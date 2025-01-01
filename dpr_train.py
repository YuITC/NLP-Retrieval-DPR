import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # NOTE: Để an toàn thì tốt nhất vẫn nên cho hiện warningho hiện warning

import os
import json
import functools
import math, random
from tqdm import tqdm
from itertools import chain

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import transformers
from transformers import BertTokenizer, BertModel

transformers.logging.set_verbosity_error()    # Suppress Transformer logs for cleaner output
os.environ['TOKENIZERS_PARALLELISM'] = 'true' # Enable parallelism for tokenizers # NOTE: Chưa biết có tác dụng hay không

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from utils.dpr_utils import get_logger, set_seed, normalize_document, normalize_query, get_config, get_linear_scheduler
logger = get_logger(__file__) # Customized logger

class DualEncoder(nn.Module):
    """
    A dual encoder model for DPR, consisting of separate encoders for queries and documents.
    """
    def __init__(self, query_encoder, doc_encoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder   = doc_encoder
        
    def forward(self, query_input_ids, query_attention_mask, query_token_type_ids, 
                        doc_input_ids,   doc_attention_mask,   doc_token_type_ids):
        """
        Args:
            query_input_ids          ([bs, seq_len]): Tokenized query IDs.
            query_attention_mask     ([bs, seq_len]): Mask indicating attention for query tokens.
            query_token_type_ids     ([bs, seq_len]): Token type IDs for distinguishing segments.
            doc_input_ids      ([bs*n_doc, seq_len]): ...
            doc_attention_mask ([bs*n_doc, seq_len]): ...
            doc_token_type_ids ([bs*n_doc, seq_len]): ...
        
        Returns:
            query_embedding       ([bs, n_dim]): Query embeddings.
            doc_embedding   ([bs*n_doc, n_dim]): Document embeddings.
        
        Notes:  
            .last_hidden_state ((bs, seq_len, hidden_size)): Tensor lưu giữ thông tin ngữ nghĩa cho mỗi token trong chuỗi sau khi đi qua toàn bộ các layer của model.
            .token_type_ids are unused in some models like DistilBERT.
        """
        CLS_POS = 0 # Position of [CLS] token for extracting sentence-level embeddings
        
        query_embedding = self.query_encoder(query_input_ids, query_attention_mask, query_token_type_ids).last_hidden_state[:, CLS_POS, :]
        doc_embedding   = self.doc_encoder(doc_input_ids, doc_attention_mask, doc_token_type_ids).last_hidden_state[:, CLS_POS, :]
        
        return query_embedding, doc_embedding
    
def calc_dpr_loss(matching_score, labels):
    """
    Calculate loss for DPR using negative log-likelihood. (Eq. (2) in paper).

    Args:
        matching_score (torch.Tensor): Similarity scores between queries and documents.
        labels         (torch.Tensor): Ground truth labels.
    """
    return F.nll_loss(input=F.log_softmax(matching_score, dim=1), target=labels)

def calc_hit_cnt(matching_score, labels):
    """
    Compute the number of correct top-1 predictions (hit count).
    """
    _, max_ids = torch.max(matching_score, 1)
    return (max_ids == labels).sum()

def calc_avg_rank(matching_score, labels):
    """
    Compute the average rank of the correct documents. With rank starts at 1.
    """
    _, indices = torch.sort(matching_score, dim=1, descending=True)
    
    ranks = []
    for idx, label in enumerate(labels):
        rank = ((indices[idx] == label).nonzero()).item() + 1
        ranks.append(rank)
    return ranks

class QADataset(Dataset):
    """
    Dataset class for loading and processing question-answer pairs for DPR.
    """
    def __init__(self, file_path):
        self.data = json.load(open(file_path, encoding='utf-8'))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples, tokenizer, config, stage):
        """
        Prepare and tokenize inputs for queries and documents.

        Args:
            samples (list): List of samples from the dataset.
            tokenizer     : Tokenizer instance for encoding text.
            config        : Configuration object with model parameters.
            stage    (str): Training or validation stage.

        Returns:
            dict: Encoded inputs for query and document pairs.
        """
        
        # Prepare and tokenize query inputs
        queries      = [normalize_query(x['question']) for x in samples]
        query_inputs = tokenizer(queries, max_length=config.max_length, padding=True, truncation=True, return_tensors='pt')
        
        # Prepare positive document
        positive_passages = [x['positive_ctxs'][0] for x in samples]
        positive_titles   = [x['title'] for x in positive_passages]
        positive_docs     = [x['text']  for x in positive_passages]

        # Prepare negative document
        if stage == 'train':
            # Randomly sample one negative passage per sample
            negative_passages = [
                random.choice(x['hard_negative_ctxs']) if len(x['hard_negative_ctxs']) != 0 else \
                random.choice(x['negative_ctxs']) 
                for x in samples
            ]
        elif stage == 'val':
            negative_passages = list(chain.from_iterable([
                x['hard_negative_ctxs'][:min(config.num_hard_negative_ctx , len(x['hard_negative_ctxs']))] + \
                x['negative_ctxs']     [:min(config.num_other_negative_ctx, len(x['negative_ctxs']))] 
                for x in samples
            ])) # Flatten

        negative_titles = [x['title'] for x in negative_passages]
        negative_docs   = [x['text']  for x in negative_passages]
        
        # Prepare and tokenize document inputs
        titles     = positive_titles + negative_titles
        docs       = positive_docs   + negative_docs
        doc_inputs = tokenizer(titles, docs, max_length=config.max_length, padding=True, truncation=True, return_tensors='pt')

        return {
            'query_input_ids'     : query_inputs.input_ids,
            'query_attention_mask': query_inputs.attention_mask,
            'query_token_type_ids': query_inputs.token_type_ids,

            'doc_input_ids'     : doc_inputs.input_ids,
            'doc_attention_mask': doc_inputs.attention_mask,
            'doc_token_type_ids': doc_inputs.token_type_ids,
        }
        
def validate(model, dataloader, accelerator):
    """
    Evaluate the dual encoder model on validation data.
    """
    model.eval()
    
    # Compute embeddings for queries and documents
    query_embeddings   = []
    pos_doc_embeddings = []
    neg_doc_embeddings = []
    for batch in dataloader:
        with torch.no_grad():
            query_embedding, doc_embedding = model(**batch)
        query_num, _ = query_embedding.shape
        
        query_embeddings.append(query_embedding.cpu())
        pos_doc_embeddings.append(doc_embedding[:query_num, :].cpu())
        neg_doc_embeddings.append(doc_embedding[query_num:, :].cpu())
        
    query_embeddings = torch.cat(query_embeddings, dim=0)
    doc_embeddings   = torch.cat(pos_doc_embeddings + neg_doc_embeddings, dim=0)
    
    # Compute similarity scores and labels
    matching_score = torch.matmul(query_embeddings, doc_embeddings.permute(1, 0)) # (bs, n_pos_doc+n_neg_doc)
    labels         = torch.arange(query_embeddings.shape[0], dtype=torch.int64).to(matching_score.device)
    
    # Calculate validation loss and average rank
    loss  = calc_dpr_loss(matching_score, labels).item()
    ranks = calc_avg_rank(matching_score, labels)
    
    # Aggregate results across distributed processes if possible
    if accelerator.use_distributed and accelerator.num_processes > 1:
        ranks_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(ranks_from_all_gpus, ranks)
        
        ranks = [x for y in ranks_from_all_gpus for x in y] # Flatten, dùng cách của negative_passages cũng được

        loss_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(loss_from_all_gpus, loss)
        
        loss = sum(loss_from_all_gpus) / len(loss_from_all_gpus) 
    
    return sum(ranks) / len(ranks), loss

def initialize_components(config):
    """
    Initialize components for training.
    """
    
    # Setup Accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.grad_accumulate_steps,
        log_with=None,
        mixed_precision='fp16',
        kwargs_handlers=[kwargs]
    )

    # Query Encoder and Doc Encoder
    tokenizer     = BertTokenizer.from_pretrained(config.base_model)
    query_encoder = BertModel.from_pretrained(config.base_model, add_pooling_layer=False)
    doc_encoder   = BertModel.from_pretrained(config.base_model, add_pooling_layer=False)
    dual_encoder  = DualEncoder(query_encoder, doc_encoder)
    dual_encoder.train()
    
    # Dataset and Dataloader
    train_dataset    = QADataset(config.train_file)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.train_bs_per_device, 
        shuffle=True, 
        collate_fn=functools.partial(QADataset.collate_fn, tokenizer=tokenizer, config=config, stage='train'), 
        num_workers=config.num_workers, 
        prefetch_factor=config.prefetch_factor,
        pin_memory=True
    )
    
    val_dataset = QADataset(config.val_file)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.val_bs_per_device,
        shuffle=False,
        collate_fn=functools.partial(QADataset.collate_fn, tokenizer=tokenizer, config=config, stage='val'),
        num_workers=config.num_workers, 
        prefetch_factor=config.prefetch_factor,
        pin_memory=True
    )
    
    # Optimizer and Scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optim_grouped_params = [
        {
            'params'      : [p for n, p in dual_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay,
        },
        {
            'params'      : [p for n, p in dual_encoder.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optim_grouped_params, lr=config.lr, eps=config.adam_eps)
    
    # Prepare components with accelerator for distributed training
    dual_encoder, optimizer, train_dataloader, val_dataloader = accelerator.prepare(dual_encoder, optimizer, train_dataloader, val_dataloader)
    
    return accelerator, dual_encoder, optimizer, train_dataloader, val_dataloader, tokenizer

if __name__ == '__main__':
    
    # Setup
    config = get_config('config/config_train_dpr.yaml')
    set_seed(config.seed)
    accelerator, dual_encoder, optimizer, train_dataloader, val_dataloader, tokenizer = initialize_components(config)    
    
    UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / config.grad_accumulate_steps)
    MAX_TRAIN_STEPS   = UPDATES_PER_EPOCH * config.epochs
    MAX_TRAIN_EPOCHS  = math.ceil(MAX_TRAIN_STEPS / UPDATES_PER_EPOCH)
    EVAL_STEPS        = config.val_interval if isinstance(config.val_interval, int) else \
                        int(config.val_interval * UPDATES_PER_EPOCH)
                        
    lr_scheduler = get_linear_scheduler(optimizer, warmup_steps=config.warmup_steps, total_training_steps=MAX_TRAIN_STEPS)
    
    # Logger
    logger.info("Training ...")
    logger.info(f"Num train examples: {len(train_dataloader.dataset)}")
    logger.info(f"Num val examples  : {len(val_dataloader.dataset)}")
    logger.info(f"Updates per epoch : {UPDATES_PER_EPOCH}")
    logger.info(f"Num Steps         : {MAX_TRAIN_STEPS}")
    logger.info(f"Num Epochs        : {MAX_TRAIN_EPOCHS}")
    logger.info(f"Num Eval Step     : {EVAL_STEPS}")
    logger.info(f"Train Batch size per device: {config.train_bs_per_device}")
    logger.info(f"Val   Batch size per device: {config.val_bs_per_device}")
    logger.info(f"Gradient Accumulate steps  : {config.grad_accumulate_steps}")
    logger.info(f"Total train batch size (w. parallel, distributed & accumulation): {config.train_bs_per_device * accelerator.num_processes * config.grad_accumulate_steps}")
    
    # Training
    completed_steps = 0
    pbar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process, ncols=100)
    
    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(config.seed + epoch)
        pbar.set_description(f"Epoch {epoch + 1}/{MAX_TRAIN_EPOCHS}")
        
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dual_encoder):
                with accelerator.autocast():
                    query_embedding, doc_embedding = dual_encoder(**batch)
                    single_device_query_num, _     = query_embedding.shape
                    single_device_doc_num, _       = doc_embedding.shape
                    
                    # Gather embeddings from all devices
                    if accelerator.use_distributed: 
                        doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                        
                        doc_list[dist.get_rank()] = doc_embedding
                        doc_embedding = torch.cat(doc_list, dim=0)

                        query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                        
                        query_list[dist.get_rank()] = query_embedding
                        query_embedding = torch.cat(query_list, dim=0)
                    
                    # Compute similarity scores and loss
                    matching_score = torch.matmul(query_embedding, doc_embedding.permute(1, 0))
                    labels = torch.cat(
                        [
                            torch.arange(single_device_query_num) + gpu_index * single_device_doc_num
                            for gpu_index in range(accelerator.num_processes)
                        ],
                        dim=0
                    ).to(matching_score.device)
                    loss = calc_dpr_loss(matching_score, labels=labels)
                    
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
                    completed_steps += 1
                    
                    # Gradient clipping to avoid exploding gradient
                    accelerator.clip_grad_norm_(dual_encoder.parameters(), config.max_grad_norm)
                    
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    
                    logger.info(f"Training Loss: {loss}, Learning Rate: {lr_scheduler.get_last_lr()[0]}, Step: {completed_steps}")
                    
                    if completed_steps % EVAL_STEPS == 0:
                        avg_rank, loss = validate(dual_encoder, val_dataloader, accelerator)
                        dual_encoder.train()

                        logger.info(f"Average Rank: {avg_rank}, Loss: {loss}, Step: {completed_steps}")
                        accelerator.wait_for_everyone()
                        
                        if accelerator.is_local_main_process:
                            trained_dir = os.path.join("trained", f"step-{completed_steps}")
                            os.makedirs(trained_dir, exist_ok=True)
                            
                            unwrapped_model = accelerator.unwrap_model(dual_encoder)
                            
                            # Save query_encoder
                            unwrapped_model.query_encoder.save_pretrained(os.path.join(trained_dir, "query_encoder"))
                            tokenizer.save_pretrained(os.path.join(trained_dir, "query_encoder"))

                            # Save doc_encoder
                            unwrapped_model.doc_encoder.save_pretrained(os.path.join(trained_dir, "doc_encoder"))
                            tokenizer.save_pretrained(os.path.join(trained_dir, "doc_encoder"))

                        accelerator.wait_for_everyone()
                
                optimizer.step()
                optimizer.zero_grad()
    
    # accelerator.end_training() # NOTE: Bao giờ dùng các log tracker như WandB thì phải chỉnh lại
    accelerator.wait_for_everyone()