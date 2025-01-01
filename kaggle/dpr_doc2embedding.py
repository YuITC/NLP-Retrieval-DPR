import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # NOTE: Để an toàn thì tốt nhất vẫn nên cho hiện warning

import os
import csv
import numpy as np
import torch
import transformers
from transformers import BertTokenizer, BertModel
from accelerate import PartialState
transformers.logging.set_verbosity_error() # Suppress Transformer logs for cleaner output

from tqdm import tqdm
from dpr_utils import get_config

if __name__ == '__main__':
    
    # Setup
    config            = get_config('/kaggle/input/dpr-config/config_embedding_dpr.yaml')
    distributed_state = PartialState()
    device            = distributed_state.device
    
    # Doc Encoder
    doc_encoder = BertModel.from_pretrained(config.pretrained_model, add_pooling_layer=False)
    tokenizer   = BertTokenizer.from_pretrained(config.pretrained_model)
    doc_encoder.eval()
    doc_encoder.to(device)
    
    # Load Wikipedia passages
    wiki = [] 
    with open(config.wiki_file, encoding='utf-8') as f: # Format: id, text, title
        reader = csv.reader(f, delimiter='\t')
        for row in tqdm(reader, total=config.num_docs, disable=not distributed_state.is_main_process, ncols=100, desc='Loading Wiki'):
            if row[0] == 'id':
                continue
            wiki.append([row[2], row[1].strip('"')])
            
    # Embedding doc
    with distributed_state.split_between_processes(wiki) as shared_wiki:
        
        # Divide data into smaller batches for encoding
        shared_wiki    = [shared_wiki[i : i+config.encoding_bs] for i in range(0, len(shared_wiki), config.encoding_bs)]
        doc_embeddings = []
        
        for data in tqdm(shared_wiki, total=len(shared_wiki), disable=not distributed_state.is_main_process, ncols=100, desc='Encoding Wiki'):
            title       = [x[0] for x in data]
            passage     = [x[1] for x in data]
            model_input = tokenizer(title, passage, max_length=config.max_length, padding='max_length', truncation=True, return_tensors='pt').to(device)
            
            with torch.no_grad():
                CLS_POS = 0
                output  = doc_encoder(**model_input).last_hidden_state[:, CLS_POS, :].cpu().numpy() # Extract embeddings for the [CLS] token
                doc_embeddings.append(output)
        
        # Concatenate embeddings from all batches    
        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        os.makedirs('embeddings', exist_ok=True)
        np.save(f"/kaggle/working/embeddings/wiki_shard_{distributed_state.process_index}.npy", doc_embeddings)