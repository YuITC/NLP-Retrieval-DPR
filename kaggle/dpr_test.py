import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # NOTE: Để an toàn thì tốt nhất vẫn nên cho hiện warning

import os
import csv
import time
import faiss # For fast embedding similarity search 
import unicodedata
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import transformers
from transformers import BertModel, BertTokenizer
transformers.logging.set_verbosity_error() # Suppress Transformer logs for cleaner output

from custom_tokenizers import SimpleTokenizer
from dpr_utils import get_logger, get_config, normalize_query
logger = get_logger(__file__) # Customized logger

def normalize(text):
    """
    Normalize text using NFD (Normalization Form Decomposed).
    This decomposes characters into their base form and combining marks.
    """
    return unicodedata.normalize("NFD", text)

def has_answer(answers, doc):
    """
    Check if any of the provided answers appear in the given document.
    Uses tokenization to identify matches while ignoring case differences.

    Args:
        answers ([str]): Answers list.
        doc       (str): Document text.
    """
    tokenizer = SimpleTokenizer()
    
    doc = tokenizer.tokenize(normalize(doc)).words(uncased=True) # Normalize and tokenize document
    for answer in answers:
        answer = tokenizer.tokenize(normalize(answer)).words(uncased=True) # Normalize and tokenize each answer
        
        # Check if the tokenized answer appears in the tokenized document
        for i in range(0, len(doc) - len(answer) + 1):
            if answer == doc[i : i+len(answer)]:
                return True
    return False

def initialize_components(config):
    """
    Initialize components: load dataset, build FAISS index, and load Wikipedia passages.
    """
    
    # Load QA Dataset
    queries, answers = [], []
    with open(config.nq_test_file, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            queries.append(normalize_query(row[0]))
            answers.append(eval(row[1])) # eval(): '['A', 'B']' -> ['A', 'B']
            # NOTE: Thử dùng from ast import literal_eval, có vẻ an toàn hơn
    
    # Batch queries for efficient processing
    queries = [
        queries[i : i+config.encoding_bs] 
        for i in range(0, len(queries), config.encoding_bs)
    ]
    
    # FAISS index for inner product search
    embedding_dim = 768
    index = faiss.IndexFlatIP(embedding_dim) 
    
    # Add document embeddings (from dpr_doc2embedding.py) to the FAISS index
    for i in tqdm(range(config.num_shards), desc='Building index from embedding'):
        data = np.load(f"/kaggle/working/embeddings/wiki_shard_{i}.npy")
        index.add(data)
        
    # Load Wikipedia passages
    wiki = []
    with open(config.wiki_file, encoding='utf-8') as f: # Format: id, text, title
        reader = csv.reader(f, delimiter='\t') 
        for row in tqdm(reader, total=config.num_docs, desc='Loading wiki'):
            if row[0] == "id":
                continue
            wiki.append(row[1].strip('"'))
    
    return queries, answers, index, wiki

def retrieving(answers, index, wiki, top_k=100, visualize=True, demo=True):
    """
    Retrieve top-k documents for each query and calculate the top-k accuracy.

    Args:
        answers      (list): List of answer sets corresponding to queries.
        index (faiss.Index): FAISS index for searching embeddings.
        wiki         (list): List of Wikipedia passages.
        top_k         (int): Number of top documents to retrieve.
    """
    
    # Perform search on FAISS index
    logger.info(f"Retrieving ...")
    start = time.time()
    _, I  = index.search(query_embeddings, top_k) # inner product, top_k id
    logger.info(f"Searching for top-{top_k} takes {time.time() - start} (s)")
    
    # Track hits
    hit_lists = []
    for answer_list, id_list in tqdm(zip(answers, I), total=len(answers), desc='Calculating hits'): # Process each query
        hit_list = []
        for doc_id in id_list:
            doc = wiki[doc_id]
            hit_list.append(has_answer(answer_list, doc))
        hit_lists.append(hit_list)
        
    # Calculate top-k accuracy
    top_k_hits = [0] * top_k
    for hit_list in hit_lists:
        # Find the first correct hit
        best_hit = next((i for i, x in enumerate(hit_list) if x), None) 
        
        # If the answer appears at the top-k position, this means that it also appears at the top-h position, with h > k
        if best_hit is not None:
            top_k_hits[best_hit:] = [v+1 for v in top_k_hits[best_hit:]]
             
    top_k_accuracy = [x/len(answers) for x in top_k_hits]
    for i in range(top_k):
        if (i+1) % 10 == 0:
            logger.info(f"Top-{i+1} accuracy: {top_k_accuracy[i]}")
    
    # Visualize top-k accuracy
    if visualize:
        parts  = config.nq_test_file.split('/')
        prefix = parts[3].split('-')[1]
        name   = f"{prefix}_{parts[-1].split('.')[0]}"
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, top_k + 1), top_k_accuracy, marker='o', linestyle='-')
        
        for x, y in zip(range(1, top_k + 1), top_k_accuracy):
            if x == 1 or x % 5 == 0:
                plt.text(x, y, f"{y:.2f}", fontsize=10, ha='center', va='bottom')
        
        plt.title(f"Top-k Retrieval Accuracy on {name}")
        plt.xlabel("k (Top-k)")
        plt.ylabel("Accuracy / Hit rate (%)")
        plt.xticks(range(1, top_k + 1, 5))
        plt.grid(True)
        os.makedirs('demo', exist_ok=True)
        plt.savefig(f"/kaggle/working/demo/retrieval_performance_on_{name}.png")
        plt.show()

    # NOTE: Bổ sung phần demo for top-5
    if demo:
        k = 5
        pass

if __name__ == '__main__':
    
    # Setup
    config = get_config('/kaggle/input/dpr-config/config_test_dpr.yaml')
    queries, answers, index, wiki = initialize_components(config)
            
    # Query Encoder
    query_encoder = BertModel.from_pretrained(config.pretrained_model, add_pooling_layer=False)
    tokenizer     = BertTokenizer.from_pretrained(config.pretrained_model)
    device        = 'cuda' if torch.cuda.is_available() else 'cpu'
    query_encoder.to(device).eval()
    
    # Embedding queries
    query_embeddings = []
    for query in tqdm(queries, desc='Encoding queries'):
        with torch.no_grad():
            # Tokenize and encode queries
            query_embedding = query_encoder(
                **tokenizer(query, max_length=config.max_length, padding='max_length', truncation=True, return_tensors='pt').to(device)
            ) 
            # Extract CLS token embedding
            query_embedding = query_embedding.last_hidden_state[:, 0, :] 
        query_embeddings.append(query_embedding.cpu().detach().numpy())
    query_embeddings = np.concatenate(query_embeddings, axis=0)
    
    # Retrieve top-k documents
    retrieving(answers, index, wiki, top_k=100, visualize=True, demo=False)