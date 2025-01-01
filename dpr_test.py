import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # NOTE: Để an toàn thì tốt nhất vẫn nên cho hiện warning

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

from utils.custom_tokenizers import SimpleTokenizer
from utils.dpr_utils import get_logger, get_config, normalize_query
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
        data = np.load(f"embeddings/wiki_shard_{i}.npy")
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
        prefix = parts[1].split('_')[1] if 'sample_' in parts[1] else parts[1]
        name   = f"{prefix}_{parts[-1]}"
        
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

        plt.savefig(f"demo/retrieval_performance_on_{name}.png")
        plt.show()

    # NOTE: Bổ sung phần demo for top-5
    if demo:
        k = 5
        pass

if __name__ == '__main__':
    
    # Setup
    config = get_config('config_test_dpr.yaml')
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

""" # NOTE: Giải thích hàm retrieving() cho ai không hiểu, tại t cũng thế =))    

### NOTE: Về các biến

answers = [
    ["apple", "banana"], >> Q1: Đáp án là "apple" hoặc "banana"
    ["cat"],             >> Q2: Đáp án là "cat"
    ["dog", "elephant"]] >> Q3: Đáp án là "dog" hoặc "elephant"

wiki = [
    "This document mentions apple and other fruits", >> d0
    "A document about banana and mango",             >> d1
    "This document is about cats and dogs",          >> d2
    "Elephants are big animals",                     >> d3
    "No relevant information here"                   >> d4
]

I_extended = [       >> Kết quả FAISS Index khi tìm top-5 tài liệu cho 3 truy vấn, thứ tự từ tương đồng cao đến thấp
    [0, 1, 4, 2, 3], >> Kết quả cho Q1
    [4, 0, 2, 1, 3], >> Kết quả cho Q2
    [4, 1, 3, 2, 0], >> Kết quả cho Q3
]

### NOTE: Về quá trình xử lý 
# Ví dụ này sẽ bỏ qua hậu tố 's' trong từ
# best_hit: Vị trí sớm nhất (top-k nhỏ nhất) tìm thấy tài liệu chứa đáp án đúng
  Nếu "hit" xảy ra tại top-k, thì nó xảy ra ở tất cả các mức lớn hơn (top-k+1, top-k+2, ...)

hit_lists = [
    [True , True , False, False, False], >> Q1: d0, d1 chứa đáp án -> best_hit = 0
    [False, False, True , False, False], >> Q2: d2     chứa đáp án -> best_hit = 2
    [False, False, True , True , False], >> Q3: d3, d2 chứa đáp án -> best_hit = 2
]

top_k_hits = [0, 0, 0, 0, 0]
>> Với Q1: top_k_hits trở thành [1, 1, 1, 1, 1]
>> Với Q2: cộng thêm 1 từ chỉ số 2 trở đi: [1, 1, 2, 2, 2]
>> Với Q3: cộng thêm 1 từ chỉ số 2 trở đi: [1, 1, 3, 3, 3]
>> top_k_hits = [1, 1, 3, 3, 3]

top_k_accuracy = [x/len(answers) for x in top_k_hits]
>> top-0: 1/3 ~ 0.3
>> top-1: 1/3 ~ 0.3
>> top-2: 3/3 = 1
>> top-3: 3/3 = 1
>> top-4: 3/3 = 1

### NOTE: Mục tiêu hàm: Đánh giá xem DPR có hiệu quả thế nào ở những thứ hạng đầu (top-k).
# 1. DPR có thể đưa ra kết quả chính xác ngay những lần truy vấn đầu tiên không?
# 2. Nếu kết quả đầu tiên không chính xác, hệ thống có thể tìm ra đáp án đúng trong các kết quả tiếp theo hay không?
"""