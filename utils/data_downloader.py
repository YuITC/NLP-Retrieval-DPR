import wget, gzip
import argparse
from pathlib import Path

from dpr_utils import get_logger
logger = get_logger(__file__) # Customized logger

# Resources
NQ_LICENSE_FILES = [
    "https://dl.fbaipublicfiles.com/dpr/nq_license/LICENSE",
    "https://dl.fbaipublicfiles.com/dpr/nq_license/README",
]

RESOURCES_MAP = {
    "data.wikipedia_split.psgs_w100": {
        "link": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz",
        "ext": ".tsv",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },
    "data.retriever.nq-dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "NQ dev subset with passages pools for the Retriever train time validation",
        "license": NQ_LICENSE_FILES,
    },
    "data.retriever.nq-train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "NQ train subset with passages pools for the Retriever training",
        "license": NQ_LICENSE_FILES,
    },
    "data.retriever.nq-adv-hn-train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-adv-hn-train.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "NQ train subset with hard negative passages mined using the baseline DPR NQ encoders & wikipedia index",
        "license": NQ_LICENSE_FILES,
    },
    "data.retriever.trivia-dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "TriviaQA dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.trivia-train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "TriviaQA train subset with passages pools for the Retriever training",
    },
    "data.retriever.squad1-train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "SQUAD 1.1 train subset with passages pools for the Retriever training",
    },
    "data.retriever.squad1-dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "SQUAD 1.1 dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.webq-train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-webquestions-train.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "WebQuestions dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.webq-dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-webquestions-dev.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "WebQuestions dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.curatedtrec-train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-train.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "CuratedTrec dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.curatedtrec-dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-dev.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "CuratedTrec dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.qas.nq-dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv",
        "ext": ".csv",
        "compressed": False,
        "desc": "NQ dev subset for Retriever validation and IR results generation",
        "license": NQ_LICENSE_FILES,
    },
    "data.retriever.qas.nq-test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv",
        "ext": ".csv",
        "compressed": False,
        "desc": "NQ test subset for Retriever validation and IR results generation",
        "license": NQ_LICENSE_FILES,
    },
    "data.retriever.qas.nq-train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv",
        "ext": ".csv",
        "compressed": False,
        "desc": "NQ train subset for Retriever validation and IR results generation",
        "license": NQ_LICENSE_FILES,
    },
    #
    "data.retriever.qas.trivia-dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz",
        "ext": ".csv",
        "compressed": True,
        "desc": "Trivia dev subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.trivia-test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz",
        "ext": ".csv",
        "compressed": True,
        "desc": "Trivia test subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.trivia-train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-train.qa.csv.gz",
        "ext": ".csv",
        "compressed": True,
        "desc": "Trivia train subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.squad1-test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/squad1-test.qa.csv",
        "ext": ".csv",
        "compressed": False,
        "desc": "Trivia test subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.webq-test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/webquestions-test.qa.csv",
        "ext": ".csv",
        "compressed": False,
        "desc": "WebQuestions test subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.curatedtrec-test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever/curatedtrec-test.qa.csv",
        "ext": ".csv",
        "compressed": False,
        "desc": "CuratedTrec test subset for Retriever validation and IR results generation",
    },
    "data.gold_passages_info.nq_train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-train_gold_info.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "Original NQ (our train subset) gold positive passages and alternative question tokenization",
        "license": NQ_LICENSE_FILES,
    },
    "data.gold_passages_info.nq_dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-dev_gold_info.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "Original NQ (our dev subset) gold positive passages and alternative question tokenization",
        "license": NQ_LICENSE_FILES,
    },
    "data.gold_passages_info.nq_test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-test_gold_info.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "Original NQ (our test, original dev subset) gold positive passages and alternative question "
        "tokenization",
        "license": NQ_LICENSE_FILES,
    },
    "pretrained.fairseq.roberta-base.dict": {
        "link": "https://dl.fbaipublicfiles.com/dpr/pretrained/fairseq/roberta/dict.txt",
        "ext": ".txt",
        "compressed": False,
        "desc": "Dictionary for pretrained fairseq roberta model",
    },
    "pretrained.fairseq.roberta-base.model": {
        "link": "https://dl.fbaipublicfiles.com/dpr/pretrained/fairseq/roberta/model.pt",
        "ext": ".pt",
        "compressed": False,
        "desc": "Weights for pretrained fairseq roberta base model",
    },
    "pretrained.pytext.bert-base.model": {
        "link": "https://dl.fbaipublicfiles.com/dpr/pretrained/pytext/bert/bert-base-uncased.pt",
        "ext": ".pt",
        "compressed": False,
        "desc": "Weights for pretrained pytext bert base model",
    },
    "data.retriever_results.nq.single.wikipedia_passages": {
        "link": [
            "https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single/nq/wiki_passages_{}".format(i)
            for i in range(50)
        ],
        "ext": ".pkl",
        "compressed": False,
        "desc": "Encoded wikipedia files using a biencoder checkpoint("
        "checkpoint.retriever.single.nq.bert-base-encoder) trained on NQ dataset ",
    },
    "data.retriever_results.nq.single-adv-hn.wikipedia_passages": {
        "link": [
            "https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single-adv-hn/nq/wiki_passages_{}".format(i)
            for i in range(50)
        ],
        "ext": ".pkl",
        "compressed": False,
        "desc": "Encoded wikipedia files using a biencoder checkpoint("
        "checkpoint.retriever.single-adv-hn.nq.bert-base-encoder) trained on NQ dataset + adversarial hard negatives",
    },
    "data.retriever_results.nq.single.test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ test dataset for the encoder trained on NQ",
        "license": NQ_LICENSE_FILES,
    },
    "data.retriever_results.nq.single.dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ dev dataset for the encoder trained on NQ",
        "license": NQ_LICENSE_FILES,
    },
    "data.retriever_results.nq.single.train": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-train.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ train dataset for the encoder trained on NQ",
        "license": NQ_LICENSE_FILES,
    },
    "data.retriever_results.nq.single-adv-hn.test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single-adv-hn/nq-test.json.gz",
        "ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ test dataset for the encoder trained on NQ + adversarial hard negatives",
        "license": NQ_LICENSE_FILES,
    },
    "checkpoint.retriever.single.nq.bert-base-encoder": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriever/single/nq/hf_bert_base.cp",
        "ext": ".cp",
        "compressed": False,
        "desc": "Biencoder weights trained on NQ data and HF bert-base-uncased model",
    },
    "checkpoint.retriever.multiset.bert-base-encoder": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/multiset/hf_bert_base.cp",
        "ext": ".cp",
        "compressed": False,
        "desc": "Biencoder weights trained on multi set data and HF bert-base-uncased model",
    },
    "checkpoint.retriever.single-adv-hn.nq.bert-base-encoder": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/single-adv-hn/nq/hf_bert_base.cp",
        "ext": ".cp",
        "compressed": False,
        "desc": "Biencoder weights trained on the original DPR NQ data combined with adversarial hard negatives (See data.retriever.nq-adv-hn-train resource). "
        "The model is HF bert-base-uncased",
    },
    "data.reader.nq.single.train": {
        "link": ["https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/train.{}.pkl".format(i) for i in range(8)],
        "ext": ".pkl",
        "compressed": False,
        "desc": "Reader model NQ train dataset input data preprocessed from retriever results (also trained on NQ)",
        "license": NQ_LICENSE_FILES,
    },
    "data.reader.nq.single.dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/dev.0.pkl",
        "ext": ".pkl",
        "compressed": False,
        "desc": "Reader model NQ dev dataset input data preprocessed from retriever results (also trained on NQ)",
        "license": NQ_LICENSE_FILES,
    },
    "data.reader.nq.single.test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/test.0.pkl",
        "ext": ".pkl",
        "compressed": False,
        "desc": "Reader model NQ test dataset input data preprocessed from retriever results (also trained on NQ)",
        "license": NQ_LICENSE_FILES,
    },
    "data.reader.trivia.multi-hybrid.train": {
        "link": [
            "https://dl.fbaipublicfiles.com/dpr/data/reader/trivia/multi-hybrid/train.{}.pkl".format(i)
            for i in range(8)
        ],
        "ext": ".pkl",
        "compressed": False,
        "desc": "Reader model Trivia train dataset input data preprocessed from hybrid retriever results "
        "(where dense part is trained on multiset)",
    },
    "data.reader.trivia.multi-hybrid.dev": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/reader/trivia/multi-hybrid/dev.0.pkl",
        "ext": ".pkl",
        "compressed": False,
        "desc": "Reader model Trivia dev dataset input data preprocessed from hybrid retriever results "
        "(where dense part is trained on multiset)",
    },
    "data.reader.trivia.multi-hybrid.test": {
        "link": "https://dl.fbaipublicfiles.com/dpr/data/reader/trivia/multi-hybrid/test.0.pkl",
        "ext": ".pkl",
        "compressed": False,
        "desc": "Reader model Trivia test dataset input data preprocessed from hybrid retriever results "
        "(where dense part is trained on multiset)",
    },
    "checkpoint.reader.nq-single.hf-bert-base": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-single/hf_bert_base.cp",
        "ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on NQ-single retriever results and HF bert-base-uncased model",
    },
    "checkpoint.reader.nq-trivia-hybrid.hf-bert-base": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-trivia-hybrid/hf_bert_base.cp",
        "ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on Trivia multi hybrid retriever results and HF bert-base-uncased model",
    },
    # extra checkpoints for EfficientQA competition
    "checkpoint.reader.nq-single-subset.hf-bert-base": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-single-seen_only/hf_bert_base.cp",
        "ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on NQ-single retriever results and HF bert-base-uncased model, when only Wikipedia pages seen during training are considered",
    },
    "checkpoint.reader.nq-tfidf.hf-bert-base": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-drqa/hf_bert_base.cp",
        "ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on TFIDF results and HF bert-base-uncased model",
    },
    "checkpoint.reader.nq-tfidf-subset.hf-bert-base": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-drqa-seen_only/hf_bert_base.cp",
        "ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on TFIDF results and HF bert-base-uncased model, when only Wikipedia pages seen during training are considered",
    },
    # retrieval indexes
    "indexes.single.nq.full.index": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/full.index.dpr",
        "ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever",
    },
    "indexes.single.nq.full.index_meta": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/full.index_meta.dpr",
        "ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever (metadata)",
    },
    "indexes.single.nq.subset.index": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/seen_only.index.dpr",
        "ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever when only Wikipedia pages seen during training are considered",
    },
    "indexes.single.nq.subset.index_meta": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/seen_only.index_meta.dpr",
        "ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever when only Wikipedia pages seen during training are considered (metadata)",
    },
    "indexes.tfidf.nq.full": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/drqa/nq/full-tfidf.npz",
        "ext": ".npz",
        "compressed": False,
        "desc": "TFIDF index",
    },
    "indexes.tfidf.nq.subset": {
        "link": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/drqa/nq/seen_only-tfidf.npz",
        "ext": ".npz",
        "compressed": False,
        "desc": "TFIDF index when only Wikipedia pages seen during training are considered",
    },
    # Universal retriever project data
    "data.wikipedia_split.psgs_w100": {
        "link": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz",
        "ext": ".tsv",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },
}

def decompress(gzip_file, out_file, chunk_size=1024*1024*1024):
    """
    Decompress a gzip file.
    
    Args:
        gzip_file (Path): Path of the gzip file.
        out_file  (Path): Path of the decompressed file.
        chunk_size (int): Size of reading chunk while decompressing (default: 1 GB).
    """
    logger.info(f"Decompressing {gzip_file.name} to {out_file.name} ...")
    
    with gzip.open(gzip_file, "rb") as input_file, \
         out_file.open("wb")        as output_file:
             
        while chunk := input_file.read(chunk_size):
            output_file.write(chunk)

def download_resource(link, ext, compressed, resource_key, out_dir):
    """
    Download resources from the given link.
    
    Args:
        link         (str): URL of the resource to download.
        ext          (str): Extension of the downloaded resource.
        compressed  (bool): Whether the resource is compressed.
        resource_key (str): Identifier for the resource (check RESOURCES_MAP).
        out_dir     (Path): Output directory for resource after processed.
        
    Returns:
        tuple: (Path to save directory, Path to the downloaded file).
    """
    logger.info(f"Requesting resource from: {link} ...")
    
    # Define save directory
    path_names = resource_key.split(".") 
    root_dir   = out_dir if out_dir else Path("./").resolve()
    root_dir   = Path(str(root_dir).split("/outputs/")[0]) if "/outputs/" in str(root_dir) else root_dir 
    save_dir   = root_dir / "downloads" / Path(*path_names[:-1])
    save_dir.mkdir(parents=True, exist_ok=True) 
    
    # Avoid redundant downloads
    if (final_file := save_dir / f"{path_names[-1]}{ext}").exists():
        logger.info(f"Resource already exists: {final_file}")
        return save_dir, final_file
      
    # Download the resource
    downloaded_file = save_dir / f"{path_names[-1]}{'.tmp' if compressed else ext}"
    wget.download(link, out=str(downloaded_file))
    print()
    logger.info(f"Downloaded resource: {downloaded_file}")

    # Decompress the resource if needed
    if compressed:
        decompress(downloaded_file, decompressed_file := save_dir / f"{path_names[-1]}{ext}")
        downloaded_file.unlink()
        downloaded_file = decompressed_file
        logger.info(f"Downloaded resource (decompressed): {downloaded_file}")
        
    return save_dir, downloaded_file

def download_file(link, out_dir, file_name):
    """
    Same function as download_resource().
    """
    logger.info(f"Requesting file from: {link} ...")
    
    if (downloaded_file := out_dir / file_name).exists():
        logger.info(f"File already exists: {downloaded_file}")
        return

    wget.download(link, out=str(downloaded_file))
    logger.info(f"Downloaded file: {downloaded_file}")

def download(resource_key, out_dir=Path("./")):
    """
    Download stuff based on the given resource key.
    """
    logger.info(f"Downloading {resource_key} ...")
    
    # Find matches resouce
    if resource_key not in RESOURCES_MAP:
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        logger.info(f"Matched resources: {resources}")
        if resources:
            for key in resources:
                download(key, out_dir)
        return []
    
    # Retrieve resource information from the mapping
    download_info = RESOURCES_MAP[resource_key]
    links         = link if isinstance(link := download_info["link"], list) else [link]

    data_files = []
    for i, url in enumerate(links):
        save_dir, downloaded_file = download_resource(
            url, download_info["ext"], download_info["compressed"],
            f"{resource_key}_{i}" if len(links) > 1 else resource_key,
            out_dir
        )
        data_files.append(downloaded_file)

    # Download associated license files if specified
    if license := download_info.get("license", None):
        download_file(license[0], save_dir, "LICENSE")
        download_file(license[1], save_dir, "README")
        
    return data_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument("--resource"  , nargs="+"   , type=str, help="Resource key. See RESOURCES_MAP for all possible values.")
    args = parser.parse_args()

    for resource in args.resource:
        download(resource, Path(args.output_dir))