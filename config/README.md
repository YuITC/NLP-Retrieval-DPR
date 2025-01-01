# **1. Setup**

## Library

```bash
# requirements.txt
pip install -r requirements.txt

# Install pytorch according to the CUDA version
# https://pytorch.org/get-started/previous-versions/
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124

# Install FAISS
# https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
conda install -c conda-forge faiss-gpu
```

## Data

```bash
# Download data
python utils/data_downloader.py \
    --resource \
        data.wikipedia_split.psgs_w100 \
        data.retriever.nq \
        data.retriever.qas.nq

# Data sampling (if needed, default 20%)
python utils/data_sampler.py --sample_pct 20
```

# **2. Training**

Firstly configure distributed setting using `accelerate` library. 

(Optional) Use `wandb` for tracking performance.

```bash
accelerate config
accelerate launch dpr_train.py
```

After training, we would get a trained **Query Encoder** and **Doc Encoder**.