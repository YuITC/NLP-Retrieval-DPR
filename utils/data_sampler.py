import random
import argparse
import ijson, json
from tqdm import tqdm
from pathlib import Path
from decimal import Decimal

from dpr_utils import get_logger
logger = get_logger(__file__) # Customized logger

def get_file_size(file_path, header=False):
    """
    Determine the number of rows (.csv, .tsv) or objects (.json) in a file.
    
    Args:
        file_path (Path): Path to the file.
        header    (bool): If True, subtracts one line for a header (default: False).
    """
    with file_path.open('r', encoding='utf-8') as f:
        if file_path.suffix in ['.tsv', '.csv']:
            return sum(1 for _ in f) - (1 if header else 0)
        return sum(1 for _ in ijson.items(f, 'item'))
    
def get_sample_output_path(file_path, pct):
    """
    Simply generate the path and directory for the output sample file.
    
    Args:
        file_path (Path): Path to the original file.
        pct        (int): Percentage of the sample size.
    """
    output_file = Path(file_path.parts[0]) / f"sample_{pct}pct_data" / Path(*file_path.parts[2:])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file
    
def sampling_JSON(file_path: Path, pct=5):
    """
    Perform Reservoir Sampling.
    """
    
    # Ensure Decimal values are serialized correctly
    def decimal_default(obj):
        if isinstance(obj, Decimal):
            return float(obj)

    # Setup for sampling
    total_size  = get_file_size(file_path)
    sample_size = int(total_size * (pct / 100))
    output_file = get_sample_output_path(file_path, pct)
    
    logger.info(f"Sampling {file_path.name} with {total_size} objects by {pct}% ...")
    
    # Perform sampling and save sampled file
    sampled_data = []
    with file_path.open('r', encoding='utf-8') as infile:
        objects = ijson.items(infile, 'item')
        
        for idx, obj in enumerate(tqdm(objects, total=total_size, desc=f"Sampling {file_path.name}")):
            if idx < sample_size:
                sampled_data.append(obj)
            else:
                r = random.randint(0, idx)
                if r < sample_size:
                    sampled_data[r] = obj
                    
    with output_file.open('w', encoding='utf-8') as outfile:
        json.dump(sampled_data, outfile, ensure_ascii=False, indent=4, default=decimal_default)
        
    logger.info(f"Sample data saved to {output_file}, with {sample_size} objects")
    
def sampling_CSV(file_path: Path, pct=5, header=True):
    """
    Perform Reservoir Sampling.
    """
    
    # Setup for sampling
    total_size  = get_file_size(file_path, header=header)
    sample_size = int(total_size * (pct / 100))
    output_file = get_sample_output_path(file_path, pct)
    
    logger.info(f"Sampling {file_path.name} with {total_size} lines by {pct}% ...")
    
    # Perform sampling and save sampled file
    sampled_data = []
    header_line  = None
    with file_path.open('r', encoding='utf-8') as infile:
        if header:
            header_line = infile.readline()
        
        for idx, line in enumerate(tqdm(infile, total=total_size, desc=f"Sampling {file_path.name}")):
            if idx < sample_size:
                sampled_data.append(line)
            else:
                r = random.randint(0, idx)
                if r < sample_size:
                    sampled_data[r] = line
                    
    with output_file.open('w', encoding='utf-8') as outfile:
        if header and header_line:
            outfile.write(header_line)
        outfile.writelines(sampled_data)
        
    logger.info(f"Sample data saved to {output_file}, with {sample_size} lines")

def sampling(data_dir, pct=5, seed=42):
    random.seed(seed)
    sampling_map = {
        '.tsv' : lambda fp: sampling_CSV(fp, pct, header=True),
        '.csv' : lambda fp: sampling_CSV(fp, pct, header=False),
        '.json': lambda fp: sampling_JSON(fp, pct),
    }
    
    for file_path in data_dir.rglob('*'):
        if file_path.is_file():
            suffix = file_path.suffix
            
            if suffix in sampling_map:
                sampling_map[suffix](file_path)
            else:
                logger.info(f"Skipping LICENSE/README file: {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_pct", default=5, type=int)
    args = parser.parse_args()
    
    sampling(Path('downloads'), args.sample_pct)