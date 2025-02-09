import os
import requests
import gzip
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url: str, target_path: Path):
    """Download a file from URL to target path"""
    logger.info(f"Downloading {url} to {target_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def setup_data_directories():
    """Create necessary data directories"""
    dirs = [
        "data/genome_sequences/hg38",
        "data/processed",
        "models"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    # Create directories
    setup_data_directories()
    
    # Download sample genome data
    genome_data_url = "https://raw.githubusercontent.com/raphaelmourad/LLM-for-genomics-training/main/data/genome_sequences/hg38/sequences_hg38_200b_verysmall.csv.gz"
    target_gz = Path("data/genome_sequences/hg38/sequences_hg38_200b_verysmall.csv.gz")
    target_csv = Path("data/genome_sequences/hg38/sequences_hg38_200b_verysmall.csv")
    
    # Download and extract
    if not target_csv.exists():
        download_file(genome_data_url, target_gz)
        
        logger.info(f"Extracting {target_gz} to {target_csv}")
        with gzip.open(target_gz, 'rb') as f_in:
            with open(target_csv, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove gz file after extraction
        target_gz.unlink()
    
    logger.info("Data download and setup complete!")

if __name__ == "__main__":
    main()
