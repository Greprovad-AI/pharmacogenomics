# DeepSeek Genomics

This project adapts the genomics training framework to use DeepSeek models instead of Mistral for DNA sequence analysis.

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the sample genome data:
```bash
python scripts/download_data.py
```

## Project Structure

- `src/`: Source code for model training and inference
  - `pretraining.py`: Code for pretraining DeepSeek on DNA sequences
  - `finetuning.py`: Code for finetuning on specific tasks
  - `inference.py`: Utilities for running inference
- `data/`: Directory for storing genome sequences and datasets
- `models/`: Directory for saved model checkpoints
- `notebooks/`: Jupyter notebooks for examples and tutorials
- `scripts/`: Utility scripts for data processing and setup

## Usage

See individual Python scripts in the `src/` directory for specific tasks:
- DNA sequence pretraining
- DNA classification
- Mutation effect prediction
- Synthetic DNA generation
- DNA sequence optimization
