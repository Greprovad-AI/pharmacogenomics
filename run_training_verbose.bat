@echo on
echo Starting training process...
echo Current directory: %CD%

echo Activating virtual environment...
call .\venv\Scripts\activate.bat

echo Setting Python environment variables...
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=0
set TOKENIZERS_PARALLELISM=true

echo Checking Python version...
python --version

echo Starting training script...
python -u src/pretraining.py

echo Training script completed
pause
