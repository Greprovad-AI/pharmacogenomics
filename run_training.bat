@echo off
set PYTHONUNBUFFERED=1
call .\venv\Scripts\activate.bat
python -u src/pretraining.py
