@echo off
cd /d "%~dp0"
echo ===================================================
echo  Starting Machine Health Monitoring System...
echo ===================================================
if exist .venv\Scripts\python.exe (
    echo Using virtual environment at .venv...
    start http://localhost:8000
    .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    echo Virtual environment not found, falling back to system python...
    start http://localhost:8000
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
)
pause
