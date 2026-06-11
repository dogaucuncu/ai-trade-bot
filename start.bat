@echo off
title AI Trade Bot

echo ==========================================
echo       Starting AI Trade Bot
echo ==========================================
echo.

:: Check if the virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in the 'venv' folder.
    echo Please ensure the project is set up correctly.
    pause
    exit /b 1
)

:: Run the bot
echo [INFO] Launching the bot engine...
echo [INFO] Opening dashboard in your default browser...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8000
echo.
.\venv\Scripts\python.exe main.py

echo.
echo ==========================================
echo Bot has stopped or crashed.
pause
