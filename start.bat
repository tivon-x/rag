@echo off
echo Starting RAG Application System...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.10+ first
    pause
    exit /b 1
)

REM 检查是否存在虚拟环境
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM 激活虚拟环境
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM 安装依赖
echo Installing dependencies...
pip install -r requirements.txt

REM 检查.env文件
if not exist ".env" (
    echo Warning: .env file not found
    echo Please copy .env.example to .env and configure your API keys
    echo.
)

REM 启动应用
echo Starting RAG application...
python main.py

pause
