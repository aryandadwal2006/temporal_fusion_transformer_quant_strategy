@echo off
REM ============================================================================
REM TFT Quant Strategy - Windows Run Script
REM ============================================================================
REM This script provides an easy way to run the project on Windows
REM ============================================================================

setlocal enabledelayedexpansion

REM Change to D: drive and navigate to project directory
D:
cd D:\tft-quant-strategy-main

echo.
echo ============================================================
echo TFT Quant Strategy - Windows Runner
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run setup first:
    echo   1. Open Git Bash in this directory
    echo   2. Run: bash setup.sh
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated
echo.

REM Display menu
:menu
echo ============================================================
echo What would you like to do?
echo ============================================================
echo.
echo 1. Run Quick Test (2-5 minutes)
echo 2. Run Full Pipeline (several hours)
echo 3. Run Full Pipeline (skip conversion)
echo 4. Run Full Pipeline (skip tuning)
echo 5. Run Full Pipeline (skip training, backtest only)
echo 6. Open Python Shell
echo 7. Check GPU Status
echo 8. Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto quick_test
if "%choice%"=="2" goto full_pipeline
if "%choice%"=="3" goto skip_conversion
if "%choice%"=="4" goto skip_tuning
if "%choice%"=="5" goto backtest_only
if "%choice%"=="6" goto python_shell
if "%choice%"=="7" goto check_gpu
if "%choice%"=="8" goto end

echo.
echo [ERROR] Invalid choice. Please enter 1-8.
echo.
goto menu

:quick_test
echo.
echo ============================================================
echo Running Quick Test
echo ============================================================
echo.
python test.py
if errorlevel 1 (
    echo.
    echo [ERROR] Quick test failed!
    pause
    goto menu
)
echo.
echo [OK] Quick test completed successfully!
pause
goto menu

:full_pipeline
echo.
echo ============================================================
echo Running Full Pipeline
echo ============================================================
echo.
echo WARNING: This will take several hours!
echo.
set /p confirm="Are you sure? (y/n): "
if /i not "%confirm%"=="y" goto menu

python main_enhanced.py
if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed!
    pause
    goto menu
)
echo.
echo [OK] Pipeline completed!
pause
goto menu

:skip_conversion
echo.
echo ============================================================
echo Running Pipeline (Skipping Conversion)
echo ============================================================
echo.
python main_enhanced.py --skip-conversion
if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed!
    pause
    goto menu
)
echo.
echo [OK] Pipeline completed!
pause
goto menu

:skip_tuning
echo.
echo ============================================================
echo Running Pipeline (Skipping Hyperparameter Tuning)
echo ============================================================
echo.
python main_enhanced.py --skip-tuning
if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed!
    pause
    goto menu
)
echo.
echo [OK] Pipeline completed!
pause
goto menu

:backtest_only
echo.
echo ============================================================
echo Running Backtest Only
echo ============================================================
echo.
python main_enhanced.py --skip-conversion --skip-data-prep --skip-tuning --skip-training
if errorlevel 1 (
    echo.
    echo [ERROR] Backtest failed!
    pause
    goto menu
)
echo.
echo [OK] Backtest completed!
pause
goto menu

:python_shell
echo.
echo ============================================================
echo Opening Python Shell
echo ============================================================
echo.
echo Type 'exit()' to return to menu
echo.
python
goto menu

:check_gpu
echo.
echo ============================================================
echo Checking GPU Status
echo ============================================================
echo.
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo.

REM Also check NVIDIA-SMI if available
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU Info:
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv
) else (
    echo nvidia-smi not found
)

echo.
pause
goto menu

:end
echo.
echo Deactivating virtual environment...
call deactivate
echo.
echo Goodbye!
echo.
pause
exit /b 0