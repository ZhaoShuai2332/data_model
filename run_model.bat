@echo off
REM ==============================================================================
REM Model Runner Script for Machine Learning Classification Project (Windows)
REM ==============================================================================
REM 
REM Description:
REM   This script serves as the unified entry point for running various machine
REM   learning models in the data classification project. It supports multiple
REM   model architectures including CNN, MLP, Logistic Regression, SVM, and
REM   Transformer-based approaches.
REM
REM Author: Data Modeling Team
REM Date: December 2025
REM Version: 1.0
REM
REM Usage:
REM   run_model.bat <model_name>      Run a specific model
REM   run_model.bat all               Run all models sequentially
REM   run_model.bat help              Display help information
REM
REM Available Models:
REM   cnn          - Convolutional Neural Network (ResNet1D architecture)
REM   mlp          - Multi-Layer Perceptron
REM   logistic     - Logistic Regression classifier
REM   svm          - Support Vector Machine
REM   transformer  - Transformer-based model
REM
REM ==============================================================================

setlocal EnableDelayedExpansion

REM ==============================================================================
REM Environment Configuration
REM ==============================================================================

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ==============================================================================
REM Color Setup (Windows 10+ supports ANSI escape codes)
REM ==============================================================================

REM Enable virtual terminal processing for colors
for /f "tokens=2 delims=[]" %%a in ('ver') do set "WINVER=%%a"

REM ==============================================================================
REM Main Entry Point
REM ==============================================================================

echo.
echo ================================================================================
echo   Machine Learning Classification Project - Model Runner
echo   Version: 1.0 ^| Date: December 2025
echo ================================================================================

REM Validate command line arguments
if "%~1"=="" (
    echo.
    echo [ERROR] No model specified.
    call :print_usage
    exit /b 1
)

REM Process command line argument
if /i "%~1"=="all" (
    call :run_all_models
    exit /b !ERRORLEVEL!
)

if /i "%~1"=="help" (
    call :print_usage
    exit /b 0
)

if /i "%~1"=="-h" (
    call :print_usage
    exit /b 0
)

if /i "%~1"=="--help" (
    call :print_usage
    exit /b 0
)

REM Run specified model
call :run_model %~1
exit /b !ERRORLEVEL!

REM ==============================================================================
REM Function: print_usage
REM Description: Display usage information and available options
REM ==============================================================================
:print_usage
echo.
echo ================================================================================
echo   Machine Learning Model Runner - Usage Guide
echo ================================================================================
echo.
echo SYNOPSIS:
echo     run_model.bat ^<model_name^>
echo.
echo AVAILABLE MODELS:
echo     cnn          - Convolutional Neural Network (ResNet1D architecture)
echo     mlp          - Multi-Layer Perceptron
echo     logistic     - Logistic Regression classifier
echo     svm          - Support Vector Machine
echo     transformer  - Transformer-based model
echo     all          - Run all models sequentially
echo.
echo OPTIONS:
echo     help, -h, --help    Display this help message
echo.
echo EXAMPLES:
echo     run_model.bat cnn           # Train CNN model
echo     run_model.bat logistic      # Train Logistic Regression model
echo     run_model.bat all           # Run all models for comparison
echo.
echo OUTPUT:
echo     Results will be saved in respective model directories.
echo     Comparison metrics will be displayed after execution.
echo.
echo ================================================================================
goto :eof

REM ==============================================================================
REM Function: print_header
REM Description: Print a formatted section header
REM Arguments: %~1 - Header text to display
REM ==============================================================================
:print_header
echo.
echo ============================================================
echo   %~1
echo ============================================================
echo.
goto :eof

REM ==============================================================================
REM Function: run_model
REM Description: Execute a specific machine learning model
REM Arguments: %~1 - Model identifier (cnn, mlp, logistic, svm, transformer)
REM Returns: 0 on success, 1 on failure
REM ==============================================================================
:run_model
set "model=%~1"
set "script="
set "name="

REM Map model identifier to script path and display name
if /i "%model%"=="cnn" (
    set "script=cnn_model\execute.py"
    set "name=CNN (ResNet1D)"
    goto :run_model_execute
)

if /i "%model%"=="mlp" (
    set "script=mlp_model\execute.py"
    set "name=Multi-Layer Perceptron (MLP)"
    goto :run_model_execute
)

if /i "%model%"=="logistic" (
    set "script=logistic_model\execute.py"
    set "name=Logistic Regression"
    goto :run_model_execute
)

if /i "%model%"=="svm" (
    set "script=svm_model\execute.py"
    set "name=Support Vector Machine (SVM)"
    goto :run_model_execute
)

if /i "%model%"=="transformer" (
    set "script=transformer_model\execute.py"
    set "name=Transformer"
    goto :run_model_execute
)

REM Unknown model
echo.
echo [ERROR] Unknown model: '%model%'
echo.
echo Run 'run_model.bat help' for a list of available models.
exit /b 1

:run_model_execute
REM Verify script exists before execution
if not exist "%script%" (
    echo.
    echo [ERROR] Script not found: %script%
    exit /b 1
)

REM Display execution header
call :print_header "Executing %name% Model"

echo [INFO] Script: %script%
echo [INFO] Start Time: %date% %time%
echo.

REM Execute the Python script
python "%script%"
set "exit_code=%ERRORLEVEL%"

REM Report execution result
echo.
echo [INFO] End Time: %date% %time%

if !exit_code!==0 (
    echo [SUCCESS] !name! model execution completed successfully!
    exit /b 0
) else (
    echo [FAILED] !name! model execution failed with exit code: !exit_code!
    exit /b 1
)

REM ==============================================================================
REM Function: run_all_models
REM Description: Execute all available models sequentially and generate summary
REM Returns: 0 if all models succeed, 1 if any model fails
REM ==============================================================================
:run_all_models
call :print_header "Running All Models - Comparative Evaluation"

echo [INFO] This will execute all available models sequentially.
echo [INFO] Total models to run: 5
echo.

REM Initialize result tracking
set "result_cnn=0"
set "result_mlp=0"
set "result_logistic=0"
set "result_svm=0"
set "result_transformer=0"
set "failed_count=0"

REM Execute each model
echo.
echo [PROGRESS] Running model 1 of 5: CNN (ResNet1D)
call :run_model cnn
set "result_cnn=%ERRORLEVEL%"
if not %result_cnn%==0 set /a failed_count+=1

echo.
echo [PROGRESS] Running model 2 of 5: Multi-Layer Perceptron
call :run_model mlp
set "result_mlp=%ERRORLEVEL%"
if not %result_mlp%==0 set /a failed_count+=1

echo.
echo [PROGRESS] Running model 3 of 5: Logistic Regression
call :run_model logistic
set "result_logistic=%ERRORLEVEL%"
if not %result_logistic%==0 set /a failed_count+=1

echo.
echo [PROGRESS] Running model 4 of 5: Support Vector Machine
call :run_model svm
set "result_svm=%ERRORLEVEL%"
if not %result_svm%==0 set /a failed_count+=1

echo.
echo [PROGRESS] Running model 5 of 5: Transformer
call :run_model transformer
set "result_transformer=%ERRORLEVEL%"
if not %result_transformer%==0 set /a failed_count+=1

REM Generate execution summary
echo.
echo ================================================================================
echo   EXECUTION SUMMARY
echo ================================================================================
echo.
echo   Model                          Status
echo   ------------------------------  --------

if !result_cnn!==0 (
    echo   CNN ^(ResNet1D^)                  [PASS]
) else (
    echo   CNN ^(ResNet1D^)                  [FAIL]
)

if !result_mlp!==0 (
    echo   Multi-Layer Perceptron         [PASS]
) else (
    echo   Multi-Layer Perceptron         [FAIL]
)

if !result_logistic!==0 (
    echo   Logistic Regression            [PASS]
) else (
    echo   Logistic Regression            [FAIL]
)

if !result_svm!==0 (
    echo   Support Vector Machine         [PASS]
) else (
    echo   Support Vector Machine         [FAIL]
)

if !result_transformer!==0 (
    echo   Transformer                    [PASS]
) else (
    echo   Transformer                    [FAIL]
)

set /a passed_count=5-failed_count

echo.
echo   ------------------------------  --------
echo   Total: 5 models ^| Passed: !passed_count! ^| Failed: !failed_count!
echo.
echo ================================================================================

if !failed_count!==0 (
    echo [SUCCESS] All models executed successfully!
    exit /b 0
) else (
    echo [WARNING] !failed_count! model(s) failed during execution.
    exit /b 1
)

REM ==============================================================================
REM End of Script
REM ==============================================================================
