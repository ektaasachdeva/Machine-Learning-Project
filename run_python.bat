@echo off
echo ======================================================================
echo Running Income Classification with Gender-Fairness Audit (Python)
echo University of Delhi - Department of Computer Science
echo Student: Ekta Sachdeva ^| Roll No: 24020570063 ^| Problem 24
echo ======================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [STATUS] Python executable is not currently detected in your system PATH.
    echo.
    echo You have two options to run this code:
    echo.
    echo ----------------------------------------------------------------------
    echo OPTION 1: Run Instantly on Google Colab (Recommended)
    echo ----------------------------------------------------------------------
    echo 1. Open your browser and go to: https://colab.research.google.com/
    echo 2. Click "Upload" and choose:
    echo    "%~dp0Income_Classification_Gender_Fairness_Audit.ipynb"
    echo 3. Click "Runtime" -^> "Run all" (Ctrl+F9).
    echo    All models, figures, and fairness audits run with one click in the cloud!
    echo.
    echo ----------------------------------------------------------------------
    echo OPTION 2: Run Locally on Windows
    echo ----------------------------------------------------------------------
    echo 1. Download official Python from: https://www.python.org/downloads/
    echo    *IMPORTANT: Check the box "Add Python to PATH" during installation.*
    echo 2. Open Command Prompt and install dependencies:
    echo    pip install ucimlrepo fairlearn scikit-learn pandas numpy matplotlib seaborn scipy
    echo 3. Re-run this run_python.bat file!
    echo ======================================================================
    echo.
    pause
    exit /b
)

echo [SUCCESS] Python found! Executing income_fairness_audit.py...
python "%~dp0income_fairness_audit.py"
echo.
echo Execution complete. Figures saved in figures/ directory.
pause
