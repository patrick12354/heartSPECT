@echo off
echo ===========================================
echo   🫀 Heart SPECT AI - Prototype Launcher
echo ===========================================
echo.
echo Menjalankan aplikasi Streamlit...
echo (Aplikasi akan terbuka otomatis di browser Anda)
echo.

set VENV_PYTHON="%~dp0..\..\.venv\Scripts\python.exe"

if exist %VENV_PYTHON% (
    %VENV_PYTHON% -m streamlit run "%~dp0prototype\app.py"
) else (
    echo [ERROR] Virtual Environment tidak ditemukan di path:
    echo %VENV_PYTHON%
    echo Pastikan folder .venv masih ada di direktori utamanya.
)

echo.
pause
