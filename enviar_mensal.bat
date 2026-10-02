@echo off
setlocal EnableDelayedExpansion
title Dash Risco - Email Mensal
echo ============================================
echo   EMAIL MENSAL - BOOK JUROS
echo ============================================

cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    echo Ativando virtualenv...
    call venv\Scripts\activate.bat
)

if exist .env (
    echo Carregando .env...
    for /f "usebackq eol=# tokens=1,2 delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
) else (
    echo [erro] .env nao encontrado
    pause
    exit /b 1
)

echo.
echo [1/1] enviar_email_mensal.py (via Outlook)...
python enviar_email_mensal.py %*
if errorlevel 1 (
    echo.
    echo ============================================
    echo   ERRO ao enviar email mensal
    echo ============================================
    pause
    exit /b 1
)

echo.
echo ============================================
echo   EMAIL MENSAL ENVIADO
echo ============================================
pause
