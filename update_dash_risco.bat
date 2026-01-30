@echo off
setlocal EnableDelayedExpansion

echo ============================================
echo   DASH RISCO — PIPELINE DE ATUALIZAÇÃO
echo ============================================

REM ──────────────────────────────────────────────
REM 1) Entrar no diretório do projeto
REM ──────────────────────────────────────────────
cd /d "%~dp0"

REM ──────────────────────────────────────────────
REM 2) Ativar ambiente virtual (se existir)
REM ──────────────────────────────────────────────
if exist venv\Scripts\activate.bat (
    echo Ativando virtualenv...
    call venv\Scripts\activate.bat
)

REM ──────────────────────────────────────────────
REM 3) Atualizar PL dos fundos (AF)
REM ──────────────────────────────────────────────
echo Rodando ScrapAF3.py...
python ScrapAF3.py
if errorlevel 1 goto :error

REM ──────────────────────────────────────────────
REM 4) Atualizar dados B3
REM ──────────────────────────────────────────────
echo Rodando ScrapB3_v2.py...
python ScrapB3_v2.py
if errorlevel 1 goto :error

REM ──────────────────────────────────────────────
REM 5) Rodar ETL / normalizações
REM ──────────────────────────────────────────────
echo Rodando TransformarRetornosParquet.py...
python TransformarRetornosParquet.py
if errorlevel 1 goto :error

REM ──────────────────────────────────────────────
REM 6) Criar data no formato YYYY-MM-DD
REM ──────────────────────────────────────────────
for /f %%i in ('powershell -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i

REM ──────────────────────────────────────────────
REM 7) Git add + commit + push
REM ──────────────────────────────────────────────
echo Fazendo git add...
git add .

echo Criando commit...
git commit -m "Dash Risco - %TODAY%"

echo Fazendo git push...
git push

REM ──────────────────────────────────────────────
REM 8) (Opcional) abrir Streamlit
REM ──────────────────────────────────────────────
echo Deseja abrir o app? (S/N)
set /p OPENAPP=
if /i "%OPENAPP%"=="S" (
    streamlit run app4.py
)

echo ============================================
echo   PIPELINE FINALIZADO COM SUCESSO
echo ============================================
pause
exit /b 0

:error
echo.
echo ============================================
echo   ERRO DETECTADO - PIPELINE INTERROMPIDO
echo ============================================
pause
exit /b 1
