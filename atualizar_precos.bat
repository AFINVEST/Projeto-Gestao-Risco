@echo off
setlocal EnableDelayedExpansion
title Dash Risco - Atualizar precos (pre-lancamento)
cd /d "%~dp0"

if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
if exist .env (
    for /f "usebackq eol=# tokens=1,2 delims==" %%a in (".env") do set "%%a=%%b"
)

echo [1] ScrapB3_v2.py (precos B3)...
python ScrapB3_v2.py
if errorlevel 1 goto :erro

echo.
echo [2] TransformarRetornosParquet.py...
python TransformarRetornosParquet.py
if errorlevel 1 goto :erro

echo.
echo [3] Git push (libera a data no dashboard hospedado)...
for /f %%i in ('powershell -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i
git add Dados
git commit -m "Precos B3 - %TODAY% (pre-lancamento)"
git push

echo.
echo ============================================
echo  Precos atualizados. Aguarde ~1 min o redeploy,
echo  lance as operacoes no dashboard e depois rode
echo  update_dash_risco_v2.bat
echo ============================================
pause
exit /b 0

:erro
echo ERRO - verifique o log acima
pause
exit /b 1
