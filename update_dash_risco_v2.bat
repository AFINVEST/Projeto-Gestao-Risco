@echo off
setlocal EnableDelayedExpansion
title Dash Risco - Pipeline v2 (Fase 2c)

echo ============================================
echo   DASH RISCO - PIPELINE DE ATUALIZACAO v2
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
)

echo.
echo [0] atualizar_cdi_lft.py (CDI do BCB - CRITICO, aborta se falhar)...
python atualizar_cdi_lft.py
if errorlevel 1 (
    echo.
    echo ============================================
    echo   ERRO: CDI nao pode ser atualizado.
    echo   Pipeline abortado.
    echo ============================================
    pause
    exit /b 1
)

echo.
echo [A1] ScrapAF3.py (PL fundos)...
python ScrapAF3.py
if errorlevel 1 goto :error

echo.
echo [A2] ScrapB3_v2.py (dados B3)...
python ScrapB3_v2.py
if errorlevel 1 goto :error

echo.
echo [A3] TransformarRetornosParquet.py (CSV-parquet + BBG ETL)...
python TransformarRetornosParquet.py
if errorlevel 1 goto :error

echo.
echo [B1] migrar_basefundos_naming.py --apply (safety net)...
python migrar_basefundos_naming.py --apply
if errorlevel 1 (
    echo [warn] migracao de naming falhou, mas seguindo o pipeline...
)

echo.
echo [B2] atualizar_retornos_diarios.py (base para VaR de carteira)...
if "%SUPABASE_URL%"=="" (
    echo [B2-skip] SUPABASE_URL nao definido, pulando.
) else (
    python atualizar_retornos_diarios.py
    if errorlevel 1 (
        echo [warn] falha ao atualizar retornos diarios, mas seguindo...
    )
)

if "%SUPABASE_URL%"=="" (
    echo.
    echo [C-skip] SUPABASE_URL nao definido. Pulando snapshot/governance/email.
    goto :git_step
)

echo.
echo [C1] gravar_snapshot_diario.py...
python gravar_snapshot_diario.py
if errorlevel 1 (
    echo [warn] snapshot diario falhou. Seguindo o pipeline...
)

echo.
echo [C2] aplicar_governance.py...
python aplicar_governance.py
if errorlevel 1 (
    echo [warn] governance falhou. Seguindo o pipeline...
)

echo.
echo [C3] enviar_email_diario.py (via Outlook)...
python enviar_email_diario.py
if errorlevel 1 (
    echo [warn] envio do email diario falhou. Seguindo o pipeline...
)

:git_step
for /f %%i in ('powershell -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i

echo.
echo [D1] Git add + commit + push...
git add .
git commit -m "Dash Risco v2 - %TODAY%"
git push

echo.
echo Deseja abrir o app? (S/N)
set /p OPENAPP=
if /i "%OPENAPP%"=="S" (
    streamlit run app4.py
)

echo.
echo ============================================
echo   PIPELINE v2 FINALIZADO COM SUCESSO
echo ============================================
pause
exit /b 0

:error
echo.
echo ============================================
echo   ERRO CRITICO - PIPELINE INTERROMPIDO
echo ============================================
pause
exit /b 1
