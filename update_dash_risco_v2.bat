@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
title Dash Risco - Pipeline v2 (Fase 2c)

echo ============================================
echo   DASH RISCO - PIPELINE DE ATUALIZACAO v2
echo ============================================

REM ---- 1) cd para diretorio do projeto ----
cd /d "%~dp0"

REM ---- 2) Ambiente virtual (se existir) ----
if exist venv\Scripts\activate.bat (
    echo Ativando virtualenv...
    call venv\Scripts\activate.bat
)

REM ---- 3) Carrega .env (SUPABASE_URL, SUPABASE_KEY) se existir ----
REM  Usa "eol=#" para pular linhas de comentario que comecam com #
if exist .env (
    echo Carregando .env...
    for /f "usebackq eol=# tokens=1,2 delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

REM ============================================
REM  ETAPA 0 - CDI (obrigatorio, fail-fast)
REM ============================================
echo.
echo [0] atualizar_cdi_lft.py (CDI do BCB — CRITICO, aborta se falhar)...
python atualizar_cdi_lft.py
if errorlevel 1 (
    echo.
    echo ============================================
    echo   ERRO: CDI nao pode ser atualizado.
    echo   Pipeline abortado para evitar dados incorretos.
    echo ============================================
    pause
    exit /b 1
)

REM ============================================
REM  ETAPA A - Atualizacao dos dados de mercado
REM ============================================

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

REM ============================================
REM  ETAPA B - Alinhamento de naming e base de retornos
REM ============================================

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

REM ============================================
REM  ETAPA C - Snapshot, governance e email (fase 2c)
REM ============================================

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
REM ============================================
REM  ETAPA D - Git
REM ============================================

for /f %%i in ('powershell -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i

echo.
echo [D1] Git add + commit + push...
git add .
git commit -m "Dash Risco v2 - %TODAY%"
git push

REM ============================================
REM  ETAPA E - Streamlit (opcional)
REM ============================================

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
