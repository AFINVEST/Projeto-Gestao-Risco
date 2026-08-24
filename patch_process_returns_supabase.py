"""patch_process_returns_supabase.py - process_returns tenta Supabase primeiro (dados novos B3+3y)"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_procret_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Substitui process_returns pra usar Supabase primeiro
old = '''def process_returns(df, assets):
    df_retorno = df.copy()
    df_retorno = df_retorno[assets]
    df_retorno = df_retorno.astype(float)
    df_retorno = df_retorno.tail(1260).reset_index(drop=True)
    df_retorno = np.log(df_retorno / df_retorno.shift(1))'''

novo = '''def process_returns(df, assets):
    """Retorna DataFrame de retornos log dos ativos. Prioriza Supabase (novo metodo B3+3y).
    Fallback: metodo antigo (BBG 5y da planilha via df)."""
    # === Tentativa 1: retornos_diarios_ativo do Supabase (novo, 3y B3 em PU) ===
    try:
        from risco_carteira_core import carregar_retornos_historicos
        import pandas as _pd_pr
        _wide = carregar_retornos_historicos(_pd_pr.Timestamp.today().normalize(),
                                              ativos=list(assets), janela_dias=756)
        if not _wide.empty:
            # Filtra ativos solicitados que existem no wide
            _cols = [a for a in assets if a in _wide.columns]
            if _cols:
                _df_sup = _wide[_cols].copy()
                # Preenche ativos ausentes com 0 pra manter estrutura
                for a in assets:
                    if a not in _df_sup.columns:
                        _df_sup[a] = 0.0
                _df_sup = _df_sup[list(assets)].reset_index(drop=True)
                return _df_sup
    except Exception:
        pass
    # === Fallback: metodo antigo BBG 5y ===
    df_retorno = df.copy()
    df_retorno = df_retorno[assets]
    df_retorno = df_retorno.astype(float)
    df_retorno = df_retorno.tail(1260).reset_index(drop=True)
    df_retorno = np.log(df_retorno / df_retorno.shift(1))'''

if old in s:
    s = s.replace(old, novo)
    f.write_text(s, encoding="utf-8")
    print("[ok] process_returns agora usa Supabase (novo metodo) com fallback pro antigo")
else:
    print("[warn] bloco process_returns nao encontrado")
