"""patch_process_returns_v2.py"""
from pathlib import Path
import shutil, datetime as dt
f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_procret_v2_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

old = '''def process_returns(df, assets):
    df_retorno = df.copy()
    df_retorno = df_retorno[assets]
    df_retorno.dropna(inplace=True)
    df_retorno = df_retorno.astype(float)
    df_retorno = df_retorno.tail(1260).reset_index(drop=True)
    df_retorno = np.log(df_retorno / df_retorno.shift(1))
    return df_retorno'''

novo = '''def process_returns(df, assets):
    """Retornos log dos ativos. Prioriza Supabase (novo, 3y B3 em PU); fallback: BBG 5y."""
    try:
        from risco_carteira_core import carregar_retornos_historicos
        import pandas as _pd_pr
        _wide = carregar_retornos_historicos(_pd_pr.Timestamp.today().normalize(),
                                              ativos=list(assets), janela_dias=756)
        if not _wide.empty:
            _cols = [a for a in assets if a in _wide.columns]
            if _cols:
                _df_sup = _wide[_cols].copy()
                for a in assets:
                    if a not in _df_sup.columns:
                        _df_sup[a] = 0.0
                _df_sup = _df_sup[list(assets)].reset_index(drop=True)
                return _df_sup
    except Exception:
        pass
    # Fallback antigo BBG
    df_retorno = df.copy()
    df_retorno = df_retorno[assets]
    df_retorno.dropna(inplace=True)
    df_retorno = df_retorno.astype(float)
    df_retorno = df_retorno.tail(1260).reset_index(drop=True)
    df_retorno = np.log(df_retorno / df_retorno.shift(1))
    return df_retorno'''

if old in s:
    s = s.replace(old, novo)
    f.write_text(s, encoding="utf-8")
    print("[ok] process_returns patchado com fallback")
else:
    print("[warn] bloco nao encontrado")
