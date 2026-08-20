"""
risco_carteira_core.py
======================

Modulo standalone (SEM streamlit) que calcula VaR / CVaR / DV01 da CARTEIRA ATUAL
usando retornos historicos por ativo (tabela Supabase retornos_diarios_ativo).

Replica a logica de `calcular_metricas_por_pl` do app4.py, mas:
  - Le retornos do Supabase (nao mais da planilha BBG estatica).
  - Retorna dict simples (compatible com snapshot_diario).
  - Suporta bootstrap historico (aplica posicoes de cada dia com retornos ate o dia).

Metodos de VaR:
  - 'hist'  : quantile empirico simples (equal-weight) — mesmo do dashboard.
  - 'ewma'  : pesos exponenciais (lambda~0.99, halflife ~68 dias) para
              sensibilidade a choques recentes.

Uso tipico:
    from risco_carteira_core import calcular_var_completo
    res = calcular_var_completo(
        data_ref=pd.Timestamp("2026-07-31"),
        pl_total=650_000_000,
        janela_dias=756,   # 3 anos
    )
    print(res)
"""
from __future__ import annotations
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

ALPHA_DEFAULT = 0.05
JANELA_DIAS_DEFAULT = 756    # ~3 anos (mais responsivo; dashboard usa 5 anos como referencia)
LAMBDA_EWMA_DEFAULT = 0.99   # halflife ~68 dias (~3.4 meses)
TABELA_RETORNOS = "retornos_diarios_ativo"


# ─────────────────────────────────────────────────────────────────────────────
# Supabase
# ─────────────────────────────────────────────────────────────────────────────

_CLIENT = None
def _get_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Defina SUPABASE_URL e SUPABASE_KEY.")
    _CLIENT = create_client(url, key)
    return _CLIENT


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def carregar_retornos_historicos(data_ref: pd.Timestamp,
                                  ativos: list[str] | None = None,
                                  janela_dias: int = JANELA_DIAS_DEFAULT) -> pd.DataFrame:
    """Retorna DataFrame wide (index=Data, columns=Ativo, values=retorno).

    Filtra data_ref - janela_dias <= Data <= data_ref.
    """
    client = _get_client()
    data_ini = (pd.Timestamp(data_ref) - pd.Timedelta(days=int(janela_dias * 1.5))).date().isoformat()
    data_fim = pd.Timestamp(data_ref).date().isoformat()
    # Paginação (Supabase limita a 1000 por padrão)
    rows = []
    offset = 0
    page = 1000
    while True:
        q = (client.table(TABELA_RETORNOS)
             .select("Data,Ativo,retorno")
             .gte("Data", data_ini)
             .lte("Data", data_fim)
             .range(offset, offset + page - 1))
        if ativos:
            q = q.in_("Ativo", list(ativos))
        resp = q.execute()
        data = resp.data or []
        if not data:
            break
        rows.extend(data)
        if len(data) < page:
            break
        offset += page

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Data"] = pd.to_datetime(df["Data"])
    df["retorno"] = pd.to_numeric(df["retorno"], errors="coerce")
    # Wide
    wide = df.pivot_table(index="Data", columns="Ativo", values="retorno", aggfunc="last")
    wide = wide.sort_index()
    # Limita à janela exata
    corte = pd.Timestamp(data_ref) - pd.Timedelta(days=janela_dias)
    wide = wide.loc[wide.index >= corte]
    return wide


def _parse_ptbr(v):
    if v is None: return None
    try:
        if pd.isna(v): return None
    except Exception: pass
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    s = str(v).strip().replace("R$", "").replace(" ", "")
    if not s or s in ("-", "--"):
        return None
    s2 = s.replace(".", "").replace(",", ".")
    try: return float(s2)
    except Exception: return None


def carregar_precos_atuais(data_ref: pd.Timestamp,
                            path: str = "Dados/df_preco_de_ajuste_atual_completo.parquet") -> dict:
    """Retorna {ativo: preco} na data_ref (ou mais recente <= data_ref)."""
    df = pd.read_parquet(path)
    if "Assets" not in df.columns:
        return {}
    col_ref = data_ref.strftime("%Y-%m-%d") if hasattr(data_ref, "strftime") else str(data_ref)[:10]
    cols_data = [c for c in df.columns if c != "Assets"]
    if col_ref in cols_data:
        col_alvo = col_ref
    else:
        anteriores = [c for c in cols_data if c <= col_ref]
        if not anteriores:
            return {}
        col_alvo = max(anteriores)
    out = {}
    for _, row in df.iterrows():
        ativo = str(row["Assets"])
        v = _parse_ptbr(row[col_alvo])
        if v is not None and v > 0:
            out[ativo] = v
    return out


def carregar_posicoes_atuais(data_ref: pd.Timestamp,
                              basefundos: dict | None = None) -> dict:
    """Retorna {ativo: qty_cumulativa} na data_ref.

    Se basefundos nao dado, chama load_basefundos do cota_portfolio_core (Supabase-first).
    """
    if basefundos is None:
        sys.path.insert(0, str(Path(__file__).parent))
        from cota_portfolio_core import load_basefundos
        basefundos = load_basefundos()

    data_ref_str = pd.Timestamp(data_ref).strftime("%Y-%m-%d")
    qty = {}
    for fundo, df in basefundos.items():
        if str(fundo).upper() == "TOTAL":
            continue
        cols_qtd = [c for c in df.columns if c.endswith("Quantidade")]
        for col_q in cols_qtd:
            try:
                data_col = col_q.split()[0]
            except Exception:
                continue
            if data_col > data_ref_str:
                continue
            for ativo in df.index:
                try:
                    v = df.at[ativo, col_q]
                    if pd.notna(v) and float(v) != 0:
                        qty[ativo] = qty.get(ativo, 0.0) + float(v)
                except Exception:
                    continue
    # remove posicoes liquidas zero
    return {a: q for a, q in qty.items() if abs(q) > 1e-6}


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo do VaR
# ─────────────────────────────────────────────────────────────────────────────

def _quantile_ewma(series: pd.Series, alpha: float, lambda_: float) -> float:
    """VaR nao-parametrico com pesos EWMA (age-weighted).
    Mais peso a observacoes recentes; retorna quantile ponderado.
    """
    s = series.dropna().astype(float).values
    n = len(s)
    if n < 30:
        return float("nan")
    idx = np.arange(n)
    # pesos crescentes com o tempo (mais recente = maior peso)
    w = (lambda_ ** (n - 1 - idx)) * (1 - lambda_)
    w = w / w.sum()
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    pos = int(min(np.searchsorted(cdf, alpha), n - 1))
    return float(s_sorted[pos])


def calcular_var_carteira(retornos_df: pd.DataFrame,
                           qty_dict: dict,
                           precos_dict: dict,
                           alpha: float = ALPHA_DEFAULT,
                           metodos: tuple = ("hist", "ewma"),
                           lambda_ewma: float = LAMBDA_EWMA_DEFAULT) -> dict:
    """Calcula VaR e CVaR de carteira aplicando retornos historicos na posicao atual.

    Retorna dict com:
        mv_total          : valor de mercado total (R$)
        port_ret_series   : serie de P&L simulado (retorno %)
        var_hist_ret      : quantile 5% da serie (equal-weight)
        var_hist_R        : var em R$ (ret * mv_total)
        cvar_hist_R       : CVaR em R$
        var_ewma_ret      : VaR com pesos EWMA
        var_ewma_R        : VaR EWMA em R$
        ativos_usados     : list dos ativos com posicao + retorno + preco
    """
    # 1) Ativos com posicao != 0 que estao em retornos_df E em precos_dict
    ativos_com_pos = [a for a in qty_dict.keys() if a in retornos_df.columns and a in precos_dict]

    if not ativos_com_pos:
        return {
            "mv_total": 0.0,
            "port_ret_series": pd.Series(dtype=float),
            "var_hist_ret": 0.0, "var_hist_R": 0.0, "cvar_hist_R": 0.0,
            "var_ewma_ret": 0.0, "var_ewma_R": 0.0,
            "ativos_usados": [],
        }

    # 2) MV por ativo e pesos
    mv = pd.Series({a: qty_dict[a] * precos_dict[a] for a in ativos_com_pos})
    mv_total = float(mv.abs().sum())    # abs pra evitar cancelamento de long/short
    if mv_total == 0:
        pesos = pd.Series(np.ones(len(ativos_com_pos)) / len(ativos_com_pos), index=ativos_com_pos)
    else:
        pesos = (mv / mv_total).astype(float)

    # 3) Serie de retornos ponderados (P&L simulado do portfolio)
    ret_hist = retornos_df[ativos_com_pos].dropna(how="all")
    # Preenche NaN com 0 (ativo sem retorno naquele dia = sem contribuicao)
    ret_hist = ret_hist.fillna(0.0)
    port_ret = (ret_hist * pesos).sum(axis=1)

    out = {
        "mv_total": mv_total,
        "port_ret_series": port_ret,
        "ativos_usados": ativos_com_pos,
    }

    # 4) VaR hist (quantile empirico)
    if "hist" in metodos and len(port_ret) >= 30:
        var_hist = float(np.quantile(port_ret.values, alpha))
        cvar_hist = float(port_ret[port_ret <= var_hist].mean()) if (port_ret <= var_hist).any() else var_hist
        out["var_hist_ret"] = abs(var_hist)
        out["var_hist_R"]   = abs(var_hist * mv_total)
        out["cvar_hist_R"]  = abs(cvar_hist * mv_total)
    else:
        out["var_hist_ret"] = 0.0
        out["var_hist_R"]   = 0.0
        out["cvar_hist_R"]  = 0.0

    # 5) VaR EWMA (quantile ponderado)
    if "ewma" in metodos and len(port_ret) >= 30:
        var_ewma = _quantile_ewma(port_ret, alpha, lambda_ewma)
        if not np.isnan(var_ewma):
            out["var_ewma_ret"] = abs(var_ewma)
            out["var_ewma_R"]   = abs(var_ewma * mv_total)
        else:
            out["var_ewma_ret"] = 0.0
            out["var_ewma_R"]   = 0.0
    else:
        out["var_ewma_ret"] = 0.0
        out["var_ewma_R"]   = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orquestrador
# ─────────────────────────────────────────────────────────────────────────────

def calcular_var_completo(data_ref: pd.Timestamp,
                           pl_total: float | None = None,
                           basefundos: dict | None = None,
                           janela_dias: int = JANELA_DIAS_DEFAULT,
                           alpha: float = ALPHA_DEFAULT,
                           lambda_ewma: float = LAMBDA_EWMA_DEFAULT,
                           limite_pct_pl: float = 0.0001,   # 1 bp default
                           base_dir: str | None = None) -> dict:
    """Calcula VaR carteira completo para data_ref.

    limite_pct_pl: fracao do PL para limite de risco (1bp = 0.0001).
    Retorna dict pronto para gravar no snapshot_diario.
    """
    if base_dir:
        cwd0 = os.getcwd()
        os.chdir(base_dir)
    else:
        cwd0 = None

    try:
        # 1) Posicoes na data
        qty_dict = carregar_posicoes_atuais(data_ref, basefundos=basefundos)
        if not qty_dict:
            return {"erro": "sem_posicoes"}

        # 2) Precos na data
        precos_dict = carregar_precos_atuais(data_ref)
        if not precos_dict:
            return {"erro": "sem_precos"}

        # 3) Retornos historicos (janela)
        retornos_df = carregar_retornos_historicos(data_ref,
                                                     ativos=list(qty_dict.keys()),
                                                     janela_dias=janela_dias)
        if retornos_df.empty:
            return {"erro": "sem_retornos_historicos"}

        # 4) VaR
        res = calcular_var_carteira(retornos_df, qty_dict, precos_dict,
                                     alpha=alpha, lambda_ewma=lambda_ewma)

        # 5) Metricas normalizadas
        limite_R = float(pl_total) * limite_pct_pl if pl_total else None
        consumo_hist = (res["var_hist_R"] / limite_R) if limite_R else None
        consumo_ewma = (res["var_ewma_R"] / limite_R) if limite_R else None

        return {
            "data_ref":            pd.Timestamp(data_ref).date().isoformat(),
            "n_ativos":            len(res["ativos_usados"]),
            "ativos":              res["ativos_usados"],
            "mv_total":            res["mv_total"],
            "pl_total":            pl_total,
            "limite_R":            limite_R,
            "limite_bps":          limite_pct_pl * 10_000,
            "var_hist_ret":        res["var_hist_ret"],
            "var_hist_R":          res["var_hist_R"],
            "cvar_hist_R":         res["cvar_hist_R"],
            "consumo_hist_pct":    consumo_hist,
            "var_ewma_ret":        res["var_ewma_ret"],
            "var_ewma_R":          res["var_ewma_R"],
            "consumo_ewma_pct":    consumo_ewma,
            "janela_dias":         janela_dias,
            "lambda_ewma":         lambda_ewma,
            "alpha":               alpha,
        }
    finally:
        if cwd0:
            os.chdir(cwd0)


# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CoVaR por ativo (Euler / Component VaR via historical decomposition)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_covar_ativo(retornos_df: pd.DataFrame,
                          qty_dict: dict,
                          precos_dict: dict,
                          alpha: float = ALPHA_DEFAULT,
                          metodo: str = "hist") -> dict:
    """Decompoe VaR de carteira em contribuicao por ativo (component VaR).

    Metodo historical: pega os cenarios da cauda alpha% e computa a contribuicao
    media de cada ativo pra perda observada nesses dias.
      CoVaR_i = mean over tail of (w_i * r_i * MV_total)

    Retorna dict:
      {
        "covar_por_ativo_R": {ativo: contribuicao_media_R$},
        "covar_por_ativo_bps": {ativo: bps sobre MV},
        "covar_por_classe_R": {classe: sum_R$},
        "covar_por_classe_pct": {classe: fracao do CoVaR total},
        "var_estimado_R": soma_das_contribuicoes,
      }
    """
    ativos_com_pos = [a for a in qty_dict.keys() if a in retornos_df.columns and a in precos_dict]
    if not ativos_com_pos:
        return {"erro": "sem_posicoes"}

    mv = pd.Series({a: qty_dict[a] * precos_dict[a] for a in ativos_com_pos})
    mv_total = float(mv.abs().sum())
    if mv_total == 0:
        return {"erro": "mv_zero"}
    pesos = (mv / mv_total).astype(float)

    ret_hist = retornos_df[ativos_com_pos].dropna(how="all").fillna(0.0)
    # P&L simulado do portfolio por dia (fracao do MV)
    port_ret = (ret_hist * pesos).sum(axis=1)

    n = len(port_ret)
    if n < 30:
        return {"erro": "historico_insuficiente"}

    # Component VaR via Euler: usa janela de cenarios em torno do quantil VaR
    # (nao a cauda inteira, que seria CVaR/ES). Assim sum(CoVaR_i) ~= VaR.
    sorted_ret = port_ret.sort_values()
    idx_var = int(len(sorted_ret) * alpha)
    window = 3   # cenarios em torno do quantil VaR pra suavizar ruido
    lo = max(0, idx_var - window)
    hi = min(len(sorted_ret), idx_var + window + 1)
    scenarios = sorted_ret.iloc[lo:hi].index

    # Contribuicao SIGNED por ativo: w_i * E[r_i | port ~ VaR] * MV_total
    # (long+short netta corretamente; sum ~= -VaR)
    contrib_signed = {}
    for a in ativos_com_pos:
        r_i_win = ret_hist.loc[scenarios, a].mean()
        contrib_signed[a] = float(pesos[a] * r_i_win * mv_total)

    # Total signed ~= -VaR (perda esperada no quantil)
    sum_signed = sum(contrib_signed.values())
    var_estimado = abs(sum_signed)   # em R$

    # Reporta CoVaR por ativo como valor SIGNED (positivo = adiciona a perda; negativo = hedge)
    # Convencao: perda = valor positivo (multiplico por -1 do signed original)
    contrib_R_abs = {a: -v for a, v in contrib_signed.items()}

    # Por classe
    def _classe(a):
        au = str(a).upper()
        if au.startswith("DI_") or au.startswith("DI"): return "Juros Nominais BR"
        if au.startswith(("DAP", "NTNB")): return "Juros Reais BR"
        if "TREASURY" in au: return "Juros US"
        if au.startswith("WDO"): return "Moeda"
        return "Outros"

    classe_R = {}
    for a, v in contrib_R_abs.items():
        c = _classe(a)
        classe_R[c] = classe_R.get(c, 0.0) + v

    total = sum(classe_R.values())
    classe_pct = {c: (v/total if total else 0) for c, v in classe_R.items()}
    covar_bps = {a: (v/mv_total*10_000) for a, v in contrib_R_abs.items()}

    return {
        "covar_por_ativo_R":     contrib_R_abs,
        "covar_por_ativo_bps":   covar_bps,
        "covar_por_classe_R":    classe_R,
        "covar_por_classe_pct":  classe_pct,
        "var_estimado_R":        var_estimado,
        "n_scenarios_tail":      len(scenarios),
        "mv_total":              mv_total,
        "metodo":                metodo,
    }


def calcular_covar_completo(data_ref: pd.Timestamp,
                             basefundos: dict | None = None,
                             janela_dias: int = JANELA_DIAS_DEFAULT,
                             alpha: float = ALPHA_DEFAULT,
                             base_dir: str | None = None) -> dict:
    """Orquestrador — carrega posicoes/precos/retornos e chama calcular_covar_ativo."""
    if base_dir:
        cwd0 = os.getcwd(); os.chdir(base_dir)
    else:
        cwd0 = None
    try:
        qty_dict = carregar_posicoes_atuais(data_ref, basefundos=basefundos)
        if not qty_dict: return {"erro": "sem_posicoes"}
        precos_dict = carregar_precos_atuais(data_ref)
        if not precos_dict: return {"erro": "sem_precos"}
        retornos_df = carregar_retornos_historicos(data_ref,
                                                     ativos=list(qty_dict.keys()),
                                                     janela_dias=janela_dias)
        if retornos_df.empty: return {"erro": "sem_retornos"}
        return calcular_covar_ativo(retornos_df, qty_dict, precos_dict, alpha=alpha)
    finally:
        if cwd0: os.chdir(cwd0)



# CLI — teste rapido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Data ref YYYY-MM-DD (default: hoje)")
    ap.add_argument("--pl", type=float, default=None, help="PL total (R$)")
    ap.add_argument("--janela", type=int, default=JANELA_DIAS_DEFAULT)
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()

    data_ref = pd.Timestamp(args.data) if args.data else pd.Timestamp.today().normalize()
    res = calcular_var_completo(
        data_ref=data_ref,
        pl_total=args.pl,
        janela_dias=args.janela,
        base_dir=args.base_dir,
    )
    print("-" * 78)
    if "erro" in res:
        print(f"ERRO: {res['erro']}")
    else:
        print(f"Data ref     : {res['data_ref']}")
        print(f"# ativos     : {res['n_ativos']}  {res['ativos']}")
        print(f"MV total     : R$ {res['mv_total']:,.2f}")
        if res['pl_total']:
            print(f"PL total     : R$ {res['pl_total']:,.2f}")
            print(f"Limite ({res['limite_bps']:.1f}bp): R$ {res['limite_R']:,.2f}")
        print()
        print(f"VaR HIST 5%  : R$ {res['var_hist_R']:>15,.2f}   ({res['var_hist_ret']*100:.3f}% do MV)")
        if res.get('consumo_hist_pct') is not None:
            print(f"  consumo    : {res['consumo_hist_pct']*100:.1f}% do limite")
        print(f"CVaR HIST    : R$ {res['cvar_hist_R']:>15,.2f}")
        print()
        print(f"VaR EWMA λ={res['lambda_ewma']:.2f}: R$ {res['var_ewma_R']:>15,.2f}   ({res['var_ewma_ret']*100:.3f}% do MV)")
        if res.get('consumo_ewma_pct') is not None:
            print(f"  consumo    : {res['consumo_ewma_pct']*100:.1f}% do limite")
        print()
        print(f"Janela: {res['janela_dias']} dias  |  alpha={res['alpha']}")
    print("-" * 78)
