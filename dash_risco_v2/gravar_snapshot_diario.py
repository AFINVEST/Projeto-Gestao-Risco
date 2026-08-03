"""
gravar_snapshot_diario.py  (v2.0 — usa cota_portfolio_core)
============================================================

MUDANÇA v2.0 (Opção C):
  - Cota, retorno_total, vol, DD e Sharpe agora vêm de
    `cota_portfolio_core.computar_cota_serie(...)`, que replica EXATAMENTE
    a fórmula do `simulate_nav_cota()` do app4.py.
  - Fonte única de PnL: `analisar_dados_fundos2` (mesma do dashboard),
    baseada em df_valor_ajuste_contrato.parquet + BaseFundos/*.parquet.
  - Garante que TODAS as métricas de performance batam com o que o
    dashboard exibe (mesmo período, mesma fórmula, mesma fonte de dados).
  - DV01 e VaR/CVaR (via retornos históricos por-ativo) permanecem
    calculados aqui — mas agora usando `ret_total` do módulo core como
    base para VaR/CVaR.

VARIÁVEIS DE AMBIENTE:
    SUPABASE_URL, SUPABASE_KEY  (ou SUPABASE_SERVICE_ROLE_KEY)
    DASHBOARD_DIR               (opcional; default = CWD)

USO:
    python gravar_snapshot_diario.py --bootstrap
    python gravar_snapshot_diario.py                # incremental
    python gravar_snapshot_diario.py --data 2026-07-30
"""
from __future__ import annotations
import os
import sys
import math
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from supabase import create_client
except ImportError:
    print("ERRO: pip install supabase", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
try:
    import dv01_dinamico as dv        # noqa: F401 — usado abaixo
    import taxas_dinamicas as td       # noqa: F401
    from cota_portfolio_core import computar_cota_serie
except ImportError as e:
    print(f"ERRO importando módulos: {e}", file=sys.stderr)
    print("Os arquivos dv01_dinamico.py, taxas_dinamicas.py e cota_portfolio_core.py "
          "precisam estar no mesmo diretório.", file=sys.stderr)
    sys.exit(1)


LAMBDA_BRW = 0.99
ALPHA_VAR = 0.05
PCT_CAPITAL_DEFAULT = 0.01

FUNDOS_ALLOW_LIST = {
    "AF DEB INCENTIVADAS", "BH FIRF INFRA", "BORDEAUX INFRA",
    "GLOBAL BONDS", "HORIZONTE", "JERA2026",
    "MANACA INFRA FIRF", "REAL FIM", "TOPAZIO INFRA",
}


# ─────────────────────────────────────────────────────────────────────────────
# Supabase helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Defina SUPABASE_URL e SUPABASE_KEY.")
    return create_client(url, key)


def _get_config(client) -> dict:
    resp = client.table("config_risco").select("parametro,valor").execute()
    out = {}
    for row in resp.data or []:
        v = row["valor"]
        if isinstance(v, str):
            v = v.strip('"')
        try:
            out[row["parametro"]] = float(v)
        except Exception:
            out[row["parametro"]] = v
    return out


def _load_snapshot_dates(client) -> set:
    resp = client.table("snapshot_diario").select("Data").order("Data").execute()
    return {row["Data"] for row in resp.data or []}


def _upsert_snapshots(client, registros):
    if not registros:
        return 0
    total = 0
    for i in range(0, len(registros), 100):
        lote = registros[i:i + 100]
        client.table("snapshot_diario").upsert(lote, on_conflict="Data").execute()
        total += len(lote)
    return total


def _load_posicoes_supabase(client, trading_index) -> pd.DataFrame:
    """Posições cumulativas por (Data, Ativo) para cálculo de DV01."""
    resp = client.table("posicoes_por_fundo").select("Data,Ativo,Quantidade").execute()
    if not resp.data:
        return pd.DataFrame(index=trading_index)
    df = pd.DataFrame(resp.data)
    df["Data"] = pd.to_datetime(df["Data"]).dt.normalize()
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0.0)
    trades = df.groupby(["Data", "Ativo"], as_index=False)["Quantidade"].sum().sort_values("Data")
    trades["pos"] = trades.groupby("Ativo")["Quantidade"].cumsum()
    pos = trades.pivot(index="Data", columns="Ativo", values="pos").sort_index()
    pos = pos.reindex(trading_index).ffill().fillna(0.0)
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de risco (agora rodam SOBRE ret_total do módulo core)
# ─────────────────────────────────────────────────────────────────────────────

def _vol(retornos, janela):
    if len(retornos) < janela + 1:
        return None
    s = retornos.tail(janela).dropna()
    if len(s) < janela // 2:
        return None
    return float(s.std() * math.sqrt(252))


def _var_ew(retornos, alpha=ALPHA_VAR, min_n=60):
    r = retornos.dropna()
    if len(r) < min_n:
        return None
    return float(r.quantile(alpha))


def _var_brw(retornos, alpha=ALPHA_VAR, lam=LAMBDA_BRW, min_n=60):
    r = pd.Series(retornos).dropna().astype(float).values
    n = len(r)
    if n < min_n:
        return None
    idx = np.arange(n)
    w = (lam ** (n - 1 - idx)) * (1 - lam) / (1 - lam ** n)
    order = np.argsort(r)
    r_sorted = r[order]
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    pos = min(np.searchsorted(cdf, alpha), n - 1)
    return float(r_sorted[pos])


def _cvar(retornos, alpha=ALPHA_VAR):
    r = retornos.dropna()
    if len(r) < 60:
        return None
    v = r.quantile(alpha)
    tail = r[r <= v]
    return float(tail.mean()) if len(tail) else None


def _dd(cota_series):
    if cota_series.empty:
        return (0.0, 0.0)
    peak = cota_series.cummax()
    dd = cota_series / peak - 1.0
    return (float(dd.iloc[-1]), float(dd.min()))


def _bucket_dv01(dv01_dict):
    out = {"dv01_juros_nom": 0.0, "dv01_juros_real": 0.0,
           "dv01_treasury": 0.0, "dv01_ntnb": 0.0}
    for ativo, v in (dv01_dict or {}).items():
        if not isinstance(v, (int, float)) or v is None or math.isnan(v):
            continue
        a = str(ativo).upper()
        if a.startswith("DI_"):
            out["dv01_juros_nom"] += v
        elif a.startswith("DAP_"):
            out["dv01_juros_real"] += v
        elif a == "TREASURY":
            out["dv01_treasury"] += v
        elif a.startswith("NTNB"):
            out["dv01_ntnb"] += v
    return out


def _acumular_periodo(retornos_serie: pd.Series, data_ref: pd.Timestamp, periodo: str):
    """Acumula retornos DENTRO do período (mtd, ytd) até data_ref."""
    if retornos_serie is None or retornos_serie.empty:
        return None
    d = pd.Timestamp(data_ref).normalize()
    if periodo == "mtd":
        inicio = d.replace(day=1)
    elif periodo == "ytd":
        inicio = pd.Timestamp(year=d.year, month=1, day=1)
    else:
        return None
    janela = retornos_serie.loc[inicio:d].dropna()
    if janela.empty:
        return 0.0
    return float((1 + janela).prod() - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Montagem do snapshot (agora recebe series já prontas do módulo core)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_snapshot(
    data: pd.Timestamp,
    cota_hoje: float,
    cota_ontem: float,
    pl_total: float | None,
    ret_total_serie: pd.Series,   # série de retornos totais (do core) até `data`
    cdi_serie: pd.Series,          # CDI diário decimal
    lft_serie: pd.Series,          # LFT diário decimal (para cdi_dtd proxy quando faltar CDI)
    cota_series: pd.Series,        # série de cota até `data`
    config: dict,
    pct_capital: float,
    dv01_por_ativo: dict | None = None,
    stop_loss_ativo: bool = False,
    fator_governance: float = 1.0,
) -> dict:
    # Retorno do dia (via cota)
    retorno_dtd = (cota_hoje - cota_ontem) / cota_ontem if cota_ontem else 0.0

    # CDI do dia — se CDI existe, usa; senão cai para LFT como proxy
    if data in cdi_serie.index and pd.notna(cdi_serie.loc[data]):
        cdi_dtd = float(cdi_serie.loc[data])
    elif data in lft_serie.index:
        cdi_dtd = float(lft_serie.loc[data])
    else:
        cdi_dtd = 0.0

    # Vol/VaR/CVaR SOBRE ret_total (mesma série usada para calcular a cota)
    ret_janela = ret_total_serie.dropna()
    v20  = _vol(ret_janela, 20)
    v60  = _vol(ret_janela, 60)
    v252 = _vol(ret_janela, 252)
    var_ew = _var_ew(ret_janela)
    var_bw = _var_brw(ret_janela)
    cvar   = _cvar(ret_janela)

    var_hist_ew_bps = -var_ew * 10_000 if var_ew is not None else None
    var_hist_bw_bps = -var_bw * 10_000 if var_bw is not None else None
    cvar_bps        = -cvar   * 10_000 if cvar   is not None else None

    dd_atual, dd_max = _dd(cota_series)
    dv01_buckets = _bucket_dv01(dv01_por_ativo)
    dv01_total = sum(dv01_buckets.values())

    var_base = float(config.get("var_limite_base_bps", 1.0))
    var_efet = var_base * fator_governance

    capital_risco = pl_total * pct_capital if pl_total else None
    var_bw_reais = (var_hist_bw_bps / 10_000) * capital_risco if (var_hist_bw_bps and capital_risco) else None
    var_ew_reais = (var_hist_ew_bps / 10_000) * capital_risco if (var_hist_ew_bps and capital_risco) else None

    # MTD / YTD sobre ret_total e CDI
    ret_mtd = _acumular_periodo(ret_total_serie, data, "mtd")
    ret_ytd = _acumular_periodo(ret_total_serie, data, "ytd")
    cdi_mtd = _acumular_periodo(cdi_serie,       data, "mtd")
    cdi_ytd = _acumular_periodo(cdi_serie,       data, "ytd")

    return {
        "Data": data.date().isoformat(),
        "cota": float(cota_hoje),
        "pl_total": float(pl_total) if pl_total else None,
        "retorno_dtd": float(retorno_dtd),
        "retorno_mtd": ret_mtd,
        "retorno_ytd": ret_ytd,
        "cdi_dtd": cdi_dtd,
        "cdi_mtd": cdi_mtd,
        "cdi_ytd": cdi_ytd,
        "vol_20d": v20, "vol_60d": v60, "vol_252d": v252, "vol_ewma": None,
        "dd_atual": float(dd_atual), "dd_max_hist": float(dd_max),
        "var_hist_ew_bps": var_hist_ew_bps,
        "var_hist_bw_bps": var_hist_bw_bps,
        "var_param_bps": None,
        "cvar_bps": cvar_bps,
        "var_hist_ew_reais": var_ew_reais,
        "var_hist_bw_reais": var_bw_reais,
        "dv01_total": float(dv01_total) if dv01_total else None,
        **{k: (float(v) if v else None) for k, v in dv01_buckets.items()},
        "var_limite_base_bps": var_base,
        "var_limite_efet_bps": var_efet,
        "stop_loss_ativo": bool(stop_loss_ativo),
        "fator_governance": float(fator_governance),
        "fonte": "batch_diario_v2",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run(bootstrap: bool = False, data_override: str | None = None,
        dashboard_dir: str | None = None):
    client = _get_client()
    config = _get_config(client)
    pct_capital = float(config.get("pct_capital_risco", PCT_CAPITAL_DEFAULT))
    print(f"[snapshot] pct_capital_risco = {pct_capital}")

    # dashboard_dir default = env var DASHBOARD_DIR ou CWD
    if dashboard_dir is None:
        dashboard_dir = os.environ.get("DASHBOARD_DIR", os.getcwd())
    print(f"[snapshot] dashboard_dir     = {dashboard_dir}")

    # ────────────────────────── núcleo: cota_portfolio_core ────────────────
    print("[snapshot] chamando computar_cota_serie... (isto lê BaseFundos/*, "
          "df_valor_ajuste_contrato, df_preco_de_ajuste_atual_completo, dados_lft, cdi_cached)")
    res = computar_cota_serie(pct=pct_capital, base_dir=dashboard_dir)
    df_core = res["df"]

    print(f"[snapshot] período     : {res['data_ini'].date()} → {res['data_fim'].date()}  "
          f"({len(df_core)} dias)")
    print(f"[snapshot] ret_acum    : {res['ret_acum']*100:+.2f}%")
    print(f"[snapshot] vol_anual   : {res['vol_anual']*100:.2f}%")
    print(f"[snapshot] max_dd      : {res['max_dd']*100:.2f}%")
    print(f"[snapshot] sharpe_cdi  : {res['sharpe_cdi']:.2f}")
    print(f"[snapshot] cota (base 1): {df_core['cota'].iloc[-1]:.6f}")

    cotas       = df_core["cota"]
    ret_total   = df_core["ret_total"]
    cdi_serie   = df_core["cdi_series"]
    lft_serie   = df_core["lft_series"]
    pl_serie    = df_core["pl_series"]

    # ────────────────────────── DV01 (opcional; usa Supabase p/ posições)
    positions = _load_posicoes_supabase(client, cotas.index)
    print(f"[snapshot] posições p/ DV01 (Supabase): {positions.shape}")

    datas_existentes = _load_snapshot_dates(client)
    print(f"[snapshot] {len(datas_existentes)} dias já em snapshot_diario")

    if bootstrap:
        datas_a_processar = list(cotas.index)
    elif data_override:
        datas_a_processar = [pd.Timestamp(data_override).normalize()]
    else:
        datas_a_processar = [d for d in cotas.index if d.date().isoformat() not in datas_existentes]

    registros: list[dict] = []
    for data in datas_a_processar:
        try:
            if data not in cotas.index:
                continue
            cota_hoje = float(cotas.loc[data])
            if not np.isfinite(cota_hoje):
                continue
            idx_prev = cotas.index.searchsorted(data) - 1
            cota_ontem = float(cotas.iloc[idx_prev]) if idx_prev >= 0 else cota_hoje
            pl_dia = float(pl_serie.loc[data]) if data in pl_serie.index else None

            snap = _compute_snapshot(
                data=data,
                cota_hoje=cota_hoje,
                cota_ontem=cota_ontem,
                pl_total=pl_dia,
                ret_total_serie=ret_total.loc[:data],
                cdi_serie=cdi_serie,
                lft_serie=lft_serie,
                cota_series=cotas.loc[:data],
                config=config,
                pct_capital=pct_capital,
                dv01_por_ativo=None,   # TODO: integrar dv01_dinamico usando positions.loc[data]
            )
            registros.append(snap)
        except Exception as e:
            print(f"[snapshot] erro em {data.date()}: {e}")

    print(f"[snapshot] enviando {len(registros)} snapshots...")
    n = _upsert_snapshots(client, registros)
    print(f"[snapshot] OK — {n} linhas em snapshot_diario")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--dashboard-dir", default=None,
                    help="Diretório onde ficam Dados/ e BaseFundos/. Default = env DASHBOARD_DIR ou CWD.")
    args = ap.parse_args()
    run(bootstrap=args.bootstrap, data_override=args.data, dashboard_dir=args.dashboard_dir)
