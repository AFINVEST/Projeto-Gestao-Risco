"""
gravar_snapshot_diario.py  (v2.2 — VaR de carteira histórico + EWMA)
======================================================================

MUDANCAS v2.2:
  - Popula var_carteira_hist_reais / _bps (P&L 5% percentile da carteira
    do dia sobre retornos historicos até o dia — sem lookback).
  - Popula var_carteira_ewma_reais / _bps (mesmo, com pesos EWMA lambda=0.99).
  - Popula consumo_hist_pct e consumo_ewma_pct (VaR / limite_1bp).
  - Popula mv_total_carteira e n_ativos_carteira.
  - Usa novo modulo `risco_carteira_core` (Fase B).
  - Requer tabela `retornos_diarios_ativo` populada (Fase A).

MUDANCAS v2.1:
  - Popula dv01_juros_nom / juros_real / treasury / ntnb / total via
    dv01_dinamico + posicoes correntes.
  - Cota, retorno_dtd, cdi_dtd via cota_portfolio_core.

USO:
    python gravar_snapshot_diario.py --bootstrap        # re-processa tudo (recomendado uma vez)
    python gravar_snapshot_diario.py                    # incremental (diario no .bat)
    python gravar_snapshot_diario.py --data 2026-07-31
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
    from cota_portfolio_core import (
        computar_cota_serie, load_basefundos, _alias_ativo_novo,
    )
except ImportError as e:
    print(f"ERRO importando cota_portfolio_core: {e}", file=sys.stderr)
    sys.exit(1)

# dv01_dinamico é opcional — se não instalado, DV01 fica None
try:
    import dv01_dinamico as dv
    HAS_DV01 = True
except ImportError:
    HAS_DV01 = False
    print("[snapshot] AVISO: dv01_dinamico nao encontrado. DV01 ficara None.")

# risco_carteira_core é opcional — se não instalado, VaR carteira fica None
try:
    from risco_carteira_core import calcular_var_completo
    HAS_VAR_CARTEIRA = True
except ImportError:
    HAS_VAR_CARTEIRA = False
    print("[snapshot] AVISO: risco_carteira_core nao encontrado. VaR carteira ficara None.")


LAMBDA_BRW = 0.99
ALPHA_VAR = 0.05
PCT_CAPITAL_DEFAULT = 0.01


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


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de risco
# ─────────────────────────────────────────────────────────────────────────────

def _vol(retornos, janela):
    if len(retornos) < janela + 1: return None
    s = retornos.tail(janela).dropna()
    if len(s) < janela // 2: return None
    return float(s.std() * math.sqrt(252))

def _var_ew(retornos, alpha=ALPHA_VAR, min_n=60):
    r = retornos.dropna()
    if len(r) < min_n: return None
    return float(r.quantile(alpha))

def _var_brw(retornos, alpha=ALPHA_VAR, lam=LAMBDA_BRW, min_n=60):
    r = pd.Series(retornos).dropna().astype(float).values
    n = len(r)
    if n < min_n: return None
    idx = np.arange(n)
    w = (lam ** (n - 1 - idx)) * (1 - lam) / (1 - lam ** n)
    order = np.argsort(r)
    r_sorted = r[order]; w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    pos = min(np.searchsorted(cdf, alpha), n - 1)
    return float(r_sorted[pos])

def _cvar(retornos, alpha=ALPHA_VAR):
    r = retornos.dropna()
    if len(r) < 60: return None
    v = r.quantile(alpha)
    tail = r[r <= v]
    return float(tail.mean()) if len(tail) else None

def _dd(cota_series):
    if cota_series.empty: return (0.0, 0.0)
    peak = cota_series.cummax()
    dd = cota_series / peak - 1.0
    return (float(dd.iloc[-1]), float(dd.min()))

def _acumular_periodo(retornos_serie: pd.Series, data_ref: pd.Timestamp, periodo: str):
    if retornos_serie is None or retornos_serie.empty: return None
    d = pd.Timestamp(data_ref).normalize()
    if periodo == "mtd":
        inicio = d.replace(day=1)
    elif periodo == "ytd":
        inicio = pd.Timestamp(year=d.year, month=1, day=1)
    else:
        return None
    janela = retornos_serie.loc[inicio:d].dropna()
    if janela.empty: return 0.0
    return float((1 + janela).prod() - 1)


# ─────────────────────────────────────────────────────────────────────────────
# DV01 por classe (para a última data)
# ─────────────────────────────────────────────────────────────────────────────

def _classe(ativo: str) -> str:
    a = str(ativo).upper()
    if a.startswith("DI_"):
        return "juros_nom"
    if a.startswith("DAP") or a.startswith("NTNB"):
        return "juros_real" if a.startswith("DAP") else "ntnb"
    if a == "TREASURY":
        return "treasury"
    return "outros"


PROJECAO_IPCA_DEFAULT = 0.05   # % ao mês (default do CLI de dv01_dinamico)


def _parse_ptbr(v):
    """Converte 'R$ 83.245,19' ou '83.245,19' -> 83245.19."""
    if v is None: return None
    try:
        if pd.isna(v): return None
    except Exception: pass
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().replace("R$", "").replace(" ", "")
    if not s or s in ("-", "--"): return None
    # Remove separador de milhar '.' e troca ',' por '.'
    s2 = s.replace(".", "").replace(",", ".")
    try: return float(s2)
    except Exception: return None


def _carregar_pu_por_ativo(data_ref: pd.Timestamp,
                            path: str = "Dados/df_preco_de_ajuste_atual_completo.parquet") -> dict:
    """Le PU (Preco Unitario) por ativo na data_ref (ou a mais proxima anterior).
    Aceita valores em pt-BR ('83.245,19')."""
    try:
        df = pd.read_parquet(path)
        # 'Assets' + colunas-data. Escolhe data_ref se existe, senao a mais recente <= data_ref
        col_ref = data_ref.strftime("%Y-%m-%d")
        # colunas de data (strings YYYY-MM-DD)
        cols_data = [c for c in df.columns if c != "Assets"]
        # tenta match exato
        if col_ref in cols_data:
            col_alvo = col_ref
        else:
            # pega a mais recente <= data_ref
            anteriores = [c for c in cols_data if c <= col_ref]
            if not anteriores:
                print(f"[dv01] nenhuma coluna <= {col_ref} em {path}")
                return {}
            col_alvo = max(anteriores)
        print(f"[dv01] usando PU da coluna: {col_alvo}")
        out = {}
        for _, row in df.iterrows():
            ativo = str(row["Assets"])
            v = _parse_ptbr(row[col_alvo])
            if v is not None and v > 0:
                out[ativo] = v
        return out
    except Exception as e:
        print(f"[dv01] erro carregando {path}: {e}")
        return {}


def _dv01_hoje(basefundos: dict, data_ref: pd.Timestamp,
               projecao_ipca_pct: float = PROJECAO_IPCA_DEFAULT) -> dict:
    """Calcula DV01 por classe para a data de referência (DI e DAP apenas —
    dv01_dinamico não suporta NTNB/TREASURY)."""
    if not HAS_DV01:
        return {"dv01_juros_nom": None, "dv01_juros_real": None,
                "dv01_treasury": None, "dv01_ntnb": None, "dv01_total": None}

    # Le PU (Preco Unitario) atualizado do B3, formato pt-BR
    taxa_dict = _carregar_pu_por_ativo(data_ref)
    if not taxa_dict:
        return {"dv01_juros_nom": None, "dv01_juros_real": None,
                "dv01_treasury": None, "dv01_ntnb": None, "dv01_total": None}

    # 1) Agrega qty por ativo (posicao CUMULATIVA = soma de todos os trades <= data_ref)
    #    Colunas "YYYY-MM-DD Quantidade" são DELTAS (trades), não estoque.
    data_ref_str = pd.Timestamp(data_ref).strftime("%Y-%m-%d")
    qty_por_ativo = {}
    for fundo, df in basefundos.items():
        if fundo.upper() == "TOTAL":
            continue
        cols_qtd = [c for c in df.columns if c.endswith("Quantidade")]
        if not cols_qtd:
            continue
        # itera TODAS as colunas de Quantidade até data_ref (posicao = sum(deltas))
        for col_q in cols_qtd:
            try:
                data_col = col_q.split()[0]
            except Exception:
                continue
            if data_col > data_ref_str:
                continue   # trade após data_ref, ignora
            for ativo in df.index:
                try:
                    v = df.at[ativo, col_q]
                    if pd.notna(v) and float(v) != 0:
                        qty_por_ativo[ativo] = qty_por_ativo.get(ativo, 0.0) + float(v)
                except Exception:
                    continue
    # remove ativos com posição líquida zero (opened+closed cancelam)
    qty_por_ativo = {a: q for a, q in qty_por_ativo.items() if abs(q) > 1e-6}
    print(f"[dv01] posicoes cumulativas em {data_ref_str}: {qty_por_ativo}")

    # 2) Vencimentos e DU (para converter PU -> taxa)
    try:
        from dv01_dinamico import vencimento as _vencimento, networkdays as _networkdays, load_feriados as _load_feriados
        _feriados = _load_feriados()
    except Exception as e:
        print(f"[dv01] erro importando helpers de dv01_dinamico: {e}")
        _feriados = None

    # 3) DV01 por ativo x qty
    buckets = {"juros_nom": 0.0, "juros_real": 0.0}
    n_ok, n_skip = 0, 0
    for ativo, qty in qty_por_ativo.items():
        a = str(ativo).upper()
        if not (a.startswith("DI_") or a.startswith("DAP")):
            n_skip += 1
            continue

        pu = taxa_dict.get(ativo)   # df_inicial armazena PU, nao taxa
        if pu is None or pu <= 0:
            print(f"[dv01] sem PU para {ativo} em df_inicial")
            n_skip += 1
            continue

        # DAP nao suportado nesta iteracao (PU em pontos, precisa rs_dap)
        if a.startswith("DAP"):
            print(f"[dv01] {ativo}: DAP nao processado nesta versao (necessita rs_dap/projecao ANBIMA)")
            n_skip += 1
            continue

        try:
            venc = _vencimento(ativo, _feriados)
            du   = _networkdays(data_ref, venc, _feriados)
            if du <= 0:
                print(f"[dv01] {ativo}: DU={du} invalido, pulando")
                n_skip += 1
                continue
            # PU -> taxa (%) via formula DI padrao
            taxa_pct = ((100_000 / float(pu)) ** (252 / du) - 1) * 100

            resultado = dv.calcular_dv01(ativo, taxa_pct, data_ref)
            dv_contrato = float(resultado.get("dv01", 0.0))
            dv_signed = dv_contrato * qty   # SIGNED: long soma, short subtrai (netting da carteira)
            print(f"[dv01] {ativo}: PU={pu:.2f} du={du} taxa={taxa_pct:.4f}% dv_contrato={dv_contrato:.4f} qty={qty:.1f} dv_signed={dv_signed:.2f}")
            c = _classe(ativo)
            if c in buckets:
                buckets[c] += dv_signed
                n_ok += 1
        except Exception as e:
            print(f"[dv01] falha em {ativo}: {e}")
            n_skip += 1
            continue

    print(f"[dv01] {n_ok} ativos calculados, {n_skip} ignorados/erros")
    total = buckets["juros_nom"] + buckets["juros_real"]
    return {
        "dv01_juros_nom":  buckets["juros_nom"]  if buckets["juros_nom"]  > 0 else None,
        "dv01_juros_real": buckets["juros_real"] if buckets["juros_real"] > 0 else None,
        "dv01_treasury":   None,   # não suportado por dv01_dinamico
        "dv01_ntnb":       None,   # não suportado por dv01_dinamico
        "dv01_total":      total if total > 0 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot builder
# ─────────────────────────────────────────────────────────────────────────────

def _compute_snapshot(
    data: pd.Timestamp,
    cota_hoje: float,
    cota_ontem: float,
    pl_total: float | None,
    ret_total_serie: pd.Series,
    cdi_serie: pd.Series,
    lft_serie: pd.Series,
    cota_series: pd.Series,
    config: dict,
    pct_capital: float,
    dv01_dict: dict | None = None,
    var_carteira_dict: dict | None = None,
    stop_loss_ativo: bool = False,
    fator_governance: float = 1.0,
) -> dict:
    retorno_dtd = (cota_hoje - cota_ontem) / cota_ontem if cota_ontem else 0.0

    # SEM FALLBACK LFT: se CDI faltar, cdi_dtd = None (visivel no dashboard/email)
    if data in cdi_serie.index and pd.notna(cdi_serie.loc[data]):
        cdi_dtd = float(cdi_serie.loc[data])
    else:
        cdi_dtd = None
        print(f"[snapshot] AVISO: CDI faltando em {data.date()}. Rode atualizar_cdi_lft.py.")

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

    var_base = float(config.get("var_limite_base_bps", 1.0))
    var_efet = var_base * fator_governance

    capital_risco = pl_total * pct_capital if pl_total else None
    var_bw_reais = (var_hist_bw_bps / 10_000) * capital_risco if (var_hist_bw_bps and capital_risco) else None
    var_ew_reais = (var_hist_ew_bps / 10_000) * capital_risco if (var_hist_ew_bps and capital_risco) else None

    ret_mtd = _acumular_periodo(ret_total_serie, data, "mtd")
    ret_ytd = _acumular_periodo(ret_total_serie, data, "ytd")
    cdi_mtd = _acumular_periodo(cdi_serie,       data, "mtd")
    cdi_ytd = _acumular_periodo(cdi_serie,       data, "ytd")

    dv01_dict = dv01_dict or {}
    var_c = var_carteira_dict or {}

    # VaR de carteira em bps sobre PL total (nao PL_risco). Bate com o conceito de limite (1bp do PL).
    def _to_bps(v_reais):
        if v_reais is None or not pl_total or pl_total == 0:
            return None
        return (v_reais / pl_total) * 10_000

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
        # VaR da COTA (metrica antiga — acompanhamento historico do book realizado)
        "var_hist_ew_bps": var_hist_ew_bps,
        "var_hist_bw_bps": var_hist_bw_bps,
        "var_param_bps": None,
        "cvar_bps": cvar_bps,
        "var_hist_ew_reais": var_ew_reais,
        "var_hist_bw_reais": var_bw_reais,
        # DV01
        "dv01_total":       dv01_dict.get("dv01_total"),
        "dv01_juros_nom":   dv01_dict.get("dv01_juros_nom"),
        "dv01_juros_real":  dv01_dict.get("dv01_juros_real"),
        "dv01_treasury":    dv01_dict.get("dv01_treasury"),
        "dv01_ntnb":        dv01_dict.get("dv01_ntnb"),
        # Limites e governance
        "var_limite_base_bps": var_base,
        "var_limite_efet_bps": var_efet,
        "stop_loss_ativo": bool(stop_loss_ativo),
        "fator_governance": float(fator_governance),
        # VaR da CARTEIRA ATUAL (metrica nova — usada para controle de risco)
        "var_carteira_hist_reais":  var_c.get("var_hist_R"),
        "var_carteira_hist_bps":    _to_bps(var_c.get("var_hist_R")),
        "var_carteira_ewma_reais":  var_c.get("var_ewma_R"),
        "var_carteira_ewma_bps":    _to_bps(var_c.get("var_ewma_R")),
        "cvar_carteira_hist_reais": var_c.get("cvar_hist_R"),
        "consumo_hist_pct":         var_c.get("consumo_hist_pct"),
        "consumo_ewma_pct":         var_c.get("consumo_ewma_pct"),
        "mv_total_carteira":        var_c.get("mv_total"),
        "n_ativos_carteira":        var_c.get("n_ativos"),
        "fonte": "batch_diario_v2.2",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(bootstrap: bool = False, data_override: str | None = None,
        dashboard_dir: str | None = None):
    client = _get_client()
    config = _get_config(client)
    pct_capital = float(config.get("pct_capital_risco", PCT_CAPITAL_DEFAULT))
    print(f"[snapshot] pct_capital_risco = {pct_capital}")

    if dashboard_dir is None:
        dashboard_dir = os.environ.get("DASHBOARD_DIR", os.getcwd())
    print(f"[snapshot] dashboard_dir     = {dashboard_dir}")

    print("[snapshot] computar_cota_serie... (le BaseFundos via Supabase-first)")
    res = computar_cota_serie(pct=pct_capital, base_dir=dashboard_dir)
    df_core = res["df"]

    print(f"[snapshot] periodo    : {res['data_ini'].date()} -> {res['data_fim'].date()}  ({len(df_core)} dias)")
    print(f"[snapshot] ret_acum   : {res['ret_acum']*100:+.2f}%")
    print(f"[snapshot] vol_anual  : {res['vol_anual']*100:.2f}%")
    print(f"[snapshot] max_dd     : {res['max_dd']*100:.2f}%")
    print(f"[snapshot] sharpe_cdi : {res['sharpe_cdi']:.2f}")

    cotas       = df_core["cota"]
    ret_total   = df_core["ret_total"]
    cdi_serie   = df_core["cdi_series"]
    lft_serie   = df_core["lft_series"]
    pl_serie    = df_core["pl_series"]

    # Load basefundos UMA vez (usado para DV01 e VaR de carteira do loop)
    bf_cache = None
    try:
        bf_cache = load_basefundos()
        print(f"[snapshot] BaseFundos carregado 1x para uso no loop ({len(bf_cache)} fundos)")
    except Exception as e:
        print(f"[snapshot] erro pre-load basefundos: {e}")

    # DV01 para a última data (calculado uma única vez, usa parquets atuais)
    dv01_hoje_dict = {}
    if HAS_DV01 and bf_cache:
        try:
            dv01_hoje_dict = _dv01_hoje(bf_cache, res["data_fim"])
            print(f"[snapshot] DV01 total (hoje): R$ {dv01_hoje_dict.get('dv01_total') or 0:,.2f}/bp "
                  f"(juros_nom={dv01_hoje_dict.get('dv01_juros_nom') or 0:.2f}, "
                  f"juros_real={dv01_hoje_dict.get('dv01_juros_real') or 0:.2f}, "
                  f"treasury={dv01_hoje_dict.get('dv01_treasury') or 0:.2f}, "
                  f"ntnb={dv01_hoje_dict.get('dv01_ntnb') or 0:.2f})")
        except Exception as e:
            print(f"[snapshot] erro DV01: {e}")

    datas_existentes = _load_snapshot_dates(client)
    print(f"[snapshot] {len(datas_existentes)} dias ja em snapshot_diario")

    if bootstrap:
        datas_a_processar = list(cotas.index)
    elif data_override:
        datas_a_processar = [pd.Timestamp(data_override).normalize()]
    else:
        datas_a_processar = [d for d in cotas.index if d.date().isoformat() not in datas_existentes]

    data_ultima = res["data_fim"]

    registros: list[dict] = []
    for data in datas_a_processar:
        try:
            if data not in cotas.index: continue
            cota_hoje = float(cotas.loc[data])
            if not np.isfinite(cota_hoje): continue
            idx_prev = cotas.index.searchsorted(data) - 1
            cota_ontem = float(cotas.iloc[idx_prev]) if idx_prev >= 0 else cota_hoje
            pl_dia = float(pl_serie.loc[data]) if data in pl_serie.index else None

            # DV01 só para a última data (senão fica None; evita lookback bias)
            dv01_para_dia = dv01_hoje_dict if data == data_ultima else {}

            # VaR de carteira — para CADA dia (posicoes do dia + retornos ate o dia)
            var_carteira = {}
            if HAS_VAR_CARTEIRA and pl_dia:
                try:
                    var_carteira = calcular_var_completo(
                        data_ref=data,
                        pl_total=pl_dia,
                        basefundos=bf_cache,      # passa cache pra evitar reload
                        janela_dias=756,          # 3 anos (padrao configuravel)
                        limite_pct_pl=0.0001,     # 1 bp
                        base_dir=dashboard_dir,
                    ) or {}
                    if "erro" in var_carteira:
                        var_carteira = {}
                except Exception as e:
                    print(f"[snapshot] erro VaR carteira em {data.date()}: {e}")
                    var_carteira = {}

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
                dv01_dict=dv01_para_dia,
                var_carteira_dict=var_carteira,
            )
            registros.append(snap)
        except Exception as e:
            print(f"[snapshot] erro em {data.date()}: {e}")

    print(f"[snapshot] enviando {len(registros)} snapshots...")
    n = _upsert_snapshots(client, registros)
    print(f"[snapshot] OK - {n} linhas em snapshot_diario")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--dashboard-dir", default=None)
    args = ap.parse_args()
    run(bootstrap=args.bootstrap, data_override=args.data, dashboard_dir=args.dashboard_dir)
