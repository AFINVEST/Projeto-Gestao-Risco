"""
enviar_email_mensal.py  (v1.1)
================================

Consolidacao mensal com atribuicao por estrategia, rentabilidade historica
e indices historicos (mesmo formato do dashboard).

ESTRUTURA:
  [SECAO 1] Resumo do mes (retorno / CDI combinado, alpha, cota)
              - Chart: cota vs CDI do mes
              - Metricas: vol 20d media, Max DD mes, VaR inicio->fim
  [SECAO 2] Atribuicao de performance por estrategia (waterfall)
              - Caixa, Custos, Juros Nominais, Juros Reais, Performance, CDI
  [SECAO 3] Rentabilidade historica (tabela ano x mes + acumulado)
  [SECAO 4] Indices historicos (janelas: ano, mes, 12m, 24m, 36m, 48m, 60m, total)
  [SECAO 5] Contexto historico (cota, vol 20d, DD)

USO:
    python enviar_email_mensal.py                       # mes anterior
    python enviar_email_mensal.py --ref 2026-07         # mes especifico
    python enviar_email_mensal.py --dry-run
"""
from __future__ import annotations
import os
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
try:
    from supabase import create_client
except ImportError:
    print("ERRO: pip install supabase", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
try:
    from enviar_email_diario import (
        _svg_line_chart, _svg_donut, _nice_step,
        _fmt_reais, _fmt_pct, _fmt_bps,
        _cor_ret, _cor_dd, _cor_var,
    )
except ImportError as e:
    print(f"ERRO helpers de enviar_email_diario: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    from cota_portfolio_core import computar_cota_serie, analisar_dados_fundos2, load_basefundos
    HAS_CORE = True
except ImportError as e:
    HAS_CORE = False
    print(f"[email-mensal] AVISO: cota_portfolio_core nao disponivel — atribuicao por estrategia desabilitada. {e}")


# =============================================================================
# Supabase
# =============================================================================

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
        if isinstance(v, str): v = v.strip('"')
        try:
            if v.startswith("["):
                import json; out[row["parametro"]] = json.loads(v); continue
        except Exception: pass
        try: out[row["parametro"]] = float(v)
        except Exception: out[row["parametro"]] = v
    return out


def _load_snapshot_serie(client, data_ini_iso=None, data_fim_iso=None):
    rows = []; offset = 0; page = 1000
    while True:
        q = client.table("snapshot_diario").select("*").order("Data")
        if data_ini_iso: q = q.gte("Data", data_ini_iso)
        if data_fim_iso: q = q.lte("Data", data_fim_iso)
        resp = q.range(offset, offset + page - 1).execute()
        data = resp.data or []
        if not data: break
        rows.extend(data)
        if len(data) < page: break
        offset += page
    return rows


# =============================================================================
# Helpers
# =============================================================================

def _std(vals):
    n = len(vals)
    if n < 2: return 0.0
    m = sum(vals) / n
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


def _acumulado(retornos):
    acc = 1.0
    for r in retornos:
        acc *= (1 + r)
    return acc - 1


def _mes_referencia(hoje: date | None = None):
    if hoje is None: hoje = date.today()
    inicio_mes_atual = date(hoje.year, hoje.month, 1)
    fim_mes_ant = inicio_mes_atual - timedelta(days=1)
    ini_mes_ant = date(fim_mes_ant.year, fim_mes_ant.month, 1)
    label = fim_mes_ant.strftime("%b/%Y")
    return ini_mes_ant, fim_mes_ant, label


def _mes_from_ref(ref: str):
    y, m = ref.split("-")
    y, m = int(y), int(m)
    ini = date(y, m, 1)
    if m == 12: fim = date(y, 12, 31)
    else: fim = date(y, m + 1, 1) - timedelta(days=1)
    return ini, fim, ini.strftime("%b/%Y")


MESES_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
             "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def _mes_pt(dt: date | datetime):
    return MESES_PT[dt.month - 1] + "/" + str(dt.year)


# =============================================================================
# SVG waterfall (atribuicao)
# =============================================================================

def _svg_waterfall(items, largura=800, altura=340, titulo=""):
    """items: [(label, valor, tipo)] onde tipo in {'start','end','pos','neg','total'}.
    Valores em decimal (0.01 = 1%). Retorna SVG.
    """
    if not items:
        return f'<div style="color:#999">Sem dados</div>'

    n = len(items)
    # Calcula posicoes (base y de cada barra)
    # Waterfall: start=absoluto, pos/neg=relativo ao cumulativo, total=absoluto do zero
    cum = 0
    bases = []
    tops = []
    valores = []
    for i, (lbl, v, tp) in enumerate(items):
        if tp in ("start", "total", "end"):
            base = 0
            top = v
        else:
            base = cum
            top = cum + v
            cum = top
        # Se start ou total, atualiza cum
        if tp == "start":
            cum = v
        elif tp == "total" or tp == "end":
            cum = v
        bases.append(base)
        tops.append(top)
        valores.append(v)

    all_vals = [t for t in tops] + [b for b in bases] + [0]
    ymin_raw = min(all_vals)
    ymax_raw = max(all_vals)
    rng = max(ymax_raw - ymin_raw, 1e-6)
    ymin_raw -= 0.10 * rng
    ymax_raw += 0.15 * rng
    step = _nice_step(ymax_raw - ymin_raw)
    ymin = math.floor(ymin_raw / step) * step
    ymax = math.ceil(ymax_raw / step) * step
    if ymax <= ymin: ymax = ymin + step

    margin_l = 60; margin_r = 15; margin_t = 40; margin_b = 60
    w_plot = largura - margin_l - margin_r
    h_plot = altura - margin_t - margin_b
    yrange = ymax - ymin

    bar_w = (w_plot / n) * 0.75
    gap = (w_plot / n) - bar_w

    def _x(i): return margin_l + gap/2 + i * (bar_w + gap)
    def _y(v): return margin_t + h_plot - ((v - ymin) / yrange) * h_plot

    svg = [f'<svg width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}" '
           f'xmlns="http://www.w3.org/2000/svg" style="background:#fff;font-family:Segoe UI,Arial,sans-serif;">']
    if titulo:
        svg.append(f'<text x="{largura/2:.0f}" y="18" font-size="14" font-weight="bold" '
                   f'fill="#1a3a6c" text-anchor="middle">{titulo}</text>')

    # Grid Y
    yv = ymin
    while yv <= ymax + 1e-9:
        y = _y(yv)
        svg.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{largura - margin_r}" y2="{y:.1f}" '
                   f'stroke="#e5e7eb" stroke-width="0.5"/>')
        svg.append(f'<text x="{margin_l - 4}" y="{y+3:.1f}" font-size="10" fill="#666" '
                   f'text-anchor="end">{yv*100:.2f}%</text>')
        yv += step
    # Linha do zero (destacada)
    y0 = _y(0)
    svg.append(f'<line x1="{margin_l}" y1="{y0:.1f}" x2="{largura - margin_r}" y2="{y0:.1f}" '
               f'stroke="#333" stroke-width="1"/>')

    # Barras
    for i, (lbl, v, tp) in enumerate(items):
        base = bases[i]; top = tops[i]
        y_top = _y(max(base, top))
        y_bot = _y(min(base, top))
        h = max(1.0, y_bot - y_top)
        cor = {"start": "#2e7d32", "end": "#1565c0",
               "pos": "#2e7d32", "neg": "#c62828",
               "total": "#1a237e"}[tp]
        svg.append(f'<rect x="{_x(i):.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                   f'fill="{cor}" stroke="#333" stroke-width="0.5"/>')
        # Label do valor no topo/base
        label_y = y_top - 6 if v >= 0 else y_bot + 12
        sinal = "+" if v >= 0 else ""
        svg.append(f'<text x="{_x(i) + bar_w/2:.1f}" y="{label_y:.1f}" font-size="11" font-weight="bold" '
                   f'fill="#333" text-anchor="middle">{sinal}{v*100:.2f}%</text>')
        # Nome da categoria no eixo x
        svg.append(f'<text x="{_x(i) + bar_w/2:.1f}" y="{altura - 25:.1f}" font-size="11" '
                   f'fill="#333" text-anchor="middle">{lbl}</text>')

    svg.append('</svg>')
    return "".join(svg)


# =============================================================================
# Atribuicao por estrategia (usa cota_portfolio_core)
# =============================================================================

def _atribuicao_mes(data_ini: date, data_fim: date, pct_capital=0.01, base_dir=None,
                     cdi_mes_override: float | None = None):
    """Calcula atribuicao de performance do mes por estrategia:
       Caixa (LFT), Custos/Despesas, Juros Nominais, Juros Reais, Performance, CDI.
       Retorna list of tuples (label, valor_pct, tipo).

       cdi_mes_override: se dado, usa esse valor de CDI em vez de recomputar
       (evita cdi_series vazio quando cdi_cached.csv nao existe).
    """
    if not HAS_CORE:
        return []

    try:
        res = computar_cota_serie(pct=pct_capital, data_ini=data_ini, data_fim=data_fim,
                                    base_dir=base_dir)
        df = res["df"]
        if df.empty:
            return []

        # Componentes acumulados sobre capital_ini_dia
        cap_ini = df["capital_ini_dia"]
        ganho_lft_pct  = (df["ganho_lft"]   / cap_ini).sum()
        custo_pct      = (df["custo_total"] / cap_ini).sum()
        # PnL total (somando todos os ativos)
        pnl_pct_total  = (df["pnl"] / cap_ini).sum()
        ret_total_pct  = df["ret_total"].sum()   # aproximacao
        cdi_pct        = df["cdi_series"].sum()

        # Melhor: pegar retorno mes composto
        ret_mes = float((1 + df["ret_total"]).prod() - 1)
        cdi_mes_calc = float((1 + df["cdi_series"]).prod() - 1)
        # Override se dado (usa CDI do snapshot, que e confiavel)
        cdi_mes = cdi_mes_override if cdi_mes_override is not None else cdi_mes_calc

        # Como decompor PnL por classe: precisa analisar_dados_fundos2 direto
        # Le posicoes e computa PnL agregado por ativo, filtra por mes
        pnl_nom_pct = pnl_pct_total   # default: tudo em nominais
        pnl_real_pct = 0.0

        try:
            bf = load_basefundos()
            df_pnl, _, _, _ = analisar_dados_fundos2(
                soma_pl_sem_pesos=1.0,   # nao usado
                basefundos=bf,
            )
            # Filtra colunas do mes
            df_pnl.columns = pd.to_datetime(df_pnl.columns, errors="coerce")
            cols_mes = [c for c in df_pnl.columns
                        if pd.notna(c) and data_ini <= c.date() <= data_fim]
            if cols_mes:
                pnl_mes = df_pnl[cols_mes].sum(axis=1)   # sum por (Ativo-Fundo)
                # Classe pelo nome do indice: "DI_F28 - FUNDO - P&L"
                pnl_nom = 0.0; pnl_real = 0.0
                for key, val in pnl_mes.items():
                    ativo = str(key).split(" - ")[0].upper()
                    v = float(val or 0)
                    if ativo.startswith("DI_") or ativo.startswith("DI"):
                        pnl_nom += v
                    elif ativo.startswith(("DAP", "NTNB")):
                        pnl_real += v
                # Normaliza sobre cap_ini medio
                cap_medio = float(cap_ini.mean())
                if cap_medio > 0:
                    pnl_nom_pct  = pnl_nom  / cap_medio / len(cap_ini) * len(df.index)
                    # Melhor: soma dos ret diarios (ret_nom_dia = pnl_nom_dia / cap_ini_dia)
                    # Se nao conseguirmos separar por dia, aproxima:
                    # Usar sum(pnl_class) / sum(cap_ini_medio_dia*n)
                    pnl_nom_pct  = pnl_nom  / cap_ini.sum() * len(cap_ini)
                    pnl_real_pct = pnl_real / cap_ini.sum() * len(cap_ini)
                    # Ah, melhor ainda: pnl_pct = sum(pnl_dia/cap_ini_dia)
                    # Mas nao temos pnl por classe por dia sem mais trabalho
                    # Aproximacao: usar proporcao do pnl total
                    if abs(pnl_pct_total) > 1e-9:
                        prop_nom  = pnl_nom  / (pnl_nom + pnl_real) if (pnl_nom + pnl_real) else 0
                        prop_real = 1 - prop_nom
                        pnl_nom_pct  = pnl_pct_total * prop_nom
                        pnl_real_pct = pnl_pct_total * prop_real
        except Exception as e:
            print(f"[email-mensal] atribuicao por classe: fallback (erro: {e})")

        items = [
            ("Caixa (LFT)",         ganho_lft_pct, "start"),
            ("Custos/Despesas",    -abs(custo_pct), "neg"),
        ]
        if abs(pnl_nom_pct) > 1e-6:
            tp = "pos" if pnl_nom_pct >= 0 else "neg"
            items.append(("Juros Nominais BR", pnl_nom_pct, tp))
        if abs(pnl_real_pct) > 1e-6:
            tp = "pos" if pnl_real_pct >= 0 else "neg"
            items.append(("Juros Reais BR", pnl_real_pct, tp))
        items.append(("Performance", ret_mes, "total"))
        items.append(("CDI", cdi_mes, "end"))
        return items
    except Exception as e:
        print(f"[email-mensal] erro na atribuicao: {e}")
        return []


# =============================================================================
# Rentabilidade historica (tabela ano x mes)
# =============================================================================

def _rentabilidade_historica(snapshots_all):
    """Agrupa snapshots por (ano, mes) e computa retorno composto do mes + CDI.
    Retorna dict {ano: {mes: {ret, cdi, pct_cdi}}, ano_totais: {ret, cdi, pct_cdi}}
    """
    from collections import defaultdict
    grupos = defaultdict(lambda: defaultdict(lambda: {"rets": [], "cdis": []}))
    for s in snapshots_all:
        dt = datetime.fromisoformat(s["Data"])
        r = float(s.get("retorno_dtd") or 0)
        c = float(s.get("cdi_dtd") or 0)
        grupos[dt.year][dt.month]["rets"].append(r)
        grupos[dt.year][dt.month]["cdis"].append(c)

    saida = {}
    for ano in sorted(grupos.keys()):
        meses = {}
        ret_ano_rets = []
        ret_ano_cdis = []
        for mes in range(1, 13):
            if mes not in grupos[ano]:
                meses[mes] = None
                continue
            rs = grupos[ano][mes]["rets"]
            cs = grupos[ano][mes]["cdis"]
            ret = _acumulado(rs)
            cdi = _acumulado(cs)
            pct = (ret / cdi) if cdi else None
            meses[mes] = {"ret": ret, "cdi": cdi, "pct_cdi": pct}
            ret_ano_rets.extend(rs)
            ret_ano_cdis.extend(cs)
        ret_ano = _acumulado(ret_ano_rets)
        cdi_ano = _acumulado(ret_ano_cdis)
        pct_ano = (ret_ano / cdi_ano) if cdi_ano else None
        saida[ano] = {"meses": meses, "no_ano": {"ret": ret_ano, "cdi": cdi_ano, "pct_cdi": pct_ano}}
    # Acumulado total
    all_rets = [float(s.get("retorno_dtd") or 0) for s in snapshots_all]
    all_cdis = [float(s.get("cdi_dtd") or 0) for s in snapshots_all]
    ret_tot = _acumulado(all_rets); cdi_tot = _acumulado(all_cdis)
    saida["acumulado"] = {"ret": ret_tot, "cdi": cdi_tot, "pct_cdi": (ret_tot / cdi_tot if cdi_tot else None)}
    return saida


# =============================================================================
# Indices historicos por janela
# =============================================================================

def _indices_historicos(snapshots_all, ref_date: date):
    """Computa Rentabilidade, %CDI, Vol, Sharpe para janelas ANO/MES/12/24/36/48/60/TOTAL.
    Sempre usando snapshots ate ref_date.
    """
    # Filtra ate ref_date
    ref_iso = ref_date.isoformat()
    snaps = [s for s in snapshots_all if s["Data"] <= ref_iso]
    if not snaps:
        return {}

    def _slice_by_dias(n_dias):
        return snaps[-n_dias:] if len(snaps) >= n_dias else snaps

    def _slice_ano():
        return [s for s in snaps if datetime.fromisoformat(s["Data"]).year == ref_date.year]

    def _slice_mes():
        return [s for s in snaps if
                datetime.fromisoformat(s["Data"]).year == ref_date.year and
                datetime.fromisoformat(s["Data"]).month == ref_date.month]

    janelas = {
        "ANO":       _slice_ano(),
        "NO MÊS":    _slice_mes(),
        "12 MESES":  _slice_by_dias(252),
        "24 MESES":  _slice_by_dias(504),
        "36 MESES":  _slice_by_dias(756),
        "48 MESES":  _slice_by_dias(1008),
        "60 MESES":  _slice_by_dias(1260),
        "TOTAL":     snaps,
    }

    def _metric(subs):
        if not subs: return None
        rets = [float(s.get("retorno_dtd") or 0) for s in subs]
        cdis = [float(s.get("cdi_dtd") or 0) for s in subs]
        ret = _acumulado(rets); cdi = _acumulado(cdis)
        pct = (ret / cdi) if cdi else None
        vol = _std(rets) * math.sqrt(252)
        # Sharpe: mean(r-c)/std(r-c) * sqrt(252)
        excess = [r - c for r, c in zip(rets, cdis)]
        mu = sum(excess) / len(excess) if excess else 0
        sig = _std(excess)
        sharpe = (mu / sig * math.sqrt(252)) if sig > 0 else None
        return {"ret": ret, "cdi": cdi, "pct_cdi": pct, "vol": vol, "sharpe": sharpe}

    return {k: _metric(v) for k, v in janelas.items()}


# =============================================================================
# HTML
# =============================================================================

def montar_html(snapshots_mes, snapshots_all, atrib_items, rent_hist, ind_hist,
                 label_mes, data_fim: date, pct_capital=0.01):
    if not snapshots_mes:
        return f"Dash Risco - {label_mes} - sem dados", "<p>Sem snapshots no periodo.</p>", {}

    n = len(snapshots_mes)
    primeiro = snapshots_mes[0]; ultimo = snapshots_mes[-1]
    cota_ini = float(primeiro.get("cota") or 0)
    cota_fim = float(ultimo.get("cota") or 0)
    retornos_dia = [float(s.get("retorno_dtd") or 0) for s in snapshots_mes]
    cdis_dia = [float(s.get("cdi_dtd") or 0) for s in snapshots_mes]

    ret_mes = _acumulado(retornos_dia)
    cdi_mes = _acumulado(cdis_dia)
    # Alpha composto: (1+ret)/(1+cdi) - 1  (fatorial, nao linear)
    alpha_mes = ((1 + ret_mes) / (1 + cdi_mes) - 1) if cdi_mes != -1 else 0
    pct_cdi_mes = (ret_mes / cdi_mes) if cdi_mes else None

    # Ret / CDI do ANO ate ref
    year_ref = data_fim.year
    snaps_ano = [s for s in snapshots_all
                  if datetime.fromisoformat(s["Data"]).year == year_ref
                  and s["Data"] <= data_fim.isoformat()]
    ret_ano_rets = [float(s.get("retorno_dtd") or 0) for s in snaps_ano]
    ret_ano_cdis = [float(s.get("cdi_dtd") or 0) for s in snaps_ano]
    ret_ano = _acumulado(ret_ano_rets)
    cdi_ano = _acumulado(ret_ano_cdis)
    alpha_ano = ((1 + ret_ano) / (1 + cdi_ano) - 1) if cdi_ano != -1 else 0
    pct_cdi_ano = (ret_ano / cdi_ano) if cdi_ano else None

    # Ret / CDI TOTAL (historico completo ate ref)
    snaps_tot = [s for s in snapshots_all if s["Data"] <= data_fim.isoformat()]
    ret_tot_rets = [float(s.get("retorno_dtd") or 0) for s in snaps_tot]
    ret_tot_cdis = [float(s.get("cdi_dtd") or 0) for s in snaps_tot]
    ret_tot = _acumulado(ret_tot_rets)
    cdi_tot = _acumulado(ret_tot_cdis)
    alpha_tot = ((1 + ret_tot) / (1 + cdi_tot) - 1) if cdi_tot != -1 else 0
    pct_cdi_tot = (ret_tot / cdi_tot) if cdi_tot else None

    # Vol 20d anualizada (media do mes)
    vols_20d = [float(s.get("vol_20d") or 0) for s in snapshots_mes if s.get("vol_20d")]
    vol_20d_media = sum(vols_20d) / len(vols_20d) if vols_20d else 0

    # VaR medio do mes + % do total
    vars_hist_mes = [float(s.get("var_carteira_hist_reais") or 0) for s in snapshots_mes if s.get("var_carteira_hist_reais")]
    vars_ewma_mes = [float(s.get("var_carteira_ewma_reais") or 0) for s in snapshots_mes if s.get("var_carteira_ewma_reais")]
    consumo_hist_mes = [float(s.get("consumo_hist_pct") or 0) for s in snapshots_mes if s.get("consumo_hist_pct")]
    consumo_ewma_mes = [float(s.get("consumo_ewma_pct") or 0) for s in snapshots_mes if s.get("consumo_ewma_pct")]
    var_hist_med  = sum(vars_hist_mes)  / len(vars_hist_mes)  if vars_hist_mes  else 0
    var_ewma_med  = sum(vars_ewma_mes)  / len(vars_ewma_mes)  if vars_ewma_mes  else 0
    consumo_hist_med = sum(consumo_hist_mes) / len(consumo_hist_mes) if consumo_hist_mes else 0
    consumo_ewma_med = sum(consumo_ewma_mes) / len(consumo_ewma_mes) if consumo_ewma_mes else 0

    # Max DD do mes
    cotas_mes = [float(s.get("cota") or 0) for s in snapshots_mes if s.get("cota") and 0.5 < float(s.get("cota")) < 10]
    if len(cotas_mes) >= 2:
        peak = cotas_mes[0]; dd_max_mes = 0.0
        for c in cotas_mes:
            if c > peak: peak = c
            dd = c / peak - 1
            if dd < dd_max_mes: dd_max_mes = dd
    else:
        dd_max_mes = 0.0

    # VaR inicio e fim do mes + oscilacao
    var_hist_ini = float(primeiro.get("var_carteira_hist_reais") or 0)
    var_hist_fim = float(ultimo.get("var_carteira_hist_reais") or 0)
    var_ewma_ini = float(primeiro.get("var_carteira_ewma_reais") or 0)
    var_ewma_fim = float(ultimo.get("var_carteira_ewma_reais") or 0)
    dif_hist = var_hist_fim - var_hist_ini
    dif_ewma = var_ewma_fim - var_ewma_ini

    # PL medio
    pls = [float(s.get("pl_total") or 0) for s in snapshots_mes if s.get("pl_total")]
    pl_medio = sum(pls) / len(pls) if pls else 0

    # ─── Charts ────────────────────────────
    datas_mes = [s["Data"] for s in snapshots_mes]
    # Chart em RETORNO acumulado (%): mais intuitivo que index base=1
    base_cota_mes = cota_ini if cota_ini > 0 else 1.0
    cota_mes_ret = [((float(s.get("cota") or 0) / base_cota_mes) - 1)
                     if (s.get("cota") and 0.5 < float(s.get("cota")) < 10) else None
                     for s in snapshots_mes]
    cdi_mes_ret = []; acc = 1.0
    for r in cdis_dia:
        acc *= (1 + r); cdi_mes_ret.append(acc - 1)
    chart_cota_mes = _svg_line_chart(datas_mes,
        {"Cota": cota_mes_ret, "CDI": cdi_mes_ret},
        largura=800, altura=220, titulo=f"Cota vs CDI - {label_mes} (retorno acumulado)",
        cores=["#1565c0", "#7f7f7f"], y_fmt="pct")

    # Waterfall atribuicao
    chart_waterfall = _svg_waterfall(atrib_items, largura=800, altura=340,
                                       titulo=f"Atribuicao de Performance {label_mes}")

    # Historico
    datas_h = [s["Data"] for s in snapshots_all]
    cotas_h_raw = []
    for s in snapshots_all:
        c = s.get("cota")
        if c is None: cotas_h_raw.append(None); continue
        cf = float(c)
        cotas_h_raw.append(cf if 0.5 < cf < 10 else None)
    base_h = next((c for c in cotas_h_raw if c is not None), 1.0)
    cota_h_ret = [((c / base_h) - 1) if c is not None else None for c in cotas_h_raw]
    cdi_h_diario = [float(s.get("cdi_dtd") or 0) for s in snapshots_all]
    cdi_h_ret = []; acc = 1.0
    for r in cdi_h_diario:
        acc *= (1 + r); cdi_h_ret.append(acc - 1)
    chart_cota_hist = _svg_line_chart(datas_h,
        {"Cota": cota_h_ret, "CDI": cdi_h_ret},
        largura=800, altura=200, titulo="Cota vs CDI - Historico total (retorno acumulado)",
        cores=["#1565c0", "#7f7f7f"], y_fmt="pct")

    vol_20d_h = [s.get("vol_20d") for s in snapshots_all]
    chart_vol_hist = _svg_line_chart(datas_h, {"Vol 20d anualizada": vol_20d_h},
        largura=800, altura=180, titulo="Volatilidade 20d anualizada",
        cores=["#c62828"], y_fmt="pct")

    dd_h = [s.get("dd_atual") for s in snapshots_all]
    chart_dd = _svg_line_chart(datas_h, {"Drawdown": dd_h},
        largura=800, altura=180, titulo="Drawdown historico",
        cores=["#dc3545"], y_fmt="pct", y_zero_center=True)

    # ─── HTML ────────────────────────────
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: Segoe UI, Arial, sans-serif; color: #222; font-size: 14px; max-width: 900px; margin: 0 auto; padding: 10px; }}
h1 {{ color: #1a3a6c; border-bottom: 3px solid #1a3a6c; padding-bottom: 8px; }}
h2 {{ color: #1a3a6c; margin-top: 32px; font-size: 18px; border-bottom: 2px solid #1a3a6c; padding-bottom: 4px; }}
h3 {{ color: #1a3a6c; margin-top: 20px; font-size: 15px; }}
.metric-grid {{ display: table; width: 100%; border-collapse: collapse; margin: 12px 0; }}
.metric {{ display: table-cell; padding: 12px 14px; border: 1px solid #ddd; background: #f8f9fa; vertical-align: top; }}
.metric-label {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
.metric-value {{ font-size: 22px; font-weight: bold; line-height: 1.2; }}
.metric-value-secondary {{ font-size: 14px; color: #444; font-weight: 600; margin-top: 4px; }}
.chart-container {{ margin: 16px 0; padding: 10px; background: #fff; border: 1px solid #ddd; border-radius: 4px; text-align: center; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12px; }}
th, td {{ padding: 6px 8px; border: 1px solid #ddd; text-align: center; }}
th {{ background: #1a3a6c; color: white; font-size: 11px; }}
tr:nth-child(even) td {{ background: #f8f9fa; }}
.tbl-rent td.rowlabel {{ text-align: left; font-weight: bold; background: #eef; }}
.pct-cdi {{ display: block; font-size: 10px; color: #888; font-style: italic; margin-top: 2px; }}
</style></head><body>

<h1>Dash Risco - Consolidado {label_mes}</h1>

<!-- ============ SECAO 1: RESUMO DO MÊS ============ -->
<h2>1. Resumo do mês</h2>

<div class="metric-grid">
  <div class="metric" style="width:33%;">
    <div class="metric-label">Mês</div>
    <div class="metric-value">
      <span style="color:{_cor_ret(ret_mes)}">{ret_mes*100:+.2f}%</span>
      <span style="color:#333"> / {cdi_mes*100:.2f}%</span>
    </div>
    <div style="font-size:14px;color:#444;font-weight:600;margin-top:4px;">
      {'-' if pct_cdi_mes is None else f'{pct_cdi_mes*100:.0f}% do CDI'}
      <span style="color:#555;font-weight:normal;"> &nbsp; CDI {'+' if alpha_mes >= 0 else '-'} {abs(alpha_mes*100):.2f}%</span>
    </div>
  </div>
  <div class="metric" style="width:33%;">
    <div class="metric-label">Ano ({year_ref})</div>
    <div class="metric-value">
      <span style="color:{_cor_ret(ret_ano)}">{ret_ano*100:+.2f}%</span>
      <span style="color:#333"> / {cdi_ano*100:.2f}%</span>
    </div>
    <div style="font-size:14px;color:#444;font-weight:600;margin-top:4px;">
      {'-' if pct_cdi_ano is None else f'{pct_cdi_ano*100:.0f}% do CDI'}
      <span style="color:#555;font-weight:normal;"> &nbsp; CDI {'+' if alpha_ano >= 0 else '-'} {abs(alpha_ano*100):.2f}%</span>
    </div>
  </div>
  <div class="metric" style="width:34%;">
    <div class="metric-label">Total (histórico)</div>
    <div class="metric-value">
      <span style="color:{_cor_ret(ret_tot)}">{ret_tot*100:+.2f}%</span>
      <span style="color:#333"> / {cdi_tot*100:.2f}%</span>
    </div>
    <div style="font-size:14px;color:#444;font-weight:600;margin-top:4px;">
      {'-' if pct_cdi_tot is None else f'{pct_cdi_tot*100:.0f}% do CDI'}
      <span style="color:#555;font-weight:normal;"> &nbsp; CDI {'+' if alpha_tot >= 0 else '-'} {abs(alpha_tot*100):.2f}%</span>
    </div>
  </div>
</div>

<div class="chart-container">
  {chart_cota_mes}
</div>

<h3>Métricas de risco do mês</h3>
<div class="metric-grid">
  <div class="metric" style="width:25%;">
    <div class="metric-label">Vol 20d anualizada</div>
    <div class="metric-value">{vol_20d_media*100:.2f}%</div>
  </div>
  <div class="metric" style="width:25%;">
    <div class="metric-label">Max Drawdown</div>
    <div class="metric-value" style="color:#dc3545">{dd_max_mes*100:.2f}%</div>
  </div>
  <div class="metric" style="width:25%;">
    <div class="metric-label">VaR HIST médio</div>
    <div class="metric-value">{_fmt_reais(var_hist_med)}</div>
    <div class="metric-value-secondary">({consumo_hist_med*100:.1f}% do limite)</div>
  </div>
  <div class="metric" style="width:25%;">
    <div class="metric-label">VaR EWMA médio</div>
    <div class="metric-value">{_fmt_reais(var_ewma_med)}</div>
    <div class="metric-value-secondary">({consumo_ewma_med*100:.1f}% do limite)</div>
  </div>
</div>


<!-- ============ SECAO 2: ATRIBUIÇÃO DE PERFORMANCE ============ -->
<h2>2. Atribuição de performance por estratégia</h2>
<div class="chart-container">
  {chart_waterfall}
</div>
"""

    # ─── Rentabilidade historica (tabela) ────────────
    html += """
<!-- ============ SECAO 3: RENTABILIDADE HISTÓRICA ============ -->
<h2>3. Rentabilidade histórica</h2>
<table class="tbl-rent">
<tr><th>ANO</th>"""
    for m in MESES_PT:
        html += f"<th>{m}</th>"
    html += "<th>No ano</th><th>Acumulado</th></tr>"

    # Linhas por ano
    anos = sorted([a for a in rent_hist.keys() if isinstance(a, int)])
    acumulado_running_ret = 1.0
    acumulado_running_cdi = 1.0
    for ano in anos:
        html += f'<tr><td class="rowlabel">{ano}<br><span style="font-size:9px;font-weight:normal;color:#888">%CDI</span></td>'
        info = rent_hist[ano]
        for mes in range(1, 13):
            dados = info["meses"].get(mes)
            if dados is None:
                html += "<td>—</td>"
            else:
                ret = dados["ret"]; pct = dados["pct_cdi"]
                cor = _cor_ret(ret)
                pct_str = f"({pct*100:.0f}%)" if pct is not None else ""
                html += (f'<td style="color:{cor};font-weight:bold">{ret*100:+.2f}%'
                         f'<span class="pct-cdi">{pct_str}</span></td>')
                acumulado_running_ret *= (1 + ret)
                acumulado_running_cdi *= (1 + dados["cdi"])
        # No ano
        no_ano = info["no_ano"]
        cor_no_ano = _cor_ret(no_ano["ret"])
        pct_no_ano_str = f'({no_ano["pct_cdi"]*100:.0f}%)' if no_ano["pct_cdi"] is not None else ""
        html += (f'<td style="color:{cor_no_ano};font-weight:bold">{no_ano["ret"]*100:+.2f}%'
                 f'<span class="pct-cdi">{pct_no_ano_str}</span></td>')
        # Acumulado ate esse ano
        acc_ret = acumulado_running_ret - 1
        acc_cdi = acumulado_running_cdi - 1
        acc_pct = (acc_ret / acc_cdi) if acc_cdi else None
        cor_acc = _cor_ret(acc_ret)
        acc_pct_str = f'({acc_pct*100:.0f}%)' if acc_pct is not None else ""
        html += (f'<td style="color:{cor_acc};font-weight:bold">{acc_ret*100:+.2f}%'
                 f'<span class="pct-cdi">{acc_pct_str}</span></td>')
        html += "</tr>"
    html += "</table>"

    # ─── Indices Historicos (janelas) ────────────
    html += """
<!-- ============ SECAO 4: ÍNDICES HISTÓRICOS ============ -->
<h2>4. Índices históricos</h2>
<table>
<tr><th></th><th>ANO</th><th>NO MÊS</th><th>12 MESES</th><th>24 MESES</th><th>36 MESES</th><th>48 MESES</th><th>60 MESES</th><th>TOTAL</th></tr>
"""
    janelas = ["ANO", "NO MÊS", "12 MESES", "24 MESES", "36 MESES", "48 MESES", "60 MESES", "TOTAL"]

    def _c(x, cor=None):
        if x is None: return "—"
        return f'<span style="color:{cor}" >{x}</span>' if cor else x

    # Rentabilidade
    html += '<tr><td class="rowlabel">RENTABILIDADE<br><span style="font-size:9px;font-weight:normal;color:#888">%CDI</span></td>'
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("ret") is None:
            html += "<td>—</td>"
        else:
            cor = _cor_ret(d["ret"])
            pct = d["pct_cdi"]
            pct_str = f"({pct*100:.0f}%)" if pct is not None else ""
            html += (f'<td style="color:{cor};font-weight:bold">{d["ret"]*100:+.2f}%'
                     f'<span class="pct-cdi">{pct_str}</span></td>')
    html += "</tr>"

    # Volatilidade
    html += '<tr><td class="rowlabel">VOLATILIDADE</td>'
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("vol") is None:
            html += "<td>—</td>"
        else:
            html += f'<td>{d["vol"]*100:.2f}%</td>'
    html += "</tr>"

    # Sharpe
    html += '<tr><td class="rowlabel">ÍNDICE DE SHARPE</td>'
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("sharpe") is None:
            html += "<td>—</td>"
        else:
            cor = _cor_ret(d["sharpe"])
            html += f'<td style="color:{cor};font-weight:bold">{d["sharpe"]:.2f}</td>'
    html += "</tr></table>"

    # ─── Contexto historico (charts) ────────────
    html += f"""

<!-- ============ SECAO 5: CONTEXTO HISTÓRICO ============ -->
<h2>5. Contexto histórico</h2>

<div class="chart-container">
  {chart_cota_hist}
</div>

<div class="chart-container">
  {chart_vol_hist}
</div>

<div class="chart-container">
  {chart_dd}
</div>

<p style="margin-top:32px;color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:12px;">
Consolidado gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}.
PL médio do mês: {_fmt_reais(pl_medio)}.
</p>
</body></html>"""

    assunto = f"Dash Risco - Consolidado {label_mes} - Retorno {ret_mes*100:+.2f}% ({(pct_cdi_mes*100 if pct_cdi_mes else 0):.0f}% CDI)"
    dados = {
        "label": label_mes, "ret_mes": ret_mes, "cdi_mes": cdi_mes,
        "alpha_mes": alpha_mes, "vol_20d_media": vol_20d_media, "dd_max": dd_max_mes,
        "cota_ini": cota_ini, "cota_fim": cota_fim, "n_dias": n,
    }
    return assunto, html, dados


# =============================================================================
# Envio
# =============================================================================

def enviar_via_outlook(destinatarios, assunto, html_body):
    try:
        import win32com.client
    except ImportError:
        raise RuntimeError("Instale pywin32: pip install pywin32")
    outlook = win32com.client.Dispatch("Outlook.Application")
    mail = outlook.CreateItem(0)
    mail.To = "; ".join(destinatarios)
    mail.Subject = assunto
    mail.HTMLBody = html_body
    mail.Send()
    return True


def run(dry_run=False, override_to=None, ref: str | None = None, dashboard_dir=None):
    client = _get_client()
    config = _get_config(client)
    pct_capital = float(config.get("pct_capital_risco", 0.01))
    emails = config.get("emails_mensal") or config.get("emails_diario") or ["marcos.freitas@afinvest.com.br"]
    if isinstance(emails, str): emails = [emails]
    destinatarios = [override_to] if override_to else emails

    if ref:
        ini, fim, label = _mes_from_ref(ref)
    else:
        ini, fim, label = _mes_referencia()

    print(f"[email-mensal] periodo: {ini} -> {fim}  ({label})")

    snapshots_mes = _load_snapshot_serie(client, ini.isoformat(), fim.isoformat())
    if not snapshots_mes:
        print(f"[email-mensal] sem snapshots no periodo {label}.")
        return

    snapshots_all = _load_snapshot_serie(client)

    # Rentabilidade histórica (tabela ano x mês)
    rent_hist = _rentabilidade_historica(snapshots_all)

    # Índices históricos por janela
    ind_hist = _indices_historicos(snapshots_all, ref_date=fim)

    # Calcula cdi_mes correto do snapshot (fonte confiavel)
    cdi_mes_correto = _acumulado([float(s.get("cdi_dtd") or 0) for s in snapshots_mes])

    # Atribuição por estratégia (usa cota_portfolio_core, com cdi_mes override)
    atrib_items = _atribuicao_mes(ini, fim, pct_capital=pct_capital, base_dir=dashboard_dir,
                                    cdi_mes_override=cdi_mes_correto)

    assunto, html, dados = montar_html(snapshots_mes, snapshots_all, atrib_items,
                                        rent_hist, ind_hist, label, fim, pct_capital=pct_capital)
    print(f"[email-mensal] assunto: {assunto}")
    print(f"[email-mensal] {dados['n_dias']} dias | ret {dados['ret_mes']*100:+.2f}% vs CDI {dados['cdi_mes']*100:.2f}%")
    print(f"[email-mensal] alpha {dados['alpha_mes']*10_000:+.0f} bps | vol_20d_med {dados['vol_20d_media']*100:.2f}% | DD {dados['dd_max']*100:.2f}%")

    if dry_run:
        out = Path(f"email_mensal_{label.replace('/', '_')}_preview.html")
        out.write_text(html, encoding="utf-8")
        print(f"[email-mensal] dry-run - HTML em {out.resolve()}")
        return
    try:
        enviar_via_outlook(destinatarios, assunto, html)
        print(f"[email-mensal] OK enviado")
    except Exception as e:
        print(f"[email-mensal] ERRO ao enviar: {e}")
        out = Path("email_mensal_ERRO.html")
        out.write_text(html, encoding="utf-8")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--to", default=None)
    ap.add_argument("--ref", default=None, help="Mês ref YYYY-MM (default = mês anterior)")
    ap.add_argument("--dashboard-dir", default=None)
    args = ap.parse_args()
    run(dry_run=args.dry_run, override_to=args.to, ref=args.ref, dashboard_dir=args.dashboard_dir)
