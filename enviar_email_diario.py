"""
enviar_email_diario.py  (v1.6 - com graficos SVG e layout redesenhado)
========================================================================

ESTRUTURA:
  [SECAO 1] Resumo do dia
              - Cota, retornos (dia/mes/ano) com %CDI ao lado
              - Chart: cota vs CDI no mes
              - PL Total, PL Risco
              - DV01 atual, VaR carteira (com donut de consumo)
              - Vol 20d, DD atual

  [SECAO 2] Metricas historicas
              - Chart: cota vs CDI historico
              - Chart: vol 20d anualizada historica
              - Chart: DD historico
              - Chart: utilizacao de risco (VaR carteira %) historico
              - Chart: DV01 empilhado por estrategia (limitado ao ultimo dia enquanto
                       nao houver backfill de DV01)
              - Sharpe e VaR historico da cota (equal-weight)
              - Tabela ultimos 5 dias

MUDANCAS v1.6:
  - Metricas em bps/CDI aparecem AO LADO do valor principal (nao mais em letra pequena)
  - Graficos SVG inline (compativeis com Outlook)
  - Nova organizacao em duas secoes claras
"""
from __future__ import annotations
import os
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime, date
try:
    from supabase import create_client
except ImportError:
    print("ERRO: pip install supabase", file=sys.stderr)
    sys.exit(1)


DEFAULT_ASSUNTO = "Dash Risco - {data_br} - Cota {cota:.4f} ({sinal}{ret_bps:.2f}bps) - VaR carteira EWMA {consumo_ewma:.0f}%"


# =============================================================================
# Supabase / config
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
        if isinstance(v, str):
            v = v.strip('"')
        try:
            if v.startswith("["):
                import json
                out[row["parametro"]] = json.loads(v); continue
        except Exception:
            pass
        try:
            out[row["parametro"]] = float(v)
        except Exception:
            out[row["parametro"]] = v
    return out


def _load_snapshot_serie_completa(client):
    """Carrega TODO o snapshot ordenado por data. Usado nos graficos historicos."""
    rows = []
    offset = 0
    page = 1000
    while True:
        resp = (client.table("snapshot_diario")
                     .select("*")
                     .order("Data")
                     .range(offset, offset + page - 1)
                     .execute())
        data = resp.data or []
        if not data:
            break
        rows.extend(data)
        if len(data) < page:
            break
        offset += page
    return rows


def _load_governance(client):
    resp = (client.table("governance_state").select("*").eq("regra", "stop_loss_dd").execute())
    return resp.data[0] if resp.data else None


def _load_eventos_recentes(client, n=5):
    resp = (client.table("eventos_risco").select("*").order("timestamp", desc=True).limit(n).execute())
    return resp.data or []


# =============================================================================
# Formatacao
# =============================================================================

def _fmt_reais(v, casas=0):
    if v is None: return "-"
    try:
        return f"R$ {float(v):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def _fmt_pct(v_frac, casas=2, sinal=False):
    if v_frac is None: return "-"
    try:
        v = float(v_frac) * 100
        s = f"+{v:.{casas}f}%" if sinal and v >= 0 else f"{v:.{casas}f}%"
        return s
    except Exception:
        return "-"


def _fmt_bps(v_frac, casas=2, sinal=False):
    if v_frac is None: return "-"
    try:
        v = float(v_frac) * 10_000
        s = f"+{v:.{casas}f}" if sinal and v >= 0 else f"{v:.{casas}f}"
        return f"{s} bps"
    except Exception:
        return "-"


def _cor_var(pct):
    if pct is None: return "#888"
    if pct < 0.5: return "#28a745"
    if pct < 0.8: return "#ffc107"
    if pct < 1.0: return "#fd7e14"
    return "#dc3545"


def _cor_dd(dd):
    if dd is None or dd >= 0: return "#28a745"
    if dd > -0.02: return "#28a745"
    if dd > -0.04: return "#ffc107"
    return "#dc3545"


def _cor_ret(v):
    if v is None: return "#888"
    return "#28a745" if v >= 0 else "#dc3545"


# =============================================================================
# SVG helpers
# =============================================================================

def _nice_step(rng: float) -> float:
    """Retorna step 'bonito' (1, 2, 2.5, 5, 10 x 10^k) para dividir range em ~5 partes."""
    if rng <= 0: return 1.0
    raw = rng / 5
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized < 1.5:
        step = 1 * magnitude
    elif normalized < 2.25:
        step = 2 * magnitude
    elif normalized < 3.75:
        step = 2.5 * magnitude
    elif normalized < 7.5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    return step


def _svg_line_chart(datas, series_dict, largura=560, altura=180, titulo="",
                     cores=None, y_fmt="pct", y_zero_center=False,
                     margin_left=55, margin_right=15, margin_top=25, margin_bottom=25):
    """Gera SVG line chart simples.

    datas: list of date strings (YYYY-MM-DD)
    series_dict: {"Nome1": [v1, v2, ...], "Nome2": [...]}
    y_fmt: 'pct' | 'bps' | 'reais' | 'num'
    y_zero_center: se True, centra em 0 (util pra DD).
    """
    if not datas or not series_dict:
        return f'<div style="color:#999;padding:10px;text-align:center;">Sem dados para {titulo}</div>'

    if cores is None:
        cores = ["#1565c0", "#7f7f7f", "#28a745", "#dc3545", "#fd7e14"]

    # Coleta todos os valores para escala Y
    all_vals = []
    for nome, vals in series_dict.items():
        all_vals.extend([v for v in vals if v is not None])
    if not all_vals:
        return f'<div style="color:#999;padding:10px;text-align:center;">Sem valores para {titulo}</div>'

    ymin_raw = min(all_vals)
    ymax_raw = max(all_vals)
    if y_zero_center:
        yabs = max(abs(ymin_raw), abs(ymax_raw))
        ymin_raw, ymax_raw = -yabs, 0.001

    # padding vertical 5%
    rng = max(ymax_raw - ymin_raw, 1e-9)
    ymin_raw -= 0.05 * rng
    ymax_raw += 0.05 * rng
    rng = ymax_raw - ymin_raw

    # Ajusta para step "bonito" e limites em multiplos do step
    step = _nice_step(rng)
    ymin = math.floor(ymin_raw / step) * step
    ymax = math.ceil(ymax_raw / step) * step
    if ymax <= ymin:
        ymax = ymin + step
    yrange = ymax - ymin

    w_plot = largura - margin_left - margin_right
    h_plot = altura - margin_top - margin_bottom

    n = len(datas)

    def _x(i):
        return margin_left + (i / max(n - 1, 1)) * w_plot

    def _y(v):
        return margin_top + h_plot - ((v - ymin) / yrange) * h_plot

    def _fmt_y(v):
        if y_fmt == "pct":
            return f"{v*100:.1f}%"
        if y_fmt == "bps":
            return f"{v*10_000:.0f} bps"
        if y_fmt == "reais":
            return f"R$ {v/1000:.0f}k"
        return f"{v:.2f}"

    # SVG start
    svg = [f'<svg width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}" '
           f'xmlns="http://www.w3.org/2000/svg" style="background:#fff;font-family:Segoe UI,Arial,sans-serif;">']

    # Titulo
    if titulo:
        svg.append(f'<text x="{largura/2:.0f}" y="14" font-size="13" font-weight="bold" '
                   f'fill="#1a3a6c" text-anchor="middle">{titulo}</text>')

    # Grid Y em multiplos do step "bonito"
    yv = ymin
    while yv <= ymax + 1e-12:
        y = _y(yv)
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{largura - margin_right}" y2="{y:.1f}" '
                   f'stroke="#e5e7eb" stroke-width="0.5"/>')
        svg.append(f'<text x="{margin_left - 4}" y="{y+3:.1f}" font-size="9" fill="#666" '
                   f'text-anchor="end">{_fmt_y(yv)}</text>')
        yv += step

    # Eixo X — labels em 5 pontos
    for i in range(5):
        idx = int(i / 4 * (n - 1))
        x = _x(idx)
        d = datas[idx]
        try:
            label = datetime.fromisoformat(d).strftime("%b/%y")
        except Exception:
            label = str(d)[:7]
        svg.append(f'<text x="{x:.1f}" y="{altura - 8}" font-size="9" fill="#666" '
                   f'text-anchor="middle">{label}</text>')

    # Series
    for idx_s, (nome, vals) in enumerate(series_dict.items()):
        cor = cores[idx_s % len(cores)]
        pts = []
        for i, v in enumerate(vals):
            if v is None:
                continue
            pts.append((_x(i), _y(v)))
        if len(pts) < 2:
            continue
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        svg.append(f'<path d="{path}" fill="none" stroke="{cor}" stroke-width="1.5"/>')

    # Legenda
    lx = margin_left + 10
    ly = margin_top - 5
    for idx_s, nome in enumerate(series_dict.keys()):
        cor = cores[idx_s % len(cores)]
        svg.append(f'<rect x="{lx}" y="{ly-8}" width="10" height="3" fill="{cor}"/>')
        svg.append(f'<text x="{lx + 14}" y="{ly-2}" font-size="10" fill="#333">{nome}</text>')
        lx += 90

    svg.append('</svg>')
    return "".join(svg)


def _svg_area_stacked(datas, series_dict, largura=560, altura=180, titulo="",
                       cores=None, y_fmt="reais"):
    """SVG de área empilhada."""
    if not datas or not series_dict:
        return f'<div style="color:#999;padding:10px;text-align:center;">Sem dados</div>'
    if cores is None:
        cores = ["#1565c0", "#c62828", "#2e7d32", "#f57c00", "#6a1b9a"]

    n = len(datas)
    # Todas as séries têm mesmo comprimento?
    nomes = list(series_dict.keys())
    # Calcula série cumulativa por linha
    zeros = [0.0] * n
    stacks = [zeros[:]]  # base
    for nome in nomes:
        vals = series_dict[nome]
        vals = [(v if v is not None else 0.0) for v in vals]
        cur = stacks[-1]
        stacks.append([cur[i] + vals[i] for i in range(n)])

    ymax = max(max(s) for s in stacks) if stacks else 0
    if ymax == 0:
        return f'<div style="color:#999;padding:10px;text-align:center;">{titulo}: todos zeros</div>'
    ymin = 0

    margin_left = 55
    margin_right = 15
    margin_top = 25
    margin_bottom = 25
    w_plot = largura - margin_left - margin_right
    h_plot = altura - margin_top - margin_bottom
    yrange = ymax - ymin

    def _x(i): return margin_left + (i / max(n - 1, 1)) * w_plot
    def _y(v): return margin_top + h_plot - ((v - ymin) / yrange) * h_plot

    def _fmt_y(v):
        if y_fmt == "pct": return f"{v*100:.1f}%"
        if y_fmt == "bps": return f"{v*10_000:.0f}"
        if y_fmt == "reais":
            if v >= 1e6: return f"R$ {v/1e6:.1f}M"
            if v >= 1e3: return f"R$ {v/1e3:.0f}k"
            return f"R$ {v:.0f}"
        return f"{v:.2f}"

    svg = [f'<svg width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}" '
           f'xmlns="http://www.w3.org/2000/svg" style="background:#fff;font-family:Segoe UI,Arial,sans-serif;">']
    if titulo:
        svg.append(f'<text x="{largura/2:.0f}" y="14" font-size="13" font-weight="bold" '
                   f'fill="#1a3a6c" text-anchor="middle">{titulo}</text>')

    # Grid Y
    for i in range(5):
        yv = ymin + (yrange * i / 4)
        y = _y(yv)
        svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{largura - margin_right}" y2="{y:.1f}" '
                   f'stroke="#e5e7eb" stroke-width="0.5"/>')
        svg.append(f'<text x="{margin_left - 4}" y="{y+3:.1f}" font-size="9" fill="#666" '
                   f'text-anchor="end">{_fmt_y(yv)}</text>')

    # Areas empilhadas
    for idx_s, nome in enumerate(nomes):
        cor = cores[idx_s % len(cores)]
        low = stacks[idx_s]
        high = stacks[idx_s + 1]
        pts = []
        for i in range(n):
            pts.append((_x(i), _y(high[i])))
        for i in range(n - 1, -1, -1):
            pts.append((_x(i), _y(low[i])))
        path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"
        svg.append(f'<path d="{path}" fill="{cor}" fill-opacity="0.55" stroke="{cor}" stroke-width="0.5"/>')

    # X labels
    for i in range(5):
        idx = int(i / 4 * (n - 1))
        x = _x(idx)
        d = datas[idx]
        try:
            label = datetime.fromisoformat(d).strftime("%b/%y")
        except Exception:
            label = str(d)[:7]
        svg.append(f'<text x="{x:.1f}" y="{altura - 8}" font-size="9" fill="#666" '
                   f'text-anchor="middle">{label}</text>')

    # Legenda
    lx = margin_left + 10
    ly = margin_top - 5
    for idx_s, nome in enumerate(nomes):
        cor = cores[idx_s % len(cores)]
        svg.append(f'<rect x="{lx}" y="{ly-8}" width="10" height="10" fill="{cor}" fill-opacity="0.55"/>')
        svg.append(f'<text x="{lx + 14}" y="{ly}" font-size="10" fill="#333">{nome}</text>')
        lx += max(90, len(nome) * 7 + 25)

    svg.append('</svg>')
    return "".join(svg)


def _svg_donut(pct, cor="#1565c0", raio=48):
    """SVG donut mostrando pct (0..1)."""
    if pct is None: pct = 0
    pct = max(0, min(pct, 1.5))   # permite ultrapassar 100% mas capado a 150%
    if pct > 1:
        cor = "#dc3545"
    elif pct > 0.8:
        cor = "#fd7e14"
    elif pct > 0.5:
        cor = "#ffc107"
    else:
        cor = "#28a745"

    tam = raio * 2 + 20
    cx = tam / 2
    cy = tam / 2
    r = raio
    stroke_w = 12

    circ = 2 * math.pi * r
    fill_len = circ * min(pct, 1.0)
    dash = f"{fill_len:.2f} {circ:.2f}"

    svg = [f'<svg width="{tam}" height="{tam}" viewBox="0 0 {tam} {tam}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#e5e7eb" stroke-width="{stroke_w}"/>')
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{cor}" stroke-width="{stroke_w}" '
               f'stroke-dasharray="{dash}" transform="rotate(-90 {cx} {cy})" stroke-linecap="butt"/>')
    svg.append(f'<text x="{cx}" y="{cy - 2}" font-size="18" font-weight="bold" '
               f'fill="{cor}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif">{pct*100:.0f}%</text>')
    svg.append(f'<text x="{cx}" y="{cy + 15}" font-size="9" fill="#666" text-anchor="middle" font-family="Segoe UI">do limite</text>')
    svg.append('</svg>')
    return "".join(svg)


# =============================================================================
# Sharpe robusto
# =============================================================================

def _sharpe_robusto(snap_all, vol252):
    if not snap_all or not vol252 or vol252 <= 0:
        return None
    all_ret = [float(s.get("retorno_dtd") or 0) for s in snap_all]
    all_cdi_raw = [s.get("cdi_dtd") for s in snap_all]
    n_faltando = sum(1 for c in all_cdi_raw if c is None)
    if n_faltando > 0.3 * len(all_cdi_raw):
        return None
    all_cdi = [float(c or 0) for c in all_cdi_raw]
    if len(all_ret) < 60:
        return None
    ret_mean_anual = sum(all_ret) / len(all_ret) * 252
    cdi_mean_anual = sum(all_cdi) / len(all_cdi) * 252
    return (ret_mean_anual - cdi_mean_anual) / vol252


# =============================================================================
# Montagem do HTML
# =============================================================================

def montar_html(snapshots_all, governance, eventos, pct_capital=0.01):
    if not snapshots_all:
        return "Dash Risco - sem dados", "<p>Nao ha snapshots.</p>", {}

    hoje = snapshots_all[-1]
    data_iso = hoje["Data"]
    data_br = datetime.fromisoformat(data_iso).strftime("%d/%m/%Y")

    # Metricas basicas do dia
    cota = float(hoje.get("cota") or 0)
    ret_dtd = float(hoje.get("retorno_dtd") or 0)
    cdi_dtd = float(hoje.get("cdi_dtd") or 0)
    ret_mtd = hoje.get("retorno_mtd")
    ret_ytd = hoje.get("retorno_ytd")
    cdi_mtd = hoje.get("cdi_mtd")
    cdi_ytd = hoje.get("cdi_ytd")
    pl_total = hoje.get("pl_total")
    pl_risco = (pl_total * pct_capital) if pl_total else None

    # % CDI por periodo
    def _pct_cdi(ret, cdi):
        if ret is None or cdi is None or cdi == 0: return None
        try: return float(ret) / float(cdi)
        except Exception: return None

    pct_cdi_dtd = _pct_cdi(ret_dtd, cdi_dtd)
    pct_cdi_mtd = _pct_cdi(ret_mtd, cdi_mtd)
    pct_cdi_ytd = _pct_cdi(ret_ytd, cdi_ytd)

    # VaR de carteira
    var_cart_hist_R    = hoje.get("var_carteira_hist_reais")
    var_cart_hist_bps  = hoje.get("var_carteira_hist_bps")
    var_cart_ewma_R    = hoje.get("var_carteira_ewma_reais")
    var_cart_ewma_bps  = hoje.get("var_carteira_ewma_bps")
    consumo_hist       = hoje.get("consumo_hist_pct") or 0
    consumo_ewma       = hoje.get("consumo_ewma_pct") or 0
    var_limite_efet_bps = float(hoje.get("var_limite_efet_bps") or 1.0)
    limite_R = (var_limite_efet_bps / 10_000 * pl_total) if pl_total else None

    # Cota metrics
    dd_atual = hoje.get("dd_atual")
    dd_max   = hoje.get("dd_max_hist")
    vol20    = hoje.get("vol_20d")
    vol60    = hoje.get("vol_60d")
    vol252   = hoje.get("vol_252d")
    var_hist_ew_bps = hoje.get("var_hist_ew_bps")
    var_hist_ew_R = (abs(var_hist_ew_bps) / 10_000 * pl_risco) if (var_hist_ew_bps and pl_risco) else None

    # DV01
    dv01_total = hoje.get("dv01_total")
    dv01_nom   = hoje.get("dv01_juros_nom")
    dv01_real  = hoje.get("dv01_juros_real")
    dv01_treas = hoje.get("dv01_treasury")
    dv01_ntnb  = hoje.get("dv01_ntnb")

    # Governance
    stop_ativo = bool(hoje.get("stop_loss_ativo", False))
    fator_gov = float(hoje.get("fator_governance") or 1.0)
    var_base = float(hoje.get("var_limite_base_bps") or 1.0)

    # Sharpe e vol historica total da cota (todos os retornos_dtd)
    all_ret = [float(s.get("retorno_dtd") or 0) for s in snapshots_all]
    n_ret = len(all_ret)
    if n_ret >= 2:
        m = sum(all_ret) / n_ret
        var = sum((r - m) ** 2 for r in all_ret) / (n_ret - 1)
        vol_total_anual = math.sqrt(var) * math.sqrt(252)
    else:
        vol_total_anual = None

    sharpe = _sharpe_robusto(snapshots_all, vol252)

    # ── Stress DV01 por estrategia (bps configurados) ─────────────
    # NTNB agrupado como Juros Reais. Treasury nao entra no stress.
    STRESS_BPS = {
        "juros_nom":  100,   # DI: choque de 100 bps
        "juros_real": 50,    # DAP + NTNB: 50 bps
    }
    dv01_real_total = (dv01_real or 0) + (dv01_ntnb or 0)   # agrupa DAP + NTNB
    dv01_nom_total  = (dv01_nom  or 0)
    dv01_agg_total  = dv01_nom_total + dv01_real_total

    stress_nom_R   = dv01_nom_total  * STRESS_BPS["juros_nom"]
    stress_real_R  = dv01_real_total * STRESS_BPS["juros_real"]
    stress_total_R = stress_nom_R + stress_real_R

    # Stress em bps sobre PL RISCO (nao PL total)
    def _bps_do_risco(v_reais):
        if not pl_risco or pl_risco == 0: return 0.0
        return (v_reais / pl_risco) * 10_000
    dv01_nom_bps_risco   = _bps_do_risco(dv01_nom_total)
    dv01_real_bps_risco  = _bps_do_risco(dv01_real_total)
    dv01_total_bps_risco = _bps_do_risco(dv01_agg_total)
    stress_nom_bps_risco   = _bps_do_risco(stress_nom_R)
    stress_real_bps_risco  = _bps_do_risco(stress_real_R)
    stress_total_bps_risco = _bps_do_risco(stress_total_R)

    # ─── Construcao das series historicas ────────────────────────────────
    datas = [s["Data"] for s in snapshots_all]

    # Cota vs CDI — filtra valores None/0 na cota E cotas fora do range razoavel
    # (0.5 < c < 10): elimina residuos de v1.4 (base 1000) misturados com v2.x (base 1)
    cotas_raw = []
    for s in snapshots_all:
        c = s.get("cota")
        if c is None:
            cotas_raw.append(None)
        else:
            try:
                cf = float(c)
                if 0.5 < cf < 10:
                    cotas_raw.append(cf)
                else:
                    cotas_raw.append(None)
            except Exception:
                cotas_raw.append(None)

    ret_cdi_diario = [float(s.get("cdi_dtd") or 0) for s in snapshots_all]

    # Primeiro valor valido de cota para normalizar
    base_cota = next((c for c in cotas_raw if c is not None), 1.0)
    cota_norm = [(c / base_cota) if c is not None else None for c in cotas_raw]

    # CDI acumulado normalizado a 1
    cdi_acum = []
    acc = 1.0
    for r in ret_cdi_diario:
        acc *= (1 + r)
        cdi_acum.append(acc)

    # Cota vs CDI no mes (ultimos ~21 dias)
    mes_i = max(0, len(datas) - 21)
    datas_mes = datas[mes_i:]
    cotas_mes_raw = cotas_raw[mes_i:]
    cdi_diario_mes = ret_cdi_diario[mes_i:]
    base_cota_mes = next((c for c in cotas_mes_raw if c is not None), 1.0)
    cota_mes_norm = [(c / base_cota_mes) if c is not None else None for c in cotas_mes_raw]
    cdi_mes_acum = []
    acc = 1.0
    for r in cdi_diario_mes:
        acc *= (1 + r)
        cdi_mes_acum.append(acc)

    # Vol 20d anualizada
    vol20_hist = [s.get("vol_20d") for s in snapshots_all]

    # DD historico
    dd_hist = [s.get("dd_atual") for s in snapshots_all]

    # Consumo VaR historico (hist e ewma)
    consumo_hist_serie = [s.get("consumo_hist_pct") for s in snapshots_all]
    consumo_ewma_serie = [s.get("consumo_ewma_pct") for s in snapshots_all]

    # DV01 empilhado (apenas ultimo dia tem valor)
    dv01_nom_serie   = [s.get("dv01_juros_nom") or 0 for s in snapshots_all]
    dv01_real_serie  = [s.get("dv01_juros_real") or 0 for s in snapshots_all]
    dv01_treas_serie = [s.get("dv01_treasury") or 0 for s in snapshots_all]
    dv01_ntnb_serie  = [s.get("dv01_ntnb") or 0 for s in snapshots_all]

    # ─── Graficos ───────────────────────────────────────────────────────
    chart_cota_mes = _svg_line_chart(datas_mes,
        {"Cota": cota_mes_norm, "CDI": cdi_mes_acum},
        largura=560, altura=180, titulo="Cota vs CDI - Mes",
        cores=["#1565c0", "#7f7f7f"], y_fmt="num")

    chart_cota_hist = _svg_line_chart(datas,
        {"Cota": cota_norm, "CDI": cdi_acum},
        largura=800, altura=200, titulo="Cota vs CDI - Historico",
        cores=["#1565c0", "#7f7f7f"], y_fmt="num")

    chart_vol_hist = _svg_line_chart(datas,
        {"Vol 20d anualizada": vol20_hist},
        largura=800, altura=180, titulo="Volatilidade 20d anualizada",
        cores=["#c62828"], y_fmt="pct")

    chart_dd_hist = _svg_line_chart(datas,
        {"Drawdown": dd_hist},
        largura=800, altura=180, titulo="Drawdown historico",
        cores=["#dc3545"], y_fmt="pct", y_zero_center=True)

    chart_consumo_var = _svg_line_chart(datas,
        {"HIST 3y": consumo_hist_serie, "EWMA": consumo_ewma_serie},
        largura=800, altura=180, titulo="Utilizacao do risco (VaR / limite)",
        cores=["#1565c0", "#fd7e14"], y_fmt="pct")

    # (chart DV01 histórico removido — sem backfill de DV01 nao vale plotar)

    # ─── Donut do consumo VaR (usa EWMA) ─────────────────────────────────
    donut_ewma = _svg_donut(consumo_ewma)
    donut_hist = _svg_donut(consumo_hist)

    # ─── Banners ────────────────────────────────────────────────────────
    banner = ""
    if stop_ativo:
        banner += (f'<div style="background:#dc3545;color:white;padding:14px;'
                   f'font-weight:bold;text-align:center;border-radius:6px;margin:10px 0;">'
                   f'STOP-LOSS ATIVO (por DD) — VaR reduzido para {var_limite_efet_bps:.2f} bps '
                   f'({fator_gov*100:.0f}% do orcamento base {var_base:.2f} bps)'
                   f'</div>')
    if consumo_hist > 1.0:
        banner += (f'<div style="background:#fd7e14;color:white;padding:12px;'
                   f'font-weight:bold;text-align:center;border-radius:6px;margin:8px 0;">'
                   f'ALERTA: VaR HIST de carteira EXCEDEU limite ({consumo_hist*100:.1f}%) — sinalizar reducao de posicoes'
                   f'</div>')
    if consumo_ewma > 1.0:
        banner += (f'<div style="background:#fd7e14;color:white;padding:12px;'
                   f'font-weight:bold;text-align:center;border-radius:6px;margin:8px 0;">'
                   f'ALERTA: VaR EWMA de carteira EXCEDEU limite ({consumo_ewma*100:.1f}%)'
                   f'</div>')

    # ─── HTML ───────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: Segoe UI, Arial, sans-serif; color: #222; font-size: 14px; max-width: 900px; margin: 0 auto; padding: 10px; }}
h1 {{ color: #1a3a6c; border-bottom: 3px solid #1a3a6c; padding-bottom: 8px; margin-bottom: 6px; }}
h2 {{ color: #1a3a6c; margin-top: 32px; font-size: 18px; border-bottom: 2px solid #1a3a6c; padding-bottom: 4px; }}
h3 {{ color: #1a3a6c; margin-top: 20px; font-size: 15px; }}
.metric-grid {{ display: table; width: 100%; border-collapse: collapse; margin: 12px 0; }}
.metric {{ display: table-cell; padding: 12px 14px; border: 1px solid #ddd; background: #f8f9fa; vertical-align: top; }}
.metric-label {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
.metric-value {{ font-size: 20px; font-weight: bold; line-height: 1.2; }}
.metric-value-secondary {{ font-size: 15px; color: #444; font-weight: 600; margin-top: 2px; }}
.chart-container {{ margin: 16px 0; padding: 10px; background: #fff; border: 1px solid #ddd; border-radius: 4px; text-align: center; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; font-size: 13px; }}
th {{ background: #1a3a6c; color: white; }}
tr:nth-child(even) td {{ background: #f8f9fa; }}
.section-intro {{ background:#eef; padding:10px; border-radius:6px; font-size:12px; color:#333; margin:6px 0 16px 0; }}
</style></head><body>

<h1>Dash Risco - {data_br}</h1>
{banner}

<!-- ==================== SECAO 1: RESUMO DO DIA ==================== -->
<h2>1. Resumo do dia</h2>

<!-- Cota + Retornos periodo -->
<div class="metric-grid">
  <div class="metric" style="width:22%;">
    <div class="metric-label">Cota</div>
    <div class="metric-value">{cota:.4f}</div>
  </div>
  <div class="metric" style="width:26%;">
    <div class="metric-label">Retorno DIA</div>
    <div class="metric-value" style="color:{_cor_ret(ret_dtd)}">{_fmt_bps(ret_dtd, sinal=True)}</div>
    <div class="metric-value-secondary">{'-' if pct_cdi_dtd is None else f'{pct_cdi_dtd*100:.0f}% do CDI'}</div>
  </div>
  <div class="metric" style="width:26%;">
    <div class="metric-label">Retorno MES</div>
    <div class="metric-value" style="color:{_cor_ret(ret_mtd)}">{_fmt_pct(ret_mtd, sinal=True)}</div>
    <div class="metric-value-secondary">{'-' if pct_cdi_mtd is None else f'{pct_cdi_mtd*100:.0f}% do CDI'}</div>
  </div>
  <div class="metric" style="width:26%;">
    <div class="metric-label">Retorno ANO</div>
    <div class="metric-value" style="color:{_cor_ret(ret_ytd)}">{_fmt_pct(ret_ytd, sinal=True)}</div>
    <div class="metric-value-secondary">{'-' if pct_cdi_ytd is None else f'{pct_cdi_ytd*100:.0f}% do CDI'}</div>
  </div>
</div>

<!-- Chart cota vs CDI mes -->
<div class="chart-container">
  {chart_cota_mes}
</div>

<!-- PL + Vol/DD -->
<div class="metric-grid">
  <div class="metric" style="width:25%;">
    <div class="metric-label">PL Total</div>
    <div class="metric-value">{_fmt_reais(pl_total)}</div>
  </div>
  <div class="metric" style="width:25%;">
    <div class="metric-label">PL Risco ({pct_capital*100:.0f}%)</div>
    <div class="metric-value">{_fmt_reais(pl_risco)}</div>
  </div>
  <div class="metric" style="width:25%;">
    <div class="metric-label">Vol 20d anualizada</div>
    <div class="metric-value">{_fmt_pct(vol20)}</div>
  </div>
  <div class="metric" style="width:25%;">
    <div class="metric-label">Drawdown atual</div>
    <div class="metric-value" style="color:{_cor_dd(dd_atual)}">{_fmt_pct(dd_atual)}</div>
  </div>
</div>

<!-- DV01 + VaR carteira com donut (layout limpo) -->
<h3>Risco atual da carteira</h3>
<table style="width:100%;border:1px solid #ddd;border-collapse:collapse;">
  <tr>
    <td style="width:26%;padding:12px;vertical-align:top;background:#f8f9fa;border:1px solid #ddd;">
      <div class="metric-label">DV01 e Stress do dia</div>
      <div style="font-size:13px;color:#333;margin-top:8px;line-height:1.8;">
        <b>Juros Nominais<sup>*</sup>:</b><br>
        &nbsp;&nbsp;DV01 = {_fmt_reais(dv01_nom_total, 0)} ({dv01_nom_bps_risco:.2f} bps)<br>
        &nbsp;&nbsp;Stress = {_fmt_reais(stress_nom_R, 0)} ({stress_nom_bps_risco:.1f} bps)<br><br>
        <b>Juros Reais<sup>**</sup>:</b><br>
        &nbsp;&nbsp;DV01 = {_fmt_reais(dv01_real_total, 0)} ({dv01_real_bps_risco:.2f} bps)<br>
        &nbsp;&nbsp;Stress = {_fmt_reais(stress_real_R, 0)} ({stress_real_bps_risco:.1f} bps)<br><br>
        <b style="color:#1a3a6c;">TOTAL:</b><br>
        &nbsp;&nbsp;DV01 = {_fmt_reais(dv01_agg_total, 0)} ({dv01_total_bps_risco:.2f} bps)<br>
        &nbsp;&nbsp;Stress = {_fmt_reais(stress_total_R, 0)} ({stress_total_bps_risco:.1f} bps)
      </div>
      <div style="font-size:10px;color:#666;margin-top:10px;border-top:1px solid #ddd;padding-top:6px;">
        <sup>*</sup>Juros nominais (DI): choque de {STRESS_BPS['juros_nom']} bps<br>
        <sup>**</sup>Juros reais (DAP + NTNB): choque de {STRESS_BPS['juros_real']} bps<br>
        bps calculados sobre PL Risco.
      </div>
    </td>
    <td style="width:37%;padding:12px;text-align:center;background:#f8f9fa;border:1px solid #ddd;">
      <div class="metric-label">VaR HIST 3 anos</div>
      <div class="metric-value" style="color:{_cor_var(consumo_hist)}">{_fmt_reais(var_cart_hist_R)} / {var_cart_hist_bps or 0:.2f} bps</div>
      <div style="margin-top:8px;">{donut_hist}</div>
    </td>
    <td style="width:37%;padding:12px;text-align:center;background:#f8f9fa;border:1px solid #ddd;">
      <div class="metric-label">VaR EWMA (lambda=0.99)</div>
      <div class="metric-value" style="color:{_cor_var(consumo_ewma)}">{_fmt_reais(var_cart_ewma_R)} / {var_cart_ewma_bps or 0:.2f} bps</div>
      <div style="margin-top:8px;">{donut_ewma}</div>
    </td>
  </tr>
</table>

<div class="section-intro" style="margin-top:8px;">
  <b>Limite:</b> {_fmt_reais(limite_R)} ({var_limite_efet_bps:.2f} bps do PL total).
</div>


<!-- ==================== SECAO 2: METRICAS HISTORICAS ==================== -->
<h2>2. Metricas historicas de performance e risco</h2>

<div class="chart-container">
  {chart_cota_hist}
</div>

<div class="chart-container">
  {chart_vol_hist}
</div>

<div class="chart-container">
  {chart_dd_hist}
</div>

<div class="chart-container">
  {chart_consumo_var}
</div>

<!-- Metricas historicas da cota -->
<h3>Metricas historicas da cota</h3>
<div class="metric-grid">
  <div class="metric" style="width:25%;">
    <div class="metric-label">Sharpe</div>
    <div class="metric-value" style="color:{_cor_ret(sharpe)}">{f'{sharpe:.2f}' if sharpe is not None else '-'}</div>
    <div class="metric-value-secondary" style="font-size:11px;color:#666;font-weight:normal;">(ret-CDI) / vol252</div>
  </div>
  <div class="metric" style="width:25%;">
    <div class="metric-label">Vol realizada</div>
    <div class="metric-value">{_fmt_pct(vol_total_anual)}</div>
  </div>
  <div class="metric" style="width:25%;">
    <div class="metric-label">VaR (95%)</div>
    <div class="metric-value">{_fmt_reais(var_hist_ew_R)} / {abs(var_hist_ew_bps) if var_hist_ew_bps else 0:.2f} bps</div>
  </div>
  <div class="metric" style="width:25%;">
    <div class="metric-label">Drawdown</div>
    <div class="metric-value">{_fmt_pct(dd_max)}</div>
  </div>
</div>

<h3>Ultimos 5 dias</h3>
"""

    # Tabela ultimos 5 dias
    html += ("<table><tr><th>Data</th><th>Cota</th><th>Retorno</th><th>Alpha</th>"
             "<th>VaR Cart HIST</th><th>VaR Cart EWMA</th><th>Consumo EWMA</th>"
             "<th>Vol 20d</th><th>DD</th></tr>")
    for s in snapshots_all[-5:][::-1]:
        d_br = datetime.fromisoformat(s["Data"]).strftime("%d/%m/%Y")
        _ret = float(s.get("retorno_dtd") or 0)
        _cdi = float(s.get("cdi_dtd") or 0)
        _var_ch = s.get("var_carteira_hist_reais")
        _var_ce = s.get("var_carteira_ewma_reais")
        _consumo_e = s.get("consumo_ewma_pct")
        html += (f"<tr><td>{d_br}</td>"
                 f"<td>{float(s.get('cota') or 0):.4f}</td>"
                 f"<td>{_ret * 10_000:+.2f} bps</td>"
                 f"<td>{(_ret - _cdi) * 10_000:+.2f} bps</td>"
                 f"<td>{_fmt_reais(_var_ch)}</td>"
                 f"<td>{_fmt_reais(_var_ce)}</td>"
                 f"<td>{(_consumo_e * 100 if _consumo_e else 0):.1f}%</td>"
                 f"<td>{_fmt_pct(s.get('vol_20d'))}</td>"
                 f"<td>{_fmt_pct(s.get('dd_atual'))}</td></tr>")
    html += "</table>"

    if eventos:
        html += "<h3>Eventos recentes</h3><ul>"
        for e in eventos[:5]:
            ts = e.get("timestamp", "")[:10]
            sev = e.get("severidade", "")
            cor_sev = {"critical": "#dc3545", "warn": "#ffc107", "info": "#17a2b8"}.get(sev, "#666")
            html += f'<li><span style="color:{cor_sev};font-weight:bold">[{sev.upper()}]</span> {ts} - {e.get("titulo", "")}</li>'
        html += "</ul>"

    html += f"""
<p style="margin-top:32px;color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:12px;">
Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}.
Cota = retorno total (ativo + carry LFT). PL Risco = {pct_capital*100:.0f}% do PL Total.
Stop-loss aciona apenas por DD; VaR e alerta para reducao prospectiva de posicoes.
</p>
</body></html>"""

    ret_bps = ret_dtd * 10_000
    sinal = "+" if ret_bps >= 0 else ""
    assunto = DEFAULT_ASSUNTO.format(
        data_br=data_br, cota=cota, sinal=sinal, ret_bps=ret_bps,
        consumo_ewma=(consumo_ewma * 100 if consumo_ewma else 0)
    )
    dados = {
        "data": data_iso, "cota": cota, "ret_bps": ret_bps,
        "pl_total": pl_total,
        "var_cart_hist_R": var_cart_hist_R, "consumo_hist": consumo_hist,
        "var_cart_ewma_R": var_cart_ewma_R, "consumo_ewma": consumo_ewma,
        "var_cota_ew_R": var_hist_ew_R,
        "limite_R": limite_R,
        "dd_atual": dd_atual, "stop_ativo": stop_ativo,
        "sharpe": sharpe, "pct_cdi_ytd": pct_cdi_ytd,
        "dv01_total": dv01_total,
    }
    return assunto, html, dados


# =============================================================================
# Envio via Outlook
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


def run(dry_run=False, override_to=None):
    client = _get_client()
    config = _get_config(client)
    pct_capital = float(config.get("pct_capital_risco", 0.01))
    emails_diario = config.get("emails_diario", ["marcos.freitas@afinvest.com.br"])
    if isinstance(emails_diario, str):
        emails_diario = [emails_diario]
    destinatarios = [override_to] if override_to else emails_diario

    snapshots_all = _load_snapshot_serie_completa(client)
    if not snapshots_all:
        print("[email] sem snapshots.")
        return

    governance = _load_governance(client)
    eventos = _load_eventos_recentes(client, n=5)

    assunto, html, dados = montar_html(snapshots_all, governance, eventos, pct_capital=pct_capital)
    print(f"[email] assunto: {assunto}")
    print(f"[email] cota={dados.get('cota'):.4f}  ret={dados.get('ret_bps'):+.2f}bps")
    print(f"[email] VaR CARTEIRA HIST : R$ {(dados.get('var_cart_hist_R') or 0):,.0f} = {(dados.get('consumo_hist') or 0)*100:.1f}%")
    print(f"[email] VaR CARTEIRA EWMA : R$ {(dados.get('var_cart_ewma_R') or 0):,.0f} = {(dados.get('consumo_ewma') or 0)*100:.1f}%")
    print(f"[email] DV01 total: R$ {(dados.get('dv01_total') or 0):,.0f}/bp | Sharpe: {dados.get('sharpe')}")

    if dry_run:
        out = Path("email_diario_preview.html")
        out.write_text(html, encoding="utf-8")
        print(f"[email] dry-run - HTML em {out.resolve()}")
        return
    try:
        enviar_via_outlook(destinatarios, assunto, html)
        print(f"[email] OK enviado")
    except Exception as e:
        print(f"[email] ERRO ao enviar: {e}")
        out = Path("email_diario_ERRO.html")
        out.write_text(html, encoding="utf-8")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--to", default=None)
    args = ap.parse_args()
    run(dry_run=args.dry_run, override_to=args.to)
