"""
enviar_email_diario.py  (v1.3 — Sharpe correto + % CDI)
=========================================================

MUDANÇAS v1.3:
  - Sharpe = (retorno_anualizado_carteira − CDI_anualizado) / vol_252d
  - Novo card "% CDI" (retorno_YTD / cdi_YTD × 100)
  - Novo card "Excesso vs CDI" em bps
  - Usa colunas cdi_dtd/mtd/ytd populadas pelo snapshot v1.4
"""
from __future__ import annotations
import os
import sys
import argparse
import math
from pathlib import Path
from datetime import datetime

try:
    from supabase import create_client
except ImportError:
    print("ERRO: pip install supabase", file=sys.stderr)
    sys.exit(1)


DEFAULT_ASSUNTO = "Dash Risco — {data_br} — Cota {cota:.4f} ({sinal}{ret_bps:.2f}bps) — Consumo VaR {consumo_pct:.0f}%"


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


def _load_ultimos_n(client, n=30):
    resp = (client.table("snapshot_diario")
                  .select("*").order("Data", desc=True).limit(n).execute())
    return list(reversed(resp.data or []))


def _load_all(client):
    """Carrega toda série histórica pra cálculo de Sharpe."""
    resp = (client.table("snapshot_diario")
                  .select("Data,retorno_dtd,cdi_dtd")
                  .order("Data").execute())
    return resp.data or []


def _load_governance(client):
    resp = (client.table("governance_state").select("*").eq("regra", "stop_loss_dd").execute())
    return resp.data[0] if resp.data else None


def _load_eventos_recentes(client, n=5):
    resp = (client.table("eventos_risco").select("*").order("timestamp", desc=True).limit(n).execute())
    return resp.data or []


def _fmt_reais(v):
    if v is None: return "—"
    try:
        return f"R$ {float(v):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "—"


def _fmt_pct(v_frac, casas=2):
    if v_frac is None: return "—"
    try:
        return f"{float(v_frac) * 100:.{casas}f}%"
    except Exception:
        return "—"


def _fmt_bps_raw(v):
    if v is None: return "—"
    try:
        return f"{float(v):.2f} bps"
    except Exception:
        return "—"


def _cor_var_consumido(pct):
    if pct is None: return "#888"
    if pct < 0.5: return "#28a745"
    if pct < 0.8: return "#ffc107"
    return "#dc3545"


def _cor_dd(dd_frac):
    if dd_frac is None or dd_frac >= 0: return "#28a745"
    if dd_frac > -0.02: return "#28a745"
    if dd_frac > -0.04: return "#ffc107"
    return "#dc3545"


def _cor_ret(v):
    if v is None: return "#888"
    return "#28a745" if v >= 0 else "#dc3545"


def montar_html(snapshots, snap_all, governance, eventos, pct_capital=0.01):
    if not snapshots:
        return "Dash Risco — sem dados", "<p>Não há snapshots.</p>", {}

    hoje = snapshots[-1]
    data_iso = hoje["Data"]
    data_br = datetime.fromisoformat(data_iso).strftime("%d/%m/%Y")
    cota = float(hoje.get("cota") or 0)
    ret_dtd = float(hoje.get("retorno_dtd") or 0)
    cdi_dtd = float(hoje.get("cdi_dtd") or 0)
    ret_mtd = hoje.get("retorno_mtd")
    ret_ytd = hoje.get("retorno_ytd")
    cdi_mtd = hoje.get("cdi_mtd")
    cdi_ytd = hoje.get("cdi_ytd")

    ret_bps = ret_dtd * 10_000
    cdi_bps = cdi_dtd * 10_000
    sinal = "+" if ret_bps >= 0 else ""

    pl_total = hoje.get("pl_total")
    pl_risco = (pl_total * pct_capital) if pl_total else None

    var_bw = hoje.get("var_hist_bw_bps")
    cvar = hoje.get("cvar_bps")
    var_limite_pct_total = float(hoje.get("var_limite_efet_bps") or 1.0)
    var_base = float(hoje.get("var_limite_base_bps") or 1.0)

    if var_bw is not None and var_limite_pct_total > 0:
        var_consumido_pct = (abs(var_bw) * pct_capital) / var_limite_pct_total
    else:
        var_consumido_pct = None
    cor_var = _cor_var_consumido(var_consumido_pct)

    var_bw_reais = (abs(var_bw) / 10_000 * pl_risco) if (var_bw is not None and pl_risco) else None
    cvar_reais = (abs(cvar) / 10_000 * pl_risco) if (cvar is not None and pl_risco) else None
    limite_reais = (var_limite_pct_total / 10_000 * pl_total) if pl_total else None

    dd_atual = hoje.get("dd_atual")
    dd_max = hoje.get("dd_max_hist")
    vol60 = hoje.get("vol_60d")
    vol20 = hoje.get("vol_20d")
    vol252 = hoje.get("vol_252d")

    stop_ativo = bool(hoje.get("stop_loss_ativo", False))
    fator_gov = float(hoje.get("fator_governance") or 1.0)

    # % CDI (YTD)
    pct_cdi_ytd = None
    if ret_ytd is not None and cdi_ytd and cdi_ytd != 0:
        pct_cdi_ytd = float(ret_ytd) / float(cdi_ytd)
    # Excesso vs CDI (YTD, em bps)
    excesso_bps_ytd = ((float(ret_ytd) - float(cdi_ytd)) * 10_000) if (ret_ytd is not None and cdi_ytd is not None) else None

    # Sharpe correto: (retorno anualizado − CDI anualizado) / vol_252d
    # Anualiza usando toda série disponível
    sharpe = None
    if snap_all and vol252 and vol252 > 0:
        try:
            all_ret = [float(s.get("retorno_dtd") or 0) for s in snap_all]
            all_cdi = [float(s.get("cdi_dtd") or 0) for s in snap_all]
            if len(all_ret) >= 60:
                ret_mean_anual = sum(all_ret) / len(all_ret) * 252
                cdi_mean_anual = sum(all_cdi) / len(all_cdi) * 252
                sharpe = (ret_mean_anual - cdi_mean_anual) / vol252
        except Exception:
            pass

    banner = ""
    if stop_ativo:
        banner = (f'<div style="background:#dc3545;color:white;padding:14px;'
                  f'font-weight:bold;text-align:center;border-radius:6px;margin:10px 0;">'
                  f'⚠ STOP-LOSS ATIVO — VaR reduzido para {var_limite_pct_total:.2f} bps '
                  f'({fator_gov*100:.0f}% do orçamento base de {var_base:.2f} bps)'
                  f'</div>')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: Segoe UI, Arial, sans-serif; color: #222; font-size: 14px; }}
h1 {{ color: #1a3a6c; border-bottom: 2px solid #1a3a6c; padding-bottom: 8px; }}
h2 {{ color: #1a3a6c; margin-top: 24px; font-size: 16px; }}
.metric-grid {{ display: table; width: 100%; border-collapse: collapse; margin: 12px 0; }}
.metric {{ display: table-cell; padding: 10px 14px; border: 1px solid #ddd; background: #f8f9fa; }}
.metric-label {{ font-size: 11px; color: #666; text-transform: uppercase; margin-bottom: 4px; }}
.metric-value {{ font-size: 20px; font-weight: bold; }}
.metric-sub {{ font-size: 11px; color: #666; margin-top: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: left; font-size: 13px; }}
th {{ background: #1a3a6c; color: white; }}
tr:nth-child(even) td {{ background: #f8f9fa; }}
</style></head><body>

<h1>Dash Risco — {data_br}</h1>

{banner}

<h2>📊 Resumo do dia</h2>
<div class="metric-grid">
  <div class="metric">
    <div class="metric-label">Cota</div>
    <div class="metric-value">{cota:.4f}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Retorno DTD</div>
    <div class="metric-value" style="color:{_cor_ret(ret_dtd)}">{sinal}{ret_bps:.2f} bps</div>
    <div class="metric-sub">CDI DTD: {cdi_bps:+.2f} bps</div>
  </div>
  <div class="metric">
    <div class="metric-label">PL Total</div>
    <div class="metric-value">{_fmt_reais(pl_total)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">PL Risco ({pct_capital*100:.0f}%)</div>
    <div class="metric-value">{_fmt_reais(pl_risco)}</div>
  </div>
</div>

<h2>📈 Performance vs CDI</h2>
<div class="metric-grid">
  <div class="metric">
    <div class="metric-label">Retorno MTD</div>
    <div class="metric-value" style="color:{_cor_ret(ret_mtd)}">{_fmt_pct(ret_mtd)}</div>
    <div class="metric-sub">CDI: {_fmt_pct(cdi_mtd)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Retorno YTD</div>
    <div class="metric-value" style="color:{_cor_ret(ret_ytd)}">{_fmt_pct(ret_ytd)}</div>
    <div class="metric-sub">CDI: {_fmt_pct(cdi_ytd)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">% CDI (YTD)</div>
    <div class="metric-value">{f'{pct_cdi_ytd*100:.1f}%' if pct_cdi_ytd is not None else '—'}</div>
    <div class="metric-sub">retorno_YTD / CDI_YTD</div>
  </div>
  <div class="metric">
    <div class="metric-label">Excesso vs CDI (YTD)</div>
    <div class="metric-value" style="color:{_cor_ret(excesso_bps_ytd)}">
      {f'{excesso_bps_ytd:+.0f} bps' if excesso_bps_ytd is not None else '—'}
    </div>
  </div>
</div>

<h2>⚠ VaR do dia</h2>
<div class="metric-grid">
  <div class="metric">
    <div class="metric-label">VaR (BRW λ=0.99)</div>
    <div class="metric-value" style="color:{cor_var}">{_fmt_reais(var_bw_reais)}</div>
    <div class="metric-sub">{_fmt_bps_raw(var_bw)} sobre PL Risco</div>
  </div>
  <div class="metric">
    <div class="metric-label">CVaR</div>
    <div class="metric-value">{_fmt_reais(cvar_reais)}</div>
    <div class="metric-sub">{_fmt_bps_raw(cvar)} sobre PL Risco</div>
  </div>
  <div class="metric">
    <div class="metric-label">Limite efetivo</div>
    <div class="metric-value">{_fmt_reais(limite_reais)}</div>
    <div class="metric-sub">{var_limite_pct_total:.2f} bps sobre PL Total</div>
  </div>
  <div class="metric">
    <div class="metric-label">Consumo VaR</div>
    <div class="metric-value" style="color:{cor_var}">{(var_consumido_pct*100 if var_consumido_pct else 0):.1f}%</div>
    <div class="metric-sub">VaR / Limite (em R$)</div>
  </div>
</div>

<h2>📉 Vol, DD, Sharpe</h2>
<div class="metric-grid">
  <div class="metric">
    <div class="metric-label">Vol 20d</div>
    <div class="metric-value">{_fmt_pct(vol20)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Vol 60d</div>
    <div class="metric-value">{_fmt_pct(vol60)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Vol 252d</div>
    <div class="metric-value">{_fmt_pct(vol252)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Sharpe (ret-CDI)/vol</div>
    <div class="metric-value" style="color:{_cor_ret(sharpe)}">{f'{sharpe:.2f}' if sharpe is not None else '—'}</div>
  </div>
</div>

<div class="metric-grid">
  <div class="metric">
    <div class="metric-label">Drawdown atual</div>
    <div class="metric-value" style="color:{_cor_dd(dd_atual)}">{_fmt_pct(dd_atual)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Max DD histórico</div>
    <div class="metric-value">{_fmt_pct(dd_max)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">CDI DTD</div>
    <div class="metric-value">{cdi_bps:+.2f} bps</div>
  </div>
  <div class="metric">
    <div class="metric-label">Alpha DTD (ret-CDI)</div>
    <div class="metric-value" style="color:{_cor_ret(ret_dtd - cdi_dtd)}">{(ret_dtd - cdi_dtd) * 10_000:+.2f} bps</div>
  </div>
</div>
"""

    html += "<h2>📅 Últimos 5 dias</h2>"
    html += "<table><tr><th>Data</th><th>Cota</th><th>Retorno</th><th>CDI</th><th>Alpha</th><th>VaR (R$)</th><th>Vol 60d</th><th>DD</th></tr>"
    for s in snapshots[-5:][::-1]:
        d_br = datetime.fromisoformat(s["Data"]).strftime("%d/%m/%Y")
        _var = s.get("var_hist_bw_bps")
        _pl = s.get("pl_total")
        _var_reais = (abs(_var) / 10_000 * float(_pl) * pct_capital) if (_var and _pl) else None
        _ret = float(s.get("retorno_dtd") or 0)
        _cdi = float(s.get("cdi_dtd") or 0)
        html += (f"<tr><td>{d_br}</td>"
                 f"<td>{float(s.get('cota') or 0):.4f}</td>"
                 f"<td>{_ret * 10_000:+.2f} bps</td>"
                 f"<td>{_cdi * 10_000:+.2f} bps</td>"
                 f"<td>{(_ret - _cdi) * 10_000:+.2f} bps</td>"
                 f"<td>{_fmt_reais(_var_reais)}</td>"
                 f"<td>{_fmt_pct(s.get('vol_60d'))}</td>"
                 f"<td>{_fmt_pct(s.get('dd_atual'))}</td></tr>")
    html += "</table>"

    if eventos:
        html += "<h2>🔔 Eventos recentes</h2><ul>"
        for e in eventos[:5]:
            ts = e.get("timestamp", "")[:10]
            sev = e.get("severidade", "")
            cor_sev = {"critical": "#dc3545", "warn": "#ffc107", "info": "#17a2b8"}.get(sev, "#666")
            html += f'<li><span style="color:{cor_sev};font-weight:bold">[{sev.upper()}]</span> {ts} — {e.get("titulo", "")}</li>'
        html += "</ul>"

    html += f"""
<p style="margin-top:32px;color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:12px;">
Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}. Cota reflete retorno total (ativo + carry LFT).
PL Risco = {pct_capital*100:.0f}% do PL Total. Sharpe usa vol_252d como denominador.
</p>
</body></html>"""

    assunto = DEFAULT_ASSUNTO.format(
        data_br=data_br, cota=cota, sinal=sinal, ret_bps=ret_bps,
        consumo_pct=(var_consumido_pct * 100 if var_consumido_pct else 0)
    )

    dados = {
        "data": data_iso, "cota": cota, "ret_bps": ret_bps,
        "pl_total": pl_total, "var_bw": var_bw,
        "var_reais": var_bw_reais, "limite_reais": limite_reais,
        "consumo_pct": var_consumido_pct,
        "dd_atual": dd_atual, "stop_ativo": stop_ativo,
        "sharpe": sharpe, "pct_cdi_ytd": pct_cdi_ytd,
    }
    return assunto, html, dados


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

    snapshots = _load_ultimos_n(client, n=30)
    snap_all = _load_all(client)   # série completa pra Sharpe
    if not snapshots:
        print("[email] sem snapshots.")
        return

    governance = _load_governance(client)
    eventos = _load_eventos_recentes(client, n=5)

    assunto, html, dados = montar_html(snapshots, snap_all, governance, eventos, pct_capital=pct_capital)

    print(f"[email] assunto: {assunto}")
    print(f"[email] resumo:")
    print(f"  cota: {dados.get('cota'):.4f}")
    print(f"  retorno DTD: {dados.get('ret_bps'):+.2f} bps")
    print(f"  VaR: {_fmt_reais(dados.get('var_reais'))} / limite {_fmt_reais(dados.get('limite_reais'))} = {(dados.get('consumo_pct') or 0)*100:.1f}%")
    print(f"  Sharpe: {dados.get('sharpe')}")
    print(f"  % CDI (YTD): {dados.get('pct_cdi_ytd')}")

    if dry_run:
        out = Path("email_diario_preview.html")
        out.write_text(html, encoding="utf-8")
        print(f"[email] dry-run — HTML em {out.resolve()}")
        return

    try:
        enviar_via_outlook(destinatarios, assunto, html)
        print(f"[email] ✓ enviado")
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
