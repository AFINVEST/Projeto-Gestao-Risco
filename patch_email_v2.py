"""patch_email_v2.py - aplica melhorias no enviar_email_diario.py"""
from pathlib import Path
import shutil, sys, datetime as dt

f = Path("enviar_email_diario.py")
if not f.exists():
    sys.exit("enviar_email_diario.py nao encontrado")

bak = f.with_suffix(f".py.bak_{dt.datetime.now():%Y%m%d_%H%M%S}")
shutil.copy2(f, bak)
print(f"[backup] {bak.name}")

src = f.read_text(encoding="utf-8")

# --- Fix 1: assunto novo ---
old_assunto = 'DEFAULT_ASSUNTO = "Dash Risco - {data_br} - Cota {cota:.4f} ({sinal}{ret_bps:.2f}bps) - VaR carteira EWMA {consumo_ewma:.0f}%"'
new_assunto = 'DEFAULT_ASSUNTO = "Cota {data_br} - Book Juros AF - {cota:.4f}"'
if old_assunto in src:
    src = src.replace(old_assunto, new_assunto)
    print("[ok] assunto novo aplicado")
else:
    print("[warn] linha do assunto nao encontrada (talvez ja patchada)")

# --- Fix 2: format do assunto (remove args extras) ---
old_fmt = '''    assunto = DEFAULT_ASSUNTO.format(
        data_br=data_br, cota=cota, sinal=sinal, ret_bps=ret_bps,
        consumo_ewma=(consumo_ewma * 100 if consumo_ewma else 0)
    )'''
new_fmt = '    assunto = DEFAULT_ASSUNTO.format(data_br=data_br, cota=cota)'
if old_fmt in src:
    src = src.replace(old_fmt, new_fmt)
    print("[ok] format do assunto ajustado")

# --- Fix 3: dados dict expandido (mtd/ytd + %CDI de cada periodo) ---
old_dados = '''    dados = {
        "data": data_iso, "cota": cota, "ret_bps": ret_bps,
        "pl_total": pl_total,
        "var_cart_hist_R": var_cart_hist_R, "consumo_hist": consumo_hist,
        "var_cart_ewma_R": var_cart_ewma_R, "consumo_ewma": consumo_ewma,
        "var_cota_ew_R": var_hist_ew_R,
        "limite_R": limite_R,
        "dd_atual": dd_atual, "stop_ativo": stop_ativo,
        "sharpe": sharpe, "pct_cdi_ytd": pct_cdi_ytd,
        "dv01_total": dv01_total,
    }'''
new_dados = '''    dados = {
        "data": data_iso, "cota": cota, "ret_bps": ret_bps,
        "ret_dtd": ret_dtd, "ret_mtd": ret_mtd, "ret_ytd": ret_ytd,
        "cdi_dtd": cdi_dtd, "cdi_mtd": cdi_mtd, "cdi_ytd": cdi_ytd,
        "pct_cdi_dtd": pct_cdi_dtd, "pct_cdi_mtd": pct_cdi_mtd, "pct_cdi_ytd": pct_cdi_ytd,
        "pl_total": pl_total,
        "var_cart_hist_R": var_cart_hist_R, "consumo_hist": consumo_hist,
        "var_cart_ewma_R": var_cart_ewma_R, "consumo_ewma": consumo_ewma,
        "var_cota_ew_R": var_hist_ew_R,
        "limite_R": limite_R,
        "dd_atual": dd_atual, "stop_ativo": stop_ativo,
        "sharpe": sharpe,
        "dv01_total": dv01_total,
    }'''
if old_dados in src:
    src = src.replace(old_dados, new_dados)
    print("[ok] dados dict expandido (mtd/ytd + %CDI)")

# --- Fix 4: reescreve _resumo_email_texto (fix charset + mtd/ytd + bps/CDI) ---
old_fn_marker = 'def _resumo_email_texto(dados: dict, link_html: str) -> str:'
end_marker = '</body></html>"""\n\n\ndef enviar_via_outlook'
i_start = src.find(old_fn_marker)
i_end = src.find(end_marker)
if i_start == -1 or i_end == -1:
    print("[warn] _resumo_email_texto nao encontrada, pulando")
else:
    new_fn = '''def _resumo_email_texto(dados: dict, link_html: str) -> str:
    """Gera resumo simples pra ser corpo do email (funciona em qualquer cliente)."""
    def _fmt_R(v): return _fmt_reais(v) if v is not None else "-"
    def _fmt_p(v): return f"{v*100:.2f}%" if v is not None else "-"
    def _fmt_c(v): return f"{v*100:.0f}%" if v is not None else "-"
    def _fmt_b(v): return f"{v*10_000:+.2f} bps" if v is not None else "-"

    data_br = datetime.fromisoformat(dados["data"]).strftime("%d/%m/%Y")
    ret_dtd = dados.get("ret_dtd")
    ret_mtd = dados.get("ret_mtd")
    ret_ytd = dados.get("ret_ytd")
    pct_dtd = dados.get("pct_cdi_dtd")
    pct_mtd = dados.get("pct_cdi_mtd")
    pct_ytd = dados.get("pct_cdi_ytd")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:Segoe UI,Arial,sans-serif;color:#222;font-size:14px;">
<h2 style="color:#1a3a6c;border-bottom:2px solid #1a3a6c;padding-bottom:6px;">Dash Risco - {data_br}</h2>

<h3 style="color:#1a3a6c;">Resumo</h3>
<table style="border-collapse:collapse;">
<tr><td style="padding:4px 12px;"><b>Cota:</b></td><td>{dados.get('cota'):.4f}</td></tr>
<tr><td style="padding:4px 12px;"><b>Retorno DIA:</b></td><td>{_fmt_b(ret_dtd)} &nbsp;({_fmt_c(pct_dtd)} do CDI)</td></tr>
<tr><td style="padding:4px 12px;"><b>Retorno MES:</b></td><td>{_fmt_p(ret_mtd)} &nbsp;({_fmt_c(pct_mtd)} do CDI)</td></tr>
<tr><td style="padding:4px 12px;"><b>Retorno ANO:</b></td><td>{_fmt_p(ret_ytd)} &nbsp;({_fmt_c(pct_ytd)} do CDI)</td></tr>
<tr><td style="padding:4px 12px;"><b>PL Total:</b></td><td>{_fmt_R(dados.get('pl_total'))}</td></tr>
</table>

<h3 style="color:#1a3a6c;">Risco</h3>
<table style="border-collapse:collapse;">
<tr><td style="padding:4px 12px;"><b>VaR HIST 3y:</b></td><td>{_fmt_R(dados.get('var_cart_hist_R'))} ({_fmt_c(dados.get('consumo_hist'))} do limite)</td></tr>
<tr><td style="padding:4px 12px;"><b>VaR EWMA:</b></td><td>{_fmt_R(dados.get('var_cart_ewma_R'))} ({_fmt_c(dados.get('consumo_ewma'))} do limite)</td></tr>
<tr><td style="padding:4px 12px;"><b>Limite:</b></td><td>{_fmt_R(dados.get('limite_R'))}</td></tr>
<tr><td style="padding:4px 12px;"><b>DV01 total:</b></td><td>{_fmt_R(dados.get('dv01_total'))}/bp</td></tr>
<tr><td style="padding:4px 12px;"><b>Drawdown atual:</b></td><td>{_fmt_p(dados.get('dd_atual'))}</td></tr>
<tr><td style="padding:4px 12px;"><b>Sharpe:</b></td><td>{f"{dados.get('sharpe'):.2f}" if dados.get('sharpe') is not None else '-'}</td></tr>
</table>

<p style="margin-top:20px;padding:12px;background:#eef;border-radius:6px;">
<b>Relatorio completo com graficos:</b><br>
<a href="file:///{link_html}" style="color:#1a3a6c;font-weight:bold;">
  Abrir {Path(link_html).name}
</a><br>
<span style="font-size:11px;color:#666;">
Ou copie o caminho: <code>{link_html}</code>
</span>
</p>

<p style="margin-top:32px;color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:12px;">
Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}.
</p>
</body></html>"""


'''
    src = src[:i_start] + new_fn + src[i_end+len('</body></html>"""\n\n\n'):]
    print("[ok] _resumo_email_texto reescrita (charset UTF-8 + MTD/YTD + bps/CDI)")

f.write_text(src, encoding="utf-8")
print(f"[done] {f} atualizado")
