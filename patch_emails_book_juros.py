"""patch_emails_book_juros.py - reformulacao emails diario+mensal"""
from __future__ import annotations
from pathlib import Path
import shutil, datetime as dt, sys


def _backup(path):
    bak = path.with_suffix(f"{path.suffix}.bak_{dt.datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(path, bak)
    print(f"[backup] {bak.name}")


def patch_diario():
    f = Path("enviar_email_diario.py")
    if not f.exists():
        print("[erro] enviar_email_diario.py nao encontrado"); return
    _backup(f)
    src = f.read_text(encoding="utf-8")
    n = 0

    for old_ass in [
        'DEFAULT_ASSUNTO = "Cota {data_br} - Book Juros AF - {cota:.4f}"',
        'DEFAULT_ASSUNTO = "Dash Risco - {data_br} - Cota {cota:.4f} ({sinal}{ret_bps:.2f}bps) - VaR carteira EWMA {consumo_ewma:.0f}%"',
    ]:
        if old_ass in src:
            src = src.replace(old_ass, 'DEFAULT_ASSUNTO = "Book Juros - {data_br}"')
            print("[diario] D1: assunto"); n += 1; break

    for old_fmt in [
        '    assunto = DEFAULT_ASSUNTO.format(data_br=data_br, cota=cota)',
        '''    assunto = DEFAULT_ASSUNTO.format(
        data_br=data_br, cota=cota, sinal=sinal, ret_bps=ret_bps,
        consumo_ewma=(consumo_ewma * 100 if consumo_ewma else 0)
    )''',
    ]:
        if old_fmt in src:
            src = src.replace(old_fmt, '    assunto = DEFAULT_ASSUNTO.format(data_br=data_br)')
            print("[diario] D2: format"); n += 1; break

    if '<h1>Dash Risco - {data_br}</h1>' in src:
        src = src.replace('<h1>Dash Risco - {data_br}</h1>', '<h1>Book Juros - {data_br}</h1>')
        print("[diario] D3: H1 anexo"); n += 1

    inicio = '\n\n<!-- ==================== SECAO 2: METRICAS HISTORICAS ==================== -->\n<h2>2. Metricas historicas de performance e risco</h2>'
    fim = '<h3>Ultimos 5 dias</h3>'
    i1 = src.find(inicio); i2 = src.find(fim)
    if i1 != -1 and i2 != -1 and i1 < i2:
        src = src[:i1] + '\n\n<h3>\u00daltimos 5 dias</h3>' + src[i2+len(fim):]
        print("[diario] D4: secao 2 removida + acento"); n += 1

    if '<h3>Ultimos 5 dias</h3>' in src:
        src = src.replace('<h3>Ultimos 5 dias</h3>', '<h3>\u00daltimos 5 dias</h3>')
        print("[diario] D5: acento"); n += 1

    fn_start = 'def _resumo_email_texto(dados: dict, link_html: str) -> str:'
    fn_end   = '\ndef enviar_via_outlook'
    i1 = src.find(fn_start); i2 = src.find(fn_end, i1)
    if i1 != -1 and i2 != -1:
        novo = 'def _resumo_email_texto(dados: dict, link_html: str) -> str:\n'
        novo += '    """Corpo do email: apenas Resumo. Sem Risco."""\n'
        novo += '    def _fmt_R(v): return _fmt_reais(v) if v is not None else "-"\n'
        novo += '    def _fmt_p(v): return f"{v*100:.2f}%" if v is not None else "-"\n'
        novo += '    def _fmt_c(v): return f"{v*100:.0f}%" if v is not None else "-"\n'
        novo += '    def _fmt_b(v): return f"{v*10_000:+.2f} bps" if v is not None else "-"\n\n'
        novo += '    data_br = datetime.fromisoformat(dados["data"]).strftime("%d/%m/%Y")\n'
        novo += '    ret_dtd = dados.get("ret_dtd")\n'
        novo += '    ret_mtd = dados.get("ret_mtd")\n'
        novo += '    ret_ytd = dados.get("ret_ytd")\n'
        novo += '    pct_dtd = dados.get("pct_cdi_dtd")\n'
        novo += '    pct_mtd = dados.get("pct_cdi_mtd")\n'
        novo += '    pct_ytd = dados.get("pct_cdi_ytd")\n\n'
        novo += '    return f"""<!DOCTYPE html>\n'
        novo += '<html><head><meta charset="utf-8"></head>\n'
        novo += '<body style="font-family:Segoe UI,Arial,sans-serif;color:#222;font-size:14px;">\n'
        novo += '<h2 style="color:#1a3a6c;border-bottom:2px solid #1a3a6c;padding-bottom:6px;">Book Juros - {data_br}</h2>\n\n'
        novo += '<h3 style="color:#1a3a6c;">Resumo</h3>\n'
        novo += '<table style="border-collapse:collapse;">\n'
        novo += '<tr><td style="padding:4px 12px;"><b>Cota:</b></td><td>{dados.get(\'cota\'):.4f}</td></tr>\n'
        novo += '<tr><td style="padding:4px 12px;"><b>Retorno DIA:</b></td><td>{_fmt_b(ret_dtd)} &nbsp;({_fmt_c(pct_dtd)} do CDI)</td></tr>\n'
        novo += '<tr><td style="padding:4px 12px;"><b>Retorno M\u00caS:</b></td><td>{_fmt_p(ret_mtd)} &nbsp;({_fmt_c(pct_mtd)} do CDI)</td></tr>\n'
        novo += '<tr><td style="padding:4px 12px;"><b>Retorno ANO:</b></td><td>{_fmt_p(ret_ytd)} &nbsp;({_fmt_c(pct_ytd)} do CDI)</td></tr>\n'
        novo += '<tr><td style="padding:4px 12px;"><b>PL Total:</b></td><td>{_fmt_R(dados.get(\'pl_total\'))}</td></tr>\n'
        novo += '</table>\n\n'
        novo += '<p style="margin-top:20px;padding:12px;background:#eef;border-radius:6px;">\n'
        novo += '<b>Relat\u00f3rio completo com gr\u00e1ficos:</b><br>\n'
        novo += '<a href="file:///{link_html}" style="color:#1a3a6c;font-weight:bold;">Abrir {Path(link_html).name}</a><br>\n'
        novo += '<span style="font-size:11px;color:#666;">Ou copie o caminho: <code>{link_html}</code></span>\n'
        novo += '</p>\n\n'
        novo += '<p style="margin-top:32px;color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:12px;">\n'
        novo += 'Gerado em {datetime.now().strftime(\'%d/%m/%Y %H:%M\')}.\n'
        novo += '</p>\n'
        novo += '</body></html>"""\n\n\n'
        src = src[:i1] + novo + src[i2+1:]
        print("[diario] D6: _resumo_email_texto reescrita"); n += 1

    f.write_text(src, encoding="utf-8")
    print(f"[diario] {n} mudancas\n")


def patch_mensal():
    f = Path("enviar_email_mensal.py")
    if not f.exists():
        print("[erro] enviar_email_mensal.py nao encontrado"); return
    _backup(f)
    src = f.read_text(encoding="utf-8")
    n = 0

    old = 'assunto = f"Dash Risco - Consolidado {label_mes} - Retorno {ret_mes*100:+.2f}% ({(pct_cdi_mes*100 if pct_cdi_mes else 0):.0f}% CDI)"'
    if old in src:
        src = src.replace(old, 'assunto = f"Book Juros - Consolidado {label_mes}"')
        print("[mensal] M1: assunto"); n += 1

    if '<h1>Dash Risco - Consolidado {label_mes}</h1>' in src:
        src = src.replace('<h1>Dash Risco - Consolidado {label_mes}</h1>', '<h1>Book Juros - Consolidado {label_mes}</h1>')
        print("[mensal] M2: H1"); n += 1

    old = "Dash Risco \u2014 Consolidado {dados['label']}"
    new = "Book Juros - Consolidado {dados['label']}"
    if old in src:
        src = src.replace(old, new); print("[mensal] M3: body title"); n += 1

    old = 'return f"""<html><body style="font-family:Segoe UI,Arial,sans-serif;color:#222;font-size:14px;">'
    new = 'return f"""<!DOCTYPE html>\n<html><head><meta charset="utf-8"></head>\n<body style="font-family:Segoe UI,Arial,sans-serif;color:#222;font-size:14px;">'
    if old in src:
        src = src.replace(old, new); print("[mensal] M4: charset"); n += 1

    correcoes = [
        ('titulo=f"Atribuicao de Performance {label_mes}"', 'titulo=f"Atribui\u00e7\u00e3o de Performance {label_mes}"'),
        ('titulo="Cota vs CDI - Historico total (retorno acumulado)"', 'titulo="Cota vs CDI - Hist\u00f3rico total (retorno acumulado)"'),
        ('titulo="Drawdown historico"', 'titulo="Drawdown hist\u00f3rico"'),
        ('return f"Dash Risco - {label_mes} - sem dados"', 'return f"Book Juros - Consolidado {label_mes} - sem dados"'),
        ('<b>Relatorio completo com atribuicao, rentabilidade historica e indices:</b>',
         '<b>Relat\u00f3rio completo com atribui\u00e7\u00e3o, rentabilidade hist\u00f3rica e \u00edndices:</b>'),
    ]
    for old, new in correcoes:
        if old in src and old != new:
            src = src.replace(old, new); n += 1
    print("[mensal] M5: correcoes gramaticais aplicadas")

    old_dd = 'dd_h = [s.get("dd_atual") for s in snapshots_all]\n    chart_dd = _svg_line_chart(datas_h, {"Drawdown": dd_h},\n        largura=800, altura=180, titulo="Drawdown hist\u00f3rico",\n        cores=["#dc3545"], y_fmt="pct", y_zero_center=True)'
    new_dd = old_dd + '\n\n    consumo_hist_h = [s.get("consumo_hist_pct") for s in snapshots_all]\n    consumo_ewma_h = [s.get("consumo_ewma_pct") for s in snapshots_all]\n    chart_var_hist = _svg_line_chart(datas_h,\n        {"HIST 3y": consumo_hist_h, "EWMA": consumo_ewma_h},\n        largura=800, altura=180, titulo="Utiliza\u00e7\u00e3o do risco (VaR / limite)",\n        cores=["#1565c0", "#fd7e14"], y_fmt="pct")\n\n    all_ret_hist = [float(s.get("retorno_dtd") or 0) for s in snapshots_all]\n    n_hist = len(all_ret_hist)\n    if n_hist >= 2:\n        m_h = sum(all_ret_hist) / n_hist\n        v_h = sum((r - m_h) ** 2 for r in all_ret_hist) / (n_hist - 1)\n        vol_total_hist = math.sqrt(v_h) * math.sqrt(252)\n    else:\n        vol_total_hist = None\n    all_cdi_hist = [float(s.get("cdi_dtd") or 0) for s in snapshots_all]\n    excesso = [r - c for r, c in zip(all_ret_hist, all_cdi_hist)]\n    if excesso and vol_total_hist and vol_total_hist > 0:\n        mu_ex = sum(excesso) / len(excesso)\n        sig_ex_val = _std(excesso)\n        sharpe_hist = (mu_ex / sig_ex_val * math.sqrt(252)) if sig_ex_val > 0 else None\n    else:\n        sharpe_hist = None\n    var_95_bps_h = -sorted(all_ret_hist)[int(len(all_ret_hist) * 0.05)] * 10_000 if all_ret_hist else 0\n    cotas_h_valid = [c for c in cotas_h_raw if c is not None]\n    dd_max_hist = 0.0\n    if len(cotas_h_valid) >= 2:\n        peak_h = cotas_h_valid[0]\n        for c in cotas_h_valid:\n            if c > peak_h: peak_h = c\n            dd = c / peak_h - 1\n            if dd < dd_max_hist: dd_max_hist = dd'
    if old_dd in src and 'chart_var_hist' not in src:
        src = src.replace(old_dd, new_dd); print("[mensal] M6a: calculos VaR hist + metricas cota"); n += 1

    old_rodape = '<div class="chart-container">\n  {chart_dd}\n</div>\n\n<p style="margin-top:32px;color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:12px;">\nConsolidado gerado em {datetime.now().strftime(\'%d/%m/%Y %H:%M\')}.\nPL m\u00e9dio do m\u00eas: {_fmt_reais(pl_medio)}.\n</p>\n</body></html>"""'
    new_rodape = '<div class="chart-container">\n  {chart_dd}\n</div>\n\n<div class="chart-container">\n  {chart_var_hist}\n</div>\n\n<h3>M\u00e9tricas hist\u00f3ricas da cota</h3>\n<div class="metric-grid">\n  <div class="metric" style="width:25%;">\n    <div class="metric-label">Sharpe</div>\n    <div class="metric-value" style="color:{_cor_ret(sharpe_hist) if sharpe_hist else \'#888\'}">{f\'{sharpe_hist:.2f}\' if sharpe_hist is not None else \'-\'}</div>\n    <div class="metric-value-secondary" style="font-size:11px;color:#666;font-weight:normal;">(ret-CDI) / vol</div>\n  </div>\n  <div class="metric" style="width:25%;">\n    <div class="metric-label">Vol realizada</div>\n    <div class="metric-value">{(vol_total_hist*100 if vol_total_hist else 0):.2f}%</div>\n  </div>\n  <div class="metric" style="width:25%;">\n    <div class="metric-label">VaR (95%)</div>\n    <div class="metric-value">{var_95_bps_h:.2f} bps</div>\n  </div>\n  <div class="metric" style="width:25%;">\n    <div class="metric-label">M\u00e1ximo Drawdown</div>\n    <div class="metric-value" style="color:#dc3545">{dd_max_hist*100:.2f}%</div>\n  </div>\n</div>\n\n<p style="margin-top:32px;color:#888;font-size:11px;border-top:1px solid #ddd;padding-top:12px;">\nConsolidado gerado em {datetime.now().strftime(\'%d/%m/%Y %H:%M\')}.\nPL m\u00e9dio do m\u00eas: {_fmt_reais(pl_medio)}.\n</p>\n</body></html>"""'
    if old_rodape in src and 'chart_var_hist}' not in src[:src.find(old_rodape)] if old_rodape in src else False:
        src = src.replace(old_rodape, new_rodape); print("[mensal] M6b: HTML anexo com VaR hist + metricas"); n += 1
    elif old_rodape in src:
        src = src.replace(old_rodape, new_rodape); print("[mensal] M6b: HTML anexo com VaR hist + metricas"); n += 1

    f.write_text(src, encoding="utf-8")
    print(f"[mensal] {n} mudancas\n")


if __name__ == "__main__":
    print("=== Patch emails Book Juros ===\n")
    patch_diario()
    patch_mensal()
    print("=== Concluido. Teste com --dry-run ===")
