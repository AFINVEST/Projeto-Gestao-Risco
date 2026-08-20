"""patch_stats_mensais.py - adiciona estatisticas mensais a tabela de indices historicos"""
from __future__ import annotations
from pathlib import Path
import shutil, datetime as dt

f = Path("enviar_email_mensal.py")
shutil.copy2(f, f"{f}.bak_statsmensais_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# --- Fix 1: adiciona funcao _stats_mensais + inclui no _indices_historicos ---
old_metric = '''    def _metric(subs):
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
        return {"ret": ret, "cdi": cdi, "pct_cdi": pct, "vol": vol, "sharpe": sharpe}'''

new_metric = '''    def _stats_mensais_local(subs):
        from collections import defaultdict
        g = defaultdict(lambda: {"rets": [], "cdis": []})
        for s in subs:
            d = datetime.fromisoformat(s["Data"])
            g[(d.year, d.month)]["rets"].append(float(s.get("retorno_dtd") or 0))
            g[(d.year, d.month)]["cdis"].append(float(s.get("cdi_dtd") or 0))
        meses = []
        for _, dat in g.items():
            meses.append({"ret": _acumulado(dat["rets"]), "cdi": _acumulado(dat["cdis"])})
        if not meses:
            return {"n_meses": 0, "n_pos": 0, "n_neg": 0, "max_ret": 0, "min_ret": 0, "n_acima": 0, "n_abaixo": 0}
        return {
            "n_meses":      len(meses),
            "n_pos":        sum(1 for m in meses if m["ret"] > 0),
            "n_neg":        sum(1 for m in meses if m["ret"] < 0),
            "max_ret":      max(m["ret"] for m in meses),
            "min_ret":      min(m["ret"] for m in meses),
            "n_acima":      sum(1 for m in meses if m["ret"] > m["cdi"]),
            "n_abaixo":     sum(1 for m in meses if m["ret"] < m["cdi"]),
        }

    def _metric(subs):
        if not subs: return None
        rets = [float(s.get("retorno_dtd") or 0) for s in subs]
        cdis = [float(s.get("cdi_dtd") or 0) for s in subs]
        ret = _acumulado(rets); cdi = _acumulado(cdis)
        pct = (ret / cdi) if cdi else None
        vol = _std(rets) * math.sqrt(252)
        excess = [r - c for r, c in zip(rets, cdis)]
        mu = sum(excess) / len(excess) if excess else 0
        sig = _std(excess)
        sharpe = (mu / sig * math.sqrt(252)) if sig > 0 else None
        stats = _stats_mensais_local(subs)
        return {"ret": ret, "cdi": cdi, "pct_cdi": pct, "vol": vol, "sharpe": sharpe, **stats}'''

if old_metric in s:
    s = s.replace(old_metric, new_metric)
    print("[ok] _metric expandido com stats mensais")

# --- Fix 2: adiciona novas linhas na tabela HTML de indices historicos ---
old_sharpe_row = '''    # Sharpe
    html += ''' + chr(39) + '<tr><td class="rowlabel">\u00cdNDICE DE SHARPE</td>' + chr(39) + '''
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("sharpe") is None:
            html += "<td>\u2014</td>"
        else:
            cor = _cor_ret(d["sharpe"])
            html += f''' + chr(39) + '<td style="color:{cor};font-weight:bold">{d["sharpe"]:.2f}</td>' + chr(39) + '''
    html += "</tr></table>"'''

new_sharpe_row = '''    # Sharpe
    html += ''' + chr(39) + '<tr><td class="rowlabel">\u00cdNDICE DE SHARPE</td>' + chr(39) + '''
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("sharpe") is None:
            html += "<td>\u2014</td>"
        else:
            cor = _cor_ret(d["sharpe"])
            html += f''' + chr(39) + '<td style="color:{cor};font-weight:bold">{d["sharpe"]:.2f}</td>' + chr(39) + '''
    html += "</tr>"

    # Meses positivos
    html += ''' + chr(39) + '<tr><td class="rowlabel">MESES POSITIVOS</td>' + chr(39) + '''
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("n_meses", 0) == 0:
            html += "<td>\u2014</td>"
        else:
            html += f''' + chr(39) + '<td style="color:#28a745;font-weight:bold">{d["n_pos"]}/{d["n_meses"]}</td>' + chr(39) + '''
    html += "</tr>"

    # Meses negativos
    html += ''' + chr(39) + '<tr><td class="rowlabel">MESES NEGATIVOS</td>' + chr(39) + '''
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("n_meses", 0) == 0:
            html += "<td>\u2014</td>"
        else:
            html += f''' + chr(39) + '<td style="color:#dc3545;font-weight:bold">{d["n_neg"]}/{d["n_meses"]}</td>' + chr(39) + '''
    html += "</tr>"

    # Maior retorno mensal
    html += ''' + chr(39) + '<tr><td class="rowlabel">MAIOR RET. MENSAL</td>' + chr(39) + '''
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("n_meses", 0) == 0:
            html += "<td>\u2014</td>"
        else:
            html += f''' + chr(39) + '<td style="color:#28a745;font-weight:bold">{d["max_ret"]*100:+.2f}%</td>' + chr(39) + '''
    html += "</tr>"

    # Menor retorno mensal
    html += ''' + chr(39) + '<tr><td class="rowlabel">MENOR RET. MENSAL</td>' + chr(39) + '''
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("n_meses", 0) == 0:
            html += "<td>\u2014</td>"
        else:
            cor = _cor_ret(d["min_ret"])
            html += f''' + chr(39) + '<td style="color:{cor};font-weight:bold">{d["min_ret"]*100:+.2f}%</td>' + chr(39) + '''
    html += "</tr>"

    # Meses acima do CDI
    html += ''' + chr(39) + '<tr><td class="rowlabel">MESES ACIMA CDI</td>' + chr(39) + '''
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("n_meses", 0) == 0:
            html += "<td>\u2014</td>"
        else:
            html += f''' + chr(39) + '<td style="color:#28a745;font-weight:bold">{d["n_acima"]}/{d["n_meses"]}</td>' + chr(39) + '''
    html += "</tr>"

    # Meses abaixo do CDI
    html += ''' + chr(39) + '<tr><td class="rowlabel">MESES ABAIXO CDI</td>' + chr(39) + '''
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("n_meses", 0) == 0:
            html += "<td>\u2014</td>"
        else:
            html += f''' + chr(39) + '<td style="color:#dc3545;font-weight:bold">{d["n_abaixo"]}/{d["n_meses"]}</td>' + chr(39) + '''
    html += "</tr></table>"'''

if old_sharpe_row in s:
    s = s.replace(old_sharpe_row, new_sharpe_row)
    print("[ok] 6 novas linhas adicionadas na tabela de indices")

f.write_text(s, encoding="utf-8")
print("[done] enviar_email_mensal.py atualizado")
