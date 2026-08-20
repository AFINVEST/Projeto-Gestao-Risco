"""patch_stats_separado.py - reverte stats na tabela de indices + tabela nova"""
from pathlib import Path
import shutil, datetime as dt

f = Path("enviar_email_mensal.py")
shutil.copy2(f, f"{f}.bak_statsep_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# --- Fix 1: reverte as 6 linhas extras na tabela de indices ---
# Bloco atual: sharpe row + 6 rows novas (positivos, negativos, maior, menor, acima, abaixo)
# Volta pra: sharpe row + </table>
old = '''    # Sharpe
    html += ''' + chr(39) + '<tr><td class="rowlabel">ÍNDICE DE SHARPE</td>' + chr(39) + '''
    for j in janelas:
        d = ind_hist.get(j)
        if d is None or d.get("sharpe") is None:
            html += "<td>—</td>"
        else:
            cor = _cor_ret(d["sharpe"])
            html += f''' + chr(39) + '<td style="color:{cor};font-weight:bold">{d["sharpe"]:.2f}</td>' + chr(39) + '''
    html += "</tr>"

    # Meses positivos'''
if old in s:
    # Encontra o proximo </table> e substitui todo o bloco antes dele
    i_start = s.find(old)
    # localiza o </table></tr> das rows extras (fim da tabela ampliada)
    i_end_table = s.find('MESES ABAIXO CDI', i_start)
    if i_end_table > 0:
        i_close = s.find('</table>', i_end_table)
        if i_close > 0:
            # Substitui do inicio de "# Meses positivos" ate depois de </table>
            i_extras = s.find('# Meses positivos', i_start)
            fim = i_close + len('</table>')
            # Reconstroi: mantem so a parte de sharpe ate </tr>, depois </table>
            novo = s[:i_extras] + '</table>' + s[fim:]
            s = novo
            print("[ok] linhas extras removidas da tabela de indices")

# --- Fix 2: adiciona NOVA tabela apos a de indices historicos ---
# Insere antes de "# ─── Contexto historico (charts)" (secao 5)
marker = "    # ─── Contexto historico (charts) ────────────"
nova_tabela = '''    # ─── ESTATISTICAS MENSAIS (todo o periodo) ────────────
    from collections import defaultdict as _dd
    _g = _dd(lambda: {"rets": [], "cdis": []})
    for _s in snapshots_all:
        _d = datetime.fromisoformat(_s["Data"])
        _g[(_d.year, _d.month)]["rets"].append(float(_s.get("retorno_dtd") or 0))
        _g[(_d.year, _d.month)]["cdis"].append(float(_s.get("cdi_dtd") or 0))
    _meses = [{"ret": _acumulado(dat["rets"]), "cdi": _acumulado(dat["cdis"])} for _, dat in _g.items()]
    if _meses:
        _n = len(_meses)
        _npos = sum(1 for m in _meses if m["ret"] > 0)
        _nneg = sum(1 for m in _meses if m["ret"] < 0)
        _maxr = max(m["ret"] for m in _meses)
        _minr = min(m["ret"] for m in _meses)
        _nac  = sum(1 for m in _meses if m["ret"] > m["cdi"])
        _nab  = sum(1 for m in _meses if m["ret"] < m["cdi"])
        html += f\"\"\"
<h3>Estatísticas mensais (histórico completo, {_n} meses)</h3>
<table>
<tr>
  <th>Meses positivos</th><th>Meses negativos</th>
  <th>Maior retorno mensal</th><th>Menor retorno mensal</th>
  <th>Meses acima do CDI</th><th>Meses abaixo do CDI</th>
</tr>
<tr>
  <td style="color:#28a745;font-weight:bold">{_npos}/{_n} ({_npos/_n*100:.0f}%)</td>
  <td style="color:#dc3545;font-weight:bold">{_nneg}/{_n} ({_nneg/_n*100:.0f}%)</td>
  <td style="color:#28a745;font-weight:bold">{_maxr*100:+.2f}%</td>
  <td style="color:{_cor_ret(_minr)};font-weight:bold">{_minr*100:+.2f}%</td>
  <td style="color:#28a745;font-weight:bold">{_nac}/{_n} ({_nac/_n*100:.0f}%)</td>
  <td style="color:#dc3545;font-weight:bold">{_nab}/{_n} ({_nab/_n*100:.0f}%)</td>
</tr>
</table>
\"\"\"

    '''
if marker in s and 'Estatísticas mensais' not in s:
    s = s.replace(marker, nova_tabela + marker)
    print("[ok] tabela de estatisticas mensais adicionada")

# --- Fix 3: reverte _metric expandido (nao precisa mais dos stats por janela) ---
old_metric_new = '''    def _stats_mensais_local(subs):
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

    def _metric(subs):'''
old_metric_clean = '    def _metric(subs):'
if old_metric_new in s:
    s = s.replace(old_metric_new, old_metric_clean)
    print("[ok] _stats_mensais_local removida de _indices_historicos")

# Remove o "stats = _stats_mensais_local(subs)" e "**stats" do return
s = s.replace(
    '        stats = _stats_mensais_local(subs)\n        return {"ret": ret, "cdi": cdi, "pct_cdi": pct, "vol": vol, "sharpe": sharpe, **stats}',
    '        return {"ret": ret, "cdi": cdi, "pct_cdi": pct, "vol": vol, "sharpe": sharpe}'
)

f.write_text(s, encoding="utf-8")
print("[done] enviar_email_mensal.py atualizado")
