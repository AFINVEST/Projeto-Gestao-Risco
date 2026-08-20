"""patch_fase1_stats_mover.py - move tabela stats ANTES dos charts + formata horizontal"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_stats_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Remove bloco atual de stats (esta no fim)
inicio_marker = '            # === Tabela estatisticas mensais ==='
fim_marker = '                st.dataframe(pd.DataFrame([_st]).T.rename(columns={0: "Valor"}), use_container_width=True)'
i1 = s.find(inicio_marker)
i2 = s.find(fim_marker, i1)
if i1 == -1 or i2 == -1:
    print("[warn] bloco atual nao encontrado")
else:
    i2_end = s.find("\n", i2) + 1
    s = s[:i1] + s[i2_end:]
    print("[ok] bloco antigo de stats removido do fim")

# Novo bloco: HTML horizontal + insere ANTES do primeiro chart (_fig_vol)
novo_stats = '''            # === Estatisticas mensais (horizontal, ANTES dos charts) ===
            from collections import defaultdict as _dd
            _g = _dd(lambda: {"rets": [], "cdis": []})
            for _, _row in _df_snap.iterrows():
                _y, _m = _row.name.year, _row.name.month
                _g[(_y, _m)]["rets"].append(float(_row.get("retorno_dtd") or 0))
                _g[(_y, _m)]["cdis"].append(float(_row.get("cdi_dtd") or 0))
            def _acum(rs):
                acc = 1.0
                for r in rs: acc *= (1+r)
                return acc - 1
            _meses = [{"ret": _acum(v["rets"]), "cdi": _acum(v["cdis"])} for v in _g.values()]
            if _meses:
                _n = len(_meses)
                _npos = sum(1 for m in _meses if m["ret"]>0)
                _nneg = sum(1 for m in _meses if m["ret"]<0)
                _maxr = max(m["ret"] for m in _meses)
                _minr = min(m["ret"] for m in _meses)
                _nac  = sum(1 for m in _meses if m["ret"]>m["cdi"])
                _nab  = sum(1 for m in _meses if m["ret"]<m["cdi"])
                _cor_min = "#28a745" if _minr >= 0 else "#dc3545"
                st.markdown(f"### Estatisticas mensais (historico completo, {_n} meses)")
                _html = f"""
<table style="width:100%;border-collapse:collapse;text-align:center;font-size:14px;">
<thead>
<tr style="background:#1a3a6c;color:white;">
<th style="padding:8px 10px;border:1px solid #ddd;">Meses positivos</th>
<th style="padding:8px 10px;border:1px solid #ddd;">Meses negativos</th>
<th style="padding:8px 10px;border:1px solid #ddd;">Maior retorno mensal</th>
<th style="padding:8px 10px;border:1px solid #ddd;">Menor retorno mensal</th>
<th style="padding:8px 10px;border:1px solid #ddd;">Meses acima do CDI</th>
<th style="padding:8px 10px;border:1px solid #ddd;">Meses abaixo do CDI</th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding:10px;border:1px solid #ddd;color:#28a745;font-weight:bold;">{_npos}/{_n} ({_npos/_n*100:.0f}%)</td>
<td style="padding:10px;border:1px solid #ddd;color:#dc3545;font-weight:bold;">{_nneg}/{_n} ({_nneg/_n*100:.0f}%)</td>
<td style="padding:10px;border:1px solid #ddd;color:#28a745;font-weight:bold;">{_maxr*100:+.2f}%</td>
<td style="padding:10px;border:1px solid #ddd;color:{_cor_min};font-weight:bold;">{_minr*100:+.2f}%</td>
<td style="padding:10px;border:1px solid #ddd;color:#28a745;font-weight:bold;">{_nac}/{_n} ({_nac/_n*100:.0f}%)</td>
<td style="padding:10px;border:1px solid #ddd;color:#dc3545;font-weight:bold;">{_nab}/{_n} ({_nab/_n*100:.0f}%)</td>
</tr>
</tbody>
</table>
"""
                st.markdown(_html, unsafe_allow_html=True)
                st.markdown("")   # espaco

'''

# Insere o novo bloco ANTES do "# === Vol 20d anualizada ==="
marker_insercao = '            # === Vol 20d anualizada ==='
if marker_insercao in s:
    s = s.replace(marker_insercao, novo_stats + marker_insercao)
    print("[ok] stats mensais inseridas ANTES dos charts")
else:
    print("[warn] marker Vol 20d nao encontrado")

f.write_text(s, encoding="utf-8")
print("[done] app4.py atualizado")
