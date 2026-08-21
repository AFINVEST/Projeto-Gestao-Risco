"""patch_stats_gotable.py - reescreve tabela Estatisticas mensais com go.Table (estilo uniforme)"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_stgotbl_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Substitui o bloco HTML markdown pela versao go.Table
old = '''                _cor_min = "#28a745" if _minr >= 0 else "#dc3545"
                st.markdown("### Estatísticas mensais")
                _html = f"""'''

# Precisamos capturar TODO o bloco ate o `st.markdown(_html, unsafe_allow_html=True)`
i1 = s.find(old)
if i1 == -1:
    print("[erro] bloco stats mensais nao encontrado")
else:
    fim_marker = 'st.markdown(_html, unsafe_allow_html=True)'
    i2 = s.find(fim_marker, i1)
    if i2 == -1:
        print("[erro] fim do bloco nao encontrado")
    else:
        # tambem inclui o "st.markdown('')" da linha seguinte (espacador)
        i2_end = s.find("\n", i2) + 1
        # possivelmente tem 'st.markdown("")' logo depois
        proxima = s[i2_end:i2_end+50]
        if 'st.markdown("")' in proxima:
            i2_end = s.find("\n", i2_end + proxima.index('st.markdown("")')) + 1

        bloco_antigo = s[i1:i2_end]

        novo_bloco = '''                _cor_min = "#28a745" if _minr >= 0 else "#dc3545"
                st.markdown("### Estatísticas mensais")

                # go.Table no mesmo estilo de Indices Historicos
                import plotly.graph_objects as _gotbl
                _hdr_fill = "#0A2240"
                _hdr_font = "#FFFFFF"
                _cell_fill_gray = "#F7F8FA"
                _cell_h = 30
                _tbl_header = ["Meses positivos","Meses negativos","Maior retorno mensal","Menor retorno mensal","Meses acima do CDI","Meses abaixo do CDI"]
                _tbl_row = [
                    f"{_npos}/{_n} ({_npos/_n*100:.0f}%)",
                    f"{_nneg}/{_n} ({_nneg/_n*100:.0f}%)",
                    f"{_maxr*100:+.2f}%",
                    f"{_minr*100:+.2f}%",
                    f"{_nac}/{_n} ({_nac/_n*100:.0f}%)",
                    f"{_nab}/{_n} ({_nab/_n*100:.0f}%)",
                ]
                _cell_font_colors = ["#28a745","#dc3545","#28a745",_cor_min,"#28a745","#dc3545"]
                _fig_stats = _gotbl.Figure(data=[_gotbl.Table(
                    header=dict(values=_tbl_header, fill_color=_hdr_fill, font=dict(color=_hdr_font, size=12), align="center", height=28),
                    cells=dict(values=[[v] for v in _tbl_row], align="center", height=_cell_h,
                               fill_color=[[_cell_fill_gray]]*6,
                               font=dict(color=[[c] for c in _cell_font_colors], size=12))
                )])
                _fig_stats.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=28+_cell_h+15, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(_fig_stats, use_container_width=True)

'''
        s = s.replace(bloco_antigo, novo_bloco)
        f.write_text(s, encoding="utf-8")
        print(f"[ok] tabela Estatisticas mensais reescrita com go.Table (mesma formatacao de Indices Historicos)")
