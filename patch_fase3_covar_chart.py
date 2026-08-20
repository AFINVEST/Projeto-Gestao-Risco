"""patch_fase3_covar_chart.py - adiciona chart historico CoVaR por classe"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_covarhist_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Insere APOS o caption "VaR estimado (soma)..." do bloco novo CoVaR
marker = 'st.caption(f"VaR estimado (soma): R$ {_covar_res[\'var_estimado_R\']:,.0f} — cauda: {_covar_res[\'n_scenarios_tail\']} scenarios (HIST 3y, alpha 5%)")'

novo_chart = marker + '''

                    # ============ CoVaR HISTORICO por classe (area empilhada) ============
                    st.markdown("### CoVaR histórico por estratégia")
                    _rows_h = []
                    _offset_h = 0
                    while True:
                        _rh = supabase.table("snapshot_diario").select(
                            "Data,covar_juros_nom_pct,covar_juros_real_pct,covar_moeda_pct,covar_juros_us_pct,covar_outros_pct"
                        ).order("Data").range(_offset_h, _offset_h+999).execute()
                        if not _rh.data: break
                        _rows_h.extend(_rh.data)
                        if len(_rh.data) < 1000: break
                        _offset_h += 1000

                    if _rows_h:
                        import plotly.graph_objects as _pgo_h
                        import pandas as _pd_h
                        _df_h = _pd_h.DataFrame(_rows_h)
                        _df_h["Data"] = _pd_h.to_datetime(_df_h["Data"])
                        _df_h = _df_h.sort_values("Data").set_index("Data")

                        _mapa_cor = {
                            "covar_juros_nom_pct":  ("Juros Nominais BR", "#1565c0"),
                            "covar_juros_real_pct": ("Juros Reais BR", "#dc3545"),
                            "covar_moeda_pct":      ("Moeda", "#e57373"),
                            "covar_juros_us_pct":   ("Juros US", "#0d47a1"),
                            "covar_outros_pct":     ("Outros", "#7f7f7f"),
                        }
                        _fig_h = _pgo_h.Figure()
                        for _col, (_nome, _cor) in _mapa_cor.items():
                            if _col in _df_h.columns:
                                _serie = _df_h[_col].fillna(0) * 100
                                if _serie.abs().sum() > 0:
                                    _fig_h.add_trace(_pgo_h.Scatter(
                                        x=_df_h.index, y=_serie,
                                        mode="lines", stackgroup="one",
                                        name=_nome, line=dict(width=0.5),
                                        fillcolor=_cor, hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
                                    ))
                        _fig_h.update_layout(
                            title=dict(text="CoVaR por estratégia (composição %)", x=0.5, xanchor="center", font=dict(size=14, color="#1a3a6c")),
                            hovermode="x unified",
                            margin=dict(l=20, r=20, t=40, b=20),
                            yaxis=dict(ticksuffix="%", title="Proporção do CoVaR"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=380,
                        )
                        st.plotly_chart(_fig_h, use_container_width=True)
                    else:
                        st.info("Sem histórico de CoVaR disponível no snapshot_diario.")'''

if marker in s:
    s = s.replace(marker, novo_chart)
    f.write_text(s, encoding="utf-8")
    print("[ok] Chart historico CoVaR por classe adicionado")
else:
    print("[warn] marker do caption VaR estimado nao encontrado")
