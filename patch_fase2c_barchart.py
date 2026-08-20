"""patch_fase2c_barchart.py - substitui treemap por bar chart em bps, full width"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_bar_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Localiza o bloco atual do treemap (Composição do DV01) e substitui por bar chart em full width
old = '''        with COL1:

        # ===================== DV01 por ATIVO x CLASSE (TREEMAP) =====================
            st.subheader("Composição do DV01")

            dv01_asset_rs_dict  = risco.get("DV01 por ativo (bps)", {}) or {}
            if not dv01_asset_rs_dict:
                st.info("DV01 por ativo indisponível para este portfólio.")
            else:
                import plotly.express as _px
                import pandas as _pd
                def _mc(a):
                    au = str(a).upper()
                    if au.startswith("DI"): return "Juros Nominais BR"
                    if au.startswith(("DAP","NTNB")): return "Juros Reais BR"
                    if "TREASURY" in au: return "Juros US"
                    if au.startswith("WDO"): return "Moeda"
                    return "Outros"
                _cores_cls = {"Juros Nominais BR":"#1565c0","Juros Reais BR":"#dc3545","Juros US":"#0d47a1","Moeda":"#e57373","Outros":"#7f7f7f"}
                _df_tm = _pd.DataFrame([
                    {"Ativo": a, "Classe": _mc(a), "DV01_abs": abs(float(v)), "DV01_signed": float(v), "Sinal": "Long" if float(v)>=0 else "Short"}
                    for a, v in dv01_asset_rs_dict.items() if abs(float(v)) > 1e-9
                ])
                if _df_tm.empty:
                    st.info("Todos os DV01 zerados.")
                else:
                    _fig_tm = _px.treemap(
                        _df_tm, path=["Classe","Ativo"], values="DV01_abs",
                        color="Classe", color_discrete_map=_cores_cls,
                        custom_data=["DV01_signed","Sinal"]
                    )
                    _fig_tm.update_traces(
                        hovertemplate="<b>%{label}</b><br>DV01: R$ %{customdata[0]:,.0f}<br>%{customdata[1]}<extra></extra>",
                        textinfo="label+value", texttemplate="<b>%{label}</b><br>R$ %{value:,.0f}"
                    )
                    _fig_tm.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=380)
                    st.plotly_chart(_fig_tm, use_container_width=True)'''

novo = '''        # ===================== DV01 por ATIVO (BAR CHART, bps, full width, colorido por classe) =====================
        st.subheader("Composição do DV01 por ativo (bps)")

        dv01_asset_bps_dict = risco.get("DV01 por ativo (bps)", {}) or {}
        if not dv01_asset_bps_dict:
            st.info("DV01 por ativo indisponível para este portfólio.")
        else:
            import plotly.graph_objects as _pgo
            import pandas as _pd
            def _mc(a):
                au = str(a).upper()
                if au.startswith("DI"): return "Juros Nominais BR"
                if au.startswith(("DAP","NTNB")): return "Juros Reais BR"
                if "TREASURY" in au: return "Juros US"
                if au.startswith("WDO"): return "Moeda"
                return "Outros"
            _cores_cls = {"Juros Nominais BR":"#1565c0","Juros Reais BR":"#dc3545","Juros US":"#0d47a1","Moeda":"#e57373","Outros":"#7f7f7f"}
            _df_bar = _pd.DataFrame([
                {"Ativo": a, "Classe": _mc(a), "DV01_bps": float(v)}
                for a, v in dv01_asset_bps_dict.items() if abs(float(v)) > 1e-9
            ]).sort_values("DV01_bps", ascending=True)
            if _df_bar.empty:
                st.info("Todos os DV01 zerados.")
            else:
                _fig_bar = _pgo.Figure()
                for _cl in _df_bar["Classe"].unique():
                    _sub = _df_bar[_df_bar["Classe"] == _cl]
                    _fig_bar.add_trace(_pgo.Bar(
                        x=_sub["DV01_bps"], y=_sub["Ativo"], orientation="h",
                        name=_cl, marker_color=_cores_cls.get(_cl, "#7f7f7f"),
                        text=[f"{v:+.2f} bps" for v in _sub["DV01_bps"]],
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>DV01: %{x:.2f} bps<br>Classe: " + _cl + "<extra></extra>",
                    ))
                _fig_bar.update_layout(
                    xaxis=dict(title="DV01 (bps sobre PL Risco)", zeroline=True, zerolinecolor="#666", zerolinewidth=1.5),
                    yaxis=dict(title=""),
                    barmode="relative", height=max(220, 40*len(_df_bar) + 100),
                    margin=dict(l=80, r=30, t=30, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    showlegend=True,
                )
                st.plotly_chart(_fig_bar, use_container_width=True)'''

if old in s:
    s = s.replace(old, novo)
    f.write_text(s, encoding="utf-8")
    print("[ok] Treemap substituido por bar chart em bps (full width)")
else:
    print("[warn] bloco treemap nao encontrado")
