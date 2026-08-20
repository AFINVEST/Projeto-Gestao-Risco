"""patch_fase2d_dashboard_covar.py - novo display CoVaR + backup antigo"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_covar_dash_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Envelopa o CoVaR antigo em if False (backup) e insere o novo antes
old = '''        with colll1:

            st.subheader("CoVaR por classe")
            covar_bps_dict = risco.get("CoVaR por ativo (bps)", {}) or {}'''

novo = '''        with colll1:

            # ============ NOVO CoVaR: usa risco_carteira_core (HIST 3y em PU) ============
            st.subheader("CoVaR por classe")
            try:
                from risco_carteira_core import calcular_covar_completo
                from cota_portfolio_core import load_basefundos as _load_bf
                import pandas as _pd_c
                _covar_res = calcular_covar_completo(
                    data_ref=_pd_c.Timestamp.today().normalize(),
                    basefundos=_load_bf(),
                    janela_dias=756,
                )
                if "erro" in _covar_res:
                    st.info(f"CoVaR novo indisponivel: {_covar_res.get('erro')}")
                else:
                    import plotly.graph_objects as _pgo_c
                    _cls_R = _covar_res["covar_por_classe_R"]
                    _cls_pct = _covar_res["covar_por_classe_pct"]
                    _cores_cls = {
                        "Juros Nominais BR":"#1565c0","Juros Reais BR":"#dc3545",
                        "Juros US":"#0d47a1","Moeda":"#e57373","Outros":"#7f7f7f"
                    }
                    _labels = list(_cls_R.keys())
                    _vals   = list(_cls_R.values())
                    _cores  = [_cores_cls.get(l, "#999") for l in _labels]

                    _c_pie, _c_bar = st.columns([1, 1.5])
                    with _c_pie:
                        _fig_pie = _pgo_c.Figure(_pgo_c.Pie(
                            labels=_labels, values=_vals, hole=0.55,
                            marker=dict(colors=_cores),
                            textinfo="label+percent", sort=False,
                        ))
                        _fig_pie.update_layout(
                            title=dict(text="Composição do CoVaR por classe", x=0.5, xanchor="center", font=dict(size=14, color="#1a3a6c")),
                            margin=dict(l=10, r=10, t=40, b=10), height=320, showlegend=False,
                        )
                        st.plotly_chart(_fig_pie, use_container_width=True)
                    with _c_bar:
                        # Bar chart por ativo dentro de cada classe
                        _covar_ativo = _covar_res["covar_por_ativo_R"]
                        def _mc2(a):
                            au = str(a).upper()
                            if au.startswith("DI"): return "Juros Nominais BR"
                            if au.startswith(("DAP","NTNB")): return "Juros Reais BR"
                            if "TREASURY" in au: return "Juros US"
                            if au.startswith("WDO"): return "Moeda"
                            return "Outros"
                        _df_at = _pd_c.DataFrame([
                            {"Ativo": a, "Classe": _mc2(a), "CoVaR_R": v}
                            for a, v in _covar_ativo.items() if abs(v) > 1e-6
                        ]).sort_values("CoVaR_R", ascending=True)
                        _fig_bar = _pgo_c.Figure()
                        for _cl in _df_at["Classe"].unique():
                            _sub = _df_at[_df_at["Classe"] == _cl]
                            _fig_bar.add_trace(_pgo_c.Bar(
                                x=_sub["CoVaR_R"], y=_sub["Ativo"], orientation="h",
                                name=_cl, marker_color=_cores_cls.get(_cl, "#999"),
                                text=[f"R$ {v:,.0f}" for v in _sub["CoVaR_R"]],
                                textposition="outside",
                            ))
                        _fig_bar.update_layout(
                            title=dict(text="CoVaR por ativo (R$)", x=0.5, xanchor="center", font=dict(size=14, color="#1a3a6c")),
                            xaxis=dict(title="R$"), yaxis=dict(title=""),
                            barmode="relative", height=max(240, 30*len(_df_at)+120),
                            margin=dict(l=80, r=50, t=40, b=40), showlegend=False,
                        )
                        st.plotly_chart(_fig_bar, use_container_width=True)

                    st.caption(f"VaR estimado (soma): R$ {_covar_res['var_estimado_R']:,.0f} — cauda: {_covar_res['n_scenarios_tail']} scenarios (HIST 3y, alpha 5%)")
            except Exception as _ec:
                st.error(f"Erro CoVaR: {_ec}")

            # ============ BACKUP: CoVaR antigo (BBG 5y planilha) — DESATIVADO ============
            if False:
                st.subheader("CoVaR por classe (legado BBG 5y)")
                covar_bps_dict = risco.get("CoVaR por ativo (bps)", {}) or {}'''

if old in s:
    s = s.replace(old, novo)
    f.write_text(s, encoding="utf-8")
    print("[ok] Novo CoVaR display + backup do antigo (if False)")
else:
    print("[warn] bloco nao encontrado")
