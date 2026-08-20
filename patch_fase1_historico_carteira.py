"""patch_fase1_historico_carteira.py - Fix typo + charts snapshot + stats mensais"""
from pathlib import Path
import shutil, datetime as dt, re

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_fase1_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
n = 0

# --- Fix 1: typo Histortico -> Historico ---
old1 = 'aba_cart,tab_orcamento = st.tabs(["Histortico Carteira", "Portfolio Atual"])'
new1 = 'aba_cart,tab_orcamento = st.tabs(["Historico Carteira", "Portfolio Atual"])'
if old1 in s:
    s = s.replace(old1, new1); n += 1
    print("[ok] typo Histortico -> Historico")

# --- Fix 2: substitui bloco Resumo de Volatilidade (linhas ~8433-8500) ---
# Localiza pelo marcador "st.subheader(\"Resumo de Volatilidade\")" e o proximo "with tab_orcamento" ou similar
inicio_marker = '        st.subheader("Resumo de Volatilidade")'
# Fim: procurar por "st.plotly_chart(fig_vol, use_container_width=True)" (linha ~8497)
fim_marker = 'st.plotly_chart(fig_vol, use_container_width=True)'

i1 = s.find(inicio_marker)
i2 = s.find(fim_marker, i1)
if i1 == -1 or i2 == -1:
    print("[warn] marcadores Resumo de Volatilidade nao encontrados")
else:
    # localiza fim da linha do fim_marker
    i2_end = s.find("\n", i2) + 1
    bloco_antigo = s[i1:i2_end]
    novo_bloco = '''        st.subheader("Metricas historicas de risco (snapshot)")

        # Carrega serie de snapshot_diario diretamente do Supabase
        _snap_rows = []
        _offset = 0
        while True:
            _r = supabase.table("snapshot_diario").select(
                "Data,retorno_dtd,cdi_dtd,vol_20d,dd_atual,consumo_hist_pct,consumo_ewma_pct"
            ).order("Data").range(_offset, _offset+999).execute()
            if not _r.data: break
            _snap_rows.extend(_r.data)
            if len(_r.data) < 1000: break
            _offset += 1000

        if not _snap_rows:
            st.info("Sem dados de snapshot_diario para plotar.")
        else:
            _df_snap = pd.DataFrame(_snap_rows)
            _df_snap["Data"] = pd.to_datetime(_df_snap["Data"])
            _df_snap = _df_snap.sort_values("Data").set_index("Data")

            import plotly.graph_objects as _go

            # === Vol 20d anualizada ===
            _fig_vol = _go.Figure()
            _fig_vol.add_trace(_go.Scatter(
                x=_df_snap.index, y=_df_snap["vol_20d"]*100,
                mode="lines", name="Vol 20d anualizada",
                line=dict(color="#c62828", width=2)
            ))
            _fig_vol.update_layout(
                title="Volatilidade 20d anualizada (%)",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(_fig_vol, use_container_width=True)

            # === Drawdown historico ===
            _fig_dd = _go.Figure()
            _fig_dd.add_trace(_go.Scatter(
                x=_df_snap.index, y=_df_snap["dd_atual"]*100,
                mode="lines", name="Drawdown",
                line=dict(color="#dc3545", width=2),
                fill="tozeroy", fillcolor="rgba(220,53,69,0.15)"
            ))
            _fig_dd.update_layout(
                title="Drawdown historico (%)",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(ticksuffix="%"),
                shapes=[dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=0,
                             line=dict(width=1, dash="dot", color="#888"))]
            )
            st.plotly_chart(_fig_dd, use_container_width=True)

            # === Utilizacao do risco (VaR / limite) ===
            _fig_var = _go.Figure()
            _fig_var.add_trace(_go.Scatter(
                x=_df_snap.index, y=_df_snap["consumo_hist_pct"]*100,
                mode="lines", name="HIST 3y",
                line=dict(color="#1565c0", width=2)
            ))
            _fig_var.add_trace(_go.Scatter(
                x=_df_snap.index, y=_df_snap["consumo_ewma_pct"]*100,
                mode="lines", name="EWMA (lambda=0.99)",
                line=dict(color="#fd7e14", width=2)
            ))
            _fig_var.update_layout(
                title="Utilizacao do risco (VaR / limite)",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(ticksuffix="%")
            )
            st.plotly_chart(_fig_var, use_container_width=True)

            # === Tabela estatisticas mensais ===
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
                _st = {
                    "Total meses": _n,
                    "Meses positivos": f'{sum(1 for m in _meses if m["ret"]>0)}/{_n} ({sum(1 for m in _meses if m["ret"]>0)/_n*100:.0f}%)',
                    "Meses negativos": f'{sum(1 for m in _meses if m["ret"]<0)}/{_n} ({sum(1 for m in _meses if m["ret"]<0)/_n*100:.0f}%)',
                    "Maior retorno mensal": f'{max(m["ret"] for m in _meses)*100:+.2f}%',
                    "Menor retorno mensal": f'{min(m["ret"] for m in _meses)*100:+.2f}%',
                    "Meses acima do CDI": f'{sum(1 for m in _meses if m["ret"]>m["cdi"])}/{_n} ({sum(1 for m in _meses if m["ret"]>m["cdi"])/_n*100:.0f}%)',
                    "Meses abaixo do CDI": f'{sum(1 for m in _meses if m["ret"]<m["cdi"])}/{_n} ({sum(1 for m in _meses if m["ret"]<m["cdi"])/_n*100:.0f}%)',
                }
                st.markdown("### Estatisticas mensais (historico completo)")
                st.dataframe(pd.DataFrame([_st]).T.rename(columns={0: "Valor"}), use_container_width=True)

'''
    s = s.replace(bloco_antigo, novo_bloco); n += 1
    print(f"[ok] bloco Resumo de Volatilidade substituido ({len(bloco_antigo)} -> {len(novo_bloco)} chars)")

f.write_text(s, encoding="utf-8")
print(f"[done] {n} mudancas aplicadas em app4.py")
