"""patch_fase2b_donuts_var.py - Resumo de Orcamento com donuts HIST + EWMA do snapshot"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_fase2b_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
n = 0

# --- Replace content of "Resumo de Orcamento" bloco ---
# Marca inicio: st.subheader("Resumo de Orcamento") ate antes de "with COLmeio:"
inicio = '''        with col11:
            st.subheader("Resumo de Orçamento")
            c1, c2  = st.columns(2)
            c1.metric("VaR (R$ / bps)",  var_display)
            c2.metric("CVaR (R$ / bps)", cvar_display)'''

novo_bloco = '''        with col11:
            st.subheader("Consumo do orçamento (VaR)")
            # --- Puxa VaR HIST + EWMA do snapshot_diario (Supabase) ---
            _snap_ult = supabase.table("snapshot_diario").select(
                "Data,pl_total,var_carteira_hist_reais,consumo_hist_pct,var_carteira_ewma_reais,consumo_ewma_pct,var_limite_efet_bps"
            ).order("Data", desc=True).limit(1).execute().data
            if _snap_ult:
                _s = _snap_ult[0]
                _pl = _s.get("pl_total") or 0
                _lim_bps = _s.get("var_limite_efet_bps") or 1.0
                _limite_R = (_lim_bps / 10_000) * _pl if _pl else 0
                _var_hist = _s.get("var_carteira_hist_reais") or 0
                _var_ewma = _s.get("var_carteira_ewma_reais") or 0
                _cons_hist = _s.get("consumo_hist_pct") or 0
                _cons_ewma = _s.get("consumo_ewma_pct") or 0

                import plotly.graph_objects as _pgo
                def _donut(pct, titulo, valor_R):
                    if pct >= 1.0: cor = "#dc3545"
                    elif pct >= 0.8: cor = "#fd7e14"
                    elif pct >= 0.5: cor = "#ffc107"
                    else: cor = "#28a745"
                    _fig = _pgo.Figure(_pgo.Pie(
                        values=[min(pct, 1)*100, max(1-pct, 0)*100], hole=0.65,
                        marker=dict(colors=[cor, "#e5e7eb"]),
                        textinfo="none", sort=False, direction="clockwise",
                        rotation=0, showlegend=False
                    ))
                    _fig.update_layout(
                        title=dict(text=titulo, x=0.5, xanchor="center", font=dict(size=14, color="#1a3a6c")),
                        annotations=[
                            dict(text=f"<b>{pct*100:.0f}%</b>", x=0.5, y=0.55, showarrow=False, font=dict(size=28, color=cor)),
                            dict(text="do limite", x=0.5, y=0.4, showarrow=False, font=dict(size=10, color="#666")),
                        ],
                        margin=dict(l=10, r=10, t=40, b=10), height=280
                    )
                    return _fig

                _c1, _c2 = st.columns(2)
                with _c1:
                    st.plotly_chart(_donut(_cons_hist, f"VaR HIST 3 anos<br><span style=\"font-size:12px;color:#333\">R$ {_var_hist:,.0f}</span>", _var_hist), use_container_width=True)
                with _c2:
                    st.plotly_chart(_donut(_cons_ewma, f"VaR EWMA (λ=0.99)<br><span style=\"font-size:12px;color:#333\">R$ {_var_ewma:,.0f}</span>", _var_ewma), use_container_width=True)

                st.caption(f"Limite: R$ {_limite_R:,.0f} ({_lim_bps:.2f} bps do PL) — snapshot {_s.get(\"Data\")}")
            else:
                st.info("Sem snapshot_diario disponivel.")
'''

if inicio in s:
    s = s.replace(inicio, novo_bloco); n += 1
    print("[ok] Resumo de Orcamento substituido por donuts HIST + EWMA")
else:
    print("[warn] bloco Resumo de Orcamento nao encontrado")

# --- Remove antigo "Consumo do orçamento (VaR)" que ficava logo abaixo ---
old_consumo = '''        with col11:
            st.subheader("Consumo do orçamento (VaR)")
            #colA, colB = st.columns(2)
            #with colA:
            st.caption(f"VaR — {fmt_pct(pct_consumo_var(var_bps/100))} do orçamento")
            donut_chart("VaR", pct_consumo_var(var_bps/100))

            #with colB:
            #    st.caption(f"CVaR — {fmt_pct(pct_consumo_cvar(cvar_bps/100))} do orçamento")
            #    donut_chart("CVaR", pct_consumo_cvar(cvar_bps/100))'''

new_consumo = '''        # REMOVED Fase 2b: Consumo do orcamento antigo substituido pelos donuts HIST+EWMA acima
        if False:
            with col11:
                st.subheader("Consumo do orçamento (VaR)")
                st.caption(f"VaR — {fmt_pct(pct_consumo_var(var_bps/100))} do orçamento")
                donut_chart("VaR", pct_consumo_var(var_bps/100))'''

if old_consumo in s:
    s = s.replace(old_consumo, new_consumo); n += 1
    print("[ok] Consumo antigo desativado")

f.write_text(s, encoding="utf-8")
print(f"[done] {n} mudancas Fase 2b")
