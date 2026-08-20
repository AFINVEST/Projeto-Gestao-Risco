"""patch_fase2a_ocultar.py - oculta Carry, DV01 historico, Vol por ativo"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_fase2a_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
n = 0

# --- Hide Carry display (mantem computo) ---
# Envelopa 3 statements de display em if False
old = '''                st.markdown("### Carry do portfólio DI ao longo do tempo (R$)")

                # métrica do carry atual
                if len(carry_portfolio_ts) >= 2:
                    curr = float(carry_portfolio_ts.iloc[-1])
                    prev = float(carry_portfolio_ts.iloc[-2])
                    st.metric(label="Carry DI do portfólio (1d) — atual", value=f"R$ {curr:,.2f}", delta=f"{(curr-prev):,.2f}")
                elif len(carry_portfolio_ts) == 1:
                    curr = float(carry_portfolio_ts.iloc[-1])
                    st.metric(label="Carry DI do portfólio (1d) — atual", value=f"R$ {curr:,.2f}", delta=None)
                
                st.line_chart(carry_portfolio_ts)'''

new = '''                # HIDDEN Fase 2: Carry chart oculto ate refazer o calculo corretamente
                if False:
                    st.markdown("### Carry do portfólio DI ao longo do tempo (R$)")
                    if len(carry_portfolio_ts) >= 2:
                        curr = float(carry_portfolio_ts.iloc[-1])
                        prev = float(carry_portfolio_ts.iloc[-2])
                        st.metric(label="Carry DI do portfólio (1d) — atual", value=f"R$ {curr:,.2f}", delta=f"{(curr-prev):,.2f}")
                    elif len(carry_portfolio_ts) == 1:
                        curr = float(carry_portfolio_ts.iloc[-1])
                        st.metric(label="Carry DI do portfólio (1d) — atual", value=f"R$ {curr:,.2f}", delta=None)
                    st.line_chart(carry_portfolio_ts)'''

if old in s: s = s.replace(old, new); n += 1; print("[ok] Carry oculto")

# Tambem oculta o expander de contribuicao por DI
old2 = '''                if "carry_assets_ts" in locals() and not carry_assets_ts.empty:
                    last_contrib = carry_assets_ts.iloc[-1].sort_values(ascending=False)
                    with st.expander("Exibir mais — Contribuição por DI (última data)", expanded=False):
                        st.dataframe(
                            last_contrib.to_frame("Carry_R$"),
                            use_container_width=True
                        )'''
new2 = '''                if False:  # HIDDEN Fase 2
                    if "carry_assets_ts" in locals() and not carry_assets_ts.empty:
                        last_contrib = carry_assets_ts.iloc[-1].sort_values(ascending=False)
                        with st.expander("Exibir mais — Contribuição por DI (última data)", expanded=False):
                            st.dataframe(last_contrib.to_frame("Carry_R$"), use_container_width=True)'''
if old2 in s: s = s.replace(old2, new2); n += 1; print("[ok] Carry expander oculto")

# --- Hide DV01 historico + CoVaR historico display ---
# So esconde o st.subheader; o restante do bloco vai continuar rodando (computo) mas nao renderiza
old3 = '''        with COL2:
            st.subheader("Histórico de DV01 & CoVaR")'''
new3 = '''        with COL2:
            if False:  # HIDDEN Fase 2: DV01 historico oculto ate remocao definitiva
                st.subheader("Histórico de DV01 & CoVaR")'''
if old3 in s: s = s.replace(old3, new3); n += 1; print("[ok] DV01 historico subheader oculto")

# --- Remove Volatilidade historica por ativo (linhas 9142-9143) ---
old4 = '''        st.subheader("Volatilidade histórica por ativo")
        st.plotly_chart(fig_vol_assets, use_container_width=True)'''
new4 = '''        # REMOVED Fase 2: Volatilidade historica por ativo removida do dashboard
        # st.subheader("Volatilidade histórica por ativo")
        # st.plotly_chart(fig_vol_assets, use_container_width=True)'''
if old4 in s: s = s.replace(old4, new4); n += 1; print("[ok] Vol por ativo removida")

f.write_text(s, encoding="utf-8")
print(f"[done] {n} mudancas Fase 2a")
