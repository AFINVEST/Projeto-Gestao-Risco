"""patch_fase1_titulos.py - remove header extra + acentos"""
from pathlib import Path
import shutil, datetime as dt
f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_titulos_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# 1) Remove o header "Metricas historicas de risco (snapshot)"
old = '        st.subheader("Metricas historicas de risco (snapshot)")\n\n'
if old in s: s = s.replace(old, ""); print("[ok] header removido")

# 2) Title stats: "Estatisticas mensais (historico completo, N meses)" -> "Estatisticas mensais"
old = 'st.markdown(f"### Estatisticas mensais (historico completo, {_n} meses)")'
new = 'st.markdown("### Estatísticas mensais")'
if old in s: s = s.replace(old, new); print("[ok] titulo stats limpo + acento")

# 3) Acentos nos titulos dos charts plotly
correcoes = [
    ('title="Volatilidade 20d anualizada (%)"', 'title="Volatilidade 20d anualizada"'),
    ('title="Drawdown historico (%)"', 'title="Drawdown histórico"'),
    ('title="Utilizacao do risco (VaR / limite)"', 'title="Utilização do risco (VaR / limite)"'),
    ('aba_cart,tab_orcamento = st.tabs(["Historico Carteira", "Portfolio Atual"])',
     'aba_cart,tab_orcamento = st.tabs(["Histórico Carteira", "Portfólio Atual"])'),
]
for a, b in correcoes:
    if a in s: s = s.replace(a, b); print(f"[ok] {b[:40]}")

f.write_text(s, encoding="utf-8")
print("[done]")
