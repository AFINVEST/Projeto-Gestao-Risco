"""diag_var_ewma.py - inspeciona scenarios que definem HIST e EWMA VaR pra uma data"""
import os, sys, math
from pathlib import Path
import pandas as pd, numpy as np
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))
from supabase import create_client
from risco_carteira_core import calcular_var_completo
from cota_portfolio_core import load_basefundos

DATA_ALVO = "2026-04-15"   # ajuste aqui
LAM = 0.99
JANELA = 756

c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
bf = load_basefundos()

# Pega PL do dia
snap = c.table("snapshot_diario").select("*").eq("Data", DATA_ALVO).execute().data
if not snap:
    sys.exit(f"snapshot {DATA_ALVO} nao existe")
pl = snap[0]["pl_total"]
print(f"Data alvo: {DATA_ALVO}   PL: R$ {pl:,.0f}")

# Roda calcular_var_completo com verbose interno se possivel; senao chama e mostra o resultado
r = calcular_var_completo(
    data_ref=pd.Timestamp(DATA_ALVO),
    pl_total=pl,
    basefundos=bf,
    janela_dias=JANELA,
    limite_pct_pl=0.0001,
)
print(f"VaR HIST:  R$ {r.get('var_hist_R', 0):,.0f}  ({r.get('consumo_hist_pct',0)*100:.1f}% do limite)")
print(f"VaR EWMA:  R$ {r.get('var_ewma_R', 0):,.0f}  ({r.get('consumo_ewma_pct',0)*100:.1f}% do limite)")

# Reconstroi scenarios "na mao" via retornos_diarios_ativo
# Puxa retornos dos ultimos 756 DU
data_end = pd.Timestamp(DATA_ALVO)
data_start = data_end - pd.Timedelta(days=365*3+30)
qr = c.table("retornos_diarios_ativo").select("Data,Ativo,retorno") \
      .gte("Data", data_start.date().isoformat()).lte("Data", data_end.date().isoformat()) \
      .execute()
df_ret = pd.DataFrame(qr.data)
df_ret["Data"] = pd.to_datetime(df_ret["Data"])
df_ret = df_ret.pivot(index="Data", columns="Ativo", values="retorno").sort_index()
df_ret = df_ret.tail(JANELA)   # ultimos 756 DUs

# Aqui precisaria da posicao (P&L por ativo) pra montar cenario. Simplificacao:
# usa a media dos retornos dos ativos com posicao (nao ideal, so pra visualizar cauda)
print(f"\nRetornos disponiveis: {df_ret.shape[0]} DUs x {df_ret.shape[1]} ativos")
print(f"Media dos 10 piores dias (retorno medio dos ativos):")
media_diaria = df_ret.mean(axis=1)
piores = media_diaria.sort_values().head(20)
n = len(media_diaria)
# Pesos EWMA
idx = np.arange(n)
w = (LAM ** (n - 1 - idx)) * (1 - LAM) / (1 - LAM ** n)
peso_por_data = pd.Series(w, index=media_diaria.index)
for data, ret in piores.items():
    dias_atras = (data_end - data).days
    peso_ewma = peso_por_data.loc[data]
    peso_hist = 1 / n
    print(f"  {data.date()}  (h-{dias_atras:3d}d)  ret_med={ret*100:+.3f}%  "
          f"peso_HIST={peso_hist*100:.3f}%  peso_EWMA={peso_ewma*100:.3f}%  "
          f"razao_EWMA/HIST={peso_ewma/peso_hist:.2f}x")
