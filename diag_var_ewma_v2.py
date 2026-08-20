"""diag_var_ewma_v2.py - inspeciona os cenarios reais do VaR de carteira"""
import os, sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from risco_carteira_core import (
    carregar_retornos_historicos, carregar_posicoes_atuais, carregar_precos_atuais,
    calcular_var_carteira, _quantile_ewma, LAMBDA_EWMA_DEFAULT
)
from cota_portfolio_core import load_basefundos
from supabase import create_client

DATA_ALVO = pd.Timestamp("2026-04-15")   # ajuste
JANELA = 756
LAM = LAMBDA_EWMA_DEFAULT
ALPHA = 0.05

bf = load_basefundos()
qty = carregar_posicoes_atuais(DATA_ALVO, basefundos=bf)
precos = carregar_precos_atuais(DATA_ALVO)
retornos = carregar_retornos_historicos(DATA_ALVO, ativos=list(qty.keys()), janela_dias=JANELA)

print(f"Data: {DATA_ALVO.date()}")
print(f"Posicoes: {qty}")
print(f"Precos:   {precos}")
print(f"Retornos: {retornos.shape[0]} dias x {retornos.shape[1]} ativos")
print()

res = calcular_var_carteira(retornos, qty, precos, alpha=ALPHA, lambda_ewma=LAM)
port = res["port_ret_series"].sort_index()

# 5th percentile HIST (equal-weight)
var_hist = float(np.quantile(port.values, ALPHA))
# 5th percentile EWMA
var_ewma = _quantile_ewma(port, ALPHA, LAM)

print(f"5% quantile HIST (equal): {var_hist*100:+.4f}%")
print(f"5% quantile EWMA (lam={LAM}): {var_ewma*100:+.4f}%")
print()

# Top 20 worst P&L scenarios com pesos
n = len(port)
idx = np.arange(n)
w = (LAM ** (n - 1 - idx)) * (1 - LAM)
w = w / w.sum()
peso_ewma = pd.Series(w, index=port.index)

df_scen = pd.DataFrame({
    "port_ret": port,
    "peso_hist": 1/n,
    "peso_ewma": peso_ewma,
})
df_scen["dias_atras"] = (DATA_ALVO - df_scen.index).days
piores = df_scen.sort_values("port_ret").head(20)

print("TOP 20 PIORES CENARIOS (aplicando posicao atual em retornos historicos):")
print(f"{'Data':<12} {'h-Ndias':>8} {'PnL %':>10} {'peso HIST':>10} {'peso EWMA':>10} {'EWMA/HIST':>10}")
for data, r in piores.iterrows():
    razao = r['peso_ewma'] / r['peso_hist']
    print(f"{str(data.date()):<12} {r['dias_atras']:>8.0f} {r['port_ret']*100:>+9.4f}% {r['peso_hist']*100:>9.3f}% {r['peso_ewma']*100:>9.4f}% {razao:>9.2f}x")

print()
# Contagem por periodo
buckets = {"<30d": 0, "30-90d": 0, "90-180d": 0, "180-365d": 0, "365-730d": 0, ">730d": 0}
for _, r in piores.iterrows():
    d = r["dias_atras"]
    if   d < 30:   buckets["<30d"] += 1
    elif d < 90:   buckets["30-90d"] += 1
    elif d < 180:  buckets["90-180d"] += 1
    elif d < 365:  buckets["180-365d"] += 1
    elif d < 730:  buckets["365-730d"] += 1
    else:          buckets[">730d"] += 1
print("Distribuicao temporal dos top 20 piores:")
for b, c in buckets.items():
    print(f"  {b:<10}: {c}")
