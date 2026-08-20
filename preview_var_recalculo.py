"""preview_var_recalculo.py - compara VaR gravado vs recalculado com historico completo"""
import os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from supabase import create_client
from risco_carteira_core import calcular_var_completo
from cota_portfolio_core import load_basefundos

DATAS = ["2025-04-15", "2025-12-15", "2026-04-15", "2026-08-11"]

c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
bf = load_basefundos()

print(f"{'Data':<12} | {'VaR HIST gravado':>18} | {'VaR HIST novo':>18} | {'VaR EWMA gravado':>18} | {'VaR EWMA novo':>18} | {'n retornos':>10}")
print("-" * 120)
for d in DATAS:
    snap = c.table("snapshot_diario").select("*").eq("Data", d).execute().data
    if not snap:
        print(f"{d:<12} | (sem snapshot)")
        continue
    s = snap[0]
    pl = s["pl_total"]
    old_hist = s.get("var_carteira_hist_reais") or 0
    old_ewma = s.get("var_carteira_ewma_reais") or 0
    old_consumo_hist = (s.get("consumo_hist_pct") or 0) * 100
    old_consumo_ewma = (s.get("consumo_ewma_pct") or 0) * 100

    novo = calcular_var_completo(
        data_ref=pd.Timestamp(d), pl_total=pl,
        basefundos=bf, janela_dias=756, limite_pct_pl=0.0001,
    )
    if "erro" in novo:
        print(f"{d:<12} | ERRO: {novo['erro']}")
        continue
    new_hist = novo.get("var_hist_R", 0)
    new_ewma = novo.get("var_ewma_R", 0)
    new_consumo_hist = (novo.get("consumo_hist_pct") or 0) * 100
    new_consumo_ewma = (novo.get("consumo_ewma_pct") or 0) * 100
    n_ret = novo.get("n_ativos", 0)

    print(f"{d:<12} | R$ {old_hist:>10,.0f} ({old_consumo_hist:>4.1f}%) | R$ {new_hist:>10,.0f} ({new_consumo_hist:>4.1f}%) | R$ {old_ewma:>10,.0f} ({old_consumo_ewma:>4.1f}%) | R$ {new_ewma:>10,.0f} ({new_consumo_ewma:>4.1f}%) | {n_ret:>10}")
