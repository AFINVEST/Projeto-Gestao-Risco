import os
from supabase import create_client
c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

datas = ["2025-04-15","2025-08-19","2025-12-15","2026-03-20","2026-04-15","2026-08-11"]
print(f"{'Data':<12} {'hist%':>7} {'ewma%':>7} {'VaR hist R$':>15} {'VaR ewma R$':>15} {'n_ativos':>10}")
print("-" * 75)
for d in datas:
    r = c.table("snapshot_diario").select(
        "Data,consumo_hist_pct,consumo_ewma_pct,var_carteira_hist_reais,var_carteira_ewma_reais,n_ativos_carteira"
    ).eq("Data", d).execute().data
    if not r:
        print(f"{d:<12} (sem snapshot)")
        continue
    s = r[0]
    ch = (s.get("consumo_hist_pct") or 0) * 100
    ce = (s.get("consumo_ewma_pct") or 0) * 100
    vh = s.get("var_carteira_hist_reais") or 0
    ve = s.get("var_carteira_ewma_reais") or 0
    n = s.get("n_ativos_carteira") or 0
    print(f"{s['Data']:<12} {ch:>6.1f}% {ce:>6.1f}% R$ {vh:>12,.0f} R$ {ve:>12,.0f} {n:>10}")
