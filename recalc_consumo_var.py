"""recalc_consumo_var.py - consumo = VaR / limite EFETIVO (pos-governance)"""
import os
from supabase import create_client
c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

rows, off = [], 0
while True:
    r = c.table("snapshot_diario").select(
        "Data,pl_total,var_limite_efet_bps,var_carteira_hist_reais,var_carteira_ewma_reais,consumo_hist_pct,consumo_ewma_pct"
    ).order("Data").range(off, off + 999).execute()
    if not r.data: break
    rows.extend(r.data)
    if len(r.data) < 1000: break
    off += 1000

n = 0
for s in rows:
    pl = s.get("pl_total") or 0
    lim_bps = s.get("var_limite_efet_bps") or 1.0
    lim_R = pl * lim_bps / 10_000
    if not lim_R:
        continue
    h = (s.get("var_carteira_hist_reais") or 0) / lim_R
    e = (s.get("var_carteira_ewma_reais") or 0) / lim_R
    if abs(h - (s.get("consumo_hist_pct") or 0)) > 1e-6 or abs(e - (s.get("consumo_ewma_pct") or 0)) > 1e-6:
        c.table("snapshot_diario").update({"consumo_hist_pct": h, "consumo_ewma_pct": e}).eq("Data", s["Data"]).execute()
        n += 1
        print(f"  {s['Data']}  limite={lim_bps:.2f}bps  hist={h*100:.1f}%  ewma={e*100:.1f}%")
print(f"[recalc-consumo] {n} dias corrigidos")
