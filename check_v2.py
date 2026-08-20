import os
from supabase import create_client
c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
r = c.table("snapshot_diario").select(
    "Data,covar_juros_nom_pct,covar_juros_real_pct,covar_moeda_pct,covar_juros_us_pct,covar_outros_pct,covar_total_r"
).order("Data", desc=True).limit(5).execute()
print("Ultimos 5:")
for row in r.data: print(row)
r2 = c.table("snapshot_diario").select("Data", count="exact").not_.is_("covar_total_r", "null").execute()
print(f"\nCom covar_total_r populado: {r2.count}")
