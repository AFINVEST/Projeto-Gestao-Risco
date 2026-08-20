import os
from supabase import create_client
c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
r = c.table("snapshot_diario").select(
    "Data,covar_juros_nom_pct,covar_juros_real_pct,covar_moeda_pct,covar_juros_us_pct,covar_total_R"
).order("Data", desc=True).limit(5).execute()
print("Ultimos 5 snapshots:")
for row in r.data:
    print(row)

# Conta populados
r2 = c.table("snapshot_diario").select("Data", count="exact").not_.is_("covar_total_R", "null").execute()
print(f"\nSnapshots com covar_total_R populado: {r2.count}")

r3 = c.table("snapshot_diario").select("Data", count="exact").execute()
print(f"Total snapshots: {r3.count}")
