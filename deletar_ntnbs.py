import os
from supabase import create_client
c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
ntnbs = ["NTNB26","NTNB27","NTNB28","NTNB30","NTNB32","NTNB35","NTNB40","NTNB45","NTNB50","NTNB55","NTNB60"]
r = c.table("retornos_diarios_ativo").delete().in_("Ativo", ntnbs).execute()
print(f"Deletadas {len(r.data or [])} linhas de NTNBs")
