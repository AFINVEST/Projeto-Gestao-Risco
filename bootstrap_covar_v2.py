"""bootstrap_covar_v2.py - com nomes lowercase"""
import os, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from supabase import create_client
from risco_carteira_core import calcular_covar_completo
from cota_portfolio_core import load_basefundos

c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
bf = load_basefundos()

snaps = []
offset = 0
while True:
    r = c.table("snapshot_diario").select("Data").order("Data").range(offset, offset+999).execute()
    if not r.data: break
    snaps.extend(r.data)
    if len(r.data) < 1000: break
    offset += 1000
print(f"[covar-hist v2] {len(snaps)} datas")

t0 = time.time()
ok = 0; sem = 0; erros = 0
for i, row in enumerate(snaps):
    d = row["Data"]
    try:
        res = calcular_covar_completo(data_ref=pd.Timestamp(d), basefundos=bf, janela_dias=756)
        if "erro" in res: sem += 1; continue
        pct = res.get("covar_por_classe_pct", {})
        upd = {
            "covar_juros_nom_pct":   pct.get("Juros Nominais BR"),
            "covar_juros_real_pct":  pct.get("Juros Reais BR"),
            "covar_moeda_pct":       pct.get("Moeda"),
            "covar_juros_us_pct":    pct.get("Juros US"),
            "covar_outros_pct":      pct.get("Outros"),
            "covar_total_r":         res.get("var_estimado_R"),
        }
        c.table("snapshot_diario").update(upd).eq("Data", d).execute()
        ok += 1
    except Exception as e:
        erros += 1
        if erros <= 3: print(f"  [erro] {d}: {e}")
    if (i+1) % 30 == 0:
        eta = (time.time()-t0)/(i+1)*(len(snaps)-i-1)
        print(f"  [{i+1}/{len(snaps)}] eta ~{eta/60:.1f}min")
print(f"\n[done] ok={ok} sem_dados={sem} erros={erros} tempo={(time.time()-t0)/60:.1f}min")
