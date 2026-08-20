"""recalcular_var_historico.py - atualiza var_carteira_* e consumo_* de todos os snapshots"""
import os, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from supabase import create_client
from risco_carteira_core import calcular_var_completo
from cota_portfolio_core import load_basefundos

c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
bf = load_basefundos()

print("[recalc-var] carregando datas de snapshot_diario...")
snaps = []
offset = 0
while True:
    r = c.table("snapshot_diario").select("Data,pl_total").order("Data").range(offset, offset+999).execute()
    if not r.data: break
    snaps.extend(r.data)
    if len(r.data) < 1000: break
    offset += 1000
print(f"[recalc-var] {len(snaps)} snapshots a recalcular")

t0 = time.time()
updated = 0; sem_dados = 0; erros = 0
for i, row in enumerate(snaps):
    d = row["Data"]
    pl = row.get("pl_total")
    if pl is None:
        sem_dados += 1
        continue
    try:
        novo = calcular_var_completo(
            data_ref=pd.Timestamp(d), pl_total=pl,
            basefundos=bf, janela_dias=756, limite_pct_pl=0.0001,
        )
        if "erro" in novo:
            sem_dados += 1
            continue
        upd = {
            "var_carteira_hist_reais": novo.get("var_hist_R"),
            "var_carteira_hist_bps": (novo.get("var_hist_R", 0) / pl * 10_000) if pl else None,
            "var_carteira_ewma_reais": novo.get("var_ewma_R"),
            "var_carteira_ewma_bps": (novo.get("var_ewma_R", 0) / pl * 10_000) if pl else None,
            "cvar_carteira_hist_reais": novo.get("cvar_hist_R"),
            "consumo_hist_pct": novo.get("consumo_hist_pct"),
            "consumo_ewma_pct": novo.get("consumo_ewma_pct"),
            "mv_total_carteira": novo.get("mv_total"),
            "n_ativos_carteira": novo.get("n_ativos"),
        }
        c.table("snapshot_diario").update(upd).eq("Data", d).execute()
        updated += 1
    except Exception as e:
        print(f"  [erro] {d}: {e}")
        erros += 1
    if (i+1) % 20 == 0:
        eta = (time.time() - t0) / (i+1) * (len(snaps) - i - 1)
        print(f"  [{i+1}/{len(snaps)}] eta ~{eta/60:.1f}min")

print(f"\n[recalc-var] OK  updated={updated}  sem_dados={sem_dados}  erros={erros}")
print(f"[recalc-var] tempo total: {(time.time()-t0)/60:.1f}min")
