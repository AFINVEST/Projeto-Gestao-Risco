"""atualizar_covar_snapshot.py - CoVaR por estrategia no snapshot_diario.
Preenche covar_* nos snapshots vazios + ultimos 3 (self-heal). --todos recalcula tudo."""
import os, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from supabase import create_client
from risco_carteira_core import calcular_covar_completo
from cota_portfolio_core import load_basefundos

TODOS = "--todos" in sys.argv
c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

snaps, off = [], 0
while True:
    r = c.table("snapshot_diario").select("Data,covar_total_r").order("Data").range(off, off + 999).execute()
    if not r.data:
        break
    snaps += r.data
    if len(r.data) < 1000:
        break
    off += 1000

datas = [s["Data"] for s in snaps]
alvo = datas if TODOS else sorted(set(s["Data"] for s in snaps if s["covar_total_r"] is None) | set(datas[-3:]))
print(f"[covar] {len(snaps)} snapshots | a processar: {len(alvo)}" + (f" ({alvo[0]} .. {alvo[-1]})" if alvo else ""))
if not alvo:
    sys.exit(0)

bf = load_basefundos(verbose=False)
t0, ok, flat, erros = time.time(), 0, 0, 0
for i, d in enumerate(alvo):
    try:
        res = calcular_covar_completo(data_ref=pd.Timestamp(d), basefundos=bf, janela_dias=756)
        if "erro" in res:
            upd = {"covar_juros_nom_pct": None, "covar_juros_real_pct": None, "covar_moeda_pct": None,
                   "covar_juros_us_pct": None, "covar_outros_pct": None, "covar_total_r": 0.0}
            flat += 1
        else:
            pct = res.get("covar_por_classe_pct", {})
            upd = {"covar_juros_nom_pct":  pct.get("Juros Nominais BR"),
                   "covar_juros_real_pct": pct.get("Juros Reais BR"),
                   "covar_moeda_pct":      pct.get("Moeda"),
                   "covar_juros_us_pct":   pct.get("Juros US"),
                   "covar_outros_pct":     pct.get("Outros"),
                   "covar_total_r":        res.get("var_estimado_R")}
            ok += 1
        c.table("snapshot_diario").update(upd).eq("Data", d).execute()
    except Exception as e:
        erros += 1
        if erros <= 5:
            print(f"  [erro] {d}: {e}")
    if (i + 1) % 30 == 0:
        print(f"  [{i+1}/{len(alvo)}] {(time.time()-t0)/60:.1f} min")

print(f"[covar] ok={ok} sem_posicao={flat} erros={erros} ({(time.time()-t0)/60:.1f} min)")
sys.exit(1 if erros else 0)
