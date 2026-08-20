import os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from atualizar_retornos_diarios import _ler_precos_wide_df_inicial, _calcular_log_returns, _detecta_moeda, BATCH_SIZE
from supabase import create_client

NTNBS = ["NTNB26","NTNB27","NTNB28","NTNB30","NTNB32","NTNB35","NTNB40","NTNB45","NTNB50","NTNB55","NTNB60"]

c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
df = _calcular_log_returns(_ler_precos_wide_df_inicial()).dropna(subset=["retorno"])
df = df[df["Ativo"].isin(NTNBS)]
print(f"[ntnbs] {len(df)} pares BBG, cobertura {df['Data'].min().date()} -> {df['Data'].max().date()}")

r = c.table("retornos_diarios_ativo").select("Data,Ativo").in_("Ativo", NTNBS).execute()
existentes = {(row["Data"], row["Ativo"]) for row in r.data or []}
print(f"[ntnbs] {len(existentes)} pares ja no Supabase (do df_preco fresh)")

df_novo = df[~df.apply(lambda r: (r["Data"].strftime("%Y-%m-%d"), r["Ativo"]) in existentes, axis=1)]
print(f"[ntnbs] {len(df_novo)} pares novos pre-2025 a inserir")

regs = [{
    "Data": r["Data"].date().isoformat(),
    "Ativo": str(r["Ativo"]),
    "preco": float(r["preco"]),
    "retorno": float(r["retorno"]),
    "moeda": _detecta_moeda(r["Ativo"]),
    "fonte": "df_inicial_ntnb",
} for _, r in df_novo.iterrows()]

n = 0
for i in range(0, len(regs), BATCH_SIZE):
    lote = regs[i:i+BATCH_SIZE]
    c.table("retornos_diarios_ativo").upsert(lote, on_conflict="Data,Ativo").execute()
    n += len(lote)
    print(f"  [upsert] {n}/{len(regs)}")
print("[ntnbs] OK")
