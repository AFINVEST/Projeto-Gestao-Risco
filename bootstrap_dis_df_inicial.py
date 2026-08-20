"""bootstrap_dis_df_inicial.py - bootstrap SELETIVO de DIs do df_inicial
   (sem tocar NTNBs, DAPs, WDO, etc que tem drift)"""
import os, sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from atualizar_retornos_diarios import _ler_precos_wide_df_inicial, _calcular_log_returns, _detecta_moeda, BATCH_SIZE
from supabase import create_client

# Ativos compativeis (validados via comparar_fontes_local: DIs 100%)
ATIVOS_OK = ["DI_F26","DI_F27","DI_F28","DI_F29","DI_F30","DI_F31","DI_F32","DI_F33","DI_F35"]

client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

print(f"[dis-bootstrap] lendo df_inicial e filtrando ativos: {ATIVOS_OK}")
df = _calcular_log_returns(_ler_precos_wide_df_inicial()).dropna(subset=["retorno"])
df = df[df["Ativo"].isin(ATIVOS_OK)]
print(f"[dis-bootstrap] {len(df)} pares (Data,Ativo) filtrados")

# Data range
print(f"[dis-bootstrap] cobertura: {df['Data'].min().date()} -> {df['Data'].max().date()}")

# Verifica quais pares JA existem em df_preco (nao mexer)
r = client.table("retornos_diarios_ativo").select("Data,Ativo") \
          .in_("Ativo", ATIVOS_OK) \
          .execute()
existentes = {(row["Data"], row["Ativo"]) for row in r.data or []}
print(f"[dis-bootstrap] {len(existentes)} pares ja no Supabase pra esses ativos")

# Filtra so os NAO existentes (nao sobrescreve o df_preco atual)
df["chave"] = df["Data"].dt.strftime("%Y-%m-%d") + "|" + df["Ativo"]
df_novo = df[~df.apply(lambda r: (r["Data"].strftime("%Y-%m-%d"), r["Ativo"]) in existentes, axis=1)]
print(f"[dis-bootstrap] {len(df_novo)} pares novos a inserir")

if len(df_novo) == 0:
    print("[dis-bootstrap] nada a fazer.")
    sys.exit(0)

regs = []
for _, r in df_novo.iterrows():
    regs.append({
        "Data": r["Data"].date().isoformat(),
        "Ativo": str(r["Ativo"]),
        "preco": float(r["preco"]),
        "retorno": float(r["retorno"]),
        "moeda": _detecta_moeda(r["Ativo"]),
        "fonte": "df_inicial_dis",   # label distinto pra rastreabilidade
    })

print(f"[dis-bootstrap] upsertando {len(regs)}...")
n = 0
for i in range(0, len(regs), BATCH_SIZE):
    lote = regs[i:i+BATCH_SIZE]
    client.table("retornos_diarios_ativo").upsert(lote, on_conflict="Data,Ativo").execute()
    n += len(lote)
    print(f"  [upsert] {n}/{len(regs)}")

# Confere estado final
r = client.table("retornos_diarios_ativo").select("Data").in_("Ativo", ATIVOS_OK) \
          .order("Data").limit(1).execute()
print(f"[dis-bootstrap] primeiro retorno de DI agora: {r.data[0]['Data'] if r.data else 'N/A'}")
