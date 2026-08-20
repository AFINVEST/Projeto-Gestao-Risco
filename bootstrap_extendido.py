"""bootstrap_extendido.py - recupera DAPs, WDO, TREASURY, IBOV do df_inicial (sem NTNBs)"""
import os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from atualizar_retornos_diarios import _ler_precos_wide_df_inicial, _calcular_log_returns, _detecta_moeda, BATCH_SIZE
from supabase import create_client

ATIVOS_OK = [
    "DAP_K27","DAP_K35","DAP_Q26","DAP_Q28","DAP_Q30","DAP_Q32","DAP_Q40",
    "TREASURY","WDO1","IBOV",
]
# NTNBs excluidos ate corrigir FechamentoNTNBs (sheet_name='Planilha1')

client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
print(f"[extendido] ativos: {ATIVOS_OK}")

df = _calcular_log_returns(_ler_precos_wide_df_inicial()).dropna(subset=["retorno"])
df = df[df["Ativo"].isin(ATIVOS_OK)]
print(f"[extendido] {len(df)} pares filtrados, cobertura {df['Data'].min().date()} -> {df['Data'].max().date()}")

r = client.table("retornos_diarios_ativo").select("Data,Ativo").in_("Ativo", ATIVOS_OK).execute()
existentes = {(row["Data"], row["Ativo"]) for row in r.data or []}
print(f"[extendido] {len(existentes)} pares ja existem no Supabase")

df_novo = df[~df.apply(lambda r: (r["Data"].strftime("%Y-%m-%d"), r["Ativo"]) in existentes, axis=1)]
print(f"[extendido] {len(df_novo)} pares novos (pre-2025) a inserir")

regs = [{
    "Data": r["Data"].date().isoformat(),
    "Ativo": str(r["Ativo"]),
    "preco": float(r["preco"]),
    "retorno": float(r["retorno"]),
    "moeda": _detecta_moeda(r["Ativo"]),
    "fonte": "df_inicial_extendido",
} for _, r in df_novo.iterrows()]

n = 0
for i in range(0, len(regs), BATCH_SIZE):
    lote = regs[i:i+BATCH_SIZE]
    client.table("retornos_diarios_ativo").upsert(lote, on_conflict="Data,Ativo").execute()
    n += len(lote)
    print(f"  [upsert] {n}/{len(regs)}")
print("[extendido] OK")
