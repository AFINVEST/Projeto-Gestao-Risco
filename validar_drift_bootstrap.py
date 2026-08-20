"""validar_drift_bootstrap.py - compara retornos Supabase vs df_preco recalculado"""
import os, sys
from pathlib import Path
import pandas as pd, numpy as np
from supabase import create_client

sys.path.insert(0, str(Path(__file__).parent))
# reusa parser pt-BR
from atualizar_retornos_diarios import _ler_precos_wide_df_preco, _calcular_log_returns

client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

print("[1/3] Lendo df_preco.parquet e recalculando retornos localmente...")
df_local = _ler_precos_wide_df_preco()
df_local = _calcular_log_returns(df_local).dropna(subset=["retorno"])
df_local["chave"] = df_local["Data"].dt.strftime("%Y-%m-%d") + "|" + df_local["Ativo"]
df_local = df_local.set_index("chave")[["Data","Ativo","retorno"]].rename(columns={"retorno":"ret_local"})
print(f"  {len(df_local)} pares (Data,Ativo) no df_preco local")

print("[2/3] Buscando Supabase (retornos atuais + fonte)...")
rows = []; offset = 0; page = 1000
while True:
    r = client.table("retornos_diarios_ativo").select("Data,Ativo,retorno,fonte") \
              .range(offset, offset+page-1).execute()
    if not r.data: break
    rows.extend(r.data)
    if len(r.data) < page: break
    offset += page
df_sup = pd.DataFrame(rows)
df_sup["chave"] = df_sup["Data"] + "|" + df_sup["Ativo"]
df_sup = df_sup.set_index("chave").rename(columns={"retorno":"ret_supabase"})
print(f"  {len(df_sup)} pares no Supabase")
print(f"  Distribuicao por fonte:")
for f, cnt in df_sup["fonte"].value_counts().items():
    print(f"    {f}: {cnt}")

print("[3/3] Comparando pares que existem em AMBOS...")
merged = df_local.join(df_sup[["ret_supabase","fonte"]], how="inner")
merged["diff"] = merged["ret_supabase"] - merged["ret_local"]
merged["abs_diff"] = merged["diff"].abs()

# So compara onde fonte atual == 'df_inicial' (indica overwrite)
sobrescritos = merged[merged["fonte"] == "df_inicial"]
print(f"\nPares onde df_inicial SOBRESCREVEU df_preco: {len(sobrescritos)}")
if len(sobrescritos) == 0:
    print("Nenhuma sobreposicao — bootstrap so preencheu datas novas. Tudo ok.")
    sys.exit(0)

print(f"\nEstatisticas do drift (Supabase - df_preco) em bps:")
print(f"  Media abs: {sobrescritos['abs_diff'].mean()*10_000:.4f} bps")
print(f"  Mediana abs: {sobrescritos['abs_diff'].median()*10_000:.4f} bps")
print(f"  Max abs: {sobrescritos['abs_diff'].max()*10_000:.4f} bps")
print(f"  P95 abs: {sobrescritos['abs_diff'].quantile(0.95)*10_000:.4f} bps")

# Threshold: 1 bps de diferenca
th = 0.0001
grandes = sobrescritos[sobrescritos["abs_diff"] > th].sort_values("abs_diff", ascending=False)
print(f"\nPares com drift > 1 bps: {len(grandes)} ({len(grandes)/len(sobrescritos)*100:.1f}%)")

if len(grandes) > 0:
    print(f"\nTop 20 maiores drifts:")
    print(f"{'Data':<12} {'Ativo':<10} {'ret_local':>12} {'ret_supabase':>12} {'diff bps':>10}")
    for _, r in grandes.head(20).iterrows():
        print(f"{r['Data'].strftime('%Y-%m-%d'):<12} {r['Ativo']:<10} "
              f"{r['ret_local']*100:>11.4f}% {r['ret_supabase']*100:>11.4f}% "
              f"{r['diff']*10_000:>+9.2f}")

    print(f"\nDrift por ativo (top 10 mais afetados):")
    stats = sobrescritos.groupby("Ativo").agg(
        n=("abs_diff","count"),
        media_bps=("abs_diff", lambda x: x.mean()*10_000),
        max_bps=("abs_diff", lambda x: x.max()*10_000),
    ).sort_values("media_bps", ascending=False).head(10)
    print(stats.to_string())
