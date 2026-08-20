"""comparar_fontes_local.py - drift df_inicial vs df_preco (recomputado localmente)"""
import sys
from pathlib import Path
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from atualizar_retornos_diarios import (
    _ler_precos_wide_df_preco, _ler_precos_wide_df_inicial, _calcular_log_returns
)

print("[1/3] Lendo df_preco...")
df_p = _calcular_log_returns(_ler_precos_wide_df_preco()).dropna(subset=["retorno"])
df_p["chave"] = df_p["Data"].dt.strftime("%Y-%m-%d") + "|" + df_p["Ativo"]
df_p = df_p.set_index("chave")[["Data","Ativo","retorno"]].rename(columns={"retorno":"ret_preco"})
print(f"  {len(df_p)} pares em df_preco")

print("[2/3] Lendo df_inicial...")
df_i = _calcular_log_returns(_ler_precos_wide_df_inicial()).dropna(subset=["retorno"])
df_i["chave"] = df_i["Data"].dt.strftime("%Y-%m-%d") + "|" + df_i["Ativo"]
df_i = df_i.set_index("chave")[["Data","Ativo","retorno"]].rename(columns={"retorno":"ret_inicial"})
print(f"  {len(df_i)} pares em df_inicial")

print("[3/3] Merge dos overlaps...")
m = df_p[["Data","Ativo","ret_preco"]].join(df_i[["ret_inicial"]], how="inner")
m["diff"] = m["ret_inicial"] - m["ret_preco"]
m["abs_diff"] = m["diff"].abs()
print(f"  {len(m)} pares em ambas as fontes")

# Extrai classe do ativo
def classe(a):
    a = str(a).upper()
    if a.startswith("DI_"): return "DI"
    if a.startswith("DAP"): return "DAP"
    if a.startswith("NTNB"): return "NTNB"
    if a in ("WDO1","TREASURY","IBOV"): return a
    return "OUTROS"
m["classe"] = m["Ativo"].apply(classe)

print("\n=== DRIFT MEDIO POR CLASSE ===")
stats_classe = m.groupby("classe").agg(
    n_pares=("abs_diff","count"),
    media_bps=("abs_diff", lambda x: x.mean()*10_000),
    mediana_bps=("abs_diff", lambda x: x.median()*10_000),
    max_bps=("abs_diff", lambda x: x.max()*10_000),
    p95_bps=("abs_diff", lambda x: x.quantile(0.95)*10_000),
).sort_values("media_bps", ascending=False)
print(stats_classe.to_string())

print("\n=== DRIFT MEDIO POR ATIVO (todos) ===")
stats_ativo = m.groupby("Ativo").agg(
    n=("abs_diff","count"),
    media_bps=("abs_diff", lambda x: x.mean()*10_000),
    max_bps=("abs_diff", lambda x: x.max()*10_000),
    equivalencia=("abs_diff", lambda x: (x < 0.0001).mean()*100),   # % pares com drift < 1bp
).sort_values("media_bps", ascending=False)
print(stats_ativo.to_string())

print("\n=== VEREDITO ===")
for ativo, row in stats_ativo.iterrows():
    status = "COMPATIVEL" if row["media_bps"] < 1.0 else ("SUSPEITO" if row["media_bps"] < 5.0 else "INCOMPATIVEL")
    print(f"  {ativo:<12}  {status:<12}  media {row['media_bps']:>7.2f} bps  ({row['equivalencia']:>5.1f}% dos pares batem)")
