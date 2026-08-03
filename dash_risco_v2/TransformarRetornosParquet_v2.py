"""
TransformarRetornosParquet_v2.py
=================================

Substitui o TransformarRetornosParquet.py original. Diferenças:

- ELIMINA leitura de 'Dados/BBG - ECO DASH.xlsx' (fonte antiga do BBG)
- GERA df_inicial.parquet e df_divone.parquet a partir de dados dinâmicos
  do Supabase (via taxas_dinamicas + dv01_dinamico)
- MANTÉM as demais conversões (CSVs → parquets, merge NTNB, LFT)

Coloque na raiz do projeto substituindo o original.

USO:
    python TransformarRetornosParquet.py    # (mesmo nome do antigo)
"""
from __future__ import annotations
import glob
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ---- 1) csv → parquet (mesmo comportamento antigo) ----
for file in glob.glob("*.csv"):
    df = pd.read_csv(file)
    df.to_parquet(file.replace(".csv", ".parquet"), index=False)
    os.remove(file)
    print(f"{file} convertido para parquet")
print("Conversão finalizada (root)")

for file in glob.glob("BaseFundos/*.csv"):
    df = pd.read_csv(file)
    df.to_parquet(file.replace(".csv", ".parquet"), index=False)
    os.remove(file)
    print(f"{file} convertido para parquet")
print("Conversão finalizada (BaseFundos)")


# ---- 2) df_inicial.parquet e df_divone.parquet a partir do Supabase ----
sys.path.insert(0, str(Path("dash_risco_v2")))
try:
    import taxas_dinamicas as td
    print("\n[v2] Gerando df_inicial.parquet do Supabase (taxas_dinamicas)...")
    td.gerar_df_inicial("Dados/df_inicial.parquet")

    print("\n[v2] Gerando df_divone.parquet do Supabase (dv01_dinamico)...")
    td.gerar_df_divone("Dados/df_divone.parquet")
except Exception as e:
    print(f"\n[v2] AVISO: geração dinâmica falhou: {e}")
    print("[v2] Os arquivos anteriores serão mantidos como fallback.")
    print("[v2] Verifique SUPABASE_URL/SUPABASE_KEY e conectividade.")


# ---- 3) NTNB merge (mesma lógica do original) ----
try:
    df_ntnb = pd.read_excel("Dados/FechamentoNTNBs.xlsx")
    df_ntnb.columns = df_ntnb.iloc[0]
    df_ntnb = df_ntnb.drop(df_ntnb.index[0])
    df_ntnb = df_ntnb.drop(df_ntnb.index[0])
    df_ntnb.dropna(inplace=True)
    df_ntnb.rename(columns={"Nome": "Assets"}, inplace=True)

    df_ntnb["Assets"] = pd.to_datetime(df_ntnb["Assets"]).dt.date
    df_ntnb["Assets"] = df_ntnb["Assets"].astype(str)
    df_ntnb = df_ntnb.T
    df_ntnb.columns = df_ntnb.iloc[0]
    df_ntnb = df_ntnb.drop(df_ntnb.index[0])
    df_ntnb.reset_index(inplace=True)
    df_ntnb.rename(columns={0: "Assets"}, inplace=True)

    for col in df_ntnb.columns:
        df_ntnb[col] = df_ntnb[col].apply(lambda x: str(x).replace(".", ","))

    df_b3 = pd.read_parquet("Dados/df_preco_de_ajuste_atual_completo.parquet")
    df_precos = pd.concat([df_b3, df_ntnb], axis=0)
    id_col = "Assets"
    date_cols = [c for c in df_precos.columns if c != id_col]
    date_cols = sorted(date_cols, key=pd.to_datetime)
    df_precos = df_precos[[id_col] + date_cols]
    df_precos[date_cols] = df_precos[date_cols].ffill(axis=1)
    df_precos.to_parquet("Dados/df_preco_de_ajuste_atual_completo.parquet")
    print("[v2] NTNBs mesclados em df_preco_de_ajuste_atual_completo.parquet")
except FileNotFoundError as e:
    print(f"[v2] AVISO: NTNB merge pulado ({e}).")


# ---- 4) LFT (mesma lógica do original) ----
try:
    dados = pd.read_excel(
        r"Z:\Asset Management\FUNDOS e CLUBES\Gerencial\dashboard LFT.xlsx",
        sheet_name="Historico preços"
    )
    dados.rename(columns={"Unnamed: 0": "Data"}, inplace=True)
    dados.drop(index=[0, 1], inplace=True)
    dados = dados[["Data", "BLFT 0 06/01/30"]]
    dados.rename(columns={"BLFT 0 06/01/30": "RetornoLFT"}, inplace=True)
    dados.to_csv("Dados/dados_lft.csv", index=False)
    print("[v2] LFT salvo em Dados/dados_lft.csv")
except Exception as e:
    print(f"[v2] AVISO: LFT falhou ({e}).")

print("\n[v2] TransformarRetornosParquet finalizado.")
