import pandas as pd
pl  = pd.read_parquet("Dados/pl_fundos_teste.parquet")
c_pl = [c for c in pl.columns if c not in ("Fundos/Carteiras Adm", "Último Valor")]
cdi = pd.read_csv("Dados/cdi_cached.csv", parse_dates=["Data"])
lft = pd.read_csv("Dados/dados_lft.csv", parse_dates=["Data"]).dropna()
b3  = [c for c in pd.read_parquet("Dados/df_preco_de_ajuste_atual_completo.parquet").columns if c != "Assets"]
print("PL fundos (ScrapAF3):", max(pd.to_datetime(c_pl)).date())
print("CDI (BCB)           :", cdi["Data"].max().date())
print("LFT (BCB)           :", lft["Data"].max().date())
print("Precos B3 (ScrapB3) :", max(pd.to_datetime(b3)).date())
