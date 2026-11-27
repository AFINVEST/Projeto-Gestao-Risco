import os, warnings, subprocess
warnings.filterwarnings("ignore")
import pandas as pd
from supabase import create_client
import cota_portfolio_core as C

D = ["2026-11-23", "2026-11-24", "2026-11-25"]
A = ["DI_F28", "DI_F29", "DI_F35", "WDO_Z26"]
for nome, path in [("PRECO", "Dados/df_preco_de_ajuste_atual_completo.parquet"),
                   ("VALOR AJUSTE", "Dados/df_valor_ajuste_contrato.parquet")]:
    w = pd.read_parquet(path)
    dup = w["Assets"][w["Assets"].duplicated(keep=False)].value_counts()
    print(f"\n=== {nome} (local) | {len(w)} linhas | ultimas colunas: {[c for c in w.columns if c != 'Assets'][-3:]}")
    print(f"    duplicados: {dict(dup) if len(dup) else 'nenhum'}")
    sub = w[w["Assets"].isin(A)][["Assets"] + [x for x in D if x in w.columns]]
    print(sub.to_string(index=False))

bf = C._migrate_basefundos_names(C.load_basefundos(verbose=False), verbose=False)
pl, _ = C.load_pl_series()
pn, _, _, _ = C.analisar_dados_fundos2(float(pl.iloc[-1]), C.load_b3_prices(), C.load_ajustes(), bf)
pn = pn.drop(columns="Total", errors="ignore").apply(pd.to_numeric, errors="coerce")
pn.columns = pd.to_datetime(pn.columns).strftime("%Y-%m-%d")
pn = pn.groupby(pn.index.str.split(" - ").str[0]).sum().reindex(columns=D)
print("\n=== P&L local (mesmo calculo do snapshot) ===")
print(pn[pn.abs().sum(axis=1) > 0].round(0).to_string())
print("TOTAL:", pn.sum().round(0).to_dict())

c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
print("\n=== SNAPSHOT ===")
print(pd.DataFrame(c.table("snapshot_diario").select("Data,cota,retorno_dtd").in_("Data", D).order("Data").execute().data).to_string(index=False))

print("\n=== GIT ===")
print(subprocess.run(["git", "status", "--short", "Dados/df_valor_ajuste_contrato.parquet", "Dados/df_preco_de_ajuste_atual_completo.parquet"], capture_output=True, text=True).stdout or "(local = commit)")
print(subprocess.run(["git", "log", "-3", "--format=%h %ad %s", "--date=format:%d/%m %H:%M", "--", "Dados/df_valor_ajuste_contrato.parquet"], capture_output=True, text=True).stdout)
