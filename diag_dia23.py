import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from supabase import create_client
import cota_portfolio_core as C

D0, D1 = "2026-11-18", "2026-11-24"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)

# 1) Operacoes cadastradas recentemente (data >= 17/11)
c = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
rows, off = [], 0
while True:
    r = c.table("posicoes_por_fundo").select("Fundo,Ativo,Data,Quantidade,Preco_Compra,Preco_Fechamento").gte("Data", "2026-11-17").range(off, off+999).execute()
    if not r.data: break
    rows += r.data; off += 1000
ops = pd.DataFrame(rows)
print("\n=== 1) OPERACOES com data >= 17/11 (soma por ativo/dia) ===")
if len(ops):
    ops["Quantidade"] = pd.to_numeric(ops["Quantidade"], errors="coerce")
    print(ops.groupby(["Data","Ativo"]).agg(qtd=("Quantidade","sum"), pc_min=("Preco_Compra","min"), pc_max=("Preco_Compra","max"), pf=("Preco_Fechamento","max")).to_string())
else:
    print("nenhuma")

# 2) P&L por ativo e por dia (R$)
bf = C._migrate_basefundos_names(C.load_basefundos(verbose=False), verbose=False)
pl, _ = C.load_pl_series()
df_pnl, _, df_desp, _ = C.analisar_dados_fundos2(float(pl.iloc[-1]), C.load_b3_prices(), C.load_ajustes(), bf)
pn = df_pnl.drop(columns="Total", errors="ignore").apply(pd.to_numeric, errors="coerce")
pn.columns = pd.to_datetime(pn.columns)
pn = pn.groupby(pn.index.str.split(" - ").str[0]).sum()
pn = pn.loc[:, (pn.columns >= D0) & (pn.columns <= D1)]
print("\n=== 2) P&L POR ATIVO (R$) ===")
print(pn[(pn.abs().sum(axis=1) > 0)].round(0).to_string())

# 3) PL, LFT, despesas e capital
lft = C.load_lft_series()
idx = pl.loc[D0:D1].index
cap = 0.01 * pl
desp = df_desp.loc["Despesas"] if "Despesas" in df_desp.index else pd.Series(dtype=float)
desp.index = pd.to_datetime(desp.index)
t = pd.DataFrame({"PL_total": pl.reindex(idx), "capital_1%": cap.reindex(idx),
                  "lft_ret_bps": lft.reindex(idx)*1e4, "despesas": desp.reindex(idx).fillna(0),
                  "pnl_total": pn.sum().reindex(idx)})
t["ret_aprox_bps"] = (t["pnl_total"] + cap.shift(1).reindex(idx)*lft.reindex(idx) - t["despesas"]) / cap.shift(1).reindex(idx) * 1e4
print("\n=== 3) PL / CAPITAL / RETORNO APROX ===")
print(t.round(2).to_string())

# 4) Precos B3 dos ativos em carteira
b3 = C.load_b3_prices().set_index("Assets")
cols = [x for x in b3.columns if D0 <= x <= D1]
ativos = [a for a in pn.index if a in b3.index] + [a for a in ("WDO1",) if a in b3.index]
print("\n=== 4) PRECOS B3 (WDO ja x10) ===")
print(b3.loc[sorted(set(ativos)), cols].round(2).to_string())

# 5) Snapshot gravado
s = c.table("snapshot_diario").select("Data,cota,retorno_dtd,retorno_mtd").gte("Data", D0).lte("Data", D1).order("Data").execute().data
print("\n=== 5) SNAPSHOT ===")
print(pd.DataFrame(s).to_string(index=False))
