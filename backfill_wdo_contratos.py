"""backfill_wdo_contratos.py - cria WDO_<venc> no historico dos wides a partir da base longa."""
import shutil, datetime as dt
import pandas as pd
import ScrapB3_v2 as S

TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
longa = pd.read_parquet(S.PATH_LONG)
longa["Data_Referencia"] = pd.to_datetime(longa["Data_Referencia"])

precos, valores = {}, {}
for d, df_dia in longa.groupby(longa["Data_Referencia"].dt.date):
    sp, sv = S.construir_colunas_wide_duplas(df_dia.copy(), d)
    sp = sp[sp.index.astype(str).str.startswith("WDO_")]
    sv = sv[sv.index.astype(str).str.startswith("WDO_")]
    if len(sp): precos[sp.name] = sp
    if len(sv): valores[sv.name] = sv

novo_p, novo_v = pd.DataFrame(precos), pd.DataFrame(valores)
print(f"[backfill] {novo_p.shape[1]} datas, contratos: {sorted(novo_p.index)}")

def gravar(path, path_csv, novo):
    shutil.copy2(path, f"{path}.bak_wdo_{TS}")
    w = pd.read_parquet(path)
    w = w[~w["Assets"].astype(str).str.startswith("WDO_")].set_index("Assets")
    cols = [c for c in w.columns if c in novo.columns]
    add = novo[cols].apply(lambda c: c.map(S._fmt_ptbr_2dec)).reindex(columns=w.columns)
    add.index.name = "Assets"
    out = pd.concat([w, add]).reset_index()
    out.to_parquet(path, index=False)
    out.to_csv(path_csv, index=False, encoding="utf-8")
    print(f"[ok] {path}: +{len(add)} linhas WDO_ ({len(cols)} datas de {len(w.columns)})")

gravar(S.PATH_PRECO, S.PATH_PRECO_CSV, novo_p)
gravar(S.PATH_VALOR, S.PATH_VALOR_CSV, novo_v)
