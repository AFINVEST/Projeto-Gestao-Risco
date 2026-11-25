"""corrigir_dia_b3.py - rebaixa o pregao EXATO de uma data e corrige wides, base longa e retornos.
Uso: python corrigir_dia_b3.py 2026-11-23"""
import os, sys, shutil, datetime as dt
import numpy as np
import pandas as pd
import ScrapB3_v2 as S

d = dt.date.fromisoformat(sys.argv[1])
col = d.strftime("%Y-%m-%d")
df = S.buscar_dia_EXATO_sem_backfill(d)
if df is None or df.empty:
    print(f"[ERRO] B3 nao retornou o pregao {col}. Nada alterado."); sys.exit(1)

sp, sv = S.construir_colunas_wide_duplas(df.copy(), d)
TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def corrigir(path, path_csv, s_novo, rotulo):
    shutil.copy2(path, f"{path}.bak_fix_{TS}")
    w = pd.read_parquet(path).set_index("Assets")
    if col not in w.columns:
        print(f"[ERRO] coluna {col} nao existe em {path}"); sys.exit(1)
    antes = w[col].copy()
    for a, v in s_novo.items():
        if a in w.index:
            w.at[a, col] = S._fmt_ptbr_2dec(v)
    mud = [(a, antes.get(a), w.at[a, col]) for a in s_novo.index if a in w.index and str(antes.get(a)) != str(w.at[a, col])]
    out = w.reset_index()
    out.to_parquet(path, index=False)
    out.to_csv(path_csv, index=False, encoding="utf-8")
    print(f"[ok] {rotulo}: {len(mud)} ativos corrigidos em {col}")
    for a, x, y in mud[:12]:
        print(f"     {a:10} {x} -> {y}")

corrigir(S.PATH_PRECO, S.PATH_PRECO_CSV, sp, "precos de ajuste")
corrigir(S.PATH_VALOR, S.PATH_VALOR_CSV, sv, "valor de ajuste")
S.incrementar_base_ajuste(S.PATH_LONG, df)
print("[ok] base longa atualizada")

import atualizar_retornos_diarios as A
cli = A._get_client()
lg = A._calcular_log_returns(A._ler_precos_wide("df_preco"))
datas_b3 = sorted(pd.to_datetime([c for c in pd.read_parquet(S.PATH_PRECO).columns if c != "Assets"]))
i = datas_b3.index(pd.Timestamp(d))
alvo = set(datas_b3[i:i + 2])
lg = lg[lg["Data"].isin(alvo) & lg["Ativo"].isin(set(sp.index)) & lg["retorno"].notna()]
reg = [{"Data": r.Data.date().isoformat(), "Ativo": str(r.Ativo), "preco": float(r.preco),
        "retorno": float(r.retorno), "moeda": A._detecta_moeda(r.Ativo), "fonte": "df_preco"}
       for r in lg.itertuples()]
A._upsert_batch(cli, reg)
print(f"[ok] retornos_diarios_ativo: {len(reg)} linhas reenviadas ({', '.join(x.strftime('%d/%m') for x in sorted(alvo))})")
