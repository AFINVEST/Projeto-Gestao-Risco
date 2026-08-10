"""
repopular_retornos_ago.py — repopula retornos_diarios_ativo pros dias Aug 03-06
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np
from supabase import create_client


def _parse_ptbr(v):
    if v is None: return None
    try:
        if pd.isna(v): return None
    except (TypeError, ValueError): pass
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    s = str(v).strip().replace("R$", "").replace(" ", "")
    if not s or s in ("-", "--"): return None
    s = s.replace(".", "").replace(",", ".")
    try: return float(s)
    except (TypeError, ValueError): return None


def _detecta_moeda(ativo):
    return "USD" if str(ativo).upper() == "TREASURY" else "BRL"


def main(alvos=("2026-08-03","2026-08-04","2026-08-05","2026-08-06")):
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ERRO: SUPABASE_URL / SUPABASE_KEY nao definidos.", file=sys.stderr); sys.exit(1)
    client = create_client(url, key)

    df = pd.read_parquet("Dados/df_preco_de_ajuste_atual_completo.parquet").set_index("Assets")
    cols_data = sorted([c for c in df.columns], key=pd.to_datetime)
    print(f"[repop] parquet lido: {len(df)} ativos, {len(cols_data)} datas")

    regs = []
    for ativo in df.index:
        for d_alvo in alvos:
            if d_alvo not in cols_data:
                continue
            i = cols_data.index(d_alvo)
            if i == 0: continue
            d_prev = cols_data[i-1]
            p_hoje = _parse_ptbr(df.at[ativo, d_alvo])
            p_ont  = _parse_ptbr(df.at[ativo, d_prev])
            if p_hoje and p_ont and p_ont > 0:
                ret = float(np.log(p_hoje / p_ont))
                regs.append({
                    "Data": d_alvo, "Ativo": str(ativo),
                    "preco": p_hoje, "retorno": ret,
                    "moeda": _detecta_moeda(ativo), "fonte": "df_preco",
                })

    print(f"[repop] Upserting {len(regs)} pares...")
    total = 0
    for i in range(0, len(regs), 500):
        lote = regs[i:i+500]
        client.table("retornos_diarios_ativo").upsert(lote, on_conflict="Data,Ativo").execute()
        total += len(lote)
        print(f"  [upsert] {total}/{len(regs)}")
    print(f"[repop] OK - {total} linhas em retornos_diarios_ativo")


if __name__ == "__main__":
    main()
