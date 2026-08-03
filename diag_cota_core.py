"""
diag_cota_core.py  (v2 - ASCII only, foco em PnL por-ativo)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cota_portfolio_core import (
    load_b3_prices, load_ajustes, load_basefundos, load_lft_series,
    load_pl_series, analisar_dados_fundos2, computar_cota_serie,
)

def sep(t): print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)

# ---------------------------------------------------------------------------
sep("1) INVENTARIO — ativos em cada fonte")

df_b3 = load_b3_prices()
df_aj = load_ajustes()
bf    = load_basefundos()

ativos_b3 = set(df_b3["Assets"].astype(str))
ativos_aj = set(df_aj["Assets"].astype(str))
ativos_bf = set()
for fundo, dff in bf.items():
    if fundo.upper() == "TOTAL": continue
    idx = dff["Ativo"] if "Ativo" in dff.columns else dff.index
    for a in idx: ativos_bf.add(str(a))

print(f"df_preco  : {len(ativos_b3)} ativos")
print(f"df_ajuste : {len(ativos_aj)} ativos")
print(f"BaseFundos: {len(ativos_bf)} ativos  -> {sorted(ativos_bf)}")

falta_preco = sorted(ativos_bf - ativos_b3)
falta_ajuste = sorted(ativos_bf - ativos_aj)
print(f"\nBaseFundos NAO encontrados em df_preco : {falta_preco}")
print(f"BaseFundos NAO encontrados em df_ajuste: {falta_ajuste}")

# ---------------------------------------------------------------------------
sep("2) COBERTURA TEMPORAL POR ATIVO DO BASEFUNDOS")

def _cover(df, ativo):
    """Retorna (n_nonzero, primeira_data_nonzero, ultima_data_nonzero) para o ativo."""
    if ativo not in df["Assets"].values:
        return (0, None, None)
    row = df[df["Assets"] == ativo].iloc[0]
    cols = df.columns[1:]
    vals = pd.to_numeric(row[cols], errors="coerce")
    dts  = pd.to_datetime(cols, errors="coerce")
    mask = vals.abs() > 1e-9
    idx_ok = dts[mask.values]
    if len(idx_ok) == 0: return (0, None, None)
    return (len(idx_ok), idx_ok.min(), idx_ok.max())

print(f"{'Ativo':<15} | {'PRECO nonzero':>12} {'inicio':>12} {'fim':>12} | {'AJUSTE nonzero':>13} {'inicio':>12} {'fim':>12}")
print("-" * 100)
for a in sorted(ativos_bf):
    np_, p_i, p_f = _cover(df_b3, a)
    na, a_i, a_f = _cover(df_aj, a)
    p_i_s = p_i.date().isoformat() if p_i is not None else '-'
    p_f_s = p_f.date().isoformat() if p_f is not None else '-'
    a_i_s = a_i.date().isoformat() if a_i is not None else '-'
    a_f_s = a_f.date().isoformat() if a_f is not None else '-'
    print(f"{a:<15} | {np_:>12} {p_i_s:>12} {p_f_s:>12} | {na:>13} {a_i_s:>12} {a_f_s:>12}")

# ---------------------------------------------------------------------------
sep("3) analisar_dados_fundos2 -> perfil de PnL")

pl_series, _ = load_pl_series()
df_pnl, _, df_desp, fin_ntnb = analisar_dados_fundos2(
    soma_pl_sem_pesos=float(pl_series.iloc[-1]),
    df_b3_fechamento=df_b3,
    df_ajuste=df_aj,
    basefundos=bf,
)

print(f"df_pnl shape: {df_pnl.shape}   (linhas={df_pnl.shape[0]}, colunas={df_pnl.shape[1]})")
print(f"df_desp shape: {df_desp.shape}")
print(f"fin_ntnb: {len(fin_ntnb)} entradas")

pnl = (df_pnl.drop(columns="Total", errors="ignore")
              .apply(pd.to_numeric, errors="coerce")
              .sum(axis=0)
              .replace([np.inf, -np.inf], np.nan)
              .dropna())
pnl.index = pd.to_datetime(pnl.index); pnl = pnl.sort_index()

nao_zero = pnl[pnl.abs() > 0.01]
print(f"\nPnL total diario:")
print(f"  periodo: {pnl.index.min().date()} -> {pnl.index.max().date()}  ({len(pnl)} dias)")
print(f"  soma acum: R$ {pnl.sum():+,.2f}")
print(f"  min/max  : R$ {pnl.min():+,.2f} / R$ {pnl.max():+,.2f}")
print(f"  std daily: R$ {pnl.std():,.2f}")
print(f"  dias c/ |pnl|>R$0.01: {len(nao_zero)} de {len(pnl)}  ({len(nao_zero)/len(pnl)*100:.1f}%)")

# Top contribuintes
contrib = df_pnl.drop(columns="Total", errors="ignore").abs().sum(axis=1).sort_values(ascending=False)
print(f"\nTop 20 (Ativo - Fundo) por |pnl| acumulado:")
for k, v in contrib.head(20).items():
    n_dias = (df_pnl.loc[k].drop("Total", errors="ignore").abs() > 0.01).sum()
    print(f"  {str(k)[:55]:<55}  R$ {v:>15,.2f}   dias!=0: {n_dias}")

zeros = (contrib.abs() < 0.01).sum()
print(f"\nlinhas totalmente zeradas: {zeros} de {len(contrib)}")

# ---------------------------------------------------------------------------
sep("4) serie_ntnb")
if fin_ntnb:
    s = pd.Series(fin_ntnb).astype(float); s.index = pd.to_datetime(s.index); s = s.sort_index()
    print(f"periodo: {s.index.min().date()} -> {s.index.max().date()}  ({len(s)} pontos)")
    print(f"min/max: R$ {s.min():+,.2f} / R$ {s.max():+,.2f}   ultimo: R$ {s.iloc[-1]:+,.2f}")
else:
    print("vazio")

# ---------------------------------------------------------------------------
sep("5) Cota do core (metricas)")
res = computar_cota_serie(pct=0.01)
df = res["df"]
print(f"periodo: {res['data_ini'].date()} -> {res['data_fim'].date()}  ({len(df)} dias)")
print(f"capital medio: R$ {df['capital_dia'].mean():,.2f}")
print(f"pnl medio    : R$ {df['pnl'].mean():+,.2f}/dia   std: R$ {df['pnl'].std():,.2f}")
print(f"ganho_lft med: R$ {df['ganho_lft'].mean():,.2f}/dia")
print(f"custo medio  : R$ {df['custo_total'].mean():,.2f}/dia")
print(f"vol anual (ret_total): {df['ret_total'].std()*np.sqrt(252)*100:.2f}%")
print(f"vol anual (SO pnl/cap): {(df['pnl']/df['capital_ini_dia']).std()*np.sqrt(252)*100:.2f}%")
