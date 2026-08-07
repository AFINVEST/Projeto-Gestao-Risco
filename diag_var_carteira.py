"""
diag_var_carteira.py
====================

Diagnostico completo do VaR de carteira. Responde:

1) P&L direto (sum((PU_t - PU_{t-1}) * qty_atual)) bate com meu metodo
   (log_return * peso * MV)?
2) Por que VaR HIST ~= VaR EWMA? Onde estao as piores observacoes na
   linha do tempo?
3) Qual o impacto da janela historica?

USO:
    python diag_var_carteira.py --data 2026-07-31 --pl 653744640
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from risco_carteira_core import (
    carregar_posicoes_atuais,
    carregar_precos_atuais,
    carregar_retornos_historicos,
    _quantile_ewma,
    _parse_ptbr,
)


def _load_pu_wide_hist(data_ref, ativos, path="Dados/df_preco_de_ajuste_atual_completo.parquet"):
    """Retorna DataFrame wide (Data x Ativo) com PU historico."""
    df = pd.read_parquet(path)
    if "Assets" not in df.columns:
        return pd.DataFrame()
    cols_data = [c for c in df.columns if c != "Assets"]
    df_long = df.melt(id_vars="Assets", value_vars=cols_data, var_name="Data", value_name="pu_raw")
    df_long["pu"] = df_long["pu_raw"].map(_parse_ptbr)
    df_long["Data"] = pd.to_datetime(df_long["Data"], errors="coerce")
    df_long = df_long.dropna(subset=["Data", "pu"])
    df_long = df_long[df_long["pu"] > 0]
    df_long = df_long.rename(columns={"Assets": "Ativo"})
    df_long = df_long[df_long["Ativo"].isin(ativos)]
    df_long = df_long[df_long["Data"] <= pd.Timestamp(data_ref)]
    df_long = df_long.drop_duplicates(subset=["Data", "Ativo"], keep="first")
    wide = df_long.pivot(index="Data", columns="Ativo", values="pu").sort_index()
    return wide


def sep(t): print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def run(data_ref, pl_total):
    sep("1) POSICOES E PRECOS ATUAIS")

    qty_dict = carregar_posicoes_atuais(data_ref)
    precos_dict = carregar_precos_atuais(data_ref)
    ativos = sorted(qty_dict.keys())

    print(f"Posicoes em {data_ref.date()}:")
    mv_total = 0.0
    for a in ativos:
        pu = precos_dict.get(a)
        qty = qty_dict[a]
        mv = qty * pu if pu else 0.0
        mv_total += mv
        print(f"  {a}: qty={qty:.1f}  PU={pu:,.2f}  MV=R$ {mv:,.2f}")
    print(f"MV total: R$ {mv_total:,.2f}")

    sep("2) SERIE HISTORICA DE PU E LOG-RETURN")

    pu_wide = _load_pu_wide_hist(data_ref, ativos)
    print(f"pu_wide.shape = {pu_wide.shape}  (dias x ativos)")
    print(f"periodo: {pu_wide.index.min().date()} -> {pu_wide.index.max().date()}")
    print(pu_wide.tail(5))

    ret_wide = np.log(pu_wide / pu_wide.shift(1))
    ret_wide = ret_wide.dropna(how="any")   # so dias com ambos os ativos
    print(f"\nret_wide.shape = {ret_wide.shape}  (dias com log-return valido para AMBOS)")
    print(f"periodo com dados: {ret_wide.index.min().date()} -> {ret_wide.index.max().date()}")

    sep("3) P&L DIRETO — sum((PU_t - PU_{t-1}) * qty_atual)")

    delta_pu = pu_wide.diff().dropna(how="any")
    qty_series = pd.Series(qty_dict).reindex(ativos)
    pnl_direto = (delta_pu * qty_series).sum(axis=1)
    print(f"pnl_direto.shape = {pnl_direto.shape}")
    print(f"stats: mean={pnl_direto.mean():+,.2f}  std={pnl_direto.std():,.2f}")
    print(f"       min ={pnl_direto.min():+,.2f}  max ={pnl_direto.max():+,.2f}")
    print(f"       quantile 5%: R$ {pnl_direto.quantile(0.05):+,.2f}")
    print(f"       quantile 1%: R$ {pnl_direto.quantile(0.01):+,.2f}")

    sep("4) P&L VIA LOG-RETURN (metodo do risco_carteira_core)")

    pesos = pd.Series({a: qty_dict[a] * precos_dict[a] / mv_total for a in ativos})
    port_ret = (ret_wide * pesos).sum(axis=1)
    pnl_via_ret = port_ret * mv_total

    print(f"port_ret.shape = {port_ret.shape}")
    print(f"stats port_ret: mean={port_ret.mean()*100:+.4f}%  std={port_ret.std()*100:.4f}%")
    print(f"stats pnl_via_ret: mean={pnl_via_ret.mean():+,.2f}  std={pnl_via_ret.std():,.2f}")
    print(f"       min ={pnl_via_ret.min():+,.2f}  max ={pnl_via_ret.max():+,.2f}")
    print(f"       quantile 5%: R$ {pnl_via_ret.quantile(0.05):+,.2f}")

    sep("5) COMPARACAO — pnl_direto vs pnl_via_ret")

    idx_comum = pnl_direto.index.intersection(pnl_via_ret.index)
    dif = (pnl_direto.reindex(idx_comum) - pnl_via_ret.reindex(idx_comum))
    dif_pct = (dif / pnl_direto.reindex(idx_comum).abs()) * 100
    print(f"n dias comparados: {len(idx_comum)}")
    print(f"dif absoluta (R$): mean={dif.mean():+,.2f}  std={dif.std():,.2f}  max_abs={dif.abs().max():,.2f}")
    print(f"corr(direto, via_ret): {pnl_direto.reindex(idx_comum).corr(pnl_via_ret.reindex(idx_comum)):.6f}")
    print(f"\nSe correlacao > 0.99 e dif_std pequeno, metodo esta OK.")

    sep("6) VaR nas 3 janelas — HIST vs EWMA (lambda=0.99)")

    print(f"{'Janela':<15} {'n_dias':<8} {'VaR HIST 5%':<18} {'VaR EWMA 5%':<18} {'CVaR HIST':<15}")
    print("-" * 78)
    for nome, dias in [("1 ano", 252), ("3 anos", 756), ("Total", 10_000)]:
        janela = pnl_direto.tail(min(dias, len(pnl_direto)))
        if len(janela) < 30:
            print(f"{nome:<15} {len(janela):<8} <insuficiente>")
            continue
        var_hist = janela.quantile(0.05)
        cvar_hist = janela[janela <= var_hist].mean()
        var_ewma = _quantile_ewma(janela, 0.05, 0.99)
        print(f"{nome:<15} {len(janela):<8} R$ {var_hist:>13,.0f}   R$ {var_ewma:>13,.0f}   R$ {cvar_hist:>10,.0f}")

    sep("7) DISTRIBUICAO DAS PIORES OBSERVACOES — quando ocorreram?")

    piores = pnl_direto.nsmallest(20)
    print(f"20 piores dias de P&L:")
    for d, v in piores.items():
        idade_dias = (pd.Timestamp(data_ref) - d).days
        print(f"  {d.date()}  R$ {v:+,.2f}   ({idade_dias:>3}d atras)")

    # Concentracao no tempo
    q_5pct = pnl_direto.quantile(0.05)
    piores_5pct = pnl_direto[pnl_direto <= q_5pct]
    if len(piores_5pct) > 0:
        idades = [(pd.Timestamp(data_ref) - d).days for d in piores_5pct.index]
        print(f"\nDos {len(piores_5pct)} dias no tail 5%:")
        print(f"  media de idade: {np.mean(idades):.0f} dias atras")
        print(f"  mediana:        {np.median(idades):.0f} dias atras")
        print(f"  ultimos 90 dias: {sum(1 for i in idades if i <= 90)} obs")
        print(f"  ultimos 180d:    {sum(1 for i in idades if i <= 180)} obs")
        print(f"  ultimos 365d:    {sum(1 for i in idades if i <= 365)} obs")

    sep("8) LIMITE E CONSUMO")

    limite_1bp = pl_total * 0.0001
    print(f"PL total: R$ {pl_total:,.2f}")
    print(f"Limite (1 bp): R$ {limite_1bp:,.2f}")
    print()
    for nome, dias in [("3 anos (HIST)", 756), ("3 anos (EWMA lam=0.99)", 756)]:
        janela = pnl_direto.tail(min(dias, len(pnl_direto)))
        if len(janela) < 30:
            continue
        if "HIST" in nome:
            v = abs(janela.quantile(0.05))
        else:
            v = abs(_quantile_ewma(janela, 0.05, 0.99))
        consumo = v / limite_1bp * 100
        print(f"{nome:<25} VaR R$ {v:>13,.0f}   consumo {consumo:5.1f}% do limite")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--pl", type=float, required=True)
    args = ap.parse_args()
    data_ref = pd.Timestamp(args.data) if args.data else pd.Timestamp.today().normalize()
    run(data_ref, args.pl)
