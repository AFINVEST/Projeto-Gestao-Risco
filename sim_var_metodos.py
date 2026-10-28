"""sim_var_metodos.py - compara metodos de VaR em datas-chave"""
import numpy as np, pandas as pd
from risco_carteira_core import (carregar_posicoes_atuais, carregar_precos_atuais,
    carregar_retornos_historicos, calcular_var_carteira, _quantile_ewma)
from cota_portfolio_core import load_basefundos

bf = load_basefundos()
for d in ["2026-10-23", "2026-10-26"]:
    ref = pd.Timestamp(d)
    qty = carregar_posicoes_atuais(ref, basefundos=bf)
    px  = carregar_precos_atuais(ref)
    ret = carregar_retornos_historicos(ref, ativos=list(qty), janela_dias=756)
    res = calcular_var_carteira(ret, qty, px)
    port, mv = res["port_ret_series"].sort_index(), res["mv_total"]

    hist   = -np.quantile(port.values, 0.05) * mv
    brw99  = -_quantile_ewma(port, 0.05, 0.99) * mv
    brw94  = -_quantile_ewma(port, 0.05, 0.94) * mv
    # EWMA parametrico (RiskMetrics): sigma recursiva, VaR = 1.645 * sigma * MV
    var_ = port.iloc[:20].var()
    for r in port.values:
        var_ = 0.94 * var_ + 0.06 * r * r
    param94 = 1.645 * np.sqrt(var_) * mv

    print(f"\n{d}  ultimo ret carteira={port.iloc[-1]*100:+.2f}%  MV=R$ {mv:,.0f}")
    print(f"  HIST 3y            : R$ {hist:>10,.0f}")
    print(f"  EWMA quantil l=0.99: R$ {brw99:>10,.0f}   (atual)")
    print(f"  EWMA quantil l=0.94: R$ {brw94:>10,.0f}")
    print(f"  EWMA parametr l=0.94: R$ {param94:>10,.0f}   (RiskMetrics)")
