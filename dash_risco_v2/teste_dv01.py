"""
teste_dv01.py
=============

Valida o módulo dv01_dinamico.py contra os valores da planilha
di_curvab3vsimplicita.xlsx (aba PnL) em 29/07/2026.

Uso:
    python teste_dv01.py

Espera encontrar Dados/feriados_anbima.parquet e Dados/ni_ipca.parquet
no diretório de execução (ou passa --dados-dir).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# Adiciona pasta atual ao path pra importar dv01_dinamico
sys.path.insert(0, str(Path(__file__).parent))
import dv01_dinamico as dv


DATA_REF = pd.Timestamp("2026-07-29")
PROJECAO_MENSAL_PCT = 0.05   # ANBIMA em 28/07/26
# NI IPCA usado na planilha do usuário (fallback pra 15/12/2025 pois série
# NI local dele estava desatualizada). Aqui a gente força o mesmo valor pra
# validar que a fórmula bate exatamente com a planilha.
NI_REF_OVERRIDE_TESTE = 7403.29
# A planilha tem uma inconsistência: aba PnL usa PU calculado com
# data 29/07/26 mas o named range `rs_dap` aponta pra Base de dados!J5
# (calculado com data 28/10/25 e NI de set/25, projeção 0.67% mensal).
# Portanto rs_dap efetivo = 1.8447976818146532. Passa esse valor
# override pra bater o teste com o gabarito.
RS_DAP_OVERRIDE_TESTE = 1.8447976818146532

# Ground truth extraído da planilha (Aba PnL, valores computados)
# Formato: (ticker_novo, DU_planilha, taxa_pct, PU_esperado, contrato_R$_esperado, DV01_R$_esperado)
GABARITO_DI = [
    # ticker,    DU,   taxa,    PU,          DV01
    ("DI_F27",   107, 13.8730,  94_633.21,    3.5284),
    ("DI_J27",   167, 13.9080,  91_732.16,    5.3364),
    ("DI_N27",   230, 13.9340,  88_775.39,    7.1110),
    ("DI_F28",   358, 13.9760,  83_040.30,   10.3493),
    ("DI_N28",   482, 14.0960,  77_706.79,   13.0251),
    ("DI_F29",   606, 14.2230,  72_630.07,   15.2887),
    ("DI_N29",   730, 14.3330,  67_840.19,   17.1856),
    ("DI_F30",   855, 14.3850,  63_381.43,   18.7964),
    ("DI_F31",  1107, 14.4670,  55_236.46,   21.1929),
    ("DI_F32",  1359, 14.5590,  48_046.74,   22.6117),
    ("DI_F33",  1611, 14.6860,  41_644.58,   23.2062),
    ("DI_F34",  1862, 14.6180,  36_491.14,   23.5155),
    ("DI_F35",  2110, 14.6400,  31_854.97,   23.2566),
    ("DI_F37",  2612, 14.6620,  24_216.54,   21.8801),
]

GABARITO_DAP = [
    # ticker,    DU,   taxa,    PU_pts,      Contrato R$,  DV01 R$
    ("DAP_Q26",    13, 15.5240,  99_258.32,   183_749.46,    0.8176),
    ("DAP_K27",   198,  8.8100,  93_581.24,   173_239.91,   12.4652),
    ("DAP_Q28",   513,  8.5200,  84_666.62,   156_736.95,   29.2959),
    ("DAP_K29",   697,  8.3250,  80_157.62,   148_389.78,   37.7504),
    ("DAP_Q30",  1011,  8.3350,  72_528.87,   134_267.25,   49.5383),
    ("DAP_Q32",  1515,  8.2900,  61_952.40,   114_687.81,   63.4293),
    ("DAP_K33",  1702,  8.2500,  58_543.01,   108_376.27,   67.3596),
    ("DAP_K35",  2201,  7.0900,  54_975.50,   101_772.00,   82.6782),
    ("DAP_Q40",  3520,  7.8250,  34_911.31,    64_628.69,   83.3751),
    ("DAP_K45",  4707,  7.6750,  25_126.96,    46_515.65,   80.3379),
]

# Constantes IPCA esperadas
IPCA_ESPERADO = {
    'ni_ref': 7403.29,
    'du_desde': 10,
    'du_entre': 23,
    'ipca_pro_rata': 7404.89918351212,
    'rs_dap': 1.85122479587803,
}


def _fmt_pct_diff(esperado: float, obtido: float, tol_abs: float = 0.02) -> str:
    """Retorna string tipo '✓ 0.00%' ou '✗ 0.15%'."""
    if esperado == 0:
        return "n/a"
    diff_abs = abs(esperado - obtido)
    diff_pct = diff_abs / abs(esperado) * 100 if esperado else 0
    marker = "✓" if diff_abs < tol_abs else "✗"
    return f"{marker} {diff_pct:>5.2f}%"


def teste_calendario():
    print("=" * 70)
    print("TESTE 1 — Calendário (vencimentos e DU)")
    print("=" * 70)
    feriados = dv.load_feriados()
    print(f"Feriados carregados: {len(feriados)}")

    print(f"\n{'Ticker':<10} {'Vencimento':<12} {'DU planilha':>12} {'DU calc':>10} {'diff':>6}")
    ok = 0
    ko = 0
    for ticker, du_esp, *_ in GABARITO_DI + GABARITO_DAP:
        venc = dv.vencimento(ticker, feriados)
        du_calc = dv.networkdays(DATA_REF, venc, feriados)
        diff = du_calc - du_esp
        marker = "✓" if diff == 0 else "✗"
        print(f"{ticker:<10} {venc.date()!s:<12} {du_esp:>12} {du_calc:>10} {diff:>+5} {marker}")
        (ok if diff == 0 else ko).__add__ if False else None
        if diff == 0:
            ok += 1
        else:
            ko += 1
    print(f"\nResultado: {ok} ok / {ko} divergências")
    return ko == 0


def teste_ipca_pro_rata():
    print("\n" + "=" * 70)
    print("TESTE 2 — IPCA pro-rata (constantes da planilha em 29/07/26)")
    print("=" * 70)
    result = dv.ipca_pro_rata(DATA_REF, PROJECAO_MENSAL_PCT,
                              ni_ref_override=NI_REF_OVERRIDE_TESTE)
    print(f"{'Campo':<20} {'Planilha':>18} {'Calculado':>18} {'Status':>10}")
    ok = 0
    ko = 0
    for k, esp in IPCA_ESPERADO.items():
        obt = result[k]
        diff = abs(esp - obt)
        marker = "✓" if diff < 1e-4 else "✗"
        print(f"{k:<20} {esp:>18,.6f} {obt:>18,.6f} {marker:>10}")
        if diff < 1e-4:
            ok += 1
        else:
            ko += 1
    print(f"\nDatas: IPCA_ref_aj={result['ipca_ref_ajustado'].date()} | proximo_aj={result['proximo_ajustado'].date()}")
    print(f"Resultado: {ok} ok / {ko} divergências")
    return ko == 0


def teste_di():
    print("\n" + "=" * 70)
    print(f"TESTE 3 — DV01 DI (data_ref = {DATA_REF.date()})")
    print("=" * 70)
    print(f"{'Ticker':<8} {'Taxa%':>8} {'DU':>5} "
          f"{'PU esp':>12} {'PU calc':>12} {'ΔPU':>8}   "
          f"{'DV01 esp':>10} {'DV01 calc':>10} {'ΔDV01':>8}")
    ok = 0
    ko = 0
    for ticker, du_esp, taxa, pu_esp, dv01_esp in GABARITO_DI:
        res = dv.calcular_dv01(ticker, taxa, DATA_REF)
        pu_calc = res['pu']
        dv01_calc = res['dv01']
        d_pu   = _fmt_pct_diff(pu_esp, pu_calc, tol_abs=0.05)
        d_dv01 = _fmt_pct_diff(dv01_esp, dv01_calc, tol_abs=0.005)
        print(f"{ticker:<8} {taxa:>8.4f} {res['du']:>5} "
              f"{pu_esp:>12,.2f} {pu_calc:>12,.2f} {d_pu:>8}   "
              f"{dv01_esp:>10.4f} {dv01_calc:>10.4f} {d_dv01:>8}")
        if abs(pu_esp - pu_calc) < 0.05 and abs(dv01_esp - dv01_calc) < 0.005:
            ok += 1
        else:
            ko += 1
    print(f"\nResultado: {ok} ok / {ko} divergências")
    return ko == 0


def teste_dap():
    print("\n" + "=" * 70)
    print(f"TESTE 4 — DV01 DAP (data_ref = {DATA_REF.date()}, proj={PROJECAO_MENSAL_PCT}%)")
    print("=" * 70)
    print(f"  Usando rs_dap = {RS_DAP_OVERRIDE_TESTE:.10f} (o mesmo named range da planilha).")
    print(f"  OBS: coluna Contr esp da planilha usa rs_dap DIFERENTE (C43 recalculado com 29/07/26),")
    print(f"       que é uma inconsistência interna da planilha. Nosso Contr calc reflete rs_dap único.\n")
    print(f"{'Ticker':<9} {'Taxa%':>8} {'DU':>5} "
          f"{'PU pts esp':>12} {'PU pts calc':>12} "
          f"{'DV01 esp':>10} {'DV01 calc':>10} {'ΔDV01':>10}")
    ok = 0
    ko = 0
    for ticker, du_esp, taxa, pu_esp, contr_esp, dv01_esp in GABARITO_DAP:
        res = dv.calcular_dv01(ticker, taxa, DATA_REF,
                               projecao_mensal_pct=PROJECAO_MENSAL_PCT,
                               ni_ref_override=NI_REF_OVERRIDE_TESTE,
                               rs_dap_override=RS_DAP_OVERRIDE_TESTE)
        pu_calc    = res['pu_pontos']
        dv01_calc  = res['dv01']
        d_dv01 = _fmt_pct_diff(dv01_esp, dv01_calc, tol_abs=0.01)
        print(f"{ticker:<9} {taxa:>8.4f} {res['du']:>5} "
              f"{pu_esp:>12,.2f} {pu_calc:>12,.2f} "
              f"{dv01_esp:>10.4f} {dv01_calc:>10.4f} {d_dv01:>10}")
        if abs(dv01_esp - dv01_calc) < 0.01:
            ok += 1
        else:
            ko += 1
    print(f"\nResultado: {ok} ok / {ko} divergências")
    return ko == 0


def main():
    print(f"\nTeste de validação dv01_dinamico vs planilha\n")
    r1 = teste_calendario()
    r2 = teste_ipca_pro_rata()
    r3 = teste_di()
    r4 = teste_dap()
    print("\n" + "=" * 70)
    total_ok = sum([r1, r2, r3, r4])
    if total_ok == 4:
        print("*** TODOS OS TESTES PASSARAM ***")
    else:
        print(f"*** {4 - total_ok} DE 4 GRUPOS COM DIVERGÊNCIAS — revisar ***")
    print("=" * 70)


if __name__ == "__main__":
    main()
