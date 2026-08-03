"""
dv01_dinamico.py
================

Cálculo dinâmico de DV01 para contratos DI1 (juros nominais) e DAP
(juros reais IPCA+), substituindo o input estático que hoje vem do
AF_Trading.xlsm.

Fórmulas replicam exatamente a aba PnL da planilha
di_curvab3vsimplicita.xlsx:

    DI:  PU  = 100_000 / (1 + taxa/100)^(DU/252)
         PU' = 100_000 / (1 + (taxa+0.01)/100)^(DU/252)
         DV01 = PU - PU'    (em R$; nominal já é R$)

    DAP: PU_pts  = idem DI (em "pontos", contrato nominal = 100_000 pts)
         PU'_pts = idem
         DV01_pts = PU_pts - PU'_pts
         IPCA_pro_rata = NI_ref × (1 + projeção%/100)^(DU_desde / DU_entre)
         rs_dap        = 0.00025 × IPCA_pro_rata     [R$ por ponto]
         DV01_DAP      = DV01_pts × rs_dap           [R$]

Onde:
    NI_ref     = NI IPCA do mês (M-1) do IPCA_ref
    IPCA_ref   = 15 do mês corrente se today>15, senão 15 do mês anterior
    DU_desde   = dias úteis de IPCA_ref até today
    DU_entre   = dias úteis de IPCA_ref até próximo IPCA_ref
    projeção   = ANBIMA (mensal, em %, ex: 0.05 significa 0.05%)

Naming dos tickers:
    DI:  DI_F<YY> (jan), DI_J<YY> (abr), DI_N<YY> (jul), DI_V<YY> (out)
    DAP: DAP_K<YY> (mai) ou DAP_Q<YY> (ago)

Requer:
    Dados/feriados_anbima.parquet   (extraído da planilha, entregue na Fase 1)
    Dados/ni_ipca.parquet           (idem)
"""
from __future__ import annotations
import re
from pathlib import Path
from datetime import date, datetime
from functools import lru_cache
import pandas as pd
import numpy as np


# =====================================================================
# Constantes
# =====================================================================

NOMINAL = 100_000.0            # valor nominal do contrato DI/DAP no vencimento
RS_POR_PONTO_DAP = 0.00025      # R$ por ponto para DAP
DIAS_UTEIS_ANO = 252
BPS_SHIFT = 0.01                # 1 bp = 0.01% na taxa (para diferença finita)

# Letra do contrato → mês (convenção B3)
LETRA_MES = {
    'F': 1,  'G': 2,  'H': 3,  'J': 4,  'K': 5,  'M': 6,
    'N': 7,  'Q': 8,  'U': 9,  'V': 10, 'X': 11, 'Z': 12,
}
MES_LETRA = {v: k for k, v in LETRA_MES.items()}

# Vencimentos padrão que o sistema conhece
LETRAS_DI  = ['F', 'J', 'N', 'V']     # jan, abr, jul, out
ANOS_DI    = list(range(26, 41))       # 2026..2040
ANOS_DAP   = list(range(26, 46))       # 2026..2045 (K/Q dependendo da paridade)


def _letra_dap(ano_yy: int) -> str:
    """Regra por paridade: par=Q (agosto), ímpar=K (maio)."""
    return 'Q' if (ano_yy % 2) == 0 else 'K'


# Tabela de vencimentos padrão (75 contratos)
VENCIMENTOS_PADRAO_DI  = [f"DI_{L}{yy}"  for yy in ANOS_DI  for L in LETRAS_DI]
VENCIMENTOS_PADRAO_DAP = [f"DAP_{_letra_dap(yy)}{yy}" for yy in ANOS_DAP]
VENCIMENTOS_PADRAO     = VENCIMENTOS_PADRAO_DI + VENCIMENTOS_PADRAO_DAP


# =====================================================================
# Loader de dados de apoio
# =====================================================================

@lru_cache(maxsize=1)
def load_feriados(caminho: str = "Dados/feriados_anbima.parquet") -> frozenset:
    """Carrega feriados Anbima como set de pd.Timestamp normalizados."""
    df = pd.read_parquet(caminho)
    dates = pd.to_datetime(df["Data"]).dt.normalize()
    return frozenset(dates.tolist())


@lru_cache(maxsize=1)
def load_ni_ipca(caminho: str = "Dados/ni_ipca.parquet") -> pd.DataFrame:
    """Carrega série NI IPCA (colunas: Data_Ref [15 do mês], NI)."""
    df = pd.read_parquet(caminho)
    df["Data_Ref"] = pd.to_datetime(df["Data_Ref"]).dt.normalize()
    return df.set_index("Data_Ref").sort_index()


# =====================================================================
# Utilitários de calendário
# =====================================================================

def workday(data: pd.Timestamp, offset: int, feriados: frozenset) -> pd.Timestamp:
    """Equivalente ao WORKDAY do Excel: pula `offset` dias úteis a partir
    de `data` (excluindo weekends e feriados)."""
    d = pd.Timestamp(data).normalize()
    step = 1 if offset >= 0 else -1
    restam = abs(offset)
    while restam > 0:
        d = d + pd.Timedelta(days=step)
        if d.weekday() < 5 and d not in feriados:
            restam -= 1
    return d


def networkdays(inicio: pd.Timestamp, fim: pd.Timestamp, feriados: frozenset) -> int:
    """Equivalente ao NETWORKDAYS do Excel:
    conta dias úteis entre inicio e fim, inclusive. Excel retorna a
    diferença - 1 quando usado como em `=NETWORKDAYS(A, B)-1`, que é o
    padrão dessa planilha (DU strictos entre as duas datas).

    Aqui devolvemos JÁ o valor "- 1" (dias úteis exclusive do início).
    Isso bate exatamente com a coluna E da aba PnL.
    """
    a = pd.Timestamp(inicio).normalize()
    b = pd.Timestamp(fim).normalize()
    if b < a:
        return -networkdays(b, a, feriados)
    # gera todos os dias entre a e b (inclusive)
    dias = pd.bdate_range(a, b, freq='C', weekmask='1111100')  # bday
    du = sum(1 for d in dias if d not in feriados)
    # NETWORKDAYS(a,b) inclui os dois extremos se forem business days;
    # a planilha usa "- 1" para excluir o inicio (data_ref)
    return du - 1


# =====================================================================
# Parser de ticker
# =====================================================================

_RE_DI  = re.compile(r"^DI_([FGHJKMNQUVXZ])(\d{2})$")
_RE_DAP = re.compile(r"^DAP_([KQ])(\d{2})$")


def parse_ticker(ticker: str) -> dict:
    """Retorna dict com: tipo, letra, mes, ano_yy, ano_yyyy.
    Levanta ValueError se ticker desconhecido."""
    m = _RE_DI.match(ticker)
    if m:
        letra, yy = m.group(1), int(m.group(2))
        return {
            'tipo': 'DI',
            'letra': letra,
            'mes': LETRA_MES[letra],
            'ano_yy': yy,
            'ano_yyyy': 2000 + yy,
        }
    m = _RE_DAP.match(ticker)
    if m:
        letra, yy = m.group(1), int(m.group(2))
        return {
            'tipo': 'DAP',
            'letra': letra,
            'mes': LETRA_MES[letra],
            'ano_yy': yy,
            'ano_yyyy': 2000 + yy,
        }
    raise ValueError(f"Ticker não reconhecido: {ticker}")


def vencimento(ticker: str, feriados: frozenset | None = None) -> pd.Timestamp:
    """Data de vencimento efetiva do contrato (já ajustada por feriados).

    DI:  1º dia útil do mês (WORKDAY(dia_1 - 1, 1, feriados))
    DAP: 15 do mês ajustado (WORKDAY(dia_14, 1, feriados))
    """
    if feriados is None:
        feriados = load_feriados()
    info = parse_ticker(ticker)
    if info['tipo'] == 'DI':
        base = pd.Timestamp(year=info['ano_yyyy'], month=info['mes'], day=1)
        # WORKDAY(dia_1 - 1, 1, feriados) = próximo DU >= dia_1
        return workday(base - pd.Timedelta(days=1), 1, feriados)
    else:  # DAP
        base = pd.Timestamp(year=info['ano_yyyy'], month=info['mes'], day=15)
        # WORKDAY(dia_14, 1, feriados) = próximo DU >= dia_15
        return workday(base - pd.Timedelta(days=1), 1, feriados)


def du_ate_vencimento(ticker: str, data_ref: pd.Timestamp, feriados: frozenset | None = None) -> int:
    """Dias úteis de data_ref (exclusive) até o vencimento (inclusive)."""
    if feriados is None:
        feriados = load_feriados()
    venc = vencimento(ticker, feriados)
    return networkdays(pd.Timestamp(data_ref).normalize(), venc, feriados)


# =====================================================================
# PU e DV01 — DI
# =====================================================================

def pu_di(taxa_pct: float, du: int) -> float:
    """PU do contrato DI dado a taxa em % e DU."""
    return NOMINAL / (1.0 + taxa_pct / 100.0) ** (du / DIAS_UTEIS_ANO)


def dv01_di(taxa_pct: float, du: int) -> float:
    """DV01 de um DI (diferença finita, 1 bp de shock).
    Retorna R$ (o nominal já está em R$)."""
    pu   = pu_di(taxa_pct, du)
    pu_s = pu_di(taxa_pct + BPS_SHIFT, du)
    return pu - pu_s


# =====================================================================
# IPCA pro-rata (para DAP)
# =====================================================================

def _ipca_ref_para_data(data_ref: pd.Timestamp) -> pd.Timestamp:
    """Retorna a data 15/mm/yyyy que é a referência de IPCA para a data.
    Se dia > 15, usa 15 do mês corrente. Senão, 15 do mês anterior."""
    d = pd.Timestamp(data_ref).normalize()
    if d.day > 15:
        return pd.Timestamp(year=d.year, month=d.month, day=15)
    else:
        # dia <= 15 → mês anterior
        mes = d.month - 1
        ano = d.year
        if mes < 1:
            mes = 12
            ano -= 1
        return pd.Timestamp(year=ano, month=mes, day=15)


def _proxima_ipca_ref(ipca_ref: pd.Timestamp) -> pd.Timestamp:
    """Retorna o 15 do mês seguinte à IPCA_ref (não ajustado por feriado)."""
    d = pd.Timestamp(ipca_ref)
    mes = d.month + 1
    ano = d.year
    if mes > 12:
        mes = 1
        ano += 1
    return pd.Timestamp(year=ano, month=mes, day=15)


def ipca_pro_rata(data_ref: pd.Timestamp,
                  projecao_mensal_pct: float,
                  ni_ipca: pd.DataFrame | None = None,
                  feriados: frozenset | None = None,
                  ni_ref_override: float | None = None) -> dict:
    """Replica o cálculo da aba PnL linhas 34-42.

    Retorna dict com: ipca_ref, ipca_ref_ajustado, proximo_ajustado,
    du_desde, du_entre, ni_ref, ipca_pro_rata, rs_dap.

    Se ni_ref_override for passado, ignora o lookup e usa o valor
    fornecido (útil pra testes e backtests históricos).
    """
    if ni_ipca is None:
        ni_ipca = load_ni_ipca()
    if feriados is None:
        feriados = load_feriados()

    ipca_ref = _ipca_ref_para_data(data_ref)
    proximo  = _proxima_ipca_ref(ipca_ref)

    # Ajustes por dia útil (workday(-1, 1))
    ipca_ref_aj = workday(ipca_ref - pd.Timedelta(days=1), 1, feriados)
    proximo_aj  = workday(proximo   - pd.Timedelta(days=1), 1, feriados)

    du_desde = networkdays(ipca_ref_aj, pd.Timestamp(data_ref).normalize(), feriados)
    du_entre = networkdays(ipca_ref_aj, proximo_aj, feriados)

    # NI_ref = NI do mês (M-1) do IPCA_ref
    if ni_ref_override is not None:
        ni_ref = float(ni_ref_override)
        data_ni_efetiva = None
    else:
        mes_ni = ipca_ref.month - 1
        ano_ni = ipca_ref.year
        if mes_ni < 1:
            mes_ni = 12
            ano_ni -= 1
        data_ni = pd.Timestamp(year=ano_ni, month=mes_ni, day=15)
        if data_ni in ni_ipca.index:
            ni_ref = float(ni_ipca.loc[data_ni, 'NI'])
            data_ni_efetiva = data_ni
        else:
            # Fallback: usa último NI disponível com aviso
            import warnings
            ultimo = ni_ipca.index.max()
            ni_ref = float(ni_ipca.loc[ultimo, 'NI'])
            data_ni_efetiva = ultimo
            warnings.warn(
                f"NI IPCA não disponível para {data_ni.date()}. "
                f"Usando último disponível: {ultimo.date()} (NI={ni_ref:.4f}). "
                f"Rode atualiza_ni_ipca.py para pegar dados novos do IPEADATA."
            )

    # IPCA pro-rata (projeção mensal em % → decimal)
    proj_dec = projecao_mensal_pct / 100.0
    if du_entre <= 0:
        ipca_pr = ni_ref
    else:
        ipca_pr = ni_ref * (1.0 + proj_dec) ** (du_desde / du_entre)

    rs_dap = RS_POR_PONTO_DAP * ipca_pr

    return {
        'ipca_ref': ipca_ref,
        'ipca_ref_ajustado': ipca_ref_aj,
        'proximo_ajustado': proximo_aj,
        'du_desde': du_desde,
        'du_entre': du_entre,
        'ni_ref': ni_ref,
        'ipca_pro_rata': ipca_pr,
        'rs_dap': rs_dap,
    }


# =====================================================================
# PU e DV01 — DAP
# =====================================================================

def pu_dap_pontos(taxa_pct: float, du: int) -> float:
    """PU em pontos (mesma fórmula do DI, nominal = 100_000)."""
    return NOMINAL / (1.0 + taxa_pct / 100.0) ** (du / DIAS_UTEIS_ANO)


def dv01_dap(taxa_pct: float, du: int, rs_dap: float) -> float:
    """DV01 do DAP em R$. Precisa do rs_dap (calculado por ipca_pro_rata)."""
    pu_p   = pu_dap_pontos(taxa_pct, du)
    pu_p_s = pu_dap_pontos(taxa_pct + BPS_SHIFT, du)
    dv01_pts = pu_p - pu_p_s
    return dv01_pts * rs_dap


def contrato_dap_reais(taxa_pct: float, du: int, rs_dap: float) -> float:
    """Valor do contrato DAP em R$ (PU × R$/ponto)."""
    return pu_dap_pontos(taxa_pct, du) * rs_dap


# =====================================================================
# Dispatcher genérico
# =====================================================================

def calcular_dv01(ticker: str,
                  taxa_pct: float,
                  data_ref: pd.Timestamp | date | str,
                  projecao_mensal_pct: float | None = None,
                  ni_ipca: pd.DataFrame | None = None,
                  feriados: frozenset | None = None,
                  ni_ref_override: float | None = None,
                  rs_dap_override: float | None = None) -> dict:
    """Ponto de entrada único. Aceita DI ou DAP e retorna todos os
    valores relevantes num dict.

    rs_dap_override: se passado, pula o cálculo de IPCA pro-rata e usa
    este valor diretamente (útil pra backtest histórico ou pra bater
    contra planilha com rs_dap fixo)."""
    if isinstance(data_ref, str):
        data_ref = pd.Timestamp(data_ref)
    data_ref = pd.Timestamp(data_ref).normalize()

    if feriados is None:
        feriados = load_feriados()

    info = parse_ticker(ticker)
    venc = vencimento(ticker, feriados)
    du   = networkdays(data_ref, venc, feriados)

    out = {
        'ticker': ticker,
        'tipo': info['tipo'],
        'data_ref': data_ref,
        'vencimento': venc,
        'du': du,
        'taxa_pct': taxa_pct,
    }

    if info['tipo'] == 'DI':
        out['pu']   = pu_di(taxa_pct, du)
        out['dv01'] = dv01_di(taxa_pct, du)
        out['contrato_reais'] = out['pu']
    else:  # DAP
        pu_pts = pu_dap_pontos(taxa_pct, du)
        if rs_dap_override is not None:
            rs_dap_val = float(rs_dap_override)
            out.update({
                'pu_pontos': pu_pts,
                'rs_dap': rs_dap_val,
                'ni_ref': None,
                'ipca_pro_rata': None,
            })
        else:
            if projecao_mensal_pct is None:
                raise ValueError("projecao_mensal_pct é obrigatório para DAP")
            if ni_ipca is None:
                ni_ipca = load_ni_ipca()
            ipca = ipca_pro_rata(data_ref, projecao_mensal_pct, ni_ipca, feriados,
                                 ni_ref_override=ni_ref_override)
            rs_dap_val = ipca['rs_dap']
            out.update({
                'pu_pontos': pu_pts,
                'rs_dap': rs_dap_val,
                'ni_ref': ipca['ni_ref'],
                'ipca_pro_rata': ipca['ipca_pro_rata'],
            })
        out['contrato_reais'] = pu_pts * rs_dap_val
        out['dv01'] = dv01_dap(taxa_pct, du, rs_dap_val)

    return out


# =====================================================================
# CLI (uso rápido no terminal)
# =====================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Calcula DV01 dinâmico de DI/DAP.")
    ap.add_argument("ticker", help="Ex: DI_F29 ou DAP_K27")
    ap.add_argument("taxa",   type=float, help="Taxa em %% (ex: 14.22)")
    ap.add_argument("--data", default=None, help="Data ref YYYY-MM-DD (default: hoje)")
    ap.add_argument("--proj", type=float, default=0.05, help="Projeção IPCA mensal em %% (default: 0.05)")
    args = ap.parse_args()
    data = pd.Timestamp(args.data) if args.data else pd.Timestamp.today().normalize()
    res = calcular_dv01(args.ticker, args.taxa, data, projecao_mensal_pct=args.proj)
    for k, v in res.items():
        print(f"  {k}: {v}")
