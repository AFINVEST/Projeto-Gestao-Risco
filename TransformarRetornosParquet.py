"""
TransformarRetornosParquet.py  (v2 — aplica alias de naming antigo -> novo)
============================================================================

MUDANÇAS EM RELAÇÃO AO ORIGINAL:
  - Ao converter BaseFundos/*.csv -> BaseFundos/*.parquet, aplica alias
    antigo -> novo na coluna 'Ativo' (DAP30 -> DAP_Q30, DI_27 -> DI_F27, etc).
  - Idempotente: nomes que já estão no naming novo passam direto.
  - Restante do ETL (BBG excel, df_divone, LFT) permanece IDÊNTICO ao original.
"""
import pandas as pd
import numpy as np
import os
import re
import glob


# ─────────────────────────────────────────────────────────────────────────────
# Alias antigo -> novo (mesma lógica de cota_portfolio_core e migrar_basefundos)
# ─────────────────────────────────────────────────────────────────────────────
_RE_DAP_NOVO   = re.compile(r"^DAP_[KQ]\d{2}$")
_RE_DAP_ANTIGO = re.compile(r"^DAP(\d{2})$")
_RE_DI_NOVO    = re.compile(r"^DI_[FJNV]\d{2}$")
_RE_DI_ANTIGO  = re.compile(r"^DI_(\d{2})$")


def _alias_ativo_novo(ativo) -> str:
    s = str(ativo).strip()
    if _RE_DAP_NOVO.match(s) or _RE_DI_NOVO.match(s):
        return s
    m = _RE_DAP_ANTIGO.match(s)
    if m:
        yy = int(m.group(1))
        letra = "Q" if yy % 2 == 0 else "K"
        return f"DAP_{letra}{yy:02d}"
    m = _RE_DI_ANTIGO.match(s)
    if m:
        return f"DI_F{m.group(1)}"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 1) CSVs top-level -> parquet (comportamento original)
# ─────────────────────────────────────────────────────────────────────────────
files = glob.glob('*.csv')
for file in files:
    df = pd.read_csv(file)
    df.to_parquet(file.replace('.csv', '.parquet'), index=False)
    os.remove(file)
    print(f'{file} convertido para parquet')
print('Conversão finalizada (top-level)')


# ─────────────────────────────────────────────────────────────────────────────
# 2) CSVs de BaseFundos/ -> parquet — COM APLICAÇÃO DE ALIAS
# ─────────────────────────────────────────────────────────────────────────────
files = glob.glob('BaseFundos/*.csv')
for file in files:
    df = pd.read_csv(file)

    # Aplica alias na coluna 'Ativo' (se existir)
    col_ativo = None
    for cand in ("Ativo", "ativo", "ATIVO"):
        if cand in df.columns:
            col_ativo = cand
            break
    if col_ativo is not None:
        antigos = df[col_ativo].astype(str).tolist()
        novos = [_alias_ativo_novo(a) for a in antigos]
        renames = [(a, b) for a, b in zip(antigos, novos) if a != b]
        if renames:
            df[col_ativo] = novos
            renames_unicos = sorted(set(renames))
            print(f'[alias] {os.path.basename(file)}: {len(renames)} renames — {renames_unicos}')

    df.to_parquet(file.replace('.csv', '.parquet'), index=False)
    os.remove(file)
    print(f'{file} convertido para parquet')
print('Conversão finalizada (BaseFundos)')


# ─────────────────────────────────────────────────────────────────────────────
# 3) ETL do PL dos fundos (comportamento original)
# ─────────────────────────────────────────────────────────────────────────────
file_pl = "Dados/pl_fundos.parquet"
df2 = pd.read_parquet(file_pl)
df2 = df2.set_index(df2.columns[0])

df3 = pd.read_parquet('Dados/pl_fundos_teste.parquet')


# ─────────────────────────────────────────────────────────────────────────────
# 4) BBG (ECO DASH) — DI/DAP/WDO/TREASURY/IBOV/NTNB (comportamento original)
# ─────────────────────────────────────────────────────────────────────────────
file_bbg = 'Dados/BBG - ECO DASH.xlsx'
df = pd.read_excel(file_bbg, sheet_name='BZ RATES',
                   skiprows=1, thousands='.', decimal=',')

df.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2',
        'Unnamed: 3', 'Unnamed: 25'], axis=1, inplace=True)
df.columns.values[0] = 'Date'
df = df.drop([0])
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.drop(['WSP1 Index'], axis=1, inplace=True)

df.columns = [
    'Date', 'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
    'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP26', 'DAP27',
    'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'TREASURY', 'IBOV',
    'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40', 'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60'
]

df.to_parquet('Dados/df_inicial.parquet')


# ─────────────────────────────────────────────────────────────────────────────
# 5) DIV01 — comportamento original
# ─────────────────────────────────────────────────────────────────────────────
df_divone = pd.read_excel(file_bbg, sheet_name='DIV01',
                          skiprows=1, usecols='E:F', nrows=31)
df_divone = df_divone.T

columns = [
    'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
    'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP26', 'DAP27',
    'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'TREASURY', 'IBOV', 'S&P',
    'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40', 'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60'
]

df_divone.columns = columns
df_divone = df_divone.drop(df_divone.index[0])
df_divone.to_parquet('Dados/df_divone.parquet')


# ─────────────────────────────────────────────────────────────────────────────
# 6) NTNB — comportamento original
# ─────────────────────────────────────────────────────────────────────────────
df_ntnb = pd.read_excel('Dados/FechamentoNTNBs.xlsx')
df_ntnb.columns = df_ntnb.iloc[0]
df_ntnb = df_ntnb.drop(df_ntnb.index[0])
df_ntnb = df_ntnb.drop(df_ntnb.index[0])
df_ntnb.dropna(inplace=True)
df_ntnb.rename(columns={'Nome': 'Assets'}, inplace=True)
df_ntnb['Assets'] = pd.to_datetime(df_ntnb['Assets']).dt.date
df_ntnb['Assets'] = df_ntnb['Assets'].astype(str)
df_ntnb = df_ntnb.T
df_ntnb.columns = df_ntnb.iloc[0]
df_ntnb = df_ntnb.drop(df_ntnb.index[0])
df_ntnb.reset_index(inplace=True)
df_ntnb.rename(columns={0: 'Assets'}, inplace=True)

colunas = df_ntnb.columns
for col in colunas:
    df_ntnb[col] = df_ntnb[col].apply(lambda x: str(x).replace('.', ','))

df_b3 = pd.read_parquet('Dados/df_preco_de_ajuste_atual_completo.parquet')
df_precos = pd.concat([df_b3, df_ntnb], axis=0)
id_col = "Assets"
date_cols = [c for c in df_precos.columns if c != id_col]
date_cols = sorted(date_cols, key=pd.to_datetime)
df_precos = df_precos[[id_col] + date_cols]
df_precos[date_cols] = df_precos[date_cols].ffill(axis=1)
df_precos.to_parquet('Dados/df_preco_de_ajuste_atual_completo.parquet')


# ─────────────────────────────────────────────────────────────────────────────
# 7) LFT — comportamento original
# ─────────────────────────────────────────────────────────────────────────────
dados = pd.read_excel(r'Z:\Asset Management\FUNDOS e CLUBES\Gerencial\dashboard LFT.xlsx', sheet_name='Historico preços')
dados.rename(columns={'Unnamed: 0': 'Data'}, inplace=True)
dados.drop(index=[0, 1], inplace=True)
dados = dados[['Data', 'BLFT 0 06/01/30']]
dados.rename(columns={'BLFT 0 06/01/30': 'RetornoLFT'}, inplace=True)
dados.to_csv('Dados/dados_lft.csv', index=False)
