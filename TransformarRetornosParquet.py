"""
TransformarRetornosParquet.py  (v3 — naming NOVO em df_inicial + df_divone)
============================================================================

MUDANÇAS EM RELAÇÃO AO ORIGINAL:
  - CSV -> parquet em BaseFundos/ aplica alias antigo -> novo na coluna 'Ativo'.
  - df_inicial.parquet e df_divone.parquet gravados com naming NOVO
    (DI_F26, DAP_Q30 etc.). Elimina divergência entre load_basefundos
    (naming novo, patchado) e load_and_process_excel (que consome df_inicial).
  - Restante do ETL (BBG excel, LFT) permanece IDÊNTICO ao original.

Idempotente: se o CSV/planilha já vier com naming novo, alias é no-op.
"""
import pandas as pd
import numpy as np
import os
import re
import glob


# ─────────────────────────────────────────────────────────────────────────────
# Alias antigo -> novo (idempotente)
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


def _alias_columns(cols):
    """Aplica alias a uma lista/Index de nomes de colunas."""
    return [_alias_ativo_novo(c) for c in cols]


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
# 2) CSVs de BaseFundos/ -> parquet — APLICA ALIAS NA COLUNA Ativo
# ─────────────────────────────────────────────────────────────────────────────
files = glob.glob('BaseFundos/*.csv')
for file in files:
    df = pd.read_csv(file)

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
            print(f'[alias] {os.path.basename(file)}: {len(renames)} renames — {sorted(set(renames))}')

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
# 4) BBG (ECO DASH) -> df_inicial.parquet — NOMES COM NAMING NOVO
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

# NAMING NOVO (era: DI_26, DI_27, ..., DAP26, DAP27, ...)
df.columns = [
    'Date',
    'DI_F26', 'DI_F27', 'DI_F28', 'DI_F29', 'DI_F30',
    'DI_F31', 'DI_F32', 'DI_F33', 'DI_F35',
    'DAP_Q26', 'DAP_K27', 'DAP_Q28', 'DAP_Q30', 'DAP_Q32', 'DAP_K35', 'DAP_Q40',
    'WDO1', 'TREASURY', 'IBOV',
    'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40',
    'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60'
]

df.to_parquet('Dados/df_inicial.parquet')
print(f'df_inicial.parquet salvo com naming novo ({df.shape[1]-1} cols)')


# ─────────────────────────────────────────────────────────────────────────────
# 5) DIV01 -> df_divone.parquet — NOMES COM NAMING NOVO
# ─────────────────────────────────────────────────────────────────────────────
df_divone = pd.read_excel(file_bbg, sheet_name='DIV01',
                          skiprows=1, usecols='E:F', nrows=31)
df_divone = df_divone.T

columns_novo = [
    'DI_F26', 'DI_F27', 'DI_F28', 'DI_F29', 'DI_F30',
    'DI_F31', 'DI_F32', 'DI_F33', 'DI_F35',
    'DAP_Q26', 'DAP_K27', 'DAP_Q28', 'DAP_Q30', 'DAP_Q32', 'DAP_K35', 'DAP_Q40',
    'WDO1', 'TREASURY', 'IBOV', 'S&P',
    'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40',
    'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60'
]

df_divone.columns = columns_novo
df_divone = df_divone.drop(df_divone.index[0])
df_divone.to_parquet('Dados/df_divone.parquet')
print(f'df_divone.parquet salvo com naming novo ({df_divone.shape[1]} cols)')


# ─────────────────────────────────────────────────────────────────────────────
# 6) NTNB (comportamento original)
# ─────────────────────────────────────────────────────────────────────────────
df_ntnb = pd.read_excel('Dados/FechamentoNTNBs.xlsx', sheet_name='Planilha1')  # PUs (Planilha2 tem yields)
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
# 7) LFT — DESATIVADO (agora atualizado por atualizar_cdi_lft.py via BCB SELIC)
# ─────────────────────────────────────────────────────────────────────────────
# Historico da planilha 'dashboard LFT.xlsx' permanece em dados_lft.csv.
# Novos dias sao extrapolados via CDI+0.10% pelo script atualizar_cdi_lft.py
# no fluxo do .bat v2. A planilha manual/BBG NAO e mais lida.
#
# Se voce quiser reativar a leitura da planilha (por exemplo para atualizar
# retroativamente o historico com dados reais da LFT), rode:
#
#   import pandas as pd
#   dados = pd.read_excel(r'Z:\Asset Management\FUNDOS e CLUBES\Gerencial\dashboard LFT.xlsx',
#                         sheet_name='Historico preços')
#   dados.rename(columns={'Unnamed: 0': 'Data'}, inplace=True)
#   dados.drop(index=[0, 1], inplace=True)
#   dados = dados[['Data', 'BLFT 0 06/01/30']]
#   dados.rename(columns={'BLFT 0 06/01/30': 'RetornoLFT'}, inplace=True)
#   dados.to_csv('Dados/dados_lft.csv', index=False)
print('LFT: leitura de planilha desativada (usar atualizar_cdi_lft.py via BCB).')

print('\n[OK] TransformarRetornosParquet v3 finalizado.')
