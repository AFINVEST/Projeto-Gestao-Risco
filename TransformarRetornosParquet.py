import pandas as pd
import numpy as np
import os
import glob

files = glob.glob('*.csv')

# Percorrer todos os arquivos csv
for file in files:
    # Ler o arquivo csv
    df = pd.read_csv(file)
    # Transformar em parquet
    df.to_parquet(file.replace('.csv', '.parquet'), index=False)
    # Remover o arquivo csv
    os.remove(file)
    print(f'{file} convertido para parquet')

print('Conversão finalizada')

# Preciso ler todos os arquivos dentro de BaseFundos e se for um arquivo csv transformar em parquet

# Pegar todos os arquivos csv
files = glob.glob('BaseFundos/*.csv')

# Percorrer todos os arquivos csv
for file in files:
    # Ler o arquivo csv
    df = pd.read_csv(file)
    # Transformar em parquet
    df.to_parquet(file.replace('.csv', '.parquet'), index=False)
    # Remover o arquivo csv
    os.remove(file)
    print(f'{file} convertido para parquet')

print('Conversão finalizada')

df = pd.read_parquet('Dados/portifolio_posições.parquet')

file_pl = "Dados/pl_fundos.parquet"
df2 = pd.read_parquet(file_pl)
# Colocar como indice a coluna 0
df2 = df2.set_index(df2.columns[0])


df3 = pd.read_parquet('Dados/pl_fundos_teste.parquet')


file_bbg = 'Dados/BBG - ECO DASH.xlsx'
df = pd.read_excel(file_bbg, sheet_name='BZ RATES',
                   skiprows=1, thousands='.', decimal=',')

df.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2',
        'Unnamed: 3', 'Unnamed: 26'], axis=1, inplace=True)
df.columns.values[0] = 'Date'
df = df.drop([0])  # Remove a primeira linha
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.drop(['WSP1 Index'], axis=1, inplace=True)

df.columns = [
    'Date', 'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
    'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP25', 'DAP26', 'DAP27',
    'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'TREASURY', 'IBOV',
    'NTNB25', 'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40', 'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60'
]
df.to_parquet('Dados/df_inicial.parquet')

df_divone = pd.read_excel(file_bbg, sheet_name='DIV01',
                          skiprows=1, usecols='E:F', nrows=33)
df_divone = df_divone.T
columns = [
    'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
    'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP25', 'DAP26', 'DAP27',
    'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'TREASURY', 'IBOV', 'S&P',
    'NTNB25', 'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40', 'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60'
]
df_divone.columns = columns
df_divone = df_divone.drop(df_divone.index[0])

df_divone.to_parquet('Dados/df_divone.parquet')

df_ntnb = pd.read_excel('Dados/FechamentoNTNBs.xlsx')
df_ntnb.columns = df_ntnb.iloc[0]
df_ntnb = df_ntnb.drop(df_ntnb.index[0])
# Dropar a primeira linha
df_ntnb = df_ntnb.drop(df_ntnb.index[0])
df_ntnb.dropna(inplace=True)
df_ntnb.rename(columns={'Nome': 'Assets'}, inplace=True)
# Tirar o horario das datas da coluna Assets
df_ntnb['Assets'] = pd.to_datetime(df_ntnb['Assets']).dt.date
df_ntnb['Assets'] = df_ntnb['Assets'].astype(str)
df_ntnb = df_ntnb.T
df_ntnb.columns = df_ntnb.iloc[0]
df_ntnb = df_ntnb.drop(df_ntnb.index[0])
df_ntnb.reset_index(inplace=True)
df_ntnb.rename(columns={0: 'Assets'}, inplace=True)
# Mudar todos os "." por ","
colunas = df_ntnb.columns
for col in colunas:
    df_ntnb[col] = df_ntnb[col].apply(lambda x: str(x).replace('.', ','))

df_b3 = pd.read_parquet('Dados/df_preco_de_ajuste_atual.parquet')


# Fazer o merge dos dataframes
df_precos = pd.concat([df_b3, df_ntnb], axis=0)

df_precos.to_parquet('Dados/df_preco_de_ajuste_atual_completo.parquet')
