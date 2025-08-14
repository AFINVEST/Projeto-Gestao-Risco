import psycopg2
from psycopg2 import sql
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from datetime import date
from supabase import create_client
from plotnine import (
    ggplot, aes, geom_col, geom_line, labs,
    scale_fill_brewer, scale_color_manual, scale_x_datetime,
    theme_minimal, theme,
    element_text, element_rect, element_line
)
# ── depois dos imports pandas/streamlit ────────────────────
import functools, os, datetime as dt


# ==========================================================
#               FUNÇÕES AUXILIARES (MESMAS)
# ==========================================================

# Configuração do Supabase (substitua pelas suas credenciais)
SUPABASE_URL = 'https://obgwfekirteetqzjydry.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9iZ3dmZWtpcnRlZXRxemp5ZHJ5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczODk2MDc1MCwiZXhwIjoyMDU0NTM2NzUwfQ.k7-Haw1txbCEwb_MzkynOeEuRJpfgt3msePdvQavWAc'
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def process_portfolio(df_pl, Weights):
    df_pl['PL'] = (
        df_pl['PL']
        .str.replace('R$', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .replace('--', np.nan)
        .astype(float)
    )
    df_pl = df_pl.iloc[[4, 8, 9, 10, 16, 17, 18, 19, 21]]
    weights_zero = []
    for weight in Weights:
        if weight == 0:
            weights_zero.append(0)
        else:
            weights_zero.append(1)
    df_pl['Weights Zero'] = weights_zero
    df_pl['PL_ZeroPeso'] = df_pl['PL'] * df_pl['Weights Zero']
    soma_sem_pesos = df_pl['PL_ZeroPeso'].sum()
    df_pl.drop(['Weights Zero', 'PL_ZeroPeso'], axis=1, inplace=True)
    df_pl['Weights'] = Weights
    df_pl['Weights'] = df_pl['Weights'].astype(float)
    df_pl['PL_atualizado'] = df_pl['PL'] * df_pl['Weights']
    df_pl['Adm'] = ['SANTANDER', 'BTG', 'SANTANDER',
                    'SANTANDER', 'BTG', 'BTG', 'BTG', 'BTG', 'BTG']
    
    return df_pl, df_pl['PL_atualizado'].sum(), soma_sem_pesos


def process_portfolio_especifico(df_pl, Weights, fundo):
    df_pl = df_pl.iloc[[4, 8, 9, 10, 16, 17, 18, 19, 21]]
    weights_zero = []
    for weight in Weights:
        if weight == 0:
            weights_zero.append(0)
        else:
            weights_zero.append(1)
    df_pl['Weights Zero'] = weights_zero
    df_pl['PL_ZeroPeso'] = df_pl['PL'] * df_pl['Weights Zero']
    df_pl['Weights'] = Weights
    df_pl['Weights'] = df_pl['Weights'].astype(float)
    df_pl['Adm'] = ['SANTANDER', 'BTG', 'SANTANDER',
                    'SANTANDER', 'BTG', 'BTG', 'BTG', 'BTG', 'BTG']
    df_pl = df_pl[df_pl['Fundos/Carteiras Adm'] == fundo]
    soma_sem_pesos = df_pl['PL_ZeroPeso'].sum()
    df_pl['PL_atualizado'] = df_pl['PL'] * df_pl['Weights']
    df_pl.drop(['Weights Zero', 'PL_ZeroPeso'], axis=1, inplace=True)

    return df_pl, df_pl['PL_atualizado'].sum(), soma_sem_pesos


def load_and_process_excel(df_excel, assets_sel):
    df = df_excel.copy()
    df_precos = df.tail(1)
    df_precos['TREASURY'] = df_precos['TREASURY'] * df_precos['WDO1'] / 10000
    df_precos = df_precos[assets_sel]
    return df_precos, df


def load_and_process_divone(file_bbg, df_excel):
    df_divone = pd.read_parquet('Dados/df_divone.parquet')

    df_pe = df_excel.copy()
    df_pe = df_pe.tail(1)

    df_divone['TREASURY'] = df_divone['TREASURY'].astype(
        str).str.replace(',', '.', regex=False)
    df_pe['WDO1'] = df_pe['WDO1'].astype(
        str).str.replace(',', '.', regex=False)

    df_divone['TREASURY'] = pd.to_numeric(
        df_divone['TREASURY'], errors='coerce')
    
    df_pe['WDO1'] = pd.to_numeric(df_pe['WDO1'], errors='coerce')

    df_divone['TREASURY'] = df_divone['TREASURY'] * \
        df_pe['WDO1'].iloc[0] / 10000
    dolar = df_pe['WDO1'].iloc[0]
    treasury = df_pe['TREASURY'].iloc[0]

    return df_divone, dolar, treasury


def load_and_process_divone2(file_bbg, df_excel):
    # --- 1) Carrega o df_divone existente ---
    path_parquet = 'Dados/df_divone.parquet'
    df_divone = pd.read_parquet(path_parquet)

    # --- 2) Seu pré-processamento (como você já fez) ---
    # DI
    di = pd.read_excel('Dados/AF_Trading.xlsm', sheet_name='Base CDI', skiprows=1, usecols='G:M')
    di = di.iloc[:14, :].iloc[4:, :]         # mantém 14 linhas, depois dropa as 4 primeiras
    di = di.drop(di.index[1])                # dropa a segunda linha
    di['Nome'] = ['DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30', 'DI_32', 'DI_33', 'DI_35', 'DI_37']
    di = di[['Nome', 'DV01']]

    # DAP
    dap = pd.read_excel('Dados/AF_Trading.xlsm', sheet_name='Base IPCA', skiprows=16, usecols='A:M')
    dap = dap.iloc[:9, :]
    dap['Nome'] = ['DAP26', 'DAP27', 'DAP28', 'DAP29', 'DAP30', 'DAP32', 'DAP33', 'DAP35', 'DAP40']
    dap = dap[dap['Nome'] != 'DAP33']
    dap = dap[['Nome', 'DV01']]

    # --- 3) Monta a série de atualização (DI + DAP) ---
    s_di  = di.set_index('Nome')['DV01'].astype(float)
    s_dap = dap.set_index('Nome')['DV01'].astype(float)
    s_upd = pd.concat([s_di, s_dap])  # índice = nomes dos ativos, valores = DV01

    # Filtra apenas os ativos que existem no df_divone
    cols_existentes = [c for c in df_divone.columns if c in s_upd.index]

    # --- 4) Descobre qual linha é a de DV01 no df_divone ---
    linha_alvo = None
    for cand in ['FUT_TICK_VAL', 'DV01', 'BPV', 'PVBP']:
        if cand in df_divone.index:
            linha_alvo = cand
            break

    # Se não achou, mas o df tem 1 linha, usa a única linha
    if linha_alvo is None and df_divone.shape[0] == 1:
        linha_alvo = df_divone.index[0]

    if linha_alvo is None:
        raise ValueError("Não encontrei a linha de DV01 em df_divone (ex.: 'FUT_TICK_VAL').")

    # --- 5) Atualiza os valores de DV01 nessa linha, somente para DI_* e DAP* que existirem ---
    df_divone.loc[linha_alvo, cols_existentes] = s_upd.loc[cols_existentes].values

    # (opcional) Reporta o que foi atualizado e o que ficou de fora
    atualizados = cols_existentes
    nao_encontrados = sorted(set(s_upd.index) - set(cols_existentes))
    print(f"Atualizados ({len(atualizados)}): {atualizados}")
    if nao_encontrados:
        print(f"Ignorados (não existem no df_divone): {nao_encontrados}")


    df_pe = df_excel.copy()
    df_pe = df_pe.tail(1)

    df_divone['TREASURY'] = df_divone['TREASURY'].astype(
        str).str.replace(',', '.', regex=False)
    df_pe['WDO1'] = df_pe['WDO1'].astype(
        str).str.replace(',', '.', regex=False)

    df_divone['TREASURY'] = pd.to_numeric(
        df_divone['TREASURY'], errors='coerce')
    
    df_pe['WDO1'] = pd.to_numeric(df_pe['WDO1'], errors='coerce')

    df_divone['TREASURY'] = df_divone['TREASURY'] * \
        df_pe['WDO1'].iloc[0] / 10000
    dolar = df_pe['WDO1'].iloc[0]
    treasury = df_pe['TREASURY'].iloc[0]

    return df_divone, dolar, treasury


def process_returns(df, assets):
    df_retorno = df.copy()
    df_retorno = df_retorno[assets]
    df_retorno.dropna(inplace=True)
    df_retorno = df_retorno.astype(float)
    df_retorno = df_retorno.tail(756).reset_index(drop=True)
    df_retorno = np.log(df_retorno / df_retorno.shift(1))
    return df_retorno


def process_returns2(df, assets):
    #Colocar a coluna 'Date' como índice
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    df_retorno = df.copy()
    df_retorno = df_retorno[assets]
    df_retorno.dropna(inplace=True)
    df_retorno = df_retorno.astype(float)
    df_retorno = df_retorno.tail(756)
    df_retorno = np.log(df_retorno / df_retorno.shift(1))
    return df_retorno


def var_not_parametric(data, alpha=0.05):
    return data.quantile(alpha)


def adjust_prices_with_var(df_precos, var_ativos):
    df_precos = df_precos.T
    df_precos.columns = ['Valor Fechamento']
    df_precos['Valor Fechamento Ajustado pelo Var'] = (
        df_precos['Valor Fechamento'] * var_ativos.values
    )
    return df_precos


def calculate_contracts_per_fund(df_pl, df_precos):
    for i in range(len(df_precos)):
        df_pl[f'Max Contratos {df_precos.index[i]}'] = (
            df_pl['PL_atualizado'] /
            df_pl['PL_atualizado'].sum()
        ) * df_precos.iloc[i]['Valor Total']

    return df_pl


def calculate_contracts_per_fund_input(df_pl, df_precos):
    for i in range(len(df_precos)):
        df_pl[f'Contratos {df_precos.index[i]}'] = (
            df_pl['PL_atualizado'] /
            df_pl['PL_atualizado'].sum()
        ) * df_precos.iloc[i]['Quantidade']
        # df_pl[f'Contratos {df_precos.index[i]}'] = df_pl[f'Contratos {df_precos.index[i]}'].apply(
        #    lambda x: round(x, 0))
    return df_pl


def calculate_portfolio_values(df_precos, df_pl_processado, var_bps):
    pl_santander = df_pl_processado[df_pl_processado['Adm']
                                    == 'SANTANDER']['PL_atualizado'].sum()
    pl_btg = df_pl_processado[df_pl_processado['Adm']
                              == 'BTG']['PL_atualizado'].sum()
    portifolio_santander = pl_santander * var_bps
    portifolio_btg = pl_btg * var_bps
    df_precos['Santander'] = portifolio_santander / \
        df_precos['Valor Fechamento Ajustado pelo Var']
    df_precos['BTG'] = portifolio_btg / \
        df_precos['Valor Fechamento Ajustado pelo Var']
    df_precos['Valor Total'] = df_precos['Santander'] + df_precos['BTG']
    for col in df_precos.columns:
        df_precos[col] = df_precos[col].apply(lambda x: round(x, 0))

    return df_precos.abs()


def processar_b3_portifolio():
    """
    Exemplo de função que carrega dois dataframes de parquet:
     - Dados/df_preco_de_ajuste_atual_completo.parquet : preços de fechamento (colunas de datas)
     - df_variacao.parquet : variação diária dos ativos (colunas de datas)
    """
    df_b3_fechamento = pd.read_parquet(
        "Dados/df_preco_de_ajuste_atual_completo.parquet")
    # df_b3_fechamento possui colunas: ['Ativo', '17/01/2025', '18/01/2025', ...]

    # df_b3_variacao possui colunas: ['Ativo', '17/01/2025', '18/01/2025', ...]
    # Tirar o pontos das casas de milhar
    df_b3_fechamento = df_b3_fechamento.replace('\.', '', regex=True)

    # Trocar Virgula por Ponto
    df_b3_fechamento = df_b3_fechamento.replace(',', '.', regex=True)

    # Converter para Float menos a primeira coluna
    df_b3_fechamento.iloc[:, 1:] = df_b3_fechamento.iloc[:, 1:].astype(float)

    # Multiplicar a linha da treasury por 1000
    df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'TREASURY', df_b3_fechamento.columns !=
                         'Assets'] = df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'TREASURY', df_b3_fechamento.columns != 'Assets'] * 1000
    df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'WDO1', df_b3_fechamento.columns !=
                         'Assets'] = df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'WDO1', df_b3_fechamento.columns != 'Assets'] * 10

    return df_b3_fechamento


################ DEIXAR ESSA FUNÇÃO ATUALIZADA COM A LISTA DE ASSETS DEFAUTL DO PORTIFÓLIO E SUAS QUANTIDADES ################
def processar_dados_port():
    df_assets = pd.read_parquet("Dados/portifolio_posições.parquet")
    df_assets.rename(columns={'Unnamed: 0': 'Ativo'}, inplace=True)
    df_portifolio_default = df_assets.copy()
    # Agrupar por ativo e somar as quantidades
    df_assets = df_assets.groupby('Ativo').sum().reset_index()
    # Criar uma lista com os ativos, convertendo para string e ignorando valores nulos
    assets_iniciais = df_assets['Ativo'].tolist()
    # Criar uma lista com as quantidades, convertendo para inteiros e ignorando valores nulos
    quantidades_iniciais = (
        df_assets['Quantidade']
        # Converte para inteiro, garantindo valores válidos
        .apply(lambda x: int(x) if x == x else 0)
        .tolist()
    )
    # Criar um dicionário com os ativos e suas quantidades
    assets_quantidades = dict(zip(assets_iniciais, quantidades_iniciais))

    return assets_iniciais, assets_quantidades, df_portifolio_default


# NA SEGUNDA PÁGINA TEM QUE CRIAR UM CAMPO PARA O USUÁRIO INSERIR A DATA QUE ELE COMPROU DETERMINADO ATIVO E SE ELE QUER INSERIR UM VALOR DE COMPRA ESPECÍFICO
# ATUALIZAR PRA ESSA LISTA DE QUANTIDADES LER O PORTIFÓLIO_POSIÇÕES E PEGAR AS QUANTIDADES DE LÁ, MAS PERMITIR O USUÁRIO ALTERAR TANTO AS QUANTIDADES QUANTO OS ATIVOS SELECIONADOS. OU SEJA, QUERO QUE OS INPUTS DO USUÁRIO NA TELA PRINCIPAL ATUALIZEM O PORTIFÓLIO_POSIÇÕES
# compra_especifica = {'DAP28': 5.5, 'TREASURY': 105.5}
# dia_compra = {'DAP28': '2025-01-17', 'DI_31': '2025-01-17',
#               'WDO1': '2025-01-17', 'TREASURY': '2025-01-17'}


def checkar_portifolio(assets, quantidades, compra_especifica, dia_compra, df_contratos):
    """
    Verifica ou cria um parquet de portfólio. Exibe dois DataFrames:
      1) Posição atual salva no parquet.
      2) Novo DataFrame com os ativos e dados recebidos como input.
    Permite concatenar o novo DataFrame ao existente e salvar uma versão compilada
    (agrupada por Ativo + Dia, com soma da Quantidade e média ponderada do Preço).
    """

    st.write('---')
    st.title("Gestão de Portfólio")

    nome_arquivo_portifolio = 'Dados/portifolio_posições.parquet'
    # <-- Sua função existente para pegar dados B3
    df_b3_fechamento = processar_b3_portifolio()

    # Carregar (ou criar) o portfólio existente
    if os.path.exists(nome_arquivo_portifolio):
        df_portifolio_salvo = pd.read_parquet(nome_arquivo_portifolio)
    else:
        df_portifolio_salvo = pd.DataFrame(columns=[
            'Ativo',
            'Quantidade',
            'Dia de Compra',
            'Preço de Compra',
            'Preço de Ajuste Atual',
            'Rendimento'
        ])

    # Criar novo DataFrame baseado nos inputs
    novo_portifolio = pd.DataFrame(columns=[
        'Ativo',
        'Quantidade',
        'Dia de Compra',
        'Preço de Compra',
        'Preço de Ajuste Atual',
        'Rendimento'
    ])

    # Última data de fechamento disponível
    ultimo_fechamento = df_b3_fechamento.columns[-1]
    dolar = df_b3_fechamento.loc[df_b3_fechamento['Assets']
                                 == 'WDO1', ultimo_fechamento].values[0]

    # Converter dia_compra para string 'YYYY-MM-DD'
    dia_compra = (
        {k: v.strftime('%Y-%m-%d') if isinstance(v, datetime.date)
         else v for k, v in dia_compra.items()}
        if isinstance(dia_compra, dict)
        else dia_compra.strftime('%Y-%m-%d') if isinstance(dia_compra, datetime.date)
        else dia_compra
    )

    # Se for dict, pegamos o primeiro valor só para exibir, mas cada ativo usa seu dia
    if isinstance(dia_compra, dict):
        dia_compra_unico = next(iter(dia_compra.values()))
    else:
        dia_compra_unico = dia_compra

    dia_compra_unico = pd.to_datetime(dia_compra_unico).strftime('%Y-%m-%d')

    # -----------------------------------------------------------------------
    # Gerar o novo_portifolio (várias operações, inclusive para o mesmo dia e ativo)
    # -----------------------------------------------------------------------
    for asset in assets:
        qtd_final = quantidades[asset]
        dia_de_compra_atual = dia_compra[asset] if isinstance(
            dia_compra, dict) else dia_compra_unico

        try:
            # Filtrar os dados no DataFrame de fechamento
            filtro_ativo = (df_b3_fechamento['Assets'] == asset)

            # Se não houver preço especificado manualmente, usamos o do B3
            if compra_especifica.get(asset) is None:
                preco_compra = df_b3_fechamento.loc[filtro_ativo,
                                                    dia_de_compra_atual].values[0]
            else:
                preco_compra = compra_especifica[asset]

            preco_fechamento_atual = df_b3_fechamento.loc[
                df_b3_fechamento["Assets"] == asset,
                ultimo_fechamento
            ].values[0]
            preco_fechamento_atual = pd.to_numeric(
                preco_fechamento_atual, errors='coerce')

            # Calcular rendimento
            if asset == 'TREASURY':
                rendimento = qtd_final * \
                    (preco_fechamento_atual - preco_compra) * (dolar / 10000)

            elif 'DAP' in asset:
                df_ajuste = pd.read_parquet(
                    'Dados/df_valor_ajuste_contrato.parquet')

                # Selecionar apenas as colunas de data (ignorando a primeira, que é "Assets")
                colunas_datas_originais = df_ajuste.columns[1:]
                df_ajuste[colunas_datas_originais] = (
                    df_ajuste[colunas_datas_originais]
                    .replace('\.', '', regex=True)
                    .replace(',', '.', regex=True)
                    .astype(float)
                )

                # Renomear as colunas de data para string no formato 'YYYY-MM-DD'
                novos_nomes_colunas = [
                    pd.to_datetime(col, errors='coerce').strftime('%Y-%m-%d') for col in colunas_datas_originais
                ]
                renomear_colunas = dict(
                    zip(colunas_datas_originais, novos_nomes_colunas))
                df_ajuste.rename(columns=renomear_colunas, inplace=True)

                # Obter a data da compra específica do ativo
                data_compra_raw = dia_compra.get(asset)
                if data_compra_raw is None:
                    st.error(
                        f"Data de compra não encontrada para o ativo {asset}")
                else:
                    coluna_dia_compra_str = pd.to_datetime(
                        data_compra_raw).strftime('%Y-%m-%d')

                    # Verificar se a coluna existe no DataFrame antes de tentar acessar
                    if coluna_dia_compra_str not in df_ajuste.columns:
                        st.error(
                            f"Coluna de data '{coluna_dia_compra_str}' não encontrada no DataFrame.")
                    else:
                        filtro = df_ajuste['Assets'] == asset
                        if filtro.any():
                            valor_ajuste = df_ajuste.loc[filtro,
                                                         coluna_dia_compra_str].values[0]
                            # Ver se tem alguma coluna de ajuste posterior a data de compra
                            colunas_uteis = df_ajuste.columns[df_ajuste.columns >
                                                              coluna_dia_compra_str]
                            colunas_uteis = colunas_uteis[colunas_uteis != 'Assets']
                            colunas_uteis = colunas_uteis[:-1]
                            if len(colunas_uteis) > 0:
                                # Pegar o valor do ajuste mais recente
                                rendimento = valor_ajuste * qtd_final
                            else:
                                rendimento = 0
                        else:
                            st.error(
                                f"Ativo {asset} não encontrado no DataFrame.")
            elif 'DI' in asset:
                df_ajuste = pd.read_parquet(
                    'Dados/df_valor_ajuste_contrato.parquet')
                
                # Selecionar apenas as colunas de data (ignorando a primeira, que é "Assets")
                colunas_datas_originais = df_ajuste.columns[1:]
                df_ajuste[colunas_datas_originais] = (
                    df_ajuste[colunas_datas_originais]
                    .replace('\.', '', regex=True)
                    .replace(',', '.', regex=True)
                    .astype(float)
                )

                # Renomear as colunas de data para string no formato 'YYYY-MM-DD'
                novos_nomes_colunas = [
                    pd.to_datetime(col, errors='coerce').strftime('%Y-%m-%d') for col in colunas_datas_originais
                ]
                renomear_colunas = dict(
                    zip(colunas_datas_originais, novos_nomes_colunas))
                df_ajuste.rename(columns=renomear_colunas, inplace=True)

                # Obter a data da compra específica do ativo
                data_compra_raw = dia_compra.get(asset)
                if data_compra_raw is None:
                    st.error(
                        f"Data de compra não encontrada para o ativo {asset}")
                else:
                    coluna_dia_compra_str = pd.to_datetime(
                        data_compra_raw).strftime('%Y-%m-%d')

                    # Verificar se a coluna existe no DataFrame antes de tentar acessar
                    if coluna_dia_compra_str not in df_ajuste.columns:
                        st.error(
                            f"Coluna de data '{coluna_dia_compra_str}' não encontrada no DataFrame.")
                    else:
                        filtro = df_ajuste['Assets'] == asset
                        if filtro.any():
                            valor_ajuste = df_ajuste.loc[filtro,
                                                         coluna_dia_compra_str].values[0]
                            # Ver se tem alguma coluna de ajuste posterior a data de compra
                            colunas_uteis = df_ajuste.columns[df_ajuste.columns >
                                                              coluna_dia_compra_str]
                            colunas_uteis = colunas_uteis[colunas_uteis != 'Assets']
                            #Dropar a ultima coluna
                            colunas_uteis = colunas_uteis[:-1]
                            if len(colunas_uteis) > 0:
                                # Pegar o valor do ajuste mais recente
                                rendimento = valor_ajuste * qtd_final
                            else:
                                rendimento = qtd_final * (preco_fechamento_atual - preco_compra)

                        else:
                            st.error(
                                f"Ativo {asset} não encontrado no DataFrame.")

            else:
                rendimento = qtd_final * \
                    (preco_fechamento_atual - preco_compra)

            # Adicionar linha ao novo DataFrame
            nova_linha = {
                'Ativo': asset,
                'Quantidade': qtd_final,
                'Dia de Compra': dia_de_compra_atual,
                'Preço de Compra': preco_compra,
                'Preço de Ajuste Atual': preco_fechamento_atual,
                'Rendimento': rendimento
            }
            novo_portifolio = pd.concat(
                [novo_portifolio, pd.DataFrame([nova_linha])], ignore_index=True)

        except Exception as e:
            st.error(f"Erro ao processar o ativo {asset}: {e}")

    # -----------------------------------------------------------------------
    # 1) Exibir o "Portfólio Atual" (linhas já salvas no parquet)
    # 2) Exibir as "Novas Operações"
    # 3) Exibir uma visualização "processada" (somada) mas ainda sem agrupar Ativo+Dia
    # 4) Exibir a versão "compilada" (agrupada por Ativo+Dia), com média do preço
    # -----------------------------------------------------------------------
    col_p1, col_p3, col_p2 = st.columns([4.9, 0.2, 4.9])

    with col_p1:
        #st.subheader("Portfólio Atual (já salvo)")
        #st.table(df_portifolio_salvo.set_index('Ativo'))

        # Agrupar para ver quantidades consolidadas do portfólio atual
        df_atual = df_portifolio_salvo.groupby('Ativo', as_index=False)[
            'Quantidade'].sum()
        assets_atual = df_atual['Ativo'].tolist()
        quantidades_atual = df_atual['Quantidade'].tolist()

    with col_p3:
        st.html(
            '''
            <div class="divider-vertical-lines"></div>
            <style>
                .divider-vertical-lines {
                    border-left: 2px solid rgba(49, 51, 63, 0.2);
                    height: 40vh;
                    margin: auto;
                }
                @media (max-width: 768px) {
                    .divider-vertical-lines {
                        display: none;
                    }
                }
            </style>
            '''
        )

    with col_p2:
        st.subheader("Novas Operações")
        st.table(novo_portifolio.set_index('Ativo'))

        # "Portfólio processado" (bruto, apenas concat do atual + novo)
        st.subheader("Portfólio processado")
        df_teste = pd.concat(
            [df_portifolio_salvo, novo_portifolio], ignore_index=True)
        if st.checkbox("Mostrar tabela com compras por dia", False):
            st.table(df_teste.set_index('Ativo'))
        df_teste['Dia de Compra'] = pd.to_datetime(
            df_teste['Dia de Compra'], format='%Y-%m-%d', errors='coerce')

        # 1) Ordenar por Dia de Compra, para garantir que a última linha tenha o dia mais recente
        df_teste = df_teste.sort_values('Dia de Compra')

        def agrupar_por_ativo(subdf):
            # Quantidade total
            qtd_total = subdf['Quantidade'].sum()

            if qtd_total != 0:
                # Média ponderada de Preço de Compra:
                # (soma de [Quantidade_i * PreçoCompra_i]) / soma(Quantidade_i)
                preco_medio_compra = (
                    (subdf['Quantidade'] * subdf['Preço de Compra']).sum()
                    / qtd_total
                )
            else:
                preco_medio_compra = 0

            # Preço de Ajuste Atual mais recente (a linha final depois de sort)
            preco_ajuste_atual = subdf.iloc[-1]['Preço de Ajuste Atual']

            # Rendimento recalculado:
            # (Preço Ajuste Atual - Preço Compra Médio) * Quantidade total
            # Rendimento por ativo DAP
            if 'DAP' in subdf['Ativo'].iloc[0]:
                df_ajuste = pd.read_parquet(
                    'Dados/df_valor_ajuste_contrato.parquet')
                # Ver se a ultima coluna é igual a penultima
                if df_ajuste.iloc[:, -1].equals(df_ajuste.iloc[:, -2]):
                    df_ajuste = df_ajuste.iloc[:, :-1]
                # Corrigir formatação
                colunas_datas = df_ajuste.columns[1:]
                df_ajuste[colunas_datas] = df_ajuste[colunas_datas].replace(
                    '\.', '', regex=True).replace(',', '.', regex=True)
                df_ajuste[colunas_datas] = df_ajuste[colunas_datas].astype(
                    float)

                # Converter nomes de colunas (datas) para datetime
                datas_convertidas = pd.to_datetime(
                    colunas_datas, errors='coerce')
                colunas_datas_validas = [col for col, data in zip(
                    df_ajuste.columns[1:], datas_convertidas) if pd.notnull(data)]

                # Garantir que vamos usar apenas datas válidas
                df_ajuste = df_ajuste[['Assets'] + colunas_datas_validas]

                # Pegar o nome do ativo (assumindo que todas as linhas do subdf têm o mesmo ativo)
                ativo = subdf['Ativo'].iloc[0]

                # DataFrame de ajustes do ativo
                linha_ajuste = df_ajuste[df_ajuste['Assets'] == ativo].drop(
                    columns='Assets')

                # Converter colunas novamente para datetime para indexar
                datas_ajuste = pd.to_datetime(linha_ajuste.columns)

                # Agora iterar sobre cada linha de compra do subdf
                rendimentos = []
                for _, row in subdf.iterrows():
                    dia_compra = pd.to_datetime(row['Dia de Compra'])
                    quantidade = row['Quantidade']

                    # Filtrar colunas de datas iguais ou posteriores à data de compra
                    colunas_uteis = linha_ajuste.columns[datas_ajuste > dia_compra]

                    # Soma dos valores de ajuste após a data de compra
                    rendimento = linha_ajuste[colunas_uteis].sum(
                        axis=1).values[0] * quantidade
                    rendimentos.append(rendimento)

                # Atribuir os rendimentos ao subdf
                subdf['Rendimento'] = rendimentos
                rendimento_final = sum(rendimentos)

            elif 'DI' in subdf['Ativo'].iloc[0]:
                df_ajuste = pd.read_parquet('Dados/df_valor_ajuste_contrato.parquet')

                # Se a última coluna é igual à penúltima, remove a última (cópia)
                if df_ajuste.iloc[:, -1].equals(df_ajuste.iloc[:, -2]):
                    df_ajuste = df_ajuste.iloc[:, :-1]

                # Corrigir formatação
                colunas_datas = df_ajuste.columns[1:]
                df_ajuste[colunas_datas] = (
                    df_ajuste[colunas_datas]
                    .replace('\.', '', regex=True)
                    .replace(',', '.', regex=True)
                    .astype(float)
                )

                # Converter nomes de colunas (datas) para datetime e manter apenas válidas
                datas_convertidas = pd.to_datetime(colunas_datas, errors='coerce')
                colunas_datas_validas = [
                    col for col, data in zip(df_ajuste.columns[1:], datas_convertidas) if pd.notnull(data)
                ]
                df_ajuste = df_ajuste[['Assets'] + colunas_datas_validas]

                # Linha de ajuste do ativo
                ativo = subdf['Ativo'].iloc[0]
                linha_ajuste = df_ajuste[df_ajuste['Assets'] == ativo].drop(columns='Assets')
                datas_ajuste = pd.to_datetime(linha_ajuste.columns) if not linha_ajuste.empty else pd.to_datetime([])

                # Itera cada compra
                rendimentos = []
                for _, row in subdf.iterrows():
                    dia_compra  = pd.to_datetime(row['Dia de Compra'])
                    quantidade  = row['Quantidade']

                    # 1) PnL do DIA DA COMPRA (D0) por unidade
                    pnl_d0_unit = (row['Preço de Ajuste Atual'] - row['Preço de Compra'])

                    # 2) Ajustes a partir de D+1
                    if linha_ajuste.empty:
                        soma_ajustes_unit = 0.0
                    else:
                        colunas_uteis = linha_ajuste.columns[datas_ajuste > dia_compra]  # estritamente > D0
                        soma_ajustes_unit = linha_ajuste[colunas_uteis].sum(axis=1).values[0] if len(colunas_uteis) else 0.0

                    rendimento = (pnl_d0_unit + soma_ajustes_unit) * quantidade
                    rendimentos.append(rendimento)

                subdf['Rendimento'] = rendimentos
                rendimento_final = sum(rendimentos)
            else:
                rendimento_final = (preco_ajuste_atual -
                                    preco_medio_compra) * qtd_total

            return pd.Series({
                'Quantidade': qtd_total,
                'Dia de Compra': subdf.iloc[-1]['Dia de Compra'],
                'Preço de Compra Médio': preco_medio_compra,
                'Preço de Ajuste Atual (Mais Recente)': preco_ajuste_atual,
                'Rendimento': rendimento_final
            })

        df_compilado = (
            df_teste
            .groupby('Ativo', as_index=False)
            .apply(agrupar_por_ativo)
        )

        st.table(df_compilado.set_index('Ativo'))

    st.write("---")

    # --------------------------------------------------------------------
    # Lógica para df_contratos etc.
    # --------------------------------------------------------------------
    df_contratos_2 = read_atual_contratos()
    for col in df_contratos_2.columns:
        df_contratos_2.rename(columns={col: f'Contratos {col}'}, inplace=True)

    df_contratos = df_contratos.apply(pd.to_numeric, errors='coerce')
    df_contratos_2 = df_contratos_2.apply(pd.to_numeric, errors='coerce')
    df_contratos = df_contratos.add(df_contratos_2, fill_value=0)

    df_contratos.drop(['Adm', 'PL', 'PL_atualizado', 'Weights'],
                      axis=1, inplace=True, errors='ignore')
    df_teste['Dia de Compra'] = pd.to_datetime(
        df_teste['Dia de Compra'], format='%Y-%m-%d', errors='coerce')

    opp = st.sidebar.checkbox("Verificar visão do portifólio", value=False)
    if opp:
        col_pp1, col_pp3, col_pp2 = st.columns([4.9, 0.2, 4.9])
        with col_pp1:
            st.write("## Portfólio Atual (visão consolidada)")
            soma_pl_sem_pesos = calcular_metricas_de_port(
                assets_atual, quantidades_atual, df_contratos_2)
        with col_pp2:
            st.write("## Novo Portfólio (visão consolidada)")
            # Vamos agrupar para exibir a soma final de df_compilado ou df_teste
            # Para simplificar, usaremos os Ativos e Quantidades totais de df_compilado
            # (agrupado, mas sem re-somar por dia).
            st.write(df_compilado)
            df_final_group = df_compilado.groupby('Ativo', as_index=False)[
                'Quantidade'].sum()
            assets_teste = df_final_group['Ativo'].tolist()
            quantidades_teste = df_final_group['Quantidade'].tolist()

            soma_pl_sem_pesos_novo = calcular_metricas_de_port(
                assets_teste, quantidades_teste, df_contratos)
        with col_pp3:
            st.html(
                '''
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 2px solid rgba(49, 51, 63, 0.2);
                        height: 110vh;
                        margin: auto;
                    }
                    @media (max-width: 768px) {
                        .divider-vertical-line {
                            display: none;
                        }
                    }
                </style>
                '''
            )
        st.write("---")
    else:
        # Sua lógica original de PL, etc.
        file_pl = "Dados/pl_fundos.parquet"
        df_pl = pd.read_parquet(file_pl)
        df_pl = df_pl.set_index(df_pl.columns[0])
        file_bbg = "Dados/BBG - ECO DASH.xlsx"
        dict_pesos = {
            'GLOBAL BONDS': 4,
            'HORIZONTE': 1,
            'JERA2026': 1,
            'REAL FIM': 1,
            'BH FIRF INFRA': 1,
            'BORDEAUX INFRA': 1,
            'TOPAZIO INFRA': 1,
            'MANACA INFRA FIRF': 1,
            'AF DEB INCENTIVADAS': 3
        }
        for idx, row in df_contratos_2.iterrows():
            if idx == 'Total':
                continue
            else:
                fundo = idx
                check = 0
                for asset in assets_atual:
                    # Se ainda existe no df_contratos_2 com quantity != 0
                    if int(row.get(f'Contratos {asset}', 0)) != 0:
                        check = 1
                if check == 0:
                    dict_pesos[fundo] = 0
        Weights = list(dict_pesos.values())
        df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio(
            df_pl, Weights)

    st.write("## Analise por Fundo")
    st.write("### Selecione os filtros")
    lista_fundos = df_contratos.index.tolist()
    lista_fundos = [str(x)
                    for x in df_contratos.index.tolist() if str(x) != 'Total']

    colll1, colll2 = st.columns([4.9, 4.9])
    with colll1:
        fundos = st.multiselect(
            "Selecione os fundos que deseja analisar", lista_fundos, default=lista_fundos)
    with colll2:
        op1 = st.checkbox("CoVaR / % Risco Total", value=False)
        op2 = st.checkbox("Div01 / Stress", value=False)

    cool1, cool3, cool2 = st.columns([4.9, 0.2, 4.9])
    with cool1:
        fundos0 = fundos.copy()
        if fundos0:
            st.write("## Portfólio Atual (original, consolidado)")
            soma_pl_sem_pesos2 = calcular_metricas_de_fundo(
                assets_atual,
                quantidades_atual,
                df_contratos_2,
                fundos0,
                op1,
                op2
            )
    with cool2:
        fundos1 = fundos.copy()
        # Tirar horas do df_teste
        df_teste['Dia de Compra'] = pd.to_datetime(
            df_teste['Dia de Compra'], format='%Y-%m-%d', errors='coerce')

        if fundos1:
            st.write("## Novo Portfólio (após inserções)")
            # Aqui podemos usar os ativos/quantidades do df_compilado (ou agrupar novamente)
            df_final_group = df_compilado.groupby('Ativo', as_index=False)[
                'Quantidade'].sum()
            assets_teste = df_final_group['Ativo'].tolist()
            quantidades_teste = df_final_group['Quantidade'].tolist()

            soma_pl_sem_pesos2_novo = calcular_metricas_de_fundo(
                assets_teste,
                quantidades_teste,
                df_contratos,
                fundos1,
                op1,
                op2
            )

        # Botão para salvar
        #
        st.write("### Salvar novo portfólio compilado")
        if st.button("Salvar novo portfólio"):
            # Tirar horas do df_teste
            df_teste['Dia de Compra'] = pd.to_datetime(
                df_teste['Dia de Compra'], format='%Y-%m-%d', errors='coerce')
            df_teste['Dia de Compra'] = df_teste['Dia de Compra'].astype(str)

            max_id = df_teste['Id'].max()
            # se estiver totalmente nula, defina como 0
            if pd.isna(max_id):
                max_id = 0

            # contar quantos NaNs existem
            num_missing = df_teste['Id'].isna().sum()

            # criar novos IDs sequenciais a partir do max_id
            new_ids = range(int(max_id) + 1, int(max_id) + 1 + num_missing)

            # substituir os NaNs com esses novos valores
            df_teste.loc[df_teste['Id'].isna(), 'Id'] = list(new_ids)

            # garantir que a coluna seja inteira, se quiser
            df_teste['Id'] = df_teste['Id'].astype(int)
            # df_compilado contém a versão agrupara por (Ativo, Dia de Compra)
            df_teste.to_parquet(nome_arquivo_portifolio, index=False)

            # Caso você deseje salvar a versão "bruta" também,
            # troque "df_compilado" por "df_teste" ou junte ambos.
            add_data(df_teste.to_dict(orient="records"))

            st.success("Novo portfólio compilado salvo com sucesso!")
            st.dataframe(df_teste)
            key = True
        else:
            key = False

    with cool3:
        st.html(
            '''
            <div class="divider-vertical-line"></div>
            <style>
                .divider-vertical-line {
                    border-left: 2px solid rgba(49, 51, 63, 0.2);
                    height: 80vh;
                    margin: auto;
                }
                @media (max-width: 768px) {
                    .divider-vertical-line {
                        display: none;
                    }
                }
            </style>
            '''
        )
    st.write("---")

    return df_teste, key, soma_pl_sem_pesos


def read_atual_contratos():
    files = os.listdir('BaseFundos')
    df_fundos = pd.DataFrame()
    lista_files = []
    for file in files:
        df_fundos = pd.concat(
            [df_fundos, pd.read_parquet(f'BaseFundos/{file}')])
        # Adiciona o nome do arquivo como uma linha da coluna Fundo
        contagem = df_fundos['Ativo'].unique()
        contagem = len(contagem)
        for i in range(contagem):
            lista_files.append(file.split('.')[0])
    # Pegar a quantidade de ativos unicos
    df_fundos['Fundo'] = lista_files
    df_fundos = df_fundos.reset_index(drop=True)
    contratos = []
    for fundo in df_fundos['Fundo'].unique():
        ativos = df_fundos[df_fundos['Fundo'] == fundo]
        col_quantidade = [
            col for col in ativos.columns if col.endswith('Quantidade')]

        # Agrupar por ativo e somar as quantidades
        agrupados = ativos.groupby('Ativo')[col_quantidade].sum().reset_index()

        for idx, row in agrupados.iterrows():
            # Soma total das colunas 'Quantidade'
            quantidade = row[col_quantidade].sum()
            contratos.append([fundo, row['Ativo'], quantidade])

    # DF_CONTRATOS TEM O FUNDO COMO INDEX, ATIVO COMO COLUNA E A QUANTIDADE DE CONTRATOS COMO VALOR
    df_contratos = pd.DataFrame(
        contratos, columns=['Fundo', 'Ativo', 'Quantidade'])
    df_contratos = df_contratos.pivot(
        index='Fundo', columns='Ativo', values='Quantidade')
    df_contratos = df_contratos.fillna(0)

    return df_contratos


def att_portifosições():
    df_fechamento_b3 = processar_b3_portifolio()

    # Obtendo o valor do dólar para conversão do Treasury
    dolar = df_fechamento_b3.loc[df_fechamento_b3['Assets']
                                 == 'WDO1', df_fechamento_b3.columns[-1]].values[0]

    df_portifolio = pd.read_parquet('Dados/portifolio_posições.parquet')

    # Criar dicionário com os preços de fechamento por ativo
    fechamento_dict = df_fechamento_b3.set_index(
        'Assets')[df_fechamento_b3.columns[-1]].to_dict()

    # Atualizar preço de ajuste considerando múltiplas ocorrências de ativos
    df_portifolio['Preço de Ajuste Atual'] = df_portifolio['Ativo'].map(
        fechamento_dict)

    # Carregar o DataFrame de ajustes (caso tenha DAPs)
    df_ajuste = pd.read_parquet('Dados/df_valor_ajuste_contrato.parquet')
    # Ver se a ultima coluna é igual a penultima
    if df_ajuste.iloc[:, -1].equals(df_ajuste.iloc[:, -2]):
        df_ajuste = df_ajuste.iloc[:, :-1]
    colunas_datas = df_ajuste.columns[1:]
    df_ajuste[colunas_datas] = df_ajuste[colunas_datas].replace(
        '\.', '', regex=True).replace(',', '.', regex=True)
    df_ajuste[colunas_datas] = df_ajuste[colunas_datas].astype(float)

    # Converter nomes das colunas para datetime
    datas_convertidas = pd.to_datetime(colunas_datas, errors='coerce')
    colunas_datas_validas = [col for col, data in zip(
        df_ajuste.columns[1:], datas_convertidas) if pd.notnull(data)]
    df_ajuste = df_ajuste[['Assets'] + colunas_datas_validas]
    datas_validas = pd.to_datetime(colunas_datas_validas)

    # Função para calcular rendimento linha a linha
    def calcular_rendimento(row):
        preco_compra = row['Preço de Compra']
        preco_de_ajuste = row['Preço de Ajuste Atual']
        quantidade = row['Quantidade']
        ativo = row['Ativo']

        if ativo == 'TREASURY':
            return quantidade * (preco_de_ajuste - preco_compra) * (dolar / 10000)

        elif 'DAP' in ativo:
            dia_compra = pd.to_datetime(row['Dia de Compra'])

            # Pega linha do ativo no df_ajuste
            linha_ajuste = df_ajuste[df_ajuste['Assets']
                                     == ativo].drop(columns='Assets')

            if linha_ajuste.empty:
                return 0  # Ativo não encontrado

            # Selecionar colunas com datas >= data de compra
            colunas_uteis = linha_ajuste.columns[datas_validas > dia_compra]

            # Soma os ajustes após a data de compra e multiplica pela quantidade
            return linha_ajuste[colunas_uteis].sum(axis=1).values[0] * quantidade
        
        elif 'DI' in ativo:
            dia_compra = pd.to_datetime(row['Dia de Compra'])

            # Pega linha do ativo no df_ajuste
            linha_ajuste = df_ajuste[df_ajuste['Assets'] == ativo].drop(columns='Assets')
            if linha_ajuste.empty:
                return 0  # Ativo não encontrado

            # 1) PnL do DIA DA COMPRA (D0)
            pnl_d0 = (preco_de_ajuste - preco_compra) * quantidade

            # 2) Ajustes a partir do DIA SEGUINTE (D+1)
            colunas_uteis = linha_ajuste.columns[datas_validas > dia_compra]
            soma_ajustes = linha_ajuste[colunas_uteis].sum(axis=1).values[0] if len(colunas_uteis) else 0

            return pnl_d0 + soma_ajustes * quantidade


        else:
            return quantidade * (preco_de_ajuste - preco_compra)

    df_portifolio['Rendimento'] = df_portifolio.apply(
        calcular_rendimento, axis=1)

    max_id = df_portifolio['Id'].max()

    # se estiver totalmente nula, defina como 0
    if pd.isna(max_id):
        max_id = 0

    # contar quantos NaNs existem
    num_missing = df_portifolio['Id'].isna().sum()

    # criar novos IDs sequenciais a partir do max_id
    new_ids = range(int(max_id) + 1, int(max_id) + 1 + num_missing)

    # substituir os NaNs com esses novos valores
    df_portifolio.loc[df_portifolio['Id'].isna(), 'Id'] = list(new_ids)

    # garantir que a coluna seja inteira, se quiser
    df_portifolio['Id'] = df_portifolio['Id'].astype(int)

    # Salvar parquet atualizado
    df_portifolio.to_parquet('Dados/portifolio_posições.parquet', index=False)

    # Chamar a função add_data(data) para atualizar portifolio_posições no banco de dados
    add_data(df_portifolio.to_dict(orient="records"))

    return

def calcular_metricas_de_fundo3(assets, df_contratos, fundos, op1=True, op2=True):

    # ---------------- helpers locais ----------------
    def _ativos_por(assets, quantidade_nomes, *contains):
        out = []
        for a in assets:
            if any(s in a for s in contains) and float(quantidade_nomes.get(a, 0)) != 0:
                out.append(a)
        return out

    def _get_or_zero(series_like, key):
        try:
            if isinstance(series_like, pd.Series):
                return float(series_like.get(key, 0.0))
            elif isinstance(series_like, pd.DataFrame):
                if key in series_like.index:
                    return float(series_like.loc[key].squeeze())
                return 0.0
            else:
                return float(series_like)
        except Exception:
            return 0.0

    # ---------------- pré-processamento ----------------
    df_tira = df_contratos.copy()
    df_tira.reset_index(inplace=True)
    df_tira = df_tira[df_tira.columns.drop(list(df_tira.filter(regex='Max|Adm')))]
    df_tira.rename(columns={'index': 'Fundo'}, inplace=True)

    lista_remove = []
    for fundo2 in fundos[:]:
        linha = df_tira[df_tira['Fundo'] == fundo2].select_dtypes(include=['number'])
        if not linha.empty and (linha == 0).all(axis=1).values[0]:
            lista_remove.append(fundo2)
    for f in lista_remove:
        if f in fundos:
            fundos.remove(f)

    if not fundos:
        st.write("Nenhum fundo selecionado / Nenhum contrato cadastrado")
        return

    # ---------------- cargas e pesos ----------------
    file_pl = "Dados/pl_fundos.parquet"
    df_pl = pd.read_parquet(file_pl)
    file_bbg = "Dados/BBG - ECO DASH.xlsx"

    dict_pesos = {
        'GLOBAL BONDS': 4, 'HORIZONTE': 1, 'JERA2026': 1, 'REAL FIM': 1,
        'BH FIRF INFRA': 1, 'BORDEAUX INFRA': 1, 'TOPAZIO INFRA': 1,
        'MANACA INFRA FIRF': 1, 'AF DEB INCENTIVADAS': 3
    }

    for idx, row in df_contratos.iterrows():
        if str(idx).strip().upper() == 'TOTAL':
            continue
        check = 0
        for asset in assets:
            col = f'Contratos {asset}'
            if col in row.index and pd.notna(row[col]) and int(row[col]) != 0:
                check = 1
                break
        if check == 0 and idx in dict_pesos:
            dict_pesos[idx] = 0

    Weights = list(dict_pesos.values())

    df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio(df_pl, Weights)

    df = pd.read_parquet('Dados/df_inicial.parquet')
    df_precos, df_completo = load_and_process_excel(df, assets)
    df_retorno = process_returns(df_completo, assets)
    var_ativos = var_not_parametric(df_retorno).abs()
    df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)

    quantidade_nomes = {}
    tabela_dados_fundos = pd.DataFrame()
    tabela_dados_riscos = pd.DataFrame()
    df_portfolio_final = pd.DataFrame()

    # ---------------- loop nos fundos ----------------
    for idx, row in df_contratos.iterrows():
        if idx not in fundos:
            continue

        df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio_especifico(df_pl, Weights, idx)
        soma_pl = soma_pl * 0.01  # converte para porcentagem
        # >>> NOVO: 1% do PL do fundo como base (pl_ref) <<<
        pl_ref = (soma_pl_sem_pesos or 0.0) * 0.01  # evita None

        quantidade_nomes = {}
        for a in assets:
            col = f'Contratos {a}'
            quantidade_nomes[a] = row[col] if col in row.index and pd.notna(row[col]) else 0.0

        quantidade = np.array([quantidade_nomes[a] for a in assets], dtype=float)

        if "Valor Fechamento" not in df_precos_ajustados.columns:
            st.write("Coluna 'Valor Fechamento' não encontrada em df_precos_ajustados.")
            continue

        vp = df_precos_ajustados['Valor Fechamento'].reindex(assets).fillna(0.0).values * np.abs(quantidade)
        vp_soma = float(np.nansum(vp))
        if vp_soma == 0:
            continue

        pesos = (quantidade * df_precos_ajustados['Valor Fechamento'].reindex(assets).fillna(0.0).values) / vp_soma

        df_returns_portifolio = (df_retorno.reindex(columns=assets).fillna(0.0)
                                 * pd.Series(pesos, index=assets)).copy()
        df_returns_portifolio['Portifolio'] = df_returns_portifolio.sum(axis=1)

        var_port = abs(var_not_parametric(df_returns_portifolio['Portifolio']))
        var_port_dinheiro = vp_soma * var_port

        var_bps = 1.0 / 10000.0
        var_limite = 1.0

        vol_port_retornos = df_returns_portifolio['Portifolio'].std()
        vol_port_analitica = vol_port_retornos * np.sqrt(252)

        df_retorno['Portifolio'] = df_returns_portifolio['Portifolio']
        cov = df_retorno.cov()
        cov_port = cov['Portifolio'].drop('Portifolio')
        df_beta = cov_port / (vol_port_retornos**2 if vol_port_retornos != 0 else np.nan)
        df_mvar = df_beta * var_port
        df_mvar_dinheiro = df_mvar * df_precos_ajustados['Valor Fechamento'].reindex(df_mvar.index).fillna(0.0)

        covar = df_mvar.reindex(df_precos_ajustados.index).fillna(0.0) \
                * pd.Series(pesos, index=assets).reindex(df_mvar.index).fillna(0.0).values \
                * vp_soma
        covar_perc = covar / covar.sum() if covar.sum() != 0 else covar

        cvar = abs(df_retorno[df_retorno['Portifolio'] < var_not_parametric(df_retorno['Portifolio'])]['Portifolio'].mean())
        cvar_dinheiro = vp_soma * cvar

        # -------------- DV01 / Stress --------------
        df_divone, dolar, treasury = load_and_process_divone2(file_bbg, df_completo)   
        
        lista_juros_interno      = _ativos_por(assets, quantidade_nomes, "DI")
        lista_juros_interno_real = _ativos_por(assets, quantidade_nomes, "DAP", "NTNB")
        lista_juros_externo      = _ativos_por(assets, quantidade_nomes, "TREASURY")
        lista_dolar              = _ativos_por(assets, quantidade_nomes, "WDO1")

        base_idx = df_divone.index
        zero_series = pd.Series(0.0, index=base_idx)

        if lista_juros_interno:
            q_nom = np.array([quantidade_nomes[a] for a in lista_juros_interno], dtype=float)
            df_divone_juros_nominais = (df_divone[lista_juros_interno] * q_nom).sum(axis=1)
        else:
            df_divone_juros_nominais = zero_series.copy()

        if lista_juros_interno_real:
            q_real = np.array([quantidade_nomes[a] for a in lista_juros_interno_real], dtype=float)
            df_divone_juros_real = (df_divone[lista_juros_interno_real] * q_real).sum(axis=1)
        else:
            df_divone_juros_real = zero_series.copy()

        if lista_juros_externo:
            q_ext = np.array([quantidade_nomes[a] for a in lista_juros_externo], dtype=float)
            df_divone_juros_externo = (df_divone[lista_juros_externo] * q_ext).sum(axis=1)
            valor_acumulado_treasury = (1 + df_retorno['TREASURY']).cumprod()
            pico_max_treasury = valor_acumulado_treasury.max()
            drawndown_treasury = (valor_acumulado_treasury - pico_max_treasury) / pico_max_treasury
            drawndown_treasury = drawndown_treasury.min()
            drawndown_treasury = df_retorno['TREASURY'].min()
            stress_test_juros_externo = abs(drawndown_treasury) * treasury * dolar / 10000 * float(q_ext.sum())
        else:
            df_divone_juros_externo = zero_series.copy()
            stress_test_juros_externo = 0.0

        if lista_dolar:
            quantidade_dolar = float(quantidade_nomes[lista_dolar[0]])
            valor_acumulado = (1 + df_retorno['WDO1']).cumprod()
            pico_max = valor_acumulado.max()
            drawndown_dolar = (valor_acumulado - pico_max) / pico_max
            drawndown_dolar = drawndown_dolar.min()
            drawndown_dolar = df_retorno['WDO1'].min()
            df_divone_dolar = abs(drawndown_dolar) * dolar * quantidade_dolar
            stress_dolar = float(df_divone_dolar)
        else:
            df_divone_dolar = 0.0
            stress_dolar = 0.0

        # >>> NOVO: normalização por 1% do PL do fundo (pl_ref) <<<
        denom = pl_ref if pl_ref not in (None, 0) else np.nan

        stress_test_juros_interno_Nominais = df_divone_juros_nominais * 100
        stress_test_juros_interno_Reais    = df_divone_juros_real * 50

        if np.isnan(denom):
            stress_test_juros_interno_Nominais_percent = zero_series.copy()
            stress_test_juros_interno_Reais_percent    = zero_series.copy()
            stress_test_juros_externo_percent          = 0.0
            stress_dolar_percent                       = 0.0
        else:
            stress_test_juros_interno_Nominais_percent = stress_test_juros_interno_Nominais / denom * 10000
            stress_test_juros_interno_Reais_percent    = stress_test_juros_interno_Reais / denom * 10000
            stress_test_juros_externo_percent          = stress_test_juros_externo / denom * 10000
            stress_dolar_percent                       = stress_dolar / denom * 10000

        df_divone_juros_externo_certo = df_divone_juros_externo

        df_stress_div01 = pd.DataFrame({
            'DIV01': [
                f"R${abs(_get_or_zero(df_divone_juros_nominais, 'FUT_TICK_VAL')):,.2f}",
                f"R${abs(_get_or_zero(df_divone_juros_real,     'FUT_TICK_VAL')):,.2f}",
                f"R${abs(_get_or_zero(df_divone_juros_externo_certo, 'FUT_TICK_VAL')):,.2f}",
                f"R${abs(df_divone_dolar):,.2f}" if lista_dolar else 0
            ],
            'Stress (R$)': [
                f"R${abs(_get_or_zero(stress_test_juros_interno_Nominais, 'FUT_TICK_VAL')):,.2f}",
                f"R${abs(_get_or_zero(stress_test_juros_interno_Reais,     'FUT_TICK_VAL')):,.2f}",
                f"R${abs(stress_test_juros_externo):,.2f}",
                f"R${abs(stress_dolar):,.2f}"
            ],
            # >>> NOVO: bps sobre 1% do PL (pl_ref) <<<
            'Stress (bps)': [
                f"{abs(_get_or_zero(stress_test_juros_interno_Nominais_percent, 'FUT_TICK_VAL')):,.2f}bps",
                f"{abs(_get_or_zero(stress_test_juros_interno_Reais_percent,     'FUT_TICK_VAL')):,.2f}bps",
                f"{abs(stress_test_juros_externo_percent):,.2f}bps",
                f"{abs(stress_dolar_percent):,.2f}bps"
            ]
        }, index=['Juros Nominais Brasil', 'Juros Reais Brasil', 'Juros US', 'Moedas'])

        sum_row = pd.DataFrame({
            'DIV01': [
                f"R${abs(_get_or_zero(df_divone_juros_nominais, 'FUT_TICK_VAL'))
                   + abs(_get_or_zero(df_divone_juros_real, 'FUT_TICK_VAL'))
                   + abs(_get_or_zero(df_divone_juros_externo_certo, 'FUT_TICK_VAL'))
                   + (abs(df_divone_dolar) if lista_dolar else 0):,.2f}"
            ],
            'Stress (R$)': [
                f"R${abs(_get_or_zero(stress_test_juros_interno_Nominais, 'FUT_TICK_VAL'))
                   + abs(_get_or_zero(stress_test_juros_interno_Reais, 'FUT_TICK_VAL'))
                   + abs(stress_test_juros_externo)
                   + abs(stress_dolar):,.2f}"
            ],
            'Stress (bps)': [
                f"{abs(_get_or_zero(stress_test_juros_interno_Nominais_percent, 'FUT_TICK_VAL'))
                   + abs(_get_or_zero(stress_test_juros_interno_Reais_percent, 'FUT_TICK_VAL'))
                   + abs(stress_test_juros_externo_percent)
                   + abs(stress_dolar_percent):,.2f}bps"
            ]
        }, index=['Total'])
        df_stress_div01 = pd.concat([df_stress_div01, sum_row])

        # --- métricas agregadas por fundo (bps sobre 1% do PL) ---
        var_limite_comparativo = soma_pl * var_bps * var_limite  # em R$
        dados_fundo = {
            'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
            # >>> NOVO: divide por pl_ref
            'VaR Limite (bps)':     f"{(var_limite_comparativo / pl_ref * 10000):,.2f}bps" if pl_ref else "0.00bps",
            'VaR Portfólio (bps)':  f"{(var_port_dinheiro    / pl_ref * 10000):,.2f}bps" if pl_ref else "0.00bps",
            'CVaR (bps)':           f"{(abs(cvar * vp_soma)  / pl_ref * 10000):,.2f}bps" if pl_ref else "0.00bps",
            'Volatilidade': f"{vol_port_analitica:.2%}",
            '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}" if var_limite_comparativo else "0.00%"
        }
        df_portfolio_final = pd.concat([df_portfolio_final, pd.DataFrame(dados_fundo, index=[idx])])

        # ---------- tabela de risco por ativo (mantida) ----------
        df_dados = pd.DataFrame({
            'Beta': df_beta,
            'MVar(R$)': df_mvar_dinheiro,
            'CoVaR(R$)': covar,
            'CoVaR(%)': covar_perc,
            'Var': var_ativos[assets],
            '% do Risco Total': covar_perc * abs(covar.sum() / var_limite_comparativo) if var_limite_comparativo else 0.0
        })

        dados_formatados = {}
        for ativo in df_dados.index.tolist():
            linha = df_dados.loc[ativo]
            valores_formatados = " / ".join([
                f"{linha['CoVaR(%)'] * 100:.2f}%",
                f"{linha['% do Risco Total']* 100:.2f}%",
            ])
            dados_formatados[ativo] = [valores_formatados]

        dados_formatados['Total'] = [" / ".join([
            f"{df_dados['CoVaR(%)'].sum() * 100:.2f}%",
            f"{(df_dados['% do Risco Total']).sum() * 100:.2f}%",
        ])]

        tabela_dados_risco = pd.DataFrame(dados_formatados, index=[idx])
        tabela_dados_riscos = pd.concat([tabela_dados_riscos, tabela_dados_risco], axis=0)

        colunas_selecionadas = ['Beta', 'MVar(R$)', 'CoVaR(R$)', 'CoVaR(%)', 'Var', '% do Risco Total']
        if 'CoVaR(R$)' in colunas_selecionadas:
            df_dados['CoVaR(R$)'] = df_dados['CoVaR(R$)'].apply(lambda x: f"R${x:,.0f}")
        if 'MVar(R$)' in colunas_selecionadas:
            df_dados['MVar(R$)'] = df_dados['MVar(R$)'].apply(lambda x: f"R${x:,.0f}")
        if 'CoVaR(%)' in colunas_selecionadas:
            df_dados['CoVaR(%)'] = df_dados['CoVaR(%)'].apply(lambda x: f"{x:.2%}")
        if 'Beta' in colunas_selecionadas:
            df_dados['Beta'] = df_dados['Beta'].apply(lambda x: f"{x:.4f}")
        if 'Var' in colunas_selecionadas:
            df_dados['Var'] = df_dados['Var'].apply(lambda x: f"{x:.4f}%")
        if '% do Risco Total' in colunas_selecionadas:
            df_dados['% do Risco Total'] = df_dados['% do Risco Total'].apply(lambda x: f"{x:.2%}")

        # ----- tabela de estratégias (DIV01 bps / Stress bps) -----
        dados = {
            'Juros Nominais Brasil': [
                f"{abs(_get_or_zero(df_divone_juros_nominais, 'FUT_TICK_VAL') / (pl_ref or 1) * 10000):,.2f}bps / "
                f"{abs(_get_or_zero(stress_test_juros_interno_Nominais_percent, 'FUT_TICK_VAL')):,.2f}bps"
            ],
            'Juros Reais Brasil': [
                f"{abs(_get_or_zero(df_divone_juros_real, 'FUT_TICK_VAL') / (pl_ref or 1) * 10000):,.2f}bps / "
                f"{abs(_get_or_zero(stress_test_juros_interno_Reais_percent, 'FUT_TICK_VAL')):,.2f}bps"
            ],
            'Juros US': [
                f"{abs(_get_or_zero(df_divone_juros_externo_certo, 'FUT_TICK_VAL') / (pl_ref or 1) * 10000):,.2f}bps / "
                f"{abs(stress_test_juros_externo_percent):,.2f}bps"
            ],
            'Moedas': [
                f"{(abs(df_divone_dolar) / (pl_ref or 1) * 10000) if lista_dolar else 0:,.2f}bps / "
                f"{abs(stress_dolar_percent):,.2f}bps"
            ]
        }
        tabela_dados_fundo = pd.DataFrame(dados, index=[idx])
        tabela_dados_fundos = pd.concat([tabela_dados_fundos, tabela_dados_fundo], axis=0)

    # ---------------- pós-loop: totais/relatórios ----------------
    #st.table(df_portfolio_final)

    tabela_dados_fundos_p1 = tabela_dados_fundos.applymap(lambda x: x.replace('bps', '') if isinstance(x, str) else x)
    tabela_dados_fundos_p1 = tabela_dados_fundos_p1.applymap(lambda x: x.split('/')[0] if isinstance(x, str) and '/' in x else x).astype(float)
    tabela_dados_fundos_p1['Total'] = tabela_dados_fundos_p1.sum(axis=1).round(2)

    tabela_dados_fundos_p2 = tabela_dados_fundos.applymap(lambda x: x.replace('bps', '') if isinstance(x, str) else x)
    tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(lambda x: x.split('/')[1] if isinstance(x, str) and '/' in x else x)
    tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(lambda x: float(x) if isinstance(x, str) and x != '0' else 0)
    tabela_dados_fundos_p2['Total'] = tabela_dados_fundos_p2[['Juros Nominais Brasil', 'Juros Reais Brasil', 'Moedas', 'Juros US']].sum(axis=1).round(2)

    tabela_dados_fundos['Total'] = tabela_dados_fundos_p1['Total'].astype(str) + 'bps / ' + tabela_dados_fundos_p2['Total'].astype(str) + 'bps'

    mapeamento_categorias = {
        "Juros Nominais Brasil": [col for col in tabela_dados_riscos.columns if "DI" in col],
        "Juros Reais Brasil":    [col for col in tabela_dados_riscos.columns if ("DAP" in col) or ("NTNB" in col)],
        "Juros US":              ["TREASURY"],
        "Moedas":                ["WDO1"]
    }

    nova_tabela = pd.DataFrame()
    nova_tabela["Fundos"] = tabela_dados_riscos.index
    soma_total_antes, soma_total_depois = [], []

    for categoria, colunas in mapeamento_categorias.items():
        valores_antes, valores_depois = [], []
        for _, row in tabela_dados_riscos.iterrows():
            soma_antes = 0.0; soma_depois = 0.0
            for col in colunas:
                if col in tabela_dados_riscos.columns and isinstance(row[col], str) and " / " in row[col]:
                    partes = row[col].split(" / ")
                    try:
                        soma_antes  += float(partes[0].replace("%", ""))
                        soma_depois += float(partes[1].replace("%", ""))
                    except Exception:
                        pass
            valores_antes.append(soma_antes)
            valores_depois.append(soma_depois)

        nova_tabela[categoria] = [f"{a:.2f}% / {b:.2f}%" for a, b in zip(valores_antes, valores_depois)]

        if not soma_total_antes:
            soma_total_antes = valores_antes
            soma_total_depois = valores_depois
        else:
            soma_total_antes  = [x + y for x, y in zip(soma_total_antes, valores_antes)]
            soma_total_depois = [x + y for x, y in zip(soma_total_depois, valores_depois)]

    nova_tabela["Total"] = [f"{a:.2f}% / {b:.2f}%" for a, b in zip(soma_total_antes, soma_total_depois)]
    nova_tabela.set_index("Fundos", inplace=True)

    if op1:
        st.write("### Analise Risco por Categoria")
        st.table(nova_tabela)
        st.markdown("<p style='font-size: 13px; font-style: italic;'>(CoVaR bps / % Risco Total)</p>", unsafe_allow_html=True)

    if op2:
        st.write("### Analise Estratégias")
        st.table(tabela_dados_fundos)
        st.markdown("<p style='font-size: 13px; font-style: italic;'>(Div01 bps / Stress bps)</p>", unsafe_allow_html=True)

def calcular_metricas_de_fundo2(assets, df_contratos, fundos, op1 = True, op2 = True):
    df_tira = df_contratos.copy()
    df_tira.reset_index(inplace=True)
    # Tirar colunas que contenham o nome 'Max' ou 'Adm'
    df_tira = df_tira[df_tira.columns.drop(
        list(df_tira.filter(regex='Max')))]
    df_tira.rename(columns={'index': 'Fundo'}, inplace=True)
    lista_remove = []
    for fundo2 in fundos[:]:
        # Filtrar a linha do fundo
        linha = df_tira[df_tira['Fundo'] == fundo2].select_dtypes(include=['number'])

        # Verificar se todas as colunas são zero
        # `axis=1` verifica todas as colunas na linha
        if (linha == 0).all(axis=1).values[0]:
            lista_remove.append(fundo2)
    for fundo in lista_remove:
        fundos.remove(fundo)
    if fundos:
        # st.write(df_contratos)
        file_pl = "Dados/pl_fundos.parquet"
        df_pl = pd.read_parquet(file_pl)
        file_bbg = "Dados/BBG - ECO DASH.xlsx"

        # Dicionário de pesos fixo (pode-se tornar dinâmico no futuro)
        dict_pesos = {
            'GLOBAL BONDS': 4,
            'HORIZONTE': 1,
            'JERA2026': 1,
            'REAL FIM': 1,
            'BH FIRF INFRA': 1,
            'BORDEAUX INFRA': 1,
            'TOPAZIO INFRA': 1,
            'MANACA INFRA FIRF': 1,
            'AF DEB INCENTIVADAS': 3
        }
        # Zerar os pesos de fundos que não tem contratos
        for idx, row in df_contratos.iterrows():
            if idx == 'Total':
                continue
            else:
                fundo = idx
                check = 0
                for asset in assets:
                    if int(row[f'Contratos {asset}']) != 0:
                        check = 1
                if check == 0:
                    dict_pesos[fundo] = 0

        Weights = list(dict_pesos.values())

        df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio(
            df_pl, Weights)

        df = pd.read_parquet('Dados/df_inicial.parquet')
        df_precos, df_completo = load_and_process_excel(df, assets)
        df_retorno = process_returns(df_completo, assets)

        var_ativos = var_not_parametric(df_retorno).abs()
        df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)
        quantidade_nomes = {}
        tabela_dados_fundos = pd.DataFrame()
        tabela_dados_riscos = pd.DataFrame()
        # Antes do loop principal, inicialize o dicionário de dados:

        df_portfolio_final = pd.DataFrame()
        for idx, row in df_contratos.iterrows():
            if idx in fundos:
                df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio_especifico(
                    df_pl, Weights, idx)
                for i in range(len(assets)):
                    quantidade_nomes[assets[i]] = row[f'Contratos {assets[i]}']
                quantidade = np.array(list(quantidade_nomes.values()))
                vp = df_precos_ajustados['Valor Fechamento'] * abs(quantidade)
                vp_soma = vp.sum()
                if vp_soma == 0:
                    pass
                else:

                    pesos = quantidade * \
                        df_precos_ajustados['Valor Fechamento'] / vp_soma
                    df_returns_portifolio = df_retorno * pesos
                    df_returns_portifolio['Portifolio'] = df_returns_portifolio.sum(
                        axis=1)

                    # VaR
                    var_port = var_not_parametric(
                        df_returns_portifolio['Portifolio'])
                    var_port = abs(var_port)
                    var_port_dinheiro = vp_soma * var_port

                    # ######################################################################################
                    # ######################################################################################
                    # #####################################################################################
                    ######################################################################################
                    ######## var bps e var limite são fixos, mas podem ser dinâmicos no futuro ########################

                    var_bps = 1.0
                    var_bps = var_bps / 10000

                    var_limite = 1.0

                    vol_port_retornos = df_returns_portifolio['Portifolio'].std(
                    )
                    vol_port_analitica = vol_port_retornos * np.sqrt(252)

                    df_retorno['Portifolio'] = df_returns_portifolio['Portifolio']
                    cov = df_retorno.cov()
                    cov_port = cov['Portifolio'].drop('Portifolio')
                    df_beta = cov_port / (vol_port_retornos**2)
                    df_mvar = df_beta * var_port
                    df_mvar_dinheiro = df_mvar * \
                        df_precos_ajustados['Valor Fechamento']

                    covar = df_mvar * pesos.values * vp_soma
                    covar_perc = covar / covar.sum()

                    cvar = df_retorno[df_retorno['Portifolio'] < var_not_parametric(
                        df_retorno['Portifolio'])]['Portifolio'].mean()
                    
                    cvar = abs(cvar)

                    cvar_dinheiro = vp_soma * cvar

                    df_divone, dolar, treasury = load_and_process_divone2(
                        'Dados/BBG - ECO DASH.xlsx', df_completo)

                    lista_juros_interno = [
                        asset for asset in assets if 'DI' in asset]
                    df_divone_juros_nominais = df_divone[lista_juros_interno]

                    lista_quantidade = [quantidade_nomes[asset]
                                        for asset in lista_juros_interno]
                    df_divone_juros_nominais = df_divone_juros_nominais * \
                        np.array(lista_quantidade)
                    df_divone_juros_nominais = df_divone_juros_nominais.sum(
                        axis=1)

                    lista_juros_interno_real = [a for a in assets if ('DAP' in a) or ('NTNB' in a)]

                    df_divone_juros_real = df_divone[lista_juros_interno_real]

                    lista_quantidade = [quantidade_nomes[asset]
                                        for asset in lista_juros_interno_real]

                    df_divone_juros_real = df_divone_juros_real * \
                        np.array(lista_quantidade)

                    df_divone_juros_real = df_divone_juros_real.sum(axis=1)

                    lista_juros_externo = [
                        asset for asset in assets if 'TREASURY' in asset]

                    df_divone_juros_externo = df_divone[lista_juros_externo]

                    lista_quantidade = [quantidade_nomes[asset]
                                        for asset in lista_juros_externo]

                    df_divone_juros_externo = df_divone_juros_externo * \
                        np.array(lista_quantidade)

                    df_divone_juros_externo = df_divone_juros_externo.sum(
                        axis=1)

                    stress_test_juros_interno_Nominais = df_divone_juros_nominais * 100
                    stress_test_juros_interno_Nominais_percent = stress_test_juros_interno_Nominais / \
                        soma_pl_sem_pesos * 10000

                    stress_test_juros_interno_Reais = df_divone_juros_real * 50
                    stress_test_juros_interno_Reais_percent = stress_test_juros_interno_Reais / \
                        soma_pl_sem_pesos * 10000

                    df_divone_juros_externo_certo = df_divone_juros_externo

                    if lista_juros_externo:
                        valor_acumulado_treasury = (
                            1+df_retorno['TREASURY']).cumprod()
                        pico_max_treasury = valor_acumulado_treasury.max()
                        drawndown_treasury = (valor_acumulado_treasury - pico_max_treasury) / \
                            pico_max_treasury
                        drawndown_treasury = drawndown_treasury.min()
                        drawndown_treasury = df_retorno['TREASURY'].min()
                        df_divone_juros_externo = drawndown_treasury
                        df_divone_juros_externo = abs(
                            df_divone_juros_externo) * treasury * dolar / 10000
                        df_divone_juros_externo = df_divone_juros_externo * \
                            np.array(lista_quantidade)
                        df_divone_juros_externo = df_divone_juros_externo.sum()

                    stress_test_juros_externo = df_divone_juros_externo

                    stress_test_juros_externo_percent = stress_test_juros_externo / \
                        soma_pl_sem_pesos * 10000

                    lista_dolar = [
                        asset for asset in assets if 'WDO1' in asset]
                    if lista_dolar:
                        quantidade_dolar = quantidade_nomes[lista_dolar[0]]
                        stress_dolar = quantidade_dolar * dolar * 0.02
                        valor_acumulado = (1+df_retorno['WDO1']).cumprod()
                        pico_max = valor_acumulado.max()
                        drawndown_dolar = (
                            valor_acumulado - pico_max) / pico_max
                        drawndown_dolar = drawndown_dolar.min()
                        #st.write(f"Drawndown Dolar: {drawndown_dolar}")
                        drawndown_dolar = df_retorno['WDO1'].min()
                        #st.write(f"Drawndown Dolar: {drawndown_dolar}")
                        df_divone_dolar = drawndown_dolar
                        df_divone_dolar = df_divone_dolar * quantidade_dolar
                        df_divone_dolar = abs(df_divone_dolar) * dolar
                        stress_dolar = df_divone_dolar
                        stress_dolar_percent = stress_dolar / soma_pl_sem_pesos * 10000
                        df_divone_dolar = df_divone[lista_dolar] * \
                            np.array(quantidade_dolar)
                        df_divone_dolar = df_divone_dolar.sum()

                    else:
                        stress_dolar = 0
                        df_divone_dolar = 0

                    stress_dolar_percent = stress_dolar / soma_pl_sem_pesos * 10000
                    df_stress_div01 = pd.DataFrame({
                        'DIV01': [
                            f"R${abs(df_divone_juros_nominais.iloc[0]):,.2f}",
                            f"R${abs(df_divone_juros_real.iloc[0]):,.2f}",
                            f"R${abs(df_divone_juros_externo_certo.iloc[0]):,.2f}",
                            f'R${abs(df_divone_dolar.iloc[0]):,.2f}' if lista_dolar else 0
                        ],
                        'Stress (R$)': [
                            f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']):,.2f}",
                            f"R${abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']):,.2f}",
                            f"R${abs(stress_test_juros_externo):,.2f}" if lista_juros_externo else f"R${abs(stress_test_juros_externo['FUT_TICK_VAL']):,.2f}",
                            f"R${abs(stress_dolar):,.2f}"
                        ],
                        'Stress (bps)': [
                            f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']):,.2f}bps",
                            f"{abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']):,.2f}bps",
                            f"{abs(stress_test_juros_externo_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_externo_percent['FUT_TICK_VAL']):,.2f}bps",
                            f"{abs(stress_dolar_percent):,.2f}bps"
                        ]
                    }, index=['Juros Nominais Brasil', 'Juros Reais Brasil', 'Juros US', 'Moedas'])

                    sum_row = pd.DataFrame({
                        'DIV01': [f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real[0]) + abs(df_divone_juros_externo_certo.iloc[0]) + abs(df_divone_dolar.iloc[0]):,.2f}"] if lista_dolar else [f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real[0]) + abs(df_divone_juros_externo_certo.iloc[0]):,.2f}"],
                        'Stress (R$)': [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo) + abs(stress_dolar):,.2f}"] if lista_juros_externo else [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo['FUT_TICK_VAL']) + abs(stress_dolar):,.2f}"],
                        'Stress (bps)': [f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent) + abs(stress_dolar_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent['FUT_TICK_VAL']) + abs(stress_dolar_percent):,.2f}bps"]
                    }, index=['Total'])
                    df_stress_div01 = pd.concat([df_stress_div01, sum_row])
                    # --- Layout ---
        #            st.write("## Dados do Portfólio")
        #            st.write(f"**PL: R$ {soma_pl_sem_pesos:,.0f}**")
        #
                    var_limite_comparativo = soma_pl * var_bps * var_limite
        #            st.write(
        #                f"**VaR Limite** (Peso de {var_limite:.1%}): R${var_limite_comparativo:,.0f}"
        #            )
        #
        #            st.write(
        #                f"**VaR do Portfólio**: R${var_port_dinheiro:,.0f} : **{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps**"
        #            )
        #            st.write(
        #                f"**CVaR**: R${abs(cvar * vp_soma):,.0f} : **{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps**"
        #            )
        #            st.write(f"**Volatilidade**: {vol_port_analitica:.2%}")
        #            st.table(df_stress_div01)
        #
        #            st.write("---")
        #            st.write(
        #                f"### {abs(covar.sum()/ var_limite_comparativo):.2%} do risco total")

                    # Coletar dados formatados para cada fundo
                    dados_fundo = {
                        'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
                        'VaR Limite (bps)': f"{var_limite_comparativo/soma_pl_sem_pesos * 10000:,.2f}bps",
                        'VaR Portfólio (bps)': f"{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps",
                        'CVaR (bps)': f"{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps",
                        'Volatilidade': f"{vol_port_analitica:.2%}",
                        '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}"
                    }
        #            dados_fundo = {
        #                'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
        #                'VaR Limite (R$)': f"R${var_limite_comparativo:,.0f}",
        #                'VaR Portfólio (R$/bps)': f"R${var_port_dinheiro:,.0f} / {var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps",
        #                'CVaR (R$/bps)': f"R${abs(cvar * vp_soma):,.0f} / {abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps",
        #                'Volatilidade': f"{vol_port_analitica:.2%}",
        #                '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}"
        #            }

                    # Adicionar como nova linha ao DataFrame
                    df_portfolio_final = pd.concat([
                        df_portfolio_final,
                        pd.DataFrame(dados_fundo, index=[idx])
                    ])

                    # Exiba a tabela formatada no Streamlit:
                    # Supondo que df_dados já está criado com múltiplos índices (ativos)
                    df_dados = pd.DataFrame({
                        'Beta': df_beta,
                        'MVar(R$)': df_mvar_dinheiro,
                        'CoVaR(R$)': covar,
                        'CoVaR(%)': covar_perc,
                        'Var': var_ativos[assets],
                        '% do Risco Total': covar_perc * abs(covar.sum() / var_limite_comparativo)
                    })

                    # Lista de ativos (nomes das colunas)
                    lista_ativos = df_dados.index.tolist()

                    # Dicionário para armazenar os valores formatados de cada ativo
                    dados_formatados = {}

                    # Iterar sobre cada ativo (linha do df_dados)
                    for ativo in lista_ativos:
                        # Extrair dados do ativo
                        linha = df_dados.loc[ativo]

                        # Formatar os valores em uma string com "/"
                        valores_formatados = " / ".join([
                            # f"{linha['Beta']:.6f}",                   # Beta
                            # f"R${linha['MVar(R$)']:,.2f}",            # MVar(R$)
                            # f"R${linha['CoVaR(R$)']:,.2f}",           # CoVaR(R$)
                            # CoVaR(%)
                            f"{linha['CoVaR(%)'] * 100:.2f}%",
                            # f"R${linha['Var']:,.2f}",                 # Var
                            # % do Risco Total
                            f"{linha['% do Risco Total']* 100:.2f}%",
                        ])

                        # Adicionar ao dicionário (chave = nome do ativo)
                        dados_formatados[ativo] = [valores_formatados]
                    # Criar uma coluna de Total
                    dados_formatados['Total'] = [
                        " / ".join([
                            # f"{df_beta.sum():.6f}",                   # Beta
                            # f"R${df_mvar_dinheiro.sum():,.2f}",       # MVar(R$)
                            # f"R${covar.sum():,.2f}",                  # CoVaR(R$)
                            # CoVaR(%)
                            f"{covar_perc.sum() * 100:.2f}%",
                            # f"R${var_ativos[assets].sum():,.2f}",      # Var
                            # % do Risco Total
                            f"{(covar_perc * abs(covar.sum() / var_limite_comparativo)).sum() * 100:.2f}%",
                        ])
                    ]
                    # Criar DataFrame com idx como única linha
                    tabela_dados_risco = pd.DataFrame(
                        dados_formatados,  # Colunas = ativos, Valores = strings formatadas
                        index=[idx]        # Única linha = idx
                    )
                    colunas_selecionadas = []
                    colunas_selecionadas.append('Beta')
                    colunas_selecionadas.append('MVar(R$)')
                    colunas_selecionadas.append('CoVaR(R$)')
                    colunas_selecionadas.append('CoVaR(%)')
                    colunas_selecionadas.append('Var')
                    colunas_selecionadas.append('% do Risco Total')

                    # Concatenando os dados se houver mais de uma iteração
                    tabela_dados_riscos = pd.concat(
                        [tabela_dados_riscos, tabela_dados_risco], axis=0)

                    # st.write("## Risco")
                    if 'CoVaR(R$)' in colunas_selecionadas:
                        df_dados['CoVaR(R$)'] = df_dados['CoVaR(R$)'].apply(
                            lambda x: f"R${x:,.0f}")
                    if 'MVar(R$)' in colunas_selecionadas:
                        df_dados['MVar(R$)'] = df_dados['MVar(R$)'].apply(
                            lambda x: f"R${x:,.0f}")
                    if 'CoVaR(%)' in colunas_selecionadas:
                        df_dados['CoVaR(%)'] = df_dados['CoVaR(%)'].apply(
                            lambda x: f"{x:.2%}")
                    if 'Beta' in colunas_selecionadas:
                        df_dados['Beta'] = df_dados['Beta'].apply(
                            lambda x: f"{x:.4f}")
                    if 'Var' in colunas_selecionadas:
                        df_dados['Var'] = df_dados['Var'].apply(
                            lambda x: f"{x:.4f}%")
                    if '% do Risco Total' in colunas_selecionadas:
                        df_dados['% do Risco Total'] = df_dados['% do Risco Total'].apply(
                            lambda x: f"{x:.2%}")

                # Display the filtered table
                    if colunas_selecionadas:
                        # st.write("Tabela de Dados Selecionados:")
                        tabela_filtrada = df_dados[colunas_selecionadas]

                        # Adicionar uma linha de soma
                        sum_row = tabela_filtrada.select_dtypes(
                            include='number').sum()
                        sum_row['Beta'] = df_beta.sum()
                        sum_row['MVar(R$)'] = df_mvar_dinheiro.sum()
                        sum_row['CoVaR(R$)'] = covar.sum()
                        sum_row['CoVaR(%)'] = covar_perc.sum()
                        sum_row['Var'] = var_ativos[assets].sum()
                        sum_row['% do Risco Total'] = (
                            covar_perc * abs(covar.sum() / var_limite_comparativo)).sum()
                        sum_row = sum_row.to_frame().T
                        sum_row['Beta'] = sum_row['Beta'].apply(
                            lambda x: f"{x:.4f}")
                        sum_row['MVar(R$)'] = sum_row['MVar(R$)'].apply(
                            lambda x: f"R${x:,.0f}")
                        sum_row['CoVaR(R$)'] = sum_row['CoVaR(R$)'].apply(
                            lambda x: f"R${x:,.0f}")
                        sum_row['CoVaR(%)'] = sum_row['CoVaR(%)'].apply(
                            lambda x: f"{x:.2%}")
                        sum_row['Var'] = sum_row['Var'].apply(
                            lambda x: f"{x:.4f}")
                        sum_row['% do Risco Total'] = sum_row['% do Risco Total'].apply(
                            lambda x: f"{x:.2%}")

                        sum_row = sum_row[colunas_selecionadas]
                        # Adicionar índice 'Total'
                        sum_row.index = ['Total']
                        # Adicionar a linha de soma na tabela filtrada
                        tabela_filtrada_com_soma = pd.concat(
                            [tabela_filtrada, sum_row])
                        # Preciso criar uma linha com dados nas colunas de todos os fundos
                        # Criando um dicionário para armazenar os valores formatados corretamente
                        dados = {
                            'Juros Nominais Brasil': [
                                f"{abs(df_divone_juros_nominais.iloc[0] / soma_pl_sem_pesos * 10000):,.2f}bps / "
                                f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']):,.2f}bps"
                            ],
                            'Juros Reais Brasil': [
                                f"{abs(df_divone_juros_real.iloc[0]/ soma_pl_sem_pesos * 10000):,.2f}bps / "
                                f"{abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']):,.2f}bps"
                            ],
                            'Juros US': [
                                f"{abs(df_divone_juros_externo_certo.iloc[0]/ soma_pl_sem_pesos * 10000):,.2f}bps / "
                                f"{abs(stress_test_juros_externo_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_externo_percent['FUT_TICK_VAL']):,.2f}bps"
                            ],
                            'Moedas': [
                                f"{abs(df_divone_dolar.iloc[0]/ soma_pl_sem_pesos * 10000):,.2f}bps /"
                                f"{abs(stress_dolar_percent):,.2f}bps" if lista_dolar else f"{abs(stress_dolar_percent):,.2f}bps"
                            ]
                        }

                        

                        # Criando a linha "Total"
                        # dados['Total'] = [
                        #    f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real.iloc[0]) + abs(df_divone_juros_externo_certo.iloc[0]) + (abs(df_divone_dolar.iloc[0]) if lista_dolar else 0):,.2f} / "
                        #    f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo) + abs(stress_dolar):,.2f}" if lista_juros_externo else f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo['FUT_TICK_VAL']) + abs(stress_dolar):,.2f} / "
                        #    f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent) + abs(stress_dolar_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent['FUT_TICK_VAL']) + abs(stress_dolar_percent):,.2f}bps"
                        # ]

                        # Criando o DataFrame
                        tabela_dados_fundo = pd.DataFrame(dados, index=[idx])

                        # Concatenando os dados se houver mais de uma iteração
                        tabela_dados_fundos = pd.concat(
                            [tabela_dados_fundos, tabela_dados_fundo], axis=0)

                    #st.table(tabela_filtrada_com_soma)
                    
        # Criar coluna de Total
        # sum_row = pd.DataFrame({
        #    'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
        #    'VaR Limite (bps)': f"{var_limite_comparativo/ soma_pl_sem_pesos *10000 :,.2f}bps",
        #    'VaR Portfólio (bps)': f"{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps",
        #    'CVaR (bps)': f"{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps",
        #    'Volatilidade': f"{vol_port_analitica:.2%}",
        #    '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}"
        # }, index=['Total'])
        # df_portfolio_final = pd.concat([df_portfolio_final, sum_row])
        st.write("## Dados de cada Fundo")
        st.table(df_portfolio_final)
        # Criar coluna de Total
        # Tirar bps e / de cada célula
        # Processar tabela_dados_fundos_p1 (valor antes de '/')
        tabela_dados_fundos_p1 = tabela_dados_fundos.copy()
        tabela_dados_fundos_p1 = tabela_dados_fundos_p1.applymap(
            lambda x: x.replace('bps', '') if isinstance(x, str) else x)
        tabela_dados_fundos_p1 = tabela_dados_fundos_p1.applymap(
            lambda x: x.split('/')[0] if isinstance(x, str) and '/' in x else x)
        # Converter os valores para float
        tabela_dados_fundos_p1 = tabela_dados_fundos_p1.astype(float)
        # Adicionar a coluna 'Total' somando as colunas
        tabela_dados_fundos_p1['Total'] = tabela_dados_fundos_p1.sum(
            axis=1).round(2)  # Soma por linha (axis=1)

        # Processar tabela_dados_fundos_p2 (valor após '/')
        tabela_dados_fundos_p2 = tabela_dados_fundos.copy()
        tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(
            lambda x: x.replace('bps', '') if isinstance(x, str) else x)
        tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(
            lambda x: x.split('/')[1] if isinstance(x, str) and '/' in x else x)

        # Aqui, é necessário tratar os valores corretamente
        # Se o valor for '0', defina como 0, caso contrário converta para float
        tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(
            lambda x: float(x) if isinstance(x, str) and x != '0' else 0)

        # Somar as colunas específicas para obter o total
        try:
            tabela_dados_fundos_p2['Total'] = tabela_dados_fundos_p2[[
                'Juros Nominais Brasil', 'Juros Reais Brasil', 'Moedas', 'Juros US']].sum(axis=1).round(2)  # Soma por linha (axis=1)

            # Concatenar a informação de 'Total' nas duas tabelas (tabela_dados_fundos_p1 e tabela_dados_fundos_p2)
            tabela_dados_fundos_p1['Total'] = tabela_dados_fundos_p1['Total'].astype(
                str) + 'bps / ' + tabela_dados_fundos_p2['Total'].astype(str) + 'bps'

            # Copiar a coluna 'Total' para a tabela final
            tabela_dados_fundos['Total'] = tabela_dados_fundos_p1['Total']
            # Preciso juntar as colunas que sejam Juros Nominais Brasil, Juros Reais Brasil, Moedas e Juros US
            # Identificar categorias por colunas
            mapeamento_categorias = {
                "Juros Nominais Brasil": [col for col in tabela_dados_riscos.columns if "DI" in col],
                "Juros Reais Brasil": [col for col in tabela_dados_riscos.columns if "DAP" in col],
                "Juros US": ["TREASURY"],
                "Moedas": ["WDO1"]
            }
            # Criar novo DataFrame com as categorias
            # Criar novo DataFrame com as categorias
            nova_tabela = pd.DataFrame()
            nova_tabela["Fundos"] = tabela_dados_riscos.index

            # Dicionário para armazenar valores da soma total
            soma_total_antes = []
            soma_total_depois = []

            for categoria, colunas in mapeamento_categorias.items():
                valores_antes = []
                valores_depois = []

                for _, row in tabela_dados_riscos.iterrows():
                    soma_antes = 0
                    soma_depois = 0

                    for col in colunas:
                        if col in tabela_dados_riscos.columns:
                            partes = row[col].split(" / ")
                            soma_antes += float(partes[0].replace("%", ""))
                            soma_depois += float(partes[1].replace("%", ""))

                    valores_antes.append(soma_antes)
                    valores_depois.append(soma_depois)

                # Adicionar ao DataFrame
                nova_tabela[categoria] = [
                    f"{antes:.2f}% / {depois:.2f}%" for antes, depois in zip(valores_antes, valores_depois)]

                # Acumulando valores para a coluna Total
                if not soma_total_antes:
                    soma_total_antes = valores_antes
                    soma_total_depois = valores_depois
                else:
                    soma_total_antes = [
                        x + y for x, y in zip(soma_total_antes, valores_antes)]
                    soma_total_depois = [
                        x + y for x, y in zip(soma_total_depois, valores_depois)]

            # Criar coluna Total
            nova_tabela["Total"] = [
                f"{antes:.2f}% / {depois:.2f}%" for antes, depois in zip(soma_total_antes, soma_total_depois)]
            nova_tabela.set_index("Fundos", inplace=True)
            # Exibir tabela reorganizada
            if op1:
                st.write("### Risco por Fundo")
                st.table(nova_tabela)
                st.markdown(
                    "<p style='font-size: 13px; font-style: italic;'>(CoVaR bps / % Risco Total)</p>", unsafe_allow_html=True)

            if op2:
                st.write("### Stress & DIV01 por Fundo")
                st.table(tabela_dados_fundos)
                st.markdown(
                    "<p style='font-size: 13px; font-style: italic;'>(Div01 bps / Stress bps)</p>", unsafe_allow_html=True)
            return
        except:
            st.write("Nenhum fundo selecionado / Nenhum contrato cadastrado")
            return
    else:
        st.write("Nenhum fundo selecionado / Nenhum contrato cadastrado")
        return



def calcular_metricas_de_fundo(assets, quantidades, df_contratos, fundos, op1, op2):
    df_tira = df_contratos.copy()
    df_tira.reset_index(inplace=True)
    # Tirar colunas que contenham o nome 'Max' ou 'Adm'
    df_tira = df_tira[df_tira.columns.drop(
        list(df_tira.filter(regex='Max')))]
    df_tira.rename(columns={'index': 'Fundo'}, inplace=True)
    lista_remove = []
    for fundo2 in fundos[:]:
        # Filtrar a linha do fundo
        linha = df_tira[df_tira['Fundo'] == fundo2].select_dtypes(include=['number'])

        # Verificar se todas as colunas são zero
        # `axis=1` verifica todas as colunas na linha
        if (linha == 0).all(axis=1).values[0]:
            lista_remove.append(fundo2)
    for fundo in lista_remove:
        fundos.remove(fundo)
    if fundos:
        # st.write(df_contratos)
        file_pl = "Dados/pl_fundos.parquet"
        df_pl = pd.read_parquet(file_pl)
        file_bbg = "Dados/BBG - ECO DASH.xlsx"

        # Dicionário de pesos fixo (pode-se tornar dinâmico no futuro)
        dict_pesos = {
            'GLOBAL BONDS': 4,
            'HORIZONTE': 1,
            'JERA2026': 1,
            'REAL FIM': 1,
            'BH FIRF INFRA': 1,
            'BORDEAUX INFRA': 1,
            'TOPAZIO INFRA': 1,
            'MANACA INFRA FIRF': 1,
            'AF DEB INCENTIVADAS': 3
        }
        # Zerar os pesos de fundos que não tem contratos
        for idx, row in df_contratos.iterrows():
            if idx == 'Total':
                continue
            else:
                fundo = idx
                check = 0
                for asset in assets:
                    if int(row[f'Contratos {asset}']) != 0:
                        check = 1
                if check == 0:
                    dict_pesos[fundo] = 0

        Weights = list(dict_pesos.values())
        df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio(
            df_pl, Weights)

        df = pd.read_parquet('Dados/df_inicial.parquet')
        df_precos, df_completo = load_and_process_excel(df, assets)
        df_retorno = process_returns(df_completo, assets)

        var_ativos = var_not_parametric(df_retorno).abs()
  
        df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)
        quantidade_nomes = {}
        tabela_dados_fundos = pd.DataFrame()
        tabela_dados_riscos = pd.DataFrame()
        # Antes do loop principal, inicialize o dicionário de dados:

        df_portfolio_final = pd.DataFrame()
        for idx, row in df_contratos.iterrows():
            if idx in fundos:
                df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio_especifico(
                    df_pl, Weights, idx)
                for i in range(len(assets)):
                    quantidade_nomes[assets[i]] = row[f'Contratos {assets[i]}']
                quantidade = np.array(list(quantidade_nomes.values()))
                vp = df_precos_ajustados['Valor Fechamento'] * abs(quantidade)
                vp_soma = vp.sum()
                if vp_soma == 0:
                    pass
                else:

                    pesos = quantidade * \
                        df_precos_ajustados['Valor Fechamento'] / vp_soma
                    df_returns_portifolio = df_retorno * pesos
                    df_returns_portifolio['Portifolio'] = df_returns_portifolio.sum(
                        axis=1)

                    # VaR
                    var_port = var_not_parametric(
                        df_returns_portifolio['Portifolio'])
                    var_port = abs(var_port)
                    var_port_dinheiro = vp_soma * var_port

                    # ######################################################################################
                    # ######################################################################################
                    # #####################################################################################
                    ######################################################################################
                    ######## var bps e var limite são fixos, mas podem ser dinâmicos no futuro ########################

                    var_bps = 1.0
                    var_bps = var_bps / 10000

                    var_limite = 1.0

                    vol_port_retornos = df_returns_portifolio['Portifolio'].std(
                    )
                    vol_port_analitica = vol_port_retornos * np.sqrt(252)

                    df_retorno['Portifolio'] = df_returns_portifolio['Portifolio']
                    cov = df_retorno.cov()
                    cov_port = cov['Portifolio'].drop('Portifolio')
                    df_beta = cov_port / (vol_port_retornos**2)
                    df_mvar = df_beta * var_port
                    df_mvar_dinheiro = df_mvar * \
                        df_precos_ajustados['Valor Fechamento']

                    covar = df_mvar * pesos.values * vp_soma
                    covar_perc = covar / covar.sum()

                    cvar = df_retorno[df_retorno['Portifolio'] < var_not_parametric(
                        df_retorno['Portifolio'])]['Portifolio'].mean()
                    cvar = abs(cvar)
                    cvar_dinheiro = vp_soma * cvar

                    df_divone, dolar, treasury = load_and_process_divone(
                        'Dados/BBG - ECO DASH.xlsx', df_completo)

                    lista_juros_interno = [
                        asset for asset in assets if 'DI' in asset]
                    df_divone_juros_nominais = df_divone[lista_juros_interno]

                    lista_quantidade = [quantidade_nomes[asset]
                                        for asset in lista_juros_interno]
                    df_divone_juros_nominais = df_divone_juros_nominais * \
                        np.array(lista_quantidade)
                    df_divone_juros_nominais = df_divone_juros_nominais.sum(
                        axis=1)

                    lista_juros_interno_real = [a for a in assets if ('DAP' in a) or ('NTNB' in a)]

                    df_divone_juros_real = df_divone[lista_juros_interno_real]

                    lista_quantidade = [quantidade_nomes[asset]
                                        for asset in lista_juros_interno_real]

                    df_divone_juros_real = df_divone_juros_real * \
                        np.array(lista_quantidade)

                    df_divone_juros_real = df_divone_juros_real.sum(axis=1)

                    lista_juros_externo = [
                        asset for asset in assets if 'TREASURY' in asset]

                    df_divone_juros_externo = df_divone[lista_juros_externo]

                    lista_quantidade = [quantidade_nomes[asset]
                                        for asset in lista_juros_externo]

                    df_divone_juros_externo = df_divone_juros_externo * \
                        np.array(lista_quantidade)

                    df_divone_juros_externo = df_divone_juros_externo.sum(
                        axis=1)

                    stress_test_juros_interno_Nominais = df_divone_juros_nominais * 100
                    stress_test_juros_interno_Nominais_percent = stress_test_juros_interno_Nominais / \
                        soma_pl_sem_pesos * 10000

                    stress_test_juros_interno_Reais = df_divone_juros_real * 50
                    stress_test_juros_interno_Reais_percent = stress_test_juros_interno_Reais / \
                        soma_pl_sem_pesos * 10000

                    df_divone_juros_externo_certo = df_divone_juros_externo

                    if lista_juros_externo:
                        valor_acumulado_treasury = (
                            1+df_retorno['TREASURY']).cumprod()
                        pico_max_treasury = valor_acumulado_treasury.max()
                        drawndown_treasury = (valor_acumulado_treasury - pico_max_treasury) / \
                            pico_max_treasury
                        drawndown_treasury = drawndown_treasury.min()
                        drawndown_treasury = df_retorno['TREASURY'].min()
                        df_divone_juros_externo = drawndown_treasury
                        df_divone_juros_externo = abs(
                            df_divone_juros_externo) * treasury * dolar / 10000
                        df_divone_juros_externo = df_divone_juros_externo * \
                            np.array(lista_quantidade)
                        df_divone_juros_externo = df_divone_juros_externo.sum()

                    stress_test_juros_externo = df_divone_juros_externo

                    stress_test_juros_externo_percent = stress_test_juros_externo / \
                        soma_pl_sem_pesos * 10000

                    lista_dolar = [
                        asset for asset in assets if 'WDO1' in asset]
                    if lista_dolar:
                        quantidade_dolar = quantidade_nomes[lista_dolar[0]]
                        stress_dolar = quantidade_dolar * dolar * 0.02
                        valor_acumulado = (1+df_retorno['WDO1']).cumprod()
                        pico_max = valor_acumulado.max()
                        drawndown_dolar = (
                            valor_acumulado - pico_max) / pico_max
                        drawndown_dolar = drawndown_dolar.min()
                        #st.write(f"Drawndown Dolar: {drawndown_dolar}")
                        drawndown_dolar = df_retorno['WDO1'].min()
                        #st.write(f"Drawndown Dolar: {drawndown_dolar}")
                        df_divone_dolar = drawndown_dolar
                        df_divone_dolar = df_divone_dolar * quantidade_dolar
                        df_divone_dolar = abs(df_divone_dolar) * dolar
                        stress_dolar = df_divone_dolar
                        stress_dolar_percent = stress_dolar / soma_pl_sem_pesos * 10000
                        df_divone_dolar = df_divone[lista_dolar] * \
                            np.array(quantidade_dolar)
                        df_divone_dolar = df_divone_dolar.sum()

                    else:
                        stress_dolar = 0
                        df_divone_dolar = 0

                    stress_dolar_percent = stress_dolar / soma_pl_sem_pesos * 10000
                    df_stress_div01 = pd.DataFrame({
                        'DIV01': [
                            f"R${abs(df_divone_juros_nominais.iloc[0]):,.2f}",
                            f"R${abs(df_divone_juros_real.iloc[0]):,.2f}",
                            f"R${abs(df_divone_juros_externo_certo.iloc[0]):,.2f}",
                            f'R${abs(df_divone_dolar.iloc[0]):,.2f}' if lista_dolar else 0
                        ],
                        'Stress (R$)': [
                            f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']):,.2f}",
                            f"R${abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']):,.2f}",
                            f"R${abs(stress_test_juros_externo):,.2f}" if lista_juros_externo else f"R${abs(stress_test_juros_externo['FUT_TICK_VAL']):,.2f}",
                            f"R${abs(stress_dolar):,.2f}"
                        ],
                        'Stress (bps)': [
                            f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']):,.2f}bps",
                            f"{abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']):,.2f}bps",
                            f"{abs(stress_test_juros_externo_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_externo_percent['FUT_TICK_VAL']):,.2f}bps",
                            f"{abs(stress_dolar_percent):,.2f}bps"
                        ]
                    }, index=['Juros Nominais Brasil', 'Juros Reais Brasil', 'Juros US', 'Moedas'])

                    sum_row = pd.DataFrame({
                        'DIV01': [f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real[0]) + abs(df_divone_juros_externo_certo.iloc[0]) + abs(df_divone_dolar.iloc[0]):,.2f}"] if lista_dolar else [f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real[0]) + abs(df_divone_juros_externo_certo.iloc[0]):,.2f}"],
                        'Stress (R$)': [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo) + abs(stress_dolar):,.2f}"] if lista_juros_externo else [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo['FUT_TICK_VAL']) + abs(stress_dolar):,.2f}"],
                        'Stress (bps)': [f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent) + abs(stress_dolar_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent['FUT_TICK_VAL']) + abs(stress_dolar_percent):,.2f}bps"]
                    }, index=['Total'])
                    df_stress_div01 = pd.concat([df_stress_div01, sum_row])
                    # --- Layout ---
        #            st.write("## Dados do Portfólio")
        #            st.write(f"**PL: R$ {soma_pl_sem_pesos:,.0f}**")
        #
                    var_limite_comparativo = soma_pl * var_bps * var_limite
        #            st.write(
        #                f"**VaR Limite** (Peso de {var_limite:.1%}): R${var_limite_comparativo:,.0f}"
        #            )
        #
        #            st.write(
        #                f"**VaR do Portfólio**: R${var_port_dinheiro:,.0f} : **{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps**"
        #            )
        #            st.write(
        #                f"**CVaR**: R${abs(cvar * vp_soma):,.0f} : **{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps**"
        #            )
        #            st.write(f"**Volatilidade**: {vol_port_analitica:.2%}")
        #            st.table(df_stress_div01)
        #
        #            st.write("---")
        #            st.write(
        #                f"### {abs(covar.sum()/ var_limite_comparativo):.2%} do risco total")

                    # Coletar dados formatados para cada fundo
                    dados_fundo = {
                        'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
                        'VaR Limite (bps)': f"{var_limite_comparativo/soma_pl_sem_pesos * 10000:,.2f}bps",
                        'VaR Portfólio (bps)': f"{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps",
                        'CVaR (bps)': f"{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps",
                        'Volatilidade': f"{vol_port_analitica:.2%}",
                        '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}"
                    }
        #            dados_fundo = {
        #                'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
        #                'VaR Limite (R$)': f"R${var_limite_comparativo:,.0f}",
        #                'VaR Portfólio (R$/bps)': f"R${var_port_dinheiro:,.0f} / {var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps",
        #                'CVaR (R$/bps)': f"R${abs(cvar * vp_soma):,.0f} / {abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps",
        #                'Volatilidade': f"{vol_port_analitica:.2%}",
        #                '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}"
        #            }

                    # Adicionar como nova linha ao DataFrame
                    df_portfolio_final = pd.concat([
                        df_portfolio_final,
                        pd.DataFrame(dados_fundo, index=[idx])
                    ])

                    # Exiba a tabela formatada no Streamlit:
                    # Supondo que df_dados já está criado com múltiplos índices (ativos)
                    df_dados = pd.DataFrame({
                        'Beta': df_beta,
                        'MVar(R$)': df_mvar_dinheiro,
                        'CoVaR(R$)': covar,
                        'CoVaR(%)': covar_perc,
                        'Var': var_ativos[assets],
                        '% do Risco Total': covar_perc * abs(covar.sum() / var_limite_comparativo)
                    })

                    # Lista de ativos (nomes das colunas)
                    lista_ativos = df_dados.index.tolist()

                    # Dicionário para armazenar os valores formatados de cada ativo
                    dados_formatados = {}

                    # Iterar sobre cada ativo (linha do df_dados)
                    for ativo in lista_ativos:
                        # Extrair dados do ativo
                        linha = df_dados.loc[ativo]

                        # Formatar os valores em uma string com "/"
                        valores_formatados = " / ".join([
                            # f"{linha['Beta']:.6f}",                   # Beta
                            # f"R${linha['MVar(R$)']:,.2f}",            # MVar(R$)
                            # f"R${linha['CoVaR(R$)']:,.2f}",           # CoVaR(R$)
                            # CoVaR(%)
                            f"{linha['CoVaR(%)'] * 100:.2f}%",
                            # f"R${linha['Var']:,.2f}",                 # Var
                            # % do Risco Total
                            f"{linha['% do Risco Total']* 100:.2f}%",
                        ])

                        # Adicionar ao dicionário (chave = nome do ativo)
                        dados_formatados[ativo] = [valores_formatados]
                    # Criar uma coluna de Total
                    dados_formatados['Total'] = [
                        " / ".join([
                            # f"{df_beta.sum():.6f}",                   # Beta
                            # f"R${df_mvar_dinheiro.sum():,.2f}",       # MVar(R$)
                            # f"R${covar.sum():,.2f}",                  # CoVaR(R$)
                            # CoVaR(%)
                            f"{covar_perc.sum() * 100:.2f}%",
                            # f"R${var_ativos[assets].sum():,.2f}",      # Var
                            # % do Risco Total
                            f"{(covar_perc * abs(covar.sum() / var_limite_comparativo)).sum() * 100:.2f}%",
                        ])
                    ]
                    # Criar DataFrame com idx como única linha
                    tabela_dados_risco = pd.DataFrame(
                        dados_formatados,  # Colunas = ativos, Valores = strings formatadas
                        index=[idx]        # Única linha = idx
                    )
                    colunas_selecionadas = []
                    colunas_selecionadas.append('Beta')
                    colunas_selecionadas.append('MVar(R$)')
                    colunas_selecionadas.append('CoVaR(R$)')
                    colunas_selecionadas.append('CoVaR(%)')
                    colunas_selecionadas.append('Var')
                    colunas_selecionadas.append('% do Risco Total')

                    # Concatenando os dados se houver mais de uma iteração
                    tabela_dados_riscos = pd.concat(
                        [tabela_dados_riscos, tabela_dados_risco], axis=0)

                    # st.write("## Risco")
                    if 'CoVaR(R$)' in colunas_selecionadas:
                        df_dados['CoVaR(R$)'] = df_dados['CoVaR(R$)'].apply(
                            lambda x: f"R${x:,.0f}")
                    if 'MVar(R$)' in colunas_selecionadas:
                        df_dados['MVar(R$)'] = df_dados['MVar(R$)'].apply(
                            lambda x: f"R${x:,.0f}")
                    if 'CoVaR(%)' in colunas_selecionadas:
                        df_dados['CoVaR(%)'] = df_dados['CoVaR(%)'].apply(
                            lambda x: f"{x:.2%}")
                    if 'Beta' in colunas_selecionadas:
                        df_dados['Beta'] = df_dados['Beta'].apply(
                            lambda x: f"{x:.4f}")
                    if 'Var' in colunas_selecionadas:
                        df_dados['Var'] = df_dados['Var'].apply(
                            lambda x: f"{x:.4f}%")
                    if '% do Risco Total' in colunas_selecionadas:
                        df_dados['% do Risco Total'] = df_dados['% do Risco Total'].apply(
                            lambda x: f"{x:.2%}")

                # Display the filtered table
                    if colunas_selecionadas:
                        # st.write("Tabela de Dados Selecionados:")
                        tabela_filtrada = df_dados[colunas_selecionadas]

                        # Adicionar uma linha de soma
                        sum_row = tabela_filtrada.select_dtypes(
                            include='number').sum()
                        sum_row['Beta'] = df_beta.sum()
                        sum_row['MVar(R$)'] = df_mvar_dinheiro.sum()
                        sum_row['CoVaR(R$)'] = covar.sum()
                        sum_row['CoVaR(%)'] = covar_perc.sum()
                        sum_row['Var'] = var_ativos[assets].sum()
                        sum_row['% do Risco Total'] = (
                            covar_perc * abs(covar.sum() / var_limite_comparativo)).sum()
                        sum_row = sum_row.to_frame().T
                        sum_row['Beta'] = sum_row['Beta'].apply(
                            lambda x: f"{x:.4f}")
                        sum_row['MVar(R$)'] = sum_row['MVar(R$)'].apply(
                            lambda x: f"R${x:,.0f}")
                        sum_row['CoVaR(R$)'] = sum_row['CoVaR(R$)'].apply(
                            lambda x: f"R${x:,.0f}")
                        sum_row['CoVaR(%)'] = sum_row['CoVaR(%)'].apply(
                            lambda x: f"{x:.2%}")
                        sum_row['Var'] = sum_row['Var'].apply(
                            lambda x: f"{x:.4f}")
                        sum_row['% do Risco Total'] = sum_row['% do Risco Total'].apply(
                            lambda x: f"{x:.2%}")

                        sum_row = sum_row[colunas_selecionadas]
                        # Adicionar índice 'Total'
                        sum_row.index = ['Total']
                        # Adicionar a linha de soma na tabela filtrada
                        tabela_filtrada_com_soma = pd.concat(
                            [tabela_filtrada, sum_row])
                        # Preciso criar uma linha com dados nas colunas de todos os fundos
                        # Criando um dicionário para armazenar os valores formatados corretamente
                        dados = {
                            'Juros Nominais Brasil': [
                                f"{abs(df_divone_juros_nominais.iloc[0] / soma_pl_sem_pesos * 10000):,.2f}bps / "
                                f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']):,.2f}bps"
                            ],
                            'Juros Reais Brasil': [
                                f"{abs(df_divone_juros_real.iloc[0]/ soma_pl_sem_pesos * 10000):,.2f}bps / "
                                f"{abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']):,.2f}bps"
                            ],
                            'Juros US': [
                                f"{abs(df_divone_juros_externo_certo.iloc[0]/ soma_pl_sem_pesos * 10000):,.2f}bps / "
                                f"{abs(stress_test_juros_externo_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_externo_percent['FUT_TICK_VAL']):,.2f}bps"
                            ],
                            'Moedas': [
                                f"{abs(df_divone_dolar.iloc[0]/ soma_pl_sem_pesos * 10000):,.2f}bps /"
                                f"{abs(stress_dolar_percent):,.2f}bps" if lista_dolar else f"{abs(stress_dolar_percent):,.2f}bps"
                            ]
                        }

                        # Criando a linha "Total"
                        # dados['Total'] = [
                        #    f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real.iloc[0]) + abs(df_divone_juros_externo_certo.iloc[0]) + (abs(df_divone_dolar.iloc[0]) if lista_dolar else 0):,.2f} / "
                        #    f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo) + abs(stress_dolar):,.2f}" if lista_juros_externo else f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo['FUT_TICK_VAL']) + abs(stress_dolar):,.2f} / "
                        #    f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent) + abs(stress_dolar_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent['FUT_TICK_VAL']) + abs(stress_dolar_percent):,.2f}bps"
                        # ]

                        # Criando o DataFrame
                        tabela_dados_fundo = pd.DataFrame(dados, index=[idx])

                        # Concatenando os dados se houver mais de uma iteração
                        tabela_dados_fundos = pd.concat(
                            [tabela_dados_fundos, tabela_dados_fundo], axis=0)

                    # st.table(tabela_filtrada_com_soma)
        # Criar coluna de Total
        # sum_row = pd.DataFrame({
        #    'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
        #    'VaR Limite (bps)': f"{var_limite_comparativo/ soma_pl_sem_pesos *10000 :,.2f}bps",
        #    'VaR Portfólio (bps)': f"{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps",
        #    'CVaR (bps)': f"{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps",
        #    'Volatilidade': f"{vol_port_analitica:.2%}",
        #    '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}"
        # }, index=['Total'])
        # df_portfolio_final = pd.concat([df_portfolio_final, sum_row])
        st.table(df_portfolio_final)
        # Criar coluna de Total
        # Tirar bps e / de cada célula
        # Processar tabela_dados_fundos_p1 (valor antes de '/')
        tabela_dados_fundos_p1 = tabela_dados_fundos.copy()
        tabela_dados_fundos_p1 = tabela_dados_fundos_p1.applymap(
            lambda x: x.replace('bps', '') if isinstance(x, str) else x)
        tabela_dados_fundos_p1 = tabela_dados_fundos_p1.applymap(
            lambda x: x.split('/')[0] if isinstance(x, str) and '/' in x else x)
        # Converter os valores para float
        tabela_dados_fundos_p1 = tabela_dados_fundos_p1.astype(float)
        # Adicionar a coluna 'Total' somando as colunas
        tabela_dados_fundos_p1['Total'] = tabela_dados_fundos_p1.sum(
            axis=1).round(2)  # Soma por linha (axis=1)

        # Processar tabela_dados_fundos_p2 (valor após '/')
        tabela_dados_fundos_p2 = tabela_dados_fundos.copy()
        tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(
            lambda x: x.replace('bps', '') if isinstance(x, str) else x)
        tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(
            lambda x: x.split('/')[1] if isinstance(x, str) and '/' in x else x)

        # Aqui, é necessário tratar os valores corretamente
        # Se o valor for '0', defina como 0, caso contrário converta para float
        tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(
            lambda x: float(x) if isinstance(x, str) and x != '0' else 0)

        # Somar as colunas específicas para obter o total
        try:
            tabela_dados_fundos_p2['Total'] = tabela_dados_fundos_p2[[
                'Juros Nominais Brasil', 'Juros Reais Brasil', 'Moedas', 'Juros US']].sum(axis=1).round(2)  # Soma por linha (axis=1)

            # Concatenar a informação de 'Total' nas duas tabelas (tabela_dados_fundos_p1 e tabela_dados_fundos_p2)
            tabela_dados_fundos_p1['Total'] = tabela_dados_fundos_p1['Total'].astype(
                str) + 'bps / ' + tabela_dados_fundos_p2['Total'].astype(str) + 'bps'

            # Copiar a coluna 'Total' para a tabela final
            tabela_dados_fundos['Total'] = tabela_dados_fundos_p1['Total']
            # Preciso juntar as colunas que sejam Juros Nominais Brasil, Juros Reais Brasil, Moedas e Juros US
            # Identificar categorias por colunas
            mapeamento_categorias = {
                "Juros Nominais Brasil": [col for col in tabela_dados_riscos.columns if "DI" in col],
                "Juros Reais Brasil": [col for col in tabela_dados_riscos.columns if "DAP" in col],
                "Juros US": ["TREASURY"],
                "Moedas": ["WDO1"]
            }
            # Criar novo DataFrame com as categorias
            # Criar novo DataFrame com as categorias
            nova_tabela = pd.DataFrame()
            nova_tabela["Fundos"] = tabela_dados_riscos.index

            # Dicionário para armazenar valores da soma total
            soma_total_antes = []
            soma_total_depois = []

            for categoria, colunas in mapeamento_categorias.items():
                valores_antes = []
                valores_depois = []

                for _, row in tabela_dados_riscos.iterrows():
                    soma_antes = 0
                    soma_depois = 0

                    for col in colunas:
                        if col in tabela_dados_riscos.columns:
                            partes = row[col].split(" / ")
                            soma_antes += float(partes[0].replace("%", ""))
                            soma_depois += float(partes[1].replace("%", ""))

                    valores_antes.append(soma_antes)
                    valores_depois.append(soma_depois)

                # Adicionar ao DataFrame
                nova_tabela[categoria] = [
                    f"{antes:.2f}% / {depois:.2f}%" for antes, depois in zip(valores_antes, valores_depois)]

                # Acumulando valores para a coluna Total
                if not soma_total_antes:
                    soma_total_antes = valores_antes
                    soma_total_depois = valores_depois
                else:
                    soma_total_antes = [
                        x + y for x, y in zip(soma_total_antes, valores_antes)]
                    soma_total_depois = [
                        x + y for x, y in zip(soma_total_depois, valores_depois)]

            # Criar coluna Total
            nova_tabela["Total"] = [
                f"{antes:.2f}% / {depois:.2f}%" for antes, depois in zip(soma_total_antes, soma_total_depois)]
            nova_tabela.set_index("Fundos", inplace=True)
            # Exibir tabela reorganizada
            if op1:
                st.write("### Analise Risco por Categoria")
                st.table(nova_tabela)
                st.markdown(
                    "<p style='font-size: 13px; font-style: italic;'>(CoVaR bps / % Risco Total)</p>", unsafe_allow_html=True)

            if op2:
                st.write("### Analise Estratégias")
                st.table(tabela_dados_fundos)
                st.markdown(
                    "<p style='font-size: 13px; font-style: italic;'>(Div01 bps / Stress bps)</p>", unsafe_allow_html=True)
            return
        except:
            st.write("Nenhum fundo selecionado / Nenhum contrato cadastrado")
            return
    else:
        st.write("Nenhum fundo selecionado / Nenhum contrato cadastrado")
        return

def _pl_ref(pl_total: pd.Series, data: pd.Timestamp) -> float:
    """
    1 % do PL total para a data escolhida.
    """
    try:
        return float(pl_total.loc[data] * 0.01)  # 1 %
    except KeyError as e:
        raise KeyError(f"PL total não encontrado para a data {data}") from e


def calcular_metricas_de_fundo_analise(assets, quantidades, df_contratos, fundos, op1, op2):
    df_tira = df_contratos.copy()
    df_tira.reset_index(inplace=True)
    # Tirar colunas que contenham o nome 'Max' ou 'Adm'
    df_tira = df_tira[df_tira.columns.drop(
        list(df_tira.filter(regex='Max')))]
    df_tira.rename(columns={'index': 'Fundo'}, inplace=True)
    lista_remove = []
    for fundo2 in fundos[:]:
        # Filtrar a linha do fundo
        linha = df_tira[df_tira['Fundo'] == fundo2].select_dtypes(include=[
                                                                  'number'])

        # Verificar se todas as colunas são zero
        # `axis=1` verifica todas as colunas na linha
        if (linha == 0).all(axis=1).values[0]:
            lista_remove.append(fundo2)
    for fundo in lista_remove:
        fundos.remove(fundo)
    if fundos:
        # st.write(df_contratos)
        file_pl = "Dados/pl_fundos.parquet"
        df_pl = pd.read_parquet(file_pl)
        file_bbg = "Dados/BBG - ECO DASH.xlsx"

        # Dicionário de pesos fixo (pode-se tornar dinâmico no futuro)
        dict_pesos = {
            'GLOBAL BONDS': 4,
            'HORIZONTE': 1,
            'JERA2026': 1,
            'REAL FIM': 1,
            'BH FIRF INFRA': 1,
            'BORDEAUX INFRA': 1,
            'TOPAZIO INFRA': 1,
            'MANACA INFRA FIRF': 1,
            'AF DEB INCENTIVADAS': 3
        }
        # Zerar os pesos de fundos que não tem contratos
        for idx, row in df_contratos.iterrows():
            if idx == 'Total':
                continue
            else:
                fundo = idx
                check = 0
                for asset in assets:
                    if int(row[f'Contratos {asset}']) != 0:
                        check = 1
                if check == 0:
                    dict_pesos[fundo] = 0

        Weights = list(dict_pesos.values())
        df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio(
            df_pl, Weights)

        df = pd.read_parquet('Dados/df_inicial.parquet')

        df_precos, df_completo = load_and_process_excel(df, assets)
        df_retorno = process_returns(df_completo, assets)
        var_ativos = var_not_parametric(df_retorno).abs()
        df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)
        quantidade_nomes = {}
        tabela_dados_fundos = pd.DataFrame()
        tabela_dados_riscos = pd.DataFrame()
        # Antes do loop principal, inicialize o dicionário de dados:

        df_portfolio_final = pd.DataFrame()

        for idx, row in df_contratos.iterrows():
            if idx in fundos:
                df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio_especifico(
                    df_pl, Weights, idx)
                for i in range(len(assets)):
                    quantidade_nomes[assets[i]] = row[f'Contratos {assets[i]}']
                quantidade = np.array(list(quantidade_nomes.values()))
                vp = df_precos_ajustados['Valor Fechamento'] * abs(quantidade)
                vp_soma = vp.sum()
                if vp_soma == 0:
                    break

                pesos = quantidade * \
                    df_precos_ajustados['Valor Fechamento'] / vp_soma
                df_returns_portifolio = df_retorno * pesos
                df_returns_portifolio['Portifolio'] = df_returns_portifolio.sum(
                    axis=1)

                # VaR
                var_port = var_not_parametric(
                    df_returns_portifolio['Portifolio'])
                var_port = abs(var_port)
                var_port_dinheiro = vp_soma * var_port

                # ######################################################################################
                # ######################################################################################
                # #####################################################################################
                ######################################################################################
                ######## var bps e var limite são fixos, mas podem ser dinâmicos no futuro ########################

                var_bps = 1.0
                var_bps = var_bps / 10000

                var_limite = 1.0

                vol_port_retornos = df_returns_portifolio['Portifolio'].std()
                vol_port_analitica = vol_port_retornos * np.sqrt(252)

                df_retorno['Portifolio'] = df_returns_portifolio['Portifolio']
                cov = df_retorno.cov()
                cov_port = cov['Portifolio'].drop('Portifolio')
                df_beta = cov_port / (vol_port_retornos**2)
                df_mvar = df_beta * var_port
                df_mvar_dinheiro = df_mvar * \
                    df_precos_ajustados['Valor Fechamento']

                covar = df_mvar * pesos.values * vp_soma
                covar_perc = covar / covar.sum()

                cvar = df_retorno[df_retorno['Portifolio'] < var_not_parametric(
                    df_retorno['Portifolio'])]['Portifolio'].mean()
                cvar = abs(cvar)
                cvar_dinheiro = vp_soma * cvar

                df_divone, dolar, treasury = load_and_process_divone(
                    'Dados/BBG - ECO DASH.xlsx', df_completo)

                lista_juros_interno = [
                    asset for asset in assets if 'DI' in asset]
                df_divone_juros_nominais = df_divone[lista_juros_interno]

                lista_quantidade = [quantidade_nomes[asset]
                                    for asset in lista_juros_interno]
                df_divone_juros_nominais = df_divone_juros_nominais * \
                    np.array(lista_quantidade)
                df_divone_juros_nominais = df_divone_juros_nominais.sum(axis=1)

                lista_juros_interno_real = [a for a in assets if ('DAP' in a) or ('NTNB' in a)]

                df_divone_juros_real = df_divone[lista_juros_interno_real]

                lista_quantidade = [quantidade_nomes[asset]
                                    for asset in lista_juros_interno_real]

                df_divone_juros_real = df_divone_juros_real * \
                    np.array(lista_quantidade)

                df_divone_juros_real = df_divone_juros_real.sum(axis=1)

                lista_juros_externo = [
                    asset for asset in assets if 'TREASURY' in asset]

                df_divone_juros_externo = df_divone[lista_juros_externo]

                lista_quantidade = [quantidade_nomes[asset]
                                    for asset in lista_juros_externo]

                df_divone_juros_externo = df_divone_juros_externo * \
                    np.array(lista_quantidade)

                df_divone_juros_externo = df_divone_juros_externo.sum(axis=1)

                stress_test_juros_interno_Nominais = df_divone_juros_nominais * 100
                stress_test_juros_interno_Nominais_percent = stress_test_juros_interno_Nominais / \
                    soma_pl_sem_pesos * 10000

                stress_test_juros_interno_Reais = df_divone_juros_real * 50
                stress_test_juros_interno_Reais_percent = stress_test_juros_interno_Reais / \
                    soma_pl_sem_pesos * 10000

                df_divone_juros_externo_certo = df_divone_juros_externo

                if lista_juros_externo:
                    valor_acumulado_treasury = (
                        1+df_retorno['TREASURY']).cumprod()
                    pico_max_treasury = valor_acumulado_treasury.max()
                    drawndown_treasury = (valor_acumulado_treasury - pico_max_treasury) / \
                        pico_max_treasury
                    drawndown_treasury = drawndown_treasury.min()
                    drawndown_treasury = df_retorno['TREASURY'].min()
                    df_divone_juros_externo = drawndown_treasury
                    df_divone_juros_externo = abs(
                        df_divone_juros_externo) * treasury * dolar / 10000
                    df_divone_juros_externo = df_divone_juros_externo * \
                        np.array(lista_quantidade)
                    df_divone_juros_externo = df_divone_juros_externo.sum()

                stress_test_juros_externo = df_divone_juros_externo

                stress_test_juros_externo_percent = stress_test_juros_externo / \
                    soma_pl_sem_pesos * 10000

                lista_dolar = [
                    asset for asset in assets if 'WDO1' in asset]
                if lista_dolar:
                    quantidade_dolar = quantidade_nomes[lista_dolar[0]]
                    stress_dolar = quantidade_dolar * dolar * 0.02
                    valor_acumulado = (1+df_retorno['WDO1']).cumprod()
                    pico_max = valor_acumulado.max()
                    drawndown_dolar = (valor_acumulado - pico_max) / pico_max
                    drawndown_dolar = drawndown_dolar.min()
                    drawndown_dolar = df_retorno['WDO1'].min()
                    df_divone_dolar = drawndown_dolar
                    df_divone_dolar = df_divone_dolar * quantidade_dolar
                    df_divone_dolar = abs(df_divone_dolar) * dolar
                    stress_dolar = df_divone_dolar
                    stress_dolar_percent = stress_dolar / soma_pl_sem_pesos * 10000
                    df_divone_dolar = df_divone[lista_dolar] * \
                        np.array(quantidade_dolar)
                    df_divone_dolar = df_divone_dolar.sum()

                else:
                    stress_dolar = 0
                    df_divone_dolar = 0

                stress_dolar_percent = stress_dolar / soma_pl_sem_pesos * 10000
                df_stress_div01 = pd.DataFrame({
                    'DIV01': [
                        f"R${abs(df_divone_juros_nominais.iloc[0]):,.2f}",
                        f"R${abs(df_divone_juros_real.iloc[0]):,.2f}",
                        f"R${abs(df_divone_juros_externo_certo.iloc[0]):,.2f}",
                        f'R${abs(df_divone_dolar.iloc[0]):,.2f}' if lista_dolar else 0
                    ],
                    'Stress (R$)': [
                        f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']):,.2f}",
                        f"R${abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']):,.2f}",
                        f"R${abs(stress_test_juros_externo):,.2f}" if lista_juros_externo else f"R${abs(stress_test_juros_externo['FUT_TICK_VAL']):,.2f}",
                        f"R${abs(stress_dolar):,.2f}"
                    ],
                    'Stress (bps)': [
                        f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']):,.2f}bps",
                        f"{abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']):,.2f}bps",
                        f"{abs(stress_test_juros_externo_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_externo_percent['FUT_TICK_VAL']):,.2f}bps",
                        f"{abs(stress_dolar_percent):,.2f}bps"
                    ]
                }, index=['Juros Nominais Brasil', 'Juros Reais Brasil', 'Juros US', 'Moedas'])

                sum_row = pd.DataFrame({
                    'DIV01': [f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real[0]) + abs(df_divone_juros_externo_certo.iloc[0]) + abs(df_divone_dolar.iloc[0]):,.2f}"] if lista_dolar else [f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real[0]) + abs(df_divone_juros_externo_certo.iloc[0]):,.2f}"],
                    'Stress (R$)': [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo) + abs(stress_dolar):,.2f}"] if lista_juros_externo else [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo['FUT_TICK_VAL']) + abs(stress_dolar):,.2f}"],
                    'Stress (bps)': [f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent) + abs(stress_dolar_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent['FUT_TICK_VAL']) + abs(stress_dolar_percent):,.2f}bps"]
                }, index=['Total'])
                df_stress_div01 = pd.concat([df_stress_div01, sum_row])
                # --- Layout ---
    #            st.write("## Dados do Portfólio")
    #            st.write(f"**PL: R$ {soma_pl_sem_pesos:,.0f}**")
    #
                var_limite_comparativo = soma_pl * var_bps * var_limite
    #            st.write(
    #                f"**VaR Limite** (Peso de {var_limite:.1%}): R${var_limite_comparativo:,.0f}"
    #            )
    #
    #            st.write(
    #                f"**VaR do Portfólio**: R${var_port_dinheiro:,.0f} : **{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps**"
    #            )
    #            st.write(
    #                f"**CVaR**: R${abs(cvar * vp_soma):,.0f} : **{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps**"
    #            )
    #            st.write(f"**Volatilidade**: {vol_port_analitica:.2%}")
    #            st.table(df_stress_div01)
    #
    #            st.write("---")
    #            st.write(
    #                f"### {abs(covar.sum()/ var_limite_comparativo):.2%} do risco total")

                # Coletar dados formatados para cada fundo
                dados_fundo = {
                    'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
                    'VaR Limite (bps)': f"{var_limite_comparativo/soma_pl_sem_pesos * 10000:,.2f}bps",
                    'VaR Portfólio (bps)': f"{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps",
                    'CVaR (bps)': f"{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps",
                    'Volatilidade': f"{vol_port_analitica:.2%}",
                    '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}"
                }
    #            dados_fundo = {
    #                'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
    #                'VaR Limite (R$)': f"R${var_limite_comparativo:,.0f}",
    #                'VaR Portfólio (R$/bps)': f"R${var_port_dinheiro:,.0f} / {var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps",
    #                'CVaR (R$/bps)': f"R${abs(cvar * vp_soma):,.0f} / {abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps",
    #                'Volatilidade': f"{vol_port_analitica:.2%}",
    #                '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}"
    #            }

                # Adicionar como nova linha ao DataFrame
                df_portfolio_final = pd.concat([
                    df_portfolio_final,
                    pd.DataFrame(dados_fundo, index=[idx])
                ])

                #st.write(df_portfolio_final)

                # Exiba a tabela formatada no Streamlit:
                # Supondo que df_dados já está criado com múltiplos índices (ativos)
                df_dados = pd.DataFrame({
                    'Beta': df_beta,
                    'MVar(R$)': df_mvar_dinheiro,
                    'CoVaR(R$)': covar,
                    'CoVaR(%)': covar_perc,
                    'Var': var_ativos[assets],
                    '% do Risco Total': covar_perc * abs(covar.sum() / var_limite_comparativo)
                })

                # Lista de ativos (nomes das colunas)
                lista_ativos = df_dados.index.tolist()

                # Dicionário para armazenar os valores formatados de cada ativo
                dados_formatados = {}

                # Iterar sobre cada ativo (linha do df_dados)
                for ativo in lista_ativos:
                    # Extrair dados do ativo
                    linha = df_dados.loc[ativo]

                    # Formatar os valores em uma string com "/"
                    valores_formatados = " / ".join([
                        # f"{linha['Beta']:.6f}",                   # Beta
                        # f"R${linha['MVar(R$)']:,.2f}",            # MVar(R$)
                        # f"R${linha['CoVaR(R$)']:,.2f}",           # CoVaR(R$)
                        # CoVaR(%)
                        f"{linha['CoVaR(%)'] * 100:.2f}%",
                        # f"R${linha['Var']:,.2f}",                 # Var
                        # % do Risco Total
                        f"{linha['% do Risco Total']* 100:.2f}%",
                    ])

                    # Adicionar ao dicionário (chave = nome do ativo)
                    dados_formatados[ativo] = [valores_formatados]
                # Criar uma coluna de Total
                dados_formatados['Total'] = [
                    " / ".join([
                        # f"{df_beta.sum():.6f}",                   # Beta
                        # f"R${df_mvar_dinheiro.sum():,.2f}",       # MVar(R$)
                        # f"R${covar.sum():,.2f}",                  # CoVaR(R$)
                        f"{covar_perc.sum() * 100:.2f}%",              # CoVaR(%)
                        # f"R${var_ativos[assets].sum():,.2f}",      # Var
                        # % do Risco Total
                        f"{(covar_perc * abs(covar.sum() / var_limite_comparativo)).sum() * 100:.2f}%",
                    ])
                ]
                # Criar DataFrame com idx como única linha
                tabela_dados_risco = pd.DataFrame(
                    dados_formatados,  # Colunas = ativos, Valores = strings formatadas
                    index=[idx]        # Única linha = idx
                )
                colunas_selecionadas = []
                colunas_selecionadas.append('Beta')
                colunas_selecionadas.append('MVar(R$)')
                colunas_selecionadas.append('CoVaR(R$)')
                colunas_selecionadas.append('CoVaR(%)')
                colunas_selecionadas.append('Var')
                colunas_selecionadas.append('% do Risco Total')

                # Concatenando os dados se houver mais de uma iteração
                tabela_dados_riscos = pd.concat(
                    [tabela_dados_riscos, tabela_dados_risco], axis=0)

                # st.write("## Risco")
                if 'CoVaR(R$)' in colunas_selecionadas:
                    df_dados['CoVaR(R$)'] = df_dados['CoVaR(R$)'].apply(
                        lambda x: f"R${x:,.0f}")
                if 'MVar(R$)' in colunas_selecionadas:
                    df_dados['MVar(R$)'] = df_dados['MVar(R$)'].apply(
                        lambda x: f"R${x:,.0f}")
                if 'CoVaR(%)' in colunas_selecionadas:
                    df_dados['CoVaR(%)'] = df_dados['CoVaR(%)'].apply(
                        lambda x: f"{x:.2%}")
                if 'Beta' in colunas_selecionadas:
                    df_dados['Beta'] = df_dados['Beta'].apply(
                        lambda x: f"{x:.4f}")
                if 'Var' in colunas_selecionadas:
                    df_dados['Var'] = df_dados['Var'].apply(
                        lambda x: f"{x:.4f}%")
                if '% do Risco Total' in colunas_selecionadas:
                    df_dados['% do Risco Total'] = df_dados['% do Risco Total'].apply(
                        lambda x: f"{x:.2%}")

            # Display the filtered table
                if colunas_selecionadas:
                    # st.write("Tabela de Dados Selecionados:")
                    tabela_filtrada = df_dados[colunas_selecionadas]

                    # Adicionar uma linha de soma
                    sum_row = tabela_filtrada.select_dtypes(
                        include='number').sum()
                    sum_row['Beta'] = df_beta.sum()
                    sum_row['MVar(R$)'] = df_mvar_dinheiro.sum()
                    sum_row['CoVaR(R$)'] = covar.sum()
                    sum_row['CoVaR(%)'] = covar_perc.sum()
                    sum_row['Var'] = var_ativos[assets].sum()
                    sum_row['% do Risco Total'] = (
                        covar_perc * abs(covar.sum() / var_limite_comparativo)).sum()
                    sum_row = sum_row.to_frame().T
                    sum_row['Beta'] = sum_row['Beta'].apply(
                        lambda x: f"{x:.4f}")
                    sum_row['MVar(R$)'] = sum_row['MVar(R$)'].apply(
                        lambda x: f"R${x:,.0f}")
                    sum_row['CoVaR(R$)'] = sum_row['CoVaR(R$)'].apply(
                        lambda x: f"R${x:,.0f}")
                    sum_row['CoVaR(%)'] = sum_row['CoVaR(%)'].apply(
                        lambda x: f"{x:.2%}")
                    sum_row['Var'] = sum_row['Var'].apply(lambda x: f"{x:.4f}")
                    sum_row['% do Risco Total'] = sum_row['% do Risco Total'].apply(
                        lambda x: f"{x:.2%}")

                    sum_row = sum_row[colunas_selecionadas]
                    # Adicionar índice 'Total'
                    sum_row.index = ['Total']
                    # Adicionar a linha de soma na tabela filtrada
                    tabela_filtrada_com_soma = pd.concat(
                        [tabela_filtrada, sum_row])
                    # Preciso criar uma linha com dados nas colunas de todos os fundos
                    # Criando um dicionário para armazenar os valores formatados corretamente
                    dados = {
                        'Juros Nominais Brasil': [
                            f"{abs(df_divone_juros_nominais.iloc[0] / soma_pl_sem_pesos * 10000):,.2f}bps / "
                            f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']):,.2f}bps"
                        ],
                        'Juros Reais Brasil': [
                            f"{abs(df_divone_juros_real.iloc[0]/ soma_pl_sem_pesos * 10000):,.2f}bps / "
                            f"{abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']):,.2f}bps"
                        ],
                        'Juros US': [
                            f"{abs(df_divone_juros_externo_certo.iloc[0]/ soma_pl_sem_pesos * 10000):,.2f}bps / "
                            f"{abs(stress_test_juros_externo_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_externo_percent['FUT_TICK_VAL']):,.2f}bps"
                        ],
                        'Moedas': [
                            f"{abs(df_divone_dolar.iloc[0]/ soma_pl_sem_pesos * 10000):,.2f}bps /"
                            f"{abs(stress_dolar_percent):,.2f}bps" if lista_dolar else f"{abs(stress_dolar_percent):,.2f}bps"
                        ]
                    }

                    # Criando a linha "Total"
                    # dados['Total'] = [
                    #    f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real.iloc[0]) + abs(df_divone_juros_externo_certo.iloc[0]) + (abs(df_divone_dolar.iloc[0]) if lista_dolar else 0):,.2f} / "
                    #    f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo) + abs(stress_dolar):,.2f}" if lista_juros_externo else f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo['FUT_TICK_VAL']) + abs(stress_dolar):,.2f} / "
                    #    f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent) + abs(stress_dolar_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent['FUT_TICK_VAL']) + abs(stress_dolar_percent):,.2f}bps"
                    # ]

                    # Criando o DataFrame
                    tabela_dados_fundo = pd.DataFrame(dados, index=[idx])

                    # Concatenando os dados se houver mais de uma iteração
                    tabela_dados_fundos = pd.concat(
                        [tabela_dados_fundos, tabela_dados_fundo], axis=0)

                    # st.table(tabela_filtrada_com_soma)
        # Criar coluna de Total
        # sum_row = pd.DataFrame({
        #    'PL_Sem_Peso (R$)': f"R${soma_pl_sem_pesos:,.0f}",
        #    'VaR Limite (bps)': f"{var_limite_comparativo/ soma_pl_sem_pesos *10000 :,.2f}bps",
        #    'VaR Portfólio (bps)': f"{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps",
        #    'CVaR (bps)': f"{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps",
        #    'Volatilidade': f"{vol_port_analitica:.2%}",
        #    '% Risco Total': f"{abs(covar.sum()/ var_limite_comparativo):.2%}"
        # }, index=['Total'])
        # df_portfolio_final = pd.concat([df_portfolio_final, sum_row])
        # Criar coluna de Total
        # Tirar bps e / de cada célula
        # Processar tabela_dados_fundos_p1 (valor antes de '/')
        tabela_dados_fundos_p1 = tabela_dados_fundos.copy()
        tabela_dados_fundos_p1 = tabela_dados_fundos_p1.applymap(
            lambda x: x.replace('bps', '') if isinstance(x, str) else x)
        tabela_dados_fundos_p1 = tabela_dados_fundos_p1.applymap(
            lambda x: x.split('/')[0] if isinstance(x, str) and '/' in x else x)
        # Converter os valores para float
        tabela_dados_fundos_p1 = tabela_dados_fundos_p1.astype(float)
        # Adicionar a coluna 'Total' somando as colunas
        tabela_dados_fundos_p1['Total'] = tabela_dados_fundos_p1.sum(
            axis=1).round(2)  # Soma por linha (axis=1)

        # Processar tabela_dados_fundos_p2 (valor após '/')
        tabela_dados_fundos_p2 = tabela_dados_fundos.copy()
        tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(
            lambda x: x.replace('bps', '') if isinstance(x, str) else x)
        tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(
            lambda x: x.split('/')[1] if isinstance(x, str) and '/' in x else x)

        # Aqui, é necessário tratar os valores corretamente
        # Se o valor for '0', defina como 0, caso contrário converta para float
        tabela_dados_fundos_p2 = tabela_dados_fundos_p2.applymap(
            lambda x: float(x) if isinstance(x, str) and x != '0' else 0)

        # Somar as colunas específicas para obter o total
        try:
            tabela_dados_fundos_p2['Total'] = tabela_dados_fundos_p2[[
                'Juros Nominais Brasil', 'Juros Reais Brasil', 'Moedas', 'Juros US']].sum(axis=1).round(2)  # Soma por linha (axis=1)

            # Concatenar a informação de 'Total' nas duas tabelas (tabela_dados_fundos_p1 e tabela_dados_fundos_p2)
            tabela_dados_fundos_p1['Total'] = tabela_dados_fundos_p1['Total'].astype(
                str) + 'bps / ' + tabela_dados_fundos_p2['Total'].astype(str) + 'bps'

            # Copiar a coluna 'Total' para a tabela final
            tabela_dados_fundos['Total'] = tabela_dados_fundos_p1['Total']
            # Preciso juntar as colunas que sejam Juros Nominais Brasil, Juros Reais Brasil, Moedas e Juros US
            # Identificar categorias por colunas
            mapeamento_categorias = {
                "Juros Nominais Brasil": [col for col in tabela_dados_riscos.columns if "DI" in col],
                "Juros Reais Brasil": [col for col in tabela_dados_riscos.columns if "DAP" in col],
                "Juros US": ["TREASURY"],
                "Moedas": ["WDO1"]
            }
            # Criar novo DataFrame com as categorias
            # Criar novo DataFrame com as categorias
            nova_tabela = pd.DataFrame()
            nova_tabela["Fundos"] = tabela_dados_riscos.index

            # Dicionário para armazenar valores da soma total
            soma_total_antes = []
            soma_total_depois = []

            for categoria, colunas in mapeamento_categorias.items():
                valores_antes = []
                valores_depois = []

                for _, row in tabela_dados_riscos.iterrows():
                    soma_antes = 0
                    soma_depois = 0

                    for col in colunas:
                        if col in tabela_dados_riscos.columns:
                            partes = row[col].split(" / ")
                            soma_antes += float(partes[0].replace("%", ""))
                            soma_depois += float(partes[1].replace("%", ""))

                    valores_antes.append(soma_antes)
                    valores_depois.append(soma_depois)

                # Adicionar ao DataFrame
                nova_tabela[categoria] = [
                    f"{antes:.2f}% / {depois:.2f}%" for antes, depois in zip(valores_antes, valores_depois)]

                # Acumulando valores para a coluna Total
                if not soma_total_antes:
                    soma_total_antes = valores_antes
                    soma_total_depois = valores_depois
                else:
                    soma_total_antes = [
                        x + y for x, y in zip(soma_total_antes, valores_antes)]
                    soma_total_depois = [
                        x + y for x, y in zip(soma_total_depois, valores_depois)]

            # Criar coluna Total
            nova_tabela["Total"] = [
                f"{antes:.2f}% / {depois:.2f}%" for antes, depois in zip(soma_total_antes, soma_total_depois)]
            nova_tabela.set_index("Fundos", inplace=True)
            colu1, colu3, colu2 = st.columns([4.9, 0.2, 4.9])
            with colu1:
                st.table(df_portfolio_final)
            # Exibir tabela reorganizada
            if op1:
                with colu2:
                    st.write("### Analise Risco por Categoria")
                    st.table(nova_tabela)
                    st.markdown(
                        "<p style='font-size: 18px; font-style: italic;'>(CoVaR bps / % Risco Total)</p>", unsafe_allow_html=True)

            if op2:
                with colu2:
                    st.write("### Analise Estratégias")
                    st.table(tabela_dados_fundos)
                    st.markdown(
                        "<p style='font-size: 18px; font-style: italic;'>(Div01 bps / Stress bps)</p>", unsafe_allow_html=True)
            with colu3:
                st.html(
                    '''
                    <div class="divider-vertical-line"></div>
                    <style>
                        .divider-vertical-line {
                            border-left: 2px solid rgba(49, 51, 63, 0.2);
                            height: 80vh;
                            margin: auto;
                        }
                        @media (max-width: 768px) {
                            .divider-vertical-line {
                                display: none;
                            }
                        }

                    </style>
                    '''
                )

            return
        except:
            st.write("Nenhum fundo selecionado / Nenhum contrato cadastrado")
            return
    else:
        st.write("Nenhum fundo selecionado / Nenhum contrato cadastrado")
        return


def calcular_metricas_de_port(assets, quantidades, df_contratos):
    file_pl = "Dados/pl_fundos.parquet"
    df_pl = pd.read_parquet(file_pl)
    df_pl = df_pl.set_index(df_pl.columns[0])
    file_bbg = "Dados/BBG - ECO DASH.xlsx"
    # Dicionário de pesos fixo (pode-se tornar dinâmico no futuro)
    dict_pesos = {
        'GLOBAL BONDS': 4,
        'HORIZONTE': 1,
        'JERA2026': 1,
        'REAL FIM': 1,
        'BH FIRF INFRA': 1,
        'BORDEAUX INFRA': 1,
        'TOPAZIO INFRA': 1,
        'MANACA INFRA FIRF': 1,
        'AF DEB INCENTIVADAS': 3
    }
    # Zerar os pesos de fundos que não tem contratos
    for idx, row in df_contratos.iterrows():
        if idx == 'Total':
            continue
        else:
            fundo = idx
            check = 0
            for asset in assets:
                if int(row[f'Contratos {asset}']) != 0:
                    check = 1
            if check == 0:
                dict_pesos[fundo] = 0

    Weights = list(dict_pesos.values())
    df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio(
        df_pl, Weights)

    df = pd.read_parquet('Dados/df_inicial.parquet')

    df_precos, df_completo = load_and_process_excel(df, assets)
    df_retorno = process_returns(df_completo, assets)
    var_ativos = var_not_parametric(df_retorno).abs()
    df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)
    quantidade_nomes = {}
    for i in range(len(assets)):
        quantidade_nomes[assets[i]] = quantidades[i]

    quantidade = np.array(list(quantidade_nomes.values()))

    # Valor do Portfólio (soma simples)
    vp = df_precos_ajustados['Valor Fechamento'] * abs(quantidade)

    vp_soma = vp.sum()

    # Pesos (p/ cálculo de VaR, etc.)
    pesos = quantidade * \
        df_precos_ajustados['Valor Fechamento'] / vp_soma
    df_returns_portifolio = df_retorno * pesos.values
    df_returns_portifolio['Portifolio'] = df_returns_portifolio.sum(
        axis=1)

   # Var

    # VaR
    var_port = var_not_parametric(df_returns_portifolio['Portifolio'])
    var_port = abs(var_port)
    var_port_dinheiro = vp_soma * var_port

    # ######################################################################################
    # ######################################################################################
    # #####################################################################################
    ######################################################################################
    ######## var bps e var limite são fixos, mas podem ser dinâmicos no futuro ########################

    var_bps = 1.0
    var_bps = var_bps / 10000

    var_limite = 1.0

    vol_port_retornos = df_returns_portifolio['Portifolio'].std()
    vol_port_analitica = vol_port_retornos * np.sqrt(252)

    df_retorno['Portifolio'] = df_returns_portifolio['Portifolio']
    cov = df_retorno.cov()
    cov_port = cov['Portifolio'].drop('Portifolio')
    df_beta = cov_port / (vol_port_retornos**2)
    df_mvar = df_beta * var_port
    df_mvar_dinheiro = df_mvar * \
        df_precos_ajustados['Valor Fechamento']

    covar = df_mvar * pesos.values * vp_soma
    covar_perc = covar / covar.sum()

    cvar = df_retorno[df_retorno['Portifolio'] < var_not_parametric(
        df_returns_portifolio['Portifolio'])]['Portifolio'].mean()

    df_divone, dolar, treasury = load_and_process_divone(
        'Dados/BBG - ECO DASH.xlsx', df)
    # --- Exemplo de cálculo de stress e DIVONE (mesmo que seu original) ---
    lista_juros_interno = [
        asset for asset in assets if 'DI' in asset]
    df_divone_juros_nominais = df_divone[lista_juros_interno]

    lista_quantidade = [quantidade_nomes[asset]
                        for asset in lista_juros_interno]
    df_divone_juros_nominais = df_divone_juros_nominais * \
        np.array(lista_quantidade)
    df_divone_juros_nominais = df_divone_juros_nominais.sum(axis=1)

    lista_juros_interno_real = [a for a in assets if ('DAP' in a) or ('NTNB' in a)]

    df_divone_juros_real = df_divone[lista_juros_interno_real]

    lista_quantidade = [quantidade_nomes[asset]
                        for asset in lista_juros_interno_real]

    df_divone_juros_real = df_divone_juros_real * \
        np.array(lista_quantidade)

    df_divone_juros_real = df_divone_juros_real.sum(axis=1)

    lista_juros_externo = [
        asset for asset in assets if 'TREASURY' in asset]

    df_divone_juros_externo = df_divone[lista_juros_externo]

    lista_quantidade = [quantidade_nomes[asset]
                        for asset in lista_juros_externo]

    df_divone_juros_externo = df_divone_juros_externo * \
        np.array(lista_quantidade)

    df_divone_juros_externo = df_divone_juros_externo.sum(axis=1)

    stress_test_juros_interno_Nominais = df_divone_juros_nominais * 100
    stress_test_juros_interno_Nominais_percent = stress_test_juros_interno_Nominais / \
        soma_pl_sem_pesos * 10000

    stress_test_juros_interno_Reais = df_divone_juros_real * 50
    stress_test_juros_interno_Reais_percent = stress_test_juros_interno_Reais / \
        soma_pl_sem_pesos * 10000

    df_divone_juros_externo_certo = df_divone_juros_externo

    if lista_juros_externo:
        valor_acumulado_treasury = (1+df_retorno['TREASURY']).cumprod()
        pico_max_treasury = valor_acumulado_treasury.max()
        drawndown_treasury = (valor_acumulado_treasury - pico_max_treasury) / \
            pico_max_treasury
        drawndown_treasury = drawndown_treasury.min()
        drawndown_treasury = df_retorno['TREASURY'].min()
        df_divone_juros_externo = drawndown_treasury
        df_divone_juros_externo = abs(
            df_divone_juros_externo) * treasury * dolar / 10000
        df_divone_juros_externo = df_divone_juros_externo * \
            np.array(lista_quantidade)
        df_divone_juros_externo = df_divone_juros_externo.sum()

    stress_test_juros_externo = df_divone_juros_externo

    stress_test_juros_externo_percent = stress_test_juros_externo / \
        soma_pl_sem_pesos * 10000

    lista_dolar = [
        asset for asset in assets if 'WDO1' in asset]
    if lista_dolar:
        quantidade_dolar = quantidade_nomes[lista_dolar[0]]
        stress_dolar = quantidade_dolar * dolar * 0.02
        valor_acumulado = (1+df_retorno['WDO1']).cumprod()
        pico_max = valor_acumulado.max()
        drawndown_dolar = (valor_acumulado - pico_max) / pico_max
        drawndown_dolar = drawndown_dolar.min()
        drawndown_dolar = df_retorno['WDO1'].min()
        df_divone_dolar = drawndown_dolar
        df_divone_dolar = df_divone_dolar * quantidade_dolar
        df_divone_dolar = abs(df_divone_dolar) * dolar
        stress_dolar = df_divone_dolar
        stress_dolar_percent = stress_dolar / soma_pl_sem_pesos * 10000
        df_divone_dolar = df_divone[lista_dolar] * \
            np.array(quantidade_dolar)
        df_divone_dolar = df_divone_dolar.sum()

    else:
        stress_dolar = 0
        df_divone_dolar = 0

    stress_dolar_percent = stress_dolar / soma_pl_sem_pesos * 10000
    df_stress_div01 = pd.DataFrame({
        'DIV01': [
            f"R${abs(df_divone_juros_nominais.iloc[0]):,.2f}",
            f"R${abs(df_divone_juros_real.iloc[0]):,.2f}",
            f"R${abs(df_divone_juros_externo_certo.iloc[0]):,.2f}",
            f'R${abs(df_divone_dolar.iloc[0]):,.2f}' if lista_dolar else 0
        ],
        'Stress (R$)': [
            f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']):,.2f}",
            f"R${abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']):,.2f}",
            f"R${abs(stress_test_juros_externo):,.2f}" if lista_juros_externo else f"R${abs(stress_test_juros_externo['FUT_TICK_VAL']):,.2f}",
            f"R${abs(stress_dolar):,.2f}"
        ],
        'Stress (bps)': [
            f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']):,.2f}bps",
            f"{abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']):,.2f}bps",
            f"{abs(stress_test_juros_externo_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_externo_percent['FUT_TICK_VAL']):,.2f}bps",
            f"{abs(stress_dolar_percent):,.2f}bps"
        ]
    }, index=['Juros Nominais Brasil', 'Juros Reais Brasil', 'Juros US', 'Moedas'])

    sum_row = pd.DataFrame({
        'DIV01': [f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real[0]) + abs(df_divone_juros_externo_certo.iloc[0]) + abs(df_divone_dolar.iloc[0]):,.2f}"] if lista_dolar else [f"R${abs(df_divone_juros_nominais.iloc[0]) + abs(df_divone_juros_real[0]) + abs(df_divone_juros_externo_certo.iloc[0]):,.2f}"],
        'Stress (R$)': [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo) + abs(stress_dolar):,.2f}"] if lista_juros_externo else [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo['FUT_TICK_VAL']) + abs(stress_dolar):,.2f}"],
        'Stress (bps)': [f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent) + abs(stress_dolar_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent['FUT_TICK_VAL']) + abs(stress_dolar_percent):,.2f}bps"]
    }, index=['Total'])
    df_stress_div01 = pd.concat([df_stress_div01, sum_row])

    df_precos_ajustados = calculate_portfolio_values(
        df_precos_ajustados, df_pl_processado, var_bps)
    df_pl_processado = calculate_contracts_per_fund(
        df_pl_processado, df_precos_ajustados)

    # --- Layout ---
    st.write("## Dados do Portfólio")
    st.write(f"**PL: R$ {soma_pl_sem_pesos:,.0f}**")

    var_limite_comparativo = soma_pl * var_bps * var_limite
    st.write(
        f"**VaR Limite** (Peso de {var_limite:.1%}): R${var_limite_comparativo:,.0f}"
    )

    st.write(
        f"**VaR do Portfólio**: R${var_port_dinheiro:,.0f} : **{var_port_dinheiro/soma_pl_sem_pesos * 10000:.2f}bps**"
    )
    st.write(
        f"**CVaR**: R${abs(cvar * vp_soma):,.0f} : **{abs(cvar * vp_soma)/soma_pl_sem_pesos * 10000:.2f}bps**"
    )
    st.write(f"**Volatilidade**: {vol_port_analitica:.2%}")
    st.table(df_stress_div01)

    st.write("---")
    st.write(
        f"### {abs(covar.sum()/ var_limite_comparativo):.2%} do risco total")

    df_dados = pd.DataFrame({
        'Beta': df_beta,
        'MVar(R$)': df_mvar_dinheiro,
        'CoVaR(R$)': covar,
        'CoVaR(%)': covar_perc,
        'Var': var_ativos[assets],
        '% do Risco Total': covar_perc * abs(covar.sum() / var_limite_comparativo)
    })

    colunas_selecionadas = []
    colunas_selecionadas.append('Beta')
    colunas_selecionadas.append('MVar(R$)')
    colunas_selecionadas.append('CoVaR(R$)')
    colunas_selecionadas.append('CoVaR(%)')
    colunas_selecionadas.append('Var')
    colunas_selecionadas.append('% do Risco Total')

    st.write("## Risco")
    if 'CoVaR(R$)' in colunas_selecionadas:
        df_dados['CoVaR(R$)'] = df_dados['CoVaR(R$)'].apply(
            lambda x: f"R${x:,.0f}")
    if 'MVar(R$)' in colunas_selecionadas:
        df_dados['MVar(R$)'] = df_dados['MVar(R$)'].apply(
            lambda x: f"R${x:,.0f}")
    if 'CoVaR(%)' in colunas_selecionadas:
        df_dados['CoVaR(%)'] = df_dados['CoVaR(%)'].apply(
            lambda x: f"{x:.2%}")
    if 'Beta' in colunas_selecionadas:
        df_dados['Beta'] = df_dados['Beta'].apply(
            lambda x: f"{x:.4f}")
    if 'Var' in colunas_selecionadas:
        df_dados['Var'] = df_dados['Var'].apply(
            lambda x: f"{x:.4f}%")
    if '% do Risco Total' in colunas_selecionadas:
        df_dados['% do Risco Total'] = df_dados['% do Risco Total'].apply(
            lambda x: f"{x:.2%}")

# Display the filtered table
    if colunas_selecionadas:
        st.write("Tabela de Dados Selecionados:")
        tabela_filtrada = df_dados[colunas_selecionadas]

        # Adicionar uma linha de soma
        sum_row = tabela_filtrada.select_dtypes(
            include='number').sum()
        sum_row['Beta'] = df_beta.sum()
        sum_row['MVar(R$)'] = df_mvar_dinheiro.sum()
        sum_row['CoVaR(R$)'] = covar.sum()
        sum_row['CoVaR(%)'] = covar_perc.sum()
        sum_row['Var'] = var_ativos[assets].sum()
        sum_row['% do Risco Total'] = (
            covar_perc * abs(covar.sum() / var_limite_comparativo)).sum()
        sum_row = sum_row.to_frame().T
        sum_row['Beta'] = sum_row['Beta'].apply(
            lambda x: f"{x:.4f}")
        sum_row['MVar(R$)'] = sum_row['MVar(R$)'].apply(
            lambda x: f"R${x:,.0f}")
        sum_row['CoVaR(R$)'] = sum_row['CoVaR(R$)'].apply(
            lambda x: f"R${x:,.0f}")
        sum_row['CoVaR(%)'] = sum_row['CoVaR(%)'].apply(
            lambda x: f"{x:.2%}")
        sum_row['Var'] = sum_row['Var'].apply(lambda x: f"{x:.4f}")
        sum_row['% do Risco Total'] = sum_row['% do Risco Total'].apply(
            lambda x: f"{x:.2%}")

        sum_row = sum_row[colunas_selecionadas]
        # Adicionar índice 'Total'
        sum_row.index = ['Total']
        # Adicionar a linha de soma na tabela filtrada
        tabela_filtrada_com_soma = pd.concat(
            [tabela_filtrada, sum_row])
        st.table(tabela_filtrada_com_soma)
    return soma_pl_sem_pesos


def checkar2_portifolio(assets,
                        quantidades,
                        # dict { 'PETR4': 200, 'VALE3': 150 } etc.
                        # dict { 'PETR4': 200, 'VALE3': 150 } etc.
                        compra_especifica,
                        # dict { 'PETR4': '17/01/2025', 'VALE3': '16/01/2025' }
                        dia_compra,
                        ):
    """
    Verifica (ou cria) um parquet de portfólio e atualiza as posições de acordo com:
      1) Lista de ativos (assets) e suas quantidades (quantidades).
      2) Dicionário de compra_especifica (caso o usuário tenha quantidades específicas para algum ativo).
      3) Dia de compra, que pode ser um único valor ou um dicionário.
      4) Atualiza Preço de Compra, Preço de Ajuste Atual, Variação e Rendimento.
    """
    nome_arquivo_portifolio = 'Dados/portifolio_posições.parquet'
    df_b3_fechamento = processar_b3_portifolio()
    # 1) Carrega portfólio existente (se existir)
    if os.path.exists(nome_arquivo_portifolio):
        df_portifolio = pd.read_parquet(nome_arquivo_portifolio, index_col=0)

    else:
        # Podemos criar um DataFrame com colunas definidas para evitar problemas de colunas inexistentes
        df_portifolio = pd.DataFrame(columns=[
            'Quantidade',
            'Dia de Compra',
            'Preço de Compra',
            'Preço de Ajuste Atual',
            'Rendimento'
        ])

    # 2) Remover do DataFrame os ativos que NÃO estão na nova lista de assets
    #    (significa que o usuário removeu aquele ativo do portfólio)
    ativos_existentes = df_portifolio.index.tolist()
    for ativo_em_port in ativos_existentes:
        if ativo_em_port not in assets:
            df_portifolio.drop(index=ativo_em_port, inplace=True)

    # 3) Adicionar ou atualizar as quantidades dos ativos que estão na nova lista
    #    Se há compra_especifica, ela sobrescreve a quantidade
    #    Também atualizamos o 'Dia de Compra' se fornecido
    for asset in assets:

        qtd_final = quantidades[asset]

        # Verifica se o dia_compra é dict ou valor único
        if isinstance(dia_compra, dict):
            dia_de_compra_atual = dia_compra.get(asset, None)
        else:
            # Se dia_compra for apenas uma string (valor único)
            dia_de_compra_atual = dia_compra
        if asset not in df_portifolio.index:
            # Se não existe, cria linha nova
            df_portifolio.loc[asset, 'Quantidade'] = qtd_final
            df_portifolio.loc[asset, 'Dia de Compra'] = dia_de_compra_atual
        else:
            # Se já existe, apenas atualiza
            df_portifolio.loc[asset, 'Quantidade'] = qtd_final
            # Se dia_compra foi passado como dict e existe valor pra esse ativo, atualiza

            if dia_de_compra_atual is not None:
                df_portifolio.loc[asset, 'Dia de Compra'] = dia_de_compra_atual

    # 4) Para cada ativo no DataFrame final, calculamos
    #    - Preço de Compra (com base em df_b3_fechamento na data de compra)
    #    - Preço de Ajuste Atual (com base em df_b3_fechamento na data de hoje)
    #    - Variação de Taxa (com base em df_b3_variacao na data de compra ou acumulada, depende da necessidade)
    #    - Rendimento (Quantidade * (Preço de Ajuste Atual - Preço de Compra)) ou vice-versa

    # Monta a string da data de hoje no mesmo formato das colunas do parquet
    # As colunas do parquet podem estar em formato "dd/mm/yyyy" ou "yyyy-mm-dd", dependendo de como foi salvo
    # Ajuste conforme a forma em que foram salvas:

    ############################## ---- PROBLEMA ---- ##############################
    #
    data_hoje_str = datetime.date.today().strftime('%Y-%m-%d')

    for asset in df_portifolio.index:
        try:
            # Dia de compra que foi salvo
            dia_compra_ativo = df_portifolio.loc[asset, 'Dia de Compra']
            # (Opcional) se for "yyyy-mm-dd", converter para "dd/mm/yyyy" para casar com o parquet
            # Caso já esteja no formato correto, você pode pular esta conversão.
            # Exemplo (caso precise):
            # dia_compra_dt = datetime.datetime.strptime(dia_compra_ativo, '%Y-%m-%d')
            # dia_compra_ativo = dia_compra_dt.strftime('%d/%m/%Y')

            # PREÇO DE COMPRA
            # Localiza a linha do ativo e depois pega o valor na coluna da data de compra
            filtro_ativo = (df_b3_fechamento['Assets'] == asset)

            # Se o usuário informou um valor de compra_especifica para este ativo, sobrescreve
            if compra_especifica and (asset in compra_especifica):
                preco_compra = compra_especifica[asset]
                print(preco_compra)
            else:
                preco_compra = df_b3_fechamento.loc[filtro_ativo,
                                                    dia_compra_ativo].values[0]
                print(preco_compra)
                print(
                    df_b3_fechamento.loc[filtro_ativo, dia_compra_ativo].values[0])

            df_portifolio.loc[asset, 'Preço de Compra'] = preco_compra

            # PREÇO DE AJUSTE ATUAL
            # Usa a data de hoje (coluna de hoje) no df_b3_fechamento
            preco_ajuste_atual = df_b3_fechamento.loc[filtro_ativo,
                                                      data_hoje_str].values[0]
            df_portifolio.loc[asset,
                              'Preço de Ajuste Atual'] = preco_ajuste_atual

            # VARIAÇÃO DE TAXA
            # Depende se você quer pegar a variação apenas do dia_compra_ativo ou a variação acumulada.
            # Abaixo apenas um exemplo de pegar a variação no dia de compra (ou outro critério)

            # RENDIMENTO
            # De acordo com seu snippet original, mas usualmente seria (Preço Atual - Preço Compra).
            df_portifolio.loc[asset, 'Rendimento'] = (
                df_portifolio.loc[asset, 'Quantidade'] *
                (df_portifolio.loc[asset, 'Preço de Ajuste Atual'] -
                 df_portifolio.loc[asset, 'Preço de Compra'])
            )

        except Exception as e:
            # Caso alguma coluna não seja encontrada ou dê erro, você pode tratar aqui
            print(f"[*] Erro ao calcular valores para o ativo {asset}: {e}")

    # 5) Salva o DataFrame atualizado de volta no parquet
    df_portifolio.to_parquet(nome_arquivo_portifolio)

    return df_portifolio


def editar_ou_remover():
    # Carregar dados do Supabase
    df_supabase = load_data()
    if not df_supabase:
        st.warning("Nenhum dado encontrado na base.")
        return
    df = pd.DataFrame(df_supabase)
    # Mostrar tabela de ativos
    st.write("### Base de Dados Atual")
    st.table(df)

    # Escolher ação: Editar ou Remover
    edit_or_remove = "Remover"

    # Escolher ativo específico
    ativo_escolhido = st.selectbox("Escolha o ativo", df["Ativo"].unique())

    # Filtrar os registros do ativo escolhido
    df_filtrado = df[df["Ativo"] == ativo_escolhido]

    # Caso tenha múltiplos registros (diferentes datas de compra), permitir escolha específica
    if len(df_filtrado) > 1:
        dia_compra_escolhido = st.selectbox(
            "Escolha a data de compra",
            df_filtrado["Dia de Compra"].astype(str).unique()
        )
        df_filtrado = df_filtrado[df_filtrado["Dia de Compra"]
                                  == dia_compra_escolhido]
    else:
        dia_compra_escolhido = df_filtrado.iloc[0]["Dia de Compra"]

    # Remover ativo
    if edit_or_remove == "Remover":
        if st.button("Remover Ativo"):
            # Criar função para deletar
            delete_data(ativo_escolhido, dia_compra_escolhido)
            update_base_fundos(ativo_escolhido, dia_compra_escolhido)

            st.success(f"Ativo {ativo_escolhido} removido com sucesso!")

    # Editar ativo
    else:
        quantidade = st.number_input(
            "Quantidade", value=int(df_filtrado.iloc[0]["Quantidade"]))
        preco = st.number_input("Preço de Compra", value=float(
            df_filtrado.iloc[0]["Preço de Compra"]))
        dia_compra = st.date_input("Dia de Compra", value=datetime.datetime.strptime(
            dia_compra_escolhido, "%Y-%m-%d"))
        preco_ajuste = float(df_filtrado.iloc[0]["Preço de Ajuste Atual"])
        rendimento = (preco_ajuste - preco) * quantidade

        data_editada = {
            'Ativo': ativo_escolhido,
            'Quantidade': quantidade,
            'Dia de Compra': dia_compra.strftime("%Y-%m-%d"),
            'Preço de Compra': preco,
            'Preço de Ajuste Atual': preco_ajuste,
            'Rendimento': rendimento
        }

        if st.button("Salvar Edição"):
            update_data(data_editada)
            st.success(f"Ativo {ativo_escolhido} atualizado com sucesso!")
    st.html(
        '''
                <style>
                    div[data-testid="stSelectbox"] div {
                   color: black; /* Define o texto como preto */
                   background-color: white; /* Define o fundo como branco */
                                                    }     
            
                </style>   
        
                '''
    )


def update_base_fundos(ativo_escolhido, dia_compra_escolhido):
    try:
        pasta_base_fundos = "BaseFundos"
        data_formatada = dia_compra_escolhido
        metricas = ['PL', 'Preco_Fechamento',
                    'Preco_Compra', 'Quantidade', 'Rendimento']
        prefixo_colunas = [
            f"{data_formatada} - {metrica}" for metrica in metricas]

        resultados = {
            "arquivos_modificados": [],
            "tabelas_atualizadas": [],
            "erros": []
        }

        # Processar cada arquivo/tabela
        for arquivo in os.listdir(pasta_base_fundos):
            if arquivo.endswith(".parquet"):
                try:
                    table_name = os.path.splitext(arquivo)[0]
                    caminho_arquivo = os.path.join(pasta_base_fundos, arquivo)
                    modificado = False

                    # 1. Processar arquivo parquet
                    df = pd.read_parquet(caminho_arquivo)

                    if 'Ativo' in df.columns:
                        # Verificar se o ativo existe neste arquivo
                        mascara_ativo = df['Ativo'] == ativo_escolhido

                        if mascara_ativo.any():
                            # Remover colunas do dia específico
                            colunas_para_remover = [
                                col for col in prefixo_colunas if col in df.columns]

                            if colunas_para_remover:
                                df.drop(columns=colunas_para_remover,
                                        inplace=True, errors='ignore')
                                df.to_parquet(caminho_arquivo, index=False)
                                resultados["arquivos_modificados"].append(
                                    arquivo)
                                modificado = True

                    # 2. Atualizar tabela no Supabase
                    if modificado:
                        conn = psycopg2.connect(
                            dbname="postgres",
                            user="postgres.obgwfekirteetqzjydry",
                            password="hF81aEQwSDNzATLJ",
                            host="aws-0-sa-east-1.pooler.supabase.com",
                            port="6543"
                        )
                        cursor = conn.cursor()

                        # Obter colunas existentes de forma segura
                        cursor.execute(
                            sql.SQL("""
                                SELECT column_name 
                                FROM information_schema.columns 
                                WHERE table_name = {}
                            """).format(sql.Literal(table_name)))
                        existing_columns = {row[0]
                                            for row in cursor.fetchall()}

                        # Atualizar registros do ativo
                        for coluna in colunas_para_remover:
                            if coluna in existing_columns:
                                update_query = sql.SQL("""
                                    UPDATE {table} 
                                    SET {coluna} = NULL 
                                    WHERE "Ativo" = %s
                                """).format(
                                    table=sql.Identifier(table_name),
                                    coluna=sql.Identifier(coluna))
                                cursor.execute(
                                    update_query, (ativo_escolhido,))

                        # Verificar e remover colunas vazias
                        for coluna in colunas_para_remover:
                            if coluna in existing_columns:
                                # Query corrigida com identificadores seguros
                                cursor.execute(
                                    sql.SQL("""
                                        SELECT COUNT(*) 
                                        FROM {table} 
                                        WHERE {coluna} IS NOT NULL
                                    """).format(
                                        table=sql.Identifier(table_name),
                                        coluna=sql.Identifier(coluna))
                                )
                                if cursor.fetchone()[0] == 0:
                                    cursor.execute(
                                        sql.SQL("ALTER TABLE {table} DROP COLUMN IF EXISTS {coluna}").format(
                                            table=sql.Identifier(table_name),
                                            coluna=sql.Identifier(coluna)))

                        conn.commit()
                        cursor.close()
                        conn.close()
                        resultados["tabelas_atualizadas"].append(table_name)

                except Exception as e:
                    resultados["erros"].append({
                        "arquivo": arquivo,
                        "erro": str(e)
                    })
                    if 'conn' in locals():
                        conn.rollback()
                        if conn.closed == 0:
                            conn.close()

        return resultados

    except Exception as e:
        return {"status": "error", "erro": str(e)}


def load_data():
    try:
        response = supabase.table('portfolio_posicoes').select("*").execute()
        print("Resposta do Supabase:", response)  # Depuração
        return response.data
    except Exception as e:
        return []

# Adicionar um novo registro no Supabase


def add_data(data):
    try:
        # 1. Deleta todos os registros da tabela
        supabase.table('portfolio_posicoes').delete().neq('Ativo', 0).execute()

        # 2. Insere os dados novos
        response = supabase.table('portfolio_posicoes').insert(data).execute()
        print("Resposta do Supabase:", response)
        return response.data

    except Exception as e:
        print(f"Erro detalhado: {e}")
        return None


# Função para atualizar o registro local com o Supabase
def att_parquet_supabase():
    df_supabase = load_data()
    if not df_supabase:
        df = pd.DataFrame(columns=["Ativo", "Quantidade", "Dia de Compra",
                          "Preço de Compra", "Preço de Ajuste Atual", "Rendimento"])
        df.to_parquet("Dados/portifolio_posições.parquet", index=False)
        return
    df = pd.DataFrame(df_supabase)
    df.to_parquet("Dados/portifolio_posições.parquet", index=False)


def hist_posicoes_supabase():
    df_supabase = load_data()
    if not df_supabase:
        df = pd.DataFrame(columns=["Ativo", "Quantidade", "Dia de Compra",
                          "Preço de Compra", "Preço de Ajuste Atual", "Rendimento"])
        df.to_parquet("Dados/portifolio_posições.parquet", index=False)
        return
    df = pd.DataFrame(df_supabase)
    return df



def update_data(data):
    try:
        ativo = data['Ativo']
        dia = data['Dia de Compra']

        response = (
            supabase
            .table('portfolio_posicoes')
            .update(data)
            .eq('Ativo', ativo)  # Filtra pelo ativo correto
            .eq('Dia de Compra', dia)  # Filtra pela quantidade correta
            .execute()
        )

        print("Resposta do Supabase:", response)
        att_parquet_supabase()
        return response.data

    except Exception as e:
        return print(f"Erro detalhado: {e}")


# Função para deletar registro do Supabase
def delete_data(ativo, dia_compra):
    try:
        response = supabase.table("portfolio_posicoes").delete().match(
            {"Ativo": ativo, "Dia de Compra": dia_compra}).execute()
        print("Registro removido:", response)
    except Exception as e:
        print("Erro ao remover ativo:", e)


def atualizar_parquet_fundos(
    df_current,         # DataFrame do dia atual (1 linha por Fundo)
    dia_operacao,       # Exemplo: "2025-01-20"
    # DF de transações: [Ativo, Quantidade, Dia de Compra, Preço de Compra, ...]
    df_info,
    # DF de preços de fechamento B3: colunas ["Assets", <data1>, <data2>, ...]
    quantidade_nomes,
):

    # 1) Ler df_fechamento_b3 e tratar
    df_fechamento_b3 = pd.read_parquet(
        "Dados/df_preco_de_ajuste_atual_completo.parquet")
    df_fechamento_b3 = df_fechamento_b3.replace('\.', '', regex=True)
    df_fechamento_b3 = df_fechamento_b3.replace({',': '.'}, regex=True)
    # Converter para float todas as colunas menos a primeira
    df_fechamento_b3.iloc[:, 2:] = df_fechamento_b3.iloc[:, 2:].astype(float)

    # Multiplicar a linha que tem o Ativo TREASURY por 1000 e WDO1 por 10
    df_fechamento_b3.loc[df_fechamento_b3['Assets'] == 'TREASURY',
                         df_fechamento_b3.columns != 'Assets'] *= 1000
    df_fechamento_b3.loc[df_fechamento_b3['Assets'] == 'WDO1',
                         df_fechamento_b3.columns != 'Assets'] *= 10

    ultimo_fechamento = df_fechamento_b3.columns[-1]
    dolar = df_fechamento_b3.loc[
        df_fechamento_b3['Assets'] == 'WDO1', ultimo_fechamento
    ].values[0]

    # 2) Ler pl_dias e tratar
    pl_dias = pd.read_parquet("Dados/pl_fundos_teste.parquet")
    pl_dias = pl_dias.replace('\.', '', regex=True)
    pl_dias = pl_dias.replace({',': '.'}, regex=True)
    for col in pl_dias.columns:
        pl_dias[col] = pl_dias[col].astype(str)
        if col != 'Fundos/Carteiras Adm':
            pl_dias[col] = pl_dias[col].str.replace('R$', '')
            pl_dias[col] = pl_dias[col].replace('--', np.nan)
            pl_dias[col] = pl_dias[col].astype(float, errors='ignore')

    pl_dias = pl_dias.set_index("Fundos/Carteiras Adm")
    pl_dias_vetor = []

    # 3) Placeholder e mensagens
    status_container = st.empty()
    mensagens = ["⏳ Aguarde até o Total ser concluído..."]
    status_container.markdown(" | ".join(mensagens))
    # Carregar o DataFrame de ajustes (caso tenha DAPs)
    df_ajuste = pd.read_parquet('Dados/df_valor_ajuste_contrato.parquet')
    # Ver se a ultima coluna é igual a penultima
    if df_ajuste.iloc[:, -1].equals(df_ajuste.iloc[:, -2]):
        df_ajuste = df_ajuste.iloc[:, :-1]
    colunas_datas = df_ajuste.columns[1:]
    df_ajuste[colunas_datas] = df_ajuste[colunas_datas].replace(
        '\.', '', regex=True).replace(',', '.', regex=True)
    df_ajuste[colunas_datas] = df_ajuste[colunas_datas].astype(float)

    # Converter nomes das colunas para datetime
    datas_convertidas = pd.to_datetime(colunas_datas, errors='coerce')
    colunas_datas_validas = [col for col, data in zip(
        df_ajuste.columns[1:], datas_convertidas) if pd.notnull(data)]
    df_ajuste = df_ajuste[['Assets'] + colunas_datas_validas]
    datas_validas = pd.to_datetime(colunas_datas_validas)

    # 4) Loop principal: para cada Fundo (linha de df_current)
    for fundo, row_fundo in df_current.iterrows():
        nome_arquivo_parquet = os.path.join("BaseFundos", f"{fundo}.parquet")

        # 4.1) Carrega ou cria df_fundo
        if os.path.exists(nome_arquivo_parquet):
            df_fundo = pd.read_parquet(nome_arquivo_parquet)
            # REMOVE O TRECHO DE DROP que existia antes:
            # if df_fundo.columns.str.startswith(dia_operacao).any():
            #     df_fundo = df_fundo.drop(
            #         columns=df_fundo.columns[df_fundo.columns.str.startswith(dia_operacao)]
            #     )
            if "Ativo" in df_fundo.columns:
                df_fundo.set_index("Ativo", inplace=True, drop=False)
        else:
            df_fundo = pd.DataFrame(
                columns=["Ativo", "Preco_Fechamento_Atual"])
            if "Ativo" in df_fundo.columns:
                df_fundo.set_index("Ativo", inplace=True, drop=False)

        # 4.2) Filtra as transações do dia para encontrar os ativos
        subset = df_info[df_info["Dia de Compra"] == dia_operacao]

        # pegar as keys do dicionario de quantidade_nomes e adicionar as keys na lista de assets
        lista_assets = list(quantidade_nomes.keys())

        # 4.3) Para cada Ativo, atualizar ou inserir
        for asset in lista_assets:
            # -----------------------------
            # DETERMINAR as colunas do dia
            # -----------------------------
            col_PL = f"{dia_operacao} - PL"
            col_PFech = f"{dia_operacao} - Preco_Fechamento"
            col_PComp = f"{dia_operacao} - Preco_Compra"
            col_Qtd = f"{dia_operacao} - Quantidade"
            col_Rend = f"{dia_operacao} - Rendimento"

            # Se o ativo não existe no df_fundo, criamos a linha básica
            if asset not in df_fundo.index:
                df_fundo.loc[asset, "Ativo"] = asset

                # PL
                if fundo == "Total":
                    soma_pl = sum(pl_dias_vetor)
                    df_fundo.loc[asset, col_PL] = soma_pl
                else:
                    df_fundo.loc[asset,
                                 col_PL] = pl_dias.loc[fundo, dia_operacao]
                    pl_dias_vetor.append(pl_dias.loc[fundo, dia_operacao])

                # Preço de Fechamento Atual + Dia
                preco_fechamento_atual = df_fechamento_b3.loc[
                    df_fechamento_b3["Assets"] == asset, ultimo_fechamento
                ].values[0]
                preco_fechamento_atual = pd.to_numeric(
                    preco_fechamento_atual, errors='coerce')
                df_fundo.loc[asset,
                             "Preco_Fechamento_Atual"] = preco_fechamento_atual

                preco_fechamento_dia = df_fechamento_b3.loc[
                    df_fechamento_b3["Assets"] == asset, dia_operacao
                ].values[0]
                preco_fechamento_dia = pd.to_numeric(
                    preco_fechamento_dia, errors='coerce')
                df_fundo.loc[asset, col_PFech] = preco_fechamento_dia

                # Preço de Compra
                preco_compra = df_info.loc[
                    (df_info["Ativo"] == asset) & (
                        df_info["Dia de Compra"] == dia_operacao),
                    "Preço de Compra"
                ].values[0]
                preco_compra = pd.to_numeric(preco_compra, errors='coerce')
                df_fundo.loc[asset, col_PComp] = preco_compra

                # Quantidade
                quantidade = row_fundo[f'Contratos {asset}']
                quantidade = pd.to_numeric(quantidade, errors='coerce')
                df_fundo.loc[asset, col_Qtd] = quantidade

                # Rendimento
                if asset == 'TREASURY':
                    rendimento = (preco_fechamento_dia -
                                  preco_compra) * dolar / 10000
                elif 'DAP' in asset:
                    dia_compra = pd.to_datetime(dia_operacao)
                    # Pega linha do ativo no df_ajuste
                    linha_ajuste = df_ajuste[df_ajuste['Assets'] == asset].drop(
                        columns='Assets')

                    if linha_ajuste.empty:
                        rendimento = 0  # Ativo não encontrado

                    # Selecionar colunas com datas >= data de compra
                    colunas_uteis = linha_ajuste.columns[datas_validas > dia_compra]

                    # Soma os ajustes após a data de compra e multiplica pela quantidade
                    rendimento = linha_ajuste[colunas_uteis].sum(
                        axis=1).values[0] * quantidade
                    
                elif 'DI' in asset:
                    dia_compra = pd.to_datetime(dia_operacao)

                    # linha do ativo no df_ajuste
                    linha_ajuste = df_ajuste[df_ajuste['Assets'] == asset].drop(columns='Assets')

                    # 1) PnL do DIA DA COMPRA (D0)
                    pnl_d0 = (preco_fechamento_dia - preco_compra)

                    if linha_ajuste.empty:
                        # Sem ajustes disponíveis → fica só o D0
                        rendimento = pnl_d0
                    else:
                        # 2) Ajustes a partir do DIA SEGUINTE (D+1)
                        colunas_uteis = linha_ajuste.columns[datas_validas > dia_compra]
                        soma_ajustes = linha_ajuste[colunas_uteis].sum(axis=1).values[0] if len(colunas_uteis) else 0
                        rendimento = pnl_d0 + soma_ajustes

                else:
                    rendimento = (preco_fechamento_dia - preco_compra)
                df_fundo.loc[asset, col_Rend] = quantidade * rendimento

            else:
                # Se o ativo já existe, precisamos mesclar valores antigos e novos

                # 1) PL (continua igual ao código original)
                if fundo == "Total":
                    df_fundo.loc[asset,
                                 col_PL] = pl_dias.loc['TOTAL', dia_operacao]
                else:
                    df_fundo.loc[asset,
                                 col_PL] = pl_dias.loc[fundo, dia_operacao]

                # 2) Preço Fechamento Atual + dia
                preco_fechamento_atual = df_fechamento_b3.loc[
                    df_fechamento_b3["Assets"] == asset, ultimo_fechamento
                ].values[0]
                preco_fechamento_atual = pd.to_numeric(
                    preco_fechamento_atual, errors='coerce')
                df_fundo.loc[asset,
                             "Preco_Fechamento_Atual"] = preco_fechamento_atual

                preco_fechamento_dia = df_fechamento_b3.loc[
                    df_fechamento_b3["Assets"] == asset, dia_operacao
                ].values[0]
                preco_fechamento_dia = pd.to_numeric(
                    preco_fechamento_dia, errors='coerce')
                # Sobrescreve com o novo
                df_fundo.loc[asset, col_PFech] = preco_fechamento_dia

                # 3) Pegar dados antigos
                old_qty = df_fundo.loc[asset,
                                       col_Qtd] if col_Qtd in df_fundo.columns else np.nan
                if pd.isna(old_qty):
                    old_qty = 0.0

                old_compra = df_fundo.loc[asset,
                                          col_PComp] if col_PComp in df_fundo.columns else np.nan
                if pd.isna(old_compra):
                    old_compra = 0.0

                old_rend = df_fundo.loc[asset,
                                        col_Rend] if col_Rend in df_fundo.columns else np.nan
                if pd.isna(old_rend):
                    old_rend = 0.0

                # 4) Dados novos
                preco_compra_new = df_info.loc[
                    (df_info["Ativo"] == asset) & (
                        df_info["Dia de Compra"] == dia_operacao),
                    "Preço de Compra"
                ].values[0]
                preco_compra_new = pd.to_numeric(
                    preco_compra_new, errors='coerce')

                quantidade_new = row_fundo[f'Contratos {asset}']
                quantidade_new = pd.to_numeric(quantidade_new, errors='coerce')
                if pd.isna(quantidade_new):
                    quantidade_new = 0.0

                # 5) Soma de Quantidades
                qty_total = old_qty + quantidade_new
                df_fundo.loc[asset, col_Qtd] = qty_total

                # 6) Média ponderada do Preço de Compra
                old_total = old_qty * old_compra
                new_total = quantidade_new * preco_compra_new
                if qty_total > 0:
                    preco_compra_final = (old_total + new_total) / qty_total
                else:
                    preco_compra_final = preco_compra_new
                df_fundo.loc[asset, col_PComp] = preco_compra_final

                # 7) Rendimento
                if asset == 'TREASURY':
                    rendimento_unit = (
                        preco_fechamento_dia - preco_compra_new) * dolar / 10000

                elif 'DAP' in asset:
                    dia_compra = pd.to_datetime(dia_operacao)
                    # Pega linha do ativo no df_ajuste
                    linha_ajuste = df_ajuste[df_ajuste['Assets'] == asset].drop(
                        columns='Assets')

                    if linha_ajuste.empty:
                        rendimento_unit = 0  # Ativo não encontrado

                    # Selecionar colunas com datas >= data de compra
                    colunas_uteis = linha_ajuste.columns[datas_validas >= dia_compra]

                    # Soma os ajustes após a data de compra e multiplica pela quantidade
                    rendimento_unit = linha_ajuste[colunas_uteis].sum(
                        axis=1).values[0] * quantidade_new
                    
                elif 'DI' in asset:
                    dia_compra = pd.to_datetime(dia_operacao)
                    # Pega linha do ativo no df_ajuste
                    linha_ajuste = df_ajuste[df_ajuste['Assets'] == asset].drop(columns='Assets')

                    # 1) PnL do DIA DA COMPRA (D0) — por unidade
                    pnl_d0_unit = (preco_fechamento_dia - preco_compra_new)

                    if linha_ajuste.empty:
                        # Sem ajustes disponíveis → fica só o D0
                        rendimento_unit = quantidade_new * pnl_d0_unit
                    else:
                        # 2) Ajustes a partir do DIA SEGUINTE (D+1)
                        colunas_uteis = linha_ajuste.columns[datas_validas > dia_compra]  # estritamente > D0
                        soma_ajustes_unit = linha_ajuste[colunas_uteis].sum(axis=1).values[0] if len(colunas_uteis) else 0
                        # total = D0 + ajustes (já multiplicando pela quantidade_new conforme seu padrão aqui)
                        rendimento_unit = quantidade_new * (pnl_d0_unit + soma_ajustes_unit)

                else:
                    rendimento_unit = (preco_fechamento_dia - preco_compra_new)

                rendimento_new = quantidade_new * rendimento_unit
                df_fundo.loc[asset, col_Rend] = old_rend + rendimento_new

        # 4.4) Salvar ao final, como no fluxo original##
        df_fundo.reset_index(drop=True, inplace=True)
        df_fundo.to_parquet(nome_arquivo_parquet, index=False)
        table_name = nome_arquivo_parquet.replace(".parquet", "")
        table_name = table_name.replace("BaseFundos/", "")

        mensagens.append(f"✅ {table_name} carregado!")
        status_container.markdown(" | ".join(mensagens))
        add_data_2(df_fundo, table_name)  # permanece a mesma chamada
        print(f"[{fundo}] -> parquet atualizado: {nome_arquivo_parquet}")


def analisar_performance_fundos(
    data_inicial,
    data_final,
    lista_estrategias,
    lista_ativos
):
    """
    Lê todos os arquivos parquet na pasta 'BaseFundos', extrai colunas diárias
    de Rendimento (ex: "YYYY-MM-DD - Rendimento"), monta um DataFrame 'long'
    com colunas [date, fundo, Ativo, Rendimento_diario], faz mapeamento
    de Estratégia e filtra o intervalo [data_inicial, data_final].

    Retorna um dicionário contendo DataFrames de performance diária:
      - df_diario_fundo_ativo:
          colunas = [date, fundo, Ativo, Rendimento_diario, Estratégia]
      - df_diario_fundo_estrategia:
          colunas = [date, fundo, Estratégia, Rendimento_diario]
      - df_diario_fundo_total:
          colunas = [date, fundo, Rendimento_diario]
    """
    dt_ini = datetime.datetime.strptime(data_inicial, "%Y-%m-%d").date()
    dt_fim = datetime.datetime.strptime(data_final,   "%Y-%m-%d").date()

    def mapear_estrategia(ativo):
        """Mapeia o ativo para a estratégia, usando substring."""
        for prefixo, nome_estrategia in lista_estrategias.items():
            if prefixo in ativo:  # substring "DI", "DAP", "WDO1", etc.
                return nome_estrategia
        return "Estratégia Desconhecida"

    pasta_fundos = "BaseFundos"
    if not os.path.isdir(pasta_fundos):
        raise FileNotFoundError(f"Pasta '{pasta_fundos}' não encontrada.")

    arquivos = [arq for arq in os.listdir(
        pasta_fundos) if arq.endswith(".parquet")]

    registros = []
    # Cada elemento em 'registros' será um dict:
    # { "date": dt_col, "fundo": nome_fundo, "Ativo": ativo, "Rendimento_diario": rend_val }

    for arquivo_parquet in arquivos:
        caminho_parquet = os.path.join(pasta_fundos, arquivo_parquet)
        nome_fundo = arquivo_parquet.replace(".parquet", "")
        df_fundo = pd.read_parquet(caminho_parquet)
        if "Ativo" not in df_fundo.columns:
            continue  # ignora se não tiver coluna 'Ativo'

        # Todas as colunas do tipo "YYYY-MM-DD - Rendimento"
        colunas_rendimento = [
            c for c in df_fundo.columns if c.endswith(" - Rendimento")]

        for _, row_ in df_fundo.iterrows():
            ativo = row_["Ativo"]
            # Se não quiser considerar "PL" em hipótese alguma, você pode ignorar aqui.
            # (Também podemos filtrar depois, no Streamlit.)

            for col_rend in colunas_rendimento:
                try:
                    data_str = col_rend.replace(" - Rendimento", "").strip()
                    dt_col = datetime.datetime.strptime(
                        data_str, "%Y-%m-%d").date()
                except:
                    continue  # erro de formato de data

                if dt_ini <= dt_col <= dt_fim:
                    rend_val = row_[col_rend]
                    if pd.isna(rend_val):
                        rend_val = 0.0
                    registros.append({
                        "date": dt_col,
                        "fundo": nome_fundo,
                        "Ativo": ativo,
                        "Rendimento_diario": rend_val
                    })

    if len(registros) == 0:
        # Nada encontrado no período
        df_diario_fundo_ativo = pd.DataFrame(
            columns=["date", "fundo", "Ativo", "Rendimento_diario", "Estratégia"])
        df_diario_fundo_estrategia = pd.DataFrame(
            columns=["date", "fundo", "Estratégia", "Rendimento_diario"])
        df_diario_fundo_total = pd.DataFrame(
            columns=["date", "fundo", "Rendimento_diario"])
        return {
            "df_diario_fundo_ativo": df_diario_fundo_ativo,
            "df_diario_fundo_estrategia": df_diario_fundo_estrategia,
            "df_diario_fundo_total": df_diario_fundo_total
        }

    # Monta DataFrame consolidado
    df_all = pd.DataFrame(registros)
    df_all["Estratégia"] = df_all["Ativo"].apply(mapear_estrategia)

    # 1) Performance diária por Fundo x Ativo
    df_diario_fundo_ativo = df_all[[
        "date", "fundo", "Ativo", "Rendimento_diario", "Estratégia"]].copy()
    df_diario_fundo_ativo.sort_values(["fundo", "date", "Ativo"], inplace=True)

    # 2) Performance diária por Fundo x Estratégia
    df_diario_fundo_estrategia = (
        df_all
        .groupby(["date", "fundo", "Estratégia"], as_index=False)["Rendimento_diario"]
        .sum()
        .sort_values(["fundo", "date", "Estratégia"])
    )

    # 3) Performance diária agregada do Fundo (somando todos os ativos)
    df_diario_fundo_total = (
        df_all
        .groupby(["date", "fundo"], as_index=False)["Rendimento_diario"]
        .sum()
        .sort_values(["fundo", "date"])
    )

    return {
        "df_diario_fundo_ativo": df_diario_fundo_ativo,
        "df_diario_fundo_estrategia": df_diario_fundo_estrategia,
        "df_diario_fundo_total": df_diario_fundo_total
    }


def atualizar_base_fundos():
    pasta_fundos = "BaseFundos"
    if not os.path.isdir(pasta_fundos):
        raise FileNotFoundError(f"Pasta '{pasta_fundos}' não encontrada.")

    arquivos = [arq for arq in os.listdir(
        pasta_fundos) if arq.endswith(".parquet")]

    registros = []
    # Cada elemento em 'registros' será um dict:
    # { "date": dt_col, "fundo": nome_fundo, "Ativo": ativo, "Rendimento_diario": rend_val }

    for arquivo_parquet in arquivos:
        # Tirar parquet do nome
        arquivo_nome = arquivo_parquet.replace(".parquet", "")
        df_salvo = pd.read_parquet(f"{pasta_fundos}/{arquivo_parquet}")
        df_novo = load_data_base(arquivo_nome)
        if not df_salvo.equals(df_novo):
            df_salvo = df_novo.copy()
            df_salvo.to_parquet(
                f"{pasta_fundos}/{arquivo_parquet}", index=False)

def fmt_money(series):  return series.apply(lambda x: f"R${x:,.2f}")
def fmt_int(series):    return series.apply(lambda x: f"{x:.0f}")

def add_data_2(df, table_name):
    # Converter NaN para None e remover colunas de datas completamente nulas
    df = df.replace({np.nan: None})

    # Identificar colunas de data (formato "YYYY-MM-DD - Métrica")
    date_columns = {}
    for col in df.columns:
        if " - " in col:
            base_date, metric = col.split(" - ", 1)  # Divide no primeiro " - "
            if base_date not in date_columns:
                date_columns[base_date] = []
            date_columns[base_date].append(col)

    # Verificar quais grupos de data têm todas as colunas nulas
    columns_to_drop = []
    for base_date, cols in date_columns.items():
        if df[cols].isnull().all().all():  # Verifica se todas as colunas estão totalmente nulas
            columns_to_drop.extend(cols)

    # Remover colunas do DataFrame
    df = df.drop(columns=columns_to_drop)
    data = df.to_dict(orient='records')

    try:
        # Conexão com o PostgreSQL
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres.obgwfekirteetqzjydry",
            password="hF81aEQwSDNzATLJ",
            host="aws-0-sa-east-1.pooler.supabase.com",
            port="6543"
        )
        cursor = conn.cursor()

        # --- PASSO 1: GERENCIAMENTO DE COLUNAS ---
        # Obter colunas existentes
        cursor.execute(f"""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = '{table_name}';
        """)
        existing_columns = {row[0] for row in cursor.fetchall()}

        # Remover colunas obsoletas do banco (se existirem)
        for col in columns_to_drop:
            if col in existing_columns:
                cursor.execute(
                    f'ALTER TABLE "{table_name}" DROP COLUMN IF EXISTS "{col}"')

        # Atualizar lista de colunas após remoção
        cursor.execute(f"""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = '{table_name}';
        """)
        existing_columns = {row[0] for row in cursor.fetchall()}

        # Adicionar novas colunas ausentes
        new_columns = [
            col for col in df.columns if col not in existing_columns]
        for col in new_columns:
            cursor.execute(
                f'ALTER TABLE "{table_name}" ADD COLUMN IF NOT EXISTS "{col}" FLOAT;')

        # --- PASSO 2: DELETAR REGISTROS OBSOLETOS ---
        current_ativos = [str(record["Ativo"])
                          for record in data if "Ativo" in record]

        cursor.execute(sql.SQL("SELECT DISTINCT \"Ativo\" FROM {}").format(
            sql.Identifier(table_name)
        ))
        existing_ativos = [row[0] for row in cursor.fetchall()]

        ativos_para_excluir = list(set(existing_ativos) - set(current_ativos))

        if ativos_para_excluir:
            delete_query = sql.SQL("DELETE FROM {} WHERE \"Ativo\" = ANY(%s)").format(
                sql.Identifier(table_name)
            )
            cursor.execute(delete_query, (ativos_para_excluir,))

        # --- PASSO 3: UPSERT DOS NOVOS REGISTROS ---
        for record in data:
            columns = list(record.keys())
            values = list(record.values())

            update_set = sql.SQL(', ').join([
                sql.SQL("{} = EXCLUDED.{}").format(
                    sql.Identifier(col),
                    sql.Identifier(col)
                ) for col in columns if col != "Ativo"
            ])

            insert_query = sql.SQL("""
                INSERT INTO {table} ({cols})
                VALUES ({vals})
                ON CONFLICT ("Ativo") DO UPDATE
                SET {updates}
            """).format(
                table=sql.Identifier(table_name),
                cols=sql.SQL(', ').join(map(sql.Identifier, columns)),
                vals=sql.SQL(', ').join(map(sql.Literal, values)),
                updates=update_set
            )

            cursor.execute(insert_query)

        conn.commit()
        cursor.close()
        conn.close()

        return {
            "status": "success",
            "rows_upserted": len(data),
            "rows_deleted": len(ativos_para_excluir),
            "columns_dropped": columns_to_drop
        }

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        print(f"Erro detalhado: {e}")
        return None


def load_data_base(table_name):
    try:
        response = supabase.table(table_name).select("*").execute()
        data = response.data
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        return []


def apagar_dados_data(data_apag):
    """
    Apaga os dados de um dia específico de todos os arquivos parquet na pasta 'BaseFundos'.
    """

    pasta_fundos = "BaseFundos"

    if not os.path.isdir(pasta_fundos):
        raise FileNotFoundError(f"Pasta '{pasta_fundos}' não encontrada.")

    arquivos = [arq for arq in os.listdir(
        pasta_fundos) if arq.endswith(".parquet")]

    for arquivo_parquet in arquivos:
        caminho_parquet = os.path.join(pasta_fundos, arquivo_parquet)
        nome_fundo = arquivo_parquet.replace(".parquet", "")

        df_fundo = pd.read_parquet(caminho_parquet)
        # Todas as colunas do tipo "YYYY-MM-DD"

        colunas_selecionadas = [
            c for c in df_fundo.columns if c.startswith(f"{data_apag}")]

        df_fundo.drop(columns=colunas_selecionadas, inplace=True)

        # Verificar se as linhas estão vazias
        df_teste = df_fundo.drop(
            columns=['Ativo', 'Preco_Fechamento_Atual'], axis=1)
        for index, row in df_teste.iterrows():
            if row.isnull().all():
                df_fundo = df_fundo.drop(index)
                df_teste = df_teste.drop(index)

        if df_teste.empty:
            # Excluir o arquivo
            os.remove(caminho_parquet)
        else:
            df_fundo.to_parquet(caminho_parquet, index=False, encoding="utf-8")

        print(f"[{nome_fundo}] -> parquet atualizado: {caminho_parquet}")

    # Atualizar portifólio_posições.parquet
    nome_arquivo_portifolio = 'Dados/portifolio_posições.parquet'
    df_portifolio = pd.read_parquet(nome_arquivo_portifolio, index_col=0)
    # Dropar todas as linhas que a coluna "Dia de Compra" SEJA igual a data_apag
    df_portifolio = df_portifolio[df_portifolio['Dia de Compra'] != data_apag]
    df_portifolio.to_parquet(nome_arquivo_portifolio)
    print(
        f"[{nome_arquivo_portifolio}] -> parquet atualizado: {nome_arquivo_portifolio}")

def analisar_dados_fundos(
        soma_pl_sem_pesos: float,
        df_b3_fechamento: pd.DataFrame | None = None,
        df_ajuste:        pd.DataFrame | None = None,
        basefundos:       dict[str, pd.DataFrame] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # ------------------------------------------------------------------ 1. entradas
    if df_b3_fechamento is None:
        df_b3_fechamento = load_b3_prices()          # ← já cacheado
    if df_ajuste is None:
        df_ajuste = load_ajustes()                  # ← já cacheado
    if basefundos is None:
        basefundos = load_basefundos()              # ← já cacheado

    preco_lookup = df_b3_fechamento.set_index("Assets")
    dia_atual    = df_b3_fechamento.columns[-1]
    dolar        = preco_lookup.loc["WDO1", dia_atual]

    # 👉  prepara df_ajuste só UMA vez
    df_ajuste = df_ajuste.copy()
    df_ajuste.columns = (
        ["Assets"]
        + [pd.to_datetime(c, errors="coerce").strftime("%Y-%m-%d")
           for c in df_ajuste.columns[1:]]
    )
    df_ajuste.set_index("Assets", inplace=True)           # lookup rápido

    # ------------------------------------------------------------------ 2. loop único
    df_cash, df_bps = {}, {}                              # acumuladores

    for file_name, df_fundos in basefundos.items():       # sem re-ler parquet
        fundo = file_name                                 # já sem extensão
        # --- garante que o índice seja 'Ativo' sem provocar KeyError -------------
        if 'Ativo' in df_fundos.columns:                 # caso 1: a coluna existe
            df_fundos = df_fundos.set_index('Ativo')
        else:                                            # caso 2: já está no índice
            if df_fundos.index.name is None:             # ainda sem nome → nomeia
                df_fundos.index.name = 'Ativo'
            # nada a fazer se o índice já se chama 'Ativo'

        # ---------- cada ativo / cada operação de compra ----------------
        for ativo, row in df_fundos.iterrows():
            col_qtd = [c for c in df_fundos.columns if c.endswith("Quantidade")]

            for c in col_qtd:
                qtd = row[c]
                if not qtd or pd.isna(qtd):
                    continue

                data_op   = c.split()[0]                  # “YYYY-MM-DD”
                p_compra  = row[c.replace("Quantidade", "Preco_Compra")]
                pl_op     = row[c.replace("Quantidade", "PL")]

                p_ant = p_compra

                # percorre TODAS as datas de fechamento já em ordem
                for data_fech in df_b3_fechamento.columns[1:]:
                    if data_fech < data_op:
                        continue

                    # ---------- preço de mercado do ativo
                    try:
                        p_fech = preco_lookup.at[ativo, data_fech]
                    except KeyError:                       # ativo não existe no preço
                        break

                    # ---------- rendimento em R$
                    if ativo == "TREASURY":
                        rend_cash = (p_fech - p_ant) * qtd * dolar / 10_000

                    elif "DAP" in ativo:
                        # ──────────────────────────────────────────────
                        # regra antiga:  1) na data da compra ⇒ 0
                        #                2) depois da compra ⇒ ajuste do próprio dia * quantidade
                        # ──────────────────────────────────────────────
                        if data_fech == data_op:           # mesma data da compra?  ⇒  rendimento zero
                            rend_cash = 0
                        else:
                            try:
                                ajuste_dia = df_ajuste.at[ativo, data_fech]   # ajuste do PRÓPRIO dia
                                rend_cash  = ajuste_dia * qtd
                            except KeyError:                                  # sem ajuste disponível
                                rend_cash = 0
                    
                    
                    elif "DI" in ativo:
                        # ──────────────────────────────────────────────
                        # regra antiga:  1) na data da compra ⇒ 0
                        #                2) depois da compra ⇒ ajuste do próprio dia * quantidade
                        # ──────────────────────────────────────────────
                        if data_fech == data_op:           # mesma data da compra?  ⇒  rendimento zero
                            rend_cash = (p_fech - p_ant) * qtd
                        else:
                            try:
                                ajuste_dia = df_ajuste.at[ativo, data_fech]   # ajuste do PRÓPRIO dia
                                rend_cash  = ajuste_dia * qtd
                            except KeyError:                                  # sem ajuste disponível
                                rend_cash = 0

                    else:                                  # DI, NTNB, etc.
                        rend_cash = (p_fech - p_ant) * qtd

                    # ---------- guarda acumulando
                    chave = f"{ativo} - {fundo} - P&L"
                    df_cash.setdefault(chave, pd.Series()).at[data_fech] = \
                        df_cash.get(chave, pd.Series()).get(data_fech, 0) + rend_cash
                    df_bps .setdefault(chave, pd.Series()).at[data_fech] = \
                        df_bps .get(chave, pd.Series()).get(data_fech, 0) + rend_cash / pl_op * 10_000

                    p_ant = p_fech

    # ------------------------------------------------------------------ 3. saída
    df_final    = pd.DataFrame(df_cash).T.fillna(0)
    df_final_pl = pd.DataFrame(df_bps ).T.fillna(0)
    # Se o P&L do dia é 0 → o bps TEM de ser 0 também
    ZERO_EPS = 1e-9                   # tolerância numérica
    df_final_pl = df_final_pl.mask(df_final.abs() < ZERO_EPS, 0)
    TOL_QTD = 1e-6     # ou use 0 se quantidade é inteira
    TOL_PL  = 0.01     # < 50 centavos tratamos como zero PL

    # -- 1) se você tem DataFrame de quantidades
    # df_final_pl = df_final_pl.where(df_quant.abs() > TOL_QTD, 0)

    # -- 2) se não tem quantidades, use o próprio PL
    df_final_pl = df_final_pl.where(df_final_pl.abs() >= TOL_PL, 0)

    df_final["Total"]    = df_final.sum(axis=1)
    df_final_pl["Total"] = df_final_pl.sum(axis=1)

    return df_final, df_final_pl


def pl_dia(df, tipo_agrupamento="Semanal"):
    # 1) Ler e limpar pl_dias
    pl_dias = pd.read_parquet("Dados/pl_fundos_teste.parquet")
    # Se a planilha estiver em outro formato, ajuste `sep`, `decimal`, etc.
    pl_dias.set_index('Fundos/Carteiras Adm', inplace=True)

    # Remove possíveis caracteres de R$, converte para float
    for col in pl_dias.columns:
        pl_dias[col] = pl_dias[col].astype(str)
        pl_dias[col] = pl_dias[col].str.replace('R$', '', regex=True)\
                                   .str.replace('--',  '', regex=True)\
                                   .str.strip()
    pl_dias.replace('', np.nan, inplace=True)
    # Converter para float (ignorando colunas não-numéricas)
    # Identificar a coluna que é nome do fundo (ex.: 'Fundos/Carteiras Adm')
    for c in pl_dias.columns:
        pl_dias[c] = pd.to_numeric(pl_dias[c], errors='ignore')

    # 2) Ordenar colunas por data e aplicar forward-fill
    #    Supondo que as colunas sejam strings como '2025-01-01' etc.
    #    Convertemos para datetime para ordenar corretamente:
    novas_datas = pd.to_datetime(
        pl_dias.columns, format='%Y-%m-%d', errors='coerce')
    # Nem sempre todas são datas válidas. Vamos separar o que for data.
    mapping_datas = {}
    datas_validas = []
    for old_col, dt_col in zip(pl_dias.columns, novas_datas):
        if pd.notnull(dt_col):
            mapping_datas[old_col] = dt_col
            datas_validas.append(dt_col)
        else:
            # se não for data, deixamos como está
            mapping_datas[old_col] = old_col

    pl_dias.rename(columns=mapping_datas, inplace=True)

    # Agora separamos apenas as colunas que são datas para ordenar:
    colunas_data = [c for c in pl_dias.columns if isinstance(c, pd.Timestamp)]
    colunas_nao_data = [c for c in pl_dias.columns if c not in colunas_data]

    pl_dias_data = pl_dias[colunas_data].copy()
    pl_dias_data = pl_dias_data.reindex(sorted(pl_dias_data.columns), axis=1)
    # Aplica ffill para preencher valores ausentes ou zero
    # Primeiro substituímos zeros por NaN, caso queira tratá-los igual a "valor ausente"
    pl_dias_data.replace(0, np.nan, inplace=True)
    pl_dias_data = pl_dias_data.ffill(axis=1)

    # Junta de volta as colunas não-data com as colunas de datas preenchidas
    pl_dias = pd.concat([pl_dias[colunas_nao_data], pl_dias_data], axis=1)

    # 3) Extrair nome do fundo do índice de df
    #    Se o índice estiver assim: "DAP35 - BH FIRF INFRA - P&L"
    #    vamos pegar só a parte do meio (fundo).
    def extrair_fundo(nome_index):
        # exemplo de split: "DAP35 - BH FIRF INFRA - P&L" => ["DAP35", "BH FIRF INFRA", "P&L"]
        partes = nome_index.split(" - ")
        if len(partes) == 3:
            return partes[1].strip()
        elif len(partes) == 2:
            # Se vier "Ativo - Fundo" ou "Fundo - P&L"
            return partes[0]  # ou partes[1], dependendo do seu caso real
        else:
            return nome_index  # fallback

    df_copy = df.copy()
    colunas_df_data = pd.to_datetime(
        df_copy.columns, format='%Y-%m-%d', errors='coerce')
    # Também precisamos garantir que as colunas de df sejam datas
    df_copy.columns = colunas_df_data
    df_copy["Fundo"] = df_copy.index.to_series().apply(extrair_fundo)
    # 4) Finalmente, dividir cada célula de df pelo PL do fundo, na data equivalente.
    #    Podemos fazer um loop sobre as colunas, ou um apply row-a-row.
    #    Para maior eficiência, normalmente pivotar e usar .div() seria melhor.
    #    Mas como você já tem esse layout, dá pra fazer via apply:

    def dividir_por_pl(row):
        fundo = row["Fundo"]
        # Para cada coluna de data, substituir pelo valor / PL
        saida = {}
        for c in row.index:
            if c == "Fundo":
                saida[c] = fundo
            else:
                # c é uma data (Timestamp)
                if fundo in pl_dias.index and c in pl_dias.columns:
                    pl_val = pl_dias.loc[fundo, c]
                    valor = row[c]
                    if pd.notnull(valor) and pd.notnull(pl_val) and pl_val != 0:
                        pl_val = pl_val.replace('.', '')
                        pl_val = pl_val.replace('R$', '')
                        pl_val = pl_val.replace(',', '.')
                        pl_val = float(pl_val)
                        # Ver o tipo de valor
                        saida[c] = valor / pl_val * 10000
                        # Aplicar estilo
                        saida[c] = "{:.2f}bps".format(saida[c])
                    else:
                        saida[c] = np.nan
                else:
                    saida[c] = np.nan
        return pd.Series(saida)

    df_result = df_copy.apply(dividir_por_pl, axis=1)

    return df_result


def calcular_retorno_sobre_pl(df_fundos, df2, pl_parquet_path="Dados/pl_fundos_teste.parquet"):
    """
    df_fundos: DataFrame que contém os fundos que você quer usar no cálculo.
               Pode ter pelo menos a coluna 'Fundo' ou então o index com o nome do fundo.
    df2:       DataFrame com as colunas ['date', 'estratégia', 'Rendimento_diario'].
               A coluna 'date' está em formato do tipo "19 Jan 2025".
    pl_parquet_path: caminho do parquet de PL (como no seu exemplo).

    Retorna: df2 com uma nova coluna 'Retorno_sobre_PL' = Rendimento_diario / soma_do_PL.
            Onde soma_do_PL é a soma do PL de todos os fundos do df_fundos, na data da linha.
    """

    # ------------------------------------------------------------------------------
    # 1) Ler e limpar pl_dias
    # ------------------------------------------------------------------------------
    pl_dias = pd.read_parquet(pl_parquet_path)

    # Se no arquivo parquet existir a coluna 'Fundos/Carteiras Adm' com o nome do fundo:
    if 'Fundos/Carteiras Adm' in pl_dias.columns:
        pl_dias.set_index('Fundos/Carteiras Adm', inplace=True)

    # Remover possíveis caracteres de R$, etc., e converter para float
    for col in pl_dias.columns:
        pl_dias[col] = (pl_dias[col].astype(str)
                                    .str.replace('R\$', '', regex=True)
                                    .str.replace('--', '', regex=True)
                                    .str.strip())
    pl_dias.replace('', np.nan, inplace=True)

    # Converte para numérico (onde for possível):
    for c in pl_dias.columns:
        pl_dias[c] = pd.to_numeric(pl_dias[c], errors='ignore')

    # ------------------------------------------------------------------------------
    # 2) Converter as colunas de pl_dias para datetime e aplicar forward-fill
    # ------------------------------------------------------------------------------
    colunas_orig = pl_dias.columns
    datas_convertidas = pd.to_datetime(
        colunas_orig, format='%Y-%m-%d', errors='coerce')

    mapping = {}
    col_datas = []
    for col_original, dt in zip(colunas_orig, datas_convertidas):
        if pd.notnull(dt):
            mapping[col_original] = dt
            col_datas.append(dt)
        else:
            # Se não conseguir converter em data, mantemos o nome original
            mapping[col_original] = col_original

    pl_dias.rename(columns=mapping, inplace=True)

    # Separa colunas que são datas daquelas que não são
    colunas_data = [c for c in pl_dias.columns if isinstance(c, pd.Timestamp)]
    colunas_nao_data = [c for c in pl_dias.columns if c not in colunas_data]

    # Concentra as colunas de data em um novo DF para ordenar e fazer ffill
    pl_dias_data = pl_dias[colunas_data].copy()
    pl_dias_data = pl_dias_data.reindex(sorted(pl_dias_data.columns), axis=1)

    # Substituir zeros por NaN para tratar como "valor ausente"
    pl_dias_data.replace(0, np.nan, inplace=True)

    # Forward fill para a direita (preenche valores ausentes com o último valor conhecido)
    pl_dias_data = pl_dias_data.ffill(axis=1)

    # Junta novamente com as colunas não-data
    pl_dias_limpo = pd.concat(
        [pl_dias[colunas_nao_data], pl_dias_data], axis=1)

    # ------------------------------------------------------------------------------
    # 3) Obter a lista de fundos do primeiro DF (df_fundos).
    #    Vamos supor que existe uma coluna 'Fundo' ou que o índice seja o nome do fundo.
    # ------------------------------------------------------------------------------
    if 'Fundo' in df_fundos.columns:
        lista_fundos = df_fundos['Fundo'].unique().tolist()
    else:
        # Se não houver coluna 'Fundo', assumimos que o índice é o nome
        lista_fundos = df_fundos.index.unique().tolist()

    # ------------------------------------------------------------------------------
    # 4) Para cada linha em df2, converter a data e achar o PL total (soma) desses fundos
    # ------------------------------------------------------------------------------
    df2 = df2.copy()
    df2['date_parsed'] = pd.to_datetime(
        df2['date'], format='%d %b %Y', errors='coerce')
    lista_fundos = [f.split('-')[1].split('-')[0] for f in lista_fundos]
    lista_fundos = [f[:-1] if f.endswith(" ") else f for f in lista_fundos]
    # Tirar o primeiro espaço
    lista_fundos = [f[1:] if f.startswith(" ") else f for f in lista_fundos]
    # Função auxiliar para achar a melhor coluna de data (igual ou anterior)

    def achar_coluna_pl(data):
        if data in pl_dias_limpo.columns:
            return data
        # Se não houver a data exata, pegar a maior data anterior
        datas_existentes = sorted(
            c for c in pl_dias_limpo.columns if isinstance(c, pd.Timestamp))
        col_anterior = None
        for d in datas_existentes:
            if d <= data:
                col_anterior = d
            else:
                break
        return col_anterior

    pl_totais = []
    for i, row in df2.iterrows():
        data_evento = row['date_parsed']
        rendimento = row['Rendimento_diario']

        if pd.isnull(data_evento):
            # data inválida
            pl_totais.append(np.nan)
            continue

        col_data_pl = achar_coluna_pl(data_evento)
        if col_data_pl is None:
            pl_totais.append(np.nan)
            continue

        # Soma o PL de todos os fundos listados em df_fundos
        soma_pl = 0.0
        for f in lista_fundos:
            if f in pl_dias_limpo.index:
                pl_fundo = pl_dias_limpo.loc[f, col_data_pl]
                pl_fundo = pl_fundo.replace('.', '')
                pl_fundo = pl_fundo.replace(',', '.')
                pl_fundo = float(pl_fundo)
                if pd.notnull(pl_fundo):
                    soma_pl += pl_fundo

        # Dividir Rendimento_diario pelo PL total
        if soma_pl != 0 and not np.isnan(soma_pl):
            pl_totais.append(rendimento / soma_pl * 10000)
        else:
            pl_totais.append(np.nan)

    # Cria a coluna de resultado
    df2['Retorno_sobre_PL'] = pl_totais
    return df2


def add_custom_css():
    st.markdown(
        """
        <style>

         /* Alterar a cor de todo o texto na barra lateral */
        section[data-testid="stSidebar"] * {
            color: White; /* Cor padrão para textos na barra lateral */
        }

        div[class="stDateInput"] div[class="st-b8"] input {
                color: white;
                }
            div[role="presentation"] div{
            color: white;
            }

        div[data-baseweb="calendar"] button  {
            color:white;
            }
            
        /* Alterar a cor do texto no campo de entrada do st.number_input */
        input[data-testid="stNumberInput-Input"] {
            color: black !important; /* Define a cor do texto como preto */
        }

        input[data-testid="stNumberInputField"] {
            color: black !important; /* Define a cor do texto como preto */
            }

        /* Estiliza os botões de incremento e decremento */
        button[data-testid="stNumberInputStepDown"], 
        button[data-testid="stNumberInputStepUp"] {
            color: black !important; /* Define a cor do ícone ou texto como preto */
            fill: black !important;  /* Caso o ícone SVG precise ser estilizado */
        }

        /* Estiliza o ícone dentro dos botões */
        button[data-testid="stNumberInputStepDown"] svg, 
        button[data-testid="stNumberInputStepUp"] svg {
            fill: black !important;  /* Garante que os ícones sejam pretos */
        }
        

    /* Estiliza o fundo do container do multiselect */
        div[class="st-ak st-al st-bd st-be st-bf st-as st-bg st-bh st-ar st-bi st-bj st-bk st-bl"] {
            background-color: White !important; /* Altera o fundo para cinza */
        }

        /* Estiliza o fundo do input dentro do multiselect */
        div[class="st-al st-bm st-bn st-bo st-bp st-bq st-br st-bs st-bt st-ak st-bu st-bv st-bw st-bx st-by st-bi st-bj st-bz st-bl st-c0 st-c1"] input {
            background-color: White !important; /* Altera o fundo do campo de entrada para cinza */
        }

        /* Estiliza o fundo do botão ou elemento de "Escolher uma opção" */
        div[class="st-cc st-bn st-ar st-cd st-ce st-cf"] {
            background-color: White !important; /* Altera o fundo do botão de opção para cinza */
        }

        /* Estiliza o ícone dentro do botão de decremento */
        button[data-testid="stNumberInput-StepDown"] svg {
            fill: black !important; /* Garante que o ícone seja preto */
            
            div[data-testid="stNumberInput"] input {
            color: black; /* Define o texto como preto */
        }
        
        div[data-testid="stDateInput"] input {
            color: black;
        };
        </style>

        
        """,
        unsafe_allow_html=True,
    )


def get_last_weekday():
    today = datetime.date.today()
    # Se hoje for segunda-feira (0), pega a sexta-feira anterior
    if today.weekday() == 0:
        last_weekday = today - datetime.timedelta(days=3)
    # Se hoje for terça-feira (1), pega a segunda-feira anterior
    elif today.weekday() == 1:
        last_weekday = today - datetime.timedelta(days=1)
    # Se hoje for quarta-feira (2), pega a terça-feira anterior
    elif today.weekday() == 2:
        last_weekday = today - datetime.timedelta(days=1)
    # Se hoje for quinta-feira (3), pega a quarta-feira anterior
    elif today.weekday() == 3:
        last_weekday = today - datetime.timedelta(days=1)
    # Se hoje for sexta-feira (4), pega a quinta-feira anterior
    elif today.weekday() == 4:
        last_weekday = today - datetime.timedelta(days=1)
    # Se hoje for sábado (5), pega a sexta-feira anterior
    elif today.weekday() == 5:
        last_weekday = today - datetime.timedelta(days=1)
    # Se hoje for domingo (6), pega a sexta-feira anterior
    else:
        last_weekday = today - datetime.timedelta(days=2)

    return last_weekday


# --------------------------------------------------------
#            CALLBACKS (para evitar double-click)
# --------------------------------------------------------
def switch_to_page2():
    """Callback para ir para a página 2, usando dados temporários."""
    # Transferir o que guardamos no 'posicoes_temp' e 'ativos_temp' para as chaves finais
    st.session_state["posicoes"] = st.session_state.get("posicoes_temp", {})
    st.session_state["ativos_selecionados"] = st.session_state.get(
        "ativos_temp", [])
    st.session_state["current_page"] = "page2"


def switch_to_main():
    """Callback para voltar para a página principal."""
    st.session_state["current_page"] = "main"


#Ajustes
@st.cache_data(show_spinner=False, ttl=3600)        # 1 h de cache
def load_b3_prices() -> pd.DataFrame:
    df = pd.read_parquet("Dados/df_preco_de_ajuste_atual_completo.parquet")
    df = (df.replace(r'\.', '', regex=True)
            .replace(',', '.', regex=True))
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    df.loc[df['Assets'] == 'TREASURY', df.columns != 'Assets'] *= 1000
    df.loc[df['Assets'] == 'WDO1',     df.columns != 'Assets'] *= 10
    return df

# aproveita a função já existente, mas cacheia a saída
@st.cache_data(show_spinner=False)
def read_atual_contratos_cached():
    return read_atual_contratos()


@st.cache_data(show_spinner=False)
def load_ajustes() -> pd.DataFrame:
    df = pd.read_parquet("Dados/df_valor_ajuste_contrato.parquet")
    if df.iloc[:, -1].equals(df.iloc[:, -2]):       # tira duplicata de “última coluna”
        df = df.iloc[:, :-1]
    qtd_cols = df.columns[1:]
    df[qtd_cols] = (df[qtd_cols]
                    .replace(r'\.', '', regex=True)
                    .replace(',', '.', regex=True)
                    .astype(float))
    return df


@st.cache_data(show_spinner=False)
def load_basefundos() -> dict[str, pd.DataFrame]:
    """Lê todos os arquivos de BaseFundos apenas 1x por sessão."""
    out = {}
    for f in os.listdir("BaseFundos"):
        nome = f.rsplit(".", 1)[0]
        df   = pd.read_parquet(f"BaseFundos/{f}").set_index("Ativo")
        out[nome] = df
    return out


@st.cache_data(show_spinner="Lendo PL…", ttl=3600)      # 1 h de validade
def load_total_pl() -> tuple[float, str]:
    """
    Soma o PL dos fundos autorizados na penúltima coluna-data
    de Dados/pl_fundos_teste.parquet.
    Devolve (pl_total_float, data_ref_yyyy-mm-dd).
    """
    import pandas as pd, numpy as np

    # ---------- 1. lista de fundos que entram na soma ----------
    allow_list = {
        'AF DEB INCENTIVADAS',
        'BH FIRF INFRA',
        'BORDEAUX INFRA',
        'GLOBAL BONDS',
        'HORIZONTE',
        'JERA2026',
        'MANACA INFRA FIRF',
        'REAL FIM',
        'TOPAZIO INFRA',
    }

    # ---------- 2. lê parquet ----------
    df = pd.read_parquet("Dados/pl_fundos_teste.parquet")

    # mantém só os fundos permitidos
    df = df[df["Fundos/Carteiras Adm"].isin(allow_list)].copy()

    # ---------- 3. identifica penúltima coluna-data ----------
    cols_data = [c for c in df.columns
                 if c not in ("Fundos/Carteiras Adm", "Último Valor")]
    cols_data = sorted(cols_data)        # ordem cronológica
    if not cols_data:
        raise ValueError("Parquet não contém colunas-data.")
    data_ref = cols_data[-1]             # penúltima = última válida

    # ---------- 4. converte texto "R$ 12.753.920,35" → float ----------
    df[data_ref] = (df[data_ref].astype(str)
                    .str.replace(r"[R$\s\.]", "", regex=True)  # tira R$, espaço, ponto
                    .str.replace(",", ".", regex=False)        # vírgula → ponto
                    .replace({"": np.nan, "--": np.nan})
                    .astype(float)
                    .fillna(0.0))

    # ---------- 5. soma ----------------
    pl_total = float(df[data_ref].sum())

    return pl_total, data_ref


_ALLOW_FUNDS = {
    'AF DEB INCENTIVADAS', 'BH FIRF INFRA', 'BORDEAUX INFRA',
    'GLOBAL BONDS', 'HORIZONTE', 'JERA2026',
    'MANACA INFRA FIRF', 'REAL FIM', 'TOPAZIO INFRA'
}

TAXA_MAP = {
    "GLOBAL BONDS":            0.0050,
    "HORIZONTE":               0.0100,
    "JERA2026":                0.0028,
    "REAL FIM":                0.0033,
    "BH FIRF INFRA":           0.0020,
    "BORDEAUX INFRA":          0.0005,
    "TOPAZIO INFRA":           0.0054,
    "MANACA INFRA FIRF":       0.0005,
    "AF DEB INCENTIVADAS":     0.0100,
}


@st.cache_data(ttl=3600)
def load_pl_series(return_fee=True, dias_ano=252):
    """
    Lê o parquet e retorna:
      - pl_series:  PL total diário (Series)
      - fee_series: taxa total cobrada no dia (Series), usando capitalização diária:
                    rate_adm_dia = (1 + taxa_anual) ** (1/dias_ano) - 1
    """
    df = pd.read_parquet("Dados/pl_fundos_teste.parquet")
    df = df[df["Fundos/Carteiras Adm"].isin(_ALLOW_FUNDS)].copy()

    cols_data = [c for c in df.columns if c not in ("Fundos/Carteiras Adm", "Último Valor")]

    def _to_float(s):
        return (s.astype(str)
                 .str.replace(r"[R$\s\.]", "", regex=True)
                 .str.replace(",", ".", regex=False)
                 .replace({"": np.nan, "--": np.nan})
                 .astype(float)
                 .fillna(0.0))

    for c in cols_data:
        df[c] = _to_float(df[c])

    # PL por fundo (linhas = fundos, colunas = datas)
    pl_by_fund = df.set_index("Fundos/Carteiras Adm")[cols_data]
    pl_by_fund.columns = pd.to_datetime(pl_by_fund.columns)


    # Série total
    pl_series = pl_by_fund.sum(axis=0).sort_index()
    pl_series.name = "PL_total"

    if not return_fee:
        return pl_series

    # Taxa anual -> diária composta
    taxas_anual = pd.Series(TAXA_MAP, name="taxa_anual").reindex(pl_by_fund.index).fillna(0.0)
    rate_adm_dia = (1 + taxas_anual)**(1/dias_ano) - 1  # <- AQUI

    # Taxa cobrada no dia (R$): PL_dia_fundo * rate_adm_dia_fundo
    pl_by_fund = pl_by_fund * 0.01
    fee_by_fund = pl_by_fund.mul(rate_adm_dia, axis=0)


    # Soma por dia
    fee_series = fee_by_fund.sum(axis=0).sort_index()
    fee_series.name = "taxa_total_dia"

    return pl_series, fee_series


@st.cache_data(ttl=24*3600)          # só renova 1 vez por dia
def load_lft_series() -> pd.Series:
    """
    Lê `Dados/dados_lft.csv` que contém a *cotação* da LFT
        Data, RetornoLFT   (ex.: 113.2481)
    Converte-a em **variação percentual diária** (decimal) e
    devolve uma Series indexada por datetime.
    """
    df = (pd.read_csv("Dados/dados_lft.csv", parse_dates=["Data"])
            .rename(columns={"RetornoLFT": "preco"}))
    df.dropna(inplace=True)  # remove linhas sem data ou preço

    # garante número
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")

    # ordena por data e calcula retorno:  P_t / P_{t-1} − 1
    df = df.sort_values("Data")
    df["lft_ret"] = df["preco"].pct_change().fillna(0.0)

    # limpa infinitos ou NaN residuais (podem aparecer se P_{t-1}=0)
    df["lft_ret"] = (df["lft_ret"]
                       .replace([np.inf, -np.inf], np.nan)
                       .fillna(0.0))

    return df.set_index("Data")["lft_ret"].astype(float)

def analisar_dados_fundos2(
        soma_pl_sem_pesos: float,
        df_b3_fechamento : pd.DataFrame | None = None,
        df_ajuste        : pd.DataFrame | None = None,
        basefundos       : dict[str, pd.DataFrame] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import numpy as np, pandas as pd

    # ───────── 1. entradas
    if df_b3_fechamento is None:
        df_b3_fechamento = load_b3_prices()
    if df_ajuste is None:
        df_ajuste = load_ajustes()
    if basefundos is None:
        basefundos = load_basefundos()

    preco_lookup = df_b3_fechamento.set_index("Assets")          # lookup rápido
    dolar        = preco_lookup.loc["WDO1", df_b3_fechamento.columns[-1]]

    # prepara df_ajuste com cabeçalhos-data
    df_ajuste = df_ajuste.copy()
    df_ajuste.columns = (["Assets"] +
                         [pd.to_datetime(c, errors="coerce").strftime("%Y-%m-%d")
                          for c in df_ajuste.columns[1:]])
    df_ajuste.set_index("Assets", inplace=True)

    pnl_cash, pnl_bps = {}, {}
    despesas_cash : dict[str, pd.Series] = {}        # novo!

    PU_FINAL = 100_000  # já existia
    #Tenho os valores atuais dos ativos em df_b3_fechamento
    #Formula pra calcular a desepesas do DI vai ser:
    # despesas = (PU_final - PU_atual) * 0.03 * qtd * 0.005
    # Onde PU_final é 100000, PU_atual é o valor atual do ativo, qtd é a quantidade de contratos
    #Contratos de DAP e TREASURY são fixos, então não entram na fórmula
    #Contrato de dolar também é fixo, então não entra na fórmula

    DESPESAS_FIXAS = {
        "DAP"     : 3.0,
        "DI"      : 3.0,
        "TREASURY": 10.0,
        "WDO1"    : 1.0,
    }


    # ───────── 2. loop fundos / operações
    for fundo, df_f in basefundos.items():
        if 'Ativo' in df_f.columns:
            df_f = df_f.set_index('Ativo')
        elif df_f.index.name is None:
            df_f.index.name = 'Ativo'

        cols_qtd = [c for c in df_f.columns if c.endswith("Quantidade")]

        for ativo, linha in df_f.iterrows():
            for col_q in cols_qtd:
                qtd = float(linha[col_q])        # garante número
                if pd.isna(qtd) or qtd == 0:
                    continue

                data_op  = col_q.split()[0]
                p_compra = linha[col_q.replace("Quantidade", "Preco_Compra")]
                pl_op    = linha[col_q.replace("Quantidade", "PL")]

                # ------------------- DESPESA FIXA --------------------------
                if fundo.upper() == "TOTAL":
                    continue
                else:
                    # 2) ROOT do ativo
                    raiz = ativo.split("_")[0]
                    if raiz.startswith(("DAP", "DI")):   # normaliza
                        raiz = raiz[:3]                  # “DAP30” → “DAP”,  “DI_27”→“DI”
                        
                    # 3) calcula despesa
                    if raiz == "DI":
                        try:
                            # preço na data da operação (melhor) – se não houver, usa último fechamento
                            PU_atual = preco_lookup.at[ativo, data_op]
                        except KeyError:
                            PU_atual = preco_lookup.at[ativo, df_b3_fechamento.columns[-1]]

                        custo_op = (PU_FINAL - PU_atual) * 0.03 * abs(qtd) * 0.005
                        #st.write(f"Despesa DI: {custo_op:.2f} para {ativo} no dia {data_op}")
                        #st.write(f"PU atual: {PU_atual:.2f}, PU final: {PU_FINAL:.2f}, qtd: {qtd}")
                        #st.write(preco_lookup)
                    elif raiz == "WDO1":
                        try:
                            # preço na data da operação (melhor) – se não houver, usa último fechamento
                            PU_atual = preco_lookup.at[ativo, data_op]
                        except KeyError:
                            PU_atual = preco_lookup.at[ativo, df_b3_fechamento.columns[-1]]
                        #st.write(f"PU atual: {PU_atual:.2f} para {ativo} no dia {data_op}")
                        #custo_op = (PU_atual) * 10 *  0.02 * abs(qtd) * 0.005
                        custo_op = (PU_atual) *  0.02 * abs(qtd) * 0.005
                        #st.write(f"Despesa DI: {custo_op:.2f} para {ativo} no dia {data_op}")
                        #st.write(f"PU atual: {PU_atual:.2f}, PU final: {PU_FINAL:.2f}, qtd: {qtd}")
                        #st.write(preco_lookup)

                    else:
                        # custo fixo por contrato
                        custo_op = DESPESAS_FIXAS.get(raiz, 0.0) * abs(qtd)

                    if custo_op:
                        despesas_cash.setdefault("Despesas", pd.Series()).at[data_op] = \
                            despesas_cash.get("Despesas", pd.Series()).get(data_op, 0.0) + custo_op

                usa_bps = not pd.isna(pl_op) and pl_op != 0
                p_ant   = p_compra

                for data_fech in df_b3_fechamento.columns[1:]:
                    if data_fech < data_op:
                        continue

                    try:
                        p_fech = preco_lookup.at[ativo, data_fech]
                    except KeyError:
                        break       # sem preço, pula resto

                    # regra de rendimento
                    if ativo == "TREASURY":
                        rend = (p_fech - p_ant) * qtd * dolar / 10_000
                    elif "DAP" in ativo:
                        if data_fech == data_op:
                            rend = 0
                        else:
                            ajuste = df_ajuste.get(data_fech, pd.Series()).get(ativo, 0)
                            rend   = ajuste * qtd
                    elif "DI" in ativo:
                        if data_fech == data_op:
                            rend = (p_fech - p_ant) * qtd
                        else:
                            ajuste = df_ajuste.get(data_fech, pd.Series()).get(ativo, 0)
                            rend   = ajuste * qtd
                    else:
                        rend = (p_fech - p_ant) * qtd

                    chave = f"{ativo} - {fundo} - P&L"

                    pnl_cash.setdefault(chave, pd.Series()).at[data_fech] = \
                        pnl_cash.get(chave, pd.Series()).get(data_fech, 0.0) + rend

                    if usa_bps:
                        pnl_bps.setdefault(chave, pd.Series()).at[data_fech] = \
                            pnl_bps.get(chave, pd.Series()).get(data_fech, 0.0) + rend / pl_op * 10_000

                    p_ant = p_fech

    # ───────── 3. saída limpa
    df_final    = pd.DataFrame(pnl_cash).T.fillna(0.0)
    df_final_pl = pd.DataFrame(pnl_bps ).T.fillna(0.0)
    df_despesas = (pd.DataFrame(despesas_cash)
                    .T                       # 1 linha (“Despesas”)
                    .rename_axis("Conta")    # deixa claro
    )
    df_despesas.columns = pd.to_datetime(df_despesas.columns, errors="coerce")
    df_despesas = df_despesas.loc[:, ~df_despesas.columns.isna()].fillna(0.0)

    # zera bps quando P&L ~ 0
    ZERO_EPS = 1e-9
    df_final_pl = df_final_pl.mask(df_final.abs() < ZERO_EPS, 0.0)

    df_final["Total"]    = df_final.sum(axis=1)
    df_final_pl["Total"] = df_final_pl.sum(axis=1)
    #Preciso dropar todos os indices que tem 'Total' no nome
    df_final = df_final[~df_final.index.str.contains("Total")]
    df_final_pl = df_final_pl[~df_final_pl.index.str.contains("Total")]

    return df_final, df_final_pl, df_despesas

# ---------------------------------------------------------------
#  FUNÇÃO AUXILIAR – gráfico Altair da cota
# ---------------------------------------------------------------
import altair as alt

def plot_cota_altair(cota: pd.Series, cdi_cum: pd.Series | None = None) -> alt.Chart:
    # --- alinhar índices (interseção) ---
    if cdi_cum is not None:
        idx = cota.index.intersection(cdi_cum.index)
        df = pd.DataFrame({
            "Data": idx,
            "Carteira": cota.reindex(idx).values,
            "CDI":      cdi_cum.reindex(idx).values,
        })
        cols_val = ["Carteira", "CDI"]
        y_min = float(np.nanmin(df[cols_val].values))
        y_max = float(np.nanmax(df[cols_val].values))
    else:
        df = pd.DataFrame({"Data": cota.index, "Carteira": cota.values})
        cols_val = ["Carteira"]
        y_min = float(df["Carteira"].min())
        y_max = float(df["Carteira"].max())

    # padding vertical suave
    pad = 0.002 * (y_max - y_min) if y_max > y_min else 0.01

    # dados em formato "long" com fold
    base = (
        alt.Chart(df)
        .transform_fold(
            fold=cols_val,  # ["Carteira"] ou ["Carteira","CDI"]
            as_=["Série", "Índice"]
        )
    )

    # mapeamento de cores fixas (azul carteira, cinza CDI)
    color_scale = alt.Scale(
        domain=["Carteira", "CDI"],
        range=["#1565c0", "#7f7f7f"]
    )

    # linhas
    linhas = (
        base
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Data:T", title="", axis=alt.Axis(format="%d-%b")),
            y=alt.Y("Índice:Q",
                    title="Índice (base=1)",
                    scale=alt.Scale(domain=[y_min - pad, y_max + pad])),
            color=alt.Color("Série:N", title="", scale=color_scale),
            tooltip=[
                alt.Tooltip("Data:T", title="Data", format="%d %b %Y"),
                alt.Tooltip("Série:N", title="Série"),
                alt.Tooltip("Índice:Q", title="Valor", format=".4f"),
            ],
        )
        .properties(height=320)
        .interactive()
    )

    return linhas

# 2) ---- BARRAS DO RETORNO DIÁRIO ----------------------------------
def plot_ret_diario(ret: pd.Series) -> alt.Chart:
    df = (
        ret.replace([np.inf, -np.inf], np.nan)
           .dropna()
           .reset_index()
           .rename(columns={"index": "Data", 0: "Ret"})
           .sort_values("Data")
    )
    return (
        alt.Chart(df)
           .mark_bar()
           .encode(
               x=alt.X("Data:T", title="", axis=alt.Axis(format="%d-%b")),
               y=alt.Y("Ret:Q", title="Retorno diário"),
               color=alt.condition("datum.Ret >= 0",
                                   alt.value("#2e7d32"),  # verde
                                   alt.value("#c62828"))  # vermelho
           )
           .properties(height=240)
    )

import requests
from datetime import date

#@st.cache_data(ttl=24*3600)         # consulta ao SGS no máx. 1 vez/dia
def load_cdi_series(cache_csv: str = "Dados/cdi_cached.csv") -> pd.Series:
    """
    Retorna Series diária do CDI (decimal) indexada por datetime.

    1) Tenta ler o cache CSV (muito mais rápido).
    2) Se não existir ou estiver vazio, faz download da série 12 do SGS
       via requests e grava o cache.
    """
    # ───────── tenta cache ───────────────────────────────────────
    try:
        s = (pd.read_csv(cache_csv, parse_dates=["Data"])
                .set_index("Data")["cdi"]
                .astype(float)
                .sort_index())
        if not s.empty:
            return s
    except FileNotFoundError:
        pass                                          # segue para o download

    # ───────── download via API do BCB ───────────────────────────
    SGS_ID  = 12                      # CDI Over / Taxa DI
    dt_ini  = "2025-01-01"            # pode ajustar
    dt_fim  = dt.date.today().strftime("%Y-%m-%d")

    url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs/{SGS_ID}/dados"
           f"?formato=json&dataInicial={dt_ini}&dataFinal={dt_fim}")

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()           # dispara se 4xx/5xx

    df = (pd.DataFrame(resp.json())          # [{'data':'01/07/2024','valor':'0.0155'}, …]
            .rename(columns={"data": "Data", "valor": "cdi"})
            .assign(Data=lambda d: pd.to_datetime(d["Data"], dayfirst=True),
                    cdi =lambda d: pd.to_numeric(d["cdi"],
                                                 errors="coerce") / 100)   # 1,55% → 0.0155
            .dropna(subset=["cdi"]))

    serie = df.set_index("Data")["cdi"].sort_index()

    # guarda cache para os próximos 24 h
    serie.to_csv(cache_csv, index=True)

    return serie

# ----------------- helpers de datas -----------------
def _ensure_dt_index_series(s: pd.Series) -> pd.Series:
    """Garante DatetimeIndex (tenta 'date' no index name; normaliza e ordena)."""
    s2 = s.copy()
    if not isinstance(s2.index, pd.DatetimeIndex):
        s2.index = pd.to_datetime(s2.index, dayfirst=True, errors="coerce")
    s2 = s2[~s2.index.isna()].sort_index()
    # normaliza para meia-noite p/ evitar comparação com horas
    s2.index = s2.index.normalize()
    return s2

def _ensure_dt_index_df(df: pd.DataFrame) -> pd.DataFrame:
    """Garante DatetimeIndex no DataFrame (usa coluna 'date' se existir)."""
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        # tenta coluna 'date' / 'Date' / 'DATA'
        cand = None
        for c in ("date", "Date", "DATA"):
            if c in d.columns:
                cand = c
                break
        if cand is not None:
            d[cand] = pd.to_datetime(d[cand], dayfirst=True, errors="coerce")
            d = d.dropna(subset=[cand]).set_index(cand)
        else:
            d.index = pd.to_datetime(d.index, dayfirst=True, errors="coerce")
            d = d[~d.index.isna()]
    d = d.sort_index()
    d.index = d.index.normalize()
    return d

def _choose_effective_date(candidate: pd.Timestamp | str | None,
                           common_idx: pd.DatetimeIndex) -> pd.Timestamp:
    """Escolhe a data efetiva: última data ≤ candidate; se None, usa a última do índice comum."""
    if common_idx.empty:
        raise ValueError("Não há interseção de datas entre pl_series e df_retorno.")
    if candidate is None:
        return common_idx.max()
    candidate = pd.to_datetime(candidate, dayfirst=True, errors="coerce")
    if pd.isna(candidate):
        return common_idx.max()
    # normaliza
    candidate = candidate.normalize()
    # pega última data <= candidate
    left = common_idx[common_idx <= candidate]
    if left.size == 0:
        # se não há <= candidate, usa a menor disponível
        return common_idx.min()
    return left.max()

# ----------------- helpers de risco -----------------
def _pl_ref(pl_series: pd.Series, data: pd.Timestamp) -> float:
    """Retorna 1% do PL total na data."""
    return float(pl_series.loc[data] * 0.01)

def _var_cvar(series_ret: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """VaR e CVaR históricos (magnitudes positivas)."""
    s = pd.to_numeric(series_ret, errors="coerce").dropna()
    if s.empty:
        return 0.0, 0.0
    var_q = np.quantile(s, alpha)
    cvar = s[s <= var_q].mean() if (s <= var_q).any() else var_q
    return abs(var_q), abs(cvar)

def _stress_percent(valor_R: float, pl_ref_R: float) -> float:
    return 0.0 if pl_ref_R == 0 else (valor_R / pl_ref_R * 10_000)

# ----------------- função principal (versão corrigida) -----------------
# ── Wrappers de UI: retornam defaults quando ui=False ─────────────────────────────
def ui_checkbox(label, *, value=False, key=None, ui=True):
    return st.sidebar.checkbox(label, value=value, key=key) if ui else value

def ui_number_input(label, *, value=0.0, min_value=None, max_value=None, step=None, format=None, key=None, ui=True):
    return st.sidebar.number_input(label, value=value, min_value=min_value, max_value=max_value,
                                step=step, format=format if format else "%f", key=key) if ui else value

def ui_radio(label, options, *, index=0, horizontal=False, key=None, ui=True):
    return st.radio(label, options=options, index=index, horizontal=horizontal, key=key) if ui else options[index]

def ui_selectbox(label, options, *, index=0, key=None, ui=True):
    return st.selectbox(label, options, index=index, key=key) if ui else options[index]

def calcular_metricas_por_pl(
    pl_series: pd.Series,
    data: pd.Timestamp | str | None = None,
    alpha: float = 0.05,
    tick_val: float = 100.0,   # ignorado (compatibilidade)
    debug: bool = False,
    ui=True
):
    """
    Métricas do portfólio (VaR, CVaR, CoVaR, DV01 e DV01-stress) normalizadas pelo PL_ref (1% do PL do dia).
    Retorna (dict, default_assets).
    """

    # ---------------- 1) Base/retornos ----------------
    default_assets, quantidade_inicial, portifolio_default = processar_dados_port()
    df_base = pd.read_parquet('Dados/df_inicial.parquet')
    df_precos, df_completo = load_and_process_excel(df_base, default_assets)
    df_retorno = process_returns2(df_completo, default_assets)

    # ---------------- 2) Datas e interseção segura ----------------
    pl_series   = _ensure_dt_index_series(pl_series)
    df_retorno  = _ensure_dt_index_df(df_retorno)
    df_completo = _ensure_dt_index_df(df_completo)

    common_idx = df_retorno.index.intersection(pl_series.index)
    data_eff = _choose_effective_date(data, common_idx)

    # ---------------- 3) Ajustes e retornos até a data ----------------
    df_retorno_hist = df_retorno.loc[df_retorno.index <= data_eff].copy()
    var_ativos = var_not_parametric(df_retorno_hist).abs()
    df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)

    # ---------------- 4) Quantidades agregadas (todos os fundos) ----------------
    df_contratos_2 = read_atual_contratos_cached()
    if 'Fundo' in df_contratos_2.columns:
        df_contratos_2 = df_contratos_2.set_index('Fundo')

    asset_cols = [c for c in df_contratos_2.columns if c.startswith("Contratos ")]
    if not asset_cols:
        df_contratos_2 = df_contratos_2.rename(columns={c: f"Contratos {c}" for c in df_contratos_2.columns})
        asset_cols = list(df_contratos_2.columns)

    df_contratos_2 = df_contratos_2.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df_contratos_2 = df_contratos_2.loc[[i for i in df_contratos_2.index if str(i).strip().upper() != "TOTAL"]]

    s_quant = df_contratos_2[asset_cols].sum(axis=0)
    assets_from_contracts = [c.replace("Contratos ", "") for c in s_quant.index]
    quantidades = pd.Series(s_quant.values, index=assets_from_contracts).astype(float)

    # Universo final (para não perder histórico)
    assets_universe = sorted(set(default_assets).union(set(quantidades.index)))
    df_precos_u, df_completo_u = load_and_process_excel(df_base, assets_universe)
    df_retorno_u = process_returns2(df_completo_u, assets_universe)
    df_retorno_u = _ensure_dt_index_df(df_retorno_u)
    df_completo_u = _ensure_dt_index_df(df_completo_u)

    # Histórico até data_eff
    df_retorno_hist = df_retorno_u.loc[df_retorno_u.index <= data_eff].copy()
    if df_retorno_hist.empty:
        pl_ref = _pl_ref(pl_series, data_eff)
        return ({
            "PL_ref (R$)"               : float(pl_ref),
            "VaR (bps)"                 : 0.0,
            "CVaR (bps)"                : 0.0,
            "VaR (R$)"                  : 0.0,
            "CVaR (R$)"                 : 0.0,
            "DV01 Port (R$/bp)"         : 0.0,
            "DV01 Port (bps do PL_ref)" : 0.0,
            "DV01 Stress (R$)"          : 0.0,
            "DV01 Stress (bps)"         : 0.0,
            "VaR (% de 1bp)"            : 0.0,
            "CVaR (% de 1bp)"           : 0.0,
            "CoVaR total (R$)"          : 0.0,
            "CoVaR total (bps)"         : 0.0,
            "CoVaR por ativo (R$)"      : {},
            "CoVaR por ativo (bps)"     : {},
            "CoVaR por ativo (% de 1bp)": {},
        }, default_assets)

    # ---------------- 5) Pesos por MV e MV_total ----------------
    if "Valor Fechamento" in df_precos_ajustados.columns:
        precos_ult = df_precos_ajustados["Valor Fechamento"].reindex(assets_universe)
    else:
        precos_ult = df_completo_u[assets_universe].ffill().loc[:data_eff].iloc[-1]

    quantidades = quantidades.reindex(assets_universe).fillna(0.0)
    mv = (precos_ult * quantidades.abs()).fillna(0.0)     # MV por ativo (R$)
    mv_total = float(mv.sum())

    if mv_total > 0:
        cols = [c for c in assets_universe if c in df_retorno_hist.columns]
        mv = mv.reindex(cols).fillna(0.0)
        pesos = (mv / float(mv.sum())).values              # pesos de MV
        df_retorno_hist = df_retorno_hist[cols]
    else:
        cols = list(df_retorno_hist.columns)
        pesos = np.ones(len(cols)) / len(cols) if cols else np.array([])

    # ---------------- 6) VaR / CVaR ----------------
    port_ret = (df_retorno_hist * pesos).sum(axis=1) if len(pesos) else pd.Series(dtype=float)
    var_ret, cvar_ret = _var_cvar(port_ret, alpha=alpha)   # (em retorno)
    var_R  = float(var_ret  * mv_total)
    cvar_R = float(cvar_ret * mv_total)

    # Normalizações por PL_ref
    pl_ref  = float(_pl_ref(pl_series, data_eff))   # 1% do PL total (R$)
    one_bp_R = pl_ref * 1e-4

    var_bps  = float((var_R  / pl_ref) * 1e4) if pl_ref else 0.0
    cvar_bps = float((cvar_R / pl_ref) * 1e4) if pl_ref else 0.0
    var_pct_of_1bp  = float((var_R  / one_bp_R) * 100.0) if one_bp_R else 0.0
    cvar_pct_of_1bp = float((cvar_R / one_bp_R) * 100.0) if one_bp_R else 0.0

    # ---------------- 7) DV01 do portfólio (df_divone) ----------------
    file_bbg = "Dados/BBG - ECO DASH.xlsx"
    df_divone, _, _ = load_and_process_divone2(file_bbg, df_completo_u)
    #Colocar uma opção para aparecer os div01 -> um check na sidebar
    show_dv01 = ui_checkbox("Mostrar DV01", value=False, key="mostrar_dv01", ui=ui)
    if show_dv01:
        st.write(df_divone)

    candidates = ["FUT_TICK_VAL", "DV01", "BPV", "PVBP"]
    dv01_label = next((lab for lab in candidates if lab in df_divone.index), None)
    if dv01_label is None:
        dv01_per_contract = df_divone.loc[df_divone.index[0]].astype(float)
    else:
        dv01_per_contract = df_divone.loc[dv01_label].astype(float)

    intersec = [a for a in cols if a in dv01_per_contract.index]
    q_signed = quantidades.reindex(intersec).fillna(0.0)
    dv01_pc  = dv01_per_contract.reindex(intersec).fillna(0.0)
    dv01_asset_R = (dv01_pc * q_signed).astype(float)   # R$/bp por ativo (já agregado por quantidade)
    dv01_asset_bps = ((dv01_asset_R / pl_ref) * 1e4).replace([np.inf, -np.inf], 0.0).fillna(0.0) if pl_ref else dv01_asset_R*0.0
    dv01_port_R  = float(dv01_asset_R.sum())            # DV01 do portfólio (R$/bp)
    dv01_port_bps_of_PLref = float((dv01_port_R / pl_ref) * 1e4) if pl_ref else 0.0

    # ---------------- 8) DV01 "stress" por drawdown ----------------
    # ---------------- 8) Stress por ativo / por classe / portfólio ----------------
    # Helpers de classificação
    is_nominal = lambda a: a.startswith("DI_")
    is_real    = lambda a: a.startswith(("DAP", "NTNB"))
    is_us      = lambda a: "TREASURY" in a
    is_fx      = lambda a: "WDO1" in a

    nominais = [a for a in intersec if is_nominal(a)]
    reais    = [a for a in intersec if is_real(a)]
    us_rates = [a for a in intersec if is_us(a)]
    moeda    = [a for a in intersec if is_fx(a)]

    # Drawdown por ativo (retorno mínimo histórico até data_eff)
    dd_min_series = df_retorno_hist[intersec].min(axis=0).astype(float)
    mv_intersec   = mv.reindex(intersec).fillna(0.0).astype(float)

    # Stress POR ATIVO em R$ conforme regra
    stress_asset_R = {}
    for a in intersec:
        if is_nominal(a):
            # 100 bps sobre DV01
            #st.write(f"Ativo nominal: {a}, DV01: {dv01_asset_R.get(a, 0.0)}")
            #st.write(f"Drawdown: {dd_min_series.get(a, 0.0)}")
            #st.write(f"MV: {mv_intersec.get(a, 0.0)}")
            stress_asset_R[a] = abs(float(dv01_asset_R.get(a, 0.0))) * 100.0
            #st.write(f"Stress nominal R$: {stress_asset_R[a]}")
        elif is_real(a):
            # 50 bps sobre DV01
            stress_asset_R[a] = abs(float(dv01_asset_R.get(a, 0.0))) * 50.0
        elif is_us(a) or is_fx(a):
            # drawdown × MV
            stress_asset_R[a] = abs(float(dd_min_series.get(a, 0.0))) * float(mv_intersec.get(a, 0.0))
        else:
            # fallback: drawdown × MV
            stress_asset_R[a] = abs(float(dd_min_series.get(a, 0.0))) * float(mv_intersec.get(a, 0.0))

    # Versão em bps do PL_ref
    stress_asset_bps = {a: ((v / pl_ref) * 1e4 if pl_ref else 0.0) for a, v in stress_asset_R.items()}

    # Totais POR CLASSE (R$)
    stress_nom_R  = sum(stress_asset_R.get(a, 0.0) for a in nominais)
    stress_real_R = sum(stress_asset_R.get(a, 0.0) for a in reais)
    stress_us_R   = sum(stress_asset_R.get(a, 0.0) for a in us_rates)
    stress_fx_R   = sum(stress_asset_R.get(a, 0.0) for a in moeda)

    stress_cat_R = {
        "JUROS NOMINAIS BRASIL": stress_nom_R,
        "JUROS REAIS BRASIL"  : stress_real_R,
        "JUROS US"            : stress_us_R,
        "MOEDA"               : stress_fx_R,
    }
    stress_cat_bps = {k: ((v / pl_ref) * 1e4 if pl_ref else 0.0) for k, v in stress_cat_R.items()}

    # Stress COMBINADO das classes (aplica todos os choques simultaneamente)
    stress_combined_R   = stress_nom_R + stress_real_R + stress_us_R + stress_fx_R
    stress_combined_bps = (stress_combined_R / pl_ref) * 1e4 if pl_ref else 0.0

    # Stress agregado por ativo (que você já fazia): soma |dd_i| × MV_i
    dv01_stress_R   = float((dd_min_series.abs() * mv_intersec).sum())
    dv01_stress_bps = (dv01_stress_R / pl_ref) * 1e4 if pl_ref else 0.0

    # Drawdown do PORTFÓLIO (retorno mínimo do portfólio × MV_total)
    dd_port              = abs(float(port_ret.min())) if len(port_ret) else 0.0
    stress_port_dd_R     = dd_port * mv_total
    stress_port_dd_bps   = (stress_port_dd_R / pl_ref) * 1e4 if pl_ref else 0.0

    # ---------------- 9) CoVaR (procedimentos do seu trecho) ----------------
    # VaR do portfólio em retorno:
    var_port = abs(var_ret)

    # Volatilidade do portfólio (retorno)
    vol_port_retornos = float(port_ret.std())

    # Preço base para mVaR_em_dinheiro (como no seu código): usa "Valor Fechamento" se existir
    if "Valor Fechamento" in df_precos_ajustados.columns:
        precos_base = df_precos_ajustados["Valor Fechamento"].reindex(cols).fillna(0.0)
    else:
        precos_base = precos_ult.reindex(cols).fillna(0.0)

    # Monta DF temporário com a coluna 'Portifolio' para a covariância
    df_tmp = df_retorno_hist.copy()
    df_tmp["Portifolio"] = port_ret

    cov = df_tmp.cov()
    if "Portifolio" in cov.columns:
        cov_port = cov["Portifolio"].drop(labels=["Portifolio"], errors="ignore")
    else:
        cov_port = pd.Series(0.0, index=cols)

    denom = (vol_port_retornos ** 2) if vol_port_retornos != 0 else np.nan
    df_beta = cov_port / denom

    # mVaR no espaço de retorno
    df_mvar = df_beta * var_port

    # mVaR em R$ (seguindo seu código original: multiplicando por preço base)
    df_mvar_dinheiro = (df_mvar * precos_base.reindex(df_mvar.index).fillna(0.0)).astype(float)

    # CoVaR por ativo (R$): mVaR_i * peso_i * MV_total
    pesos_series = pd.Series(pesos, index=cols)
    covar_R = (df_mvar.reindex(cols).fillna(0.0) * pesos_series * mv_total).astype(float)

    # Distribuição percentual dentro do total de CoVaR (como no seu código)
    covar_total_R = float(covar_R.sum())
    covar_perc = (covar_R / covar_total_R) if covar_total_R != 0 else covar_R.copy()

    # CoVaR em bps do PL_ref e % de 1 bp
    covar_bps       = ((covar_R / pl_ref) * 1e4).replace([np.inf, -np.inf], 0.0).fillna(0.0) if pl_ref else covar_R*0.0
    covar_pct_1bp   = ((covar_R / one_bp_R) * 100.0).replace([np.inf, -np.inf], 0.0).fillna(0.0) if one_bp_R else covar_R*0.0
    covar_total_bps = float((covar_total_R / pl_ref) * 1e4) if pl_ref else 0.0

    # ---------------- 10) Saída ----------------
    out = {
        # Bases
        "PL_ref (R$)"               : pl_ref,
        "MV_total (R$)"             : mv_total,

        # VaR/CVaR
        "VaR (R$)"                  : var_R,
        "CVaR (R$)"                 : cvar_R,
        "VaR (bps)"                 : var_bps,
        "CVaR (bps)"                : cvar_bps,
        "VaR (% de 1bp)"            : var_pct_of_1bp,
        "CVaR (% de 1bp)"           : cvar_pct_of_1bp,

        # DV01
        "DV01 Port (R$/bp)"         : dv01_port_R,
        "DV01 Port (bps do PL_ref)" : dv01_port_bps_of_PLref,
        "DV01 Stress (R$)"          : dv01_stress_R,
        "DV01 Stress (bps)"         : dv01_stress_bps,
        "DV01 por ativo (R$/bp)"    : dv01_asset_R.to_dict(),
        "DV01 por ativo (bps)"      : dv01_asset_bps.to_dict(),

        # CoVaR (novo)
        "CoVaR total (R$)"          : covar_total_R,
        "CoVaR total (bps)"         : covar_total_bps,
        "CoVaR por ativo (R$)"      : covar_R.to_dict(),
        "CoVaR por ativo (bps)"     : covar_bps.to_dict(),
        "CoVaR por ativo (% de 1bp)": covar_pct_1bp.to_dict(),

        # Diagnósticos úteis
        "Beta por ativo"            : df_beta.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_dict(),
        "mVaR por ativo (ret)"      : df_mvar.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_dict(),
        "mVaR por ativo (R$)"       : df_mvar_dinheiro.to_dict(),
        "Peso MV por ativo"         : pesos_series.to_dict(),
    }

    out.update({
        # Stress por ativo
        "Stress por ativo (R$)" : stress_asset_R,
        "Stress por ativo (bps)": stress_asset_bps,

        # Stress por classe
        "Stress por classe (R$)" : stress_cat_R,
        "Stress por classe (bps)": stress_cat_bps,

        # Stress combinado (todas as classes)
        "Stress combinado classes (R$)" : stress_combined_R,
        "Stress combinado classes (bps)": stress_combined_bps,

        # Portfólio: 2 óticas de drawdown
        "Stress Portfólio (agregado por ativo) (R$)"  : dv01_stress_R,
        "Stress Portfólio (agregado por ativo) (bps)" : dv01_stress_bps,
        "Stress Portfólio (retorno min do port) (R$)" : stress_port_dd_R,
        "Stress Portfólio (retorno min do port) (bps)": stress_port_dd_bps,
    })

    out["DV01 por classe (R$/bp)"] = {
        "JUROS NOMINAIS BRASIL": float(dv01_asset_R.reindex(nominais).sum()),
        "JUROS REAIS BRASIL"  : float(dv01_asset_R.reindex(reais).sum()),
        "JUROS US"            : float(dv01_asset_R.reindex(us_rates).sum()),
        "MOEDA"               : float(dv01_asset_R.reindex(moeda).sum()),
    }

    if debug:
        st.write("Ativos usados:", cols)
        st.write("Vol port (ret):", vol_port_retornos)
        st.write("Beta:", df_beta)
        st.write("mVaR (ret):", df_mvar)
        st.write("mVaR (R$):", df_mvar_dinheiro)
        st.write("CoVaR (R$):", covar_R)
        st.write("CoVaR (% dentro do total):", covar_perc)
        st.write("CoVaR (bps do PL_ref):", covar_bps)
        st.write("CoVaR (% de 1bp):", covar_pct_1bp)


    return out, default_assets

# ============================================================
# POSIÇÕES HISTÓRICAS (delta ou nível) a partir do Supabase
# ============================================================
def _nearest_left(idx: pd.DatetimeIndex, alvo: pd.Timestamp) -> pd.Timestamp:
    pos = idx.searchsorted(alvo, side="right") - 1
    if pos < 0:
        return idx[0]
    return idx[pos]

@st.cache_data(show_spinner=False)
def build_positions_timeseries(
    _trading_index,                       # <- ignorado no hash (começa com _)
    interpret_quantities: str = "delta",
    trading_sig: tuple | None = None      # <- entra no hash (first,last,len)
) -> pd.DataFrame:
    """Retorna DataFrame (datas x ativos) com quantidade vigente por dia."""
    # Reconstroi o índice a partir do argumento “ignorado no hash”
    trading_index = pd.DatetimeIndex(pd.to_datetime(list(_trading_index)))

    df = hist_posicoes_supabase()
    if df is None or df.empty:
        return pd.DataFrame(index=trading_index)

    df = df.copy()
    df["Dia de Compra"] = pd.to_datetime(df["Dia de Compra"]).dt.normalize()
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0.0)

    if interpret_quantities.lower() == "delta":
        delta = (
            df.pivot_table(index="Dia de Compra", columns="Ativo",
                           values="Quantidade", aggfunc="sum")
              .sort_index()
              .reindex(trading_index, fill_value=0.0)
        )
        pos = delta.cumsum().reindex(trading_index).fillna(0.0)
    else:  # "level"
        last = (
            df.pivot_table(index="Dia de Compra", columns="Ativo",
                           values="Quantidade", aggfunc="last")
              .sort_index()
        )
        pos = last.reindex(trading_index).ffill().fillna(0.0)
    

    return pos.where(np.isfinite(pos), 0.0)

# ============================================================
# CÁLCULO LEVE POR DATA: DV01(d) e CoVaR(d)
# ============================================================
def _pl_ref_from_series(pl_series: pd.Series, d: pd.Timestamp) -> float:
    # 1% do PL do dia útil <= d
    di = _nearest_left(pl_series.index, d)
    return float(pl_series.loc[di]) * 0.01


# ============================================================
# HISTÓRICO NORMALIZADO (Top-N + Outros) PARA ÁREA EMPILHADA
# ============================================================
def _normalize_topN_from_dict(d: dict, top_n=8, use_abs=True, only_positive=False, covar_tot_rs: float | None = None) -> pd.Series:
    s = pd.Series(d, dtype=float)
    s = s.replace({np.inf: np.nan, -np.inf: np.nan}).dropna()
    if only_positive:
        s = s.clip(lower=0.0)
    elif use_abs:
        s = s.abs()
    tot = float(s.sum())
    if tot == 0:
        return pd.Series(dtype=float)
    s = s.sort_values(ascending=False)
    if len(s) > top_n:
        top = s.head(top_n)
        outros = pd.Series({"Outros": s.iloc[top_n:].sum()})
        s = pd.concat([top, outros])
    if covar_tot_rs != 0:
        return (s / covar_tot_rs)
    else:
        return (s / float(s.sum()))
    #return (s)

def build_history_normalized(
    dates: pd.DatetimeIndex,
    kind: str = "dv01",         # "dv01" | "covar"
    top_n: int = 8,
    weekly: bool = True,
    only_positive: bool = False,
    window: int = 126,          # janela para risco (CoVaR)
    alpha: float = 0.05,        # VaR para CoVaR
    pl_series: pd.Series | None = None,
    covar_tot_rs: float | None = None

) -> pd.DataFrame:
    """Retorna df (datas x ativosTopN+Outros) com shares (0..1) normalizados por linha."""
    #if not only_positive:
    #    covar_tot_rs = 0.0 
    b = st.session_state.get("_risk_bundle")
    if b is None:
        raise RuntimeError("Bundle não encontrado em session_state['_risk_bundle'].")

    # Redução semanal (sexta) para aliviar custo
    if weekly:
        dates = pd.DatetimeIndex(dates).to_series().groupby(pd.Grouper(freq="W-FRI")).last().dropna().index

    rows = []
    idxs = []
    for d in dates:
        date_val = int(pd.Timestamp(d).value)
        # calcula só o necessário por data
        res = calc_contribs_for_date_cached(
            date_val=date_val,
            window=window,
            alpha=alpha,
            bundle_signature=b["signature"],
            return_dv01=(kind=="dv01"),
            return_covar=(kind=="covar"),
        )
        if kind == "dv01":
            series_map = res["dv01_R$"]
            norm = _normalize_topN_from_dict(series_map, top_n=top_n, use_abs=True, only_positive=False, covar_tot_rs=covar_tot_rs)
            #norm = series_map
        else:
            series_map = res["covar_R$"]
            # para CoVaR você pediu positivos → only_positive=True
            norm = _normalize_topN_from_dict(series_map, top_n=top_n, use_abs=False, only_positive=True, covar_tot_rs=covar_tot_rs)
            #norm = series_map
        if norm.empty:
            continue
        rows.append(norm)
        idxs.append(pd.to_datetime(d))

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows, index=idxs).sort_index().fillna(0.0)
    return df_out

@st.cache_data(show_spinner=False)
def calc_contribs_for_date_cached(
    date_val: int,
    window: int,
    alpha: float,
    bundle_signature: tuple,
    return_dv01: bool = True,
    return_covar: bool = True,
) -> dict:
    """
    Retorna dicionário: {'dv01_R$': Series-like, 'covar_R$': Series-like, 'pesos': Series}
    A função busca os dados no bundle via st.session_state (que você define quando cria o bundle).
    O cache depende só de: data, janela, alpha e assinatura do bundle (não há widgets aqui).
    """
    b = st.session_state.get("_risk_bundle")
    if b is None:
        raise RuntimeError("Bundle não encontrado em session_state['_risk_bundle'].")
    
    # Garantir que date_val seja convertido para Timestamp
    d = pd.to_datetime(date_val, unit='ns')  # Convertendo 'date_val' para Timestamp

    # ---- Fatias na data ----
    rets = b["df_retorno_u"]
    cols = b["cols_returns"]

    # Garantir que o primeiro índice de rets seja do tipo Timestamp
    rets_index_first = pd.to_datetime(rets.index[0])  # Convertendo o primeiro índice de rets para Timestamp

    # Verifique se 'd' é menor que o primeiro índice de retorno e retorne dicionário vazio
    if d < rets_index_first:
        return {"dv01_R$": {}, "covar_R$": {}, "pesos": {}}

    # Ajuste para usar o índice correto no pandas (DataFrame com Timestamps)
    df_hist = rets.loc[:d, cols].tail(window)
    if df_hist.empty:
        return {"dv01_R$": {}, "covar_R$": {}, "pesos": {}}

    # Garantir que os índices estejam como Timestamps
    b["df_precos_u"].index = pd.to_datetime(b["df_precos_u"].index)

    # Agora a linha de fatiamento funcionará corretamente
    precos_d = b["df_precos_u"][cols].loc[:d].ffill().tail(1).squeeze()

    # Converta precos_d para formato numérico
    precos_d = pd.to_numeric(precos_d, errors="coerce").fillna(0.0)

    # Quantidades na data (<= d) - Ajuste de índice
    q_d = b["positions_ts"].reindex(columns=cols)

    # Caso as posições estejam vazias, retorne um dicionário vazio
    if q_d.empty:
        return {"dv01_R$": {}, "covar_R$": {}, "pesos": {}}

    # Ajuste no cálculo das quantidades
    # Ao invés de get_loc() com method='pad', vamos usar a fatiagem direta
    q_d = q_d.loc[q_d.index <= d].iloc[-1].fillna(0.0)

    # MV e pesos
    mv_d = (precos_d * q_d.abs()).fillna(0.0)
    mv_total_d = float(mv_d.sum())

    if mv_total_d <= 0:
        pesos_d = pd.Series(0.0, index=cols)
    else:
        pesos_d = (mv_d / mv_total_d).astype(float)

    out = {"dv01_R$": {}, "covar_R$": {}, "pesos": pesos_d.to_dict()}

    # ---------- DV01(d) ---------- 
    if return_dv01:
        dv01_R_d = (b["dv01_pc"].reindex(cols).fillna(0.0) * q_d).astype(float)
        out["dv01_R$"] = dv01_R_d.to_dict()

    # ---------- CoVaR(d) ----------
    if return_covar:
        # Retorno do portfólio com os pesos na data
        port_ret = (df_hist * pesos_d.reindex(df_hist.columns).values).sum(axis=1)
        std_p = float(port_ret.std())

        if std_p == 0 or np.isnan(std_p):
            out["covar_R$"] = {c: 0.0 for c in cols}
        else:
            # Calculando a covariância e o beta
            cov = df_hist.cov()
            var_p = float(port_ret.var())
            cov_port = cov[port_ret.name] if port_ret.name in cov.columns else df_hist.apply(lambda x: x.cov(port_ret))
            beta = cov_port / var_p

            # VaR do portfólio (retorno) no alfa
            var_port = abs(np.nanpercentile(port_ret.values, alpha*100.0))

            mvar = beta * var_port  # mVaR (ret)
            covar_R = (mvar.reindex(cols).fillna(0.0) * pesos_d * mv_total_d).astype(float)
            out["covar_R$"] = covar_R.to_dict()

    return out

def normalize_topN(series_map: dict, top_n=8, use_abs=True, only_positive=False):
    """Converte dict{name->value} em df normalizado (share)."""
    s = pd.Series(series_map, dtype=float).replace({np.inf: np.nan, -np.inf: np.nan}).dropna()
    if only_positive:
        s = s.clip(lower=0)
    elif use_abs:
        s = s.abs()
    if s.sum() == 0:
        return pd.DataFrame({"label":[],"share":[]})
    s = s.sort_values(ascending=False)
    if len(s) > top_n:
        top = s.head(top_n)
        outros = pd.Series({"Outros": s.iloc[top_n:].sum()})
        s = pd.concat([top, outros])
    df = s.reset_index()
    df.columns = ["label","share"]
    df["label_short"] = df["label"].map(lambda x: _short(x, 18))
    df["pct"] = df["share"] / df["share"].sum()
    return df

# Carrega e prepara tudo que NÃO muda por data
# Ajustando o código para garantir que a posição não seja zerada indevidamente e utilizando os dados de fechamento atualizados

# Função para ajustar as posições, garantindo que não sejam zeradas quando não houver operação
def adjust_positions(positions_ts, date, assets):
    """Ajusta as posições dos ativos para garantir que não sejam zeradas indevidamente."""
    # Verifica se a posição existe na data
    if date in positions_ts.index:
        return positions_ts.loc[date]
    # Caso não haja posição na data, mantém a posição do dia anterior
    previous_date = positions_ts.index[positions_ts.index.get_loc(date, method="pad")]
    return positions_ts.loc[previous_date]

# Ajustando o processo de retornos
def process_returns_with_b3(df_b3_fechamento, assets):
    """Processa os retornos utilizando os dados de fechamento mais recentes."""
    # Transpor o DataFrame para que as datas sejam as linhas
    df_b3_fechamento = df_b3_fechamento.set_index('Assets').transpose()
    
    # Garantir que as colunas sejam convertidas corretamente para valores numéricos
    df_b3_fechamento = df_b3_fechamento.apply(pd.to_numeric, errors='coerce')
    
    # Verificar se os ativos solicitados estão presentes nas colunas do DataFrame
    available_assets = df_b3_fechamento.columns
    assets_to_use = [asset for asset in assets if asset in available_assets]

    if not assets_to_use:
        raise KeyError(f"Nenhum dos ativos solicitados está presente em df_b3_fechamento: {assets}")

    # Seleciona os ativos presentes e usa os últimos 756 dias de dados (aproximadamente 3 anos)
    df_b3_fechamento = df_b3_fechamento[assets_to_use].tail(756) 
    
    # Calculando os retornos logarítmicos
    df_returns = np.log(df_b3_fechamento / df_b3_fechamento.shift(1))

    return df_returns

# Atualizando a função `get_risk_static_bundle` para refletir o uso correto do `df_b3_fechamento`
@st.cache_resource(show_spinner=False)
def get_risk_static_bundle(pl_signature: tuple, interpret_quantities: str = "delta"):
    """
    Atualiza a função para garantir o uso correto dos dados de fechamento e correção das posições.
    """
    # Verificar se o bundle já está no session_state
    if '_risk_bundle' in st.session_state:
        return st.session_state['_risk_bundle']  # Se já estiver no session_state, apenas retorna

    # Caso não esteja no session_state, cria o bundle
    b = {}

    # Processar dados do portfólio
    default_assets, _, _ = processar_dados_port()
    df_base = pd.read_parquet('Dados/df_inicial.parquet')
    df_precos, df_completo = load_and_process_excel(df_base, default_assets)
    df_retorno = process_returns2(df_completo, default_assets)
    df_retorno = df_retorno.copy()
    df_retorno.index = pd.to_datetime(df_retorno.index)
    assets_universe = sorted(set(default_assets) | set(df_retorno.columns))
    df_precos_u, df_completo_u = load_and_process_excel(df_base, assets_universe)


    # Processando os retornos com o df_b3_fechamento
    df_b3_fechamento = load_b3_prices()  # Carregar os preços mais atualizados
    df_retorno_u = process_returns_with_b3(df_b3_fechamento, default_assets)  # Usar a função ajustada para processar os retornos
    df_retorno_u.index = pd.to_datetime(df_retorno_u.index)

    trading_index = df_retorno_u.index

    trading_sig = (
        int(trading_index[0].value),
        int(trading_index[-1].value),
        int(len(trading_index)),
    )

    # Construção de positions_ts (exemplo de como fazer a construção das posições)
    positions_ts = build_positions_timeseries(
        trading_index,
        interpret_quantities=interpret_quantities,
        trading_sig=trading_sig
    )

    # Adicionar dv01_pc ao bundle - Aqui você pode calcular ou carregar esses dados
    # Exemplo de como calcular ou carregar 'dv01_pc' (deve ser adaptado ao seu caso)
    # DV01 por contrato (BBG) – estático
    file_bbg = "Dados/BBG - ECO DASH.xlsx"
    df_divone, _, _ = load_and_process_divone2(file_bbg, df_completo_u)
    
    # Verificando se a chave 'DV01' existe na indexação de df_divone e processando
    for cand in ("FUT_TICK_VAL", "DV01", "BPV", "PVBP"):
        if cand in df_divone.index:
            dv01_per_contract = pd.to_numeric(df_divone.loc[cand], errors="coerce").fillna(0.0)
            break
    else:
        # fallback: primeira linha
        dv01_per_contract = pd.to_numeric(df_divone.loc[df_divone.index[0]], errors="coerce").fillna(0.0)
    # ou se for algum dado fixo
    # df_dv01_pc = pd.Series(...)

    # Assinatura hashable para o cache
    bundle_signature = (
        int(trading_index[0].value), int(trading_index[-1].value),
        tuple(df_retorno_u.columns), len(df_retorno_u.columns)
    )

    # Criação do bundle e armazenamento no session_state
    b = {
        "df_retorno_u": df_retorno_u,
        "df_precos_u": df_completo,
        "trading_index": trading_index,
        "cols_returns": list(df_retorno_u.columns),
        "positions_ts": positions_ts,
        "signature": bundle_signature,
        "dv01_pc": dv01_per_contract,  # Incluindo 'dv01_pc' no bundle
    }

    st.session_state['_risk_bundle'] = b  # Armazenar no session_state para evitar recalculo

    return b

# A partir disso, o código estará utilizando os preços mais atualizados, e as posições dos ativos serão corretamente mantidas mesmo quando não houver operações para determinado ativo.
# Vamos aplicar essas mudanças agora.





def fmt_rs_br(v, nd=0):
    try:
        s = f"{float(v):,.{nd}f}"
        return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return "R$ 0"

def _short(s, n=22):
    s = str(s);  return s if len(s) <= n else s[:n-1] + "…"

def pad_axis_from_data(pos_max, neg_min, pad_frac=0.12):
    left  = neg_min if np.isfinite(neg_min) else 0.0
    right = pos_max if np.isfinite(pos_max) else 0.0
    pad = pad_frac * max(1.0, abs(left), abs(right))
    return [left - pad, right + pad]


# ==========================================================
#   PÁGINA – Simular Cota
# ==========================================================
def simulate_nav_cota() -> None:
    """Simula a cota usando PL diário + rendimento da LFT."""
    import numpy as np, pandas as pd

    # ───────────────────────────── 1. % investido
    pct = st.sidebar.number_input(
        "Percentual do PL a investir",
        min_value=0.1, max_value=100.0,
        value=1.0, step=0.1,
        format="%.1f"
      ) / 100        # vira decimal

    # ───────────────────────────── 2. séries auxiliares
    pl_series, taxa_adm_off  = load_pl_series()           # PL total por dia (float)
    lft_series = load_lft_series()          # retorno LFT diário (decimal)

    ref_date = pl_series.index[-1]          # penúltima data válida no PL
    pl_total = pl_series.loc[ref_date]
    capital0 = pct * pl_total

    st.subheader("Resumo da Simulação")
    c1, c2 = st.columns(2)
    c1.metric("PL total", f"R$ {pl_total:,.0f}", help=f"Data ref.: {ref_date.date()}")
    c2.metric("PL Risco", f"R$ {capital0:,.0f}", help=f"{pct*100:.1f} % do PL")

    # ───────────────────────────── 3. P&L diário (regra Portfólio)
    df_pnl, _ , df_despesas = analisar_dados_fundos2(
        soma_pl_sem_pesos = pl_total,
        df_b3_fechamento  = load_b3_prices(),
        df_ajuste         = load_ajustes(),
        basefundos        = load_basefundos()
    )

    pnl = (df_pnl
           .drop(columns="Total", errors="ignore")
           .apply(pd.to_numeric, errors="coerce")
           .sum(axis=0)
           .replace([np.inf, -np.inf], np.nan)
           .dropna())
    pnl.index = pd.to_datetime(pnl.index)
    pnl       = pnl.sort_index()
    st.html(
                '''
                <style>
                div[data-testid="stDateInput"] input {
                    color: black; /* Define o texto */
                                                    }
                
                </style>   
        
                '''
    )
    # ---------------- Series de DESPESAS provenientes das operações ---------------
    # df_despesas tem só 1 linha (“Despesas”); convertemos para Series
    desp_series = (df_despesas.loc["Despesas"]        # vira Series
                            .rename("desp_op"))
    desp_series.index = pd.to_datetime(desp_series.index)
    desp_series       = desp_series.sort_index()

    # ───────────────────────────── 4. alinhar datas
    common = (
        pnl.index
        .intersection(pl_series.index)
        .intersection(lft_series.index)
    ) 
    # ─── filtro de datas ──────────────────────────────────────────────
    data_min, data_max = pl_series.index.min(), pl_series.index.max()

    d_ini = st.sidebar.date_input("Início",  value=data_min, min_value=data_min, max_value=data_max)
    d_fim = st.sidebar.date_input("Fim",     value=data_max, min_value=data_min, max_value=data_max)

    # garante ordem
    if d_ini > d_fim:
        st.sidebar.error("Data inicial > final.")
        st.stop()

    # converte p/ Timestamp e aplica corte
    mask      = (common >= pd.to_datetime(d_ini)) & (common <= pd.to_datetime(d_fim))
    common    = common[mask]           # reaproveita o mesmo índice filtrado 

    pnl        = pnl.loc[common]
    pl_series  = pl_series.loc[common]
    lft_series = lft_series.loc[common]
    taxa_adm_off = taxa_adm_off.loc[common]
    cdi_series = (
        load_cdi_series()                # já decimal diário
        .reindex(common)               # garante mesmo índice…
        .fillna(method="ffill")        # …e preenche datas faltantes
    )
    desp_series = desp_series.reindex(common).fillna(0.0)

    # 1) Bundle estático e “ponte” no session_state
    pl_sig = (int(pl_series.index[0].value), int(pl_series.index[-1].value), len(pl_series))
    bundle = get_risk_static_bundle(pl_sig, interpret_quantities="delta")

    #Printar a parte de positions_ts
    #st.write(bundle["positions_ts"])
    st.session_state["_risk_bundle"] = bundle

    #st.write(pl_series)
    # o retorno de calcular_metricas_por_pl é     return {    "PL_ref (R$)"       : float(pl_ref),    "VaR (bps)"         : float(var_bps),    "CVaR (bps)"        : float(cvar_bps),    "VaR (R$)"          : float(var_R),    "CVaR (R$)"         : float(cvar_R),    "Stress DV01 (R$)"  : float(stress_R),    "Stress DV01 (bps)" : float(stress_bps),}
    # um dicionário com as métricas de risco
    # e o PL de referência (PL_ref) para o dia
    risco, default_assets = calcular_metricas_por_pl(pl_series, data=common[-1], alpha=0.05, tick_val=100.0)
    #st.write(risco)

    if pnl.empty:
        st.warning("Datas de P&L não batem com PL/LFT disponíveis.")
        return
    
    # ------------- SIDEBAR – parâmetros de custo -----------------
    st.sidebar.markdown("### Custos diários")

    taxa_adm_on = st.sidebar.checkbox(
        "Cobrar taxa de administração ‑ 2 % a.a.", value=False
    )

    custo_pct_aa = st.sidebar.number_input(
        "Custo adicional (% do PL a.a.)", min_value=0.0, max_value=100.0,
        value=0.0, step=0.1, format="%.2f"
    ) / 100        # vira decimal anual

    custo_fixo_rs = st.sidebar.number_input(
        "Custo fixo diário (R$)", min_value=0.0, value=0.0, step=10.0
    )

    # NOVO: taxa de performance
    perf_on = st.sidebar.checkbox("Cobrar taxa de performance sobre o que exceder o CDI?", value=False)
    perf_pct = st.sidebar.number_input("Taxa de performance (%)", min_value=0.0, max_value=50.0,
                                       value=20.0, step=1.0, format="%.1f") / 100.0

    # ------------- converte para custo diário --------------------
    rate_adm_dia  = (1.02**(1/252) - 1) if taxa_adm_on else 0.0
    rate_extra_dia = ((1 + custo_pct_aa)**(1/252) - 1) if custo_pct_aa else 0.0

    # capital efetivamente investido em cada dia (mesmo % do PL)
    capital_dia = pct * pl_series

    # após definir capital_dia …
    if taxa_adm_on:
        custo_adm   = capital_dia * rate_adm_dia
    else:
        custo_adm = taxa_adm_off
    custo_extra = capital_dia * rate_extra_dia
    custo_fixo  = pd.Series(custo_fixo_rs, index=capital_dia.index)

    custo_total_parcial =  custo_adm + custo_extra + custo_fixo
    custo_total_sem_perf = custo_adm + custo_extra + custo_fixo + desp_series
    

    # ───────────────────────────── 5. ganho de ajuste com LFT
    ganho_lft    = capital_dia * lft_series        # já existia
    ganho_total_pre_perf  = pnl + ganho_lft - custo_total_sem_perf   # ▼ subtrai custos
    capital_ini_dia = capital_dia.shift(1, fill_value=capital0)

    ret_preperf    = ganho_total_pre_perf / capital_ini_dia

    # ----------------- PERF FEE (20% do excesso vs CDI c/ estorno) -----------------
    if perf_on and perf_pct > 0:
        perf_fee = []
        estoque  = []          # <<< NOVO: série do estoque (provisão acumulada)
        prov_acum = 0.0        # provisão acumulada (R$)

        for d in common:
            # excesso vs CDI **do retorno pré-performance**
            excess_day = ret_preperf.loc[d] - cdi_series.loc[d]

            # base para cálculo = capital no início do dia
            base_r = capital_ini_dia.loc[d]

            if excess_day > 0:
                # provisiona 20% do excedente
                fee_day = perf_pct * excess_day * base_r
                prov_acum += fee_day
                perf_fee.append(fee_day)
            else:
                # estorno (no máximo o que tem de provisão)
                estorno_teorico = perf_pct * (-excess_day) * base_r
                release = min(prov_acum, estorno_teorico)
                prov_acum -= release
                perf_fee.append(-release)

            estoque.append(prov_acum)

        perf_fee    = pd.Series(perf_fee, index=common, name="perf_fee$")
        perf_stock  = pd.Series(estoque,  index=common, name="perf_stock$")
    else:
        perf_fee   = pd.Series(0.0, index=common, name="perf_fee$")
        perf_stock = pd.Series(0.0, index=common, name="perf_stock$")

    # custo total FINAL inclui a taxa de performance
    custo_total = custo_total_sem_perf + perf_fee

    # ganho líquido final
    ganho_total  = pnl + ganho_lft - custo_total
    ret_total    = ganho_total / capital_ini_dia
    cota         = (1 + ret_total).cumprod()


    ret_acum      = cota.iloc[-1] - 1                       # retorno da carteira
    vol_anual     = ret_total.std() * np.sqrt(252)          # vol-anual
    max_dd        = (cota / cota.cummax() - 1).min()        # drawdown

    # —— CDI no mesmo intervalo
    cdi_cum       = (1 + cdi_series).cumprod()
    ret_cdi_acum  = cdi_cum.iloc[-1] - 1                    # retorno CDI
    excesso_acum  = ret_acum - ret_cdi_acum                 # alfa bruto
    perc_cdi = (ret_acum / ret_cdi_acum) if ret_cdi_acum != 0 else np.nan

    excesso_diario = ret_total - cdi_series

    mu_excesso   = excesso_diario.mean()        # retorno médio diário
    sigma_excesso = excesso_diario.std()        # desvio-padrão diário

    sharpe_cdi = (mu_excesso / sigma_excesso) * np.sqrt(252)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Retorno carteira",  f"{ret_acum:,.2%}")
    c2.metric("Retorno CDI",       f"{ret_cdi_acum:,.2%}")
    c3.metric("% CDI",     f"{perc_cdi:,.2%}")
    c4.metric("Vol. anual",        f"{vol_anual:,.2%}")
    c5.metric("Sharpe",  f"{sharpe_cdi:,.2f}")
    c6.metric("Máx. Drawdown",     f"{max_dd:,.2%}")
    

    aba_cart,tab_orcamento = st.tabs(["Histortico Carteira", "Portfolio Atual"])
    with aba_cart:
        st.header("Simulação de Cota – Carteira")

        # ───────────────────────────── 6. métricas
        # ─────────────── 6. métricas ──────────────────────────────

        # ───────────────────────────── 7. gráficos
        st.altair_chart(plot_cota_altair(cota, cdi_cum=cdi_cum), use_container_width=True)
        st.altair_chart(plot_ret_diario(ret_total), use_container_width=True) 

        # ───────────────────────────── 8. cards de métricas
        # ─────────────── 8. caixinhas de métricas ─────────────────

        # ───────────────────────────── 9. download CSV
        out = pd.DataFrame({
            "pnl_r$"        : pnl,
            "ajuste_lft$"   : ganho_lft,
            "custos_r$"     : -(custo_total_sem_perf - desp_series),
            "desp_op$"      : -desp_series,
            "perf_fee$"     : -perf_fee,          # taxa de performance do dia (R$)
            "perf_stock$"   : perf_stock,         # <<< NOVO: estoque acumulado (R$)
            "ganho_total$"  : ganho_total,
            "ret_total"     : ret_total,
            "cota"          : cota,
            "cdi_ret"       : cdi_series,
            "excess_ret"    : ret_total - cdi_series,
        })

        out["cdi_ret"]      = cdi_series
        out["excess_ret"]   = excesso_diario
        #out["cdi_cum"]      = cdi_cum          # opcional
        #out["excess_cum"]   = excesso_acum     # opcional


        # opcional: tabela-expander com as colunas
        # ------------------------------------------------------------
        #  Detalhe diário — nomes de colunas mais claros
        # ------------------------------------------------------------
        detalhe_fmt = {
            "P&L (R$)"                     : "R$ {:,.0f}",
            "Ajuste LFT (R$)"              : "R$ {:,.0f}",
            "Custos (R$)"                  : "R$ {:,.0f}",
            "Despesas (R$)"                : "R$ {:,.0f}",
            "Taxa de performance (R$)"     : "R$ {:,.0f}",
            "Estoque taxa perf. (R$)"      : "R$ {:,.0f}",   # <<< NOVO
            "Ganho líquido (R$)"           : "R$ {:,.0f}",
            "Retorno diário (%)"           : "{:.4%}",
            "Cota"                         : "{:.4f}",
            "CDI diário (%)"               : "{:.4%}",
            "Excesso vs CDI (%)"           : "{:.4%}"
        }

        out_renomeado = (
            out.rename(columns={
                "pnl_r$"        : "P&L (R$)",
                "ajuste_lft$"   : "Ajuste LFT (R$)",
                "custos_r$"     : "Custos (R$)",
                "desp_op$"      : "Despesas (R$)",
                "perf_fee$"     : "Taxa de performance (R$)",
                "perf_stock$"   : "Estoque taxa perf. (R$)",   # <<< NOVO
                "ganho_total$"  : "Ganho líquido (R$)",
                "ret_total"     : "Retorno diário (%)",
                "cdi_ret"       : "CDI diário (%)",
                "excess_ret"    : "Excesso vs CDI (%)"
            })
        )
        
        from io import BytesIO

        # 1) DataFrame CRU para exportação (sem formatação de R$, % etc.)
        out_xlsx = out.copy()
        out_xlsx.index.name = "Data"

        # (opcional) renomear para nomes sem símbolos
        col_rename = {
            "pnl_r$"       : "pnl_rs",
            "ajuste_lft$"  : "ajuste_lft_rs",
            "custos_r$"    : "custos_rs",
            "desp_op$"     : "despesas_operacionais_rs",
            "perf_fee$"    : "taxa_performance_rs",
            "perf_stock$"  : "estoque_taxa_performance_rs",
            "ganho_total$" : "ganho_total_rs",
            "ret_total"    : "retorno_diario",
            "cota"         : "cota",
            "cdi_ret"      : "cdi_diario",
            "excess_ret"   : "excesso_vs_cdi"
        }
        out_xlsx.rename(columns=col_rename, inplace=True)

        # 2) Excel em memória
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            out_xlsx.reset_index().to_excel(writer, sheet_name="simulacao", index=False)
        buf.seek(0)
        # ───────────────────── 1) Retorno mensal ─────────────────────
        # ------------------------------------------------------------
        #  Retorno mensal (tabela + gráfico sem título, rótulos destaque)
        # ------------------------------------------------------------
        #st.write("## Retorno Mensal da Carteira")

        # 1) retornos mensais (já em ordem cronológica)
        # ─── 1) dados mensais ------------------------------------------------
        #ret_mensal_port = (1 + ret_total).resample("M").prod().sub(1)
        #ret_mensal_cdi  = (1 + cdi_series).resample("M").prod().sub(1)
        #ret_mensal_cdi  = ret_mensal_cdi.reindex(ret_mensal_port.index)
    #
        #df_mensal = (pd.DataFrame({
        #                "Mês"     : ret_mensal_port.index.strftime("%b / %y"),
        #                "Carteira": ret_mensal_port.values,
        #                "CDI"     : ret_mensal_cdi.values})
        #            .melt(id_vars="Mês", var_name="Série", value_name="Retorno"))
    #
        #ordem = df_mensal["Mês"].unique().tolist()     # eixo‑X cronológico
    #
        ## ─── 2) gráfico ------------------------------------------------------
        #barras = (
        #    alt.Chart(df_mensal)
        #    .mark_bar(size=26)                      # largura da barra
        #    .encode(
        #        x       = alt.X("Mês:O", sort=ordem, title=""),
        #        xOffset = "Série:N",                # barras lado‑a‑lado
        #        y       = alt.Y("Retorno:Q",
        #                        title="",
        #                        axis=alt.Axis(format=".1%")),
        #        color   = alt.Color("Série:N",
        #                            scale=alt.Scale(
        #                                domain=["Carteira", "CDI"],
        #                                range =["#084594", "#2ca02c"]))  # verde & laranja
        #    )
        #)
        #
        #labels = (
        #    barras.mark_text(
        #            dy=-8, fontSize=11, fontWeight="bold", color="black")
        #        .encode(text=alt.Text("Retorno:Q", format=".1%"))
        #)
        #
        #st.altair_chart(
        #    (barras + labels)
        #    .properties(height=320)
        #    .configure_axis(labelFontSize=12),
        #    use_container_width=True
        #)

        # ───────────────────── Retorno Mensal ───────────────────────
        st.write("## Retorno Mensal da Carteira")
        ret_mensal_port = (1 + ret_total ).resample("M").prod().sub(1)
        ret_mensal_cdi  = (1 + cdi_series).resample("M").prod().sub(1)          \
                                        .reindex(ret_mensal_port.index)      \
                                        .fillna(method="ffill")

        # rótulos sempre em ordem cronológica
        idx   = ret_mensal_port.index
        rotul = idx.strftime("%b / %y")            # ex.:  Jul / 25
        ordem = rotul.tolist()                     # lista SEM duplicatas

        # DataFrame LONG sem duplicar rótulos
        df_mensal_long = pd.concat([
            pd.DataFrame({"Mês": rotul, "Série": "Carteira",
                        "Ret":  ret_mensal_port.values}),
            pd.DataFrame({"Mês": rotul, "Série": "CDI",
                        "Ret":  ret_mensal_cdi.values})
        ])

        # ------------------------------------------------------------------
        # 2) gráfico  (barras Carteira  +  linha CDI)
        # ------------------------------------------------------------------
        import plotly.graph_objects as go

        fig = go.Figure()

        # — Carteira
        fig.add_trace(
            go.Bar(
                x=ordem,
                y=ret_mensal_port.values,
                name="Carteira",
                marker_color="#084594",
                text=[f"{v:.1%}" for v in ret_mensal_port.values],
                textposition="outside",
                textfont=dict(color="#084594", size=11),
            )
        )

        # — CDI
        fig.add_trace(
            go.Bar(
                x=ordem,
                y=ret_mensal_cdi.values,
                name="CDI",
                marker_color="#6d91f5",
                text=[f"{v:.1%}" for v in ret_mensal_cdi.values],
                textposition="outside",
                textfont=dict(color="#6d91f5", size=11),
            )
        )

        # — layout geral
        fig.update_layout(
            height=460,
            barmode="group",          # <-- barras lado‑a‑lado
            bargap=0.25,
            yaxis=dict(title="", tickformat=".1%"),
            xaxis=dict(title="", tickangle=0),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
            margin=dict(l=20, r=20, t=10, b=70),
            plot_bgcolor="white",
        )

        # grade horizontal suave
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Detalhe diário"):
            st.dataframe(out_renomeado.style.format(detalhe_fmt))
            st.download_button(
            label="⬇️ Baixar planilha (Excel)",
            data=buf,
            file_name=f"simulacao_{common[0]:%Y%m%d}_{common[-1]:%Y%m%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
                )

    with aba_cart:

        #Gráfico de Waterfall plot
        st.header("Performance por Estratégia")

        # 1) ------------- mapeia cada linha do df_pnl ----------------
        MAPA_ESTRAT = {
            "DI"      : "Juros Nominais BR",
            "DAP"     : "Juros Reais BR",
            #"NTNB"    : "Juros Reais BR",
            "TREASURY": "Juros US",
            "WDO"    : "Moedas",
            #NTNB"    : "Juros Reais BR",

        }

        # fatia o df_pnl somente nas datas filtradas
        cols_dt = pd.to_datetime(df_pnl.columns, errors="coerce")
        df_pnl.columns = cols_dt                 # agora as datas são datetime64
        df_pnl = df_pnl.loc[:, ~df_pnl.columns.isna()]   # remove colunas que viraram NaT

        df_pnl_clip = df_pnl.loc[:, df_pnl.columns.intersection(common)]

        # cria colunas auxiliares
        aux = (df_pnl_clip
                .assign(Ativo = df_pnl_clip.index.str.split(" - ").str[0])
                .assign(Estratégia = lambda d:
                        d["Ativo"].str.extract(r"^([A-Z]+)", expand=False)     # DI / DAP / …
                        .replace({"DAP\d*":"DAP", "DI":"DI",}, regex=True)
                        .map(MAPA_ESTRAT)
                        .fillna("Outros"))
            )

        # ------------- soma P&L por estratégia -----------------------
        pnl_estrat = (aux.drop(columns=["Ativo"])       # só números + “Estratégia”
                        .groupby("Estratégia")
                        .sum())                      # linhas = estratégia
        
        df_strat = pnl_estrat.T            # datas nas linhas
        df_strat.index.name = "Data"

        df_comp = pd.DataFrame({
            "Ajuste LFT"   : ganho_lft,
            "Custos Fixos" : -custo_total_parcial,
            "Despesas Op"  : -desp_series,
            "Taxa Perf"    : -perf_fee,          # <<< NOVO
        })

        df_full = pd.concat([df_strat, df_comp], axis=1).fillna(0.0)

        # ---------------------------------------------------------------
        # 2. Seletor de horizonte
        # ---------------------------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Horizonte para *attribution*")
        opcao = st.sidebar.selectbox(
            "Tipo de recorte",
            ("Dia", "Mês", "Período"),
            index=1           # ← “Mês” como default
        )
        # ----------------------------------------------------------------------------
        # lista de todas as datas válidas já filtradas pelo restante do app
        # (common já é DatetimeIndex ordenado)
        datas_disponiveis = common

        # =============================== DIA ========================================
        if opcao == "Dia":
            dia_escolhido = st.sidebar.date_input(
                "Escolha o dia",
                value=datas_disponiveis[-1].date(),              # default = último
                min_value=datas_disponiveis[0].date(),
                max_value=datas_disponiveis[-1].date()
            )
            dia_escolhido = pd.to_datetime(dia_escolhido)

            # se o usuário escolheu dia sem P&L, ajusta para o dia útil anterior
            if dia_escolhido not in datas_disponiveis:
                dia_escolhido = datas_disponiveis[datas_disponiveis.get_loc(
                    dia_escolhido, method="pad")]

            ini = fim = dia_escolhido

        # =============================== MÊS ========================================
        elif opcao == "Mês":
            # gera lista de meses disponíveis
            meses_idx = (datas_disponiveis.to_period("M")
                                    .unique()
                                    .sort_values())

            mes_strs  = [m.strftime("%b / %Y") for m in meses_idx]
            mes_map   = dict(zip(mes_strs, meses_idx))

            mes_sel_str = st.sidebar.selectbox(
                "Escolha o mês",
                mes_strs,
                index=len(mes_strs)-1          # default = último mês
            )
            mes_sel = mes_map[mes_sel_str]

            ini = mes_sel.to_timestamp("D")                # 1º dia do mês
            fim = (mes_sel + 1).to_timestamp("D") - pd.Timedelta(days=1)

            # garante que ini/fim caem dentro das datas disponíveis
            ini_pos = datas_disponiveis.get_indexer([ini], method="bfill")[0]
            ini     = datas_disponiveis[ini_pos]

            fim_pos = datas_disponiveis.get_indexer([fim], method="ffill")[0]
            fim     = datas_disponiveis[fim_pos]

        # ============================ PERÍODO =======================================
        else:   # “Período”
            ini = st.sidebar.date_input(
                "Início",
                value=datas_disponiveis[0].date(),
                min_value=datas_disponiveis[0].date(),
                max_value=datas_disponiveis[-1].date(),
                key="per_ini"
            )
            fim = st.sidebar.date_input(
                "Fim",
                value=datas_disponiveis[-1].date(),
                min_value=datas_disponiveis[0].date(),
                max_value=datas_disponiveis[-1].date(),
                key="per_fim"
            )
            if ini > fim:
                st.sidebar.error("Data inicial > final.")
                st.stop()

            ini = pd.to_datetime(ini)
            fim = pd.to_datetime(fim)

            # ajusta para datas efetivamente disponíveis
            ini_pos = datas_disponiveis.get_indexer([ini], method="bfill")[0]
            ini     = datas_disponiveis[ini_pos]

            fim_pos = datas_disponiveis.get_indexer([fim], method="ffill")[0]
            fim     = datas_disponiveis[fim_pos]

        # ─────────── intervalo escolhido pronto ───────────
        st.sidebar.success(f"Intervalo: {ini:%d/%m/%Y} → {fim:%d/%m/%Y}")

        # aplica corte
        mask = (df_full.index >= ini) & (df_full.index <= fim)
        df_periodo = df_full.loc[mask]

        # ---------------------------------------------------------------
        # 3. Contribuição em p.p. de retorno
        # ---------------------------------------------------------------
        # utilitário – devolve a data existente mais próxima (<= alvo)
        def nearest_left(idx: pd.DatetimeIndex, alvo: pd.Timestamp) -> pd.Timestamp:
            pos = idx.searchsorted(alvo, side="right") - 1
            if pos < 0:                                 # alvo antes da 1.ª data
                raise KeyError(f"{alvo} fora do intervalo")
            return idx[pos]

        # ─── substitua estas duas linhas que quebraram ───────────────────
        # capital_ini = capital_dia.loc[ini]
        # ret_port    = (cota.loc[fim] / cota.loc[ini]) - 1
        # ─────────────────────────────────────────────────────────────────
        ini_eff  = nearest_left(cota.index, ini)
        fim_eff  = nearest_left(cota.index, fim)

        capital_ini = capital_dia.loc[ini_eff]
        ret_port    = (cota.loc[fim_eff] / cota.loc[ini_eff]) - 1

        # ---------------------------------------------------------------
        # 3. Contribuição em p.p. de retorno
        # ---------------------------------------------------------------
        contrib = (df_periodo.sum() / capital_ini).copy()          # Series

        # 3‑B) renomeia / consolida componentes ------------------------
        mapa = {
            "Ajuste LFT":   "Caixa",
            "Custos Fixos": "Custos/Despesas",
            "Despesas Op":  "Custos/Despesas",
            "Taxa Perf":    "Custos/Despesas",   # <<< NOVO
        }
        
        contrib = (contrib
                .groupby(lambda x: mapa.get(x, x))   # aplica o mapa
                .sum())                              # soma Custos + Despesas

        # 3‑C)  ► acrescenta CDI e impõe a ordem final ------------------
        d0 = cdi_cum.index[cdi_cum.index.get_indexer([ini_eff], method="pad")[0]]
        d1 = cdi_cum.index[cdi_cum.index.get_indexer([fim_eff], method="pad")[0]]

        # pega o valor do dia imediatamente anterior a d0
        i0 = cdi_cum.index.get_loc(d0)
        base = cdi_cum.iloc[i0-1] if i0 > 0 else 1.0

        ret_cdi_periodo = (cdi_cum.loc[d1] / base) - 1

        df_wf = contrib.reset_index().rename(columns={"index": "Componente",
                                                    0:      "ret_pl"})

        # ‑‑ Performance (total)
        if "Performance" not in df_wf["Componente"].values:
            df_wf = pd.concat([df_wf,
                            pd.DataFrame({"Componente": ["Performance"],
                                            "ret_pl":    [df_wf["ret_pl"].sum()]})],
                            ignore_index=True)

        # ‑‑ CDI (benchmark)
        df_wf = pd.concat([df_wf,
                        pd.DataFrame({"Componente": ["CDI"],
                                        "ret_pl":    [ret_cdi_periodo]})],
                        ignore_index=True)

        # ordem: Caixa ▸ Custos/Despesas ▸ outros ▸ Performance ▸ CDI
        ordem_fix   = ["Caixa", "Custos/Despesas"]
        resto       = [c for c in df_wf["Componente"]
                        if c not in ordem_fix + ["Performance", "CDI"]]
        
        ordem_final = ordem_fix + resto + ["Performance", "CDI"]

        df_wf["Componente"] = pd.Categorical(df_wf["Componente"],
                                            categories=ordem_final,
                                            ordered=True)
        df_wf = df_wf.sort_values("Componente").reset_index(drop=True)

        # ---------------------------------------------------------------
        # 4. Waterfall (Plotly)
        # ---------------------------------------------------------------
        # medidas: relative (padrão) | total | absolute (CDI)
        df_wf["measure"] = np.select(
                [df_wf["Componente"] == "Performance",
                df_wf["Componente"] == "CDI"],
                ["total", "absolute"],
                default="relative")

        df_wf["text"] = df_wf["ret_pl"].map(lambda x: f"{x:+.2%}")

        # cores
        cores = dict(Positive="#1a7519",     # verde
                    Negative="#d62728",     # vermelho
                    Total   ="#000080",     # azul‑escuro
                    Bench   ="#7f7f7f")     # cinza – CDI

        def cor(v, c):
            if c == "Performance": return cores["Total"]
            if c == "CDI":         return cores["Bench"]
            return cores["Positive"] if v > 0 else cores["Negative"]

        colors = [cor(v, c) for v, c in zip(df_wf["ret_pl"], df_wf["Componente"])]

        # remove barras ~0 (salvo Performance │ CDI)
        tol = 1e-9
        df_wf = df_wf.loc[
            ~((df_wf["Componente"].isin(["Performance", "CDI"]) == False)
            & (df_wf["ret_pl"].abs() < tol))
        ]

        # gráfico -------------------------------------------------------
        fig = go.Figure()

        fig.add_trace(
            go.Waterfall(
                x       = df_wf.query("Componente != 'CDI'")["Componente"],
                y       = df_wf.query("Componente != 'CDI'")["ret_pl"],
                measure = df_wf.query("Componente != 'CDI'")["measure"],
                text    = df_wf.query("Componente != 'CDI'")["text"],
                textposition="outside",
                connector = dict(line=dict(color="grey")),
                increasing = dict(marker=dict(color="#1a7519")),  # verde
                decreasing = dict(marker=dict(color="#d62728")),  # vermelho
                totals     = dict(marker=dict(color="#000080")),  # azul‑escuro
                name="Atribuição"
            )
        )

        # ───── segunda trace: CDI isolado ───────────────────────────────────
        cdi_val = df_wf.loc[df_wf["Componente"]=="CDI","ret_pl"].values[0]
        fig.add_trace(
            go.Waterfall(
                x       = ["CDI"],
                y       = [cdi_val],
                measure = ["absolute"],
                text    = [f"{cdi_val:+.2%}"],
                textposition="outside",
                increasing = dict(marker=dict(color="#7f7f7f")),  # cinza
                name="CDI"
            )
        )

        # ───── layout (o mesmo de antes) ────────────────────────────────────
        fig.update_layout(
            height = 460,
            title  = dict(text=f"Atribuição de Performance {ini_eff:%d/%m/%Y} – "
                            f"{fim_eff:%d/%m/%Y}",
                        x=0.5, xanchor="center", font=dict(size=18)),
            yaxis_tickformat = ".2%",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=60, r=40, t=60, b=60),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # tabela opcional ----------------------------------------------
        with st.expander("Detalhe do período"):
            st.dataframe(df_wf.set_index("Componente")[["ret_pl"]]
                        .rename(columns={"ret_pl": "pontos‑percentuais"})
                        .style.format({"pontos‑percentuais": "{:+.2%}"}))
        # 3) Botões de download

        # ------------- gráfico Waterfall plot ------------------------
    #with aba_div01:
    #st.header("Análise Portfolio Compilado")

    # ---- helpers ----
    def fmt_rs(x):
        try: return f"R${float(x):,.0f}"
        except: return "R$0"
    def fmt_bps_raw(x):
        try: return f"{float(x):,.2f}"
        except: return "0,00"
    def fmt_pct(x):
        try: return f"{float(x):,.2f}%"
        except: return "0,00%"

    # ---- extrai do dicionário 'risco' ----
    var_rs          = float(risco.get("VaR (R$)", 0.0))
    cvar_rs         = float(risco.get("CVaR (R$)", 0.0))
    var_bps         = float(risco.get("VaR (bps)", 0.0))
    cvar_bps        = float(risco.get("CVaR (bps)", 0.0))

    dv01_port_rs    = float(risco.get("DV01 Port (R$/bp)", 0.0))
    dv01_port_bps   = float(risco.get("DV01 Port (bps do PL_ref)", 0.0))  # se vier 0, mostramos só R$
    dv01_stress_rs  = float(risco.get("DV01 Stress (R$)", 0.0))
    dv01_stress_bps = float(risco.get("DV01 Stress (bps)", 0.0))

    covar_tot_rs    = float(risco.get("CoVaR total (R$)", 0.0))
    covar_tot_bps   = float(risco.get("CoVaR total (bps)", 0.0))
    covar_bps_dict  = risco.get("CoVaR por ativo (bps)", {}) or {}
    covar_rs_dict   = risco.get("CoVaR por ativo (R$)",  {}) or {}

    # ---- strings combinadas (R$ / bps) ----
    var_display        = f"{fmt_rs(var_rs)} / {fmt_bps_raw(var_bps)}bps"
    cvar_display       = f"{fmt_rs(cvar_rs)} / {fmt_bps_raw(cvar_bps)}bps"
    dv01_port_display  = f"{fmt_rs(dv01_port_rs)} / {fmt_bps_raw(dv01_port_bps)}bps" if dv01_port_bps else fmt_rs(dv01_port_rs)
    dv01_strss_display = f"{fmt_rs(dv01_stress_rs)} / {fmt_bps_raw(dv01_stress_bps)}bps" if dv01_stress_bps else fmt_rs(dv01_stress_rs)
    covar_tot_display  = f"{fmt_rs(covar_tot_rs)} / {fmt_bps_raw(covar_tot_bps)}bps"


    def pct_consumo_var(bps):  # % do orçamento escolhido
        try:
            return max(0.0, (abs(float(bps)) / float(orcamento_bps_var)) * 100.0)
        except:
            return 0.0

    def pct_consumo_cvar(bps):  # % do orçamento escolhido
        try:
            return max(0.0, (abs(float(bps)) / float(orcamento_bps_cvar)) * 100.0)
        except:
            return 0.0

    # ==========================
    # Cards principais (sem delta/setinhas)
    # ==========================
    import plotly.graph_objects as go

    with tab_orcamento:
        col11,col22 = st.columns(2)
        COL1, COLmeio, COL2 = st.columns([4.8, 0.2, 4.8])
        with col11:
            st.subheader("Resumo de Orçamento")
            c1, c2  = st.columns(2)
            c1.metric("VaR (R$ / bps)",  var_display)
            c2.metric("CVaR (R$ / bps)", cvar_display)
            # ---- orçamento de risco (1/2/3 bps) ----
            coll1,coll2 = st.columns(2)
            with coll1:
                orcamento_bps_var = st.sidebar.radio(
                    "Orçamento de risco VaR (%)",
                options=[1, 2, 3],
                index=0,
                horizontal=True
            )
            with coll2:
                orcamento_bps_cvar = st.sidebar.radio(
                    "Orçamento de risco CVaR (%)",
                    options=[1, 2, 3],
                    index=2,
                    horizontal=True
                )

        with COLmeio:
            # Adicionar linha vertical
            st.html(
                '''
                        <div class="divider-vertical-lines"></div>
                        <style>
                            .divider-vertical-lines {
                                border-left: 2px solid rgba(49, 51, 63, 0.2);
                                height: 40vh;
                                margin: auto;
                            }
                            @media (max-width: 768px) {
                                .divider-vertical-lines {
                                    display: none;
                                }
                            }
                        </style>
                        '''
            )




    # ======================= DRAWNDOWN: série, atual e dias =======================
    dd_series = (cota / cota.cummax()) - 1
    dd_atual = float(dd_series.iloc[-1])

    rolling_max = cota.cummax()
    is_peak = (cota == rolling_max)
    ultima_data_pico = is_peak[is_peak].index[-1] if is_peak.any() else cota.index[0]
    dias_em_dd = (cota.index[-1] - ultima_data_pico).days if dd_atual < 0 else 0
    with tab_orcamento:
        with COL1:
            st.subheader("Resumo de DV01")
            c4, c5= st.columns(2)
            c4.metric("DV01 Port (R$/bp / bps)", dv01_port_display)
            c5.metric("DV01 Stress (R$ / bps)",  dv01_strss_display)
            #c6.metric("CoVaR Total (R$ / bps)",  covar_tot_display)



    # ======================= VOL CURTA (1M, 6M, 1Y) ==============================
    def vol_annualized(ser: pd.Series) -> float:
        ser = ser.dropna()
        if len(ser) < 5:
            return np.nan
        return float(ser.std() * np.sqrt(252))

    ret_clean = ret_total.dropna()
    n_obs = len(ret_clean)

    # janelas alvo
    janelas = {"1M": 21, "6M": 126, "1Y": 252}

    # cálculo "robusto": se não houver N exatos, usa o que existir (cap no disponível)
    vols_finais = {}
    for nome, win in janelas.items():
        janela_real = min(win, n_obs)  # garante valor mesmo com < win
        if janela_real >= 5:
            vols_finais[nome] = vol_annualized(ret_clean.iloc[-janela_real:])
        else:
            vols_finais[nome] = np.nan


    


    with aba_cart:
        st.subheader("Resumo de Volatilidade")
        c7, c8, c9, c10 = st.columns(4)
        c7.metric("Vol (últ. 1M)", f"{vols_finais['1M']:,.2%}" if np.isfinite(vols_finais['1M']) else "—")
        c8.metric("Vol (últ. 6M)", f"{vols_finais['6M']:,.2%}" if np.isfinite(vols_finais['6M']) else "—")
        c9.metric("Vol (últ. 1Y)", f"{vols_finais['1Y']:,.2%}" if np.isfinite(vols_finais['1Y']) else "—",
                help="Se não houver 252 dias, usa tudo que existir e anualiza.")
        c10.metric("Drawdown corrente", f"{dd_atual:,.2%}", help=f"Dias no DD: {dias_em_dd}d")

        # ======================= GRÁFICOS LADO A LADO (AZUIS) ========================
        g1, g2 = st.columns(2)

        # --- Gráfico: Drawdown histórico (linha azul, zero tracejado) ---
        with g1:
            fig_dd = go.Figure()

            # Adicionar a linha do Drawdown
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series.values,
                mode="lines", name="Drawdown", line=dict(color="#1f77b4", width=2),
                fill='tozeroy',  # Preencher a área abaixo da linha com a cor desejada
                fillcolor='rgba(173, 216, 230, 0.3)'  # Cor azul clarinho com transparência
            ))

            fig_dd.update_layout(
                title="Drawdown",
                xaxis_title="",
                yaxis_title="",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=30, b=10),
                shapes=[dict(
                    type="line", xref="paper", x0=0, x1=1, yref="y", y0=0, y1=0,
                    line=dict(width=1, dash="dot", color="#888")
                )]
            )

            fig_dd.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig_dd, use_container_width=True)

            # --- Gráfico: Volatilidade histórica (uma linha) ---
            # Para ficar parecido ao print, use a vol rolling de 21 dias (1M).
            vol_1m_rolling = ret_total.rolling(21, min_periods=5).std() * np.sqrt(252)

        with g2:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_1m_rolling.index, y=vol_1m_rolling.values,
                mode="lines", name="Vol 1M (rolling)",
                line=dict(color="#1f77b4", width=2)
            ))
            fig_vol.update_layout(
                title="Volatilidade",
                xaxis_title="",
                yaxis_title="",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=30, b=10)
            )
            fig_vol.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig_vol, use_container_width=True)



    #c6.metric("Orçamento selecionado", f"{orcamento_bps_var} bps")
    #c7.metric("Orçamento CVaR selecionado", f"{orcamento_bps_cvar} bps")



    # ==========================
    # STRESS — visão geral
    # ==========================
    #st.subheader("Stress — visão geral")

    # leitura segura
    def _f(key, default=0.0):
        try: return float(risco.get(key, default))
        except: return default

    def _fd(key):
        try: return {k: float(v) for k,v in (risco.get(key, {}) or {}).items()}
        except: return {}

    # stress por ativo (se quiser usar depois)
    stress_asset_R   = _fd("Stress por ativo (R$)")
    stress_asset_bps = _fd("Stress por ativo (bps)")

    # por classe / combinado / portfólio
    stress_cls_R   = _fd("Stress por classe (R$)")
    stress_cls_bps = _fd("Stress por classe (bps)")

    stress_comb_R   = _f("Stress combinado classes (R$)")
    stress_comb_bps = _f("Stress combinado classes (bps)")

    stress_port_aggr_R   = _f("Stress Portfólio (agregado por ativo) (R$)")
    stress_port_aggr_bps = _f("Stress Portfólio (agregado por ativo) (bps)")

    stress_port_dd_R   = _f("Stress Portfólio (retorno min do port) (R$)")
    stress_port_dd_bps = _f("Stress Portfólio (retorno min do port) (bps)")

    # strings R$ / bps
    def join_rs_bps(rs, bps): return f"{fmt_rs(rs)} / {fmt_bps_raw(bps)}bps"

    #colA, colB, colC = st.columns(3)

    #colA.metric("Stress combinado (todas as classes)", join_rs_bps(stress_comb_R, stress_comb_bps))
    #colB.metric("Stress portfólio (agregado por ativo)", join_rs_bps(stress_port_aggr_R, stress_port_aggr_bps))
    #colC.metric("Stress portfólio (drawdown do portfólio)", join_rs_bps(stress_port_dd_R, stress_port_dd_bps))


    # ==========================
    # STRESS por CLASSE — R$ e bps + composição 100%
    # ==========================
    #st.subheader("Stress por classe")
#
    #if len(stress_cls_R) > 0:
    #    import pandas as pd
    #    import plotly.express as px
#
    #    # dataframe base
    #    df_sc = pd.DataFrame({"Classe": list(stress_cls_R.keys()),
    #                        "Stress_R$": [float(v) for v in stress_cls_R.values()]})
    #    df_sc["Stress_bps"] = df_sc["Classe"].map(stress_cls_bps).fillna(0.0).astype(float)
#
    #    # ordena segundo classes_order, se existirem
    #    df_sc["ord"] = df_sc["Classe"].apply(lambda c: classes_order.index(c) if c in classes_order else 999)
    #    df_sc = df_sc.sort_values("ord").drop(columns="ord")
#
    #    col1, col2 = st.columns(2)
#
    #    # --- barras R$ ---
    #    with col1:
    #        fig_r = px.bar(
    #            df_sc, x="Classe", y="Stress_R$",
    #            labels={"Stress_R$": "Stress (R$)", "Classe": ""},
    #            text=df_sc["Stress_R$"].map(lambda v: fmt_rs(v))
    #        )
    #        fig_r.update_traces(marker_color="#2563EB", textposition="outside")
    #        fig_r.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=40),
    #                            xaxis=dict(tickangle=-20), yaxis=dict(tickformat=",.0f"))
    #        st.plotly_chart(fig_r, use_container_width=True)
#
    #    # --- barras bps ---
    #    with col2:
    #        fig_b = px.bar(
    #            df_sc, x="Classe", y="Stress_bps",
    #            labels={"Stress_bps": "Stress (bps)", "Classe": ""},
    #            text=df_sc["Stress_bps"].map(lambda v: f"{v:,.2f}bps")
    #        )
    #        fig_b.update_traces(marker_color="#9333EA", textposition="outside")
    #        fig_b.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=40),
    #                            xaxis=dict(tickangle=-20))
    #        st.plotly_chart(fig_b, use_container_width=True)
#
    #    # --- composição 100% do stress combinado (só positivos; se zero, usa absoluto) ---
    #    st.caption("Composição do stress combinado (100%) por classe")
    #    df_share = df_sc.copy()
    #    # só positivos; se a soma positivar for zero, usa valor absoluto
    #    soma_pos = float(df_share["Stress_R$"].clip(lower=0.0).sum())
    #    if soma_pos > 0:
    #        df_share["base"] = df_share["Stress_R$"].clip(lower=0.0)
    #    else:
    #        df_share["base"] = df_share["Stress_R$"].abs()
#
    #    total_base = float(df_share["base"].sum())
    #    if total_base > 0:
    #        df_share["share"] = (df_share["base"] / total_base).astype(float)
#
    #        fig100 = px.bar(
    #            df_share, x="Classe", y="share",
    #            labels={"share": "Participação (100%)", "Classe": ""},
    #            text=df_share["share"].map(lambda v: f"{v:.2%}")
    #        )
    #        fig100.update_traces(marker_color="#0EA5E9", textposition="outside")
    #        fig100.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=40),
    #                            xaxis=dict(tickangle=-20), yaxis=dict(range=[0,1], tickformat=".0%"))
    #        st.plotly_chart(fig100, use_container_width=True)
    #    else:
    #        st.info("Não há valor para distribuir na composição 100% das classes.")
#
    #else:
    #    st.info("Stress por classe indisponível.")


    # ==========================
    # Donuts de consumo do orçamento (VaR e CVaR)
    # ==========================
    with tab_orcamento:
        import pandas as pd

        def donut_chart(label, pct):
            # clamp para o gráfico, mas mostramos o número real ao lado
            pct_clamped = max(0.0, min(100.0, pct))
            df_donut = pd.DataFrame({
                "categoria": ["Consumido", "Livre"],
                "valor": [pct_clamped, 100.0 - pct_clamped]
            })
            try:
                import altair as alt
                base = alt.Chart(df_donut).encode(
                    theta="valor:Q",
                    color=alt.Color("categoria:N",
                                    scale=alt.Scale(domain=["Consumido","Livre"],
                                                    range=["#4F46E5","#E5E7EB"]),
                                    legend=None)
                )
                donut = base.mark_arc(outerRadius=80, innerRadius=55)
                texto = alt.Chart(pd.DataFrame({"txt":[f"{pct:.2f}%"]})).mark_text(fontSize=18, fontWeight="bold").encode(text="txt:N")
                st.altair_chart(donut | texto, use_container_width=True)  # Ajustar altura do gráfico
            except Exception:
                # fallback
                st.progress(pct_clamped/100.0)
        with col22:
            st.subheader("Consumo do orçamento (VaR e CVaR)")
            colA, colB = st.columns(2)
            with colA:
                st.caption(f"VaR — {fmt_pct(pct_consumo_var(var_bps/100))} do orçamento")
                donut_chart("VaR", pct_consumo_var(var_bps/100))
            with colB:
                st.caption(f"CVaR — {fmt_pct(pct_consumo_cvar(cvar_bps/100))} do orçamento")
                donut_chart("CVaR", pct_consumo_cvar(cvar_bps/100))

        # ========= Helpers NOVOS (podem ficar no topo do tab) =========
        import plotly.express as px
        import plotly.graph_objects as go

        with COL1:

        # ===================== DV01 por CLASSE com CATEGORIAS empilhadas =====================
            st.subheader("DV01 por classe")

            dv01_asset_rs_dict  = risco.get("DV01 por ativo (R$/bp)", {}) or {}
            if not dv01_asset_rs_dict:
                st.info("DV01 por ativo indisponível para este portfólio.")
            else:
                import re, numpy as np, pandas as pd
                import plotly.graph_objects as go

                # --- helper de classe (mesmas regras que você usa) ---
                def map_classe(a: str) -> str:
                    au = str(a).upper()
                    if au.startswith("DI_") or au.startswith("DI"):
                        return "JUROS NOMINAIS BRASIL"
                    if au.startswith(("DAP","NTNB")):
                        return "JUROS REAIS BRASIL"
                    if "TREASURY" in au:
                        return "JUROS US"
                    if au.startswith("WDO"):
                        return "MOEDA"
                    return "OUTROS"

                # opcional: encurta rótulos de ativos na legenda
                def _short(s, n=20):
                    s = str(s)
                    return s if len(s) <= n else s[:n-1] + "…"

                # --- base por ativo ---
                df_dv = pd.DataFrame({
                    "Ativo": list(dv01_asset_rs_dict.keys()),
                    "DV01_R$": [float(v) for v in dv01_asset_rs_dict.values()],
                })
                df_dv = df_dv[df_dv["DV01_R$"] != 0.0].copy()
                if df_dv.empty:
                    st.info("Todos os DV01 por ativo estão zerados.")
                else:
                    df_dv["Classe"] = df_dv["Ativo"].map(map_classe)

                    # wide: linhas = Classe, colunas = ATIVOS, valores = DV01_R$
                    wide = (
                        df_dv.pivot_table(index="Classe", columns="Ativo", values="DV01_R$", aggfunc="sum")
                            .fillna(0.0)
                    )

                    # ordem amigável de classes
                    classes_order = ["JUROS NOMINAIS BRASIL", "JUROS REAIS BRASIL", "JUROS US", "MOEDA", "OUTROS"]
                    present = [c for c in classes_order if c in wide.index] + [c for c in wide.index if c not in classes_order]
                    wide = wide.loc[present]

                    # remove ATIVOS totalmente zerados (não aparecem na legenda)
                    wide = wide.loc[:, (wide != 0).any(axis=0)]
                    if wide.shape[1] == 0:
                        st.info("Sem ativos com DV01 diferente de zero para plotar.")
                    else:
                        col_left, col_right = st.columns([7, 3])

                        # ---------------- barras empilhadas: Classe × (ativos) ----------------
                        with col_left:
                            fig = go.Figure()

                            # range do eixo X com base na soma dos positivos/negativos por classe
                            pos_max = float(wide.clip(lower=0).sum(axis=1).max())
                            neg_min = float(wide.clip(upper=0).sum(axis=1).min())

                            # traço = 1 ativo (empilhado por classe)
                            for ativo in wide.columns:
                                x = wide[ativo]
                                if (x == 0).all():
                                    continue
                                fig.add_trace(go.Bar(
                                    y=wide.index,
                                    x=x,
                                    orientation="h",
                                    name=_short(ativo, 22),                  # legenda curta
                                    customdata=np.array([ativo]*len(x)),     # hover com nome completo
                                    hovertemplate="<b>%{y}</b><br>Ativo: %{customdata}"
                                                "<br>DV01: %{x:,.0f} R$/bp<extra></extra>",
                                    # rótulo só em barras “maiores” (evita poluição visual)
                                    text=[f"{v:,.0f}" if abs(v) >= np.nanpercentile(np.abs(wide.values), 75) else "" for v in x],
                                    textposition="outside",
                                    cliponaxis=False,
                                ))

                            fig.update_layout(
                                barmode="relative",   # empilha positivos/negativos
                                #height=max(260, 64 * len(wide.index)),
                                margin=dict(l=20, r=20, t=10, b=10),
                                xaxis=dict(title="DV01 (R$/bp)", tickformat=",.0f",
                                        showgrid=True, gridcolor="#F3F4F6"),
                                yaxis=dict(title=""),
                                legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", title="Ativos"),
                                plot_bgcolor="white",
                                shapes=[dict(type="line", xref="x", x0=0, x1=0, yref="paper", y0=0, y1=1,
                                            line=dict(width=1, dash="dot", color="#9CA3AF"))],
                            )
                            # range com “folga” visual
                            pad = 0.12 * max(1.0, abs(pos_max), abs(neg_min))
                            fig.update_xaxes(range=[neg_min - pad, pos_max + pad])

                            st.plotly_chart(fig, use_container_width=True)

                        # ---------------- donut: distribuição por ativo (Top-8 + Outros) ----------------
                        PIE_HEIGHT = 340
                        with col_right:
                            df_norm = normalize_topN(dv01_asset_rs_dict, top_n=8, use_abs=True, only_positive=False)
                            if df_norm.empty:
                                st.info("Sem DV01 para normalizar.")
                            else:
                                fig_p = go.Figure(go.Pie(
                                    labels=df_norm["label_short"],
                                    values=df_norm["pct"],
                                    hole=0.55, sort=False, direction="clockwise",
                                    textinfo="percent+label",
                                    textposition="inside",
                                    insidetextorientation="radial"
                                ))
                                fig_p.update_traces(textfont=dict(size=12))
                                fig_p.update_layout(
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    showlegend=False,
                                    height=PIE_HEIGHT
                                )
                                st.plotly_chart(fig_p, use_container_width=True)
                                st.caption("Distribuição do DV01 (|R$|): Top-8 + “Outros”.")


        # ========================== CoVaR por ativo (AJUSTES + PIZZA) ==========================
        # ===================== CoVaR por CLASSE (ativos empilhados) + donut =====================
        #with COL2:
        colll1, colllmeio, colll2 = st.columns([4.8, 0.2, 4.8])
        with colll1:

            st.subheader("CoVaR por classe")
            covar_bps_dict = risco.get("CoVaR por ativo (bps)", {}) or {}
            if not covar_bps_dict:
                st.info("CoVaR por ativo indisponível para este portfólio.")
            else:
                import plotly.graph_objects as go

                # ── mesma regra de classe usada no DV01 ───────────────────────────
                def map_classe(a: str) -> str:
                    au = str(a).upper()
                    if au.startswith("DI_") or au.startswith("DI"):
                        return "JUROS NOMINAIS BRASIL"
                    if au.startswith(("DAP","NTNB")):
                        return "JUROS REAIS BRASIL"
                    if "TREASURY" in au:
                        return "JUROS US"
                    if au.startswith("WDO"):
                        return "MOEDA"
                    return "OUTROS"

                def _short(s, n=22):
                    s = str(s);  return s if len(s) <= n else s[:n-1] + "…"

                # ── base por ativo ────────────────────────────────────────────────
                df_cv = pd.DataFrame({
                    "Ativo": list(covar_bps_dict.keys()),
                    "CoVaR_bps": [float(v) for v in covar_bps_dict.values()],
                })

                # Convertendo CoVaR de bps para percentual
                df_cv["CoVaR_percent"] = df_cv["CoVaR_bps"]
                df_cv = df_cv[df_cv["CoVaR_percent"] != 0.0].copy()

                if df_cv.empty:
                    st.info("Todos os CoVaR por ativo estão zerados.")
                else:
                    df_cv["Classe"] = df_cv["Ativo"].map(map_classe)

                    # wide: linhas = Classe, colunas = Ativo, valores = CoVaR (percentual)
                    wide = (
                        df_cv.pivot_table(index="Classe", columns="Ativo", values="CoVaR_percent", aggfunc="sum")
                            .fillna(0.0)
                    )

                    # ordem amigável de classes
                    classes_order = ["JUROS NOMINAIS BRASIL", "JUROS REAIS BRASIL", "JUROS US", "MOEDA", "OUTROS"]
                    present = [c for c in classes_order if c in wide.index] + [c for c in wide.index if c not in classes_order]
                    wide = wide.loc[present]

                    # remove ativos totalmente zerados (não aparecem na legenda)
                    wide = wide.loc[:, (wide != 0).any(axis=0)]
                    if wide.shape[1] == 0:
                        st.info("Sem ativos com CoVaR diferente de zero para plotar.")
                    else:
                        col_left, col_right = st.columns([7, 3])

                        # ---------------- barras empilhadas: Classe × (ativos) ----------------
                        with col_left:
                            fig = go.Figure()

                            # range do eixo X com base na soma dos positivos/negativos por classe
                            pos_max = float(wide.clip(lower=0).sum(axis=1).max())
                            neg_min = float(wide.clip(upper=0).sum(axis=1).min())

                            # cada trace = 1 ativo empilhado por classe
                            # (barmode="relative" empilha positivos e negativos)
                            threshold = np.nanpercentile(np.abs(wide.values), 75) if wide.size else 0.0
                            for ativo in wide.columns:
                                x = wide[ativo]
                                if (x == 0).all():
                                    continue
                                fig.add_trace(go.Bar(
                                    y=wide.index,
                                    x=x,
                                    orientation="h",
                                    name=_short(ativo, 22),
                                    customdata=np.array([ativo]*len(x)),
                                    hovertemplate="<b>%{y}</b><br>Ativo: %{customdata}<br>CoVaR: %{x:,.2f}%<extra></extra>",
                                    text=[f"{v:.2f}%" if abs(v) >= threshold else "" for v in x],
                                    textposition="outside",
                                    cliponaxis=False,
                                ))

                            fig.update_layout(
                                barmode="relative",
                                margin=dict(l=20, r=20, t=10, b=10),
                                xaxis=dict(title="CoVaR (%)", tickformat=",.2f", showgrid=True, gridcolor="#F3F4F6"),
                                yaxis=dict(title=""),
                                legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", title="Ativos"),
                                plot_bgcolor="white",
                                shapes=[dict(type="line", xref="x", x0=0, x1=0, yref="paper", y0=0, y1=1,
                                            line=dict(width=1, dash="dot", color="#9CA3AF"))],
                            )
                            pad = 0.12 * max(1.0, abs(pos_max), abs(neg_min))
                            fig.update_xaxes(range=[neg_min - pad, pos_max + pad])

                            st.plotly_chart(fig, use_container_width=True)

                        # ---------------- donut: distribuição (apenas positivos) ----------------
                        # ── CoVaR (pizza) ───────────────────────────────────────────────
                        with col_right:
                            df_norm = normalize_topN(
                                dict(zip(df_cv["Ativo"], df_cv["CoVaR_bps"])),
                                top_n=8, use_abs=False, only_positive=True
                            )
                            if df_norm.empty:
                                st.info("Sem CoVaR positivo para normalizar.")
                            else:
                                fig_p = go.Figure(go.Pie(
                                    labels=df_norm["label_short"],
                                    values=df_norm["pct"],
                                    hole=0.55, sort=False, direction="clockwise",
                                    textinfo="percent+label",
                                    textposition="inside",
                                    insidetextorientation="radial"
                                ))
                                fig_p.update_traces(textfont=dict(size=12))
                                fig_p.update_layout(
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    showlegend=False,
                                    height=PIE_HEIGHT
                                )
                                st.plotly_chart(fig_p, use_container_width=True)
                                st.caption("Distribuição do CoVaR (apenas positivos): Top-8 + “Outros”.")

        with colllmeio:
            # Adicionar linha vertical
            st.html(
                '''
                        <div class="divider-vertical-lines"></div>
                        <style>
                            .divider-vertical-lines {
                                border-left: 2px solid rgba(49, 51, 63, 0.2);
                                height: 40vh;
                                margin: auto;
                            }
                            @media (max-width: 768px) {
                                .divider-vertical-lines {
                                    display: none;
                                }
                            }
                        </style>
                        '''
            )
        #with colll1:
        # ========================== Histórico empilhado (NOVO) ==========================
        with COL2:
            st.subheader("Histórico de DV01 & CoVaR")
            st.subheader("Em desenvolvimento")
        opt_freq = "Diária"
        weekly = (opt_freq == "Semanal")

        # janelas: use o mesmo 'common' do app (período filtrado)
        dates_hist = pd.DatetimeIndex(common)
        dates_sig = (int(dates_hist[0].value), int(dates_hist[-1].value), int(len(dates_hist)))

        covar_tot_rs = capital0 * 0.01

        #Gerar bundle
    
        # DV01 normalizado (|R$|) — histórico
        df_hist_dv = build_history_normalized(
            dates_hist, kind="dv01", top_n=8, weekly=True, only_positive=False, window=126, alpha=0.05, covar_tot_rs=covar_tot_rs
        )
        #st.write(df_hist_dv)

        # CoVaR normalizado (apenas positivos) — histórico
        df_hist_cv = build_history_normalized(
            dates_hist, kind="covar", top_n=8, weekly=True, only_positive=True, window=126, alpha=0.05, covar_tot_rs=covar_tot_rs
        )
        #st.write(df_hist_cv)

        #colH1, colH2 = st.columns(2)

        #with colH1:
        import plotly.express as px
      

        # Mapeamento de ativos para estratégias
        estrategias = {
            "Juros nominais": ['DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30', 'DI_31', 'DI_32', 'DI_33', 'DI_35'],
            "Juros reais": ['DAP26', 'DAP27', 'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40', 'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60'],
            "Moeda": ['WDO1'],
            "Juros US": ['TREASURY']
        }

        # Criar um mapeamento inverso de ativo -> estratégia
        ativos_para_estrategia = {ativo: estrategia for estrategia, ativos in estrategias.items() for ativo in ativos}

        # =========================================================================================================================
        # Geração dos gráficos
        # =========================================================================================================================

        # DV01 normalizado por estratégia (área empilhada)
        if df_hist_dv.empty:
            st.info("Sem histórico suficiente para DV01.")
        else:
            with COL2:
                st.caption("DV01 por estratégia (área empilhada)")
                # 1. Agrupar os dados por estratégia (reutilizando a lógica)
                existing_cols = [col for col in df_hist_dv.columns if col in ativos_para_estrategia]
                df_hist_dv_filtrado = df_hist_dv[existing_cols]
                mapper = {col: ativos_para_estrategia[col] for col in existing_cols}
                df_hist_dv_estrategia = df_hist_dv_filtrado.groupby(by=mapper, axis=1).sum()

                # 2. Normalizar os dados agrupados
                #df_hist_dv_normalized = df_hist_dv_estrategia.divide(df_hist_dv_estrategia.sum(axis=1), axis=0)
                df_hist_dv_normalized = df_hist_dv_estrategia

                # 3. Plotar o gráfico de área empilhada
                fig_area = px.area(df_hist_dv_normalized, x=df_hist_dv_normalized.index, y=df_hist_dv_normalized.columns,
                                labels={'value': 'Proporção', 'variable': 'Estratégia'},
                                title="DV01 Normalizado por Estratégia")
                fig_area.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
                # Limitar o eixo y a 6% (0.06)
                fig_area.update_yaxes(range=[0, 0.06], tickformat=".0%", title="Proporção de DV01")
                fig_area.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Share: %{y:.2%}<extra></extra>")

                # Exibir o gráfico
                st.plotly_chart(fig_area, use_container_width=True)


        # CoVaR normalizado por estratégia (apenas positivos) — área empilhada
        if df_hist_cv.empty:
            st.info("Sem histórico suficiente para CoVaR.")
        else:
            with colll2:
                st.caption("CoVaR por estratégia (apenas positivos) — área empilhada")
                # 1. Agrupar os dados por estratégia (reutilizando a lógica)
                df_hist_cv_positive = df_hist_cv.clip(lower=0)
                existing_cols = [col for col in df_hist_cv_positive.columns if col in ativos_para_estrategia]
                df_hist_cv_filtrado = df_hist_cv_positive[existing_cols]
                mapper = {col: ativos_para_estrategia[col] for col in existing_cols}
                df_hist_cv_estrategia = df_hist_cv_filtrado.groupby(by=mapper, axis=1).sum()
                
                # 2. Normalizar os dados agrupados
                #df_hist_cv_normalized = df_hist_cv_estrategia.divide(df_hist_cv_estrategia.sum(axis=1), axis=0)
                df_hist_cv_normalized = df_hist_cv_estrategia
                # 3. Plotar o gráfico de área empilhada
                fig_area2 = px.area(df_hist_cv_normalized, x=df_hist_cv_normalized.index, y=df_hist_cv_normalized.columns,
                                    labels={'value': 'Proporção', 'variable': 'Estratégia'},
                                    title="CoVaR Normalizado por Estratégia (Positivos)")
                fig_area2.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
                fig_area2.update_yaxes(range=[0, 1], tickformat=".0%", title="Proporção de CoVaR")
                fig_area2.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Share: %{y:.2%}<extra></extra>")
                st.plotly_chart(fig_area2, use_container_width=True)
        

                    

    #with tab_fundos:
    #    df_contratos_2 = read_atual_contratos()
    #    for col in df_contratos_2.columns:
    #        df_contratos_2.rename(columns={col: f'Contratos {col}'}, inplace=True)
#
    #    df_contratos_2 = df_contratos_2.apply(pd.to_numeric, errors='coerce')
    #    lista_fundos = df_contratos_2.index.tolist()
    #    lista_fundos = [str(x)
    #                    for x in df_contratos_2.index.tolist() if str(x) != 'Total']
    #    #assets, df_contratos, fundos
    #    #calcular_metricas_de_fundo2(default_assets, df_contratos_2, lista_fundos)
    #    calcular_metricas_de_fundo3(default_assets, df_contratos_2, lista_fundos)
#
#
    #    #d6.metric("Stress DV01 (R$)", f"{risco['Stress DV01 (R$)']:,.2f}")



        
    
    

# ==========================================================
#   FUNÇÃO DA PÁGINA 1 (Dashboard Principal)
# ==========================================================
def main_page():
    st.title("Dashboard de Análise de Risco de Portfólio")
    atualizar_base_fundos()
    att_parquet_supabase()
    file_pl = "Dados/pl_fundos.parquet"
    df_pl = pd.read_parquet(file_pl)
    file_bbg = "Dados/BBG - ECO DASH.xlsx"
    

    # Dicionário de pesos fixo (pode-se tornar dinâmico no futuro)
    dict_pesos = {
        'GLOBAL BONDS': 4,
        'HORIZONTE': 1,
        'JERA2026': 1,
        'REAL FIM': 1,
        'BH FIRF INFRA': 1,
        'BORDEAUX INFRA': 1,
        'TOPAZIO INFRA': 1,
        'MANACA INFRA FIRF': 1,
        'AF DEB INCENTIVADAS': 3
    }

    Weights = list(dict_pesos.values())

    st.sidebar.write("## Pesos dos Fundos")
    st.sidebar.write("Caso queira, defina os pesos de cada fundo:")
    fundos = st.sidebar.multiselect("Selecione os Fundos:",
                                    list(dict_pesos.keys()),
                                    None)
    if fundos:
        for fundo in fundos:
            peso = st.sidebar.number_input(
                f"Peso para {fundo}:", min_value=0.0, value=1.0, step=0.1
            )
            dict_pesos[fundo] = peso
    Weights = list(dict_pesos.values())

    df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio(
        df_pl, Weights)
    
    # guarda no estado para não refazer se o usuário mudar só filtros visuais
    st.session_state.setdefault("df_pl_processado", df_pl_processado)
    st.session_state.setdefault("df_precos_ajustados_base", None)

    df = pd.read_parquet('Dados/df_inicial.parquet')

    default_assets, quantidade_inicial, portifolio_default = processar_dados_port()
    st.sidebar.write("## OPÇÕES DO DASHBOARD")
    opti = st.sidebar.radio("Escolha uma opção:", [
                            "Ver Portfólio", "Adicionar Ativos", "Remover Ativos", "Simular Cota"])
    if opti == "Adicionar Ativos":
        st.sidebar.write("## Ativos do Portfólio")

        default_or_not = st.sidebar.checkbox(
            "Usar ativos portfólio", value=False)

        if default_or_not:
            assets = st.sidebar.multiselect("Selecione os ativos:",
                                            list(df.columns),
                                            default_assets)
        else:
            assets = st.sidebar.multiselect("Selecione os ativos:",
                                            list(df.columns),
                                            'WDO1')

        if len(assets) > 0:
            df_precos, df_completo = load_and_process_excel(df, assets)
            df_retorno = process_returns(df_completo, assets)
            var_ativos = var_not_parametric(df_retorno).abs()
            df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)

            # ---------------------
            # ENTRADA DE QUANTIDADES
            # ---------------------
            qtd_input = []
            quantidade_nomes = {}
            # Função para obter o último weekday

            last_weekday = get_last_weekday()
            df_b3_fechamento = load_b3_prices()
            ultimo_dia_dados_b3 = df_b3_fechamento.columns[-1]
            ultimo_dia_dados_b3 = datetime.datetime.strptime(
                ultimo_dia_dados_b3, "%Y-%m-%d")

            data_compra_todos = st.sidebar.date_input(
                "Dia de Compra dos Ativos:", value=ultimo_dia_dados_b3, max_value=ultimo_dia_dados_b3)

            # Conferir se a data escolhida esta dentre as colunas do df_b3_fechamento
            data_compra_todos2 = str(data_compra_todos)
            if data_compra_todos2 not in df_b3_fechamento.columns:
                st.sidebar.error(
                    f"Data de compra inválida! Possível final de semana ou feriado.")
                st.stop()
            # Converter datas disponíveis para lista
            # Criar o selectbox
            st.html(
                '''
                <style>
                div[data-testid="stDateInput"] input {
                    color: black; /* Define o texto */
                                                    }
                
                </style>   
        
                '''
            )
            # Garante que "quantidade_inicial" esteja em st.session_state

            data_compra_todos = data_compra_todos.strftime("%Y-%m-%d")
            precos_user = {}

            for col in df_precos_ajustados.index:
                # Verifica se existe valor em st.session_state["quantidade_inicial"] para a ação col
                # Número de contratos
                val = st.sidebar.number_input(
                    f"Quantidade para {col}:",
                    min_value=-10000,
                    value=1,
                    step=1
                )
                # Preço (você pode ou não persistir em session_state)
                precos_user[col] = st.sidebar.number_input(
                    f"Preço de {col}:",
                    min_value=0.0,
                    value=0.0,
                    step=0.5
                )

                qtd_input.append(val)
                quantidade_nomes[col] = val

            quantidade = np.array(qtd_input)

            data_compra = {}
            for ativo in assets:
                data_compra[ativo] = data_compra_todos
            # Valor do Portfólio (soma simples)
            vp = df_precos_ajustados['Valor Fechamento'] * abs(quantidade)
            vp_soma = vp.sum()
            # Pesos (p/ cálculo de VaR, etc.)
            pesos = quantidade * \
                df_precos_ajustados['Valor Fechamento'] / vp_soma
            df_returns_portifolio = df_retorno * pesos.values
            df_returns_portifolio['Portifolio'] = df_returns_portifolio.sum(
                axis=1)

            # VaR
            var_port = var_not_parametric(df_returns_portifolio['Portifolio'])
            var_port = abs(var_port)
            var_port_dinheiro = vp_soma * var_port

            st.sidebar.write("---")
            st.sidebar.write("## Limite do Var do Portfólio")
            var_bps = st.sidebar.slider(
                "VaR do Portfólio (bps)", min_value=1.0, max_value=20.0, value=1.0, step=0.5
            )
            var_bps = var_bps / 10000
            var_din = st.sidebar.checkbox("Exibir Limite de VaR em dinheiro")
            if var_din:
                var_lim_din = st.sidebar.number_input(
                    "Valor do VaR em dinheiro:", min_value=0.0, value=float(var_port_dinheiro), step=1.0
                )
            else:
                var_limite = st.sidebar.slider(
                    "Limite para VaR gasto do Portfólio", min_value=0.1, max_value=1.0, value=1.0, step=0.01
                )

            vol_port_retornos = df_returns_portifolio['Portifolio'].std()
            vol_port_analitica = vol_port_retornos * np.sqrt(252)

            df_retorno['Portifolio'] = df_returns_portifolio['Portifolio']
            cov = df_retorno.cov()
            cov_port = cov['Portifolio'].drop('Portifolio')
            df_beta = cov_port / (vol_port_retornos**2)
            df_mvar = df_beta * var_port
            df_mvar_dinheiro = df_mvar * \
                df_precos_ajustados['Valor Fechamento']

            covar = df_mvar * pesos.values * vp_soma
            covar_perc = covar / covar.sum()

            cvar = df_retorno[df_retorno['Portifolio'] < var_not_parametric(
                df_returns_portifolio['Portifolio'])]['Portifolio'].mean()
            
            df_divone, dolar, treasury = load_and_process_divone(file_bbg, df)
            
            # --- Exemplo de cálculo de stress e DIVONE (mesmo que seu original) ---
            lista_juros_interno = [
                asset for asset in assets if 'DI' in asset]
            df_divone_juros_nominais = df_divone[lista_juros_interno]

            lista_quantidade = [quantidade_nomes[asset]
                                for asset in lista_juros_interno]
            df_divone_juros_nominais = df_divone_juros_nominais * \
                np.array(lista_quantidade)
            df_divone_juros_nominais = df_divone_juros_nominais.sum(axis=1)

            lista_juros_interno_real = [a for a in assets if ('DAP' in a) or ('NTNB' in a)]
            df_divone_juros_real = df_divone[lista_juros_interno_real]
            lista_quantidade = [quantidade_nomes[asset]
                                for asset in lista_juros_interno_real]
            df_divone_juros_real = df_divone_juros_real * \
                np.array(lista_quantidade)

            df_divone_juros_real = df_divone_juros_real.sum(axis=1)

            lista_juros_externo = [
                asset for asset in assets if 'TREASURY' in asset]
            df_divone_juros_externo = df_divone[lista_juros_externo]
            lista_quantidade = [quantidade_nomes[asset]
                                for asset in lista_juros_externo]
            df_divone_juros_externo = df_divone_juros_externo * \
                np.array(lista_quantidade)
            df_divone_juros_externo = df_divone_juros_externo.sum(axis=1)

            stress_test_juros_interno_Nominais = df_divone_juros_nominais * 100
            stress_test_juros_interno_Nominais_percent = stress_test_juros_interno_Nominais / \
                soma_pl_sem_pesos * 10000

            stress_test_juros_interno_Reais = df_divone_juros_real * 50
            stress_test_juros_interno_Reais_percent = stress_test_juros_interno_Reais / \
                soma_pl_sem_pesos * 10000

            if lista_juros_externo:
                df_divone_juros_externo = df_retorno['TREASURY'].min()
                df_divone_juros_externo = df_divone_juros_externo * \
                    np.array(lista_quantidade)
                df_divone_juros_externo = df_divone_juros_externo.sum()
                df_divone_juros_externo = abs(
                    df_divone_juros_externo) * treasury

            stress_test_juros_externo = df_divone_juros_externo * 100
            stress_test_juros_externo_percent = stress_test_juros_externo / \
                soma_pl_sem_pesos * 10000

            lista_dolar = [asset for asset in assets if 'WDO1' in asset]
            if lista_dolar:
                quantidade_dolar = quantidade_nomes[lista_dolar[0]]
                stress_dolar = quantidade_dolar * dolar * 0.02
                df_divone_dolar = df_retorno['WDO1'].min()
                df_divone_dolar = df_divone_dolar * quantidade_dolar
                df_divone_dolar = abs(df_divone_dolar) * dolar

            else:
                stress_dolar = 0
                df_divone_dolar = 0

            stress_dolar_percent = stress_dolar / soma_pl_sem_pesos * 10000
            df_stress_div01 = pd.DataFrame({
                'DIV01': [
                    f"R${df_divone_juros_nominais.iloc[0]:,.2f}",
                    f"R${df_divone_juros_real.iloc[0]:,.2f}",
                    f"R${df_divone_juros_externo:,.2f}" if lista_juros_externo else f"R${df_divone_juros_externo.iloc[0]:,.2f}",
                    f'R${df_divone_dolar:,.2f}'
                ],
                'Stress (R$)': [
                    f"R${stress_test_juros_interno_Nominais['FUT_TICK_VAL']:,.2f}",
                    f"R${stress_test_juros_interno_Reais['FUT_TICK_VAL']:,.2f}",
                    f"R${stress_test_juros_externo:,.2f}" if lista_juros_externo else f"R${stress_test_juros_externo['FUT_TICK_VAL']:,.2f}",
                    # teste
                    f"R${stress_dolar:.2f}"
                ],
                'Stress (bps)': [
                    f"{stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']:,.2f}bps",
                    f"{stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']:,.2f}bps",
                    f"{stress_test_juros_externo_percent:,.2f}bps" if lista_juros_externo else f"{stress_test_juros_externo_percent['FUT_TICK_VAL']:,.2f}bps",
                    f"{stress_dolar_percent:,.2f}bps"
                ]
            }, index=['Juros Nominais Brasil', 'Juros Reais Brasil', 'Juros US', 'Moedas'])

            sum_row = pd.DataFrame({
                'DIV01': [f"R${df_divone_juros_nominais.iloc[0] + df_divone_juros_real[0] + df_divone_juros_externo + df_divone_dolar:,.2f}" if lista_juros_externo else f"R${df_divone_juros_nominais.iloc[0] + df_divone_juros_real.iloc[0] + df_divone_juros_externo.iloc[0] + df_divone_dolar:,.2f}"],
                'Stress (R$)': [f"R${stress_test_juros_interno_Nominais['FUT_TICK_VAL'] + stress_test_juros_interno_Reais['FUT_TICK_VAL'] + stress_test_juros_externo + stress_dolar:,.2f}"] if lista_juros_externo else [f"R${stress_test_juros_interno_Nominais['FUT_TICK_VAL'] + stress_test_juros_interno_Reais['FUT_TICK_VAL'] + stress_test_juros_externo['FUT_TICK_VAL'] + stress_dolar:,.2f}"],
                'Stress (bps)': [f"{stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL'] + stress_test_juros_interno_Reais_percent['FUT_TICK_VAL'] + stress_test_juros_externo_percent + stress_dolar_percent:,.2f}bps" if lista_juros_externo else f"{stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL'] + stress_test_juros_interno_Reais_percent['FUT_TICK_VAL'] + stress_test_juros_externo_percent['FUT_TICK_VAL'] + stress_dolar_percent:,.2f}bps"]
            }, index=['Total'])
            df_stress_div01 = pd.concat([df_stress_div01, sum_row])

            df_precos_ajustados = calculate_portfolio_values(
                df_precos_ajustados, df_pl_processado, var_bps)

            df_pl_processado = calculate_contracts_per_fund(
                df_pl_processado, df_precos_ajustados)

            # st.session_state["posicoes_temp"] = quantidade_nomes
            # st.session_state["ativos_temp"] = list(quantidade_nomes.keys())

            # # Botão com callback (1 clique = troca de página)
            # st.button(
            #     "Adicionar ao Portifólio",
            #     on_click=switch_to_page2,
            #     key="go_page2"
            # )

            # ------------------------------------------------
            #   TABELA DF_PL (FILTROS) & Contratos por Fundo
            # ------------------------------------------------

            # Formatações
            df_precos_ajustados['Valor Fechamento'] = df_precos_ajustados['Valor Fechamento'].apply(
                lambda x: f"R${x:,.0f}")
            df_precos_ajustados['Valor Fechamento Ajustado pelo Var'] = df_precos_ajustados['Valor Fechamento Ajustado pelo Var'].apply(
                lambda x: f"R${x:,.0f}")
            df_precos_ajustados['Santander'] = df_precos_ajustados['Santander'].apply(
                lambda x: f"{x:.0f}")
            df_precos_ajustados['BTG'] = df_precos_ajustados['BTG'].apply(
                lambda x: f"{x:.0f}")
            df_precos_ajustados['Valor Total'] = df_precos_ajustados['Valor Total'].apply(
                lambda x: f"{x:.0f}")

            df_precos_ajustados['Quantidade'] = quantidade

            df_precos_ajustados_copy = df_precos_ajustados.copy()

            df_contratos_2 = read_atual_contratos()
            # Adicionar os contratos já existentes df_precos_ajustados
            for col in df_contratos_2.columns:
                if col in df_precos_ajustados_copy.index:
                    df_precos_ajustados_copy.loc[col, 'Quantidade'] = df_precos_ajustados_copy.loc[col,
                                                                                                   'Quantidade'] + df_contratos_2.loc['Total', col]

            df_pl_processado_input = calculate_contracts_per_fund_input(
                df_pl_processado, df_precos_ajustados_copy)

            # Cria uma cópia para exibição final
            df_pl_processado_print = df_pl_processado_input.copy()
            sum_row = df_pl_processado_print.select_dtypes(
                include='number').sum()
            sum_row['Fundos/Carteiras Adm'] = 'Total'
            sum_row['Adm'] = ''
            df_pl_processado_print = pd.concat(
                [df_pl_processado_print, sum_row.to_frame().T], ignore_index=True)

            df_pl_processado_print.set_index(
                'Fundos/Carteiras Adm', inplace=True)
            df_pl_processado_print = df_pl_processado_print.drop(
                ['PL', 'Adm'], axis=1)

            # Checkboxes para exibir colunas
            default_columns = ['Adm', 'Weights']
            for asset in assets:
                col_name = f'Contratos {asset}'
                if col_name in df_pl_processado_print.columns:
                    default_columns.append(col_name)

            # Ajusta para remover colunas de "Max Contratos" que não estão sendo usadas
            colunas_df_processado = []
            for asset in assets:
                col_name = f'Max Contratos {asset}'
                if col_name in df_pl_processado.columns:
                    colunas_df_processado.append(col_name)
            df_copy_processado = df_pl_processado.copy()
            df_copy_processado.drop(
                colunas_df_processado, axis=1, inplace=True)

            columns_sem_fundo = df_copy_processado.columns.tolist()
            if 'Fundos/Carteiras Adm' in columns_sem_fundo:
                columns_sem_fundo.remove('Fundos/Carteiras Adm')
            if 'PL' in columns_sem_fundo:
                columns_sem_fundo.remove('PL')
            st.write("### Selecione as colunas")
            col1_, col2_, col3_ = st.columns([4, 3, 3])
            columns = []
            for i, col_name in enumerate(columns_sem_fundo):
                if i % 3 == 0:
                    with col1_:
                        if st.checkbox(col_name, value=(col_name in default_columns), key=f"check_{col_name}"):
                            columns.append(col_name)
                elif i % 3 == 1:
                    with col2_:
                        if st.checkbox(col_name, value=(col_name in default_columns), key=f"check_{col_name}"):
                            columns.append(col_name)
                else:
                    with col3_:
                        if st.checkbox(col_name, value=(col_name in default_columns), key=f"check_{col_name}"):
                            columns.append(col_name)

            coll1, coll2 = st.columns([7, 3])

            with coll2:
                st.write("### Filtrar por Adm")
                filtro_adm = []
                for adm in df_pl_processado["Adm"].unique():
                    if st.checkbox(adm, key=f"checkbox_adm_{adm}"):
                        filtro_adm.append(adm)

            with coll1:
                st.write("### Filtrar por Fundos/Carteiras")
                filtro_fundo = st.multiselect(
                    'Filtrar por Fundos/Carteiras Adm',
                    df_pl_processado["Fundos/Carteiras Adm"].unique()
                )

            filtered_df = df_pl_processado_input.copy()
            quantidade_nomes_original = quantidade_nomes.copy()
            for asset in assets:
                quantidade_nomes[asset] = df_precos_ajustados_copy.loc[asset, 'Quantidade']

            for asset in assets:
                # Verifica se a soma dos contratos arredondados está correta
                soma_atual = filtered_df[f'Contratos {asset}'].apply(
                    lambda x: round(x)).sum()

                if soma_atual == quantidade_nomes[asset]:
                    filtered_df[f'Contratos {asset}'] = filtered_df[f'Contratos {asset}'].apply(
                        lambda x: round(x))
                else:
                    # Calcula o número de contratos que faltam ou excedem
                    diferencas = quantidade_nomes[asset] - soma_atual

                    # Calcula o peso relativo e os resíduos
                    filtered_df['Peso Relativo'] = (
                        filtered_df[f'Contratos {asset}'] /
                        filtered_df[f'Contratos {asset}'].sum()
                    )
                    filtered_df['Contratos Proporcionais'] = (
                        filtered_df['Peso Relativo'] * quantidade_nomes[asset]
                    )
                    filtered_df['Contratos Inteiros'] = filtered_df['Contratos Proporcionais'].apply(
                        np.floor).astype(int)
                    filtered_df['Resíduo'] = filtered_df['Contratos Proporcionais'] - \
                        filtered_df['Contratos Inteiros']

                    # Ajusta contratos restantes com base nos pesos e resíduos
                    contratos_restantes = quantidade_nomes[asset] - \
                        filtered_df['Contratos Inteiros'].sum()

                    if contratos_restantes != 0:
                        # Ordena para priorizar maior resíduo e maior peso relativo
                        ordem_ajuste = filtered_df.sort_values(
                            by=['Resíduo', 'Peso Relativo'], ascending=[False, False]).index if contratos_restantes > 0 else \
                            filtered_df.sort_values(
                                by=['Resíduo', 'Peso Relativo'], ascending=[True, True]).index

                        for i in range(abs(contratos_restantes)):
                            idx_to_adjust = ordem_ajuste[i]
                            filtered_df.at[idx_to_adjust,
                                           'Contratos Inteiros'] += 1 if contratos_restantes > 0 else -1

                    # Atualiza a coluna final com os contratos ajustados
                    filtered_df[f'Contratos {asset}'] = filtered_df['Contratos Inteiros']
                    # Remove colunas auxiliares
                    filtered_df.drop(columns=[
                        'Peso Relativo', 'Contratos Proporcionais', 'Contratos Inteiros', 'Resíduo'], inplace=True)
                    # Criar tabela editável

            for asset in assets:
                for idx, row in filtered_df.iterrows():
                    fundo = row['Fundos/Carteiras Adm']
                    if asset in df_contratos_2.columns:
                        valor_contrato = df_contratos_2.loc[fundo, asset]
                    else:
                        valor_contrato = 0
                    filtered_df.at[idx, f'Contratos {asset}'] -= valor_contrato

            quantidade_nomes = quantidade_nomes_original.copy()
            column_config = {}
            for col in filtered_df.columns:
                # Bloquear edição das outras colunas
                if not col.startswith("Contratos"):
                    column_config[col] = st.column_config.TextColumn(
                        col, disabled=True)
                else:
                    column_config[col] = st.column_config.NumberColumn(
                        col, step=1)  # Permitir edição

            # # OPERAÇÃO AQUI
            # for asset in assets:
            #     # Verifica se a soma dos contratos arredondados está correta
            #     if filtered_df[f'Contratos {asset}'].apply(lambda x: round(x)).sum() == quantidade_nomes[asset]:
            #         st.write(f"Quantidade de {asset} está correta.")
            #     else:
            #         # Calcula o desvio (diferença entre a soma atual e a quantidade desejada)
            #         diferencas = quantidade_nomes[asset] - filtered_df[f'Contratos {asset}'].apply(
            #             lambda x: round(x)).sum()

            #         # Itera para ajustar os contratos até que a diferença seja 0
            #         while diferencas != 0:
            #             # Calcula o erro relativo para cada fundo (linha)
            #             filtered_df['Erro Relativo'] = (
            #                 filtered_df[f'Contratos {asset}'] -
            #                 filtered_df[f'Contratos {asset}'].apply(
            #                     lambda x: round(x))
            #             )

            #             # Seleciona o índice do fundo com o maior ou menor erro relativo
            #             if diferencas > 0:  # Ajustar para cima
            #                 idx_to_adjust = filtered_df['Erro Relativo'].idxmax(
            #                 )
            #                 filtered_df.at[idx_to_adjust,
            #                                f'Contratos {asset}'] += 1
            #                 diferencas -= 1
            #             else:  # Ajustar para baixo
            #                 idx_to_adjust = filtered_df['Erro Relativo'].idxmin(
            #                 )
            #                 filtered_df.at[idx_to_adjust,
            #                                f'Contratos {asset}'] -= 1
            #                 diferencas += 1

            #         # Confirma ajuste
            #         st.write(
            #             f"Quantidade de {asset} ajustada corretamente para {quantidade_nomes[asset]}.")

            if filtro_fundo:
                filtered_df = filtered_df[filtered_df["Fundos/Carteiras Adm"].isin(
                    filtro_fundo)]
            if filtro_adm:
                filtered_df = filtered_df[filtered_df["Adm"].isin(filtro_adm)]

            filtered_df.index = filtered_df['Fundos/Carteiras Adm']

            filtered_df_2 = filtered_df.copy()
            col_max = [
                col for col in filtered_df_2.columns if col.startswith("Max")]
            filtered_df_2 = filtered_df_2.drop(
                ['PL', 'Adm', 'Weights', 'PL_atualizado'], axis=1)
            filtered_df_2 = filtered_df_2.drop(col_max, axis=1)
            filtered_df_2 = filtered_df_2.set_index('Fundos/Carteiras Adm')
            cool1, cool2, cool3 = st.columns([3.8, 0.2, 6.0])
            with cool1:
                st.write('### Edição de Contratos')
                df_editado = st.data_editor(
                    filtered_df_2,
                    hide_index=False,  # Mostrar o índice
                    num_rows="fixed",  # Impede adicionar ou remover linhas
                    key="data_editor",
                    column_config=column_config)

            quantidade_nova_assets = {}
            for col in df_editado.columns:
                if 'Contratos' in col:
                    quantidade_nova_assets[col.split(
                        ' ')[1]] = df_editado[col].sum()

            if quantidade_nova_assets != quantidade_nomes:
                quantidade_nomes = quantidade_nova_assets.copy()

            for idx, row in filtered_df.iterrows():
                for asset in assets:
                    fundo = row['Fundos/Carteiras Adm']
                    filtered_df.at[idx,
                                   f'Contratos {asset}'] = df_editado.loc[fundo, f'Contratos {asset}']
                    if asset not in quantidade_nova_assets:
                        quantidade_nova_assets[asset] = 0

                    quantidade_nova_assets[asset] += df_editado.loc[fundo,
                                                                    f'Contratos {asset}']
                    # Transformar linha em número
                    filtered_df.at[idx, f'Contratos {asset}'] = int(
                        filtered_df.at[idx, f'Contratos {asset}'])

            ###################### Atualização de quantidade_inicial ######################

            ############################################## DEU ERRADO ##############################################

            ############################################## DEU ERRADO ##############################################

            # Adicionar sumrow
            # Resetar indice para poder adicionar a linha de soma
            # Resetar índice para poder adicionar a linha de soma
            # Resetar índice para poder adicionar a linha de soma (sem adicionar a coluna 'Fundos/Carteiras Adm' novamente)
            filtered_df = filtered_df.reset_index(drop=True)

            # Calculando a soma das colunas numéricas
            sum_row = filtered_df.select_dtypes(include='number').sum()

            # Adicionando a linha de soma e atribuindo o nome 'Total' à coluna 'Fundos/Carteiras Adm'
            sum_row['Fundos/Carteiras Adm'] = 'Total'
            sum_row['Adm'] = ''

            # Adicionando a linha de soma no final do DataFrame
            filtered_df = pd.concat(
                [filtered_df, sum_row.to_frame().T], ignore_index=True)

            # Restaurar a coluna 'Fundos/Carteiras Adm' como índice
            filtered_df.set_index('Fundos/Carteiras Adm', inplace=True)

            # formatação
            for col_ in filtered_df.columns:
                if col_.startswith("Contratos"):
                    filtered_df[col_] = filtered_df[col_].apply(
                        lambda x: f"{x:.0f}")
                elif col_ == 'PL_atualizado':
                    filtered_df[col_] = filtered_df[col_].apply(
                        lambda x: f"R${x:,.2f}")
                elif col_ == 'PL':
                    filtered_df[col_] = filtered_df[col_].apply(
                        lambda x: f"R${x:,.0f}")
                elif col_ == 'Weights':
                    filtered_df[col_] = filtered_df[col_].apply(
                        lambda x: f"{x:.0f}")

            st.write("OBS: Os contratos estão arrendodandos para inteiros.")

            with cool2:
                # Adicionar linha vertical
                st.html(
                    '''
                            <div class="divider-vertical-lines"></div>
                            <style>
                                .divider-vertical-lines {
                                    border-left: 2px solid rgba(49, 51, 63, 0.2);
                                    height: 200vh;
                                    margin: auto;
                                }
                                @media (max-width: 768px) {
                                    .divider-vertical-lines {
                                        display: none;
                                    }
                                }
                            </style>
                            '''
                )

            df_port, key, soma_pl_sem_pesos = checkar_portifolio(
                assets, quantidade_nomes, precos_user, data_compra, filtered_df)
            
            if key == True:
                atualizar_parquet_fundos(
                    filtered_df, data_compra_todos, df_port, quantidade_nomes)
            with cool3:
                st.write("### Portfólio Atualizado")
                st.table(filtered_df[columns])
            # Formata tabela final
            for col_ in df_pl_processado_print.columns:
                if col_ != 'PL_atualizado':
                    df_pl_processado_print[col_] = df_pl_processado_print[col_].apply(
                        lambda x: f"{x:.2f}")
                else:
                    df_pl_processado_print[col_] = df_pl_processado_print[col_].apply(
                        lambda x: f"R${x:,.2f}")

            st.write("### Quantidade Máxima de Contratos por Adm")
            df_precos_plot = df_precos_ajustados.drop(
                ['Quantidade', 'Valor Fechamento', 'Valor Fechamento Ajustado pelo Var'], axis=1
            )
            st.table(df_precos_plot)
            st.write("### Quantidade Máxima de Contratos por Fundo")
            st.table(df_pl_processado_print)
            st.write("---")
            # -------------------------------------------------
            # Armazena dados em chaves "temporárias" p/ callback
            # -------------------------------------------------

        else:
            st.write("Nenhum Ativo selecionado.")

    elif opti == "Ver Portfólio":
        with st.spinner("Carregando dados do portfólio…"):
            # Agrupamento corrigido
            # ------------------------------------------------------------------
            # 1) helper para média ponderada
            # ------------------------------------------------------------------
            def wavg(df, value_col, weight_col="Quantidade"):
                w = df[weight_col].abs()            # usa abs() se quiser média “bruta”
                return np.nan if w.sum() == 0 else (df[value_col] * w).sum() / w.sum()

            # ------------------------------------------------------------------
            # 2) consolida por ativo
            # ------------------------------------------------------------------
            df_portifolio_default = (
                portifolio_default
                    .groupby("Ativo", as_index=False)
                    .apply(lambda g: pd.Series({
                        "Quantidade"           : g["Quantidade"].sum(),
                        "Preço de Compra"      : wavg(g, "Preço de Compra"),
                        "Preço de Ajuste Atual": wavg(g, "Preço de Ajuste Atual"),
                        "Rendimento"           : g["Rendimento"].sum()
                    }))
            )

            # descarta posições zeradas
            df_portifolio_default = df_portifolio_default[df_portifolio_default["Quantidade"] != 0]

            # ------------------------------------------------------------------
            # 3) linha Total • as colunas-preço usam média ponderada
            # ------------------------------------------------------------------
            q_tot  = df_portifolio_default["Quantidade"].sum()
            vlc_pc = (df_portifolio_default["Quantidade"] * df_portifolio_default["Preço de Compra"]).sum()
            vlc_pa = (df_portifolio_default["Quantidade"] * df_portifolio_default["Preço de Ajuste Atual"]).sum()

            total_row = {
                "Ativo"               : "Total",
                "Quantidade"          : q_tot,
                "Preço de Compra"      : np.nan if q_tot == 0 else vlc_pc / q_tot,
                "Preço de Ajuste Atual": np.nan if q_tot == 0 else vlc_pa / q_tot,
                "Rendimento"          : df_portifolio_default["Rendimento"].sum()
            }

            df_portifolio_default = (
                pd.concat([df_portifolio_default, pd.DataFrame([total_row])], ignore_index=True)
                .set_index("Ativo")
            )

            # ------------------------------------------------------------------
            # 4) formatação/renome igual ao seu código -------------------------
            df_fmt = df_portifolio_default.copy()
            df_fmt["Quantidade"]            = fmt_int(df_fmt["Quantidade"])
            df_fmt["Preço de Compra"]       = fmt_money(df_fmt["Preço de Compra"])
            df_fmt["Preço de Ajuste Atual"] = fmt_money(df_fmt["Preço de Ajuste Atual"])
            df_fmt["Rendimento"]            = fmt_money(df_fmt["Rendimento"])

            df_fmt = df_fmt.rename(columns={
                "Quantidade"           : "Quantidade Total",
                "Preço de Compra"      : "Preço de Compra Médio",
                "Preço de Ajuste Atual": "Preço de Ajuste Atual Médio",
                "Rendimento"           : "P&L"
            })

            # limpa linhas/colunas a gosto, depois…
            df_fmt2 = df_fmt.copy()
            df_fmt2 = df_fmt2.drop(columns=["P&L"])
            ###############
            ###############
            #TIRANDO A TABELA DE QUANTIDADE

            #st.table(df_fmt2)
            st.sidebar.write("---")
            st.sidebar.write("## Limite do Var do Portifólio")
            var_bps = st.sidebar.slider(
                "VaR do Portfólio (bps)", min_value=1.0, max_value=20.0, value=1.0, step=0.5
            )
            var_bps = var_bps / 10000
            var_din = st.sidebar.checkbox("Exibir Limite de VaR em dinheiro")
            if var_din:
                var_lim_din = st.sidebar.number_input(
                    "Valor do VaR em dinheiro:", min_value=0.0, value=float(var_port_dinheiro), step=1.0
                )
            else:
                var_limite = st.sidebar.slider(
                    "Limite para VaR gasto do Portfólio", min_value=0.1, max_value=1.0, value=1.0, step=0.01
                )
            
            df_b3_fechamento = load_b3_prices()
            ultimo_dia_dados_b3 = df_b3_fechamento.columns[-1]
            ultimo_dia_dados_b3 = datetime.datetime.strptime(
                ultimo_dia_dados_b3, "%Y-%m-%d")

            #st.write(
            #    f"OBS: O preço de compra é o preço médio de compra do ativo -- Atualizado em {ultimo_dia_dados_b3.strftime('%d/%m/%Y')}")
            #st.write("---")
            # print(df_portifolio_default_copy)
            quantidade = []
            df_contratos = read_atual_contratos_cached()

            file_pl = "Dados/pl_fundos.parquet"
            df_pl = pd.read_parquet(file_pl)
            file_bbg = "Dados/BBG - ECO DASH.xlsx"

            # Dicionário de pesos fixo (pode-se tornar dinâmico no futuro)
            dict_pesos = {
                'GLOBAL BONDS': 4,
                'HORIZONTE': 1,
                'JERA2026': 1,
                'REAL FIM': 1,
                'BH FIRF INFRA': 1,
                'BORDEAUX INFRA': 1,
                'TOPAZIO INFRA': 1,
                'MANACA INFRA FIRF': 1,
                'AF DEB INCENTIVADAS': 3
            }
            # Zerar os pesos de fundos que não tem contratos
            for idx, row in df_contratos.iterrows():
                if idx == 'Total':
                    continue
                else:
                    fundo = idx
                    check = 0
                    for asset in default_assets:
                        if int(row[asset]) != 0:
                            check = 1
                    if check == 0:
                        dict_pesos[fundo] = 0

            Weights = list(dict_pesos.values())
            df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio(
                df_pl, Weights)
            for asset in default_assets:
                quantidade.append(quantidade_inicial[asset])
            # Ver se todos os elementos da lista quantidade são zeros
            if all(elem == 0 for elem in quantidade):
                sta = False
            else:
                sta = True
            if default_assets and sta:
                df_precos, df_completo = load_and_process_excel(df, default_assets)
                df_retorno = process_returns(df_completo, default_assets)
                var_ativos = var_not_parametric(df_retorno).abs()
                if st.session_state["df_precos_ajustados_base"] is None:
                    st.session_state["df_precos_ajustados_base"] = adjust_prices_with_var(df_precos, var_ativos)
                df_precos_ajustados = st.session_state["df_precos_ajustados_base"].copy()
                df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)
                quantidade = np.array(quantidade)
                df_contratos_2 = read_atual_contratos_cached()
                for col in df_contratos_2.columns:
                    df_contratos_2.rename(
                        columns={col: f'Contratos {col}'}, inplace=True)
                df_contratos_2 = df_contratos_2.apply(
                    pd.to_numeric, errors='coerce')  # O mesmo para df_contratos_2

                

                # ------------------------------------------------
                #   TABELA DF_PL (FILTROS) & Contratos por Fundo
                # ------------------------------------------------
                #st.write("---")
                st.write("## Quantidade de Contratos por Fundo")
                df_precos_ajustados = calculate_portfolio_values(
                    df_precos_ajustados, df_pl_processado, var_bps)
                df_pl_processado = calculate_contracts_per_fund(
                    df_pl_processado, df_precos_ajustados)
                
                vp = df_precos_ajustados['Valor Fechamento'] * abs(quantidade)
                vp_soma = vp.sum()

                # Pesos (p/ cálculo de VaR, etc.)
                pesos = quantidade * \
                    df_precos_ajustados['Valor Fechamento'] / vp_soma
                df_returns_portifolio = df_retorno * pesos.values
                df_returns_portifolio['Portifolio'] = df_returns_portifolio.sum(
                    axis=1)

                # VaR
                var_port = var_not_parametric(df_returns_portifolio['Portifolio'])
                var_port = abs(var_port)
                var_port_dinheiro = vp_soma * var_port


                df_precos_ajustados = calculate_portfolio_values(
                    df_precos_ajustados, df_pl_processado, var_bps)
                df_pl_processado = calculate_contracts_per_fund(
                    df_pl_processado, df_precos_ajustados)

                # st.session_state["posicoes_temp"] = quantidade_inicia

                # Formatações
                df_precos_ajustados['Valor Fechamento'] = df_precos_ajustados['Valor Fechamento'].apply(
                    lambda x: f"R${x:,.0f}")
                df_precos_ajustados['Valor Fechamento Ajustado pelo Var'] = df_precos_ajustados['Valor Fechamento Ajustado pelo Var'].apply(
                    lambda x: f"R${x:,.0f}")
                df_precos_ajustados['Santander'] = df_precos_ajustados['Santander'].apply(
                    lambda x: f"{x:.0f}")
                df_precos_ajustados['BTG'] = df_precos_ajustados['BTG'].apply(
                    lambda x: f"{x:.0f}")
                df_precos_ajustados['Valor Total'] = df_precos_ajustados['Valor Total'].apply(
                    lambda x: f"{x:.0f}")

                df_precos_ajustados['Quantidade'] = quantidade
                df_pl_processado_input = calculate_contracts_per_fund_input(
                    df_pl_processado, df_precos_ajustados)

                # Cria uma cópia para exibição final
                df_pl_processado_print = df_pl_processado_input.copy()
                sum_row = df_pl_processado_print.select_dtypes(
                    include='number').sum()
                sum_row['Fundos/Carteiras Adm'] = 'Total'
                sum_row['Adm'] = ''
                df_pl_processado_print = pd.concat(
                    [df_pl_processado_print, sum_row.to_frame().T], ignore_index=True)

                df_pl_processado_print.set_index(
                    'Fundos/Carteiras Adm', inplace=True)
                df_pl_processado_print = df_pl_processado_print.drop(
                    ['PL', 'Adm'], axis=1)

                # Checkboxes para exibir colunas
                default_columns = ['Adm', 'PL_atualizado']
                for asset in default_assets:
                    col_name = f'Contratos {asset}'
                    if col_name in df_pl_processado_print.columns:
                        default_columns.append(col_name)

                # Ajusta para remover colunas de "Max Contratos" que não estão sendo usadas
                colunas_df_processado = []
                for asset in default_assets:
                    col_name = f'Max Contratos {asset}'
                    if col_name in df_pl_processado.columns:
                        colunas_df_processado.append(col_name)
                df_copy_processado = df_pl_processado.copy()
                df_copy_processado.drop(
                    colunas_df_processado, axis=1, inplace=True)

                columns_sem_fundo = df_copy_processado.columns.tolist()
                if 'Fundos/Carteiras Adm' in columns_sem_fundo:
                    columns_sem_fundo.remove('Fundos/Carteiras Adm')
                if 'PL' in columns_sem_fundo:
                    columns_sem_fundo.remove('PL')
                #
                #st.write("### Selecione as colunas")
                col1_, col2_, col3_ = st.columns([4, 3, 3])
                columns = columns_sem_fundo
                #for i, col_name in enumerate(columns_sem_fundo):
                #    if i % 3 == 0:
                #        with col1_:
                #            if st.checkbox(col_name, value=(col_name in default_columns), key=f"check_{col_name}"):
                #                columns.append(col_name)
                #    elif i % 3 == 1:
                #        with col2_:
                #            if st.checkbox(col_name, value=(col_name in default_columns), key=f"check_{col_name}"):
                #                columns.append(col_name)
                #    else:
                #        with col3_:
                #            if st.checkbox(col_name, value=(col_name in default_columns), key=f"check_{col_name}"):
                #                columns.append(col_name)
                #
                #coll1, coll2 = st.columns([7, 3])
                #
                #with coll2:
                #    st.write("### Filtrar por Adm")
                #    filtro_adm = []
                #    for adm in df_pl_processado["Adm"].unique():
                #        if st.checkbox(adm, key=f"checkbox_adm_{adm}"):
                #            filtro_adm.append(adm)
                #
                #with coll1:
                #    st.write("### Filtrar por Fundos/Carteiras")
                #    filtro_fundo = st.multiselect(
                #        'Filtrar por Fundos/Carteiras Adm',
                #        df_pl_processado["Fundos/Carteiras Adm"].unique()
                #    )
                #
                filtered_df = df_pl_processado_input.copy()

                for asset in default_assets:
                    # Verifica se a soma dos contratos arredondados está correta
                    soma_atual = filtered_df[f'Contratos {asset}'].apply(
                        lambda x: round(x)).sum()
                    if soma_atual == quantidade_inicial[asset]:
                        continue
                    else:
                        # Calcula o número de contratos que faltam ou excedem
                        diferencas = quantidade_inicial[asset] - soma_atual
                        # Calcula o peso relativo e os resíduos
                        filtered_df['Peso Relativo'] = (
                            filtered_df[f'Contratos {asset}'] /
                            filtered_df[f'Contratos {asset}'].sum()
                        )
                        filtered_df['Contratos Proporcionais'] = (
                            filtered_df['Peso Relativo'] *
                            quantidade_inicial[asset]
                        )
                        filtered_df['Contratos Inteiros'] = filtered_df['Contratos Proporcionais'].apply(
                            np.floor).astype(int)
                        filtered_df['Resíduo'] = filtered_df['Contratos Proporcionais'] - \
                            filtered_df['Contratos Inteiros']

                        # Ajusta contratos restantes com base nos pesos e resíduos
                        contratos_restantes = quantidade_inicial[asset] - \
                            filtered_df['Contratos Inteiros'].sum()

                        if contratos_restantes != 0:
                            # Ordena para priorizar maior resíduo e maior peso relativo
                            ordem_ajuste = filtered_df.sort_values(
                                by=['Resíduo', 'Peso Relativo'], ascending=[False, False]).index if contratos_restantes > 0 else \
                                filtered_df.sort_values(
                                    by=['Resíduo', 'Peso Relativo'], ascending=[True, True]).index

                            for i in range(abs(contratos_restantes)):
                                idx_to_adjust = ordem_ajuste[i]
                                filtered_df.at[idx_to_adjust,
                                            'Contratos Inteiros'] += 1 if contratos_restantes > 0 else -1

                        # Atualiza a coluna final com os contratos ajustados
                        filtered_df[f'Contratos {asset}'] = filtered_df['Contratos Inteiros']

                        # Remove colunas auxiliares
                        filtered_df.drop(columns=[
                            'Peso Relativo', 'Contratos Proporcionais', 'Contratos Inteiros', 'Resíduo'], inplace=True)

                #if filtro_fundo:
                #    filtered_df = filtered_df[filtered_df["Fundos/Carteiras Adm"].isin(
                #        filtro_fundo)]
                #if filtro_adm:
                #    filtered_df = filtered_df[filtered_df["Adm"].isin(filtro_adm)]

                sum_row = filtered_df.select_dtypes(include='number').sum()
                sum_row['Fundos/Carteiras Adm'] = 'Total'
                sum_row['Adm'] = ''
                filtered_df = pd.concat(
                    [filtered_df, sum_row.to_frame().T], ignore_index=True)

                filtered_df.index = filtered_df['Fundos/Carteiras Adm']
                df_contratos = read_atual_contratos_cached()
                # Pegar as colunas que começam com 'Contratos' e remover os espaços
                col_contratos = [
                    col for col in filtered_df.columns if col.startswith('Contratos')]
                for col in col_contratos:
                    filtered_df[col] = df_contratos[col.replace('Contratos ', '')]
                #if columns:
                    # Formatação
                for c in columns:
                    if c == 'PL_atualizado':
                        filtered_df[c] = filtered_df[c].apply(
                            lambda x: f"R${x:,.0f}")
                    else:
                        if c == 'Adm' or 'Fundos/Carteiras Adm':
                            filtered_df[c] = filtered_df[c].apply(lambda x: x)
                        else:
                            filtered_df[c] = filtered_df[c].apply(
                                lambda x: f"{x:.2f}")
                for col in filtered_df.columns:
                    if col.startswith("Contratos"):
                        filtered_df[col] = filtered_df[col].apply(
                            lambda x: f"{x:.0f}")
                        
                # ===== 1) identificar colunas de contratos e rótulo "Total" =====
                df_calc = filtered_df.copy()

                contract_cols = [c for c in df_calc.columns if str(c).startswith('Contratos')]
                non_contract_cols = [c for c in df_calc.columns if c not in contract_cols]

                # tenta achar a linha "Total" (case/espacos robusto)
                total_label = next((idx for idx in df_calc.index if str(idx).strip().lower() == 'total'), None)

                # converter contratos para numérico (sem afetar o df original)
                df_num = df_calc[contract_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

                # ===== 2) somar contratos (excluindo a linha Total, se existir) =====
                if total_label is not None:
                    sum_by_col = df_num.drop(index=total_label, errors='ignore').sum(axis=0)
                else:
                    sum_by_col = df_num.sum(axis=0)

                # ===== 3) atualizar a linha Total no df original, apenas nas colunas de contratos =====
                if total_label is not None:
                    filtered_df.loc[total_label, contract_cols] = sum_by_col.round().astype(int)

                # ===== 4) montar visão para tela: manter não-contratos + contratos com soma != 0 =====
                keep_contract_cols = [c for c in contract_cols if abs(sum_by_col.get(c, 0)) != 0]

                df_view = filtered_df[non_contract_cols + keep_contract_cols].copy()

                # renomear "Contratos XXX" -> "XXX"
                rename_map = {c: c.replace('Contratos ', '') for c in keep_contract_cols}
                df_view = df_view.rename(columns=rename_map)

                asset_cols_clean = list(rename_map.values())

                # arredondar contratos para inteiros na visão
                df_view[asset_cols_clean] = (
                    df_view[asset_cols_clean]
                    .apply(pd.to_numeric, errors='coerce')
                    .round(0)
                    .astype('Int64')              # preserva NaN, se houver
                )

                # (opcional) remover também colunas cujo total (excluindo Total) ficou 0 após arredondamento
                if total_label is not None:
                    zero_cols = [c for c in asset_cols_clean
                                if (pd.to_numeric(df_view.loc[df_view.index != total_label, c], errors='coerce')
                                    .fillna(0).sum() == 0)]
                    if zero_cols:
                        df_view = df_view.drop(columns=zero_cols)
                        asset_cols_clean = [c for c in asset_cols_clean if c not in zero_cols]

                # (opcional) ordenar ativos por soma absoluta decrescente
                if asset_cols_clean:
                    soma_ativos = {c: abs(pd.to_numeric(df_view.loc[df_view.index != total_label, c], errors='coerce')
                                            .fillna(0).sum()) for c in asset_cols_clean}
                    order_assets = sorted(asset_cols_clean, key=lambda k: soma_ativos[k], reverse=True)
                    df_view = df_view[non_contract_cols + order_assets]
                
                #Tirar as colunas "Fundos/Carteiras Adm", "PL", "Weights", "PL_atualizado", "Adm" e todas que começarem com "Max"
                columns_to_drop = ['Fundos/Carteiras Adm', 'PL', 'Weights', 'PL_atualizado', 'Adm']
                columns_to_drop += [col for col in df_view.columns if col.startswith("Max")]
                df_view = df_view.drop(columns=columns_to_drop, errors='ignore')
     
                st.table(df_view)
                st.write("---")
                
                st.write("## Analise por Fundo")
                st.write("### Selecione os filtros")

                lista_fundos = df_contratos.index.tolist()
                lista_fundos = [
                    str(x) for x in df_contratos.index.tolist() if str(x) != 'Total']
                colll1, colll2 = st.columns([4.9, 4.9])
                with colll1:
                    fundos = st.multiselect(
                        "Selecione os fundos que deseja analisar", lista_fundos, default=lista_fundos)
                with colll2:
                    op1 = st.checkbox("CoVaR / % Risco Total", value=True)
                    op2 = st.checkbox("Div01 / Stress", value=True)

                fundos0 = fundos.copy()
                if fundos0:
                    st.write("## Portfólio Atual")
                    soma_pl_sem_pesos2 = calcular_metricas_de_fundo_analise(
                        default_assets, quantidade, df_contratos_2, fundos0, op1, op2)
                # Transforma em lista para poder usar no cálculo
                # Valor do Portfólio (soma simples)

                # st.session_state["ativos_temp"] = list(
                #     quantidade_inicial.keys())

                # # Botão com callback (1 clique = troca de página)
                # st.button(
                #     "Ir para a tela de Preços de Compra/Venda",
                #     on_click=switch_to_page2,
                #     key="go_page2"
                # )

                lista_estrategias = {
                    'DI': 'Juros Nominais Brasil',
                    'DAP': 'Juros Reais Brasil',
                    'NTNB': 'Juros Reais Brasil',
                    'TREASURY': 'Juros US',
                    'WDO1': 'Moedas'
                }
                lista_ativos = [
                    'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
                    'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP26', 'DAP27',
                    'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'TREASURY',
                    'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40', 'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60'
                ]
                st.write('---')
                st.title("Análise de Performance dos Fundos")

                # --- Seletor de Filtro de Tempo ---
                visao = st.sidebar.selectbox("Escolha o tipo de visão", [
                                            "Fundo", "Estratégia"], index=0)
                tipo_filtro = st.sidebar.selectbox("Escolha o filtro de tempo", [
                                                "Diário", "Semanal", "Mensal"], index=2)

                dados_portifolio_atual = pd.read_parquet(
                    'Dados/portifolio_posições.parquet')
                ultimo_dia_dados = dados_portifolio_atual['Dia de Compra'].max()
                ultimo_dia_dados = datetime.datetime.strptime(
                    ultimo_dia_dados, "%Y-%m-%d")
                primeiro_dia_dados = dados_portifolio_atual['Dia de Compra'].min()
                primeiro_dia_dados = datetime.datetime.strptime(
                    primeiro_dia_dados, "%Y-%m-%d")

                df_b3_fechamento = load_b3_prices()
                ultimo_dia_dados_b3 = df_b3_fechamento.columns[-1]
                ultimo_dia_dados_b3 = datetime.datetime.strptime(
                    ultimo_dia_dados_b3, "%Y-%m-%d")

                data_inicial = st.sidebar.date_input(
                    "Data inicial", value=primeiro_dia_dados, min_value=primeiro_dia_dados, max_value=ultimo_dia_dados_b3.date())
                # Múltiplas datas
                if tipo_filtro == "Diário":
                    # soamr 7 dias a data inicial
                    # data_final = data_inicial + datetime.timedelta(days=8)
                    # if data_final >= ultimo_dia_dados_b3.date():
                    data_final = st.sidebar.date_input(
                        "Data final",   value=ultimo_dia_dados_b3, min_value=primeiro_dia_dados)

                else:
                    data_final = st.sidebar.date_input(
                        "Data final",   value=ultimo_dia_dados_b3, min_value=primeiro_dia_dados)

                # Caso o usuário escolha no date_input (retorna objeto date)
                # Convertendo para string
                data_inicial = data_inicial.strftime("%Y-%m-%d")
                data_final = data_final.strftime("%Y-%m-%d")
                st.html(
                    '''
                    <style>
                    div[data-testid="stDateInput"] input {
                        color: black; /* Define o texto */
                                                        }

                        div[data-testid="stSelectbox"] div {
                    color: black; /* Define o texto como preto */
                                                        }                 
                    </style>   
            
                    '''
                )

                # --- Seletor de Tipo de Visão ---
                df_final, df_final_pl = analisar_dados_fundos(
                        soma_pl_sem_pesos,
                        df_b3_fechamento = load_b3_prices(),
                        df_ajuste        = load_ajustes(),
                        basefundos       = load_basefundos()
                )
                #Dropar coluna Total em cada um
                df_final = df_final.drop(columns=['Total'], errors='ignore')
                df_final_pl = df_final_pl.drop(columns=['Total'], errors='ignore')

                df_final.columns = pd.to_datetime(df_final.columns)
                df_final_pl.columns = pd.to_datetime(df_final_pl.columns)

                # Preciso dropar as colunas das datas que estiverem fora do range
                # Filtrando as colunas do DataFrame de acordo com o intervalo de datas fornecido
                df_final = df_final.loc[:, (df_final.columns >= pd.to_datetime(
                    data_inicial)) & (df_final.columns <= pd.to_datetime(data_final))]
                df_final_pl = df_final_pl.loc[:, (df_final_pl.columns >= pd.to_datetime(
                    data_inicial)) & (df_final_pl.columns <= pd.to_datetime(data_final))]

                # VOLTAR AS COLUNAS PARA O FORMATO ORIGINAL
                df_final.columns = df_final.columns.strftime('%d-%b-%y')
                df_final_pl.columns = df_final_pl.columns.strftime('%d-%b-%y')

                # Convertendo as colunas para datetime e ordenando
                df_final = df_final[sorted(df_final.columns, key=pd.to_datetime)]
                df_final_pl = df_final_pl[sorted(
                    df_final_pl.columns, key=pd.to_datetime)]

                if tipo_filtro == "Semanal":
                    df_final_T = df_final.T  # Transpomos para ter datas como índice
                    df_final_T.index = pd.to_datetime(
                        df_final_T.index)  # Convertendo para datetime

                    # Agrupando por semana, somando rendimentos
                    df_semanal = df_final_T.resample('W').sum()
                    df_semanal = df_semanal.T  # Transpomos de volta
                    df_final = df_semanal
                    # Removendo o horário das colunas
                    df_final.columns = pd.to_datetime(
                        df_final.columns).strftime('%d-%b-%y')

                    df_final_pl_T = df_final_pl.T  # Transpomos para ter datas como índice
                    df_final_pl_T.index = pd.to_datetime(
                        df_final_pl_T.index)  # Convertendo para datetime

                    # Agrupando por semana, somando rendimentos
                    df_semanal_pl = df_final_pl_T.resample('W').sum()
                    df_semanal_pl = df_semanal_pl.T  # Transpomos de volta
                    df_final_pl = df_semanal_pl
                    # Removendo o horário das colunas
                    df_final_pl.columns = pd.to_datetime(
                        df_final_pl.columns).strftime('%d-%b-%y')
                    df_result = pl_dia(df_final)

                elif tipo_filtro == "Mensal":
                    df_final_T = df_final.T  # Transpomos para ter datas como índice
                    df_final_T.index = pd.to_datetime(
                        df_final_T.index)  # Convertendo para datetime
                    # Agrupando por mês, somando rendimentos
                    df_mensal = df_final_T.resample('M').sum()
                    df_mensal = df_mensal.T  # Transpomos de volta
                    df_final = df_mensal
                    # Removendo o horário das colunas
                    df_final.columns = pd.to_datetime(
                        df_final.columns).strftime('%d-%b-%y')

                    df_final_pl_T = df_final_pl.T  # Transpomos para ter datas como índice
                    df_final_pl_T.index = pd.to_datetime(
                        df_final_pl_T.index)  # Convertendo para datetime
                    # Agrupando por mês, somando rendimentos
                    df_mensal_pl = df_final_pl_T.resample('M').sum()
                    df_mensal_pl = df_mensal_pl.T  # Transpomos de volta
                    df_final_pl = df_mensal_pl
                    # Removendo o horário das colunas
                    df_final_pl.columns = pd.to_datetime(
                        df_final_pl.columns).strftime('%d-%b-%y')

                df_totais = df_final_pl.copy()
                # Deixar somente as linhas que no indice tem a palavra 'Total'
                df_totais = df_totais[df_totais.index.str.contains('Total')]
                # Tirar tudo depois do espaço no index
                df_totais.index = df_totais.index.str.split(' - ').str[0]
                df_totais = df_totais.sum()
                # Colocar a primeira linha como 'Total'
                df_totais = df_totais.to_frame().T
                df_totais.index = ['Total']

                # ADICIONAR UMA COLUNA DE TOTAL PARA O DF_FINAL
                df_final['Total'] = df_final.sum(axis=1)
                df_final_pl['Total'] = df_final_pl.sum(axis=1)
                linha_total = df_final_pl.index.str.contains('Total')
                linha_total = df_final_pl.loc[linha_total]
                pl_total = df_final.index.str.contains('Total')
                pl_total = df_final.loc[pl_total]
                df_soma = pl_total['Total']
                # Somar as linhas
                pl_total = pl_total.sum()
                pl_total = pl_total.to_frame().T
                pl_total.index = ['Total']
                pl_total['Total'] = df_soma.sum()
                pl_total = pl_total['Total']
                # pl_total.index = ['Total']
                # Renomear Index
                df_soma = linha_total['Total']
                # Somar as linhas
                linha_total = linha_total.sum()
                linha_total = linha_total.to_frame().T
                linha_total.index = ['Total']
                linha_total['Total'] = df_soma.sum()
                linha_total = linha_total['Total']
                # linha_total.index = ['Total']
                # Adicionar a linha de soma na tabela filtrada

                if visao == "Fundo":
                    lista_fundos = df_final.index
                    lista_fundos = lista_fundos.to_list()
                    # Extrai os nomes dos fundos
                    lista_fundos = [i.split(' - ')[1] for i in lista_fundos]
                    lista_fundos = list(set(lista_fundos))  # Remove duplicatas
                    lista_fundos.remove('Total')
                    lista_ativos = df_final.index
                    lista_ativos = lista_ativos.tolist()
                    lista_ativos = [i.split(' - ')[0] for i in lista_ativos]
                    lista_ativos = list(set(lista_ativos))

                    col111, col222 = st.columns([5, 5])
                    with col111:
                        fundos_lista = st.multiselect(
                            "Escolha os fundos", lista_fundos, default=lista_fundos)
                    with col222:
                        ativos_lista = st.multiselect(
                            "Escolha os ativos", lista_ativos)

                    if fundos_lista:
                        # Filtra os dados com base nos fundos escolhidos
                        df_fundos = df_final.loc[df_final.index.str.contains(
                            '|'.join(fundos_lista))]
                        if ativos_lista:
                            df_fundos = df_fundos.loc[df_fundos.index.str.contains(
                                '|'.join(ativos_lista))]

                        # Iterar por todos os fundos e calcular quanto cada um rendeu somando o rendimento de cada ativo
                        # Extrair o nome do fundo (antes de "- P&L")
                        df_fundos['Fundo'] = [
                            i.split(' - ')[1] for i in df_fundos.index.to_list()]
                        df_fundos = df_fundos.groupby('Fundo').sum()
                        TOL_REAIS = 0.01 # 1 centavo (ajuste se quiser)
                        df_fundos = df_fundos.where(df_fundos.abs() >= TOL_REAIS, 0)

                        df_fundos_copy = df_fundos.copy()
                        df_fundos_copy.loc['Total'] = df_fundos.sum()
                        total_fundos = df_fundos_copy.loc['Total', 'Total']

                        for col in df_fundos_copy.columns:

                            df_fundos_copy[col] = df_fundos_copy[col].apply(
                                lambda x: f"R${x:,.2f}")

                        df_fundos_grana = df_fundos_copy
                        # Pegar o valor da coluna total e da linha total
                        # Transforma o DataFrame de formato largo para longo
                        # T (transpose) para transformar colunas em linhas
                        df_fundos_long = df_fundos.T.reset_index()
                        # A primeira coluna será a data e as demais são os fundos
                        df_fundos_long.columns = ['date'] + list(df_fundos.index)
                        df_fundos_long = df_fundos_long.melt(
                            id_vars=["date"], var_name="fundo", value_name="Rendimento_diario")

                        # Filtra as linhas que são datas reais (elimina "Total" ou outros valores não-datas)
                        df_fundos_long = df_fundos_long[pd.to_datetime(
                            df_fundos_long['date'], errors='coerce').notna()]

                        # Convertendo a coluna 'date' para datetime
                        df_fundos_long['date'] = pd.to_datetime(
                            df_fundos_long['date'])

                        # Convertendo a coluna 'date' para datetime
                        df_fundos_long['date'] = pd.to_datetime(
                            df_fundos_long['date'])
                        # Remover a parte da hora
                        if tipo_filtro == "Diário":
                            df_fundos_long['date'] = df_fundos_long['date'].dt.strftime(
                                '%d %b %Y')
                        elif tipo_filtro == "Semanal":
                            df_fundos_long['date'] = df_fundos_long['date'].dt.strftime(
                                '%d %b %Y')
                        else:
                            df_fundos_long['date'] = df_fundos_long['date'].dt.strftime(
                                '%b %Y')

                        # Criando o gráfico de barras
                        # Cria a figura e os eixos
                        from plotnine import (
                            ggplot, aes, geom_col, geom_line,
                            labs, scale_fill_brewer, scale_color_manual, scale_x_datetime,
                            theme_minimal, theme,
                            element_rect, element_line, element_text, scale_fill_manual
                        )

                        # --------------------------------------------------------------------------- #
                        # 1)  RENDIMENTO DIÁRIO POR FUNDO  (barras agrupadas)
                        # --------------------------------------------------------------------------- #

                        # paleta personalizada – uma cor distinta para cada fundo
                        palette_fundos = {
                            "GLOBAL BONDS": "#003366",  # azul‑escuro
                            "HORIZONTE": "#B03A2E",  # vermelho tijolo
                            "JERA2026": "#138D75",  # verde
                            "REAL FIM": "#7F3C8D",  # roxo
                            "BH FIRF INFRA": "#D35400",  # laranja queimado
                            "BORDEAUX INFRA": "#1ABC9C",  # turquesa
                            "TOPAZIO INFRA": "#34495E",  # cinza‑azulado
                            "MANACA INFRA FIRF": "#C27E00",  # mostarda
                            "AF DEB INCENTIVADAS": "#E0115F"   # magenta
                        }

                        from typing import Optional, Tuple

                        def _looks_like_long_fundos(df: pd.DataFrame) -> bool:
                            return {"date", "Rendimento_diario"}.issubset(df.columns)

                        def _ensure_fundo_col(df: pd.DataFrame) -> pd.DataFrame:
                            if "fundo" in df.columns:
                                return df.rename(columns={"fundo": "fundo"})
                            # tenta alternativas usuais
                            for c in ("estratégia", "estrategia", "classe", "categoria"):
                                if c in df.columns:
                                    return df.rename(columns={c: "fundo"})
                            # senão, usa a primeira coluna categórica
                            first_cat = [c for c in df.columns if c not in {"date", "Rendimento_diario"}]
                            if not first_cat:
                                raise ValueError("Não encontrei coluna que identifique o 'fundo'.")
                            return df.rename(columns={first_cat[0]: "fundo"})

                        def _fix_long_fundos_bps(df_long: pd.DataFrame,
                                                total_label: str = "Total") -> pd.DataFrame:
                            df = df_long.copy()
                            df = df.reset_index()
                            df = _ensure_fundo_col(df)
                            df["fundo"] = df["fundo"].astype(str)

                            # datas
                            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                            df = df.dropna(subset=["date"])

                            # numéricos, trata NaN/inf
                            df["Rendimento_diario"] = (
                                pd.to_numeric(df["Rendimento_diario"], errors="coerce")
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0.0)
                            )

                            # remove total antigo se houver
                            df = df[df["fundo"].str.upper() != total_label.upper()]

                            # recalcula total (se fizer sentido para você manter o Total)
                            total = (
                                df.groupby("date", as_index=False)["Rendimento_diario"]
                                .sum()
                                .assign(fundo=total_label)
                            )
                            df_out = pd.concat([df, total], ignore_index=True)

                            return df_out[["fundo", "date", "Rendimento_diario"]]

                        def _fix_wide_fundos_bps(df_wide: pd.DataFrame,
                                                total_label: str = "Total") -> pd.DataFrame:
                            df = df_wide.copy()
                            df = df.reset_index()
                    

                            first_col = df.columns[0]
                            df = df.rename(columns={first_col: "fundo"})
                            df["fundo"] = df["fundo"].astype(str)

                            # identifica/garante a linha total
                            mask_total = df["fundo"].str.strip().str.upper() == total_label.upper()
                            if not mask_total.any():
                                df = pd.concat([df, pd.DataFrame({"fundo": [total_label]})], ignore_index=True)
                                mask_total = df["fundo"].str.strip().str.upper() == total_label.upper()

                            # colunas de datas (ignora coluna chamada 'Total' se existir)
                            date_cols = [c for c in df.columns[1:] if c.strip().upper() != "TOTAL"]

                            # numérico + limpeza
                            df[date_cols] = (
                                df[date_cols]
                                .apply(pd.to_numeric, errors="coerce")
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0.0)
                            )

                            # recalcula total
                            df.loc[mask_total, date_cols] = df.loc[~mask_total, date_cols].sum(axis=0).values

                            # já assumimos que os números JÁ estão em bps (como você pediu)
                            # se não estiverem, multiplique aqui (ex: df[date_cols] *= 10_000)

                            # para long
                            df_long = (
                                df.melt(id_vars="fundo", value_vars=date_cols, var_name="date", value_name="Rendimento_diario")
                                .assign(date=lambda d: pd.to_datetime(d["date"], dayfirst=True, errors="coerce"))
                                .dropna(subset=["date"])
                            )
                            return df_long[["fundo", "date", "Rendimento_diario"]]

                        def consertar_df_final_grafico_fundos_bps(df_raw: pd.DataFrame,
                                                                total_label: str = "Total") -> pd.DataFrame:
                            """
                            Aceita df wide (datas nas colunas) ou long (fundo, date, Rendimento_diario).
                            Considera que os valores JÁ estão em bps.
                            Retorna SEMPRE long: ['fundo', 'date', 'Rendimento_diario'] (em bps).
                            """
                            if _looks_like_long_fundos(df_raw):
                                return _fix_long_fundos_bps(df_raw, total_label=total_label)
                            else:
                                return _fix_wide_fundos_bps(df_raw, total_label=total_label)


                        # ============================================================
                        # GRÁFICOS EM PLOTLY (bps)
                        # ============================================================
                        import plotly.express as px

                        PALETTE_FUNDOS = {
                            "GLOBAL BONDS": "#003366",      # azul‑escuro
                            "HORIZONTE": "#B03A2E",         # vermelho tijolo
                            "JERA2026": "#138D75",          # verde
                            "REAL FIM": "#7F3C8D",          # roxo
                            "BH FIRF INFRA": "#D35400",     # laranja queimado
                            "BORDEAUX INFRA": "#1ABC9C",    # turquesa
                            "TOPAZIO INFRA": "#34495E",     # cinza‑azulado
                            "MANACA INFRA FIRF": "#C27E00", # mostarda
                            "AF DEB INCENTIVADAS": "#E0115F",
                            "Total": "#000000"
                        }

                        def gg_rendimento_diario_fundos_plotly(
                            df_fundos_long_bps: pd.DataFrame,
                            tol: float = 0,
                            palette: Optional[dict] = None
                        ):
                            """
                            Barras de rendimento diário por fundo, **em bps**.
                            `tol` também em bps.
                            Retorna uma figura Plotly (para usar direto no Dash).
                            """
                            if palette is None:
                                palette = PALETTE_FUNDOS
                            
                            #Tirar o fundo 'Total' se existir
                            df_fundos_long_bps = df_fundos_long_bps[df_fundos_long_bps['fundo'].str.upper() != 'TOTAL']

                            df_plot = (
                                df_fundos_long_bps
                                .copy()
                                .assign(date=pd.to_datetime(df_fundos_long_bps["date"], dayfirst=True, errors="coerce"))
                                .dropna(subset=["date", "Rendimento_diario"])
                                .loc[lambda d: d["Rendimento_diario"].abs() > tol]
                            )

                            if df_plot.empty:
                                raise ValueError("Nenhuma linha com Rendimento_diario (bps) diferente de zero acima do tol.")

                            # Ordena cronologicamente
                            df_plot = df_plot.sort_values("date")

                            fig = px.bar(
                                df_plot,
                                x="date",
                                y="Rendimento_diario",
                                color="fundo",
                                barmode="group",
                                color_discrete_map=palette,
                                labels={
                                    "date": "Data",
                                    "Rendimento_diario": "Rendimento Diário (bps)",
                                    "fundo": "Fundo"
                                },
                                title="Rendimento Diário por Fundo (bps)"
                            )

                            fig.update_layout(
                                xaxis_title="Data",
                                yaxis_title="Rendimento Diário (bps)",
                                legend_title="Fundo",
                                bargap=0.15,
                                bargroupgap=0.05,
                                template="plotly_white",
                                height=500
                            )
                            fig.update_xaxes(tickformat="%d-%b", tickangle=45)
                            return fig


                        def gg_curva_capital_fundos_plotly(
                            df_fundos_long_bps: pd.DataFrame,
                            palette: Optional[dict] = None
                        ):
                            """
                            Curva de capital acumulada por fundo **em bps** (soma dos bps diários).
                            Retorna figura Plotly.
                            """
                            if palette is None:
                                palette = PALETTE_FUNDOS
                            #Tirar o fundo 'Total' se existir
                            df_fundos_long_bps = df_fundos_long_bps[df_fundos_long_bps['fundo'].str.upper() != 'TOTAL']
                            df = df_fundos_long_bps.copy()
                            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                            df = df.dropna(subset=["date", "Rendimento_diario"])
                            df.sort_values(["fundo", "date"], inplace=True)
                            df["acumulado"] = df.groupby("fundo")["Rendimento_diario"].cumsum()

                            fig = px.line(
                                df,
                                x="date",
                                y="acumulado",
                                color="fundo",
                                color_discrete_map=palette,
                                labels={
                                    "date": "Data",
                                    "acumulado": "Acumulado (bps)",
                                    "fundo": "Fundo"
                                },
                                title="Rendimento Acumulado – Curva de Capital por Fundo (bps)"
                            )

                            fig.update_layout(
                                xaxis_title="Data",
                                yaxis_title="Acumulado (bps)",
                                legend_title="Fundo",
                                template="plotly_white",
                                height=500
                            )
                            fig.update_xaxes(tickformat="%d-%b", tickangle=45)
                            return fig
                    
                    # Exibe o gráfico com o Streamlit, passando a figura
                    # gráfico de barras
                    #fig_diario = gg_rendimento_diario_fundos(df_fundos_long).draw()
                    # curva acumulada
                    #fig_acum = gg_curva_capital_fundos(df_fundos_long).draw()
                    df_final = df_final_pl

                    if fundos_lista:
                        # Filtra os dados com base nos fundos escolhidos
                        df_fundos = df_final.loc[df_final.index.str.contains(
                            '|'.join(fundos_lista))]
                        if ativos_lista:
                            df_fundos = df_fundos.loc[df_fundos.index.str.contains(
                                '|'.join(ativos_lista))]

                        # Iterar por todos os fundos e calcular quanto cada um rendeu somando o rendimento de cada ativo
                        # Extrair o nome do fundo (antes de "- P&L")
                        df_fundos['Fundo'] = [
                            i.split(' - ')[1] for i in df_fundos.index.to_list()]
                        df_fundos = df_fundos.groupby('Fundo').sum()
                        df_fundos_copy = df_fundos.copy()
                        df_fundos_copy = pd.concat([df_fundos_copy, df_totais])

                        coluna_totais = df_fundos_copy.loc['Total']


                        # Mudar a celula da linha 'Total' e da coluna 'Total' para coluna_totais.sum()
                        # st.write(total_fundos,soma_pl_sem_pesos)
                        df_fundos_copy.iloc[-1, -1] = total_fundos / \
                            soma_pl_sem_pesos * 10000


                        # ------------------------------------------------------------------------------
                        # Mostra “0.00 bps” sempre que o valor real for zero (ou perto de zero)
                        ZERO_EPS = 0.0005     # 0,00005 bps  → ≈ 1 centavo em PL de 20 MM


                        df_fundos_copy2 = df_fundos_copy.copy()
                        df_fundos_copy2 = consertar_df_final_grafico_fundos_bps(df_fundos_copy2)
                        fig_diario = gg_rendimento_diario_fundos_plotly(df_fundos_copy2)
                        # curva acumulada
                        fig_acum = gg_curva_capital_fundos_plotly(df_fundos_copy2)

                        for col in df_fundos_copy.columns:
                            df_fundos_copy[col] = df_fundos_copy[col].apply(
                                lambda x: "0.00bps" if abs(x) < ZERO_EPS else f"{x:.2f}bps"
                            )
                        # ------------------------------------------------------------------------------
                        # df_totais
                        # Adicionar df_totais na tabela df_fundos_copy na onde tiver as mesmas datas
                        df_combinado = df_fundos_grana + " / " + df_fundos_copy

                        df_combinado.loc['Total'] = df_combinado.loc['Total'].str.split('/').str[0].str.strip()
                        # Dropar linha do total
                        # df_combinado = df_combinado.drop('Total', axis=0)
                        st.write('### Tabela de Rendimento Diário por Estratégia')
                        st.table(df_combinado)
                        st.write('### Gráficos de Rendimento Diário e Acumulado')
                        st.plotly_chart(fig_diario, use_container_width=True)
                        st.write('### Graficos de Curva de Capital')
                        st.plotly_chart(fig_acum, use_container_width=True)

                elif visao == "Estratégia":
                    lista_estrategias = {
                        'DI': 'Juros Nominais Brasil',
                        'DAP': 'Juros Reais Brasil',
                        'NTNB': 'Juros Reais Brasil',
                        'TREASURY': 'Juros US',
                        'WDO1': 'Moedas'
                    }
                    lista_ativos = df_final.index
                    lista_ativos = lista_ativos.tolist()
                    lista_ativos = [i.split(' - ')[0] for i in lista_ativos]
                    lista_ativos = list(set(lista_ativos))

                    # Criação das colunas para seleção múltipla no Streamlit
                    col111, col222 = st.columns([5, 5])
                    with col111:
                        estrategias_lista = st.multiselect("Escolha as estratégias", lista_estrategias.values(
                        ), default=lista_estrategias.values())  # Estratégias selecionadas
                    with col222:
                        ativos_lista = st.multiselect(
                            "Escolha os ativos", lista_ativos)  # Ativos selecionados

                    if estrategias_lista:
                        # Aqui, estamos mapeando os valores das estratégias para suas chaves
                        estrategias_chaves = [
                            k for k, v in lista_estrategias.items() if v in estrategias_lista]

                        # Se estratégias e ativos forem selecionados, realizamos o filtro
                        if ativos_lista:
                            # Filtra o df_final com base nas estratégias e ativos escolhidos
                            df_estrategias = df_final.loc[df_final.index.str.contains(
                                '|'.join(estrategias_chaves))]  # Filtra pela estratégia
                            df_estrategias = df_estrategias.loc[df_estrategias.index.str.contains(
                                '|'.join(ativos_lista))]  # Filtra pelo ativo

                        else:
                            # Se não selecionar nenhum ativo, apenas filtra pela estratégia
                            df_estrategias = df_final.loc[df_final.index.str.contains(
                                '|'.join(estrategias_chaves))]
                            df_estrategias['Fundo'] = [
                                i.split(' - ')[1] for i in df_estrategias.index.to_list()]
                            df_estrategias = df_estrategias[df_estrategias['Fundo'] != 'Total']
                            df_estrategias.drop(columns=['Fundo'], inplace=True)

                        # Agora, adicionar a coluna 'Estrategia' corretamente:
                        lista_estrategias_atualizar = []
                        for idx, row in df_estrategias.iterrows():
                            if 'DI' in idx:
                                lista_estrategias_atualizar.append(
                                    'JUROS NOMINAIS BRASIL')
                            elif 'DAP' in idx:
                                lista_estrategias_atualizar.append(
                                    'JUROS REAIS BRASIL')
                            elif 'TREASURY' in idx:
                                lista_estrategias_atualizar.append('JUROS US')
                            elif 'WDO1' in idx:
                                lista_estrategias_atualizar.append('MOEDAS')
                        df_estrategias['Estrategia'] = lista_estrategias_atualizar
                        df_estrategias = df_estrategias.groupby('Estrategia').sum()
                        # ------------------------------------------------------------------
                        # Zera valores muito pequenos para evitar 0,01 bps “fantasma”
                        TOL_REAIS = 0.01          # corte de 1 centavo; ajuste se quiser
                        df_estrategias = df_estrategias.where(df_estrategias.abs() >= TOL_REAIS, 0)
                        # ------------------------------------------------------------------

                        df_estrategias_copy = df_estrategias.copy()
                        df_estrategias_copy.loc['Total'] = df_estrategias_copy.sum(
                        )
                        total_estrategias = df_estrategias_copy.loc['Total', 'Total']

                        for col in df_estrategias_copy.columns:
                            df_estrategias_copy[col] = df_estrategias_copy[col].apply(
                                lambda x: f"R${x:,.2f}")
                        # Adicionar a linha Total

                        # Exibe a tabela filtrada
                        df_estrategias_grana = df_estrategias_copy.copy()

                        # Transforma o DataFrame de formato largo para longo

                        # T (transpose) para transformar colunas em linhas
                        df_estrategias_long = df_estrategias.T.reset_index()

                        # A primeira coluna será a data e as demais são as estratégias
                        df_estrategias_long.columns = [
                            'date'] + list(df_estrategias.index)
                        df_estrategias_long = df_estrategias_long.melt(
                            id_vars=["date"], var_name="estratégia", value_name="Rendimento_diario")

                        # Filtra as linhas que são datas reais (elimina "Total" ou outros valores não-datas)
                        df_estrategias_long = df_estrategias_long[pd.to_datetime(
                            df_estrategias_long['date'], errors='coerce').notna()]

                        # Convertendo a coluna 'date' para datetime
                        df_estrategias_long['date'] = pd.to_datetime(
                            df_estrategias_long['date'])

                        df_resultado = calcular_retorno_sobre_pl(
                            df_final, df_estrategias_long)
                        
                        df_final22 = df_estrategias_long.copy()
                        df_final22['Rendimento_diario'] = df_resultado['Retorno_sobre_PL']

                        # Remover a parte da hora
                        if tipo_filtro == "Diário":
                            df_estrategias_long['date'] = df_estrategias_long['date'].dt.strftime(
                                '%d %b %Y')

                        elif tipo_filtro == "Semanal":
                            df_estrategias_long['date'] = df_estrategias_long['date'].dt.strftime(
                                '%d %b %Y')

                        else:
                            df_estrategias_long['date'] = df_estrategias_long['date'].dt.strftime(
                                '%b %Y')

                        # Criando o gráfico de barras
                        # Cria a figura e os eixos

                        # --------------------------------------------------------------------------- #
                        # 1)  GRÁFICO DE BARRAS – Rendimento Diário por Estratégia
                        # --------------------------------------------------------------------------- #

                        # --------------------------------------------------------------------------------------
                        # 1) Patch – conserta o df (funciona para df wide OU long) e garante saída em bps (long)
                        # --------------------------------------------------------------------------------------
                        from typing import Tuple

                        def _looks_like_long(df: pd.DataFrame) -> bool:
                            return {"date", "Rendimento_diario"}.issubset(df.columns)

                        def _maybe_to_bps(series: pd.Series, multiplier: int = 1) -> Tuple[pd.Series, bool]:
                            """
                            Se os valores parecerem estar em retornos (ex.: 0.0012 = 12 bps),
                            converte para bps. Se já parecem bps (valores ~ dezenas/centenas),
                            não faz nada.
                            Heurística simples: se 95% do |valor| < 1, consideramos fração e multiplicamos.
                            """
                            s = pd.to_numeric(series, errors="coerce")
                            mask = s.notna()
                            if mask.sum() == 0:
                                return s, False
                            pct_small = (s[mask].abs() < 1).mean()
                            if pct_small >= 0.95:
                                return s * multiplier, True
                            return s, False

                        def _fix_long(df_long: pd.DataFrame, total_label: str = "Total") -> pd.DataFrame:
                            df = df_long.copy()

                            # Normaliza nome da coluna de estratégia
                            if "estratégia" not in df.columns:
                                if "fundo" in df.columns:
                                    df = df.rename(columns={"fundo": "estratégia"})
                                else:
                                    # se não achar, pega a primeira coluna não 'date' e não 'Rendimento_diario'
                                    first_cat = [c for c in df.columns if c not in {"date", "Rendimento_diario"}]
                                    if not first_cat:
                                        raise ValueError("Não encontrei uma coluna de estratégia/fundo no df_long.")
                                    df = df.rename(columns={first_cat[0]: "estratégia"})

                            # Tipos corretos
                            df["estratégia"] = df["estratégia"].astype(str)
                            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                            df = df.dropna(subset=["date"])

                            # Trata NaN/inf
                            df["Rendimento_diario"] = pd.to_numeric(df["Rendimento_diario"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

                            # Converte para bps se necessário
                            df["Rendimento_diario"], _ = _maybe_to_bps(df["Rendimento_diario"], 10_000)

                            # Recalcula TOTAL por data (remove eventual total antigo)
                            df = df[df["estratégia"].str.upper() != total_label.upper()]
                            total = (
                                df.groupby("date", as_index=False)["Rendimento_diario"]
                                .sum()
                                .assign(estratégia=total_label)
                            )
                            df_out = pd.concat([df, total], ignore_index=True)

                            return df_out

                        def _fix_wide(df_wide: pd.DataFrame, multiplier: int = 1, total_label: str = "Total") -> pd.DataFrame:
                            df = df_wide.copy()

                            # Primeira coluna vira 'estratégia'
                            df = df.reset_index()
                            first_col = df.columns[0]
                            df = df.rename(columns={first_col: "estratégia"})
                            df["estratégia"] = df["estratégia"].astype(str)

                            # identifica/garante a linha total
                            mask_total = df["estratégia"].str.strip().str.upper() == total_label.upper()
                            if not mask_total.any():
                                df = pd.concat([df, pd.DataFrame({"estratégia": [total_label]})], ignore_index=True)
                                mask_total = df["estratégia"].str.strip().str.upper() == total_label.upper()

                            # Colunas que são datas (exclui coluna "Total" que às vezes vem ao final)
                            date_cols = [c for c in df.columns[1:] if c.strip().upper() != "TOTAL"]

                            # Numeric + limpeza
                            df[date_cols] = (
                                df[date_cols]
                                .apply(pd.to_numeric, errors="coerce")
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0.0)
                            )

                            # Recalcula Total
                            df.loc[mask_total, date_cols] = df.loc[~mask_total, date_cols].sum(axis=0).values

                            # Converte para bps (assume que veio em fração)
                            df[date_cols] = df[date_cols] * multiplier

                            # Para long
                            df_long = (
                                df.melt(id_vars="estratégia", value_vars=date_cols, var_name="date", value_name="Rendimento_diario")
                                .assign(date=lambda d: pd.to_datetime(d["date"], dayfirst=True, errors="coerce"))
                                .dropna(subset=["date"])
                            )
                            return df_long

                        def consertar_df_final_grafico(df_raw: pd.DataFrame,
                                                    multiplier: int = 1,
                                                    total_label: str = "Total") -> pd.DataFrame:
                            """
                            - Aceita df wide (datas nas colunas) OU df long (colunas: estratégia/fundo, date, Rendimento_diario).
                            - Corrige linha Total.
                            - Limpa NaN/inf.
                            - Garante saída em bps e em formato long.
                            """
                            if _looks_like_long(df_raw):
                                return _fix_long(df_raw, total_label=total_label)
                            else:
                                return _fix_wide(df_raw, multiplier=multiplier, total_label=total_label)


                        # --------------------------------------------------------------------------------------
                        # 2) Gráficos em BPS
                        # --------------------------------------------------------------------------------------
                        from plotnine import (
                            ggplot, aes, geom_col, geom_line, labs,
                            scale_fill_manual, scale_color_manual,
                            theme_minimal, theme,
                            element_text, element_rect, element_line
                        )
                        import plotly.express as px


                        PALETTE_CURVAS = {
                            "JUROS NOMINAIS BRASIL": "#B03A2E",
                            "JUROS REAIS BRASIL":    "#003366",
                            "MOEDAS":                "#138D75",
                            "JUROS US":              "#7F3C8D",
                            "Total":                 "#000000"
                        }

                        def gg_rendimento_diario_plotly(
                            df_estrategias_long: pd.DataFrame,
                            tol: float = 0,
                            palette: dict | None = None
                        ):
                            """
                            Barras de rendimento diário **em bps** (usa Plotly).
                            `tol` também em bps.
                            Filtra fora a estratégia 'Total'.
                            """
                            if palette is None:
                                palette = PALETTE_CURVAS

                            df_plot = (
                                df_estrategias_long
                                .copy()
                                .query("estratégia != 'Total'")
                                .assign(date=lambda d: pd.to_datetime(d["date"], dayfirst=True, errors="coerce"))
                                .dropna(subset=["date", "Rendimento_diario"])
                                .loc[lambda d: d["Rendimento_diario"].abs() > tol]
                                .sort_values("date")
                            )

                            if df_plot.empty:
                                raise ValueError("Nenhuma linha com Rendimento_diario (bps) diferente de zero acima do tol.")

                            fig = px.bar(
                                df_plot,
                                x="date",
                                y="Rendimento_diario",
                                color="estratégia",
                                barmode="group",
                                color_discrete_map=palette,
                                labels={
                                    "date": "Data",
                                    "Rendimento_diario": "Rendimento Diário (bps)",
                                    "estratégia": "Estratégia"
                                },
                                title="Rendimento Diário por Estratégia (bps)"
                            )

                            fig.update_layout(
                                xaxis_title="Data",
                                yaxis_title="Rendimento Diário (bps)",
                                legend_title="Estratégia",
                                bargap=0.15,
                                bargroupgap=0.05,
                                template="plotly_white",
                                height=500
                            )
                            fig.update_xaxes(tickformat="%d-%b", tickangle=90)
                            return fig


                        def gg_curva_capital_plotly(
                            df_long: pd.DataFrame,
                            palette: dict | None = None
                        ):
                            """
                            Curva de capital acumulada **em bps** (usa Plotly).
                            Filtra fora a estratégia 'Total'.
                            """
                            if palette is None:
                                palette = PALETTE_CURVAS

                            df = (
                                df_long
                                .copy()
                                .query("estratégia != 'Total'")
                                .assign(date=lambda d: pd.to_datetime(d["date"], dayfirst=True, errors="coerce"))
                                .dropna(subset=["date", "Rendimento_diario"])
                                .sort_values(["estratégia", "date"])
                            )
                            df["acumulado"] = df.groupby("estratégia")["Rendimento_diario"].cumsum()

                            fig = px.line(
                                df,
                                x="date",
                                y="acumulado",
                                color="estratégia",
                                color_discrete_map=palette,
                                labels={
                                    "date": "Data",
                                    "acumulado": "Acumulado (bps)",
                                    "estratégia": "Estratégia"
                                },
                                title="Rendimento Acumulado – Curva de Capital (bps)"
                            )

                            fig.update_layout(
                                xaxis_title="Data",
                                yaxis_title="Acumulado (bps)",
                                legend_title="Estratégia",
                                template="plotly_white",
                                height=500
                            )
                            fig.update_xaxes(tickformat="%d-%b", tickangle=90)
                            return fig

                                                                        # plt.show()

                        # --- se estiver no Streamlit --------------------------------------------------
                        # import streamlit as st
                        # Exibe o gráfico com o Streamlit, passando a figura

                    df_final = df_final_pl
                    if estrategias_lista:
                        # Aqui, estamos mapeando os valores das estratégias para suas chaves
                        estrategias_chaves = [
                            k for k, v in lista_estrategias.items() if v in estrategias_lista]

                        # Se estratégias e ativos forem selecionados, realizamos o filtro
                        if ativos_lista:
                            # Filtra o df_final com base nas estratégias e ativos escolhidos
                            df_estrategias = df_final.loc[df_final.index.str.contains(
                                '|'.join(estrategias_chaves))]  # Filtra pela estratégia
                            df_estrategias = df_estrategias.loc[df_estrategias.index.str.contains(
                                '|'.join(ativos_lista))]  # Filtra pelo ativo

                        else:
                            # Se não selecionar nenhum ativo, apenas filtra pela estratégia
                            df_estrategias = df_final.loc[df_final.index.str.contains(
                                '|'.join(estrategias_chaves))]
                            df_estrategias['Fundo'] = [
                                i.split(' - ')[1] for i in df_estrategias.index.to_list()]
                            df_estrategias = df_estrategias[df_estrategias['Fundo'] != 'Total']
                            df_estrategias.drop(columns=['Fundo'], inplace=True)
                        # Agora, adicionar a coluna 'Estrategia' corretamente:
                        lista_estrategias_atualizar = []
                        for idx, row in df_estrategias.iterrows():
                            if 'DI' in idx:
                                lista_estrategias_atualizar.append(
                                    'JUROS NOMINAIS BRASIL')
                            elif 'DAP' in idx:
                                lista_estrategias_atualizar.append(
                                    'JUROS REAIS BRASIL')
                            elif 'TREASURY' in idx:
                                lista_estrategias_atualizar.append('JUROS US')
                            elif 'WDO1' in idx:
                                lista_estrategias_atualizar.append('MOEDAS')
                        df_estrategias['Estrategia'] = lista_estrategias_atualizar
                        df_estrategias = df_estrategias.groupby('Estrategia').sum()

                        df_estrategias_copy = df_estrategias.copy()
                        df_estrategias_copy.loc['Total'] = df_estrategias_copy.sum(
                        )
                        #Substituir a linha Total por ""
                        # Substitui os valores da linha 'Total' por string vazia

                        df_estrategias_copy.iloc[-1, -
                                                1] = total_estrategias/soma_pl_sem_pesos * 10000
                        
                        for col in df_estrategias_copy.columns:
                            df_estrategias_copy[col] = df_estrategias_copy[col].apply(
                                lambda x: f"{x:.2f}bps")
                            # Remover a parte da hora

                    df_final22 = df_final22.rename(
                        columns={'estratégia': 'Estrategia'})
                    df_final22 = df_final22.pivot(
                        index='Estrategia', columns='date', values='Rendimento_diario')
                    # Tirar o horário
                    df_final22.columns = pd.to_datetime(df_final22.columns)
                    df_final22.columns = df_final22.columns.strftime('%d-%b-%y')
                    df_final22 = pd.concat([df_final22, df_totais])

                    df_final22['Total'] = df_final22.sum(axis=1)

                    coluna_totais = df_final22.loc['Total']

                    # Mudar a celula da linha 'Total' e da coluna 'Total' para coluna_totais.sum()
                    df_final22.iloc[-1, -1] = coluna_totais.sum()
                    df_final_grafico = df_final22.copy()

                    for col in df_final22.columns:
                        df_final22[col] = df_final22[col].apply(
                            lambda x: f"{x:.2f}bps")

                    # st.write(df_estrategias_copy,df_estrategias_grana,df_final22)
                    df_combinado = df_estrategias_grana + " / " + df_final22
                    df_combinado.loc['Total'] = df_combinado.loc['Total'].str.split('/').str[0].str.strip()
                    # df_combinado = df_combinado.drop('Total', axis=0)
                    # df_combinado.drop(columns=['Total'], inplace=True)
                    # Teste
                    df_final_grafico = consertar_df_final_grafico(df_final_grafico)
                    p_barras = gg_rendimento_diario_plotly(df_final_grafico)
                    fig1 = p_barras          # converte o ggplot em Figure
                    p_curva = gg_curva_capital_plotly(df_final_grafico)
                    fig2 = p_curva

                    st.write('### Tabela de Rendimento Diário por Estratégia')
                    st.table(df_combinado)
                    st.write('### Gráficos de Rendimento Diário por Estratégia')
                    st.plotly_chart(fig1)
                    st.write('### Graficos de Curva de Capital')
                    st.plotly_chart(fig2)
                    # st.pyplot(fig2)

                elif visao == "Ativo":
                    lista_ativos = df_final.index
                    lista_ativos = lista_ativos.tolist()
                    lista_ativos = [i.split(' - ')[0] for i in lista_ativos]
                    lista_ativos = list(set(lista_ativos))

                    # Criação das colunas para seleção múltipla no Streamlit
                    ativos_lista = st.multiselect(
                        "Escolha os ativos", lista_ativos, default=lista_ativos)

                    if ativos_lista:
                        # Filtra o df_final com base nos ativos escolhidos
                        df_ativos = df_final
                        df_ativos['Fundo'] = [
                            i.split(' - ')[1] for i in df_ativos.index.to_list()]
                        df_ativos = df_ativos[df_ativos['Fundo'] != 'Total']
                        df_ativos.drop(columns=['Fundo'], inplace=True)
                        df_ativos['Ativo'] = [
                            i.split(' - ')[0] for i in df_ativos.index.to_list()]
                        df_ativos = df_ativos.loc[df_ativos['Ativo'].isin(
                            ativos_lista)]
                        df_ativos = df_ativos.groupby('Ativo').sum()

                        df_ativos_copy = df_ativos.copy()
                        df_ativos_copy.loc['Total'] = df_ativos_copy.sum()

                        for col in df_ativos_copy.columns:
                            df_ativos_copy[col] = df_ativos_copy[col].apply(
                                lambda x: f"R${x:,.2f}")

                        # Exibe a tabela filtrada
                        df_ativos_grana = df_ativos_copy.copy()

                        # Transforma o DataFrame de formato largo para longo
                        # T (transpose) para transformar colunas em linhas
                        df_ativos_long = df_ativos.T.reset_index()
                        # A primeira coluna será a data e as demais são os ativos
                        df_ativos_long.columns = ['date'] + list(df_ativos.index)
                        df_ativos_long = df_ativos_long.melt(
                            id_vars=["date"], var_name="ativo", value_name="Rendimento_diario")

                        # Filtra as linhas que são datas reais (elimina "Total" ou outros valores não-datas)
                        df_ativos_long = df_ativos_long[pd.to_datetime(
                            df_ativos_long['date'], errors='coerce').notna()]

                        # Convertendo a coluna 'date' para datetime
                        df_ativos_long['date'] = pd.to_datetime(
                            df_ativos_long['date'])

                        # Deixando somente as datas sem o horário
                        # Converter a coluna 'date' para datetime, caso não esteja
                        df_ativos_long['date'] = pd.to_datetime(
                            df_ativos_long['date'], errors='coerce')
                        # Remover a parte da hora
                        if tipo_filtro == "Diario":
                            df_ativos_long['date'] = df_ativos_long['date'].dt.strftime(
                                '%d %b')

                        elif tipo_filtro == "Semanal":
                            df_ativos_long['date'] = df_ativos_long['date'].dt.strftime(
                                '%d %b %Y')

                        else:
                            df_ativos_long['date'] = df_ativos_long['date'].dt.strftime(
                                '%b %Y')

                        # df_ativos_long['date'] = df_ativos_long['date'].dt.date
                        # df_ativos_long['date'] = df_ativos_long['date'].dt.strftime('%b %Y')
                        # Criando o gráfico de barras
                        sns.set_theme(style="whitegrid")
                        plt.rcParams.update({
                            "figure.figsize": (16, 7),
                            "axes.facecolor": "white",
                            "axes.edgecolor": "black",
                            "grid.color": "#CCCCCC",
                            "grid.linestyle": "-",
                            "grid.alpha": 1.0,
                            "axes.titleweight": "bold",
                            "axes.titlesize": 13,
                            "axes.titlecolor": "#003366",
                            "axes.labelweight": "bold",
                            "xtick.labelsize": 10,
                            "ytick.labelsize": 10,
                        })

                        # -------- função de plotagem -------------------------------------------------
                        def plot_rendimento_diario_ativo(df_ativos_long: pd.DataFrame):
                            fig, ax = plt.subplots()

                            sns.barplot(
                                x="date",
                                y="Rendimento_diario",
                                hue="ativo",
                                data=df_ativos_long,
                                ax=ax,
                                palette="Blues"
                            )

                            # Datas inclinadas
                            ax.set_xticklabels(
                                ax.get_xticklabels(),
                                rotation=45,
                                ha="right"
                            )

                            # Títulos e rótulos
                            ax.set_title("Rendimento Diário por Ativo")
                            ax.set_xlabel("Data")
                            ax.set_ylabel("Rendimento Diário")

                            # Grade menor pontilhada (para imitar panel_grid_minor)
                            ax.grid(which="minor", linestyle=":",
                                    linewidth=0.5, color="#CCCCCC")
                            ax.minorticks_on()

                            # Legenda embaixo, coerente com os demais gráficos
                            ax.legend(
                                title="Ativos",
                                loc="upper left",
                                bbox_to_anchor=(0.02, 0.98),
                                ncol=3,
                                frameon=False
                            )

                            plt.tight_layout()
                            return fig

                        # ---- exemplo de uso em Streamlit -------------------------------------------
                        # import streamlit as st
                        fig = plot_rendimento_diario_ativo(df_ativos_long)

                        # Exibe o gráfico com o Streamlit, passando a figura

                        df_final = df_final_pl
                        if ativos_lista:
                            # Filtra o df_final com base nos ativos escolhidos
                            df_ativos = df_final
                            df_ativos['Fundo'] = [
                                i.split(' - ')[1] for i in df_ativos.index.to_list()]
                            df_ativos = df_ativos[df_ativos['Fundo'] != 'Total']
                            df_ativos.drop(columns=['Fundo'], inplace=True)
                            df_ativos['Ativo'] = [
                                i.split(' - ')[0] for i in df_ativos.index.to_list()]
                            df_ativos = df_ativos.loc[df_ativos['Ativo'].isin(
                                ativos_lista)]
                            df_ativos = df_ativos.groupby('Ativo').sum()

                            df_ativos_copy = df_ativos.copy()
                            df_ativos_copy.loc['Total'] = df_ativos_copy.sum()
                            for col in df_ativos_copy.columns:
                                df_ativos_copy[col] = df_ativos_copy[col].apply(
                                    lambda x: f"{x:.2f}bps")
                        df_combinado = df_ativos_grana + " / " + df_ativos_copy
                        st.table(df_combinado)
                        # Exibe a tabela filtrada
                        st.pyplot(fig)
                elif visao == "Compilado":
                    data_final = get_last_weekday()
                    df_final = df_final.drop(columns=['Total'])
                    df_final_pl = df_final_pl.drop(columns=['Total'])
                    data_inicial = data_final - datetime.timedelta(days=7)
                    data_inicial = data_inicial.strftime("%Y-%m-%d")
                    data_final = data_final.strftime("%Y-%m-%d")
                    df_final.columns = pd.to_datetime(df_final.columns)
                    df_final_pl.columns = pd.to_datetime(df_final_pl.columns)

                    # Preciso dropar as colunas das datas que estiverem fora do range
                    # Filtrando as colunas do DataFrame de acordo com o intervalo de datas fornecido
                    df_final = df_final.loc[:, (df_final.columns >= pd.to_datetime(
                        data_inicial)) & (df_final.columns <= pd.to_datetime(data_final))]
                    df_final_pl = df_final_pl.loc[:, (df_final_pl.columns >= pd.to_datetime(
                        data_inicial)) & (df_final_pl.columns <= pd.to_datetime(data_final))]

                    # vOLTAR AS COLUNAS PARA O FORMATO ORIGINAL
                    df_final.columns = df_final.columns.strftime('%Y-%m-%d')
                    df_final_pl.columns = df_final_pl.columns.strftime('%Y-%m-%d')
                    df_final = df_final[sorted(
                        df_final.columns, key=pd.to_datetime)]
                    df_final_pl = df_final_pl[sorted(
                        df_final_pl.columns, key=pd.to_datetime)]
                    df_final_copy = df_final.copy()
                    df_final_T = df_final.T  # Transpomos para ter datas como índice
                    df_final_T.index = pd.to_datetime(
                        df_final_T.index)  # Convertendo para datetime

                    # Agrupando por semana, somando rendimentos
                    df_semanal = df_final_T.resample('W').sum()
                    df_semanal = df_semanal.T  # Transpomos de volta
                    df_final = df_semanal
                    # Removendo o horário das colunas
                    df_final.columns = pd.to_datetime(
                        df_final.columns).strftime('%Y-%m-%d')

                    df_final_pl_T = df_final_pl.T  # Transpomos para ter datas como índice
                    df_final_pl_T.index = pd.to_datetime(
                        df_final_pl_T.index)  # Convertendo para datetime

                    # Agrupando por semana, somando rendimentos
                    df_semanal_pl = df_final_pl_T.resample('W').sum()
                    df_semanal_pl = df_semanal_pl.T  # Transpomos de volta
                    df_final_pl = df_semanal_pl
                    # Removendo o horário das colunas
                    df_final_pl.columns = pd.to_datetime(
                        df_final_pl.columns).strftime('%Y-%m-%d')

                    df_semanal = df_final
                    df_semanal_pl = df_final_pl

                    df_final = df_final_copy

                    data_final = get_last_weekday()
                    data_inicial = data_final - datetime.timedelta(days=30)
                    df_final_T = df_final.T  # Transpomos para ter datas como índice
                    data_inicial = data_inicial.strftime("%Y-%m-%d")
                    data_final = data_final.strftime("%Y-%m-%d")
                    df_final.columns = pd.to_datetime(df_final.columns)
                    df_final_pl.columns = pd.to_datetime(df_final_pl.columns)

                    # Preciso dropar as colunas das datas que estiverem fora do range
                    # Filtrando as colunas do DataFrame de acordo com o intervalo de datas fornecido
                    df_final = df_final.loc[:, (df_final.columns >= pd.to_datetime(
                        data_inicial)) & (df_final.columns <= pd.to_datetime(data_final))]
                    df_final_pl = df_final_pl.loc[:, (df_final_pl.columns >= pd.to_datetime(
                        data_inicial)) & (df_final_pl.columns <= pd.to_datetime(data_final))]

                    # vOLTAR AS COLUNAS PARA O FORMATO ORIGINAL
                    df_final.columns = df_final.columns.strftime('%Y-%m-%d')
                    df_final_pl.columns = df_final_pl.columns.strftime('%Y-%m-%d')
                    df_final = df_final[sorted(
                        df_final.columns, key=pd.to_datetime)]
                    df_final_pl = df_final_pl[sorted(
                        df_final_pl.columns, key=pd.to_datetime)]
                    df_final_copy = df_final.copy()
                    df_final_T = df_final.T  # Transpomos para ter datas como índice
                    df_final_T.index = pd.to_datetime(
                        df_final_T.index)  # Convertendo para datetime

                    # Agrupando por mês, somando rendimentos
                    df_mensal = df_final_T.resample('M').sum()
                    df_mensal = df_mensal.T  # Transpomos de volta
                    df_final = df_mensal
                    # Removendo o horário das colunas
                    df_final.columns = pd.to_datetime(
                        df_final.columns).strftime('%Y-%m-%d')

                    df_final_pl_T = df_final_pl.T  # Transpomos para ter datas como índice
                    df_final_pl_T.index = pd.to_datetime(
                        df_final_pl_T.index)  # Convertendo para datetime
                    # Agrupando por mês, somando rendimentos
                    df_mensal_pl = df_final_pl_T.resample('M').sum()
                    df_mensal_pl = df_mensal_pl.T  # Transpomos de volta
                    df_final_pl = df_mensal_pl
                    # Removendo o horário das colunas
                    df_final_pl.columns = pd.to_datetime(
                        df_final_pl.columns).strftime('%Y-%m-%d')

                    df_mensal = df_final
                    df_mensal_pl = df_final_pl

                    # ADICIONAR UMA COLUNA DE TOTAL PARA O DF_FINAL
                    df_semanal['Total'] = df_semanal.sum(axis=1)
                    df_semanal_pl['Total'] = df_semanal_pl.sum(axis=1)
                    df_mensal['Total'] = df_mensal.sum(axis=1)
                    df_mensal_pl['Total'] = df_mensal_pl.sum(axis=1)

                    df_semanal_pl = df_final_pl.applymap(
                        lambda x: f"{x:.2f}bps")
                    df_semanal = df_semanal.applymap(
                        lambda x: f"R${x:,.2f}")

                    df_mensal_pl = df_mensal_pl.applymap(
                        lambda x: f"{x:.2f}bps")
                    df_mensal = df_mensal.applymap(
                        lambda x: f"R${x:,.2f}")

                    df_final_semanal = df_semanal + " / " + df_semanal_pl
                    df_final_mensal = df_mensal + " / " + df_mensal_pl

                    df_final_mensal.rename(columns={'Total': 'MoM'}, inplace=True)
                    df_final_semanal.rename(columns={'Total': 'WoW'}, inplace=True)
                    df_final_combinado = pd.concat(
                        [df_final_semanal['WoW'], df_final_mensal['MoM']], axis=1)
                    st.table(df_final_combinado)
                    st.write('Ainda está em desenvolvimento')

            else:
                st.write("Nenhum Ativo selecionado.")
            att_portifosições()
    elif opti == "Simular Cota":
        simulate_nav_cota()
    else:
        # Função para apagar os dias de dados que o usuário não quer mais
        st.write("## Apagar Dados")
        st.write(
            "Escolha o Ativo e nsira uma data para que os dados registrados dessa data sejam apagados")

        editar_ou_remover()

        # Converter para o formato '2025-01-16'

# ==========================================================
#   FUNÇÃO DA PÁGINA 2 (Entrar Preços de Compra/Venda)
# ==========================================================


# ==========================================================
#   LÓGICA DE ROTEAMENTO DAS "PÁGINAS"
# ==========================================================
st.set_page_config(
    page_title="Dashboard de Análise",
    layout="wide",
    initial_sidebar_state="expanded"
)

add_custom_css()

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "main"

if st.session_state["current_page"] == "main":
    main_page()
else:
    main_page()
# --------------------------------------------------------
