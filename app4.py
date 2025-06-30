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


def process_returns(df, assets):
    df_retorno = df.copy()
    df_retorno = df_retorno[assets]
    df_retorno.dropna(inplace=True)
    df_retorno = df_retorno.astype(float)
    df_retorno = df_retorno.tail(756).reset_index(drop=True)
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
                            if len(colunas_uteis) > 0:
                                # Pegar o valor do ajuste mais recente
                                rendimento = valor_ajuste * qtd_final
                            else:
                                rendimento = 0
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
        st.subheader("Portfólio Atual (já salvo)")
        st.table(df_portifolio_salvo.set_index('Ativo'))

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

                    lista_juros_interno_real = [
                        asset for asset in assets if 'DAP' or 'NTNB' in asset]

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

                lista_juros_interno_real = [
                    asset for asset in assets if 'DAP' or 'NTNB' in asset]

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

    lista_juros_interno_real = [
        asset for asset in assets if 'DAP' or 'NTNB' in asset]

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

 #


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
                            "Ver Portfólio", "Adicionar Ativos", "Remover Ativos"])
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

            lista_juros_interno_real = [
                asset for asset in assets if 'DAP' or 'NTNB' in asset]
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
            st.table(df_fmt2)
            df_b3_fechamento = load_b3_prices()
            ultimo_dia_dados_b3 = df_b3_fechamento.columns[-1]
            ultimo_dia_dados_b3 = datetime.datetime.strptime(
                ultimo_dia_dados_b3, "%Y-%m-%d")

            st.write(
                f"OBS: O preço de compra é o preço médio de compra do ativo -- Atualizado em {ultimo_dia_dados_b3.strftime('%d/%m/%Y')}")
            st.write("---")
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
                df_precos_ajustados = calculate_portfolio_values(
                    df_precos_ajustados, df_pl_processado, var_bps)
                df_pl_processado = calculate_contracts_per_fund(
                    df_pl_processado, df_precos_ajustados)

                # st.session_state["posicoes_temp"] = quantidade_inicial
                # st.session_state["ativos_temp"] = list(
                #     quantidade_inicial.keys())

                # # Botão com callback (1 clique = troca de página)
                # st.button(
                #     "Ir para a tela de Preços de Compra/Venda",
                #     on_click=switch_to_page2,
                #     key="go_page2"
                # )

                # ------------------------------------------------
                #   TABELA DF_PL (FILTROS) & Contratos por Fundo
                # ------------------------------------------------
                st.write("---")
                st.write("## Quantidade de Contratos por Fundo")

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

                if filtro_fundo:
                    filtered_df = filtered_df[filtered_df["Fundos/Carteiras Adm"].isin(
                        filtro_fundo)]
                if filtro_adm:
                    filtered_df = filtered_df[filtered_df["Adm"].isin(filtro_adm)]

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
                if columns:
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
                    st.table(filtered_df[columns])
                    st.write("OBS: Os contratos estão arrendodandos para inteiros.")
                else:
                    st.write("Selecione ao menos uma coluna para exibir os dados.")

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

                        def gg_rendimento_diario_fundos(
                            df_fundos_long: pd.DataFrame,
                            tol: float = 0,
                        ):
                            # -------------------------------------------------- PREPARO DOS DADOS
                            df_plot = (
                                df_fundos_long
                                .copy()
                                # 1️⃣  Converte qualquer coisa que pareça data
                                .assign(
                                    date=pd.to_datetime(
                                        df_fundos_long["date"],      # aceita “Mar 2025” ou “01/03/2025”
                                        dayfirst=True,               # 03/04/2025 → 3-abr-2025
                                        errors="coerce",             # se não der pra converter vira NaT
                                    )
                                )
                                # 2️⃣  Mantém só linhas que têm data válida e rendimento ≠ 0 (ou acima de tol)
                                .dropna(subset=["date", "Rendimento_diario"])
                                .loc[lambda d: d["Rendimento_diario"].abs() > tol]
                            )

                            if df_plot.empty:
                                raise ValueError("Nenhuma linha com Rendimento_diario diferente de zero.")

                            # -------------------------------------------------- ORDENAÇÃO DAS DATAS
                            ordered_dates = df_plot["date"].sort_values().unique()
                            ordered_labels = [d.strftime("%d-%b") for d in ordered_dates]  # 01-Mar

                            df_plot["date_str"] = pd.Categorical(
                                df_plot["date"].dt.strftime("%d-%b"),
                                categories=ordered_labels,
                                ordered=True,
                            )

                            # -------------------------------------------------- GRÁFICO
                            p = (
                                ggplot(df_plot, aes("date_str", "Rendimento_diario", fill="fundo"))
                                + geom_col(position="dodge", width=.8)
                                + labs(
                                    title="Rendimento Diário por Fundo",
                                    x="Data",
                                    y="Rendimento Diário",
                                )
                                + scale_fill_manual(values=palette_fundos)
                                + theme_minimal()
                                + theme(
                                    figure_size=(14, 6),
                                    axis_text_x=element_text(rotation=45, ha="right"),
                                    axis_title_x=element_text(weight="bold"),
                                    axis_title_y=element_text(weight="bold"),
                                    legend_position="bottom",
                                    legend_direction="horizontal",
                                    legend_title=element_text(weight="bold"),
                                    plot_title=element_text(colour="#003366", size=13),
                                )
                            )
                            return p

                        # --------------------------------------------------------------------------- #
                        # 2)  CURVA ACUMULADA POR FUNDO  (linha)
                        # --------------------------------------------------------------------------- #

                        def gg_curva_capital_fundos(df_fundos_long: pd.DataFrame):
                            df = df_fundos_long.copy()
                            df["date"] = pd.to_datetime(df["date"], dayfirst=True)
                            df.sort_values(["fundo", "date"], inplace=True)
                            df["acumulado"] = df.groupby(
                                "fundo")["Rendimento_diario"].cumsum()

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

                            p = (
                                ggplot(df,
                                    aes("date", "acumulado",
                                        colour="fundo", group="fundo"))
                                + geom_line(size=1.2)
                                + labs(title="Rendimento Acumulado – Curva de Capital por Fundo",
                                    x="Data",
                                    y="Valor Acumulado")
                                + scale_color_manual(values=palette_fundos)
                                + scale_x_datetime(date_breaks="1 days",
                                                date_labels="%d‑%b")
                                + theme_minimal()
                                + theme(
                                    figure_size=(14, 6),
                                    plot_background=element_rect(
                                        fill="white", colour=None),
                                    panel_background=element_rect(
                                        fill="white", colour="black"),
                                    panel_grid_major=element_line(
                                        colour="#CCCCCC"),
                                    panel_grid_minor=element_line(
                                        colour="#CCCCCC", linetype="dotted"),
                                    axis_text_x=element_text(
                                        rotation=45, ha="right"),
                                    axis_title_x=element_text(weight="bold"),
                                    axis_title_y=element_text(weight="bold"),
                                    legend_position="bottom",
                                    legend_direction="horizontal",
                                    legend_title=element_text(weight="bold"),
                                    plot_title=element_text(
                                        colour="#003366", size=13)
                                )
                            )
                            return p
                    
                    # Exibe o gráfico com o Streamlit, passando a figura
                    # gráfico de barras
                    fig_diario = gg_rendimento_diario_fundos(df_fundos_long).draw()
                    # curva acumulada
                    fig_acum = gg_curva_capital_fundos(df_fundos_long).draw()
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

                        df_copia_fundos = df_fundos_copy.copy()

                        # ------------------------------------------------------------------------------
                        # Mostra “0.00 bps” sempre que o valor real for zero (ou perto de zero)
                        ZERO_EPS = 0.0005     # 0,00005 bps  → ≈ 1 centavo em PL de 20 MM

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
                        st.pyplot(fig_diario)
                        st.write('### Graficos de Curva de Capital')
                        st.pyplot(fig_acum)

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

                        def gg_rendimento_diario(df_estrategias_long: pd.DataFrame, tol: float = 0):
                            """
                            Plota barras de rendimento diário.
                            • Mostra no eixo‑X apenas as datas que têm pelo menos um valor |rendimento| > tol.
                            • Converte datas para eixo categórico, mantendo ordenação cronológica.
                            """
                            from plotnine import (
                                ggplot, aes, geom_col, geom_line, labs,
                                scale_fill_brewer, scale_color_manual, scale_x_datetime,
                                theme_minimal, theme,
                                element_text, element_rect, element_line
                            )
                            # ---------------------------------------------------------- prepara dados
                            df_plot = (
                                df_estrategias_long
                                .copy()
                                .assign(date=pd.to_datetime(df_estrategias_long["date"], dayfirst=True))
                                # 1. só dados não-nulos
                                .dropna(subset=["Rendimento_diario"])
                                # 2. remove linhas com rendimento ‘nulo’ (zero ou dentro do limite tol)
                                .loc[lambda d: d["Rendimento_diario"].abs() > tol]
                            )

                            # Se nada sobrar, evita erro
                            if df_plot.empty:
                                raise ValueError(
                                    "Nenhuma linha com Rendimento_diario diferente de zero.")

                            # -------------------------------------------------- eixo X categórico
                            # ordem cronológica
                            ordered_dates = sorted(df_plot["date"].unique())
                            df_plot["date_str"] = pd.Categorical(
                                df_plot["date"].dt.strftime(
                                    "%d‑%b"),                  # rótulo
                                categories=[d.strftime("%d‑%b")
                                            for d in ordered_dates],
                                ordered=True
                            )

                            palette_curvas = {
                                "JUROS NOMINAIS BRASIL": "#B03A2E",
                                "JUROS REAIS BRASIL": "#003366",
                                "MOEDAS": "#138D75",
                                "JUROS US": "#7F3C8D"
                            }

                            # --------------------------------------------------------- gráfico
                            p = (
                                ggplot(df_plot, aes(
                                    "date_str", "Rendimento_diario", fill="estratégia"))
                                + geom_col(position="dodge", width=.8)
                                + labs(title="Rendimento Diário por Estratégia",
                                    x="Data", y="Rendimento Diário")
                                + scale_color_manual(values=palette_curvas)
                                + theme_minimal()
                                + theme(
                                    figure_size=(14, 6),
                                    plot_background=element_rect(
                                        fill="white", colour=None),
                                    panel_background=element_rect(
                                        fill="white", colour="black"),
                                    panel_grid_major=element_line(
                                        colour="#CCCCCC"),
                                    panel_grid_minor=element_line(
                                        colour="#CCCCCC", linetype="dotted"),
                                    axis_text_x=element_text(
                                        rotation=90, ha="right"),
                                    axis_title_x=element_text(weight="bold"),
                                    axis_title_y=element_text(weight="bold"),
                                    legend_position="bottom",
                                    legend_direction="horizontal",
                                    legend_box_margin=0,
                                    plot_title=element_text(
                                        colour="#003366", size=13)
                                )
                            )
                            return p

                        # --------------------------------------------------------------------------- #
                        # 2)  GRÁFICO DE LINHA – Curva de Capital (rendimento acumulado)
                        # --------------------------------------------------------------------------- #

                        def gg_curva_capital(df_long: pd.DataFrame):
                            from plotnine import (
                                ggplot, aes, geom_col, geom_line, labs,
                                scale_fill_brewer, scale_color_manual, scale_x_datetime,
                                theme_minimal, theme,
                                element_text, element_rect, element_line
                            )
                            df = df_long.copy()
                            df["date"] = pd.to_datetime(df["date"], dayfirst=True)
                            df.sort_values(["estratégia", "date"], inplace=True)
                            df["acumulado"] = df.groupby("estratégia")[
                                "Rendimento_diario"].cumsum()

                            palette_curvas = {
                                "JUROS NOMINAIS BRASIL": "#B03A2E",
                                "JUROS REAIS BRASIL": "#003366",
                                "MOEDAS": "#138D75",
                                "JUROS US": "#7F3C8D"
                            }

                            p = (
                                ggplot(df,
                                    aes("date", "acumulado", colour="estratégia", group="estratégia"))
                                + geom_line(size=1.2)
                                + labs(title="Rendimento Acumulado – Curva de Capital",
                                    x="Data",
                                    y="Valor Acumulado")
                                + scale_color_manual(values=palette_curvas)
                                # ← 2 dias
                                + scale_x_datetime(date_breaks="2 days",
                                                date_labels="%d‑%b")
                                + theme_minimal()
                                + theme(
                                    figure_size=(14, 6),
                                    plot_background=element_rect(
                                        fill="white", colour=None),
                                    panel_background=element_rect(
                                        fill="white", colour="black"),
                                    panel_grid_major=element_line(
                                        colour="#CCCCCC"),
                                    panel_grid_minor=element_line(
                                        colour="#CCCCCC", linetype="dotted"),
                                    axis_text_x=element_text(
                                        rotation=90, ha="right"),
                                    axis_title_x=element_text(weight="bold"),
                                    axis_title_y=element_text(weight="bold"),
                                    legend_position="bottom",          # legenda central abaixo
                                    legend_direction="horizontal",
                                    legend_box_margin=0,
                                    plot_title=element_text(
                                        colour="#003366", size=13)
                                )
                            )
                            return p
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

                    for col in df_final22.columns:
                        df_final22[col] = df_final22[col].apply(
                            lambda x: f"{x:.2f}bps")

                    # st.write(df_estrategias_copy,df_estrategias_grana,df_final22)
                    df_combinado = df_estrategias_grana + " / " + df_final22
                    df_combinado.loc['Total'] = df_combinado.loc['Total'].str.split('/').str[0].str.strip()
                    # df_combinado = df_combinado.drop('Total', axis=0)
                    # df_combinado.drop(columns=['Total'], inplace=True)
                    # Teste
                    p_barras = gg_rendimento_diario(df_estrategias_long)
                    fig1 = p_barras.draw()          # converte o ggplot em Figure
                    p_curva = gg_curva_capital(df_estrategias_long)
                    fig2 = p_curva.draw()

                    st.write('### Tabela de Rendimento Diário por Estratégia')
                    st.table(df_combinado)
                    st.write('### Gráficos de Rendimento Diário por Estratégia')
                    st.pyplot(fig1)
                    st.write('### Graficos de Curva de Capital')
                    st.pyplot(fig2)
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
