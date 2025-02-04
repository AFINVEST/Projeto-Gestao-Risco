import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from datetime import date

# ==========================================================
#               FUNÇÕES AUXILIARES (MESMAS)
# ==========================================================


def process_portfolio(df_pl, Weights):
    df_pl['PL'] = (
        df_pl['PL']
        .str.replace('R$', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .replace('--', np.nan)
        .astype(float)
    )
    df_pl = df_pl.iloc[[5, 9, 10, 11, 17, 18, 19, 20, 22]]
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


def load_and_process_excel(df_excel, assets_sel):
    df = df_excel.copy()
    df_precos = df.tail(1)
    df_precos['TREASURY'] = df_precos['TREASURY'] * df_precos['WDO1'] / 10000
    df_precos = df_precos[assets_sel]
    return df_precos, df


def load_and_process_divone(file_bbg, df_excel):
    df_divone = pd.read_excel(file_bbg, sheet_name='DIV01',
                              skiprows=1, usecols='E:F', nrows=21)
    df_divone = df_divone.T
    columns = [
        'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
        'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP25', 'DAP26', 'DAP27',
        'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'TREASURY', 'IBOV', 'S&P'
    ]
    df_divone.columns = columns
    df_divone = df_divone.drop(df_divone.index[0])

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
    Exemplo de função que carrega dois dataframes de CSV:
     - df_preco_de_ajuste_atual.csv : preços de fechamento (colunas de datas)
     - df_variacao.csv : variação diária dos ativos (colunas de datas)
    """
    df_b3_fechamento = pd.read_csv("df_preco_de_ajuste_atual.csv")
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
    df_assets = pd.read_csv("portifolio_posições.csv")
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
    Verifica ou cria um CSV de portfólio. Exibe dois DataFrames:
      1) Posição atual salva no CSV.
      2) Novo DataFrame com os ativos e dados recebidos como input.
    Permite concatenar o novo DataFrame ao existente.
    """
    st.write('---')
    st.title("Gestão de Portfólio")
    nome_arquivo_portifolio = 'portifolio_posições.csv'
    df_b3_fechamento = processar_b3_portifolio()

    # Carregar o portfólio existente
    if os.path.exists(nome_arquivo_portifolio):
        df_portifolio_salvo = pd.read_csv(nome_arquivo_portifolio)
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

    # Mudar para a ultima data de fechamento disponivel

    ultimo_fechamento = df_b3_fechamento.columns[-1]
    dolar = df_b3_fechamento.loc[df_b3_fechamento['Assets']
                                 == 'WDO1', ultimo_fechamento].values[0]

    dia_compra = {k: v.strftime('%Y-%m-%d') if isinstance(v, datetime.date) else v for k, v in dia_compra.items()} \
        if isinstance(dia_compra, dict) else \
        dia_compra.strftime('%Y-%m-%d') if isinstance(dia_compra,
                                                      datetime.date) else dia_compra

    # Conferir se dia de compra já foi lançado no portifólio alguma vez
    if os.path.exists(nome_arquivo_portifolio):
        df_portifolio_salvo = pd.read_csv(nome_arquivo_portifolio)

        # Verificar se `dia_compra` é um dicionário
    if isinstance(dia_compra, dict):
        # Pegar o primeiro valor do dicionário
        dia_compra_unico = next(iter(dia_compra.values()))
    else:
        dia_compra_unico = dia_compra

    # Converter o dia de compra para o formato necessário
    dia_compra_unico = pd.to_datetime(dia_compra_unico).strftime('%Y-%m-%d')

    # Conferir se o dia de compra único já existe no DataFrame
    if os.path.exists(nome_arquivo_portifolio):
        if dia_compra_unico in df_portifolio_salvo['Dia de Compra'].values:
            # Dropar dados desse dia
            df_portifolio_salvo = df_portifolio_salvo[
                df_portifolio_salvo['Dia de Compra'] != dia_compra_unico
            ]

    for asset in assets:
        qtd_final = quantidades[asset]
        dia_de_compra_atual = dia_compra[asset] if isinstance(
            dia_compra, dict) else dia_compra

        try:
            # Filtrar os dados no DataFrame de fechamento
            filtro_ativo = (df_b3_fechamento['Assets'] == asset)
            if compra_especifica.get(asset) == None:
                preco_compra = df_b3_fechamento.loc[filtro_ativo,
                                                    dia_de_compra_atual].values[0]
            else:
                preco_compra = compra_especifica[asset]

            preco_fechamento_atual = df_b3_fechamento.loc[df_b3_fechamento["Assets"]
                                                          == asset, ultimo_fechamento].values[0]
            preco_fechamento_atual = pd.to_numeric(
                preco_fechamento_atual, errors='coerce')

            if asset == 'TREASURY':

                rendimento = qtd_final * \
                    (preco_fechamento_atual - preco_compra) * (dolar / 10000)

            else:

                rendimento = qtd_final * \
                    (preco_fechamento_atual - preco_compra)

            # Adicionar linha ao novo DataFrame
            novo_portifolio = pd.concat([novo_portifolio, pd.DataFrame([{
                'Ativo': asset,
                'Quantidade': qtd_final,
                'Dia de Compra': dia_de_compra_atual,
                'Preço de Compra': preco_compra,
                'Preço de Ajuste Atual': preco_fechamento_atual,
                'Rendimento': rendimento
            }])], ignore_index=True)

        except Exception as e:
            st.error(f"Erro ao processar o ativo {asset}: {e}")

    # Exibir os dois DataFrames
    col_p1, col_p3, col_p2 = st.columns([4.9, 0.2, 4.9])
    with col_p1:
        st.subheader("Portfólio Atual")
        st.table(df_portifolio_salvo.set_index('Ativo'))
        df_atual = df_portifolio_salvo.copy()
        df_atual = df_atual.groupby('Ativo').sum().reset_index()
        df_atual.drop(['Dia de Compra', 'Preço de Compra',
                       'Preço de Ajuste Atual'], axis=1, inplace=True)

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

        df_teste = pd.concat(
            [df_portifolio_salvo, novo_portifolio], ignore_index=True)

        df_teste = df_teste.groupby('Ativo').sum().reset_index()
        df_teste.drop(['Dia de Compra', 'Preço de Compra',
                       'Preço de Ajuste Atual'], axis=1, inplace=True)
        # Separar uma lista de ativos e uma lista de quantidades
        assets_teste = df_teste['Ativo'].tolist()
        quantidades_teste = df_teste['Quantidade'].tolist()
        st.subheader("Portifólio processado")
        st.table(df_teste)
    st.write("---")
    df_contratos_2 = read_atual_contratos()
    for col in df_contratos_2.columns:
        df_contratos_2.rename(columns={col: f'Contratos {col}'}, inplace=True)
    # Somar as duas tabelas
    # Garantir que todas as colunas que são relevantes para a soma sejam numéricas
    # Coerce converte valores não numéricos para NaN
    df_contratos = df_contratos.apply(pd.to_numeric, errors='coerce')
    df_contratos_2 = df_contratos_2.apply(
        pd.to_numeric, errors='coerce')  # O mesmo para df_contratos_2

    df_contratos = df_contratos.add(df_contratos_2, fill_value=0)
    # Preciso somar os contratos e   m df_contratos_2 e df_contratos

    df_contratos.drop(['Adm', 'PL', 'PL_atualizado',
                      'Weights'], axis=1, inplace=True)

    col_pp1, col_pp3, col_pp2 = st.columns([4.9, 0.2, 4.9])
    with col_pp1:
        st.write("## Portifólio Atual")
        soma_pl_sem_pesos = calcular_metricas_de_port(
            assets_atual, quantidades_atual, df_contratos_2)
    with col_pp2:
        st.write("## Novo Portifólio")
        soma_pl_sem_pesos_novo = calcular_metricas_de_port(
            assets_teste, quantidades_teste, df_contratos)
        # Botão para concatenar os DataFrames
        if st.button("Salvar novo portfólio"):
            df_portifolio_salvo = pd.concat(
                [df_portifolio_salvo, novo_portifolio], ignore_index=True)
            df_portifolio_salvo.to_csv(nome_arquivo_portifolio, index=False)
            st.success("Novo portfólio salvo com sucesso!")
            st.dataframe(df_portifolio_salvo)
            key = True
        else:
            key = False
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
    return df_portifolio_salvo, key, soma_pl_sem_pesos


def read_atual_contratos():
    files = os.listdir('BaseFundos')
    df_fundos = pd.DataFrame()
    lista_files = []
    for file in files:
        df_fundos = pd.concat([df_fundos, pd.read_csv(f'BaseFundos/{file}')])
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
    df_portifolio = pd.read_csv('portifolio_posições.csv')
    for ativo in df_portifolio['Ativo']:
        if ativo in df_fechamento_b3['Assets'].values:
            preco_atual = df_fechamento_b3.loc[df_fechamento_b3['Assets']
                                               == ativo, df_fechamento_b3.columns[-1]].values[0]
            df_portifolio.loc[df_portifolio['Ativo'] ==
                              ativo, 'Preço de Ajuste Atual'] = preco_atual

    df_portifolio.to_csv('portifolio_posições.csv', index=False)
    return


def calcular_metricas_de_port(assets, quantidades, df_contratos):

    file_pl = "pl_fundos.csv"
    df_pl = pd.read_csv(file_pl, index_col=0)
    file_bbg = "BBG - ECO DASH.xlsx"

    # Dicionário de pesos fixo (pode-se tornar dinâmico no futuro)
    dict_pesos = {
        'GLOBAL BONDS': 4,
        'HORIZONTE': 1,
        'JERA2026': 2,
        'REAL FIM': 2,
        'BH FIRF INFRA': 2,
        'BORDEAUX INFRA': 2,
        'TOPAZIO INFRA': 2,
        'MANACA INFRA FIRF': 2,
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
        'TREASURY_AJUSTADA', 'S&P'
    ]
    df.drop(['IBOV', 'S&P', 'TREASURY_AJUSTADA'], axis=1, inplace=True)
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
        'BBG - ECO DASH.xlsx', df)
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
        asset for asset in assets if 'DAP' in asset]

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

    stress_test_juros_interno_Reais = df_divone_juros_real * 100
    stress_test_juros_interno_Reais_percent = stress_test_juros_interno_Reais / \
        soma_pl_sem_pesos * 10000

    df_divone_juros_externo_certo = df_divone_juros_externo

    if lista_juros_externo:
        df_divone_juros_externo = df_retorno['TREASURY'].min()
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
        df_divone_dolar = df_retorno['WDO1'].min()
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
        'Stress (R$)': [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo + stress_dolar):,.2f}"] if lista_juros_externo else [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais['FUT_TICK_VAL']) + abs(stress_test_juros_externo['FUT_TICK_VAL']) + abs(stress_dolar):,.2f}"],
        'Stress (bps)': [f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent) + abs(stress_dolar_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']) + abs(stress_test_juros_externo_percent['FUT_TICK_VAL']) + abs(stress_dolar_percent):,.2f}bps"]
    }, index=['Total'])
    df_stress_div01 = pd.concat([df_stress_div01, sum_row])

    df_precos_ajustados = calculate_portfolio_values(
        df_precos_ajustados, df_pl_processado, var_bps)
    df_pl_processado = calculate_contracts_per_fund(
        df_pl_processado, df_precos_ajustados)

    # --- Layout ---
    st.write("## Dados do Portifólio")
    st.write(f"**PL: R$ {soma_pl_sem_pesos:,.0f}**")

    var_limite_comparativo = soma_pl_sem_pesos * var_bps * var_limite
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
    Verifica (ou cria) um CSV de portfólio e atualiza as posições de acordo com:
      1) Lista de ativos (assets) e suas quantidades (quantidades).
      2) Dicionário de compra_especifica (caso o usuário tenha quantidades específicas para algum ativo).
      3) Dia de compra, que pode ser um único valor ou um dicionário.
      4) Atualiza Preço de Compra, Preço de Ajuste Atual, Variação e Rendimento.
    """
    nome_arquivo_portifolio = 'portifolio_posições.csv'
    df_b3_fechamento = processar_b3_portifolio()
    # 1) Carrega portfólio existente (se existir)
    if os.path.exists(nome_arquivo_portifolio):
        df_portifolio = pd.read_csv(nome_arquivo_portifolio, index_col=0)

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

    # Monta a string da data de hoje no mesmo formato das colunas do CSV
    # As colunas do CSV podem estar em formato "dd/mm/yyyy" ou "yyyy-mm-dd", dependendo de como foi salvo
    # Ajuste conforme a forma em que foram salvas:

    ############################## ---- PROBLEMA ---- ##############################
    #
    data_hoje_str = datetime.date.today().strftime('%Y-%m-%d')

    for asset in df_portifolio.index:
        try:
            # Dia de compra que foi salvo
            dia_compra_ativo = df_portifolio.loc[asset, 'Dia de Compra']
            # (Opcional) se for "yyyy-mm-dd", converter para "dd/mm/yyyy" para casar com o CSV
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

    # 5) Salva o DataFrame atualizado de volta no CSV
    df_portifolio.to_csv(nome_arquivo_portifolio)

    return df_portifolio


def atualizar_csv_fundos(
    df_current,         # DataFrame do dia atual (1 linha por Fundo)
    dia_operacao,       # Exemplo: "2025-01-20"
    # DF de transações: [Ativo, Quantidade, Dia de Compra, Preço de Compra, ...]
    df_info,
    # DF de preços de fechamento B3: colunas ["Assets", <data1>, <data2>, ...]
):

    df_fechamento_b3 = pd.read_csv("df_preco_de_ajuste_atual.csv")
    df_fechamento_b3 = df_fechamento_b3.replace('\.', '', regex=True)
    df_fechamento_b3 = df_fechamento_b3.replace({',': '.'}, regex=True)
    # Converter para float todas as colunas menos a primeira
    df_fechamento_b3.iloc[:, 2:] = df_fechamento_b3.iloc[:, 2:].astype(float)

    # Multiplicar a linha que tem o Ativo TREASURY por 1000
    # Multiplicar a linha da treasury por 1000
    df_fechamento_b3.loc[df_fechamento_b3['Assets'] == 'TREASURY', df_fechamento_b3.columns !=
                         'Assets'] = df_fechamento_b3.loc[df_fechamento_b3['Assets'] == 'TREASURY', df_fechamento_b3.columns != 'Assets'] * 1000

    df_fechamento_b3.loc[df_fechamento_b3['Assets'] == 'WDO1', df_fechamento_b3.columns !=
                         'Assets'] = df_fechamento_b3.loc[df_fechamento_b3['Assets'] == 'WDO1', df_fechamento_b3.columns != 'Assets'] * 10

    ultimo_fechamento = df_fechamento_b3.columns[-1]
    dolar = df_fechamento_b3.loc[df_fechamento_b3['Assets']
                                 == 'WDO1', ultimo_fechamento].values[0]

    pl_dias = pd.read_csv("pl_fundos_teste.csv", index_col=0)

    pl_dias = pl_dias.replace('\.', '', regex=True)
    pl_dias = pl_dias.replace({',': '.'}, regex=True)
    for col in pl_dias.columns:
        if col != 'Fundos/Carteiras Adm':
            pl_dias[col] = pl_dias[col].str.replace('R$', '')
            pl_dias[col] = pl_dias[col].replace('--', np.nan)
            pl_dias[col] = pl_dias[col].astype(float, errors='ignore')

    # Atualizar todas as colunas para float menos a primeira
    pl_dias = pl_dias.set_index("Fundos/Carteiras Adm")
    for fundo, row_fundo in df_current.iterrows():
        # Caminho do CSV do Fundo
        nome_arquivo_csv = os.path.join("BaseFundos", f"{fundo}.csv")
        # 2.1) Carregar (ou criar) o DataFrame histórico do Fundo (df_fundo)
        if os.path.exists(nome_arquivo_csv):
            df_fundo = pd.read_csv(nome_arquivo_csv, index_col=None)
            # Conferir se já existem dados para o dia de operação
            if df_fundo.columns.str.startswith(dia_operacao).any():
                df_fundo = df_fundo.drop(
                    columns=df_fundo.columns[df_fundo.columns.str.startswith(dia_operacao)])
            # 2.2) Garante que "Ativo" seja índice (mas mantendo a coluna)
            if "Ativo" in df_fundo.columns:
                df_fundo.set_index("Ativo", inplace=True, drop=False)
        else:
            df_fundo = pd.DataFrame(columns=["Ativo",
                                             "Preco_Fechamento_Atual"])
            # 2.2) Garante que "Ativo" seja índice (mas mantendo a coluna)
            if "Ativo" in df_fundo.columns:
                df_fundo.set_index("Ativo", inplace=True, drop=False)
        subset = df_info[df_info["Dia de Compra"] == dia_operacao]
        lista_assets = subset["Ativo"].unique()
        for asset in lista_assets:
            # Se o ativo ainda não existe no df_fundo, cria a linha
            if asset not in df_fundo.index:
                df_fundo.loc[asset, "Ativo"] = asset
                if fundo == "Total":
                    df_fundo.loc[asset,
                                 f"{dia_operacao} - PL"] = pl_dias.loc['TOTAL', dia_operacao]
                else:
                    df_fundo.loc[asset,
                                 f"{dia_operacao} - PL"] = pl_dias.loc[fundo, dia_operacao]

                # Garantir que o valor seja numérico
                preco_fechamento_atual = df_fechamento_b3.loc[df_fechamento_b3["Assets"]
                                                              == asset, ultimo_fechamento].values[0]
                preco_fechamento_atual = pd.to_numeric(
                    preco_fechamento_atual, errors='coerce')
                preco_fechamento_dia = df_fechamento_b3.loc[df_fechamento_b3["Assets"]
                                                            == asset, dia_operacao].values[0]
                preco_fechamento_dia = pd.to_numeric(
                    preco_fechamento_dia, errors='coerce')

                df_fundo.loc[asset,
                             f"{dia_operacao} - Preco_Fechamento"] = preco_fechamento_dia

                df_fundo.loc[asset,
                             "Preco_Fechamento_Atual"] = preco_fechamento_atual

                preco_compra = df_info.loc[(df_info["Ativo"] == asset) & (
                    df_info["Dia de Compra"] == dia_operacao), "Preço de Compra"].values[0]
                preco_compra = pd.to_numeric(preco_compra, errors='coerce')
                df_fundo.loc[asset,
                             f"{dia_operacao} - Preco_Compra"] = preco_compra

                quantidade = row_fundo[f'Contratos {asset}']
                quantidade = pd.to_numeric(quantidade, errors='coerce')
                df_fundo.loc[asset,
                             f'{dia_operacao} - Quantidade'] = quantidade

                # Calcular o rendimento
                if asset == 'TREASURY':
                    rendimento = preco_fechamento_dia - preco_compra
                    df_fundo.loc[asset,
                                 f'{dia_operacao} - Rendimento'] = quantidade * rendimento * dolar / 10000
                else:
                    rendimento = preco_fechamento_dia - preco_compra
                    df_fundo.loc[asset,
                                 f'{dia_operacao} - Rendimento'] = quantidade * rendimento

            else:
                if fundo == "Total":
                    df_fundo.loc[asset,
                                 f"{dia_operacao} - PL"] = pl_dias.loc['TOTAL', dia_operacao]
                else:
                    df_fundo.loc[asset,
                                 f"{dia_operacao} - PL"] = pl_dias.loc[fundo, dia_operacao]

                # Garantir que o valor seja numérico
                preco_fechamento_atual = df_fechamento_b3.loc[df_fechamento_b3["Assets"]
                                                              == asset, ultimo_fechamento].values[0]
                preco_fechamento_atual = pd.to_numeric(
                    preco_fechamento_atual, errors='coerce')

                preco_fechamento_dia = df_fechamento_b3.loc[df_fechamento_b3["Assets"]
                                                            == asset, dia_operacao].values[0]
                preco_fechamento_dia = pd.to_numeric(
                    preco_fechamento_dia, errors='coerce')

                df_fundo.loc[asset,
                             f"{dia_operacao} - Preco_Fechamento"] = preco_fechamento_dia

                df_fundo.loc[asset,
                             "Preco_Fechamento_Atual"] = preco_fechamento_atual

                preco_compra = df_info.loc[(df_info["Ativo"] == asset) & (
                    df_info["Dia de Compra"] == dia_operacao), "Preço de Compra"].values[0]
                preco_compra = pd.to_numeric(preco_compra, errors='coerce')
                df_fundo.loc[asset,
                             f"{dia_operacao} - Preco_Compra"] = preco_compra

                quantidade = row_fundo[f'Contratos {asset}']
                quantidade = pd.to_numeric(quantidade, errors='coerce')
                df_fundo.loc[asset,
                             f'{dia_operacao} - Quantidade'] = quantidade

                # Calcular o rendimento
                if asset == 'TREASURY':
                    rendimento = preco_fechamento_dia - preco_compra
                    df_fundo.loc[asset,
                                 f'{dia_operacao} - Rendimento'] = quantidade * rendimento * dolar / 10000
                else:
                    rendimento = preco_fechamento_dia - preco_compra
                    df_fundo.loc[asset,
                                 f'{dia_operacao} - Rendimento'] = quantidade * rendimento

        # Pegar o Preco de compra de cada ativo
        df_fundo.reset_index(drop=True, inplace=True)
        df_fundo.to_csv(nome_arquivo_csv, index=False, encoding="utf-8")

        print(f"[{fundo}] -> CSV atualizado: {nome_arquivo_csv}")


def analisar_performance_fundos(
    data_inicial,
    data_final,
    lista_estrategias,
    lista_ativos
):
    """
    Lê todos os arquivos CSV na pasta 'BaseFundos', extrai colunas diárias
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
        pasta_fundos) if arq.endswith(".csv")]

    registros = []
    # Cada elemento em 'registros' será um dict:
    # { "date": dt_col, "fundo": nome_fundo, "Ativo": ativo, "Rendimento_diario": rend_val }

    for arquivo_csv in arquivos:
        caminho_csv = os.path.join(pasta_fundos, arquivo_csv)
        nome_fundo = arquivo_csv.replace(".csv", "")

        df_fundo = pd.read_csv(caminho_csv)
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


def apagar_dados_data(data_apag):
    """
    Apaga os dados de um dia específico de todos os arquivos CSV na pasta 'BaseFundos'.
    """

    pasta_fundos = "BaseFundos"

    if not os.path.isdir(pasta_fundos):
        raise FileNotFoundError(f"Pasta '{pasta_fundos}' não encontrada.")

    arquivos = [arq for arq in os.listdir(
        pasta_fundos) if arq.endswith(".csv")]

    for arquivo_csv in arquivos:
        caminho_csv = os.path.join(pasta_fundos, arquivo_csv)
        nome_fundo = arquivo_csv.replace(".csv", "")

        df_fundo = pd.read_csv(caminho_csv)
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
            os.remove(caminho_csv)
        else:
            df_fundo.to_csv(caminho_csv, index=False, encoding="utf-8")

        print(f"[{nome_fundo}] -> CSV atualizado: {caminho_csv}")

    # Atualizar portifólio_posições.csv
    nome_arquivo_portifolio = 'portifolio_posições.csv'
    df_portifolio = pd.read_csv(nome_arquivo_portifolio, index_col=0)
    # Dropar todas as linhas que a coluna "Dia de Compra" SEJA igual a data_apag
    df_portifolio = df_portifolio[df_portifolio['Dia de Compra'] != data_apag]
    df_portifolio.to_csv(nome_arquivo_portifolio)
    print(f"[{nome_arquivo_portifolio}] -> CSV atualizado: {nome_arquivo_portifolio}")


def analisar_dados_fundos():
    files = os.listdir('BaseFundos')
    df_b3_fechamento = pd.read_csv("df_preco_de_ajuste_atual.csv")
    df_b3_fechamento = df_b3_fechamento.replace('\.', '', regex=True)
    df_b3_fechamento = df_b3_fechamento.replace(',', '.', regex=True)
    df_b3_fechamento.iloc[:, 1:] = df_b3_fechamento.iloc[:, 1:].astype(float)
    df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'TREASURY', df_b3_fechamento.columns !=
                         'Assets'] = df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'TREASURY', df_b3_fechamento.columns != 'Assets'] * 1000
    df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'WDO1', df_b3_fechamento.columns !=
                         'Assets'] = df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'WDO1', df_b3_fechamento.columns != 'Assets'] * 10

    df_final = pd.DataFrame()
    dia_atual = df_b3_fechamento.columns[-1]
    dolar = df_b3_fechamento.loc[df_b3_fechamento['Assets']
                                 == 'WDO1', dia_atual].values[0]

    # Supõe-se que `files`, `df_b3_fechamento`, e `dia_atual` estão definidos
    for file in files:
        # Lê o arquivo CSV
        df_fundos = pd.read_csv(f'BaseFundos/{file}')
        file = file.split('.')[0]  # Remove a extensão do nome do arquivo

        # Configura o índice para a coluna 'Ativo'
        df_fundos.set_index('Ativo', inplace=True)

        # Itera pelas linhas do DataFrame
        for idx, row in df_fundos.iterrows():
            # DataFrame para armazenar os rendimentos por operação e por dia
            df_rendimentos = pd.DataFrame()
            # Identifica as colunas relacionadas a 'Quantidade' e 'Preço de Compra'
            col_quantidade = [
                col for col in df_fundos.columns if col.endswith('Quantidade')]

            # Processa cada operação (quantidade/compras de cada dia)
            for col in col_quantidade:
                quantidade = row[col]

                # Ignorar operações com quantidade 0
                if quantidade == 0 or pd.isna(quantidade):
                    continue

                # Obtém o preço de compra correspondente
                preco_compra = row[col.replace('Quantidade', 'Preco_Compra')]

                # Preco Anterior
                preco_anterior = preco_compra
                # Extrai a data da coluna
                data_operacao = col.split(' ')[0]

                # Itera por cada data no DataFrame de fechamentos (até o dia atual)
                # Ignora a coluna de "Assets"
                for data_fechamento in df_b3_fechamento.columns[1:]:
                    # Calcula o rendimento apenas se a data de fechamento for posterior à data de operação
                    if datetime.datetime.strptime(data_fechamento, '%Y-%m-%d') >= datetime.datetime.strptime(data_operacao, '%Y-%m-%d'):
                        # Preço de fechamento no dia específico
                        preco_fechamento = df_b3_fechamento.loc[
                            df_b3_fechamento["Assets"] == idx, data_fechamento
                        ].values[0]
                        if idx == 'TREASURY':
                            # Calcula o rendimento
                            rendimento = (
                                preco_fechamento - preco_anterior) * quantidade * dolar / 10000
                        else:
                            rendimento = (preco_fechamento -
                                          preco_anterior) * quantidade

                        # Adiciona o rendimento ao DataFrame de resultados
                        df_rendimentos.loc[f"{idx} - {data_operacao}",
                                           data_fechamento] = rendimento

                        preco_anterior = preco_fechamento

            # Verifica se há dados no DataFrame `df_rendimentos`
            if not df_rendimentos.empty:
                # Adicionar uma linha de total
                df_rendimentos.loc['Total'] = df_rendimentos.sum()

                # Dropar as linhas que não sejam 'Total'
                df_rendimentos_append = df_rendimentos.loc[['Total']].copy()

                # Renomear a linha Total
                df_rendimentos_append.rename(
                    index={'Total': f'{idx} - {file} - P&L'}, inplace=True)

                # Adicionar a nova linha ao DataFrame final
                df_final = pd.concat([df_final, df_rendimentos_append])

    df_final_pl = pd.DataFrame()

    # Supõe-se que `files`, `df_b3_fechamento`, e `dia_atual` estão definidos
    for file in files:
        # Lê o arquivo CSV
        df_fundos = pd.read_csv(f'BaseFundos/{file}')
        file = file.split('.')[0]  # Remove a extensão do nome do arquivo

        # Configura o índice para a coluna 'Ativo'
        df_fundos.set_index('Ativo', inplace=True)

        # Itera pelas linhas do DataFrame
        for idx, row in df_fundos.iterrows():
            # DataFrame para armazenar os rendimentos por operação e por dia
            df_rendimentos = pd.DataFrame()
            # Identifica as colunas relacionadas a 'Quantidade' e 'Preço de Compra'
            col_quantidade = [
                col for col in df_fundos.columns if col.endswith('Quantidade')]

            # Processa cada operação (quantidade/compras de cada dia)
            for col in col_quantidade:
                quantidade = row[col]

                # Ignorar operações com quantidade 0
                if quantidade == 0 or pd.isna(quantidade):
                    continue

                # Obtém o preço de compra correspondente
                preco_compra = row[col.replace('Quantidade', 'Preco_Compra')]
                soma_pl = row[col.replace('Quantidade', 'PL')]
                # Extrai a data da coluna
                data_operacao = col.split(' ')[0]
                preco_anterior = preco_compra
                # Itera por cada data no DataFrame de fechamentos (até o dia atual)
                # Ignora a coluna de "Assets"
                for data_fechamento in df_b3_fechamento.columns[1:]:
                    # Calcula o rendimento apenas se a data de fechamento for posterior à data de operação
                    if datetime.datetime.strptime(data_fechamento, '%Y-%m-%d') >= datetime.datetime.strptime(data_operacao, '%Y-%m-%d'):
                        # Preço de fechamento no dia específico
                        preco_fechamento = df_b3_fechamento.loc[
                            df_b3_fechamento["Assets"] == idx, data_fechamento
                        ].values[0]

                        if idx == 'TREASURY':
                            # Calcula o rendimento
                            rendimento = (
                                preco_fechamento - preco_anterior) * quantidade * dolar / 10000
                        else:
                            rendimento = (preco_fechamento -
                                          preco_anterior) * quantidade

                        rendimento = (rendimento / soma_pl) * 10000

                        # Adiciona o rendimento ao DataFrame de resultados
                        df_rendimentos.loc[f"{idx}  EM BIPS - {data_operacao}",
                                           data_fechamento] = rendimento

                        preco_anterior = preco_fechamento
            # Verifica se há dados no DataFrame `df_rendimentos`
            if not df_rendimentos.empty:
                # Adicionar uma linha de total
                df_rendimentos.loc['Total'] = df_rendimentos.sum()

                # Dropar as linhas que não sejam 'Total'
                df_rendimentos_append = df_rendimentos.loc[['Total']].copy()

                # Renomear a linha Total
                df_rendimentos_append.rename(
                    index={'Total': f'{idx} - {file} - P&L'}, inplace=True)

                # Adicionar a nova linha ao DataFrame final
                df_final_pl = pd.concat([df_final_pl, df_rendimentos_append])

    return df_final, df_final_pl


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


# ==========================================================
#   FUNÇÃO DA PÁGINA 1 (Dashboard Principal)
# ==========================================================
def main_page():
    st.title("Dashboard de Análise de Risco de Portifólio")

    file_pl = "pl_fundos.csv"
    df_pl = pd.read_csv(file_pl, index_col=0)
    file_bbg = "BBG - ECO DASH.xlsx"

    # Dicionário de pesos fixo (pode-se tornar dinâmico no futuro)
    dict_pesos = {
        'Global Bonds': 4,
        'HORIZONTE': 1,
        'JERA2026': 2,
        'REAL FIM': 2,
        'BH FIRF INFRA': 2,
        'BORDEAUX INFRA': 2,
        'TOPAZIO': 2,
        'MANACA INFRA FIRF': 2,
        'AF DEB INCENTIVADA': 3
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
        'TREASURY_AJUSTADA', 'S&P'
    ]
    df.drop(['IBOV', 'S&P', 'TREASURY_AJUSTADA'], axis=1, inplace=True)

    default_assets, quantidade_inicial, portifolio_default = processar_dados_port()
    st.sidebar.write("## OPÇÕES DO DASHBOARD")
    opti = st.sidebar.radio("Escolha uma opção:", [
                            "Ver Portifólio", "Adicionar Ativos", "Remover Ativos"])
    if opti == "Adicionar Ativos":
        st.sidebar.write("## Ativos do Portifólio")

        default_or_not = st.sidebar.checkbox(
            "Usar ativos portifólio", value=True)

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
            df_b3_fechamento = pd.read_csv("df_preco_de_ajuste_atual.csv")
            df_b3_fechamento = df_b3_fechamento.replace('\.', '', regex=True)
            df_b3_fechamento = df_b3_fechamento.replace(',', '.', regex=True)
            df_b3_fechamento.iloc[:, 1:] = df_b3_fechamento.iloc[:, 1:].astype(
                float)
            df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'TREASURY', df_b3_fechamento.columns !=
                                 'Assets'] = df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'TREASURY', df_b3_fechamento.columns != 'Assets'] * 1000
            df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'WDO1', df_b3_fechamento.columns !=
                                 'Assets'] = df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'WDO1', df_b3_fechamento.columns != 'Assets'] * 10
            ultimo_dia_dados_b3 = df_b3_fechamento.columns[-1]
            ultimo_dia_dados_b3 = datetime.datetime.strptime(
                ultimo_dia_dados_b3, "%Y-%m-%d")

            data_compra_todos = st.sidebar.date_input(
                "Dia de Compra dos Ativos:", value=ultimo_dia_dados_b3)
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
                asset for asset in assets if 'DAP' in asset]
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

            stress_test_juros_interno_Reais = df_divone_juros_real * 100
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
                atualizar_csv_fundos(
                    filtered_df, data_compra_todos, df_port)
            with cool3:
                st.write("### Portifólio Atualizado")
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
    elif opti == "Ver Portifólio":
        # Agrupamento corrigido
        df_portifolio_default = portifolio_default.groupby("Ativo").agg({
            "Quantidade": "sum",  # Soma as quantidades
            # Média ponderada
            "Preço de Compra": lambda x: (x * portifolio_default.loc[x.index, "Quantidade"]).sum() / portifolio_default.loc[x.index, "Quantidade"].sum(),
            "Preço de Ajuste Atual": "mean",  # Exemplo, calcula a média
            "Rendimento": "sum"  # Soma os rendimentos
        }).reset_index()
        # Adicionar linha de soma
        sum_row = df_portifolio_default.select_dtypes(include='number').sum()
        sum_row['Ativo'] = 'Total'
        df_portifolio_default = pd.concat(
            [df_portifolio_default, sum_row.to_frame().T], ignore_index=True)
        df_portifolio_default.set_index('Ativo', inplace=True)
        st.subheader("Resumo do Portifólio Atual")
        df_portifolio_default_copy = df_portifolio_default.copy()
        # formatar
        df_portifolio_default_copy['Quantidade'] = df_portifolio_default_copy['Quantidade'].apply(
            lambda x: f"{x:.0f}")
        df_portifolio_default_copy['Preço de Compra'] = df_portifolio_default_copy['Preço de Compra'].apply(
            lambda x: f"R${x:,.2f}")
        df_portifolio_default_copy['Preço de Ajuste Atual'] = df_portifolio_default_copy['Preço de Ajuste Atual'].apply(
            lambda x: f"R${x:,.2f}")
        df_portifolio_default_copy['Rendimento'] = df_portifolio_default_copy['Rendimento'].apply(
            lambda x: f"R${x:,.2f}")

        st.table(df_portifolio_default_copy)
        st.write("OBS: O preço de compra é o preço médio de compra do ativo.")
        st.write("---")
        quantidade = []

        df_contratos = read_atual_contratos()

        file_pl = "pl_fundos.csv"
        df_pl = pd.read_csv(file_pl, index_col=0)
        file_bbg = "BBG - ECO DASH.xlsx"

        # Dicionário de pesos fixo (pode-se tornar dinâmico no futuro)
        dict_pesos = {
            'GLOBAL BONDS': 4,
            'HORIZONTE': 1,
            'JERA2026': 2,
            'REAL FIM': 2,
            'BH FIRF INFRA': 2,
            'BORDEAUX INFRA': 2,
            'TOPAZIO INFRA': 2,
            'MANACA INFRA FIRF': 2,
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

        if default_assets:
            df_precos, df_completo = load_and_process_excel(df, default_assets)
            df_retorno = process_returns(df_completo, default_assets)
            var_ativos = var_not_parametric(df_retorno).abs()
            df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)
            quantidade = np.array(quantidade)
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
                asset for asset in default_assets if 'DI' in asset]
            df_divone_juros_nominais = df_divone[lista_juros_interno]

            lista_quantidade = [quantidade_inicial[asset]
                                for asset in lista_juros_interno]
            df_divone_juros_nominais = df_divone_juros_nominais * \
                np.array(lista_quantidade)
            df_divone_juros_nominais = df_divone_juros_nominais.sum(axis=1)

            lista_juros_interno_real = [
                asset for asset in default_assets if 'DAP' in asset]

            df_divone_juros_real = df_divone[lista_juros_interno_real]

            lista_quantidade = [quantidade_inicial[asset]
                                for asset in lista_juros_interno_real]

            df_divone_juros_real = df_divone_juros_real * \
                np.array(lista_quantidade)

            df_divone_juros_real = df_divone_juros_real.sum(axis=1)

            lista_juros_externo = [
                asset for asset in default_assets if 'TREASURY' in asset]

            df_divone_juros_externo = df_divone[lista_juros_externo]

            lista_quantidade = [quantidade_inicial[asset]
                                for asset in lista_juros_externo]

            df_divone_juros_externo = df_divone_juros_externo * \
                np.array(lista_quantidade)

            df_divone_juros_externo = df_divone_juros_externo.sum(axis=1)

            stress_test_juros_interno_Nominais = df_divone_juros_nominais * 100
            stress_test_juros_interno_Nominais_percent = stress_test_juros_interno_Nominais / \
                soma_pl_sem_pesos * 10000

            stress_test_juros_interno_Reais = df_divone_juros_real * 100
            stress_test_juros_interno_Reais_percent = stress_test_juros_interno_Reais / \
                soma_pl_sem_pesos * 10000

            df_divone_juros_externo_certo = df_divone_juros_externo

            if lista_juros_externo:
                df_divone_juros_externo = df_retorno['TREASURY'].min()
                df_divone_juros_externo = abs(
                    df_divone_juros_externo) * treasury * dolar / 10000
                df_divone_juros_externo = df_divone_juros_externo * \
                    np.array(lista_quantidade)
                df_divone_juros_externo = df_divone_juros_externo.sum()

            stress_test_juros_externo = df_divone_juros_externo

            stress_test_juros_externo_percent = stress_test_juros_externo / \
                soma_pl_sem_pesos * 10000

            lista_dolar = [
                asset for asset in default_assets if 'WDO1' in asset]
            if lista_dolar:
                quantidade_dolar = quantidade_inicial[lista_dolar[0]]
                stress_dolar = quantidade_dolar * dolar * 0.02
                df_divone_dolar = df_retorno['WDO1'].min()
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
                'DIV01': [f"R${abs(df_divone_juros_nominais.iloc[0] + df_divone_juros_real[0] + df_divone_juros_externo_certo.iloc[0] + df_divone_dolar.iloc[0]):,.2f}"] if lista_dolar else [f"R${df_divone_juros_nominais.iloc[0] + df_divone_juros_real[0] + df_divone_juros_externo_certo.iloc[0]:,.2f}"],
                'Stress (R$)': [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL'] + stress_test_juros_interno_Reais['FUT_TICK_VAL'] + stress_test_juros_externo + stress_dolar):,.2f}"] if lista_juros_externo else [f"R${abs(stress_test_juros_interno_Nominais['FUT_TICK_VAL'] + stress_test_juros_interno_Reais['FUT_TICK_VAL'] + stress_test_juros_externo['FUT_TICK_VAL'] + stress_dolar):,.2f}"],
                'Stress (bps)': [f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL'] + stress_test_juros_interno_Reais_percent['FUT_TICK_VAL'] + stress_test_juros_externo_percent + stress_dolar_percent):,.2f}bps" if lista_juros_externo else f"{abs(stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL'] + stress_test_juros_interno_Reais_percent['FUT_TICK_VAL'] + stress_test_juros_externo_percent['FUT_TICK_VAL'] + stress_dolar_percent):,.2f}bps"]
            }, index=['Total'])
            df_stress_div01 = pd.concat([df_stress_div01, sum_row])

            df_precos_ajustados = calculate_portfolio_values(
                df_precos_ajustados, df_pl_processado, var_bps)
            df_pl_processado = calculate_contracts_per_fund(
                df_pl_processado, df_precos_ajustados)

            # --- Layout ---
            col1, col11, col2, col3 = st.columns([2.4, 0.2, 3.4, 1])
            with col3:
                st.write("Escolha as colunas a exibir:")
                beta = st.checkbox("Exibir Beta", value=False)
                mvar = st.checkbox("Exibir MVar (R$)", value=True)
                covar_rs = st.checkbox("Exibir CoVaR (R$)", value=True)
                covar_perce = st.checkbox("Exibir CoVaR (%)", value=True)
                var_check = st.checkbox("Exibir VaR", value=False)
                perc_ris_tot = st.checkbox(
                    "Exibir % do Risco Total", value=True)

            with col1:
                st.write("## Dados do Portifólio")
                st.write(f"**PL: R$ {soma_pl_sem_pesos:,.0f}**")

                if var_din:
                    st.write(f"**VaR Limite**: **R${var_lim_din:,.0f}**")
                    var_limite_comparativo = var_lim_din
                else:
                    var_limite_comparativo = soma_pl_sem_pesos * var_bps * var_limite
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

            with col11:
                st.html(
                    '''
                    <div class="divider-vertical-line"></div>
                    <style>
                        .divider-vertical-line {
                            border-left: 2px solid rgba(49, 51, 63, 0.2);
                            height: 40vh;
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

            with col2:
                df_dados = pd.DataFrame({
                    'Beta': df_beta,
                    'MVar(R$)': df_mvar_dinheiro,
                    'CoVaR(R$)': covar,
                    'CoVaR(%)': covar_perc,
                    'Var': var_ativos[default_assets],
                    '% do Risco Total': covar_perc * abs(covar.sum() / var_limite_comparativo)
                })

                colunas_selecionadas = []
                if beta:
                    colunas_selecionadas.append('Beta')
                if mvar:
                    colunas_selecionadas.append('MVar(R$)')
                if covar_rs:
                    colunas_selecionadas.append('CoVaR(R$)')
                if covar_perce:
                    colunas_selecionadas.append('CoVaR(%)')
                if var_check:
                    colunas_selecionadas.append('Var')
                if perc_ris_tot:
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
                    sum_row['Var'] = var_ativos[default_assets].sum()
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
                else:
                    st.write("Nenhuma coluna selecionada.")

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
            df_contratos = read_atual_contratos()
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
                'TREASURY': 'Juros US',
                'WDO1': 'Moedas'
            }
            lista_ativos = [
                'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
                'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP25', 'DAP26', 'DAP27',
                'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'TREASURY'
            ]
            st.write('---')
            st.title("Análise de Performance dos Fundos")

            # --- Seletor de Filtro de Tempo ---
            visao = st.sidebar.selectbox("Escolha o tipo de visão", [
                                         "Fundo", "Estratégia", "Ativo", "Compilado"], index=0)
            tipo_filtro = st.sidebar.selectbox("Escolha o filtro de tempo", [
                                               "Diário", "Semanal", "Mensal"], index=1)

            dados_portifolio_atual = pd.read_csv('portifolio_posições.csv')
            ultimo_dia_dados = dados_portifolio_atual['Dia de Compra'].max()
            ultimo_dia_dados = datetime.datetime.strptime(
                ultimo_dia_dados, "%Y-%m-%d")
            primeiro_dia_dados = dados_portifolio_atual['Dia de Compra'].min()
            primeiro_dia_dados = datetime.datetime.strptime(
                primeiro_dia_dados, "%Y-%m-%d")

            df_b3_fechamento = pd.read_csv("df_preco_de_ajuste_atual.csv")
            df_b3_fechamento = df_b3_fechamento.replace('\.', '', regex=True)
            df_b3_fechamento = df_b3_fechamento.replace(',', '.', regex=True)
            df_b3_fechamento.iloc[:, 1:] = df_b3_fechamento.iloc[:, 1:].astype(
                float)
            df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'TREASURY', df_b3_fechamento.columns !=
                                 'Assets'] = df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'TREASURY', df_b3_fechamento.columns != 'Assets'] * 1000
            df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'WDO1', df_b3_fechamento.columns !=
                                 'Assets'] = df_b3_fechamento.loc[df_b3_fechamento['Assets'] == 'WDO1', df_b3_fechamento.columns != 'Assets'] * 10
            ultimo_dia_dados_b3 = df_b3_fechamento.columns[-1]
            ultimo_dia_dados_b3 = datetime.datetime.strptime(
                ultimo_dia_dados_b3, "%Y-%m-%d")

            data_inicial = st.sidebar.date_input(
                "Data inicial", value=primeiro_dia_dados, min_value=primeiro_dia_dados, max_value=ultimo_dia_dados_b3.date())
            # Múltiplas datas
            if tipo_filtro == "Diário":
                # soamr 7 dias a data inicial
                data_final = data_inicial + datetime.timedelta(days=8)
                if data_final >= ultimo_dia_dados_b3.date():
                    data_final = ultimo_dia_dados_b3.date()

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

            df_final, df_final_pl = analisar_dados_fundos()

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

            # ADICIONAR UMA COLUNA DE TOTAL PARA O DF_FINAL
            df_final['Total'] = df_final.sum(axis=1)
            df_final_pl['Total'] = df_final_pl.sum(axis=1)

            # Chama a função de análise
            dict_result = analisar_performance_fundos(
                data_inicial,
                data_final,
                lista_estrategias,
                lista_ativos
            )

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

                    df_fundos_copy = df_fundos.copy()
                    df_fundos_copy.loc['Total'] = df_fundos_copy.sum()

                    for col in df_fundos_copy.columns:
                        df_fundos_copy[col] = df_fundos_copy[col].apply(
                            lambda x: f"R${x:,.2f}")

                    df_fundos_grana = df_fundos_copy

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

                    # Deixando somente a data sem o horário
                    df_fundos_long['date'] = df_fundos_long['date'].dt.strftime(
                        '%Y-%m-%d')

                    # Criando o gráfico de barras
                    # Cria a figura e os eixos
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='date', y='Rendimento_diario', hue='fundo',
                                data=df_fundos_long, ax=ax, palette="Blues")
                    # Rotaciona as datas para melhor visualização
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.set_title("Rendimento Diário por Fundo")
                    ax.set_xlabel("Data")
                    ax.set_ylabel("Rendimento Diário")
                    # Mudar o título da legenda para Fundos
                    ax.legend(title='Fundos')

                    plt.tight_layout()

                    # Exibe o gráfico com o Streamlit, passando a figura

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
                    df_fundos_copy.loc['Total'] = df_fundos_copy.sum()

                    for col in df_fundos_copy.columns:
                        df_fundos_copy[col] = df_fundos_copy[col].apply(
                            lambda x: f"{x:.2f}bps")

                    df_combinado = df_fundos_grana + " / " + df_fundos_copy
                    st.table(df_combinado)

                    st.pyplot(fig)

            elif visao == "Estratégia":
                lista_estrategias = {
                    'DI': 'Juros Nominais Brasil',
                    'DAP': 'Juros Reais Brasil',
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

                    df_estrategias_copy = df_estrategias.copy()
                    df_estrategias_copy.loc['Total'] = df_estrategias_copy.sum(
                    )
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

                    # Deixando somente a data sem o horário
                    df_estrategias_long['date'] = df_estrategias_long['date'].dt.date

                    # Criando o gráfico de barras
                    # Cria a figura e os eixos
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='date', y='Rendimento_diario', hue='estratégia',
                                data=df_estrategias_long, ax=ax, palette="Blues")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.set_title("Rendimento Diário por Estratégia")
                    ax.set_xlabel("Data")
                    ax.set_ylabel("Rendimento Diário")
                    ax.legend(title='Estraégias')
                    plt.tight_layout()

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
                    for col in df_estrategias_copy.columns:
                        df_estrategias_copy[col] = df_estrategias_copy[col].apply(
                            lambda x: f"{x:.2f}bps")

                df_combinado = df_estrategias_grana + " / " + df_estrategias_copy
                st.table(df_combinado)
                st.pyplot(fig)

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
                    df_ativos_long['date'] = df_ativos_long['date'].dt.date

                    # Criando o gráfico de barras
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='date', y='Rendimento_diario', hue='ativo',
                                data=df_ativos_long, ax=ax, palette="Blues")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    ax.set_title("Rendimento Diário por Ativo")
                    ax.set_xlabel("Data")
                    ax.set_ylabel("Rendimento Diário")
                    ax.legend(title='Ativos')
                    plt.tight_layout()

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
        st.write("Página ainda não está pronta.")
        # Função para apagar os dias de dados que o usuário não quer mais
        st.write("## Apagar Dados")
        st.write(
            "Insira uma data para que os dados registrados dessa data sejam apagados")
        st.sidebar.write("Selecione o dia que deseja apagar:")

        data_apag = st.sidebar.date_input(
            "Data que será apagada", value=datetime.date.today())

        st.html(
            '''
                <style>
                div[data-testid="stDateInput"] input {
                    color: black; /* Define o texto */
                                                    }
                
                </style>   
        
                '''
        )

        # Converter para o formato '2025-01-16'
        data_apag = data_apag.strftime("%Y-%m-%d")
        st.write(f"Data selecionada: {data_apag}")
        if st.sidebar.button("Apagar Dados"):
            try:
                apagar_dados_data(data_apag)
                st.write("Dados apagados com sucesso!")
            except:
                st.write("Erro ao apagar os dados. Tente novamente.")

# ==========================================================
#   FUNÇÃO DA PÁGINA 2 (Entrar Preços de Compra/Venda)
# ==========================================================


def second_page():
    st.title("Tela de Input de Preços (Compra/Venda)")

    if "posicoes" not in st.session_state or "ativos_selecionados" not in st.session_state:
        st.error(
            "Nenhuma posição encontrada. Volte à página anterior e selecione os ativos.")
        st.button("Voltar ao Dashboard Principal",
                  on_click=switch_to_main, key="back_main_no_data")
        return

    ativos = st.session_state["ativos_selecionados"]
    posicoes = st.session_state["posicoes"]

    st.sidebar.header("Insira o Preço de Compra/Venda")
    precos_user = {}
    data_compra = {}
    for ativo in ativos:
        precos_user[ativo] = st.sidebar.number_input(
            f"Preço de {ativo}:", min_value=0.0, value=None, step=0.5
        )
        data_compra[ativo] = st.sidebar.date_input(
            f"Dia de Compra de {ativo}:",
            value=datetime.date.today()
        )

    for ativo in ativos:
        data_compra[ativo] = data_compra[ativo].strftime("%Y-%m-%d")
        data_compra[ativo] = '2025-01-17'

    df_port, df = checkar_portifolio(
        ativos, posicoes, precos_user, data_compra)

    st.write("## Resumo das posições com preços informados:")
    df_resumo_port = df_port.copy()
    # Agrupar por Ativo, Quantidade e Rendimento
    df_resumo_port = df_resumo_port.groupby('Ativo').sum()
    df_resumo_port['Preço de Compra'] = df_resumo_port['Preço de Compra'] / \
        df_resumo_port['Quantidade']
    df_resumo_port.drop(
        ['Dia de Compra'], axis=1, inplace=True)
    st.table(df_resumo_port)
    st.write(
        "OBS: Os preços de compra são calculados pela média dos preços de cada ativo.")
    # Botão Voltar (1 clique)
    st.button("Voltar ao Dashboard Principal",
              on_click=switch_to_main, key="back_main_ok")


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
    second_page()

# --------------------------------------------------------
