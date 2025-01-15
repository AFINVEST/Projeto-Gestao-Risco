import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Estilo personalizado
st.markdown("""
    <style>
        .reportview-container {
            background-color: #E8F1F2;
        }
        .sidebar .sidebar-content {
            background-color: #003366;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: Arial, sans-serif;
            color: #003366;
        }
        .stButton>button {
            background-color: #003366;
            color: white;
            font-family: Arial, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Carregar os dados -- DF RETORNOS LOG

df_returns = pd.read_csv('df_returns.csv', index_col=0)
df_returns = pd.DataFrame(df_returns)


# # Patrimônio do portfólio
# df_pl = pd.read_csv('pl_fundos.csv', index_col=0)
# df_pl['PL'] = (
#     df_pl['PL']
#     .str.replace('R$', '', regex=False)
#     .str.replace('.', '', regex=False)
#     .str.replace(',', '.', regex=False)
#     .replace('--', np.nan)
#     .astype(float)
# )

# df_pl = df_pl.iloc[[5, 9, 10, 11, 17, 18, 19, 20, 22]]
# df_pl['weights'] = 1
# df_pl.loc[22, 'weights'] = 0.5
# df_pl['PL_atualizado'] = df_pl['PL'] * df_pl['weights']
# df_pl['Adm'] = ['Santander', 'BTG', 'SANTANDER',
#                 'SANTANDER', 'BTG', 'BTB', 'BTG', 'BTG', 'BTG']
# soma_pl = df_pl['PL_atualizado'].sum()

# Sidebar - Configurações de VAR -- ARBITRADO
st.sidebar.header('Configurações de Risco')
var = st.sidebar.slider('Selecione o VAR 95% desejado (%)',
                        min_value=0.01, max_value=0.1, value=0.05, step=0.01)
var = var * 0.01

# Sidebar - Ativos
st.sidebar.header('Ativos do Portfólio')
columns = ['DI_25', 'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30', 'DI_31', 'DI_32', 'DI_33', 'DI_35',
           'DAP25', 'DAP26', 'DAP27', 'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'IBOV', 'TREASURY', 'S&P']
df_returns.columns = columns

# CRIA A LISTA DE ATIVOS
assets = st.sidebar.multiselect('Selecione os Ativos', df_returns.columns.tolist(
), default=df_returns.columns.tolist()[:2])


# Sidebar - Quantidades
st.sidebar.header('Quantidade de Ativos')

df_quantidade = pd.DataFrame()

for asset in assets:

    quantidade =st.sidebar.slider(f'Selecione a porcentagem desejada para {asset}',
                        min_value=-1.00, max_value=1.00, value=0.05, step=0.01)
    
    df_quantidade[asset] = [quantidade]


quantidades = df_quantidade.values[0]


def calculate_portfoliorisk(df_returns, assets, weights):
    # Seleção dos retornos dos ativos
    df_portifolio = df_returns[assets].dropna().tail(
        252)  # Últimos 252 dias úteis

    # Passo 1: Calcular a volatilidade (desvio padrão anualizado) de cada ativo
    vol_anual = df_portifolio.std() * np.sqrt(252)

    # Passo 2: Calcular a matriz de covariância anualizada
    cov_matrix = df_portifolio.cov()

    # Passo 3: Calcular o quadrado da volatilidade ponderada e do covariância

    vol_anual = np.array(vol_anual)
    weights = np.array(weights)

    vol_anual_2 = vol_anual ** 2
    weights_2 = weights ** 2

    # Passo 4: Multiplicar a volatilidade ponderada pelo peso de cada ativo
    risco_individual = 0

    for i in range(len(assets)):
        risco_individual += vol_anual_2[i] * weights_2[i]

    # Adicionar o termo de covariância
    num_assets = len(assets)
    cov = cov_matrix.values
    risco_covariancia = 0
    numero_loops = 0
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            risco_covariancia += 2 * cov[i, j] * weights[i] * weights[j]
            numero_loops += 1

    # Soma total do risco
    risco_total = risco_individual + risco_covariancia
    risco_total = np.sqrt(risco_total)

    return risco_total


# Calcular o risco do portfólio a partir da quantidade de ativos
weights = quantidades / \
    quantidades.sum() if quantidades.sum() != 0 else np.zeros_like(quantidades)
#NAO ESTOU USANDO MAIS ESSE PESO


risco_portifolio = calculate_portfoliorisk(df_returns, assets, quantidades)


def calculate_portfolio_return(df_returns, assets, weights):
    # Seleção dos retornos dos ativos
    df_portifolio = df_returns[assets].dropna().tail(
        252)  # Últimos 252 dias úteis

    # Retorno médio anualizado
    retorno_medio = df_portifolio.mean() * 252

    # Retorno do portfólio
    portifolio_result = 0 
    for i in range(len(assets)):
        portifolio_result += weights[i] * retorno_medio[i]

    return portifolio_result


# Calcular o retorno do portfólio a partir da quantidade de ativos
portifolio_result = calculate_portfolio_return(df_returns, assets, quantidades)

# weights = quantidades / \
#     quantidades.sum() if quantidades.sum() != 0 else np.zeros_like(quantidades)

# # Cálculos de métricas
# volatilidade = df_portifolio.std() * np.sqrt(252)
# cov_matrix = df_portifolio.cov().values * 252  # Matriz de covariância anualizada

# # Ajustar pesos para cálculo de risco do portfólio
# pesos = np.array(weights).reshape(-1, 1)
# risco_portifolio = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))[0][0]

# # Retorno médio anualizado
# retorno_medio = df_portifolio.mean() * 252
# portifolio_result = np.dot(weights, retorno_medio)

# # Valor gasto em cada ativo
# df = pd.read_excel('BBG - ECO DASH.xlsx', sheet_name='BZ RATES',
#                    skiprows=1, thousands='.', decimal=',')
# df.head()
# # drop column unnamed
# df.drop('Unnamed: 1', axis=1, inplace=True)
# df.drop('Unnamed: 2', axis=1, inplace=True)
# df.drop('Unnamed: 3', axis=1, inplace=True)
# df.drop('ODF25 Comdty', axis=1, inplace=True)
# df.head()
# df.rename(columns={'Unnamed: 4': 'Date'}, inplace=True)
# # remover primeira linha
# df = df.drop([0])
# df.head()
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
# df.rename(columns={'ODF25 Comdty.1': 'ODF25 Comdty'}, inplace=True)
# df.head()
# df.drop('OI1 Comdty', axis=1, inplace=True)
# df.drop('WSP1 Index', axis=1, inplace=True)
# df.columns = ['Date', 'DI_25', 'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30', 'DI_31', 'DI_32', 'DI_33', 'DI_35',
#               'DAP25', 'DAP26', 'DAP27', 'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'IBOV', 'TREASURY', 'S&P']
# df.head()
# df = df.tail(1)

# df_quantidade = df_quantidade.rename(index={0: 'Quantidade'})
# df_quantidade = df_quantidade.T

# df_precos = df
# df_precos = df_precos[assets]
# df_precos = df_precos.T
# df_precos.columns = ['Preço']
# df_precos = df_precos.reset_index()
# df_precos.rename(columns={'index': 'Ativo'}, inplace=True)
# df_precos = df_precos.set_index('Ativo')
# df_precos = df_precos.T

# # Quantidade de cada ativo é um array com a quantidade de cada ativo
# df_quantidade = df_quantidade.T


# # Criar DataFrame final com preço e quantidade
# df_final = pd.concat([df_precos, df_quantidade])
# df_final = df_final.T
# df_final['PL'] = df_final['Preço'] * df_final['Quantidade']

# df_final['Gasto/Pl'] = df_final['PL'] / soma_pl

# df_final['Pessos'] = weights
# df_final['Gasto'] = df_final['Preço'] * df_final['Pessos']

# # Quantidade * 100 * Preço
# df_final['Posicao'] = df_final['Quantidade'] * 100 * df_final['Preço']
# df_final

# # Dado o VAR selecionado e o risco do portifólio, o gasto em cada ativo é calculado
# valor_disponivel = (soma_pl * var) - df_precos[assets].sum().sum()
# # Criar um mecanismo para sempre que a quantidade de um ativo for alterada o valor disponível seja recalculado

# # Criar dicionario com o VAR de cada ATIVO
# var_ativo = {asset: df_returns[asset].quantile(0.05) for asset in assets}

# # Criar DataFrame inicial com a quantidade de cada ativo
# unidades_compra = {
#     asset: valor_disponivel / df_precos[asset].values[0] for asset in assets
# }
# df_quantidade = pd.DataFrame(unidades_compra, index=['Quantidade'])

# df_quantidade

# # Criar DataFrame inicial com o gasto em cada ativo
# df_gasto = df_precos * df_quantidade
# df_gasto['Gasto'] = df_gasto.sum(axis=1).T

# # Criar DataFrame inicial com a posição em cada ativo
# df_posicao = pd.DataFrame({'Posicao': df_gasto['Gasto']})
# df_posicao

# # Função para atualizar os dados com base na quantidade alterada

# Carregar e preparar os dados de retorno


def prepare_returns_data(data):
    df_returns = data.dropna().tail(252)
    columns = [
        'DI_25', 'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
        'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP25', 'DAP26', 'DAP27',
        'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'IBOV',
        'TREASURY', 'S&P'
    ]
    df_returns.columns = columns
    return df_returns

# Calcular o VaR histórico


def calculate_var(df, alpha=0.05):
    return df.quantile(alpha)

# Processar o arquivo Excel


def load_and_process_excel(file_path, assets):
    df = pd.read_excel(file_path, sheet_name='BZ RATES',
                       skiprows=1, thousands='.', decimal=',')
    df.drop(['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3',
            'ODF25 Comdty'], axis=1, inplace=True)
    df.rename(columns={'Unnamed: 4': 'Date',
              'ODF25 Comdty.1': 'ODF25 Comdty'}, inplace=True)
    df = df.drop([0])  # Remover a primeira linha
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.drop(['OI1 Comdty', 'WSP1 Index'], axis=1, inplace=True)
    df.columns = [
        'Date', 'DI_25', 'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
        'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP25', 'DAP26', 'DAP27',
        'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'IBOV',
        'TREASURY', 'S&P'
    ]
    df_precos = df.tail(1)[assets]
    return df_precos

# Calcular os valores ajustados pelo VaR


def adjust_prices_with_var(df_precos, var):
    # Adicionar uma linha de valores ajustados pelo VaR
    df_precos = df_precos.T
    df_precos.columns = ['Valor Fechamento']

    # Multiplicar os valores de fechamento pelo VaR
    df_precos['Valor Fechamento Ajustado pelo Var'] = df_precos['Valor Fechamento'] * var.values

    # df_precos = df_precos * var.values
    return df_precos

# Calcular os valores por administrador

def var_not_parametric(data, alpha=0.05):
    return data.quantile(alpha)


def cvar_not_parametric(data, alpha=0.05):
    var = var_not_parametric(data, alpha)
    return data[data <= var].mean()


df_portifolio = df_returns[assets].dropna().tail(252)


def calculate_portfolio_values(df_precos, portifolio_santander, portifolio_btg):
    # df_precos = df_precos.T
    # df_precos.columns = ['Valor Fechamento Ajustado pelo Var']
    df_precos['Santander'] = portifolio_santander / \
        df_precos['Valor Fechamento Ajustado pelo Var']
    df_precos['BTG'] = portifolio_btg / \
        df_precos['Valor Fechamento Ajustado pelo Var']
    df_precos['Valor Total'] = df_precos['Santander'] + df_precos['BTG']
    return df_precos.abs().round(0)

# Criar a coluna de contratos por fundo


def calculate_contracts_per_fund(df_pl, df_precos, portifolio_santander, portifolio_btg):
    for i in range(len(df_precos)):
        df_pl[f'Contratos/Fundo {df_precos.index[i]}'] = (
            df_pl['PL_atualizado'] / df_pl[df_pl['Adm'] == 'SANTANDER']['PL_atualizado'].sum()) * df_precos.iloc[i]['Santander']
        btg_indices = [9, 17, 18, 19, 20, 22]  # Índices ajustados para BTG
        df_pl.loc[btg_indices, f'Contratos/Fundo {df_precos.index[i]}'] = (
            df_pl.loc[btg_indices, 'PL_atualizado'] / df_pl[df_pl['Adm']
                                                            == 'BTG']['PL_atualizado'].sum() * df_precos.iloc[i]['BTG']
        )
        df_pl[f'Contratos/Fundo {df_precos.index[i]}'] = df_pl[f'Contratos/Fundo {df_precos.index[i]}'] * quantidades[i]

    
     #Adicionar uma coluna de VAR
    df_pl['VAR'] = df_pl['PL_atualizado'] * var
    df_pl['VAR_UTILIZADO'] = var

    #Adicionando uma linha que soma tudo
    #df_pl.loc['Total'] = df_pl.sum()
    #Deixar com nada os valores nao numericos
    #df_pl.loc['Total', 'Fundos/Carteiras Adm'] = ''
    #df_pl.loc['Total', 'Adm'] = ''
    #df_pl.loc['Total', 'weights'] = ''
    #df_pl.loc['Total', 'VAR_UTILIZADO'] = ''

    return df_pl

# Realizar stress test financeiro


def perform_stress_test(df_pl, df_precos):
    for asset in df_precos.index:
        df_pl[f'Financeiro {asset}'] = df_pl[f'Contratos/Fundo {asset}'] * \
            df_precos.loc[asset]['Valor Fechamento']
    return df_pl

# Patrimônio do portfólio


def process_portfolio(file_path):
    df_pl = pd.read_csv(file_path, index_col=0)
    df_pl['PL'] = (
        df_pl['PL']
        .str.replace('R$', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .replace('--', np.nan)
        .astype(float)
    )
    df_pl = df_pl.iloc[[5, 9, 10, 11, 17, 18, 19, 20, 22]]
    df_pl['weights'] = 1
    df_pl['weights'] = df_pl['weights'].astype(float)
    df_pl.loc[9, 'weights'] = 0.5
    df_pl.loc[5, 'weights'] = 2

    df_pl['PL_atualizado'] = df_pl['PL'] * df_pl['weights']
    df_pl['Adm'] = ['SANTANDER', 'BTG', 'SANTANDER',
                    'SANTANDER', 'BTG', 'BTG', 'BTG', 'BTG', 'BTG']

    return df_pl, df_pl['PL_atualizado'].sum()


# Main
def main(df_returns=df_returns, assets=assets):
    df_returns = prepare_returns_data(df_returns)  # Tirar a coluna Data
    var_ativos = calculate_var(df_returns[assets])  # Calcular o VAR de cada ativo

    # Processar portfólio PL dos fundos
    df_pl, soma_pl = process_portfolio('pl_fundos.csv')

    # Processar preços dos ativos por Administrador
    portifolio_santander = df_pl[df_pl['Adm'] ==
                                'SANTANDER']['PL_atualizado'].sum() * var
    portifolio_btg = df_pl[df_pl['Adm'] == 'BTG']['PL_atualizado'].sum() * var

    # Carregar e processar os dados de preços
    df_precos = load_and_process_excel('BBG - ECO DASH.xlsx', assets)
    # Calcular o valor dos ativos com base no VAR de 95%
    df_precos = adjust_prices_with_var(df_precos, var_ativos)
    #
    df_precos = calculate_portfolio_values(
        df_precos, portifolio_santander, portifolio_btg)

    # Calcular contratos e realizar stress test
    df_pl = calculate_contracts_per_fund(
        df_pl, df_precos, portifolio_santander, portifolio_btg)

    df_pl = perform_stress_test(df_pl, df_precos)
    # Reiniciar o index de df_pl
    df_pl = df_pl.reset_index(drop=True)
    return df_precos, df_pl, soma_pl, var_ativos, portifolio_santander, portifolio_btg

df_precos, df_pl, soma_pl, var_ativos, portifolio_santander, portifolio_btg = main(df_returns, assets)

# Visualizar os resultados
# df_precos
# df_pl

# Primeira Coluna - Dados


# col1, col2 = st.columns(2)
# with col1:

if var : 
    main()

col1, col2 = st.columns(2)
with col1:

    st.subheader("Resumo do Portfólio")
    st.write(f"**PL Total do Portfólio:** R$ {soma_pl:,.2f}")
    portifolio_santander = df_pl[df_pl['Adm'] ==
                                    'SANTANDER']['PL_atualizado'].sum()
    portifolio_btg = df_pl[df_pl['Adm'] == 'BTG']['PL_atualizado'].sum()
    st.write(
        f'**Valor de PL Santander:** R$ {portifolio_santander:,.2f}')
    st.write(f'**Valor de PL BTG:** R$ {portifolio_btg:,.2f}')
    st.write(f"**Risco do Portfólio:** {risco_portifolio:.2%}")
    st.write(f"**Retorno do Portfólio:** {portifolio_result:.2%}")
    #Criar um VAR medio dos ativos vezes o peso
    st.write(f"**VAR Médio dos Ativos:**")
    var_medio = 0
    for i in range(len(assets)):
        var_medio += var_ativos[i] * quantidades[i]
        st.write(f"**{assets[i]}:** {var_ativos[i]:.2%}")
    st.write(f"**VAR Portfólio:** {var_medio:.2%}")
    st.write(f"**:**")
    st.write(df_returns.dropna().tail(252).mean())

    #Criar o retorno medio do portifolio

    st.table(df_returns['DI_26'].tail(5))
    st.table(df_returns['DI_31'].tail(5))
    df_retorno = pd.Series(0, index=df_returns.index[-252:])
    for i in range(len(assets)):
        df_retorno += df_returns[assets[i]].dropna().tail(252) * quantidades[i]
        
    retorno_log = df_retorno.mean()
    st.write(f"**Retorno Log:** {retorno_log:.2%}")
    st.table(df_retorno.tail(5))
    
    df_di26 = df_returns['DI_26'].dropna().tail(1)
    #Transformar em um numero
    df_di26 = df_di26.values[0]
    df_di31 = df_returns['DI_31'].dropna().tail(1)
    df_di31 = df_di31.values[0]
    st.table(quantidades)
    st.write(f"**Retorno DI_26:** {df_di26:.2f}")
    st.write(f"**Retorno DI_31:** {df_di31:.2f}")
    st.write(f"**Retorno DI_26 * Quantudade:** {df_di26 * quantidades[0]:.2%}")
    st.write(f"**Retorno DI_31 * Quantudade:** {df_di31 * quantidades[1]:.2%}")
    
    portifolio_teste = df_returns[assets].mean() * 252
    st.write(f"**Retorno do Portfólio Teste:** {portifolio_teste.mean():.2%}")
    st.write(f'{df_returns['DI_31'].dropna().tail(252).mean() * 252:.2%}')
#    st.write(f"**Contratos/Ativo:**")
#    ativo = st.selectbox( 'Selecione o Ativo', assets)
#    st.write(f"**Quantidade de Contratos Santander:** {df_pl[df_pl['Adm'] == 'SANTANDER'][f'Contratos/Fundo {ativo}'].sum():.2f}")
#    st.write(f"**Quantidade de Contratos BTG:** {df_pl[df_pl['Adm'] == 'BTG'][f'Contratos/Fundo {ativo}'].sum():.2f}")


with col2:
    st.subheader("VAR")
    st.write(f"**VAR:** R$ {soma_pl * var:,.2f}")
    st.write(f"**VAR Santander :** R$ {df_pl[df_pl['Adm'] ==
                                    'SANTANDER']['VAR'].sum():,.2f}")
    
    st.write(f"**VAR BTG :** R$ {df_pl[df_pl['Adm'] == 'BTG']['VAR'].sum():,.2f}")
    st.write(f"****VAR ATIVOS:****")
    for asset in assets:
        var_95 = var_not_parametric(df_portifolio[asset], alpha=0.05)
        st.write(f"**{asset}:** {var_95:.2%}")
    
default_columns = [
    'Fundos/Carteiras Adm',
    'Adm',
    'PL_atualizado',
    'VAR'

]
st.write('---')
#Adiciona as colunas de contratos e financeiro
for asset in assets:
    default_columns.append(f'Contratos/Fundo {asset}')

columns = st.multiselect(
    'Selecione as colunas', df_pl.columns.tolist(), key='unique_multiselect_columns', default= default_columns )

# Filtros adicionais
filtro_fundo = st.multiselect(
    'Filtrar por Fundos/Carteiras Adm', df_pl["Fundos/Carteiras Adm"].unique(), key="filtro_fundo"
)
filtro_adm = st.multiselect(
    'Filtrar por Adm', df_pl["Adm"].unique(), key="filtro_adm"
)


# Aplicar filtros
filtered_df = df_pl.copy()

#Adiciona uma linha que soma tudo para cada coluna FILTRADA
if filtro_fundo:
    filtered_df = filtered_df[filtered_df["Fundos/Carteiras Adm"].isin(
        filtro_fundo)]

if filtro_adm:
    filtered_df = filtered_df[filtered_df["Adm"].isin(filtro_adm)]

# Adicionar uma linha de soma total para colunas numéricas
sum_row = filtered_df.select_dtypes(include='number').sum()
sum_row['Fundos/Carteiras Adm'] = 'Total'
sum_row['Adm'] = ''  # Ou outro valor para identificar que é a linha de total
filtered_df = pd.concat([filtered_df, sum_row.to_frame().T], ignore_index=True)

# Exibir o DataFrame com a linha de soma total
#st.dataframe(filtered_df)

# Mostrar tabela com colunas filtradas
if columns:
    st.table(filtered_df[columns])
else:
    st.write("Selecione ao menos uma coluna para exibir os dados.")


st.write('---')

# Menu dropdown para análise e histogramas
analysis = st.sidebar.selectbox(
    'Selecione o Tipo de Análise',
    ['VAR 95%', 'CVAR 95%', 'Matriz de Correlação',
        'Matriz de Covariância', 'Histograma', 'Analise dos ativos', 'Analise dos Fundos'],
)

# Funções de análise


# Apresentação dos resultados
if analysis == 'VAR 95%':
    st.subheader("VAR 95%")
    for asset in assets:
        var_95 = var_not_parametric(df_portifolio[asset], alpha=0.05)
        st.write(f"**{asset}:** {var_95:.2%}")

elif analysis == 'CVAR 95%':
    st.subheader("CVAR 95%")
    for asset in assets:
        cvar_95 = cvar_not_parametric(df_portifolio[asset], alpha=0.05)
        st.write(f"**{asset}:** {cvar_95:.2%}")

elif analysis == 'Matriz de Correlação':
    st.subheader("Matriz de Correlação")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_portifolio.corr(), annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig)

elif analysis == 'Matriz de Covariância':
    st.subheader("Matriz de Covariância")
    st.write(df_portifolio.cov())

elif analysis == 'Histograma':
    st.subheader("Histograma dos Ativos")
    for asset in assets:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_portifolio[asset].dropna(),
                bins=30, color='#003366', alpha=0.8)
        ax.set_title(
            f"Distribuição dos Retornos: {asset}", fontsize=12, color='#003366')
        ax.set_facecolor('#E8F1F2')
        st.pyplot(fig)

elif analysis == 'Analise dos ativos':
    st.subheader("Análise dos Ativos")
    # Criar duas colunas dentro da coluna 1
    for asset in assets:
        st.write(f"**{asset}:**")
        st.write(
            f"**Valor de Fechamento:** {df_precos.loc[asset]['Valor Fechamento']:.2f}")
        st.write(
            f"**Valor de Fechamento Ajustado pelo VAR:** {df_precos.loc[asset]['Valor Fechamento Ajustado pelo Var']:.2f}")
        st.write(
            f"**Contratos Santander:** {df_pl[df_pl['Adm'] == 'SANTANDER'][f'Contratos/Fundo {asset}'].sum():.2f}")
        st.write(
            f"**Contratos BTG:** {df_pl[df_pl['Adm'] == 'BTG'][f'Contratos/Fundo {asset}'].sum():.2f}")
        
        st.write(f"**Peso ponderado:** {weights[assets.index(asset)]:.2%}")

        # Criar menu para selecionar Fundo
        fundo = st.selectbox(
            'Selecione o Fundo',
            df_pl['Fundos/Carteiras Adm'].unique(),
            key=f'unique_selectbox_fundo_{asset}'
        )
        st.write(
            f"**Contratos {fundo}:** {df_pl[df_pl['Fundos/Carteiras Adm'] == fundo][f'Contratos/Fundo {asset}'].values[0]:.2f}")
        
        #Adicionar uma linha entre cada loop
        st.write('---')


        # Criar a tabela de contratos e financeiro por fundo


elif analysis == 'Analise dos Fundos':
    # Adicionar filtros para os fundos
    for fundo in df_pl['Fundos/Carteiras Adm']:
        st.write(f"**{fundo}:**")
        st.write(
            f"**Financeiro Santander:** {df_pl[df_pl['Fundos/Carteiras Adm'] == fundo][f'Financeiro {asset}'].values[0]:.2f}")
        st.write(
            f"**Financeiro BTG:** {df_pl[df_pl['Fundos/Carteiras Adm'] == fundo][f'Financeiro {asset}'].values[0]:.2f}")
    st.write('---')

#df_precos
#df_pl