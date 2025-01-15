import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Estilo personalizado
st.markdown("""
    <style>
        .reportview-container {
            background-color: #1D1D1D;
            color: black;
        }
        .sidebar .sidebar-content {
            background-color: #1D1D1D;
            color: black;
        }
        h1, h2, h3, h4, h5, h6 {
            color: black;
        }
        .stButton>button {
            background-color: #003366;
            color: black;
        }
    </style>
    """, unsafe_allow_html=True)

# Carregar os dados
df_returns = pd.read_csv('df_returns.csv', index_col=0)
df_returns = pd.DataFrame(df_returns)

#Patrimonio do portifolio
df_pl = pd.read_csv('pl_fundos.csv', index_col=0)
df_pl = pd.DataFrame(df_pl)

# Tratamento da coluna 'PL' para remover 'R$', trocar ',' por '.' e converter para float
df_pl['PL'] = (
    df_pl['PL']
    .str.replace('R$', '', regex=False)  # Remove 'R$'
    .str.replace('.', '', regex=False)  # Remove os pontos dos milhares
    .str.replace(',', '.', regex=False)  # Substitui vírgula por ponto (notação decimal)
    .replace('--', np.nan)  # Substitui '--' por NaN
    .astype(float)  # Converte para float
)
#Ficar somente com as linhas 5,9,10,11,17,18,19,22
df_pl = df_pl.iloc[[5,9,10,11,17,18,19,22]]

#Criar uma coluna com os pesos, todos os pesos são iguais a 1 , menos o HORIZONTE que tem peso 0,5 
df_pl['weights'] = 1
df_pl.loc[22, 'weights'] = int(0.5)

#Criar uma coluna com o PL atualizado pelo peso
df_pl['PL_atualizado'] = df_pl['PL'] * df_pl['weights']
soma_pl = df_pl['PL_atualizado'].sum()
#Quero fazer a analise do PL TOTAL, analisar o var do PL TOTAL e ver se eu quero x exposição ao risco, quais as posições dos ativos eu posso tomar

#Adicionar uma caixa de texto para o usuário escolher o VAR 95%
st.sidebar.header('VAR 95%')
var = st.sidebar.text_input('Digite o VAR 95% desejado', 0.05)

var = float(var)

#Ver o PL total maximo que eu posso ter
st.sidebar.header('PL TOTAL Maximo do Portifolio')
st.sidebar.write(soma_pl * var)

#Ver o risco de cada ativo 

#Dentro desse primeiro total tenho a opção de escolher os ativos

# Dropdown para seleção dos ativos
assets = st.sidebar.multiselect('Selecione os Ativos', df_returns.columns.tolist(), default=df_returns.columns.tolist()[:2])
# Criação do portfólio com dados dos últimos 252 dias
df_portifolio = df_returns[assets].dropna().tail(252)
#VAR de cada ativo
def var_not_parametric(df_returns, alpha=0.05):
    # Calcular o VAR
    var = df_returns.quantile(alpha)
    return var

#Calcular o VAR para os outros ativos
df_var = pd.DataFrame()

for col in df_returns.columns:
    if pd.api.types.is_float_dtype(df_returns[col]):
        var_95 = var_not_parametric(df_returns[col], alpha=0.05)
        df_var[col] = [var_95]

#Calcular o VAR do portifolio
# Campo para adicionar a quantidade de cada ativo
df_quantidade = pd.DataFrame()
st.sidebar.header('Quantidade de Ativos')
#Somente valores inteiros
for asset in assets:  # Supondo que assets seja uma lista de nomes
    quantidade = st.sidebar.number_input(
        f"Quantidade de {asset}",
        value=0,
        step=1,
        key=f"quantidade_{asset}"  # Chave única baseada no nome do ativo
    )
    df_quantidade[asset] = [quantidade]

## Converter para um array numpy
quantidades = df_quantidade.values[0]

# Calcular os pesos do portfólio (quantidades normalizadas)
weights = quantidades / quantidades.sum() if quantidades.sum() != 0 else np.zeros_like(quantidades)

# DataFrame de retornos diários do portfólio (exemplo fictício)
# Suponha que `df_portifolio` já contém os retornos diários dos ativos

# Calcular a volatilidade anualizada de cada ativo
volatilidade = df_portifolio.std() * np.sqrt(252)

# Calcular a matriz de covariância anualizada
cov_matrix = df_portifolio.cov() * 252

# Calcular o risco do portfólio (volatilidade do portfólio)
risco_portifolio = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Calcular os retornos médios anuais de cada ativo
retorno_medio = df_portifolio.mean() * 252  # Ajuste para anualização

# Calcular o retorno esperado do portfólio
portifolio_result = np.dot(weights, retorno_medio)


#Var do portfólio
#3var = np.percentile(df_portifolio.dot(weights), 5)
# Dividir a tela em duas colunas
col1, col2 = st.columns(2)

# Garante que soma_pl seja um número
if isinstance(soma_pl, (list, pd.Series)):
    soma_pl = sum(soma_pl)

# Multiplica após garantir compatibilidade
st.sidebar.write(soma_pl * float(var))


# Primeira Coluna - Dados Formatados
with col1:
    st.subheader("Dados do Portfólio")
    #Colocar o PL total do portifolio em reais e em % do PL total
    st.write(f'**PL Total do Portifolio:** R$ {soma_pl:.2f}')
    st.write(f'**PL Total do Portifolio com VAR 95%:** R$ {soma_pl * var:.2f}')
    st.write(f"**Retorno Médio Anualizado dos Ativos:**")
    for asset, ret in zip(assets, retorno_medio):
        st.write(f"{asset}: {ret:.2%}")
    st.write(f"**Retorno Medio Anualizado do Portfólio:** {portifolio_result:.2%}")
    st.write(f"**Risco do Portfólio:** {risco_portifolio:.2%}")

# Segunda Coluna - Gráficos
with col2:
    st.subheader("Gráficos dos Ativos Selecionados")
    for asset in assets:
        st.write(f"**Histograma de {asset}**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_portifolio[asset].dropna(), bins=50, color='blue', alpha=0.7)
        ax.set_title(f"Histograma dos Retornos de {asset}")
        st.pyplot(fig)

# Análises adicionais
analysis = st.sidebar.selectbox('Selecione o Tipo de Análise', 
                                ['VAR 95%', 'CVAR 95%', 'Volatilidade', 
                                 'Matriz de Correlação', 'Matriz de Covariância'])

def var_not_parametric(data, alpha=0.05):
    return data.quantile(alpha)

def cvar_not_parametric(data, alpha=0.05):
    var = var_not_parametric(data, alpha)
    return data[data <= var].mean()

if analysis == 'VAR 95%':
    with col1:
        st.subheader("VAR 95%")
        for asset in assets:
            var = var_not_parametric(df_portifolio[asset], alpha=0.05)
            st.write(f"**{asset}:** {var:.2%}")

elif analysis == 'CVAR 95%':
    with col1:
        st.subheader("CVAR 95%")
        for asset in assets:
            cvar = cvar_not_parametric(df_portifolio[asset], alpha=0.05)
            st.write(f"**{asset}:** {cvar:.2%}")

elif analysis == 'Volatilidade':
    with col1:
        st.subheader("Volatilidade Anualizada")
        for asset in assets:
            volatility = df_portifolio[asset].std() * np.sqrt(252)
            st.write(f"**{asset}:** {volatility:.2%}")

elif analysis == 'Matriz de Correlação':
    with col2:
        st.subheader("Matriz de Correlação")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_portifolio.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

elif analysis == 'Matriz de Covariância':
    with col2:
        st.subheader("Matriz de Covariância")
        st.write(df_portifolio.cov())


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

# Carregar os dados
df_returns = pd.read_csv('df_returns.csv', index_col=0)
df_returns = pd.DataFrame(df_returns)

# Patrimônio do portfólio
df_pl = pd.read_csv('pl_fundos.csv', index_col=0)
df_pl['PL'] = (
    df_pl['PL']
    .str.replace('R$', '', regex=False)
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
    .replace('--', np.nan)
    .astype(float)
)
df_pl = df_pl.iloc[[5, 9, 10, 11, 17, 18, 19, 22]]
df_pl['weights'] = 1
df_pl.loc[22, 'weights'] = 0.5
df_pl['PL_atualizado'] = df_pl['PL'] * df_pl['weights']
soma_pl = df_pl['PL_atualizado'].sum()

# Sidebar - Configurações de VAR
st.sidebar.header('Configurações de Risco')
var = st.sidebar.slider('Selecione o VAR 95% desejado (%)', min_value=0.01, max_value=0.1, value=0.05, step=0.01)

# Sidebar - Ativos
st.sidebar.header('Ativos do Portfólio')
assets = st.sidebar.multiselect('Selecione os Ativos', df_returns.columns.tolist(), default=df_returns.columns.tolist()[:2])
df_portifolio = df_returns[assets].dropna().tail(252)

# Sidebar - Quantidades
st.sidebar.header('Quantidade de Ativos')
df_quantidade = pd.DataFrame()
for asset in assets:
    quantidade = st.sidebar.number_input(f"Quantidade de {asset}", value=0, step=1, key=f"quantidade_{asset}")
    df_quantidade[asset] = [quantidade]

quantidades = df_quantidade.values[0]
weights = quantidades / quantidades.sum() if quantidades.sum() != 0 else np.zeros_like(quantidades)

# Cálculos de métricas
volatilidade = df_portifolio.std() * np.sqrt(252)
cov_matrix = df_portifolio.cov() * 252
risco_portifolio = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
retorno_medio = df_portifolio.mean() * 252
portifolio_result = np.dot(weights, retorno_medio)

# Primeira Coluna - Dados
col1, col2 = st.columns(2)
with col1:
    st.subheader("Resumo do Portfólio")
    st.write(f"**PL Total do Portfólio:** R$ {soma_pl:.2f}")
    st.write(f"**PL Total com VAR 95%:** R$ {soma_pl * var:.2f}")
    st.write(f"**Risco do Portfólio:** {risco_portifolio:.2%}")
    st.write(f"**Retorno Médio Anualizado:** {portifolio_result:.2%}")

# Segunda Coluna - Gráficos
with col2:
    st.subheader("Visualização dos Ativos")
    for asset in assets:
        st.write(f"**Histograma de {asset}**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_portifolio[asset].dropna(), bins=30, color='#003366', alpha=0.8)
        ax.set_title(f"Distribuição dos Retornos: {asset}", fontsize=12, color='#003366')
        ax.set_facecolor('#E8F1F2')
        st.pyplot(fig)

# Seleção de análise
analysis = st.sidebar.selectbox('Selecione o Tipo de Análise', ['VAR 95%', 'CVAR 95%', 'Matriz de Correlação', 'Matriz de Covariância'])

# Funções de análise
def var_not_parametric(data, alpha=0.05):
    return data.quantile(alpha)

def cvar_not_parametric(data, alpha=0.05):
    var = var_not_parametric(data, alpha)
    return data[data <= var].mean()

# Apresentação dos resultados
if analysis == 'VAR 95%':
    with col1:
        st.subheader("VAR 95%")
        for asset in assets:
            var_95 = var_not_parametric(df_portifolio[asset], alpha=0.05)
            st.write(f"**{asset}:** {var_95:.2%}")

elif analysis == 'CVAR 95%':
    with col1:
        st.subheader("CVAR 95%")
        for asset in assets:
            cvar_95 = cvar_not_parametric(df_portifolio[asset], alpha=0.05)
            st.write(f"**{asset}:** {cvar_95:.2%}")

elif analysis == 'Matriz de Correlação':
    with col2:
        st.subheader("Matriz de Correlação")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_portifolio.corr(), annot=True, cmap='Blues', ax=ax)
        st.pyplot(fig)

elif analysis == 'Matriz de Covariância':
    with col2:
        st.subheader("Matriz de Covariância")
        st.write(df_portifolio.cov())
