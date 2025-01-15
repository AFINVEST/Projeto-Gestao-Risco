import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import itertools

#########################
# 1) ESTILO STREAMLIT
#########################
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

#########################
# 2) CARREGAR DF_RETURNS
#########################
df_returns = pd.read_csv('df_returns.csv', index_col=0)
df_returns = pd.DataFrame(df_returns)

#########################
# 3) CONFIGURAÇÕES LADO
#########################
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
assets = st.sidebar.multiselect(
    'Selecione os Ativos',
    df_returns.columns.tolist(),
    default=df_returns.columns.tolist()[:2]
)

# Sidebar - Quantidades
st.sidebar.header('Quantidade de Ativos')
df_quantidade = pd.DataFrame()
for asset in assets:
    quantidade = st.sidebar.slider(
        f'Selecione a porcentagem desejada para {asset}',
        min_value=-1.00, max_value=1.00, value=0.05, step=0.01
    )
    df_quantidade[asset] = [quantidade]

quantidades = df_quantidade.values[0]


#########################
# 4) FUNÇÕES DE RISCO E RETORNO
#########################

def calculate_portfoliorisk(df_returns, assets, weights):
    # Seleção dos retornos dos ativos
    df_portifolio = df_returns[assets].dropna().tail(252)  # Últimos 252 dias

    # Volatilidade anualizada
    vol_anual = df_portifolio.std() * np.sqrt(252)
    cov_matrix = df_portifolio.cov()

    vol_anual = np.array(vol_anual)
    weights = np.array(weights)

    vol_anual_2 = vol_anual ** 2
    weights_2 = weights ** 2

    # Risco individual (variância)
    risco_individual = 0
    for i in range(len(assets)):
        risco_individual += vol_anual_2[i] * weights_2[i]

    # Adiciona o termo de covariância
    num_assets = len(assets)
    cov = cov_matrix.values
    risco_covariancia = 0
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            risco_covariancia += 2 * cov[i, j] * weights[i] * weights[j]

    risco_total = risco_individual + risco_covariancia
    risco_total = np.sqrt(risco_total)

    return risco_total


def calculate_portfolio_return(df_returns, assets, weights):
    df_portifolio = df_returns[assets].dropna().tail(252)
    retorno_medio = df_portifolio.mean() * 252

    portifolio_result = 0
    for i in range(len(assets)):
        portifolio_result += weights[i] * retorno_medio[i]

    return portifolio_result


#########################
# 5) CALCULAR RISCO/RET
#########################
risco_portifolio = calculate_portfoliorisk(df_returns, assets, quantidades)
portifolio_result = calculate_portfolio_return(df_returns, assets, quantidades)


##############################
# 6) FUNÇÕES DE "EQUATIONS"
##############################
def equations(weights, Mvars, VarPort, Vars, CorrMatrix):
    """
    eq_i = w_i - (Mvar_i * VarPort) / ( sum_j [ w_j * Vars[j] * Vars[i] * CorrMatrix[i, j] ] )
    """
    n = len(weights)
    eqs = np.zeros(n)

    for i in range(n):
        denom_i = 0.0
        for j in range(n):
            denom_i += weights[j] * Vars[j] * Vars[i] * CorrMatrix[i, j]
        if abs(denom_i) < 1e-12:
            eqs[i] = 1e6
        else:
            eqs[i] = weights[i] - (Mvars[i] * VarPort) / denom_i
    return eqs


def objective(weights, Mvars, VarPort, Vars, CorrMatrix):
    eq_vals = equations(weights, Mvars, VarPort, Vars, CorrMatrix)
    return np.sum(eq_vals**2)


def generate_grid_nassets(n=3, step=0.001):
    """
    Gera combinações de pesos (w1..wn) com soma <=1 e w_i>0,
    em intervalos 'step' no [0,1].
    """
    w_range = np.arange(0, 1 + 1e-12, step)
    grid = []
    for combo in itertools.product(w_range, repeat=n):
        if sum(combo) <= 1.0 + 1e-12 and all(w > 0 for w in combo):
            grid.append(combo)
    return grid


#########################
# 7) RESTANTE DO SEU CÓDIGO
#########################

# -- Abaixo, as mesmas funções do seu script para preparar dados, etc.

def prepare_returns_data(data):
    df_returns2 = data.dropna().tail(252)
    columns2 = [
        'DI_25', 'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
        'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP25', 'DAP26', 'DAP27',
        'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'IBOV',
        'TREASURY', 'S&P'
    ]
    df_returns2.columns = columns2
    return df_returns2


def calculate_var(df, alpha=0.05):
    return df.quantile(alpha)


def load_and_process_excel(file_path, assets_sel):
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
    df_precos = df.tail(1)[assets_sel]
    return df_precos


def adjust_prices_with_var(df_precos, var_ativos):
    df_precos = df_precos.T
    df_precos.columns = ['Valor Fechamento']
    df_precos['Valor Fechamento Ajustado pelo Var'] = df_precos['Valor Fechamento'] * \
        var_ativos.values
    return df_precos


def var_not_parametric(data, alpha=0.05):
    return data.quantile(alpha)


def cvar_not_parametric(data, alpha=0.05):
    var_ = var_not_parametric(data, alpha)
    return data[data <= var_].mean()


df_portifolio = df_returns[assets].dropna().tail(252)


def calculate_portfolio_values(df_precos, portifolio_santander, portifolio_btg):
    df_precos['Santander'] = portifolio_santander / \
        df_precos['Valor Fechamento Ajustado pelo Var']
    df_precos['BTG'] = portifolio_btg / \
        df_precos['Valor Fechamento Ajustado pelo Var']
    df_precos['Valor Total'] = df_precos['Santander'] + df_precos['BTG']
    return df_precos.abs().round(0)


def calculate_contracts_per_fund(df_pl, df_precos, portifolio_santander, portifolio_btg):
    for i in range(len(df_precos)):
        df_pl[f'Contratos/Fundo {df_precos.index[i]}'] = (
            df_pl['PL_atualizado'] /
            df_pl[df_pl['Adm'] == 'SANTANDER']['PL_atualizado'].sum()
        ) * df_precos.iloc[i]['Santander']

        btg_indices = [9, 17, 18, 19, 20, 22]
        df_pl.loc[btg_indices, f'Contratos/Fundo {df_precos.index[i]}'] = (
            df_pl.loc[btg_indices, 'PL_atualizado'] /
            df_pl[df_pl['Adm'] == 'BTG']['PL_atualizado'].sum()
        ) * df_precos.iloc[i]['BTG']
    return df_pl


def perform_stress_test(df_pl, df_precos):
    for asset in df_precos.index:
        df_pl[f'Financeiro {asset}'] = df_pl[f'Contratos/Fundo {asset}'] * \
            df_precos.loc[asset]['Valor Fechamento']
    return df_pl


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


def main(df_returns=df_returns, assets=assets):
    df_returns2 = prepare_returns_data(df_returns)
    var_ativos = calculate_var(df_returns2[assets])
    df_pl, soma_pl = process_portfolio('pl_fundos.csv')

    portifolio_santander = df_pl[df_pl['Adm'] ==
                                 'SANTANDER']['PL_atualizado'].sum() * var
    portifolio_btg = df_pl[df_pl['Adm'] == 'BTG']['PL_atualizado'].sum() * var

    df_precos = load_and_process_excel('BBG - ECO DASH.xlsx', assets)
    df_precos = adjust_prices_with_var(df_precos, var_ativos)
    df_precos = calculate_portfolio_values(
        df_precos, portifolio_santander, portifolio_btg)

    df_pl = calculate_contracts_per_fund(
        df_pl, df_precos, portifolio_santander, portifolio_btg)
    df_pl = perform_stress_test(df_pl, df_precos)
    df_pl = df_pl.reset_index(drop=True)
    return df_precos, df_pl, soma_pl, var_ativos, portifolio_santander, portifolio_btg


df_precos, df_pl, soma_pl, var_ativos, portifolio_santander, portifolio_btg = main(
    df_returns, assets)

#########################
# 8) EXIBIR RESULTADOS
#########################
col1, col2 = st.columns(2)
with col1:
    st.subheader("Resumo do Portfólio")
    st.write(f"**PL Total do Portfólio:** R$ {soma_pl:,.2f}")
    portifolio_santander_val = df_pl[df_pl['Adm']
                                     == 'SANTANDER']['PL_atualizado'].sum()
    portifolio_btg_val = df_pl[df_pl['Adm'] == 'BTG']['PL_atualizado'].sum()
    st.write(f"**Valor de PL Santander:** R$ {portifolio_santander_val:,.2f}")
    st.write(f"**Valor de PL BTG:** R$ {portifolio_btg_val:,.2f}")
    st.write(f"**Risco do Portfólio:** {risco_portifolio:.2%}")
    st.write(f"**Retorno do Portfólio:** {portifolio_result:.2%}")

    st.write(f"**VAR dos Ativos Selecionados:**")
    var_medio = 0
    for i in range(len(assets)):
        var_medio += var_ativos[i] * quantidades[i]
        st.write(f"{assets[i]}: {var_ativos[i]:.2%}")
    st.write(f"**VAR Portfólio:** {var_medio:.2%}")

with col2:
    st.subheader("VAR")
    st.write(f"**VAR (95%) em R$:** {soma_pl * var:,.2f}")
    st.write(
        f"**VAR Santander:** R$ {df_pl[df_pl['Adm'] == 'SANTANDER']['PL_atualizado'].sum() * var:,.2f}")
    st.write(
        f"**VAR BTG:** R$ {df_pl[df_pl['Adm'] == 'BTG']['PL_atualizado'].sum() * var:,.2f}")
    st.write("**VAR de Cada Ativo (Histórico):**")
    df_portifolio_sel = df_returns[assets].dropna().tail(252)
    for asset in assets:
        var_95 = var_not_parametric(df_portifolio_sel[asset], alpha=0.05)
        st.write(f"{asset}: {var_95:.2%}")

#########################
# 9) TABELA DF_PL (FILTROS)
#########################
default_columns = ['Fundos/Carteiras Adm', 'Adm', 'PL_atualizado']

for asset in assets:
    col_name = f'Contratos/Fundo {asset}'
    if col_name in df_pl.columns:
        default_columns.append(col_name)

columns = st.multiselect(
    'Selecione as colunas para exibir',
    df_pl.columns.tolist(),
    default=default_columns
)

filtro_fundo = st.multiselect(
    'Filtrar por Fundos/Carteiras Adm',
    df_pl["Fundos/Carteiras Adm"].unique()
)

filtro_adm = st.multiselect(
    'Filtrar por Adm',
    df_pl["Adm"].unique()
)

filtered_df = df_pl.copy()

if filtro_fundo:
    filtered_df = filtered_df[filtered_df["Fundos/Carteiras Adm"].isin(
        filtro_fundo)]
if filtro_adm:
    filtered_df = filtered_df[filtered_df["Adm"].isin(filtro_adm)]

# Soma
sum_row = filtered_df.select_dtypes(include='number').sum()
sum_row['Fundos/Carteiras Adm'] = 'Total'
sum_row['Adm'] = ''
filtered_df = pd.concat([filtered_df, sum_row.to_frame().T], ignore_index=True)

st.write("---")
if columns:
    st.table(filtered_df[columns])
else:
    st.write("Selecione ao menos uma coluna para exibir os dados.")

#########################
# 10) ANÁLISES
#########################
analysis = st.sidebar.selectbox(
    'Selecione o Tipo de Análise',
    ['VAR 95%', 'CVAR 95%', 'Matriz de Correlação',
     'Matriz de Covariância', 'Histograma',
     'Analise dos ativos', 'Analise dos Fundos']
)

df_portifolio_analise = df_returns[assets].dropna().tail(252)

if analysis == 'VAR 95%':
    st.subheader("VAR 95%")
    for asset in assets:
        var_95 = var_not_parametric(df_portifolio_analise[asset], alpha=0.05)
        st.write(f"{asset}: {var_95:.2%}")

elif analysis == 'CVAR 95%':
    st.subheader("CVAR 95%")
    for asset in assets:
        cvar_95 = cvar_not_parametric(df_portifolio_analise[asset], alpha=0.05)
        st.write(f"{asset}: {cvar_95:.2%}")

elif analysis == 'Matriz de Correlação':
    st.subheader("Matriz de Correlação")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_portifolio_analise.corr(), annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig)

elif analysis == 'Matriz de Covariância':
    st.subheader("Matriz de Covariância")
    st.write(df_portifolio_analise.cov())

elif analysis == 'Histograma':
    st.subheader("Histograma dos Ativos")
    for asset in assets:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_portifolio_analise[asset].dropna(),
                bins=30, color='#003366', alpha=0.8)
        ax.set_title(
            f"Distribuição dos Retornos: {asset}", fontsize=12, color='#003366')
        ax.set_facecolor('#E8F1F2')
        st.pyplot(fig)

elif analysis == 'Analise dos ativos':
    st.subheader("Análise dos Ativos")
    for asset in assets:
        st.write(f"**Ativo:** {asset}")
        # Você pode exibir dados adicionais aqui

elif analysis == 'Analise dos Fundos':
    st.subheader("Análise dos Fundos")
    # Exemplo simples
    fundos_unicos = df_pl["Fundos/Carteiras Adm"].unique()
    for fundo in fundos_unicos:
        st.write(f"Fundo: {fundo}")
    st.write("...")

#########################
# 11) BOTÃO PARA RODAR A 'OTIMIZAÇÃO MVaR'
#########################

st.write("---")
st.header("Otimização MVaR (grade)")

st.markdown("""
Clique no botão abaixo para rodar a *grid search* que minimiza:
\\[
\\sum_{i=1}^{n} \\Bigl(w_i - \\frac{\\text{MVar}_i \\times \\text{VarPort}}{\\sum_j [w_j \\times \\text{Var}_j \\times \\text{Var}_i \\times \\text{Corr}_{i,j} ]}\\Bigr)^2
\\]

**Nota**: Aqui vamos exemplificar definindo:
- \\( \\text{VarPort} = \\text{risco_portifolio} \\)
- \\( \\text{MVar}_i \\) e \\( \\text{Var}_i \\) extraídos dos retornos / correlações selecionados.
""")

if st.button("Rodar Otimização MVaR"):

    # Determinar n
    n = len(assets)
    if n < 2:
        st.warning(
            "É necessário selecionar pelo menos 2 ativos para a otimização.")
    else:
        # Exemplo de "Mvars" e "Vars" a partir do std de df_portifolio
        df_port = df_portifolio_analise  # 252 dias
        vol_each = df_port.std()  # vol de cada ativo
        corr_matrix = df_port.corr().values

        # EXEMPLO de Mvars -> digamos que MVaR seja algo proporcional à vol
        # Na prática, você deve inserir o MVaR real de cada ativo se tiver.
        Mvars = np.array(quantidades)
        Vars = np.array(var_ativos.values)

        # VarPort = risco do portfólio "atual", ou outro
        VarPort = var

        Mvars = Mvars * VarPort

        st.write("**Ativos Selecionados:**", assets)
        st.write("**Mvars (exemplo)**:", Mvars)
        st.write("**Vars (exemplo)**:", Vars)
        st.write("**VarPort (risco do portfólio)**:", VarPort)

        # Gerar grade
        step_otim = 0.01
        grid_otim = generate_grid_nassets(n=n, step=step_otim)

        results_otim = []
        for combo in grid_otim:
            w_arr = np.array(combo)
            val_obj = objective(w_arr, Mvars, VarPort, Vars, corr_matrix)
            results_otim.append((w_arr, val_obj))

        # Ordenar
        results_otim.sort(key=lambda x: x[1])
        best_weights_otim, best_obj_val_otim = results_otim[0]

        # Mostrar
        st.write(
            f"**Melhor combinação de pesos** (min erro) - step={step_otim}")
        valores = []
        for i, w_val in enumerate(best_weights_otim):
            st.write(f"{assets[i]}: {100*w_val:.2f}%")
            valores.append(100*w_val)
        weights = np.array(valores)
        st.write(f"**Soma dos pesos**: {100*np.sum(best_weights_otim):.2f}%")
        st.write(f"**Erro (objetivo)**: {best_obj_val_otim:.4e}")


def calculate_portfolioVaR(df_returns, assets, weights):
    # Seleção dos retornos dos ativos
    df_portifolio = df_returns[assets].dropna().tail(
        252)  # Últimos 252 dias úteis

    # Passo 1: Calcular a var (5% percentil) de cada ativo
    var = df_portifolio.quantile(0.05)
    print(var)

    # Passo 2: Calcular a matriz de correlação anualizada
    cor_matrix = df_portifolio.corr()

    print(cor_matrix)

    # Passo 3: Calcular o quadrado do Var ponderada e do Peso

    var = np.array(var)
    weights = np.array(weights)

    var_2 = var ** 2
    print(var_2)
    weights_2 = weights ** 2
    print(weights_2)

    # Passo 4: Multiplicar a volatilidade ponderada pelo peso de cada ativo
    var_individual = 0
    for i in range(len(assets)):
        var_individual += var_2[i] * weights_2[i]

    print(f'Sinaliazar{var_individual}')

    # Adicionar o termo de correlação
    num_assets = len(assets)
    cor = cor_matrix.values
    var_cor = 0
    numero_loops = 0
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            var_cor += 2 * cor[i, j] * weights[i] * \
                weights[j] * var[i] * var[j]
            numero_loops += 1

    print(f'Numero de loops: {numero_loops}')
    print(var_cor)

    # Soma total do risco
    var_total = var_individual + var_cor
    var_total = np.sqrt(var_total)
    print(var_total)

    return var_total


var_tot = calculate_portfolioVaR(df_returns, assets, weights)
st.write(f"**VAR Total do Portfólio:** {var_tot:.2%}")
