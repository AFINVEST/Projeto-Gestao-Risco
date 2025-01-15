import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------- CONFIGURAÇÕES GERAIS DO STREAMLIT -----------
st.set_page_config(
    page_title="Dashboard de Análise",
    layout="wide",
    initial_sidebar_state="expanded"
)

assets = ['DI_29', 'DAP35', 'TREASURY']

# ----------- FUNÇÕES AUXILIARES -----------


def process_portfolio(df_pl, weights):
    """
    df_pl: DataFrame já lido pelo Streamlit (em vez do caminho do arquivo).
    weights: lista com os valores de pesos para cada fundo.
    """
    # Processando a coluna PL para converter texto em float
    df_pl['PL'] = (
        df_pl['PL']
        .str.replace('R$', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .replace('--', np.nan)
        .astype(float)
    )
    # Selecionando linhas de interesse (ajuste conforme a sua necessidade)
    df_pl = df_pl.iloc[[5, 9, 10, 11, 17, 18, 19, 20, 22]]
    df_pl['weights'] = weights
    df_pl['weights'] = df_pl['weights'].astype(float)
    df_pl['PL_atualizado'] = df_pl['PL'] * df_pl['weights']
    df_pl['Adm'] = ['SANTANDER', 'BTG', 'SANTANDER',
                    'SANTANDER', 'BTG', 'BTG', 'BTG', 'BTG', 'BTG']
    return df_pl, df_pl['PL_atualizado'].sum()


def load_and_process_excel(df_excel, assets_sel):
    """
    df_excel: DataFrame do Excel já lido (usando pd.read_excel no Streamlit).
    assets_sel: lista com colunas a serem selecionadas.
    """
    # Ajuste das colunas
    df = df_excel.copy()
    # Seleciona apenas a última linha para ter os "preços de fechamento"
    df_precos = df.tail(1)[assets_sel]

    return df_precos, df


def process_returns(df, assets):
    df_retorno = df.copy()
    df_retorno = df_retorno[assets]
    # Remove valores nulos
    df_retorno.dropna(inplace=True)
    # Seleciona somente os últimos 252 dias
    df_retorno = df_retorno.tail(252).reset_index(drop=True)
    # Calcula retorno percentual (pct_change)
    df_retorno = df_retorno.pct_change()
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
        df_pl[f'Contratos {df_precos.index[i]}'] = (
            df_pl['PL_atualizado'] /
            df_pl['PL_atualizado'].sum()
        ) * df_precos.iloc[i]['Valor Total']
    return df_pl


def calculate_contracts_per_fund_input(df_pl, df_precos):
    for i in range(len(df_precos)):
        df_pl[f'Contratos Input {df_precos.index[i]}'] = (
            df_pl['PL_atualizado'] /
            df_pl['PL_atualizado'].sum()
        ) * df_precos.iloc[i]['Quantidade']
        df_pl[f'Contratos Input {df_precos.index[i]}'] = df_pl[f'Contratos Input {df_precos.index[i]}'].apply(
            lambda x: round(x, 0))

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
    # Tirar casas decimais
    for col in df_precos.columns:
        df_precos[col] = df_precos[col].apply(lambda x: round(x, 0))
    return df_precos.abs()


# ----------- INÍCIO DO APLICATIVO STREAMLIT -----------

file_pl = "pl_fundos.csv"
df_pl = pd.read_csv(file_pl, index_col=0)

file_bbg = "BBG - ECO DASH.xlsx"
st.title("Dashboard de Análise de Risco de Portifólio")

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
weights = list(dict_pesos.values())

st.sidebar.write("## Pesos dos Fundos")
st.sidebar.write("Caso queira, defina os pesos de cada fundo:")
fundos = st.sidebar.multiselect("Selecione os Fundos:",
                                list(dict_pesos.keys()),
                                None)
if fundos:
    # Atualiza os pesos
    for fundo in fundos:
        peso = st.sidebar.number_input(
            f"Peso para {fundo}:", min_value=0.0, value=1.0, step=0.1)
        dict_pesos[fundo] = peso
weights = list(dict_pesos.values())
df_pl_processado, soma_pl = process_portfolio(df_pl, weights)
st.sidebar.write("---")

df = pd.read_excel(file_bbg, sheet_name='BZ RATES',
                   skiprows=1, thousands='.', decimal=',')

df.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2',
        'Unnamed: 3', 'Unnamed: 26'], axis=1, inplace=True)
df.columns.values[0] = 'Date'
df = df.drop([0])  # Remover a primeira linha
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.drop(['OI1 Comdty', 'WSP1 Index'], axis=1, inplace=True)
df.columns = [
    'Date', 'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30',
    'DI_31', 'DI_32', 'DI_33', 'DI_35', 'DAP25', 'DAP26', 'DAP27',
    'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'WDO1', 'IBOV',
    'TREASURY', 'S&P'
]

# recoloquei dolar, conferir dps
df.drop(['IBOV', 'S&P'], axis=1, inplace=True)

default_assets = ['WDO1']
# Selecionar quais ativos analisar
st.sidebar.write("## Ativos do Portifólio")
assets = st.sidebar.multiselect("Selecione os ativos:",
                                list(df.columns),
                                default_assets)


if len(assets) > 0:
    df_precos, df_completo = load_and_process_excel(
        df, assets)

    # st.subheader(
    #   "Preços de Fechamento Selecionados (última linha)")
    # st.write(df_precos)

    # Processar retornos
    df_retorno = process_returns(df_completo, assets)

    # Calcular VaR não paramétrico por ativo
    var_ativos = var_not_parametric(df_retorno)
    var_ativos = var_ativos.abs()
    # st.write("**VaR de cada ativo**", abs(var_ativos))

    # Ajustar preços com VaR
    df_precos_ajustados = adjust_prices_with_var(
        df_precos, var_ativos)
    # st.subheader("Valores de Fechamento Ajustados pelo VaR")
    # st.write(df_precos_ajustados)

    # Exemplo de Quantidade
    # Quantidades podem ser definidas manualmente
    # ou, se quiser, inserir via widget do Streamlit:
    # (Coloque valores default para cada ativo na ordem em que aparecem)
    qtd_input = []
    for col in df_precos_ajustados.index:
        val = st.sidebar.number_input(
            f"Quantidade para {col}:", min_value=-100, value=1, step=1)
        qtd_input.append(val)
    quantidade = np.array(qtd_input)

    # Somar o valor de fechamento multiplicado pela quantidade
    vp = df_precos_ajustados['Valor Fechamento'] * abs(quantidade)
    # st.write(vp)
    vp_soma = vp.sum()
    # st.write(f"**VP do portfólio**: R$ {vp_soma:,.2f}")

    # Peso de cada ativo
    pesos = quantidade * df_precos_ajustados['Valor Fechamento'] / vp_soma
    # st.write("**Pesos do portfólio**:", pesos.values)
    # st.write(pesos)
    # Retorno do portfólio
    df_returns_portifolio = df_retorno * pesos.values
    df_returns_portifolio['Portifolio'] = df_returns_portifolio.sum(
        axis=1)

    # VaR do portfólio (não paramétrico)
    var_port = var_not_parametric(
        df_returns_portifolio['Portifolio'])
    var_port = abs(var_port)
    # Em dinheiro
    st.sidebar.write("---")
    st.sidebar.write("## Limite do Var do Portifólio")
    var_port_pl_dinheiro = var_port * soma_pl
    var_port_dinheiro = vp_soma * var_port
    var_bps = st.sidebar.slider(
        "VaR do Portfólio (bps)", min_value=1.0, max_value=20.0, value=1.0, step=0.5)
    var_bps = var_bps / 10000
    var_din = st.sidebar.checkbox("Exibir Limite de VaR em dinheiro")
    if var_din:
        var_lim_din = st.sidebar.number_input(
            "Valor do VaR em dinheiro:", min_value=-1000000000000.0, value=var_port_dinheiro, step=1.0)
      #  st.write(
       #     f"**VaR Limite (dinheiro)**: R$ {var_lim_din:,.2f}")
    else:
        var_limite = st.sidebar.slider(
            "Limite para VaR gasto do Portfólio", min_value=0.1, max_value=1.0, value=1.0, step=0.01)

     #   st.write(
     #       f"**VaR Limite (dinheiro)**: R$ {var_limite * var_port_dinheiro:,.2f}")

    # st.write(
     #   f"**VaR do Portfólio (dinheiro)**: R$ {var_port_dinheiro:,.2f}")

    # Volatilidade
    vol_port_retornos = df_returns_portifolio['Portifolio'].std()
    vol_port_analitica = vol_port_retornos * np.sqrt(252)
    # st.write(
    #   f"**Volatilidade (Std) do Portfólio**: {vol_port_retornos:.4%}")

    # Matriz de covariância
    # Adicionar a coluna de retorno do portfólio no df_retorno
    df_retorno['Portifolio'] = df_returns_portifolio['Portifolio']
    cov = df_retorno.cov()
    # Cov de cada ativo com o retorno do portfólio
    cov_port = cov['Portifolio'].drop('Portifolio')
    # Beta de cada ativo
    df_beta = cov_port / (vol_port_retornos**2)

    # Mostrando Beta
    # st.write("**Beta de cada ativo em relação ao Portfólio**")
    # st.write(df_beta)
    # st.write(vp_soma)
    # st.write(f'teste{vp}')

    # Mvar (Marginal VaR) e Mvar em dinheiro
    df_mvar = df_beta * var_port
    df_mvar_dinheiro = df_mvar * df_precos_ajustados['Valor Fechamento']
    # st.write("**Marginal VaR (em Dinheiro)**")

    # COVAR e % do total
    covar = df_mvar * pesos.values * vp_soma
    covar_perc = covar / covar.sum()
    # st.write("**CoVaR (contribuição de cada ativo)**")
    # st.write(covar_perc)
    cvar = df_retorno[df_retorno['Portifolio'] < var_not_parametric(
        df_returns_portifolio['Portifolio'])]['Portifolio']
    cvar = cvar.mean()

    # st.write(f"**CoVaR Total**: R$ {covar.sum():,.2f}")


df_precos_ajustados = calculate_portfolio_values(
    df_precos_ajustados, df_pl_processado, var_bps)
df_pl_processado = calculate_contracts_per_fund(
    df_pl_processado, df_precos_ajustados)
# Definir tamanhos das colunas: col1 e col2 maiores, col3 menor
# Ajuste os valores para mudar os tamanhos relativos
col1, col2, col3 = st.columns([2.5, 3.5, 1])
with col3:
    # Coloquei as checkboxes aqui e alterei o texto para maior clareza
    st.write("Escolha as colunas a exibir:")
    beta = st.checkbox("Exibir Beta", value=False)
    mvar = st.checkbox("Exibir MVar (R$)", value=True)
    covar_rs = st.checkbox("Exibir CoVaR (R$)", value=True)
    covar_perce = st.checkbox("Exibir CoVaR (%)", value=True)
    var_check = st.checkbox("Exibir VaR", value=False)
    perc_ris_tot = st.checkbox("Exibir % do Risco Total", value=True)

with col1:
    st.write("## Dados do Portifólio")
    st.write(f"**Soma do PL atualizado: R$ {soma_pl:,.0f}**")

    if var_din:
        st.write(
            f"**VaR Limite**: **R${var_lim_din:,.0f}**")
        var_limite_comparativo = var_lim_din
    else:
        st.write(
            f"**VaR Limite**:(Peso de {var_limite:.1%}) : **R${soma_pl * var_bps * var_limite:,.0f}**")
        var_limite_comparativo = soma_pl * var_bps * var_limite
    st.write(
        f"**VaR do Portifólio**: R${var_port_dinheiro:,.0f} : **{var_port_dinheiro/soma_pl * 10000:.2f}bps**")
    st.write(
        f"**CVaR**: R${abs(cvar * vp_soma):,.0f} : **{abs(cvar * vp_soma)/soma_pl * 10000:.2f}bps**")
    st.write(f"**Volatilidade**: {vol_port_analitica:.2%}")

    st.write("---")
    # Ver quantos % do limite a soma do covar usa
    # st.write(
    #    f"**R$ {(abs(covar.sum())):,.0f}**")
    st.write(
        f"### {abs(covar.sum()/ var_limite_comparativo):.2%} do risco total")

with col2:
    # Create a table with covar, beta, mvar, and covar for each asset
    df_dados = pd.DataFrame({
        'Beta': df_beta,
        'MVar(R$)': df_mvar_dinheiro,
        'CoVaR(R$)': covar,
        'CoVaR(%)': covar_perc,
        'Var': var_ativos[assets],
        '% do Risco Total': covar_perc * abs(covar.sum() / var_limite_comparativo)
    })

    # Filter columns based on selected checkboxes
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

    # Se a coluna CoVaR(R$) foi selecionada, formatar
    if 'CoVaR(R$)' in colunas_selecionadas:
        df_dados['CoVaR(R$)'] = df_dados['CoVaR(R$)'].apply(
            lambda x: f"R${x:,.0f}")

    if 'MVar(R$)' in colunas_selecionadas:
        df_dados['MVar(R$)'] = df_dados['MVar(R$)'].apply(
            lambda x: f"R${x:,.0f}")

    if 'CoVaR(%)' in colunas_selecionadas:
        df_dados['CoVaR(%)'] = df_dados['CoVaR(%)'].apply(lambda x: f"{x:.2%}")

    if 'Beta' in colunas_selecionadas:
        df_dados['Beta'] = df_dados['Beta'].apply(lambda x: f"{x:.4f}")

    if 'Var' in colunas_selecionadas:
        df_dados['Var'] = df_dados['Var'].apply(lambda x: f"{x:.4f}%")

    if '% do Risco Total' in colunas_selecionadas:
        df_dados['% do Risco Total'] = df_dados['% do Risco Total'].apply(
            lambda x: f"{x:.2%}")

    # Display the filtered table
    if colunas_selecionadas:
        st.write("Tabela de Dados Selecionados:")
        tabela_filtrada = df_dados[colunas_selecionadas]

        # Adicionar uma linha de soma
        sum_row = tabela_filtrada.select_dtypes(include='number').sum()
        sum_row['Beta'] = df_beta.sum()
        sum_row['MVar(R$)'] = df_mvar_dinheiro.sum()
        sum_row['CoVaR(R$)'] = covar.sum()
        sum_row['CoVaR(%)'] = covar_perc.sum()
        sum_row['Var'] = var_ativos[assets].sum()
        sum_row['% do Risco Total'] = (
            covar_perc * abs(covar.sum() / var_limite_comparativo)).sum()
        sum_row = sum_row.to_frame().T
        sum_row['Beta'] = sum_row['Beta'].apply(lambda x: f"{x:.4f}")
        sum_row['MVar(R$)'] = sum_row['MVar(R$)'].apply(
            lambda x: f"R${x:,.0f}")
        sum_row['CoVaR(R$)'] = sum_row['CoVaR(R$)'].apply(
            lambda x: f"R${x:,.0f}")
        sum_row['CoVaR(%)'] = sum_row['CoVaR(%)'].apply(lambda x: f"{x:.2%}")
        sum_row['Var'] = sum_row['Var'].apply(lambda x: f"{x:.4f}")
        sum_row['% do Risco Total'] = sum_row['% do Risco Total'].apply(
            lambda x: f"{x:.2%}")

        sum_row = sum_row[colunas_selecionadas]
        # Adicionar índice 'Total'
        sum_row.index = ['Total']
        # Adicionar a linha de soma na tabela filtrada
        tabela_filtrada_com_soma = pd.concat([tabela_filtrada, sum_row])
        st.table(tabela_filtrada_com_soma)

    else:
        st.write("Nenhuma coluna selecionada.")


#########################
# 9) TABELA DF_PL (FILTROS)
#########################


st.write("---")
st.write("## Quantidade de Contratos por Fundo")

# Formatar a tabela
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


default_columns = ['Adm', 'PL_atualizado']
for asset in assets:
    col_name = f'Contratos Input {asset}'
    if col_name in df_pl_processado.columns:
        default_columns.append(col_name)


df_precos_ajustados['Quantidade'] = quantidade
df_pl_processado_input = calculate_contracts_per_fund_input(
    df_pl_processado, df_precos_ajustados)

df_pl_processado_print = df_pl_processado.copy()
# Adicionar linha de total
sum_row = df_pl_processado_input.select_dtypes(include='number').sum()
sum_row['Fundos/Carteiras Adm'] = 'Total'
sum_row['Adm'] = ''
df_pl_processado_print = pd.concat(
    [df_pl_processado_input, sum_row.to_frame().T], ignore_index=True)

df_pl_processado_print.set_index('Fundos/Carteiras Adm', inplace=True)
df_pl_processado_print = df_pl_processado_print.drop(
    ['PL', 'Adm'], axis=1)

for asset in assets:
    col_name = f'Contratos Input {asset}'
    if col_name in df_pl_processado_print.columns:
        df_pl_processado_print.drop(
            col_name, axis=1, inplace=True)
        default_columns.append(col_name)


colunas_df_processado = []
for asset in assets:
    col_name = f'Contratos {asset}'
    if col_name in df_pl_processado.columns:
        colunas_df_processado.append(col_name)

df_copy_processado = df_pl_processado.copy()
df_copy_processado.drop(colunas_df_processado, axis=1, inplace=True)
columns_sem_fundo = df_copy_processado.columns.tolist()
columns_sem_fundo.remove('Fundos/Carteiras Adm')
columns_sem_fundo.remove('PL')


st.write("### Selecione as colunas")
columns = []
col1, col2, col3 = st.columns([4, 3, 3])
# Contador para saber qual coluna estamos preenchendo
for i, col in enumerate(columns_sem_fundo):
    if i % 3 == 0:
        # Primeira coluna (col1)
        with col1:
            if st.checkbox(col, value=col in default_columns, key=f"checkbox_{i}_{col}"):
                columns.append(col)
    elif i % 3 == 1:
        # Segunda coluna (col2)
        with col2:
            if st.checkbox(col, value=col in default_columns, key=f"checkbox_{i}_{col}"):
                columns.append(col)
    else:
        # Terceira coluna (col3)
        with col3:
            if st.checkbox(col, value=col in default_columns, key=f"checkbox_{i}_{col}"):
                columns.append(col)

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


filtered_df = df_pl_processado.copy()


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

filtered_df.index = filtered_df['Fundos/Carteiras Adm']
if columns:
    # Formatar a tabela
    for column in columns:
        if column not in ['Adm', 'Fundos/Carteiras Adm']:
            if column == 'PL_atualizado':
                filtered_df[column] = filtered_df[column].apply(
                    lambda x: f"R${x:,.0f}")
            else:
                filtered_df[column] = filtered_df[column].apply(
                    lambda x: f"{x:.2f}")

    st.table(filtered_df[columns])
    # Adicionar uma OBS
    st.write("OBS: Os contratos estão arrendodandos para inteiros.")
else:
    st.write("Selecione ao menos uma coluna para exibir os dados.")


for asset in assets:
    col_name = f'Contratos {asset}'
    if col_name in df_pl_processado_print.columns:
        df_pl_processado_print.rename(
            columns={col_name: f'Max Contratos {asset}'}, inplace=True)
for col in df_pl_processado_print.columns:
    if col != 'PL_atualizado':
        df_pl_processado_print[col] = df_pl_processado_print[col].apply(
            lambda x: f"{x:.2f}")
    else:
        df_pl_processado_print[col] = df_pl_processado_print[col].apply(
            lambda x: f"R${x:,.0f}")

st.write("---")
st.write("### Quantidade Máxima de Contratos por Adm")
# Ocultar coluna de Quantidade Valor Fechamento	Valor Fechamento Ajustado pelo Var
df_precos_plot = df_precos_ajustados.drop(
    ['Quantidade', 'Valor Fechamento', 'Valor Fechamento Ajustado pelo Var'], axis=1)
st.table(df_precos_plot)

st.write("### Quantidade Máxima de Contratos por Fundo")
st.table(df_pl_processado_print)

st.write("---")


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
            color: black; /* Define o texto laranja */
        };
        </style>

        
        """,
        unsafe_allow_html=True,
    )


# Adicionar o CSS personalizado
add_custom_css()
