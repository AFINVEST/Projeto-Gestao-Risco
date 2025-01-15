import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
import numpy as np
from matplotlib import ticker as mticker

# Caminho do seu arquivo Excel
path_fundos = 'Dados/economatica (3).xlsx'


def process_df_det(path):
    # Lendo a planilha Detalhes
    df_det = pd.read_excel(path, sheet_name='Detalhes', skiprows=1)

    # Dropar coluna "Unnamed: 0"
    df_det.drop(columns=['Unnamed: 0'], inplace=True)

    # Pegar até a coluna 87
    df_det = df_det.iloc[:, :86]

    # Salvar filtros aplicáveis (pode ajustar conforme sua planilha)
    filtro = ["Classificação\nAnbima", "Gestora", "Investimento\nno Exterior"]

    # DataFrame filtrado anteriormente com os nomes que contêm "FIC"
    filtros_fic = df_det[df_det["Nome"].str.contains(
        r"\bFIC\b", case=True, na=False)]
    # print(len(filtros_fic)) = 14
    # Remove linhas do DataFrame original que possuem nomes presentes no filtros_fic
    df_det = df_det[~df_det["Nome"].isin(filtros_fic["Nome"])]

    return df_det, filtro


def process_df_cap(path):
    df_cap = pd.read_excel(path, sheet_name='Captacao')
    df_cap = df_cap.T

    # Drop the first line
    df_cap = df_cap.iloc[1:]
    df_cap.reset_index(inplace=True)
    df_cap.drop(columns=[1], inplace=True)
    # Rename columns
    df_cap.rename(columns={'index': 'Fundos', 0: 'Código'}, inplace=True)
    # Criar uma lista com a primeira linha
    new_header = df_cap.iloc[0]
    new_header = new_header.values
    new_colunms = []
    for column in new_header:
        if column != 'Datas' and column != 'Fundos':
            column = column.date()
            new_colunms.append(column)
    # Mudar o nome das colunas
    new_colunms.insert(0, 'Fundos')
    new_colunms.insert(1, 'Código')
    df_cap.columns = new_colunms
    # remover  a primeira linha
    df_cap = df_cap.iloc[1:]
    # DataFrame filtrado anteriormente com os nomes que contêm "FIC"
    filtros_fic = df_cap[df_cap["Fundos"].str.contains(
        r"\bFIC\b", case=True, na=False)]
    # print(len(filtros_fic)) = 14
    # Remove linhas do DataFrame original que possuem nomes presentes no filtros_fic
    df_cap = df_cap[~df_cap["Fundos"].isin(filtros_fic["Fundos"])]

    return df_cap


def process_df_pl(path):
    df_pl = pd.read_excel(path, sheet_name='PL')
    df_pl = df_pl.T

    # Drop the first line
    df_pl = df_pl.iloc[1:]
    df_pl.reset_index(inplace=True)
    df_pl.drop(columns=[1], inplace=True)
    # Rename columns
    df_pl.rename(columns={'index': 'Fundos', 0: 'Código'}, inplace=True)
    # Criar uma lista com a primeira linha
    new_header = df_pl.iloc[0]
    new_header = new_header.values
    new_colunms = []
    for column in new_header:
        if column != 'Datas' and column != 'Fundos':
            column = column.date()
            new_colunms.append(column)
    # Mudar o nome das colunas
    new_colunms.insert(0, 'Fundos')
    new_colunms.insert(1, 'Código')
    df_pl.columns = new_colunms
    # remover  a primeira linha
    df_pl = df_pl.iloc[1:]
    # DataFrame filtrado anteriormente com os nomes que contêm "FIC"
    filtros_fic = df_pl[df_pl["Fundos"].str.contains(
        r"\bFIC\b", case=True, na=False)]
    # print(len(filtros_fic)) = 14
    # Remove linhas do DataFrame original que possuem nomes presentes no filtros_fic
    df_pl = df_pl[~df_pl["Fundos"].isin(filtros_fic["Fundos"])]

    return df_pl


# Função principal do Streamlit
st.title("Análise de Fundos de RF")

# Lê e processa dados
df_det, filtro_det = process_df_det(path_fundos)
df_cap = process_df_cap(path_fundos)
df_pl = process_df_pl(path_fundos)

# -- SIDEBAR: Filtros de Categoria (Classificação, Gestora, Exterior) --
st.sidebar.header("Filtros de Categoria")

classificacao_options = st.sidebar.multiselect(
    "Classificação\nAnbima",
    options=sorted(df_det["Classificação\nAnbima"].dropna().unique()),
    default=None
)
gestora_options = st.sidebar.multiselect(
    "Gestora",
    options=sorted(df_det["Gestora"].dropna().unique()),
    default=None
)
exterior_options = st.sidebar.multiselect(
    "Investimento\nno Exterior",
    options=sorted(df_det["Investimento\nno Exterior"].dropna().unique()),
    default=None
)

# Aplica filtros em df_det
df_det_filtered = df_det.copy()

if classificacao_options:
    df_det_filtered = df_det_filtered[
        df_det_filtered["Classificação\nAnbima"].isin(classificacao_options)
    ]
if gestora_options:
    df_det_filtered = df_det_filtered[
        df_det_filtered["Gestora"].isin(gestora_options)
    ]
if exterior_options:
    df_det_filtered = df_det_filtered[
        df_det_filtered["Investimento\nno Exterior"].isin(exterior_options)
    ]

# -- SIDEBAR: Filtro de Datas (para a captação) --
st.sidebar.header("Filtro de Período")
# Identifica quais colunas de df_cap são datas
all_date_cols = [
    col for col in df_cap.columns
    if isinstance(col, date)
]

# Garante que existe ao menos uma coluna de data
if len(all_date_cols) > 0:
    min_date, max_date = min(all_date_cols), max(all_date_cols)

    start_date = st.sidebar.date_input(
        "Data Inicial",
        min_value=min_date, max_value=max_date,
        value=min_date
    )
    end_date = st.sidebar.date_input(
        "Data Final",
        min_value=min_date, max_value=max_date,
        value=max_date
    )

    # Filtra as colunas pelas datas selecionadas
    filtered_date_cols = [
        col for col in all_date_cols
        if col >= start_date and col <= end_date
    ]
else:
    # Se não houver colunas do tipo data, não filtra nada
    filtered_date_cols = []

st.write("## Dados Filtrados")

# -- CRUZA df_cap COM df_det_filtered PELO 'Código' --
# Seleciona apenas códigos que passaram pelo filtro de df_det
df_cap_filtered = df_cap[df_cap["Código"].isin(
    df_det_filtered["Código"])].copy()

# Para Df_Pl
df_pl_filtered = df_pl[df_pl["Código"].isin(
    df_det_filtered["Código"])].copy()

# Garante que vamos mostrar só as colunas de datas selecionadas
# + Fundos e Código (para identificação)
final_cols = ["Fundos", "Código"] + filtered_date_cols
df_cap_filtered = df_cap_filtered[final_cols]
df_pl_filtered = df_pl_filtered[final_cols]


# -- EXEMPLO DE GRÁFICO --
# Faz "melt" para transformar as datas em linhas (deixa tidy)
if len(filtered_date_cols) > 0:
    df_cap_melt = df_cap_filtered.melt(
        id_vars=["Fundos", "Código"],
        var_name="Data",
        value_name="Valor"
    )
    # Converte Data para datetime se ainda não for
    df_cap_melt["Data"] = pd.to_datetime(df_cap_melt["Data"])

    # Exemplo de agregação diária (somando todos os fundos)
    df_agg = df_cap_melt.groupby("Data")["Valor"].sum().reset_index()

    st.write("## Evolução da Captação/Resgate no Período Selecionado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_agg, x="Data", y="Valor", ax=ax)
    ax.set_title("Soma de Captações/Resgates por Data")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor (R$)")
    # Formatação do eixo Y com separadores de milhar
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Adiciona uma linha horizontal tracejada no zero
    ax.axhline(0, color='Blue', linestyle='--',
               linewidth=0.75, label='Linha do zero')

    st.pyplot(fig)

# Plotar a soma do PL dos fundos ao longo do tempo e exibir o PL total no final do período

if len(filtered_date_cols) > 0:
    df_pl_melt = df_pl_filtered.melt(
        id_vars=["Fundos", "Código"],
        var_name="Data",
        value_name="Valor"
    )
    # Converte Data para datetime se ainda não for
    df_pl_melt["Data"] = pd.to_datetime(df_pl_melt["Data"])

    # Exemplo de agregação diária (somando todos os fundos)
    df_agg = df_pl_melt.groupby("Data")["Valor"].sum().reset_index()

    st.write("## Evolução do PL dos Fundos no Período Selecionado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df_agg, x="Data", y="Valor", ax=ax)
    ax.set_title("Soma de PL por Data")
    ax.set_xlabel("Data")
    ax.set_ylabel("Valor (Milhares R$)")
    # Formatação do eixo Y com separadores de milhar
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    st.pyplot(fig)
pl_final = df_agg["Valor"].iloc[-1]
st.write(
    f"### O PL total dos fundos ao final do período foi de R$ {pl_final:,.0f}")

st.write("Tirar dúvida de FICFI")

# Exibe a tabela com os dados filtrados
st.dataframe(df_cap_filtered)


# Função para adicionar CSS personalizado

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

        div[data-baseweb="calendar"] button {
            color:white;
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
