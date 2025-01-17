









import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

# ==========================================================
#               FUNÇÕES AUXILIARES
# ==========================================================
def processar_b3_portifolio():
    """
    Exemplo de função que carrega dois dataframes de CSV:
     - df_preco_de_ajuste_atual.csv : preços de fechamento (colunas de datas)
     - df_variacao.csv : variação diária dos ativos (colunas de datas)
    Retorna df_b3_fechamento e df_b3_variacao já tratados.
    """
    df_b3_fechamento = pd.read_csv("df_preco_de_ajuste_atual.csv")  
    # df_b3_fechamento possui colunas: ['Assets', '2025-01-17', '2025-01-18', ...] (exemplo)
    
    df_b3_variacao = pd.read_csv("df_variacao.csv")
    # df_b3_variacao possui colunas: ['Assets', '2025-01-17', '2025-01-18', ...]

    # Tirar pontos das casas de milhar
    df_b3_fechamento = df_b3_fechamento.replace('\.', '', regex=True)
    df_b3_variacao = df_b3_variacao.replace('\.', '', regex=True)

    # Trocar vírgula por ponto
    df_b3_fechamento = df_b3_fechamento.replace(',', '.', regex=True)
    df_b3_variacao = df_b3_variacao.replace(',', '.', regex=True)
    
    # Converter para float (exceto a primeira coluna, que é 'Assets')
    df_b3_fechamento.iloc[:, 1:] = df_b3_fechamento.iloc[:, 1:].astype(float)
    df_b3_variacao.iloc[:, 1:] = df_b3_variacao.iloc[:, 1:].astype(float)
    
    return df_b3_fechamento, df_b3_variacao


def processar_dados_port():
    """
    Lê o arquivo CSV 'portifolio_posições.csv' e retorna a lista de ativos existentes.
    Também poderíamos retornar diretamente as quantidades.
    """
    df_assets = pd.read_csv("portifolio_posições.csv")  # Espera colunas: ['Ativo','Quantidade', 'Dia de Compra', 'Preço de Compra', ...]
    if 'Unnamed: 0' in df_assets.columns:
        df_assets.rename(columns={'Unnamed: 0': 'Ativo'}, inplace=True)
    assets_iniciais = df_assets['Ativo'].tolist()
    return assets_iniciais


def checkar_portifolio(assets, 
                       quantidades, 
                       compra_especifica,  # dict { 'PETR4': 100.0, ... } se usuário inserir valor de compra
                       dia_compra,         # dict { 'PETR4': '2025-01-17', ... } se usuário inserir data de compra
                       df_b3_fechamento, 
                       df_b3_variacao):
    """
    Atualiza (ou cria) o CSV 'portifolio_posições.csv' com as novas posições, datas e preços de compra específicos.
    Em seguida, calcula colunas como: 'Preço de Ajuste Atual', 'Variação de Taxa' e 'Rendimento', caso deseje.
    """
    
    nome_arquivo_portifolio = 'portifolio_posições.csv'
    
    # 1) Carrega portfólio existente (se existir)
    if os.path.exists(nome_arquivo_portifolio):
        df_portifolio = pd.read_csv(nome_arquivo_portifolio, index_col=None)
        # Caso a 1ª coluna seja "Unnamed: 0", ajusta:
        if 'Unnamed: 0' in df_portifolio.columns:
            df_portifolio.drop(columns=['Unnamed: 0'], inplace=True)
    else:
        # Se não existe, cria um DataFrame vazio com colunas padronizadas
        df_portifolio = pd.DataFrame(columns=[
            'Ativo', 
            'Quantidade', 
            'Dia de Compra', 
            'Preço de Compra', 
            'Preço de Ajuste Atual', 
            'Variação de Taxa', 
            'Rendimento'
        ])
    
    # Garantir que 'Ativo' será a chave principal
    if 'Ativo' not in df_portifolio.columns:
        df_portifolio['Ativo'] = []
        
    # 2) Remover do DataFrame os ativos que NÃO estão na nova lista
    #    (significa que o usuário removeu aquele ativo do portfólio)
    ativos_existentes_csv = df_portifolio['Ativo'].tolist()
    for ativo_em_port in ativos_existentes_csv:
        if ativo_em_port not in assets:
            df_portifolio = df_portifolio[df_portifolio['Ativo'] != ativo_em_port]

    # 3) Adicionar ou atualizar as quantidades dos ativos que estão na nova lista
    #    Se há compra_especifica, ela sobrescreve a coluna 'Preço de Compra'
    #    Também atualizamos o 'Dia de Compra' se fornecido
    for i, asset in enumerate(assets):
        qtd_final = quantidades[i]  # Quantidade informada pelo usuário

        # Verifica se esse ativo existe no df_portifolio
        if asset not in df_portifolio['Ativo'].values:
            # Cria uma linha nova
            new_row = {
                'Ativo': asset,
                'Quantidade': qtd_final,
                'Dia de Compra': dia_compra.get(asset, None) if isinstance(dia_compra, dict) else dia_compra,
                'Preço de Compra': compra_especifica.get(asset, np.nan) if compra_especifica else np.nan,
                'Preço de Ajuste Atual': np.nan,
                'Variação de Taxa': np.nan,
                'Rendimento': np.nan
            }
            df_portifolio = pd.concat([df_portifolio, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Se já existe, apenas atualiza
            df_portifolio.loc[df_portifolio['Ativo'] == asset, 'Quantidade'] = qtd_final
            
            # Atualiza dia_compra, se houver
            if isinstance(dia_compra, dict):
                if asset in dia_compra and dia_compra[asset] is not None:
                    df_portifolio.loc[df_portifolio['Ativo'] == asset, 'Dia de Compra'] = dia_compra[asset]
            else:
                # Se dia_compra for só uma string ou None
                if dia_compra is not None:
                    df_portifolio.loc[df_portifolio['Ativo'] == asset, 'Dia de Compra'] = dia_compra

            # Atualiza preço de compra específico, se houver
            if compra_especifica and (asset in compra_especifica):
                df_portifolio.loc[df_portifolio['Ativo'] == asset, 'Preço de Compra'] = compra_especifica[asset]

    # 4) Calcular colunas de 'Preço de Ajuste Atual', 'Variação de Taxa' e 'Rendimento'
    #    Precisamos de data_hoje_str e df_b3_fechamento/df_b3_variacao
    data_hoje_str = datetime.date.today().strftime('%Y-%m-%d')

    for index, row in df_portifolio.iterrows():
        asset = row['Ativo']

        # Localiza no df_b3_fechamento
        if asset not in df_b3_fechamento['Assets'].values:
            # Se não encontrar no df, pula ou zera
            continue
        
        # Tenta ler a data de compra
        dia_compra_ativo = row['Dia de Compra']
        # Se dia_compra_ativo é NaN ou None, não conseguimos calcular Preço de Compra pelo CSV B3
        # Mas se o usuário definiu compra_especifica, já temos algo na coluna.
        
        # PREÇO DE COMPRA (se não tiver na coluna, podemos buscar no CSV b3 se a data existir)
        preco_compra_atual = row['Preço de Compra']
        
        # PREÇO DE AJUSTE ATUAL = valor na coluna de data_hoje_str, se existir
        try:
            mask_ativo = (df_b3_fechamento['Assets'] == asset)
            if data_hoje_str in df_b3_fechamento.columns:
                preco_ajuste_atual = df_b3_fechamento.loc[mask_ativo, data_hoje_str].values[0]
            else:
                # Caso não exista a coluna com data_hoje_str no CSV, pega a última coluna
                ult_col = df_b3_fechamento.columns[-1]
                if ult_col != 'Assets':
                    preco_ajuste_atual = df_b3_fechamento.loc[mask_ativo, ult_col].values[0]
                else:
                    preco_ajuste_atual = np.nan
        except:
            preco_ajuste_atual = np.nan

        df_portifolio.loc[index, 'Preço de Ajuste Atual'] = preco_ajuste_atual

        # VARIAÇÃO DE TAXA
        # Pode ser diária ou acumulada, depende do que você deseja.
        # Para simplificar, supomos que a variação é na data de compra, mas se não tiver data de compra, pula.
        try:
            if isinstance(dia_compra_ativo, str) and dia_compra_ativo in df_b3_variacao.columns:
                variacao_taxa = df_b3_variacao.loc[df_b3_variacao['Assets'] == asset, dia_compra_ativo].values[0]
            else:
                # Se não tiver a data, pode ser NaN ou a última data
                variacao_taxa = np.nan
            df_portifolio.loc[index, 'Variação de Taxa'] = variacao_taxa
        except:
            df_portifolio.loc[index, 'Variação de Taxa'] = np.nan

        # RENDIMENTO = Quantidade * (Preço Ajuste Atual - Preço de Compra)
        # Se não tiver Preço de Compra, fica NaN
        if not pd.isna(preco_compra_atual) and not pd.isna(preco_ajuste_atual):
            df_portifolio.loc[index, 'Rendimento'] = row['Quantidade'] * (preco_ajuste_atual - preco_compra_atual)
        else:
            df_portifolio.loc[index, 'Rendimento'] = np.nan

    # 5) Salva o DataFrame atualizado de volta no CSV
    df_portifolio.reset_index(drop=True, inplace=True)
    df_portifolio.to_csv(nome_arquivo_portifolio, index=False)
    
    return df_portifolio


# ------------------------------------------------
# Exemplo de outras funções de análise/risco (opcional)
# ------------------------------------------------
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
    soma_sem_pesos = df_pl['PL'].sum()
    df_pl['Weights'] = Weights
    df_pl['Weights'] = df_pl['Weights'].astype(float)
    df_pl['PL_atualizado'] = df_pl['PL'] * df_pl['Weights']
    df_pl['Adm'] = ['SANTANDER', 'BTG', 'SANTANDER',
                    'SANTANDER', 'BTG', 'BTG', 'BTG', 'BTG', 'BTG']
    return df_pl, df_pl['PL_atualizado'].sum(), soma_sem_pesos


# ------------------------------------------------
#   Configurações do Streamlit
# ------------------------------------------------
st.set_page_config(page_title="Dashboard de Análise", layout="wide")

# CSS customizado (exemplo simplificado)
def add_custom_css():
    st.markdown(
        """
        <style>
         section[data-testid="stSidebar"] * {
            color: White;
         }
         /* Exemplo: cor de fundo do sidebar */
         div[data-testid="stSidebar"] {
            background-color: #2F4F4F;
         }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_custom_css()


# ------------------------------------------------
#   Definição das "páginas" do app
# ------------------------------------------------
def main_page():
    st.title("Página Principal - Edição de Portfólio")

    st.write("""
    Nesta página, você pode:
    1. Ler seu arquivo `portifolio_posições.csv` atual.
    2. Selecionar quais ativos quer manter no portfólio ou adicionar novos.
    3. Ajustar as quantidades.
    4. Salvar/Atualizar o CSV.
    """)

    # Carrega arquivo CSV com as posições
    assets_iniciais = processar_dados_port()  # lista de ativos
    df_positions = pd.read_csv("portifolio_posições.csv")

    # Garante colunas mínimas
    if 'Quantidade' not in df_positions.columns:
        df_positions['Quantidade'] = 0

    # Lista de ativos disponíveis (carregada do CSV) -- 
    # Se quiser permitir o usuário digitar ativos novos, podemos permitir via text_input
    st.subheader("Selecione ou adicione novos ativos")
    all_assets_currently = list(df_positions['Ativo'].unique())

    # Cria um multiselect com os ativos existentes do CSV
    selected_assets = st.multiselect(
        "Ativos do Portfólio",
        options=all_assets_currently,
        default=assets_iniciais  # Seleciona por padrão os que já estavam
    )

    # Campo para inserir um novo ativo, caso não exista
    novo_ativo = st.text_input("Adicionar Ativo Manualmente (se não estiver na lista):", "")
    if novo_ativo:
        if novo_ativo not in all_assets_currently:
            all_assets_currently.append(novo_ativo)
            selected_assets.append(novo_ativo)

    # Monta um dicionário {ativo: quantidade} para exibir e editar
    quantidades_dict = {}
    for ativo in selected_assets:
        # Pega quantidade atual do CSV (se existir)
        mask = (df_positions['Ativo'] == ativo)
        if mask.any():
            qtd_atual = float(df_positions.loc[mask, 'Quantidade'].values[0])
        else:
            qtd_atual = 0
        new_qtd = st.number_input(f"Quantidade para {ativo}:", value=qtd_atual, step=1.0)
        quantidades_dict[ativo] = new_qtd

    # Botão para atualizar o CSV
    if st.button("Atualizar/Salvar Novo Portfólio"):
        # Precisamos chamar checkar_portifolio
        # Mas checkar_portifolio espera listas e dicts bem definidos
        assets_list = list(quantidades_dict.keys())
        qtd_list = list(quantidades_dict.values())

        # Para agora, não passaremos dia_compra e compra_especifica (passaremos {}), pois isso fica na página 2
        compra_especifica = {}
        dia_compra = {}

        # Precisamos do df_b3_fechamento e df_b3_variacao:
        df_b3_fechamento, df_b3_variacao = processar_b3_portifolio()
        
        df_port_updated = checkar_portifolio(
            assets=assets_list,
            quantidades=qtd_list,
            compra_especifica=compra_especifica,
            dia_compra=dia_compra,
            df_b3_fechamento=df_b3_fechamento,
            df_b3_variacao=df_b3_variacao
        )
        st.success("Portfólio atualizado com sucesso!")
        st.write(df_port_updated)


    st.write("---")
    st.write("Visualização atual do CSV `portifolio_posições.csv`:")
    st.dataframe(df_positions)


def second_page():
    st.title("Página 2 - Dados de Compra do Portfólio")

    st.write("""
    Nesta página, você pode:
    1. Ler o portfólio já atualizado (CSV).
    2. Definir a data de compra de cada ativo.
    3. Inserir um preço de compra específico (se desejar).
    4. Salvar novamente no CSV.
    """)

    # Lê novamente as posições do CSV atualizado
    df_positions = pd.read_csv("portifolio_posições.csv")

    if df_positions.empty:
        st.warning("O portfólio está vazio. Volte à Página Principal para adicionar ativos.")
        return

    # Lê df de B3
    df_b3_fechamento, df_b3_variacao = processar_b3_portifolio()

    # Vamos construir dois dicionários para passar ao checkar_portifolio:
    #  - dia_compra = { 'AtivoX': 'YYYY-MM-DD', ... }
    #  - compra_especifica = { 'AtivoX': valor, ... }
    dia_compra_dict = {}
    compra_especifica_dict = {}

    st.write("#### Edite os campos abaixo para cada Ativo:")
    for idx, row in df_positions.iterrows():
        ativo = row['Ativo']
        col1, col2, col3 = st.columns([3,3,3])
        with col1:
            st.markdown(f"**{ativo}**")

        with col2:
            # Dia de Compra
            # Se já estiver no CSV, converta em datetime. Se não, use today como default
            default_date = datetime.date.today()
            if not pd.isna(row.get('Dia de Compra', None)):
                try:
                    default_date = datetime.datetime.strptime(str(row['Dia de Compra']), "%Y-%m-%d").date()
                except:
                    pass
            
            new_date = st.date_input(f"Dia de Compra - {ativo}", value=default_date, key=f"dia_{ativo}")
            dia_compra_dict[ativo] = str(new_date)  # salva string YYYY-MM-DD

        with col3:
            # Preço de Compra Específico
            default_compra = 0.0
            if not pd.isna(row.get('Preço de Compra', None)):
                default_compra = float(row['Preço de Compra'])
            
            new_compra = st.number_input(f"Preço Compra - {ativo}", value=default_compra, step=1.0, key=f"compra_{ativo}")
            # Se o usuário deixar 0, interpretamos como "não quero customizar"
            if new_compra > 0:
                compra_especifica_dict[ativo] = new_compra

        st.markdown("---")

    # Botão para salvar
    if st.button("Salvar Dados de Compra"):
        # Montar os assets e quantidades
        assets_list = df_positions['Ativo'].tolist()
        qtd_list = df_positions['Quantidade'].tolist()

        # Chama novamente checkar_portifolio
        updated_port = checkar_portifolio(
            assets=assets_list,
            quantidades=qtd_list,
            compra_especifica=compra_especifica_dict,
            dia_compra=dia_compra_dict,
            df_b3_fechamento=df_b3_fechamento,
            df_b3_variacao=df_b3_variacao
        )
        st.success("Dados de compra salvos com sucesso!")
        st.write(updated_port)

    st.write("### Portfólio Atualizado (visualização):")
    st.dataframe(df_positions)


# ------------------------------------------------
#   Lógica de Navegação
# ------------------------------------------------
PAGINAS = {
    "Página Principal": main_page,
    "Página 2 (Compras)": second_page
}

# Sidebar para navegar
st.sidebar.title("Navegação")
choice = st.sidebar.radio("Ir para:", list(PAGINAS.keys()))
PAGINAS[choice]()
