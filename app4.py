import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

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
    soma_sem_pesos = df_pl['PL'].sum()
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

    df_divone['TREASURY'] = df_divone['TREASURY'].astype(str).str.replace(',', '.', regex=False)
    df_pe['WDO1'] = df_pe['WDO1'].astype(str).str.replace(',', '.', regex=False)

    df_divone['TREASURY'] = pd.to_numeric(df_divone['TREASURY'], errors='coerce')
    df_pe['WDO1'] = pd.to_numeric(df_pe['WDO1'], errors='coerce')

    df_divone['TREASURY'] = df_divone['TREASURY'] * df_pe['WDO1'].iloc[0] / 10000
    dolar = df_pe['WDO1'].iloc[0]

    return df_divone, dolar

def process_returns(df, assets):
    df_retorno = df.copy()
    df_retorno = df_retorno[assets]
    df_retorno.dropna(inplace=True)
    df_retorno = df_retorno.tail(252).reset_index(drop=True)
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
        df_pl[f'Contratos {df_precos.index[i]}'] = df_pl[f'Contratos {df_precos.index[i]}'].apply(
            lambda x: round(x, 0))
    return df_pl

def calculate_portfolio_values(df_precos, df_pl_processado, var_bps):
    pl_santander = df_pl_processado[df_pl_processado['Adm'] == 'SANTANDER']['PL_atualizado'].sum()
    pl_btg = df_pl_processado[df_pl_processado['Adm'] == 'BTG']['PL_atualizado'].sum()
    portifolio_santander = pl_santander * var_bps
    portifolio_btg = pl_btg * var_bps
    df_precos['Santander'] = portifolio_santander / df_precos['Valor Fechamento Ajustado pelo Var']
    df_precos['BTG'] = portifolio_btg / df_precos['Valor Fechamento Ajustado pelo Var']
    df_precos['Valor Total'] = df_precos['Santander'] + df_precos['BTG']
    for col in df_precos.columns:
        df_precos[col] = df_precos[col].apply(lambda x: round(x, 0))
    return df_precos.abs()

import os
import pandas as pd
import datetime
import numpy as np

def processar_b3_portifolio():
    """
    Exemplo de função que carrega dois dataframes de CSV:
     - df_preco_de_ajuste_atual.csv : preços de fechamento (colunas de datas)
     - df_variacao.csv : variação diária dos ativos (colunas de datas)
    """
    df_b3_fechamento = pd.read_csv("df_preco_de_ajuste_atual.csv")  
    # df_b3_fechamento possui colunas: ['Ativo', '17/01/2025', '18/01/2025', ...]
    
    df_b3_variacao = pd.read_csv("df_variacao.csv")
    # df_b3_variacao possui colunas: ['Ativo', '17/01/2025', '18/01/2025', ...]
    #Tirar o pontos das casas de milhar
    df_b3_fechamento = df_b3_fechamento.replace('\.', '', regex=True)
    df_b3_variacao = df_b3_variacao.replace('\.', '', regex=True)

    #Trocar Virgula por Ponto
    df_b3_fechamento = df_b3_fechamento.replace(',', '.', regex=True)
    df_b3_variacao = df_b3_variacao.replace(',', '.', regex=True)
    print(df_b3_fechamento)
    #Converter para Float menos a primeira coluna
    df_b3_fechamento.iloc[:, 1:] = df_b3_fechamento.iloc[:, 1:].astype(float)
    df_b3_variacao.iloc[:, 1:] = df_b3_variacao.iloc[:, 1:].astype(float)
    #Printar o tipo das colunas
    print(df_b3_fechamento.dtypes)
    print(df_b3_variacao.dtypes)
    
    return df_b3_fechamento, df_b3_variacao 


################ DEIXAR ESSA FUNÇÃO ATUALIZADA COM A LISTA DE ASSETS DEFAUTL DO PORTIFÓLIO E SUAS QUANTIDADES ################
def processar_dados_port():
    df_assets = pd.read_csv("portifolio_posições.csv")
    df_assets.rename(columns={'Unnamed: 0': 'Ativo'}, inplace=True)
    assets_iniciais = df_assets['Ativo'].tolist()
    # Criar uma lista com as quantidades, convertendo para inteiros e ignorando valores nulos
    quantidades_iniciais = (
        df_assets['Quantidade']
        .apply(lambda x: int(x) if x == x else 0)  # Converte para inteiro, garantindo valores válidos
        .tolist()
    )
    #Criar um dicionário com os ativos e suas quantidades
    assets_quantidades = dict(zip(assets_iniciais, quantidades_iniciais))

    return assets_iniciais, assets_quantidades

#NA SEGUNDA PÁGINA TEM QUE CRIAR UM CAMPO PARA O USUÁRIO INSERIR A DATA QUE ELE COMPROU DETERMINADO ATIVO E SE ELE QUER INSERIR UM VALOR DE COMPRA ESPECÍFICO
#ATUALIZAR PRA ESSA LISTA DE QUANTIDADES LER O PORTIFÓLIO_POSIÇÕES E PEGAR AS QUANTIDADES DE LÁ, MAS PERMITIR O USUÁRIO ALTERAR TANTO AS QUANTIDADES QUANTO OS ATIVOS SELECIONADOS. OU SEJA, QUERO QUE OS INPUTS DO USUÁRIO NA TELA PRINCIPAL ATUALIZEM O PORTIFÓLIO_POSIÇÕES
compra_especifica = {'DAP28': 5.5, 'TREASURY': 105.5}
dia_compra = {'DAP28': '2025-01-17', 'DI_31': '2025-01-17', 'WDO1':'2025-01-17', 'TREASURY': '2025-01-17'}


def checkar_portifolio(assets, 
                       quantidades, 
                       compra_especifica,  # dict { 'PETR4': 200, 'VALE3': 150 } etc.
                       dia_compra,         # pode ser dict { 'PETR4': '17/01/2025', 'VALE3': '16/01/2025' } ou string única 
                    ):
    """
    Verifica (ou cria) um CSV de portfólio e atualiza as posições de acordo com:
      1) Lista de ativos (assets) e suas quantidades (quantidades).
      2) Dicionário de compra_especifica (caso o usuário tenha quantidades específicas para algum ativo).
      3) Dia de compra, que pode ser um único valor ou um dicionário.
      4) Atualiza Preço de Compra, Preço de Ajuste Atual, Variação e Rendimento.
    """
    
    nome_arquivo_portifolio = 'portifolio_posições.csv'
    df_b3_fechamento, df_b3_variacao = processar_b3_portifolio()
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
            'Variação de Taxa', 
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
                preco_compra = df_b3_fechamento.loc[filtro_ativo, dia_compra_ativo].values[0]
                print(preco_compra)
                print(df_b3_fechamento.loc[filtro_ativo, dia_compra_ativo].values[0])

            df_portifolio.loc[asset, 'Preço de Compra'] = preco_compra


            # PREÇO DE AJUSTE ATUAL
            # Usa a data de hoje (coluna de hoje) no df_b3_fechamento
            preco_ajuste_atual = df_b3_fechamento.loc[filtro_ativo, data_hoje_str].values[0]
            df_portifolio.loc[asset, 'Preço de Ajuste Atual'] = preco_ajuste_atual
            
            # VARIAÇÃO DE TAXA
            # Depende se você quer pegar a variação apenas do dia_compra_ativo ou a variação acumulada.
            # Abaixo apenas um exemplo de pegar a variação no dia de compra (ou outro critério)
            filtro_variacao = (df_b3_variacao['Assets'] == asset)
            variacao_taxa = df_b3_variacao.loc[filtro_variacao, dia_compra_ativo].values[0]
            df_portifolio.loc[asset, 'Variação de Taxa'] = variacao_taxa
            st.write(variacao_taxa)
            st.write(df_portifolio.loc[asset, 'Variação de Taxa'])
            
            # RENDIMENTO
            # De acordo com seu snippet original, mas usualmente seria (Preço Atual - Preço Compra).
            df_portifolio.loc[asset, 'Rendimento'] = (
                df_portifolio.loc[asset, 'Quantidade'] * 
                (df_portifolio.loc[asset, 'Preço de Ajuste Atual'] - df_portifolio.loc[asset, 'Preço de Compra'])
            )
            
        except Exception as e:
            # Caso alguma coluna não seja encontrada ou dê erro, você pode tratar aqui
            print(f"[*] Erro ao calcular valores para o ativo {asset}: {e}")
    
    # 5) Salva o DataFrame atualizado de volta no CSV
    df_portifolio.to_csv(nome_arquivo_portifolio)
    
    return df_portifolio



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



# --------------------------------------------------------
#            CALLBACKS (para evitar double-click)
# --------------------------------------------------------
def switch_to_page2():
    """Callback para ir para a página 2, usando dados temporários."""
    # Transferir o que guardamos no 'posicoes_temp' e 'ativos_temp' para as chaves finais
    st.session_state["posicoes"] = st.session_state.get("posicoes_temp", {})
    st.session_state["ativos_selecionados"] = st.session_state.get("ativos_temp", [])
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

    df_pl_processado, soma_pl, soma_pl_sem_pesos = process_portfolio(df_pl, Weights)

    df = pd.read_excel(file_bbg, sheet_name='BZ RATES', skiprows=1, thousands='.', decimal=',')
    df.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 26'], axis=1, inplace=True)
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

    default_assets,quantidade_inicial = processar_dados_port()

    st.sidebar.write("## Ativos do Portifólio")
    assets = st.sidebar.multiselect("Selecione os ativos:",
                                    list(df.columns),
                                    default_assets)

    if len(assets) > 0:
        df_precos, df_completo = load_and_process_excel(df, assets)
        df_retorno = process_returns(df_completo, assets)
        var_ativos = var_not_parametric(df_retorno).abs()
        df_precos_ajustados = adjust_prices_with_var(df_precos, var_ativos)

        # ---------------------
        # ENTRADA DE QUANTIDADES
        # ---------------------
        st.write(quantidade_inicial)
        st.write(df_precos_ajustados.index)
        qtd_input = []
        quantidade_nomes = {}
        for col in df_precos_ajustados.index:
            if col in quantidade_inicial:
                val = st.sidebar.number_input(
                    f"Quantidade para {col}:", min_value=-10000, value=quantidade_inicial[col], step=1
                )
            else:
                val = st.sidebar.number_input(
                    f"Quantidade para {col}:", min_value=-10000, value=1, step=1
                )
            qtd_input.append(val)
            quantidade_nomes[col] = val
        quantidade = np.array(qtd_input)

        # Valor do Portfólio (soma simples)
        vp = df_precos_ajustados['Valor Fechamento'] * abs(quantidade)
        vp_soma = vp.sum()

        # Pesos (p/ cálculo de VaR, etc.)
        pesos = quantidade * df_precos_ajustados['Valor Fechamento'] / vp_soma
        df_returns_portifolio = df_retorno * pesos.values
        df_returns_portifolio['Portifolio'] = df_returns_portifolio.sum(axis=1)

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
        df_mvar_dinheiro = df_mvar * df_precos_ajustados['Valor Fechamento']

        covar = df_mvar * pesos.values * vp_soma
        covar_perc = covar / covar.sum()

        cvar = df_retorno[df_retorno['Portifolio'] < var_not_parametric(
            df_returns_portifolio['Portifolio'])]['Portifolio'].mean()

        df_divone, dolar = load_and_process_divone(file_bbg, df)
        # --- Exemplo de cálculo de stress e DIVONE (mesmo que seu original) ---
        lista_juros_interno = [asset for asset in assets if 'DI' in asset or 'DAP' in asset]
        df_divone_juros_interno = df_divone[lista_juros_interno]

        lista_quantidade = [quantidade_nomes[asset] for asset in lista_juros_interno]
        df_divone_juros_interno = df_divone_juros_interno * np.array(lista_quantidade)
        df_divone_juros_interno = df_divone_juros_interno.sum(axis=1)

        lista_juros_externo = [asset for asset in assets if 'TREASURY' in asset]
        df_divone_juros_externo = df_divone[lista_juros_externo]
        lista_quantidade = [quantidade_nomes[asset] for asset in lista_juros_externo]
        df_divone_juros_externo = df_divone_juros_externo * np.array(lista_quantidade)
        df_divone_juros_externo = df_divone_juros_externo.sum(axis=1)

        stress_test_juros_interno = df_divone_juros_interno * 100
        stress_test_juros_interno_percent = stress_test_juros_interno / soma_pl_sem_pesos * 10000

        stress_test_juros_externo = df_divone_juros_externo * 100
        stress_test_juros_externo_percent = stress_test_juros_externo / soma_pl_sem_pesos * 10000

        lista_dolar = [asset for asset in assets if 'WDO1' in asset]
        if lista_dolar:
            quantidade_dolar = quantidade_nomes[lista_dolar[0]]
            stress_dolar = quantidade_dolar * dolar * 0.02
        else:
            stress_dolar = 0

        df_stress_div01 = pd.DataFrame({
            'DIV01': [
                f"R${df_divone_juros_interno.iloc[0]:.2f}",
                f"R${df_divone_juros_externo.iloc[0]:.2f}",
                ''
            ],
            'Stress (R$)': [
                f"R${stress_test_juros_interno['FUT_TICK_VAL']:.2f}",
                f"R${stress_test_juros_externo['FUT_TICK_VAL']:.2f}",
                f"R${stress_dolar:.2f}"
            ],
            'Stress (bps)': [
                f"{stress_test_juros_interno_percent['FUT_TICK_VAL']:.2f}bps",
                f"{stress_test_juros_externo_percent['FUT_TICK_VAL']:.2f}bps",
                ''
            ]
        }, index=['Book Brasil', 'Book US', 'Moedas'])

        sum_row = pd.DataFrame({
            'DIV01': [f"R${df_divone_juros_interno.iloc[0] + df_divone_juros_externo.iloc[0]:.2f}"],
            'Stress (R$)': [f"R${stress_test_juros_interno['FUT_TICK_VAL'] + stress_test_juros_externo['FUT_TICK_VAL'] + stress_dolar:.2f}"],
            'Stress (bps)': [""]
        }, index=['Total'])
        df_stress_div01 = pd.concat([df_stress_div01, sum_row])

        df_precos_ajustados = calculate_portfolio_values(df_precos_ajustados, df_pl_processado, var_bps)
        df_pl_processado = calculate_contracts_per_fund(df_pl_processado, df_precos_ajustados)

        # --- Layout ---
        col1, col11, col2, col3 = st.columns([2.4, 0.2, 3.4, 1])
        with col3:
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
            st.write(f"### {abs(covar.sum()/ var_limite_comparativo):.2%} do risco total")

        with col11:
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

        with col2:
            df_dados = pd.DataFrame({
                'Beta': df_beta,
                'MVar(R$)': df_mvar_dinheiro,
                'CoVaR(R$)': covar,
                'CoVaR(%)': covar_perc,
                'Var': var_ativos[assets],
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
                sum_row['CoVaR(%)'] = sum_row['CoVaR(%)'].apply(
                    lambda x: f"{x:.2%}")
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


            st.session_state["posicoes_temp"] = quantidade_nomes
            st.session_state["ativos_temp"] = list(quantidade_nomes.keys())

            # Botão com callback (1 clique = troca de página)
            st.button(
                "Ir para a tela de Preços de Compra/Venda",
                on_click=switch_to_page2,
                key="go_page2"
            )

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
        df_precos_ajustados['Santander'] = df_precos_ajustados['Santander'].apply(lambda x: f"{x:.0f}")
        df_precos_ajustados['BTG'] = df_precos_ajustados['BTG'].apply(lambda x: f"{x:.0f}")
        df_precos_ajustados['Valor Total'] = df_precos_ajustados['Valor Total'].apply(lambda x: f"{x:.0f}")

        df_precos_ajustados['Quantidade'] = quantidade
        df_pl_processado_input = calculate_contracts_per_fund_input(df_pl_processado, df_precos_ajustados)

        # Cria uma cópia para exibição final
        df_pl_processado_print = df_pl_processado_input.copy()
        sum_row = df_pl_processado_print.select_dtypes(include='number').sum()
        sum_row['Fundos/Carteiras Adm'] = 'Total'
        sum_row['Adm'] = ''
        df_pl_processado_print = pd.concat([df_pl_processado_print, sum_row.to_frame().T], ignore_index=True)

        df_pl_processado_print.set_index('Fundos/Carteiras Adm', inplace=True)
        df_pl_processado_print = df_pl_processado_print.drop(['PL', 'Adm'], axis=1)

        # Checkboxes para exibir colunas
        default_columns = ['Adm', 'PL_atualizado']
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
        df_copy_processado.drop(colunas_df_processado, axis=1, inplace=True)

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
        if filtro_fundo:
            filtered_df = filtered_df[filtered_df["Fundos/Carteiras Adm"].isin(filtro_fundo)]
        if filtro_adm:
            filtered_df = filtered_df[filtered_df["Adm"].isin(filtro_adm)]

        sum_row = filtered_df.select_dtypes(include='number').sum()
        sum_row['Fundos/Carteiras Adm'] = 'Total'
        sum_row['Adm'] = ''
        filtered_df = pd.concat([filtered_df, sum_row.to_frame().T], ignore_index=True)

        filtered_df.index = filtered_df['Fundos/Carteiras Adm']

        if columns:
            # Formatação
            for c in columns:
                if c == 'PL_atualizado':
                    filtered_df[c] = filtered_df[c].apply(lambda x: f"R${x:,.0f}")
                else:
                    if c == 'Adm' or 'Fundos/Carteiras Adm':
                        filtered_df[c] = filtered_df[c].apply(lambda x: x)
                    else:
                        filtered_df[c] = filtered_df[c].apply(lambda x: f"{x:.2f}")

            st.table(filtered_df[columns])
            st.write("OBS: Os contratos estão arrendodandos para inteiros.")
        else:
            st.write("Selecione ao menos uma coluna para exibir os dados.")

        # Formata tabela final
        for col_ in df_pl_processado_print.columns:
            if col_ != 'PL_atualizado':
                df_pl_processado_print[col_] = df_pl_processado_print[col_].apply(lambda x: f"{x:.2f}")
            else:
                df_pl_processado_print[col_] = df_pl_processado_print[col_].apply(lambda x: f"R${x:,.0f}")

        st.write("---")
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


# ==========================================================
#   FUNÇÃO DA PÁGINA 2 (Entrar Preços de Compra/Venda)
# ==========================================================
def second_page():
    st.title("Tela de Input de Preços (Compra/Venda)")

    if "posicoes" not in st.session_state or "ativos_selecionados" not in st.session_state:
        st.error("Nenhuma posição encontrada. Volte à página anterior e selecione os ativos.")
        st.button("Voltar ao Dashboard Principal", on_click=switch_to_main, key="back_main_no_data")
        return

    ativos = st.session_state["ativos_selecionados"]
    posicoes = st.session_state["posicoes"]

    st.sidebar.header("Insira o Preço de Compra/Venda")
    precos_user = {}
    data_compra = {}
    for ativo in ativos:
        precos_user[ativo] = st.sidebar.number_input(
            f"Preço de {ativo}:", min_value=0.0, value=100.0, step=0.5
        )
        data_compra[ativo] = st.sidebar.date_input(
            f"Dia de Compra de {ativo}:",
            value=datetime.date.today()
        )

    df_port = checkar_portifolio(ativos, posicoes, precos_user, data_compra)

    
    st.write("## Resumo das posições com preços informados:")
    st.table(df_port)

    # Botão Voltar (1 clique)
    st.button("Voltar ao Dashboard Principal", on_click=switch_to_main, key="back_main_ok")


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
