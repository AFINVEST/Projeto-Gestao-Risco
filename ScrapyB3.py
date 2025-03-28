from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
from datetime import datetime, timedelta
import time
import os


def processar_dados(processed_data, hoje_str):

    # Definindo as colunas de interesse
    columns = [
        'Mercadoria',
        'Vencimento',
        'Preço de ajuste anterior',
        'Preço de ajuste Atual',
        'Variação',
        'Valor do ajuste por contrato (R$)'
    ]

    # Cria DataFrame principal
    df = pd.DataFrame(processed_data, columns=columns)

    # Ajustar o nome da mercadoria para ser as 3 primeiras letras + últimas 2 do vencimento
    df['Mercadoria'] = df['Mercadoria'].str[:3] + \
        '_' + df['Vencimento'].str[-2:]

    # Tirar o '1' do nome da DI1
    df.loc[df['Mercadoria'].str.startswith(
        'DI1'), 'Mercadoria'] = 'DI_' + df['Vencimento'].str[-2:]

    # Tirar o '1' do nome da DI1
    df.loc[df['Mercadoria'].str.startswith(
        'DAP'), 'Mercadoria'] = 'DAP' + df['Vencimento'].str[-2:]

    # Mudar nome do DOL e T10 (exemplo)
    df.loc[df['Mercadoria'].str.startswith('DOL_'), 'Mercadoria'] = 'WDO1'
    df.loc[df['Mercadoria'].str.startswith('T10_'), 'Mercadoria'] = 'TREASURY'

    # Remover a coluna Vencimento do DataFrame final
    df.drop('Vencimento', axis=1, inplace=True)

    # -------------------------------------------------
    # Agora, vamos criar (ou atualizar) 2 DataFrames:
    #  1) df_preco_de_ajuste_atual
    #  2) df_variacao
    #
    # Cada um terá:
    #   - índice = Mercadoria
    #   - colunas = datas (string) no formato YYYY-MM-DD (por exemplo)
    #
    # ---------------------------------------------------

    # 1) Transformando o df original em Série para cada componente
    #    index = Mercadoria, values = coluna correspondente
    serie_preco_ajuste = pd.Series(
        df['Preço de ajuste Atual'].values, index=df['Mercadoria'])
    serie_variacao = pd.Series(df['Variação'].values, index=df['Mercadoria'])

    # Nome da coluna que será adicionada/atualizada (data do dia)
    # hoje_str = datetime.now().strftime('%Y-%m-%d')

    # 2) Carregar (ou criar) df_preco_de_ajuste_atual
    nome_arquivo_preco_ajuste = 'df_preco_de_ajuste_atual.parquet'
    if os.path.exists(nome_arquivo_preco_ajuste):
        df_preco_de_ajuste_atual = pd.read_parquet(
            nome_arquivo_preco_ajuste)
        #Colocar como index a primeira coluna
        df_preco_de_ajuste_atual = df_preco_de_ajuste_atual.set_index(df_preco_de_ajuste_atual.columns[0])
    else:
        df_preco_de_ajuste_atual = pd.DataFrame()

    # 3) Garantir que o índice contemple todas as Mercadorias
    #    (Unir o índice atual do df com o novo índice)
    mercadorias_unificadas = df_preco_de_ajuste_atual.index.union(
        serie_preco_ajuste.index)
    df_preco_de_ajuste_atual = df_preco_de_ajuste_atual.reindex(
        index=mercadorias_unificadas)

    df_preco_de_ajuste_atual.index.name = 'Assets'

    # 4) Se a coluna (data) não existir, cria; se existir e você quiser atualizar, basta sobrescrever
    df_preco_de_ajuste_atual[hoje_str] = serie_preco_ajuste

    # 5) Salvar em parquet
    df_preco_de_ajuste_atual.reset_index(inplace=True)
    df_preco_de_ajuste_atual.to_parquet(nome_arquivo_preco_ajuste)

    # # ------------------------------------------
    # #  Mesma lógica para df_variacao
    # # ------------------------------------------
    # nome_arquivo_variacao = 'df_variacao.parquet'
    # if os.path.exists(nome_arquivo_variacao):
    #     df_variacao_atual = pd.read_parquet(nome_arquivo_variacao, index_col=0)
    # else:
    #     df_variacao_atual = pd.DataFrame()

    # # Unificar índices
    # mercadorias_unificadas = df_variacao_atual.index.union(
    #     serie_variacao.index)
    # df_variacao_atual = df_variacao_atual.reindex(index=mercadorias_unificadas)
    # df_variacao_atual.index.name = 'Assets'

    # # Cria/atualiza a coluna de hoje
    # df_variacao_atual[hoje_str] = serie_variacao

    # # Salva em parquet
    # df_variacao_atual.to_parquet(nome_arquivo_variacao)

    # nome_arquivo_variacao = 'df_variacao.parquet'
    # if os.path.exists(nome_arquivo_variacao):
    #     df_variacao_atual = pd.read_parquet(nome_arquivo_variacao, index_col=0)
    # else:
    #     df_variacao_atual = pd.DataFrame()
    print("Processamento concluído!")

# Função para gerar uma lista de dias úteis a partir de uma data inicial


def obter_dias_uteis(data_inicio):
    data_atual = data_inicio
    dias_uteis = []
    while data_atual <= datetime.now():  # Inclui até a data atual
        if data_atual.weekday() < 5:  # Dias de semana (0 = segunda-feira, 4 = sexta-feira)
            dias_uteis.append(data_atual.strftime("%d/%m/%Y"))
        data_atual += timedelta(days=1)
    return dias_uteis


# Configurar o serviço do ChromeDriver
service = Service()

# Inicializar o navegador
driver = webdriver.Chrome()

# Configurar a data inicial
# data_inicial = datetime(2024, 1, 16)  # 16/01/2024
data_inicial = datetime(2025, 3, 27)  # 16/01/2024
dias_uteis = obter_dias_uteis(data_inicial)


try:
    # Acessar o site de login
    driver.get("https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/derivativos/ajustes-do-pregao/")

    # Aguarda o carregamento da página
    time.sleep(3)

    try:
        # Localizar o campo de entrada de data
        # Esperar o iframe estar disponível e alternar para ele
        iframe = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.TAG_NAME, "iframe"))
        )
        driver.switch_to.frame(iframe)

        time.sleep(2)

        for dia in dias_uteis:
            input_data = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.ID, "dData1")))
            # Limpar o campo e inserir a data
            input_data.clear()
            input_data.send_keys(dia)

            # Clicar no botão "OK"
            botao_ok = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "button.button.expand"))
            )
            botao_ok.click()
            time.sleep(5)
            # Esperar um tempo para garantir que a página carregue os resultados
            # (Opcional) Faça aqui a coleta e processamento dos dados da tabela para cada data
            # Exemplo: Adicionar uma função para capturar os dados processados
            # processar_dados(data)

            # Localizar a tabela pelo ID
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "tblDadosAjustes"))
            )

            # Rolar até a tabela para garantir visibilidade
            driver.execute_script("arguments[0].scrollIntoView(true);", table)

            # Capturar as linhas da tabela
            rows = table.find_elements(By.CSS_SELECTOR, "tr")

            # Extrair os dados da tabela
            data_list = []
            for row in rows:
                columns = row.find_elements(By.TAG_NAME, "td")
                data = [col.text.strip() for col in columns]
                if data:
                    data_list.append(data)

            # Lista para armazenar os dados processados
            processed_data = []
            # Preciso salvar em um dataframe os dados que estão em uma tabela em que a primeira coluna tem elementos que valem para multiplas linhas. como o seguinte exemplo "['ABEVO - Contrato Futuro de ABEV3', 'F25', '11,21', '11,40', '0,19', '0,19'] ['G25', '11,21', '11,47', '0,26', '0,26'] ['H25', '11,22', '11,53', '0,31', '0,31']

            # Processar os dados
            for data in data_list:
                # Verificar o número de elementos na linha atual
                if len(data) == 6:
                    # Se houver 6 elementos, é uma nova mercadoria
                    mercadoria = data[0]
                    vencimento = data[1]
                    preco_ajuste_anterior = data[2]
                    preco_ajuste_atual = data[3]
                    variacao = data[4]
                    valor_ajuste_contrato = data[5]
                    ano = vencimento[-2:]
                    ano = int(ano)
                    if mercadoria == 'DI1 - DI de 1 dia':
                        # Se o vencimento conter a letra F é pra ser adicionado
                        if 'F' in vencimento:
                            processed_data.append([mercadoria, vencimento, preco_ajuste_anterior,
                                                   preco_ajuste_atual, variacao, valor_ajuste_contrato])
                    elif mercadoria == 'DAP - Cupom de DI x IPCA': # Ano impar pegar K, Ano par será Q
                        if ano % 2 != 0:
                            if 'K' in vencimento:
                                processed_data.append([mercadoria, vencimento, preco_ajuste_anterior,
                                                   preco_ajuste_atual, variacao, valor_ajuste_contrato])
                        else:
                            if 'Q' in vencimento: 
                                processed_data.append([mercadoria, vencimento, preco_ajuste_anterior,
                                                   preco_ajuste_atual, variacao, valor_ajuste_contrato])
                        
                    elif mercadoria == 'DOL - Dólar comercial' or mercadoria == 'T10 - US T-Note 10 anos':
                        processed_data.append([mercadoria, vencimento, preco_ajuste_anterior,
                                               preco_ajuste_atual, variacao, valor_ajuste_contrato])

                else:
                    # Se houver menos de 6 elementos, é um vencimento adicional
                    vencimento = data[0]
                    preco_ajuste_anterior = data[1]
                    preco_ajuste_atual = data[2]
                    variacao = data[3]
                    valor_ajuste_contrato = data[4]
                    ano = vencimento[-2:]
                    ano = int(ano)
                    if mercadoria == 'DI1 - DI de 1 dia':
                        # Se o vencimento conter a letra F é pra ser adicionado
                        if 'F' in vencimento:
                            processed_data.append([mercadoria, vencimento, preco_ajuste_anterior,
                                                   preco_ajuste_atual, variacao, valor_ajuste_contrato])
                    elif mercadoria == 'DAP - Cupom de DI x IPCA':
                        if ano % 2 != 0:
                            if 'K' in vencimento:
                                processed_data.append([mercadoria, vencimento, preco_ajuste_anterior,
                                                   preco_ajuste_atual, variacao, valor_ajuste_contrato])
                        else:
                            if 'Q' in vencimento:
                                processed_data.append([mercadoria, vencimento, preco_ajuste_anterior,
                                                   preco_ajuste_atual, variacao, valor_ajuste_contrato])
                # Adicionar os dados processados à lista
            # mudar o dia para o formato YYYY-MM-DD
            dia = datetime.strptime(dia, "%d/%m/%Y").strftime("%Y-%m-%d")
            processar_dados(processed_data, dia)
            # Role para o topo da página
            try:
                driver.execute_script("window.scrollTo(0, 0);")
            except Exception as e:
                print(f"Erro ao rolar para o topo da página: {e}")
            print(f"Data {dia} processada com sucesso!")
            print("Aguardando 5 segundos para a próxima data...")
            time.sleep(5)

    except Exception as e:
        print(f"Erro: {e}")

    driver.quit()

except Exception as e:
    print(e)
    driver.quit()
