from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
import time

# Configurar o serviço do ChromeDriver
service = Service()

# Inicializar o navegador
driver = webdriver.Chrome()

try:
    # Acessar o site de login
    driver.get("https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/derivativos/ajustes-do-pregao/")

    # Aguarda o carregamento da página
    time.sleep(5)

    # Esperar o iframe estar disponível e alternar para ele
    iframe = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "iframe"))
    )
    driver.switch_to.frame(iframe)

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
            processed_data.append([mercadoria, vencimento, preco_ajuste_anterior,
                        preco_ajuste_atual, variacao, valor_ajuste_contrato])
    
        else:
            # Se houver menos de 6 elementos, é um vencimento adicional
            vencimento = data[0]
            preco_ajuste_anterior = data[1]
            preco_ajuste_atual = data[2]
            variacao = data[3]
            valor_ajuste_contrato = data[4]
    
        # Adicionar os dados processados à lista

    
    # Converter para DataFrame
    columns = ['Mercadoria', 'Vencimento', 'Preço de ajuste anterior',
               'Preço de ajuste Atual', 'Variação', 'Valor do ajuste por contrato (R$)']
    df = pd.DataFrame(processed_data,columns=columns)
    
    # Export to CSV
    df.to_csv("ajustes_do_pregao.csv", index=False)
    

    # Exibir o DataFrame
    print(df)

    driver.quit()
except Exception as e:
    print(e)
    driver.quit()
