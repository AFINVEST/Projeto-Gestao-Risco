from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd


# Configurar o serviço do ChromeDriver
service = Service()

# Inicializar o navegador
driver = webdriver.Chrome()

try:
    # Acessar o site de login
    driver.get("https://afinvest.com.br/login/interno")

    # Esperar até que os campos de login estejam visíveis
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "atributo")))
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "passwordLogin")))

    # Fazer login
    driver.find_element(By.ID, "atributo").send_keys(
        "emanuel.cabral@afinvest.com.br")
    driver.find_element(By.ID, "passwordLogin").send_keys("Afs@2024")
    driver.find_element(By.ID, "loginInterno").click()

    # Aguardar até que a tabela esteja visível na página
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "table_profitability_CDI")))

    # Salvar a página para visualização
    # driver.save_screenshot("tabela_rentabilidade.png")

    # Mover o cursor para a tabela para garantir que ela esteja visível
    table = driver.find_element(By.ID, "table_profitability_CDI")
    ActionChains(driver).move_to_element(table).perform()

    # Capturar o conteúdo da tabela
    rows = driver.find_elements(By.CSS_SELECTOR, "#table_profitability_CDI tr")

    # printar e salvar o conteúdo da tabela
    data_list = []

    for row in rows:
        columns = row.find_elements(By.TAG_NAME, "td")
        data = [col.text.strip() for col in columns]
        if data:
            print(data)  # Apenas para verificar os dados durante a execução
            data_list.append(data)  # Adiciona os dados à lista

    # Converte a lista de dados em um DataFrame
    df = pd.DataFrame(data_list)

    columns = "Fundos/Carteiras Adm", "PL", "17/12/2024", "Semana", "MTD", "YTD", "30D", "90D", "180D", "12M", "24M", "36M", "Desde Cri.", "Dt de Cri."

    df.columns = columns

    rows = driver.find_elements(
        By.CSS_SELECTOR, "#table_profitability_IMA-B tr")

    # add dados da tabela IMA-B
    data_list = []

    for row in rows:
        columns = row.find_elements(By.TAG_NAME, "td")
        data = [col.text.strip() for col in columns]
        if data:
            print(data)
            data_list.append(data)

    df2 = pd.DataFrame(data_list)
    columns = "Fundos/Carteiras Adm", "PL", "17/12/2024", "Semana", "MTD", "YTD", "30D", "90D", "180D", "12M", "24M", "36M", "Desde Cri.", "Dt de Cri."
    df2.columns = columns

    # concatenar os dois dataframes
    df = pd.concat([df, df2])

    # Atualizar o index
    df.reset_index(drop=True, inplace=True)

    # drop colunas menos as duas primeiras
    df = df.drop(columns=["Semana", "17/12/2024", "MTD", "YTD", "30D",
                 "90D", "180D", "12M", "24M", "36M", "Desde Cri.", "Dt de Cri."])

    print(df)
    df = df[df['Fundos/Carteiras Adm'] != 'CDI']

    # Salvar o DataFrame em um arquivo parquet
    df.to_parquet("Dados/pl_fundos.parquet")

except Exception as e:
    print(f"Ocorreu um erro: {e}")

finally:
    # Fechar o navegador
    # driver.quit()
    print("Terminou a execução")
