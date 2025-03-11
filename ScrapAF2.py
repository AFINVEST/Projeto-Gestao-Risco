from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import datetime
from datetime import datetime, timedelta

def get_last_value(row):
    for date in reversed(date_columns):  # Percorrer as colunas da esquerda para a direita
        if row[date] != "--":  # Se o valor não for "--", retorna o valor
            return row[date]
    return None  # Caso não tenha nenhum valor

# Configurar o serviço do ChromeDriver
service = Service()

# Inicializar o navegador
driver = webdriver.Chrome()

try:
    # Acessar o site de login
    driver.get("https://afinvest.com.br/login/interno")

    # Esperar até que os campos de login estejam visíveis
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "atributo")))
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "passwordLogin")))

    # Fazer login
    driver.find_element(By.ID, "atributo").send_keys("emanuel.cabral@afinvest.com.br")
    driver.find_element(By.ID, "passwordLogin").send_keys("Afs@2024")
    driver.find_element(By.ID, "loginInterno").click()

    driver.get("https://afinvest.com.br/interno/relatorios/patrimonios")
    time.sleep(5)
    try:
        # Apertar o botão de definir data
        definir_data_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "button.btn.btn-outline-primary[data-type='custom']"))
        )
        
        # Clique usando JavaScript para evitar possíveis problemas de interatividade
        driver.execute_script("arguments[0].click();", definir_data_button)
        # Clique no botão
    except Exception as e:
        print(f"Erro ao clicar no botão de definir data: {e}")

    time.sleep(5)

    # Defina o intervalo de datas
    start_date = datetime(2025, 1, 1)  # 1º de janeiro de 2025
    end_date = datetime.today()        # Data de hoje

    # Localize o campo de data pelo ID
    date_input = driver.find_element(By.ID, "date_patrimony_table_fundo")

    # Itere sobre todas as datas no intervalo
    current_date = start_date

    # Dados fornecidos
    fundos_data = [
            ("AF DEB INCENTIVADAS", "R$ 11.660.312,30"),
            ("AF INVEST GERAES PREV", "--"),
            ("AF TÁTICO", "--"),
            ("ALFA HORIZON FIA", "R$ 11.139.951,87"),
            ("AYA NMK FIM", "--"),
            ("BBRASIL FIM CP RESP", "--"),
            ("BH FIM", "--"),
            ("BH FIRF INFRA", "--"),
            ("BMG SEG", "R$ 116.604.533,42"),
            ("BORDEAUX FIM", "--"),
            ("BORDEAUX INFRA", "--"),
            ("FIA MINAS", "R$ 29.558.159,92"),
            ("FIRF GERAES", "R$ 527.440.065,49"),
            ("FIRF GERAES 30", "R$ 277.327.187,86"),
            ("GLOBAL BONDS", "R$ 28.816.807,45"),
            ("HORIZONTE", "R$ 242.504.609,40"),
            ("JERA2026", "--"),
            ("MANACA INFRA FIRF", "--"),
            ("MINAS O.N.E. FIA", "R$ 1.902.353,24"),
            ("REAL FIM", "--"),
            ("ROMEU FC FIM CP IE", "R$ 214.564.570,12"),
            ("SANKALPA FIM", "--"),
            ("SANTANA", "--"),
            ("TOPAZIO FIM", "--"),
            ("TOPAZIO INFRA", "--"),
            ("TOTAL", "R$ 1.461.518.551,07")
        ]

    # Criar o DataFrame inicial com os dados de fundos
    df_todos = pd.DataFrame(fundos_data, columns=["Fundos/Carteiras Adm", "Valor"])
    df_todos.drop(columns=["Valor"], inplace=True)

    while current_date <= end_date:
        # Formate a data no formato esperado (ex.: DD/MM/YYYY)
        formatted_date = current_date.strftime("%d/%m/%Y")
        
        # Limpe o campo de data e insira a data formatada
        date_input.clear()
        date_input.send_keys(formatted_date)
        date_input.send_keys(Keys.RETURN)
        date_input.send_keys(Keys.RETURN)

        
        # Aguarde tempo para carregar a página ou dados (ajuste conforme necessário)
        time.sleep(5)
        
        # Avance para o próximo dia
        # Aguardar até que a tabela esteja visível na página
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "table_patrimony")))

        # Salvar a página para visualização
        #driver.save_screenshot("tabela_rentabilidade.png")

        # Mover o cursor para a tabela para garantir que ela esteja visível
        table = driver.find_element(By.ID, "table_patrimony")
        ActionChains(driver).move_to_element(table).perform()

        # Capturar o conteúdo da tabela
        rows = driver.find_elements(By.CSS_SELECTOR, "#table_patrimony tbody tr")
        
        #printar e salvar o conteúdo da tabela
        data_list = []

        for row in rows:
            columns = row.find_elements(By.TAG_NAME, "td")
            data = [col.text.strip() for col in columns]
            if data:
                #print(data)  # Apenas para verificar os dados durante a execução
                data_list.append(data)  # Adiciona os dados à lista

        # Converte a lista de dados em um DataFrame
        df = pd.DataFrame(data_list)
        
        columns = "Fundos/Carteiras Adm","Data_Dia", "30 dias", "60 dias", "180 dias", "12 meses", "24 meses", "36 meses"
        
        df.columns = columns

        #Atualizar o index
        df.reset_index(drop=True, inplace=True)
        
        #drop colunas menos as duas primeiras
        df = df.drop(columns=["30 dias", "60 dias", "180 dias", "12 meses", "24 meses", "36 meses"])

        print(f'Dia {current_date.strftime("%d/%m/%Y")} coletado com sucesso!')

        # Concatenar os DataFrames e mudar data para formato 2025-01-02
        dia = current_date.strftime("%Y-%m-%d")
        df_todos[dia] = df["Data_Dia"]

        current_date += timedelta(days=1)
    try:
        date_columns = df.columns[1:]  # Exclui a coluna 'Fundo'
        # Salvar o DataFrame em um arquivo CSV
        #Criar uma coluna com a data mais recente com dados de cada fundO

        # Criar a nova coluna com o último valor
        df_todos['Último Valor'] = df.apply(get_last_value(), axis=1)
        df_todos.to_csv("pl_fundos_teste.csv")
    except Exception as e:
        print(f"Erro ao salvar o DataFrame em um arquivo CSV: {e}")

except Exception as e:
    print(f"Ocorreu um erro: {e}")

finally:
    # Fechar o navegador
    driver.quit()