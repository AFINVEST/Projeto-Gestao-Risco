from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import os
from datetime import datetime, timedelta

def get_last_value(row, date_columns):
    """Encontra o último valor não nulo/não vazio em uma linha"""
    for date in reversed(date_columns):
        value = row.get(date, "--")
        if pd.notna(value) and value != "--":
            return value
    return "--"

def main():
    # Configurações iniciais
    service = Service()
    driver = webdriver.Chrome(service=service)
    csv_path = "pl_fundos_teste.csv"
    fundos_base = [
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

    try:
        # Verificar se o arquivo existe
        if os.path.exists(csv_path):
            df_todos = pd.read_csv(csv_path)
            existing_dates = [col for col in df_todos.columns if col not in ["Fundos/Carteiras Adm", "Último Valor","Unnamed: 0"]]
            last_date = max([datetime.strptime(d, "%Y-%m-%d") for d in existing_dates]) if existing_dates else None
        else:
            df_todos = pd.DataFrame(fundos_base, columns=["Fundos/Carteiras Adm", "Valor"])
            df_todos = df_todos[["Fundos/Carteiras Adm"]]
            last_date = None

        # Definir datas de coleta
        start_date = last_date + timedelta(days=1) if last_date else datetime(2025, 1, 1)
        end_date = datetime.today()

        if start_date > end_date:
            print("Nenhuma nova data para coletar.")
            return
        
        # Login
        driver.get("https://afinvest.com.br/login/interno")
        time.sleep(2)  # Espera para renderização
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "atributo"))).send_keys("emanuel.cabral@afinvest.com.br")
        driver.find_element(By.ID, "passwordLogin").send_keys("Afs@2024")
        driver.find_element(By.ID, "loginInterno").click()

        # Navegação
        driver.get("https://afinvest.com.br/interno/relatorios/patrimonios")
        time.sleep(3)  # Espera para renderização
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "button.btn.btn-outline-primary[data-type='custom']"))).click()
        time.sleep(3)  # Espera para renderização
        date_input = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "date_patrimony_table_fundo")))

        # Coleta de dados
        current_date = start_date
        while current_date <= end_date:
            formatted_date = current_date.strftime("%d/%m/%Y")
            date_input.clear()
            date_input.send_keys(formatted_date + Keys.RETURN)
            date_input.send_keys(formatted_date + Keys.RETURN)
            
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "table_patrimony")))
            time.sleep(2)  # Espera para renderização

            # Coletar dados da tabela
            rows = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#table_patrimony tbody tr"))
            )
            
            new_data = {row.find_elements(By.TAG_NAME, "td")[0].text: row.find_elements(By.TAG_NAME, "td")[1].text 
                       for row in rows if len(row.find_elements(By.TAG_NAME, "td")) > 1}

            # Adicionar nova coluna
            col_name = current_date.strftime("%Y-%m-%d")
            df_todos[col_name] = df_todos["Fundos/Carteiras Adm"].map(new_data).fillna("--")
            
            current_date += timedelta(days=1)

        # Na seção de atualização do DataFrame:
        date_columns = sorted([col for col in df_todos.columns if col.startswith("202")], reverse=True)
        df_todos["Último Valor"] = df_todos.apply(lambda row: get_last_value(row, date_columns), axis=1)

        # Ordenar colunas
        columns_order = ["Fundos/Carteiras Adm"] + sorted(date_columns) + ["Último Valor"]
        df_todos = df_todos[columns_order]
        
        # Salvar arquivo
        df_todos.to_csv(csv_path, index=False)
        print("Dados atualizados com sucesso!")

    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
    
    finally:
        driver.quit()

if __name__ == "__main__":
    main()