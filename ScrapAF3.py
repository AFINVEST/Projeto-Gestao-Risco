from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import os
from datetime import datetime, timedelta, date
import re  # ⇢ para tratar “R$”

################### Cuidado -> Codigo pode quebrar com mudanças no site, como a adiação de novos fundos ###################

# ──────────────────────────────────────────────────────────────────────────────
# Parâmetros
# ──────────────────────────────────────────────────────────────────────────────
PARQUET_PATH = "Dados/pl_fundos_teste.parquet"
# PARQUET_PATH = "pl_fundos.parquet"

# Fundo que deve zerar a partir de CUTOFF_DATE
TARGET_FUND = "AF DEB INCENTIVADAS"
CUTOFF_DATE = datetime(2025, 9, 1).date()  # 01/09/2025
ZERO_TXT = "R$ 0,00"

# Janela para caçar “repetidos” (e re-scrapar)
LOOKBACK_DAYS = 120  # ajuste se quiser

# Índices dos fundos que compõem o TOTAL (mesma regra original)
INDICES_TOTAL = [0, 7, 10, 14, 15, 16, 17, 20, 25]


# ──────────────────────────────────────────────────────────────────────────────
# Funções de apoio
# ──────────────────────────────────────────────────────────────────────────────
def get_last_value(row, date_columns):
    """Encontra o último valor não nulo/não vazio em uma linha."""
    for date in reversed(date_columns):
        value = row.get(date, "--")
        if pd.notna(value) and value != "--":
            return value
    return "--"


def _txt_to_float(txt: str) -> float:
    """Converte 'R$ 1.234.567,89' → 1234567.89   e '--' → 0.0"""
    if not isinstance(txt, str) or txt.strip() in ("", "--"):
        return 0.0
    num = (txt.replace("R$", "").replace(".", "").replace(",", ".").strip())
    try:
        return float(num)
    except ValueError:
        return 0.0


def _float_to_txt(val: float) -> str:
    """Converte 1234567.89 → 'R$ 1.234.567,89'"""
    inteiro, frac = f"{val:,.2f}".split(".")
    inteiro = inteiro.replace(",", ".")
    return f"R$ {inteiro},{frac}"


def _list_date_columns(df: pd.DataFrame) -> list[str]:
    """Retorna colunas que são datas no formato YYYY-MM-DD, ordenadas."""
    cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c))]
    return sorted(cols)


def _to_date(s: str) -> date | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _find_repeated_dates_for_rescrape(df: pd.DataFrame,
                                      lookback_days: int,
                                      ignore_set: set[str]) -> set[date]:
    """
    Procura, nas últimas `lookback_days`, valores repetidos (igual ao dia anterior)
    que NÃO pertencem ao `ignore_set`. Retorna o conjunto de datas a re-scrapar.
    """
    if df.empty:
        return set()

    date_cols = _list_date_columns(df)
    if not date_cols:
        return set()

    # Limitar às últimas N datas
    cutoff = datetime.today().date() - timedelta(days=lookback_days)
    date_cols_recent = [c for c in date_cols if (_to_date(c) and _to_date(c) >= cutoff)]
    if len(date_cols_recent) < 2:
        return set()

    dates_to_refresh = set()

    # Itera linhas exceto TOTAL
    mask_not_total = df["Fundos/Carteiras Adm"].astype(str) != "TOTAL"
    df_fundos = df.loc[mask_not_total, ["Fundos/Carteiras Adm"] + date_cols_recent].copy()

    # Para cada fundo, compara com a coluna anterior
    for _, row in df_fundos.iterrows():
        vals = row[date_cols_recent].tolist()
        for j in range(1, len(date_cols_recent)):
            v_prev = vals[j - 1]
            v_curr = vals[j]
            if pd.isna(v_curr) or pd.isna(v_prev):
                continue
            if (str(v_curr) == str(v_prev)) and (str(v_curr).strip() not in ignore_set):
                d = _to_date(date_cols_recent[j])
                if d:
                    dates_to_refresh.add(d)

    return dates_to_refresh


# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Configurações iniciais do Selenium
    service = Service()
    driver = webdriver.Chrome(service=service)

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
        ("MINAS DIVIDENDOS", "--"),
        ("MINAS O.N.E. FIA", "R$ 1.902.353,24"),
        ("REAL FIM", "--"),
        ("ROMEU FC FIM CP IE", "R$ 214.564.570,12"),
        ("SANKALPA FIM", "--"),
        ("SANTANA", "--"),
        ("TOPAZIO FIM", "--"),
        ("TOPAZIO INFRA", "--"),
        ("TOTAL", "R$ 1.461.518.551,07"),
    ]

    try:
        # 1) Leitura do parquet (ou cria base)
        if os.path.exists(PARQUET_PATH):
            df_todos = pd.read_parquet(PARQUET_PATH)
            existing_dates = _list_date_columns(df_todos)
            last_date = (max(datetime.strptime(d, "%Y-%m-%d") for d in existing_dates)
                         if existing_dates else None)
        else:
            df_todos = pd.DataFrame(fundos_base, columns=["Fundos/Carteiras Adm", "Valor"])
            df_todos = df_todos[["Fundos/Carteiras Adm"]]
            last_date = None

        # 2) Determina datas novas e datas que precisam re-scrape por “repetidos”
        start_date = last_date + timedelta(days=1) if last_date else datetime(2025, 1, 1)
        end_date = datetime.today()

        # conjunto com datas novas
        new_dates = set()
        if start_date <= end_date:
            d = start_date.date()
            while d <= end_date.date():
                new_dates.add(d)
                d += timedelta(days=1)

        # conjunto com datas “repetidas” (para re-scrape)
        ignore_dupes = {"--", ZERO_TXT, "R$0,00", "R$0", "R$ 0,00"}
        dates_repeated = _find_repeated_dates_for_rescrape(df_todos, LOOKBACK_DAYS, ignore_dupes)

        # calendário final de scrape
        dates_to_scrape = sorted(new_dates.union(dates_repeated))

        if not dates_to_scrape:
            print("Nenhuma data nova ou repetida para atualizar.")
            # Ainda assim, reforça a regra de zerar AF DEB ≥ CUTOFF_DATE e salva
            # (para garantir consistência caso o arquivo tenha sido editado fora)
            date_cols_all = _list_date_columns(df_todos)
            for col in date_cols_all:
                dcol = _to_date(col)
                if dcol and dcol >= CUTOFF_DATE:
                    df_todos.loc[df_todos["Fundos/Carteiras Adm"] == TARGET_FUND, col] = ZERO_TXT
                    soma = df_todos.loc[INDICES_TOTAL, col].apply(_txt_to_float).sum()
                    df_todos.loc[df_todos["Fundos/Carteiras Adm"] == "TOTAL", col] = _float_to_txt(soma)
            # Atualiza “Último Valor” e salva
            _finalize_and_save(df_todos)
            return

        # 3) Login
        driver.get("https://afinvest.com.br/login/interno")
        time.sleep(2)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "atributo"))).send_keys("emanuel.cabral@afinvest.com.br")
        driver.find_element(By.ID, "passwordLogin").send_keys("Afs@2024")
        driver.find_element(By.ID, "loginInterno").click()

        # 4) Vai para o relatório e seleciona data custom
        driver.get("https://afinvest.com.br/interno/relatorios/patrimonios")
        time.sleep(3)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "button.btn.btn-outline-primary[data-type='custom']"))).click()
        time.sleep(3)
        date_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "date_patrimony_table_fundo"))
        )

        # 5) Loop nas datas (novas + repetidas)
        for d in dates_to_scrape:
            formatted_date = d.strftime("%d/%m/%Y")
            date_input.clear()
            date_input.send_keys(formatted_date + Keys.RETURN)
            date_input.send_keys(formatted_date + Keys.RETURN)
            time.sleep(3)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "table_patrimony")))
            time.sleep(2)

            # Coleta da tabela
            rows = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#table_patrimony tbody tr"))
            )
            new_data = {
                row.find_elements(By.TAG_NAME, "td")[0].text: row.find_elements(By.TAG_NAME, "td")[1].text
                for row in rows if len(row.find_elements(By.TAG_NAME, "td")) > 1
            }

            # Regra AF DEB = 0 a partir do CUTOFF_DATE
            if d >= CUTOFF_DATE:
                new_data[TARGET_FUND] = ZERO_TXT

            # Garante coluna no DF e preenche
            col_name = d.strftime("%Y-%m-%d")
            if "Fundos/Carteiras Adm" not in df_todos.columns:
                # Caso excepcional: se arquivo foi corrompido, recria col base
                df_todos = pd.DataFrame(fundos_base, columns=["Fundos/Carteiras Adm", "Valor"])[["Fundos/Carteiras Adm"]]

            if col_name not in df_todos.columns:
                df_todos[col_name] = "--"

            df_todos[col_name] = df_todos["Fundos/Carteiras Adm"].map(new_data).fillna("--")

            # Recalcula TOTAL da coluna
            soma = df_todos.loc[INDICES_TOTAL, col_name].apply(_txt_to_float).sum()
            df_todos.loc[df_todos["Fundos/Carteiras Adm"] == "TOTAL", col_name] = _float_to_txt(soma)

        # 6) Reforço: zera AF DEB e recalc TOTAL em TODAS as colunas >= CUTOFF_DATE
        date_cols_all = _list_date_columns(df_todos)
        for col in date_cols_all:
            dcol = _to_date(col)
            if dcol and dcol >= CUTOFF_DATE:
                df_todos.loc[df_todos["Fundos/Carteiras Adm"] == TARGET_FUND, col] = ZERO_TXT
                soma = df_todos.loc[INDICES_TOTAL, col].apply(_txt_to_float).sum()
                df_todos.loc[df_todos["Fundos/Carteiras Adm"] == "TOTAL", col] = _float_to_txt(soma)

        # 7) Finaliza (Último Valor, ordenar colunas, forward-fill de "--") e salva
        _finalize_and_save(df_todos)

        print("Dados atualizados com sucesso!")

    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")

    finally:
        driver.quit()


def _finalize_and_save(df_todos: pd.DataFrame) -> None:
    """Atualiza 'Último Valor', ordena colunas, aplica forward-fill de '--' e salva parquet."""
    date_columns = _list_date_columns(df_todos)
    # Atualiza “Último Valor”
    df_todos["Último Valor"] = df_todos.apply(lambda row: get_last_value(row, date_columns), axis=1)
    # Ordena colunas
    columns_order = ["Fundos/Carteiras Adm"] + date_columns + ["Último Valor"]
    df_todos = df_todos[columns_order]

    # Forward-fill apenas de "--" (mantém 0,00 quando for o caso)
    for i in range(2, len(df_todos.columns) - 1):  # pula nome e 1ª data; evita "Último Valor"
        col_atual = df_todos.columns[i]
        col_prev = df_todos.columns[i - 1]
        if col_atual not in ("Fundos/Carteiras Adm",):
            df_todos[col_atual] = df_todos[col_atual].mask(
                df_todos[col_atual] == "--", df_todos[col_prev]
            )

    # Salvar parquet
    df_todos.to_parquet(PARQUET_PATH, index=False)


if __name__ == "__main__":
    main()
