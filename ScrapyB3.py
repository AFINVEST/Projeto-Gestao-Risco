"""
AJUSTES DE PREGÃO – B3
----------------------------------------------------------
Fluxo
1. Lê e alinha df_preco_de_ajuste_atual.parquet e df_valor_ajuste_contrato.parquet
2. Descobre a última data comum gravada (last_common_dt)
3. Faz scraping (lógica original) de last_common_dt até o último dia útil antes de hoje
4. Depois do scraping, compara as **duas últimas colunas**; se todas as
   linhas forem iguais nos DOIS parquets, entende-se que a última coluna
   já é apenas uma cópia, então **não duplica** de novo.  
   Caso contrário, duplica a coluna do último dia útil real → próximo dia útil
5. Salva com índice “Assets”
----------------------------------------------------------
"""

# ───────────── imports ─────────────
import os
import time
from datetime import date, timedelta, datetime

import pandas as pd
import pandas_market_calendars as mcal
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ────────── utilitários ────────────
def ler_parquet(path: str):
    if not os.path.exists(path):
        return pd.DataFrame(), []
    df = pd.read_parquet(path)
    df = df.set_index(df.columns[0])
    df.index.name = "Assets"
    df = df[~df.index.duplicated(keep="last")]
    dates = sorted({pd.to_datetime(c).date() for c in df.columns})
    return df, dates


def garantir_coluna(df: pd.DataFrame, nova_dt: date, col_base: str):
    col = nova_dt.isoformat()
    if col not in df.columns:
        df[col] = df[col_base]


def prox_util(cal, d: date) -> date:
    return cal.valid_days(d + timedelta(days=1),
                          d + timedelta(days=10))[0].date()


def salvar(df: pd.DataFrame, path: str):
    df.index.name = "Assets"
    df.reset_index().rename(
        columns={df.reset_index().columns[0]: "Assets"}).to_parquet(path)


# ────────── caminhos / leitura ─────────
P_PRECO = "Dados/df_preco_de_ajuste_atual.parquet"
P_VALOR = "Dados/df_valor_ajuste_contrato.parquet"

df_p, dates_p = ler_parquet(P_PRECO)
df_v, dates_v = ler_parquet(P_VALOR)
if not dates_p or not dates_v:
    raise ValueError("Parquets vazios. Grave manualmente a primeira data.")

# ────────── alinhar colunas ───────────
all_dates = sorted(set(dates_p) | set(dates_v))
for d in all_dates:
    if d.isoformat() not in df_p.columns:
        prev = max(x for x in dates_p if x <= d)
        garantir_coluna(df_p, d, prev.isoformat())
    if d.isoformat() not in df_v.columns:
        prev = max(x for x in dates_v if x <= d)
        garantir_coluna(df_v, d, prev.isoformat())

dates_p = sorted({pd.to_datetime(c).date() for c in df_p.columns})
dates_v = sorted({pd.to_datetime(c).date() for c in df_v.columns})
last_common_dt = min(dates_p[-1], dates_v[-1])

# ────────── calendário B3 ────────────
cal = mcal.get_calendar("B3")
today = date.today()
ultimo_util = cal.valid_days(today - timedelta(days=10), today)[-1].date()
dias_scrap = [d.date() for d in cal.valid_days(last_common_dt, ultimo_util)]

# ────────── funções p/ normalizar e gravar ──────────


def normalizar(df_raw):
    df = pd.DataFrame(df_raw, columns=["Mercadoria", "Vencimento",
                                       "Preço de ajuste anterior", "Preço de ajuste Atual",
                                       "Variação", "Valor do ajuste por contrato (R$)"])
    df["Mercadoria"] = df["Mercadoria"].str[:3] + \
        "_" + df["Vencimento"].str[-2:]
    df.loc[df["Mercadoria"].str.startswith(
        "DI1"), "Mercadoria"] = "DI_" + df["Vencimento"].str[-2:]
    df.loc[df["Mercadoria"].str.startswith(
        "DAP"), "Mercadoria"] = "DAP" + df["Vencimento"].str[-2:]
    df.loc[df["Mercadoria"].str.startswith("DOL_"), "Mercadoria"] = "WDO1"
    df.loc[df["Mercadoria"].str.startswith("T10_"), "Mercadoria"] = "TREASURY"
    df.drop("Vencimento", axis=1, inplace=True)
    neg = df["Variação"].str.startswith("-")
    df.loc[neg, "Valor do ajuste por contrato (R$)"] = "-" + \
        df.loc[neg, "Valor do ajuste por contrato (R$)"].astype(str)
    return df


def gravar(df_norm, d: date):
    global df_p, df_v
    col = d.isoformat()
    s_preco = pd.Series(df_norm["Preço de ajuste Atual"].values,
                        index=df_norm["Mercadoria"]).groupby(level=0).last()
    s_valor = pd.Series(df_norm["Valor do ajuste por contrato (R$)"].values,
                        index=df_norm["Mercadoria"]).groupby(level=0).last()
    df_p = df_p.reindex(df_p.index.union(s_preco.index))
    df_v = df_v.reindex(df_v.index.union(s_valor.index))
    df_p[col] = s_preco
    df_v[col] = s_valor
    print(f"   • {col} gravado")


# ────────── scraping (lógica ORIGINAL) ─────────
if dias_scrap:
    print("Scraping →", ", ".join(d.strftime("%Y-%m-%d") for d in dias_scrap))
    driver = webdriver.Chrome(service=Service())
    try:
        driver.get(
            "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/derivativos/ajustes-do-pregao/")
        time.sleep(1)
        iframe = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.TAG_NAME, "iframe")))
        driver.switch_to.frame(iframe)
        time.sleep(1)

        for d in dias_scrap:
            dia_fmt = d.strftime("%d/%m/%Y")
            input_box = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.ID, "dData1")))
            input_box.clear()
            input_box.send_keys(dia_fmt)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "button.button.expand"))
            ).click()
            time.sleep(2)

            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "tblDadosAjustes")))
            driver.execute_script("arguments[0].scrollIntoView(true);", table)
            rows = table.find_elements(By.CSS_SELECTOR, "tr")

            data_list = [
                [c.text.strip() for c in r.find_elements(By.TAG_NAME, "td")]
                for r in rows if r.find_elements(By.TAG_NAME, "td")
            ]

            processed, mercadoria = [], None
            for ln in data_list:
                if len(ln) == 6:
                    mercadoria, venc, pa_ant, pa_atu, var, val_aj = ln
                else:
                    venc, pa_ant, pa_atu, var, val_aj = ln
                ano = int(venc[-2:])

                if mercadoria == 'DI1 - DI de 1 dia':
                    if 'F' not in venc:
                        continue
                elif mercadoria == 'DAP - Cupom de DI x IPCA':
                    if (ano % 2 and 'K' not in venc) or (ano % 2 == 0 and 'Q' not in venc):
                        continue
                elif mercadoria not in ('DOL - Dólar comercial', 'T10 - US T-Note 10 anos'):
                    continue

                processed.append(
                    [mercadoria, venc, pa_ant, pa_atu, var, val_aj])

            gravar(normalizar(processed), d)
            time.sleep(1.5)

    finally:
        driver.quit()
else:
    print("Nenhum dia para scraping.")

# ────────── DUPLICAR apenas se necessário ─────────
cols_sorted = sorted(df_p.columns)
if len(cols_sorted) >= 2:
    last_col, prev_col = cols_sorted[-1], cols_sorted[-2]
    iguais_preco = df_p[last_col].equals(df_p[prev_col])
    iguais_valor = df_v[last_col].equals(df_v[prev_col])
else:
    iguais_preco = iguais_valor = False   # só 1 coluna → precisa duplicar

if iguais_preco and iguais_valor:
    print("Última coluna já é cópia idêntica — não duplica de novo.")
else:
    last_dt = pd.to_datetime(cols_sorted[-1]).date()
    prox_dt = prox_util(cal, last_dt)
    garantir_coluna(df_p, prox_dt, last_col)
    garantir_coluna(df_v, prox_dt, last_col)
    print(f"Coluna duplicada para o próximo dia útil: {prox_dt}")

# ────────── salvar finais ─────────
salvar(df_p, P_PRECO)
salvar(df_v, P_VALOR)
print("Processo finalizado.")
