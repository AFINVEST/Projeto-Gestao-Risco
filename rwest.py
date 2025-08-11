# -*- coding: utf-8 -*-
"""
Atualiza o parquet exclusivo de DI (Dados/df_valor_ajuste_DI.parquet)
— apenas para os dias faltantes — via scraping da página de "Ajustes do pregão" da B3.

Critérios e fluxo:
1) Lê:
   - df_preco_de_ajuste_atual.parquet (P_PRECO)
   - df_valor_ajuste_contrato.parquet (P_VALOR)
   - df_valor_ajuste_DI.parquet (P_VALOR_DI)
2) Determina a data-base de início como a última data comum entre PREÇO e VALOR.
3) Calcula dias úteis B3 até o último dia útil antes de hoje.
4) Filtra para **apenas os dias que não existem** em P_VALOR_DI.
5) Faz scraping desses dias, **somente linhas de DI**, normaliza e grava.
6) Salva em parquet com índice "Assets" e colunas = datas ISO (YYYY-MM-DD).
"""

import os
import time
from datetime import date, timedelta
import pandas as pd
import pandas_market_calendars as mcal

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ───────── utilitários ─────────
def ler_parquet(path: str):
    if not os.path.exists(path):
        return pd.DataFrame(), []
    df = pd.read_parquet(path)
    # garante que a primeira coluna seja o índice "Assets"
    df = df.set_index(df.columns[0])
    df.index.name = "Assets"
    df = df[~df.index.duplicated(keep="last")]
    dates = sorted({pd.to_datetime(c).date() for c in df.columns})
    return df, dates

def salvar(df: pd.DataFrame, path: str):
    df.index.name = "Assets"
    df.reset_index().rename(
        columns={df.reset_index().columns[0]: "Assets"}
    ).to_parquet(path)

def normalizar_df_b3(df_raw):
    """
    Converte lista de linhas brutas (Mercadoria, Vencimento, Preço ant., Preço atual, Variação, Valor ajuste)
    no mesmo formato que você já usa, com chaves de ativo 'DI_XX'.
    """
    df = pd.DataFrame(df_raw, columns=[
        "Mercadoria", "Vencimento", "Preço de ajuste anterior",
        "Preço de ajuste Atual", "Variação", "Valor do ajuste por contrato (R$)"
    ])
    # monta o identificador e padroniza
    df["Mercadoria"] = df["Mercadoria"].str[:3] + "_" + df["Vencimento"].str[-2:]
    df.loc[df["Mercadoria"].str.startswith("DI1"), "Mercadoria"] = "DI_" + df["Vencimento"].str[-2:]
    df.drop("Vencimento", axis=1, inplace=True)

    # sinal do ajuste quando a variação é negativa
    neg = df["Variação"].astype(str).str.startswith("-")
    df.loc[neg, "Valor do ajuste por contrato (R$)"] = "-" + df.loc[neg, "Valor do ajuste por contrato (R$)"].astype(str)
    return df

def extrair_series(df_norm):
    """Retorna Series indexadas por 'Mercadoria' com preço atual e valor de ajuste."""
    s_preco = pd.Series(df_norm["Preço de ajuste Atual"].values,
                        index=df_norm["Mercadoria"]).groupby(level=0).first()
    s_valor = pd.Series(df_norm["Valor do ajuste por contrato (R$)"].values,
                        index=df_norm["Mercadoria"]).groupby(level=0).first()
    return s_preco, s_valor


# ───────── caminhos ─────────
P_PRECO    = "Dados/df_preco_de_ajuste_atual.parquet"
P_VALOR    = "Dados/df_valor_ajuste_contrato.parquet"   # geral (DAP, DI, etc.)
P_VALOR_DI = "Dados/df_valor_ajuste_DI.parquet"         # alvo (apenas DI)

# ───────── leitura bases ─────────
df_p, dates_p   = ler_parquet(P_PRECO)
df_v, dates_v   = ler_parquet(P_VALOR)
df_di, dates_di = ler_parquet(P_VALOR_DI)

if not dates_p or not dates_v:
    raise RuntimeError("Parquets base (preço/valor) não encontrados ou vazios. Rode o pipeline principal primeiro.")

# base para contagem de dias: última data comum já gravada entre PREÇO e VALOR
last_common_dt = min(dates_p[-1], dates_v[-1])

# calendário B3
cal = mcal.get_calendar("B3")
# evita instabilidade do 'hoje'
today = date.today() - timedelta(days=1)
ultimo_util = cal.valid_days(today - timedelta(days=10), today)[-1].date()

# todos os dias úteis do intervalo
dias_alvo = [d.date() for d in cal.valid_days(last_common_dt, ultimo_util)]

# já existentes no parquet DI
colunas_di_exist = {pd.to_datetime(c).date() for c in df_di.columns} if not df_di.empty else set()

# faltantes = dias_alvo que não estão no DI
dias_scrap = [d for d in dias_alvo if d not in colunas_di_exist]

if not dias_scrap:
    print("Nenhum dia faltante para DI. Nada a fazer.")
    # ainda assim salva para garantir padronização de índice/colunas
    salvar(df_di, P_VALOR_DI)
    raise SystemExit(0)

print("Dias faltantes (DI):", ", ".join(d.strftime("%Y-%m-%d") for d in dias_scrap))

# ───────── scraping apenas de DI ─────────
driver = webdriver.Chrome(service=Service())
try:
    driver.get("https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/derivativos/ajustes-do-pregao/")
    time.sleep(1)
    iframe = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
    driver.switch_to.frame(iframe)
    time.sleep(1)

    for d in dias_scrap:
        dia_fmt = d.strftime("%d/%m/%Y")

        input_box = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "dData1")))
        input_box.clear()
        input_box.send_keys(dia_fmt)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "button.button.expand"))
        ).click()
        time.sleep(2)

        table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "tblDadosAjustes")))
        driver.execute_script("arguments[0].scrollIntoView(true);", table)
        rows = table.find_elements(By.CSS_SELECTOR, "tr")

        data_list = [
            [c.text.strip() for c in r.find_elements(By.TAG_NAME, "td")]
            for r in rows if r.find_elements(By.TAG_NAME, "td")
        ]

        processed, mercadoria = [], None
        for ln in data_list:
            # linhas podem vir com 6 colunas (cabeçalho+conteúdo) ou 5 (conteúdo)
            if len(ln) == 6:
                mercadoria, venc, pa_ant, pa_atu, var, val_aj = ln
            else:
                venc, pa_ant, pa_atu, var, val_aj = ln

            # filtra APENAS DI
            if mercadoria != 'DI1 - DI de 1 dia':
                continue

            # mantém sua regra: apenas vencimentos com mês 'F'
            if 'F' not in venc:
                continue

            processed.append([mercadoria, venc, pa_ant, pa_atu, var, val_aj])

        if not processed:
            print(f"   • {d.isoformat()} sem linhas DI válidas (padrão atual).")
            continue

        df_norm = normalizar_df_b3(processed)
        _, s_valor = extrair_series(df_norm)  # só precisamos do Valor do ajuste por contrato (R$)

        # filtra novamente só "DI_"
        s_valor_di = s_valor[s_valor.index.str.startswith("DI_")]

        # garante índice e grava coluna nova
        df_di = df_di.reindex(df_di.index.union(s_valor_di.index))
        df_di[d.isoformat()] = s_valor_di
        print(f"   • {d.isoformat()} (DI) gravado")

        time.sleep(1.2)

finally:
    driver.quit()

# Ordena colunas por data
df_di = df_di.reindex(columns=sorted(df_di.columns, key=pd.to_datetime))

# Salva
salvar(df_di, P_VALOR_DI)
print(f"OK: '{P_VALOR_DI}' atualizado com {len(df_di)} ativos DI e {len(df_di.columns)} colunas (datas).")
