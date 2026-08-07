"""
atualizar_lft_selic.py  (v2 — SELIC Over + fallback CDI)
=========================================================

Atualiza `Dados/dados_lft.csv` completando os dias faltantes com SELIC Over do BCB.

Estrategia:
  - Le dados_lft.csv atual (historico LFT ate hoje)
  - Baixa SELIC Over diaria do BCB (serie 1178) — mesma taxa que LFT indexa
    Fallback: CDI Over (serie 12) + 0.10 bps de ajuste (SELIC ~= CDI + 0.10 bps a.a.)
  - Cap na data final = ontem (BCB so tem D-1)
  - Extrapola preco LFT: preco_t = preco_{t-1} * (1 + selic_dia)
  - Grava dados_lft.csv atualizado

USO:
    python atualizar_lft_selic.py                    # incremental
    python atualizar_lft_selic.py --dias 30          # forca refresh dos ultimos 30 dias
    python atualizar_lft_selic.py --dry-run          # nao grava
"""
from __future__ import annotations
import argparse
import time
from datetime import date, timedelta, datetime
from pathlib import Path
import pandas as pd
import requests

DADOS_LFT_CSV = Path("Dados/dados_lft.csv")

# BCB Series:
#   1178 = SELIC Over diaria (%)     — o que a LFT realmente indexa
#     12 = CDI Over diaria (%)       — proxy (SELIC ~= CDI + ~0.10 bps/ano)
#   4189 = SELIC Meta anualizada (%) — nao serve pra composicao diaria direta
SERIE_SELIC = 1178
SERIE_CDI   = 12
AJUSTE_CDI_PARA_SELIC = 0.10 / 100 / 252    # +0.10% a.a. em decimal diario

BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs/{serie}/dados?formato=json&dataInicial={ini}&dataFinal={fim}"
HEADERS = {"User-Agent": "AFinvest-Risco/1.0 (Python)"}


def _get_bcb(serie: int, data_ini: date, data_fim: date) -> pd.DataFrame:
    """Baixa serie do BCB SGS. Retorna DataFrame com Data (datetime) e valor (float, decimal)."""
    url = BCB_URL.format(serie=serie,
                          ini=data_ini.strftime("%d/%m/%Y"),
                          fim=data_fim.strftime("%d/%m/%Y"))
    r = requests.get(url, timeout=20, headers=HEADERS)
    if r.status_code == 404:
        print(f"  [bcb] serie {serie}: 404 (nao ha dados no range)")
        return pd.DataFrame(columns=["Data", "valor"])
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame(columns=["Data", "valor"])
    df = pd.DataFrame(data)
    df["Data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce") / 100.0  # % -> decimal
    return df[["Data", "valor"]].dropna()


def _buscar_taxa_diaria(data_ini: date, data_fim: date):
    """Tenta SELIC Over (1178). Fallback CDI Over (12) com ajuste +0.10bps a.a. diario.
    Retorna (df, fonte_usada) ou (empty, None) se nada funcionar.
    """
    # 1. Tenta SELIC Over
    try:
        print(f"[bcb] tentando SELIC Over (serie {SERIE_SELIC}) de {data_ini} a {data_fim}...")
        df = _get_bcb(SERIE_SELIC, data_ini, data_fim)
        if not df.empty:
            print(f"[bcb] SELIC Over OK — {len(df)} pontos, ultima taxa={df['valor'].iloc[-1]*100:.4f}%/dia")
            return df, "SELIC Over (BCB 1178)"
    except Exception as e:
        print(f"[bcb] SELIC Over erro: {e}")

    # 2. Fallback CDI + ajuste
    try:
        print(f"[bcb] fallback CDI Over (serie {SERIE_CDI}) + ajuste +0.10bps a.a....")
        df = _get_bcb(SERIE_CDI, data_ini, data_fim)
        if not df.empty:
            df["valor"] = df["valor"] + AJUSTE_CDI_PARA_SELIC
            print(f"[bcb] CDI+ajuste OK — {len(df)} pontos, ultima taxa={df['valor'].iloc[-1]*100:.4f}%/dia")
            return df, "CDI Over (BCB 12) + ajuste 0.10bps"
    except Exception as e:
        print(f"[bcb] CDI erro: {e}")

    return pd.DataFrame(columns=["Data", "valor"]), None


def _carregar_lft_atual() -> pd.DataFrame:
    if not DADOS_LFT_CSV.exists():
        print(f"[lft] {DADOS_LFT_CSV} nao existe — criando do zero")
        return pd.DataFrame(columns=["Data", "RetornoLFT"])
    df = pd.read_csv(DADOS_LFT_CSV, parse_dates=["Data"])
    df["RetornoLFT"] = pd.to_numeric(df["RetornoLFT"], errors="coerce")
    df = df.dropna(subset=["Data", "RetornoLFT"]).sort_values("Data").reset_index(drop=True)
    df = df[df["RetornoLFT"] > 0]
    return df


def run(dry_run: bool = False, dias_refresh: int | None = None):
    print("=" * 78)
    print("atualizar_lft_selic — completa dados_lft.csv com SELIC do BCB")
    print("=" * 78)

    df_lft = _carregar_lft_atual()
    if df_lft.empty:
        print("[lft] arquivo vazio — nao ha historico para continuar.")
        return

    ultima_data = df_lft["Data"].max().date()
    ultimo_preco = float(df_lft.iloc[-1]["RetornoLFT"])
    print(f"[lft] historico atual: {len(df_lft)} pontos, ultima data = {ultima_data}, preco={ultimo_preco:.4f}")

    if dias_refresh:
        corte = ultima_data - timedelta(days=dias_refresh)
        n_antes = len(df_lft)
        df_lft = df_lft[df_lft["Data"].dt.date < corte]
        print(f"[lft] refresh forcado: removeu {n_antes - len(df_lft)} dias apos {corte}")
        if df_lft.empty:
            print("[lft] dataframe vazio apos refresh, aborta.")
            return
        ultima_data = df_lft["Data"].max().date()
        ultimo_preco = float(df_lft.iloc[-1]["RetornoLFT"])

    # Range de datas a preencher
    hoje = date.today()
    ontem = hoje - timedelta(days=1)   # BCB tem ate D-1

    ini = ultima_data + timedelta(days=1)
    fim = ontem

    if ini > fim:
        print(f"[lft] historico ja esta em dia (ultima={ultima_data}, cap ontem={ontem}).")
        return

    print(f"[lft] periodo a preencher: {ini} → {fim} ({(fim-ini).days + 1} dias)")

    # Sempre busca range maior (60 dias antes de ini) — evita 404 quando range curto so tem
    # fim de semana ou dias sem publicacao. Filtramos localmente depois.
    query_ini = ini - timedelta(days=60)
    query_fim = fim
    print(f"[bcb] query BCB range ampliado: {query_ini} → {query_fim} (filtrado localmente depois)")

    df_tx, fonte = _buscar_taxa_diaria(query_ini, query_fim)
    if df_tx.empty:
        print(f"[lft] BCB nao retornou nada — verifica conectividade ou se API mudou.")
        # Diagnostico
        print(f"[diag] tentando query pequena (2025-01-15 a 2025-01-20) pra ver se API funciona...")
        try:
            df_test = _get_bcb(SERIE_CDI, date(2025, 1, 15), date(2025, 1, 20))
            if df_test.empty:
                print(f"[diag] API tambem retorna vazio para range antigo — pode ser bloqueio de rede.")
            else:
                print(f"[diag] API funciona ({len(df_test)} pontos historicos). Range {query_ini}-{query_fim} realmente nao tem dado.")
        except Exception as e:
            print(f"[diag] API teste falhou: {e}")
        return

    # Filtra: mantem apenas dias posteriores a ultima data do LFT
    df_tx = df_tx[df_tx["Data"].dt.date > ultima_data].copy()
    if df_tx.empty:
        print(f"[lft] apos filtro, nada novo.")
        return

    # Extrapola preco
    novos = []
    preco = ultimo_preco
    for _, row in df_tx.iterrows():
        preco = preco * (1 + float(row["valor"]))
        novos.append({"Data": row["Data"], "RetornoLFT": preco})
    df_novos = pd.DataFrame(novos)

    print(f"[lft] fonte usada: {fonte}")
    print(f"[lft] extrapolando {len(df_novos)} novos dias:")
    for i, row in df_novos.iterrows():
        if i < 5 or i > len(df_novos) - 3:
            print(f"        {row['Data'].date()}  preco={row['RetornoLFT']:.4f}  (taxa_dia={df_tx.iloc[i]['valor']*100:.4f}%)")

    df_final = pd.concat([df_lft, df_novos], ignore_index=True)
    df_final = df_final.sort_values("Data").drop_duplicates(subset=["Data"], keep="last")

    if dry_run:
        print(f"[lft] dry-run: {len(df_final)} pontos totais (novos: {len(df_novos)}). NAO GRAVOU.")
        return

    backup = DADOS_LFT_CSV.with_suffix(".csv.bak")
    if DADOS_LFT_CSV.exists():
        import shutil
        shutil.copy2(DADOS_LFT_CSV, backup)
        print(f"[lft] backup: {backup}")
    df_final.to_csv(DADOS_LFT_CSV, index=False)
    print(f"[lft] OK — {len(df_final)} pontos em {DADOS_LFT_CSV}")
    print(f"[lft] fonte: {fonte}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dias", type=int, default=None,
                    help="Forca reprocessar os ultimos N dias")
    args = ap.parse_args()
    run(dry_run=args.dry_run, dias_refresh=args.dias)
