"""
atualizar_cdi_lft.py  (unificado)
==================================

Baixa CDI diario do BCB (serie 12, endpoint novo com ponto: bcdata.sgs.12)
e atualiza:

  1. Dados/cdi_cached.csv  — CDI diario (usado por load_cdi_series no snapshot,
                              email, dashboard)
  2. Dados/dados_lft.csv   — LFT proxy = CDI + 0.10% a.a. (usado por load_lft_series)

Estrategia:
  - Le cdi_cached.csv atual e busca CDI do BCB a partir de ultima_data+1 ate ontem
  - Adiciona novos pontos ao CSV
  - Para dados_lft.csv: usa ultimo preco conhecido e extrapola com (1 + cdi_dia + ajuste)
    onde ajuste = 0.10% a.a. em base diaria

USO:
    python atualizar_cdi_lft.py                    # incremental
    python atualizar_cdi_lft.py --dias 30          # forca reprocessar ultimos N dias
    python atualizar_cdi_lft.py --dry-run
"""
from __future__ import annotations
import argparse
import shutil
import sys
import time
from datetime import date, timedelta, datetime
from pathlib import Path
import pandas as pd
import requests

CDI_CACHED_CSV = Path("Dados/cdi_cached.csv")
DADOS_LFT_CSV  = Path("Dados/dados_lft.csv")

# BCB endpoint novo (2026): bcdata.sgs.{serie} com ponto
BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json&dataInicial={ini}&dataFinal={fim}"
SERIE_CDI = 12       # CDI Over diaria (%)
AJUSTE_CDI_PARA_SELIC = 0.10 / 100 / 252   # +0.10% a.a. em decimal diario (SELIC ~= CDI + 0.10bps)
HEADERS = {"User-Agent": "AFinvest-Risco/1.0 (Python)"}


def _get_bcb(serie: int, data_ini: date, data_fim: date, tentativas: int = 2) -> pd.DataFrame:
    """Baixa serie SGS do BCB. Retorna DataFrame com Data (datetime) e valor (float, decimal)."""
    url = BCB_URL.format(serie=serie,
                          ini=data_ini.strftime("%d/%m/%Y"),
                          fim=data_fim.strftime("%d/%m/%Y"))
    for t in range(1, tentativas + 1):
        try:
            r = requests.get(url, timeout=20, headers=HEADERS)
            # BCB retorna payload de erro com HTTP 200 quando endpoint inexistente
            if r.text.strip().startswith('{"error"'):
                print(f"  [bcb] serie {serie}: erro payload ({r.text[:100]})")
                return pd.DataFrame(columns=["Data", "valor"])
            if r.status_code == 404:
                print(f"  [bcb] serie {serie}: 404 (sem dados no range)")
                return pd.DataFrame(columns=["Data", "valor"])
            if 500 <= r.status_code < 600:
                print(f"  [bcb] serie {serie} HTTP {r.status_code} tentativa {t}/{tentativas}")
                time.sleep(2)
                continue
            r.raise_for_status()
            data = r.json()
            if not data:
                return pd.DataFrame(columns=["Data", "valor"])
            df = pd.DataFrame(data)
            df["Data"] = pd.to_datetime(df["data"], dayfirst=True)
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce") / 100.0
            return df[["Data", "valor"]].dropna()
        except requests.exceptions.RequestException as e:
            print(f"  [bcb] serie {serie} tentativa {t}: {e}")
            if t < tentativas:
                time.sleep(2)
    raise RuntimeError(f"BCB serie {serie} falhou apos {tentativas} tentativas")


def _carregar_cdi_atual() -> pd.DataFrame:
    if not CDI_CACHED_CSV.exists():
        print(f"[cdi] {CDI_CACHED_CSV} nao existe — criando do zero")
        return pd.DataFrame(columns=["Data", "cdi"])
    df = pd.read_csv(CDI_CACHED_CSV, parse_dates=["Data"])
    df["cdi"] = pd.to_numeric(df["cdi"], errors="coerce")
    df = df.dropna(subset=["Data", "cdi"]).sort_values("Data").reset_index(drop=True)
    return df


def _carregar_lft_atual() -> pd.DataFrame:
    if not DADOS_LFT_CSV.exists():
        print(f"[lft] {DADOS_LFT_CSV} nao existe — criando do zero")
        return pd.DataFrame(columns=["Data", "RetornoLFT"])
    df = pd.read_csv(DADOS_LFT_CSV, parse_dates=["Data"])
    df["RetornoLFT"] = pd.to_numeric(df["RetornoLFT"], errors="coerce")
    df = df.dropna(subset=["Data", "RetornoLFT"]).sort_values("Data").reset_index(drop=True)
    return df[df["RetornoLFT"] > 0]


def run(dry_run: bool = False, dias_refresh: int | None = None):
    print("=" * 78)
    print("atualizar_cdi_lft — atualiza CDI + LFT (proxy SELIC) via BCB")
    print("=" * 78)

    df_cdi_local = _carregar_cdi_atual()
    df_lft_local = _carregar_lft_atual()

    ultima_data_cdi = df_cdi_local["Data"].max().date() if not df_cdi_local.empty else date(2020, 1, 1)
    ultima_data_lft = df_lft_local["Data"].max().date() if not df_lft_local.empty else None

    print(f"[cdi] cache atual: {len(df_cdi_local)} pontos, ultima data = {ultima_data_cdi}")
    print(f"[lft] cache atual: {len(df_lft_local)} pontos, ultima data = {ultima_data_lft}")

    if dias_refresh:
        corte = ultima_data_cdi - timedelta(days=dias_refresh)
        df_cdi_local = df_cdi_local[df_cdi_local["Data"].dt.date < corte]
        print(f"[cdi] refresh: removidos dias apos {corte}")
        ultima_data_cdi = df_cdi_local["Data"].max().date() if not df_cdi_local.empty else date(2020, 1, 1)

    hoje = date.today()
    ontem = hoje - timedelta(days=1)
    ini = ultima_data_cdi + timedelta(days=1)
    if ini > ontem:
        print(f"[cdi] ja em dia (ultima={ultima_data_cdi}, cap ontem={ontem}).")
        return

    # Range ampliado (60 dias antes) pra evitar 404 quando range curto so tem fim de semana
    query_ini = ini - timedelta(days=60)
    query_fim = ontem
    print(f"[bcb] baixando CDI serie {SERIE_CDI} de {query_ini} a {query_fim}...")

    try:
        df_cdi_novo = _get_bcb(SERIE_CDI, query_ini, query_fim)
    except Exception as e:
        print(f"[bcb] FALHA TOTAL: {e}", file=sys.stderr)
        print(f"[ERRO] CDI nao pode ser atualizado. Pipeline deve abortar.", file=sys.stderr)
        sys.exit(1)   # exit code 1 → .bat aborta

    if df_cdi_novo.empty:
        # Se cache ja estava em dia (D-1) ok. Se stale + range vazio: falha.
        gap_dias = (ontem - ultima_data_cdi).days
        if gap_dias > 3:   # gap > 3 dias uteis é suspeito
            print(f"[ERRO] CDI stale ({gap_dias} dias sem atualizar) e BCB retornou vazio.", file=sys.stderr)
            sys.exit(1)
        print(f"[bcb] retornou vazio, mas cache esta em dia (gap {gap_dias} dias).")
        return

    # Filtra: mantem apenas datas > ultima_data_cdi
    df_cdi_novo = df_cdi_novo[df_cdi_novo["Data"].dt.date > ultima_data_cdi].copy()
    if df_cdi_novo.empty:
        print(f"[cdi] apos filtro, nada de novo.")
        return

    print(f"[cdi] {len(df_cdi_novo)} novos pontos de CDI:")
    for i, row in df_cdi_novo.iterrows():
        if i < 3 or i > len(df_cdi_novo) - 3:
            print(f"      {row['Data'].date()}  cdi={row['valor']*100:.4f}%/dia")

    # ─── Atualiza CDI CSV ─────────────────────────
    df_cdi_novo_out = df_cdi_novo.rename(columns={"valor": "cdi"})[["Data", "cdi"]]
    df_cdi_final = pd.concat([df_cdi_local, df_cdi_novo_out], ignore_index=True)
    df_cdi_final = df_cdi_final.sort_values("Data").drop_duplicates(subset=["Data"], keep="last")

    # ─── Atualiza LFT CSV ─────────────────────────
    # Extrapola preco: preco_t = preco_{t-1} * (1 + cdi_dia + ajuste)
    df_lft_novo = pd.DataFrame()
    if not df_lft_local.empty and ultima_data_lft is not None:
        # Pega dias novos alinhados com CDI (apenas os apos ultima_data_lft)
        df_para_lft = df_cdi_novo[df_cdi_novo["Data"].dt.date > ultima_data_lft].copy()
        if not df_para_lft.empty:
            ultimo_preco = float(df_lft_local.iloc[-1]["RetornoLFT"])
            novos_lft = []
            preco = ultimo_preco
            for _, row in df_para_lft.iterrows():
                taxa_dia = float(row["valor"]) + AJUSTE_CDI_PARA_SELIC
                preco = preco * (1 + taxa_dia)
                novos_lft.append({"Data": row["Data"], "RetornoLFT": preco})
            df_lft_novo = pd.DataFrame(novos_lft)
            print(f"[lft] extrapolando {len(df_lft_novo)} dias com CDI+ajuste ({AJUSTE_CDI_PARA_SELIC*252*100:.2f}% a.a.):")
            for i, row in df_lft_novo.iterrows():
                if i < 3 or i > len(df_lft_novo) - 3:
                    print(f"      {row['Data'].date()}  preco={row['RetornoLFT']:.4f}")

    df_lft_final = df_lft_local.copy()
    if not df_lft_novo.empty:
        df_lft_final = pd.concat([df_lft_local, df_lft_novo], ignore_index=True)
        df_lft_final = df_lft_final.sort_values("Data").drop_duplicates(subset=["Data"], keep="last")

    if dry_run:
        print(f"[dry-run] CDI: {len(df_cdi_local)} → {len(df_cdi_final)} pontos")
        print(f"[dry-run] LFT: {len(df_lft_local)} → {len(df_lft_final)} pontos")
        print(f"[dry-run] Nada foi gravado.")
        return

    # Grava
    if CDI_CACHED_CSV.exists():
        shutil.copy2(CDI_CACHED_CSV, CDI_CACHED_CSV.with_suffix(".csv.bak"))
    df_cdi_final.to_csv(CDI_CACHED_CSV, index=False)
    print(f"[cdi] OK — {len(df_cdi_final)} pontos em {CDI_CACHED_CSV}")

    if not df_lft_novo.empty:
        if DADOS_LFT_CSV.exists():
            shutil.copy2(DADOS_LFT_CSV, DADOS_LFT_CSV.with_suffix(".csv.bak"))
        df_lft_final.to_csv(DADOS_LFT_CSV, index=False)
        print(f"[lft] OK — {len(df_lft_final)} pontos em {DADOS_LFT_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dias", type=int, default=None,
                    help="Forca reprocessar os ultimos N dias")
    args = ap.parse_args()
    run(dry_run=args.dry_run, dias_refresh=args.dias)
