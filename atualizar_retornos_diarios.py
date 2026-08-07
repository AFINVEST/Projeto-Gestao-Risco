"""
atualizar_retornos_diarios.py  (v2 — multi-fonte)
==================================================

Popula/atualiza a tabela `retornos_diarios_ativo` do Supabase.

Fontes suportadas:
  --fonte df_preco    (padrao) Dados/df_preco_de_ajuste_atual_completo.parquet
                      Formato pt-BR, atualizado diariamente pelo ScrapB3_v2.
                      Cobertura tipica: 2025-01 -> hoje.

  --fonte df_inicial  Dados/df_inicial.parquet (BBG rates/PUs, historico longo).
                      Cobertura tipica: 2021-05 -> ultima atualizacao BBG.
                      USE PARA BACKFILL DO PERIODO ANTIGO (2021-2024).

Metodo (identico ao dashboard — `process_returns2` em app4.py):
    retorno_t = ln(preco_t / preco_{t-1})    # log return

Modos:
    --bootstrap                 Processa TODOS os dias da fonte selecionada.
    (default)                   Incremental: so insere dias novos.

USO:
    # Diario (fonte B3, incremental) — dentro do .bat
    python atualizar_retornos_diarios.py

    # Bootstrap inicial (fonte B3)
    python atualizar_retornos_diarios.py --bootstrap

    # Backfill historico com BBG (2021-2024) — rodar 1 vez
    python atualizar_retornos_diarios.py --bootstrap --fonte df_inicial

VARIAVEIS DE AMBIENTE:
    SUPABASE_URL, SUPABASE_KEY (ou SUPABASE_SERVICE_ROLE_KEY)
"""
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from supabase import create_client
except ImportError:
    print("ERRO: pip install supabase", file=sys.stderr)
    sys.exit(1)


TABELA = "retornos_diarios_ativo"
BATCH_SIZE = 500


def _get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Defina SUPABASE_URL e SUPABASE_KEY.")
    return create_client(url, key)


def _parse_ptbr(v):
    """Converte 'R$ 83.245,19' ou '83.245,19' -> 83245.19; NaN -> None."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    s = str(v).strip().replace("R$", "").replace(" ", "")
    if not s or s in ("-", "--"):
        return None
    s2 = s.replace(".", "").replace(",", ".")
    try:
        return float(s2)
    except Exception:
        return None


def _detecta_moeda(ativo: str) -> str:
    return "USD" if str(ativo).upper() == "TREASURY" else "BRL"


def _ultima_data_por_ativo(client) -> dict:
    """Retorna {ativo: ultima_data_ISO} do Supabase — usado para incremental."""
    try:
        # Query eficiente: MAX(Data) por ativo
        resp = client.rpc("get_max_data_por_ativo").execute()
        if resp.data:
            return {r["Ativo"]: r["max_data"] for r in resp.data}
    except Exception:
        pass
    # Fallback: SELECT paginado
    try:
        resp = client.table(TABELA).select("Ativo,Data").order("Data", desc=True).limit(50000).execute()
        out = {}
        for r in resp.data or []:
            a = r["Ativo"]
            d = r["Data"]
            if a not in out or d > out[a]:
                out[a] = d
        return out
    except Exception as e:
        print(f"[retornos] fallback query erro: {e}")
        return {}


def _ler_precos_wide_df_preco(parquet_path: str = "Dados/df_preco_de_ajuste_atual_completo.parquet") -> pd.DataFrame:
    """Le o df_preco (B3, formato wide 'Assets' + colunas-data em string).
    Retorna long (Data, Ativo, preco)."""
    df = pd.read_parquet(parquet_path)
    if "Assets" not in df.columns:
        raise ValueError(f"'Assets' nao encontrado em {parquet_path}")

    cols_data = [c for c in df.columns if c != "Assets"]
    df_long = df.melt(id_vars="Assets", value_vars=cols_data,
                       var_name="Data", value_name="preco_raw")
    df_long = df_long.rename(columns={"Assets": "Ativo"})
    df_long["preco"] = df_long["preco_raw"].map(_parse_ptbr)
    df_long["Data"] = pd.to_datetime(df_long["Data"], errors="coerce")
    df_long = df_long.dropna(subset=["Data", "preco"])
    df_long = df_long[df_long["preco"] > 0].copy()
    df_long = df_long[["Data", "Ativo", "preco"]].sort_values(["Ativo", "Data"])
    n_antes = len(df_long)
    df_long = df_long.drop_duplicates(subset=["Data", "Ativo"], keep="first")
    n_dup = n_antes - len(df_long)
    if n_dup > 0:
        print(f"[retornos] {n_dup} pares duplicados removidos")
    return df_long


def _ler_precos_wide_df_inicial(parquet_path: str = "Dados/df_inicial.parquet") -> pd.DataFrame:
    """Le o df_inicial (BBG, formato wide 'Date' + colunas por ativo).
    Retorna long (Data, Ativo, preco)."""
    df = pd.read_parquet(parquet_path)
    if "Date" not in df.columns:
        raise ValueError(f"'Date' nao encontrado em {parquet_path}")

    cols_ativos = [c for c in df.columns if c != "Date"]
    df_long = df.melt(id_vars="Date", value_vars=cols_ativos,
                       var_name="Ativo", value_name="preco")
    df_long = df_long.rename(columns={"Date": "Data"})
    df_long["Data"] = pd.to_datetime(df_long["Data"], errors="coerce")
    df_long["preco"] = pd.to_numeric(df_long["preco"], errors="coerce")
    df_long = df_long.dropna(subset=["Data", "preco"])
    df_long = df_long[df_long["preco"] > 0].copy()
    df_long = df_long[["Data", "Ativo", "preco"]].sort_values(["Ativo", "Data"])
    df_long = df_long.drop_duplicates(subset=["Data", "Ativo"], keep="first")
    return df_long


def _ler_precos_wide(fonte: str = "df_preco"):
    """Dispatcher: le a fonte apropriada."""
    if fonte == "df_preco":
        return _ler_precos_wide_df_preco()
    elif fonte == "df_inicial":
        return _ler_precos_wide_df_inicial()
    else:
        raise ValueError(f"Fonte desconhecida: {fonte}. Use 'df_preco' ou 'df_inicial'.")


def _calcular_log_returns(df_long: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna 'retorno' (log return) a partir da série de preços por ativo."""
    df = df_long.copy()
    df["retorno"] = df.groupby("Ativo")["preco"].transform(lambda s: np.log(s / s.shift(1)))
    return df


def _upsert_batch(client, registros):
    if not registros:
        return 0
    total = 0
    for i in range(0, len(registros), BATCH_SIZE):
        lote = registros[i:i + BATCH_SIZE]
        client.table(TABELA).upsert(lote, on_conflict="Data,Ativo").execute()
        total += len(lote)
        print(f"  [upsert] {total}/{len(registros)}")
    return total


def run(bootstrap: bool = False, fonte: str = "df_preco"):
    client = _get_client()

    print(f"[retornos] fonte: {fonte}")
    df_long = _ler_precos_wide(fonte)
    print(f"[retornos] {len(df_long)} pares (Data, Ativo) com preço válido")
    print(f"[retornos] {df_long['Ativo'].nunique()} ativos distintos: {sorted(df_long['Ativo'].unique())}")

    df_long = _calcular_log_returns(df_long)
    print(f"[retornos] {df_long['retorno'].notna().sum()} retornos calculáveis (excluindo primeiro dia de cada ativo)")

    if not bootstrap:
        ultimas = _ultima_data_por_ativo(client)
        print(f"[retornos] {len(ultimas)} ativos ja tem retornos no Supabase")
        if ultimas:
            def _incrementar(row):
                ult = ultimas.get(row["Ativo"])
                if ult is None:
                    return True  # ativo novo, inclui tudo
                d = pd.Timestamp(ult).date()
                return row["Data"].date() > d
            mask = df_long.apply(_incrementar, axis=1)
            df_long = df_long[mask].copy()
            print(f"[retornos] incremental: {len(df_long)} novos pares a enviar")

    if df_long.empty:
        print("[retornos] nada novo a enviar.")
        return

    # Prepara registros
    registros = []
    for _, row in df_long.iterrows():
        retorno = row["retorno"]
        if pd.isna(retorno):
            retorno = None
        else:
            retorno = float(retorno)
        registros.append({
            "Data":    row["Data"].date().isoformat(),
            "Ativo":   str(row["Ativo"]),
            "preco":   float(row["preco"]),
            "retorno": retorno,
            "moeda":   _detecta_moeda(row["Ativo"]),
            "fonte":   fonte,
        })

    print(f"[retornos] enviando {len(registros)} registros ao Supabase (batches de {BATCH_SIZE})...")
    n = _upsert_batch(client, registros)
    print(f"[retornos] OK — {n} linhas em {TABELA}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true", help="Processa todo o historico")
    ap.add_argument("--fonte", choices=["df_preco", "df_inicial"], default="df_preco",
                    help="Fonte dos dados: df_preco (B3) ou df_inicial (BBG historico longo)")
    args = ap.parse_args()
    run(bootstrap=args.bootstrap, fonte=args.fonte)
