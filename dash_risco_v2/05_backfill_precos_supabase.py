"""
05_backfill_precos_supabase.py
================================

Popula a tabela precos_diarios no Supabase a partir dos parquets
locais que já existem no projeto:

    Fonte principal: Dados/df_inicial.parquet
        - formato wide, colunas = ativos DI/DAP, valores = taxas em %
        - histórico longo do BBG
        - marcado como fonte='bbg'

    Fonte secundária: Dados/df_preco_de_ajuste_atual_completo.parquet
        - formato longo com 'Assets' + colunas de data
        - valores = PU_ajuste (preço)
        - histórico do scraping B3
        - marcado como fonte='b3'

Estratégia: BBG primeiro (dá histórico longo), depois B3 sobrescreve
onde tiver (dado mais recente). Upsert idempotente.

USO:
    export SUPABASE_URL='https://...supabase.co'
    export SUPABASE_KEY='eyJ...service_role_key'
    python 05_backfill_precos_supabase.py

Requisitos: pip install supabase pandas pyarrow
"""
from __future__ import annotations
import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from supabase import create_client
except ImportError:
    print("ERRO: pip install supabase", file=sys.stderr)
    sys.exit(1)

# Importa nosso módulo para derivar PU dos rates BBG
sys.path.insert(0, str(Path(__file__).parent))
import dv01_dinamico as dv


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERRO: defina SUPABASE_URL e SUPABASE_KEY (ou SUPABASE_SERVICE_ROLE_KEY)", file=sys.stderr)
    sys.exit(1)


DADOS = Path("Dados")
TABLE = "precos_diarios"
BATCH = 1000


def _safe(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def _upsert_lote(sb, registros: list[dict]) -> int:
    total = 0
    for i in range(0, len(registros), BATCH):
        lote = registros[i:i + BATCH]
        (sb.table(TABLE)
           .upsert(lote, on_conflict="Data,Ativo")
           .execute())
        total += len(lote)
        print(f"    lote {i // BATCH + 1}: {len(lote)} linhas (acum: {total})")
    return total


def backfill_bbg(sb, feriados) -> int:
    """Lê df_inicial.parquet e envia para precos_diarios com fonte='bbg'.
    df_inicial tem taxas em %. Calcula PU derivado usando o vencimento."""
    caminho = DADOS / "df_inicial.parquet"
    if not caminho.exists():
        print(f"[skip] {caminho} não encontrado")
        return 0

    print(f"\n[BBG] Lendo {caminho}...")
    df = pd.read_parquet(caminho)

    # Detecta se a coluna de data é o index ou uma coluna
    if not isinstance(df.index, pd.DatetimeIndex):
        # tenta encontrar coluna de data
        for candidato in ('Data', 'data', 'Datas', 'Date'):
            if candidato in df.columns:
                df = df.set_index(candidato)
                df.index = pd.to_datetime(df.index)
                break
        else:
            # assume que primeira coluna é data
            df = df.set_index(df.columns[0])
            df.index = pd.to_datetime(df.index)

    ativos = [c for c in df.columns if dv._RE_DI.match(str(c)) or dv._RE_DAP.match(str(c))]
    print(f"[BBG] {len(ativos)} ativos DI/DAP detectados: {ativos[:5]}...{ativos[-3:] if len(ativos)>5 else ''}")

    registros = []
    puladas = 0
    for ticker in ativos:
        try:
            venc = dv.vencimento(ticker, feriados)
        except Exception as e:
            print(f"    aviso: {ticker} sem vencimento definido, pulando ({e})")
            continue
        serie = df[ticker].dropna()
        for data, taxa in serie.items():
            data_norm = pd.Timestamp(data).normalize()
            if data_norm > venc:
                puladas += 1
                continue
            taxa_v = _safe(taxa)
            if taxa_v is None:
                continue
            du = dv.networkdays(data_norm, venc, feriados)
            if du <= 0:
                continue
            pu = dv.pu_di(taxa_v, du)  # mesma fórmula p/ DI e DAP em pontos
            registros.append({
                "Data":      data_norm.date().isoformat(),
                "Ativo":     ticker,
                "PU_ajuste": pu,
                "Taxa":      taxa_v,
                "Fonte":     "bbg",
            })

    print(f"[BBG] {len(registros)} linhas para enviar ({puladas} puladas por data>venc)")
    return _upsert_lote(sb, registros) if registros else 0


def backfill_b3(sb, feriados) -> int:
    """Lê df_preco_de_ajuste_atual_completo.parquet (PU) e envia com fonte='b3'.
    Deriva a Taxa a partir do PU + DU + nominal 100k."""
    caminho = DADOS / "df_preco_de_ajuste_atual_completo.parquet"
    if not caminho.exists():
        print(f"[skip] {caminho} não encontrado")
        return 0

    print(f"\n[B3] Lendo {caminho}...")
    df = pd.read_parquet(caminho)

    # Espera coluna 'Assets' + colunas com nomes de data
    if 'Assets' not in df.columns:
        print(f"[B3] coluna 'Assets' não encontrada. Colunas: {list(df.columns)[:5]}...")
        return 0

    df_long = df.melt(id_vars=['Assets'], var_name='Data', value_name='PU')
    df_long['Data'] = pd.to_datetime(df_long['Data'], errors='coerce')
    df_long = df_long.dropna(subset=['Data', 'PU'])

    # Filtra só DI/DAP (ignora WDO1, TREASURY que não seguem PU-taxa)
    mask_di_dap = df_long['Assets'].str.match(r'^(DI|DAP)_')
    df_di_dap = df_long[mask_di_dap].copy()
    df_outros = df_long[~mask_di_dap].copy()

    print(f"[B3] {len(df_di_dap)} linhas DI/DAP + {len(df_outros)} linhas outros")

    registros = []
    for _, row in df_di_dap.iterrows():
        ticker = row['Assets']
        try:
            venc = dv.vencimento(ticker, feriados)
        except Exception:
            continue
        data = pd.Timestamp(row['Data']).normalize()
        if data > venc:
            continue
        pu = _safe(row['PU'])
        if pu is None or pu <= 0:
            continue
        du = dv.networkdays(data, venc, feriados)
        if du <= 0:
            continue
        # Deriva taxa a partir do PU: taxa = ((nominal/PU)^(252/DU) - 1) * 100
        try:
            taxa = ((dv.NOMINAL / pu) ** (dv.DIAS_UTEIS_ANO / du) - 1) * 100
        except Exception:
            taxa = None
        registros.append({
            "Data":      data.date().isoformat(),
            "Ativo":     ticker,
            "PU_ajuste": pu,
            "Taxa":      taxa,
            "Fonte":     "b3",
        })

    # Adiciona outros ativos (WDO1, TREASURY) sem taxa
    for _, row in df_outros.iterrows():
        pu = _safe(row['PU'])
        if pu is None:
            continue
        registros.append({
            "Data":      pd.Timestamp(row['Data']).normalize().date().isoformat(),
            "Ativo":     row['Assets'],
            "PU_ajuste": pu,
            "Taxa":      None,
            "Fonte":     "b3",
        })

    print(f"[B3] {len(registros)} linhas para enviar")
    return _upsert_lote(sb, registros) if registros else 0


def main():
    if not DADOS.exists():
        print(f"ERRO: pasta Dados/ não encontrada. Rode este script na raiz do projeto.", file=sys.stderr)
        sys.exit(1)

    print(f"Conectando no Supabase...")
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    print(f"Carregando feriados_anbima.parquet...")
    feriados = dv.load_feriados(str(DADOS / "feriados_anbima.parquet"))
    print(f"  {len(feriados)} feriados carregados")

    # 1) BBG primeiro (histórico longo, fonte='bbg')
    n_bbg = backfill_bbg(sb, feriados)
    print(f"[BBG] Concluído: {n_bbg} linhas enviadas")

    # 2) B3 depois (sobrescreve nas datas onde tiver, fonte='b3')
    n_b3 = backfill_b3(sb, feriados)
    print(f"[B3]  Concluído: {n_b3} linhas enviadas")

    print(f"\n{'='*60}")
    print(f"Total enviado: {n_bbg + n_b3} linhas (bbg={n_bbg}, b3={n_b3})")
    print(f"{'='*60}")
    print(f"\nVerificações no SQL Editor:")
    print(f'  select count(*), "Fonte" from precos_diarios group by "Fonte";')
    print(f'  select "Ativo", count(*), min("Data"), max("Data")')
    print(f'  from precos_diarios where "Ativo"=\'DI_F29\' group by "Ativo";')


if __name__ == "__main__":
    main()
