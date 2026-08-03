"""
taxas_dinamicas.py  (v2.5)
==========================

MUDANÇA CRÍTICA v2.5: para DI/DAP, usa APENAS o parquet local
(Dados/df_preco_de_ajuste_atual_completo.parquet), que contém PUs
corretos vindos do B3. Ignora o Supabase precos_diarios até refazermos
o backfill corretamente (o backfill BBG anterior corrompeu os PUs
tratando-os como taxas).

  DI:  valor = PU (do parquet local)
  DAP: valor = PU × rs_dap  (rs_dap dinâmico por data)
  WDO1, TREASURY, NTNB*: valor direto do parquet local

Se você quiser recuperar o histórico BBG longo depois, temos que
refazer o backfill do BBG interpretando os valores como PU (não taxa)
e recalculando a taxa corretamente.
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

try:
    from supabase import create_client
except ImportError:
    print("ERRO: pip install supabase", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
import dv01_dinamico as dv


PARQUET_B3 = Path("Dados/df_preco_de_ajuste_atual_completo.parquet")
ATIVOS_LOCAL_SPOT = ('WDO1', 'TREASURY')
PREFIXOS_LOCAL_SPOT = ('NTNB',)

_RE_DI_NOVO  = re.compile(r'^DI_[FGHJKMNQUVXZ]\d{2}$')
_RE_DAP_NOVO = re.compile(r'^DAP_[KQ]\d{2}$')
_RE_DI_OLD   = re.compile(r'^DI_(\d{2})$')
_RE_DAP_OLD  = re.compile(r'^DAP(\d{2})$')

NOMINAL = 100_000.0
DIAS_ANO = 252
RS_POR_PONTO_DAP_BASE = 0.00025


# =====================================================================
# Supabase helpers (só pra config_risco agora)
# =====================================================================

def _get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL")
            key = key or st.secrets.get("SUPABASE_KEY")
        except Exception:
            pass
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY não configuradas")
    return create_client(url, key)


def _get_projecao_anbima(client) -> float:
    try:
        resp = (client.table("config_risco")
                      .select("valor")
                      .eq("parametro", "ipca_projecao_anbima_pct")
                      .execute())
        if resp.data:
            v = resp.data[0]["valor"]
            if isinstance(v, str):
                v = v.strip('"')
            return float(v)
    except Exception as e:
        print(f"[projecao_anbima] erro: {e}. Usando 0.05.")
    return 0.05


# =====================================================================
# rs_dap por data
# =====================================================================

def _construir_rs_dap_por_data(datas, ni_ipca, feriados, projecao_pct) -> pd.Series:
    print(f"  [rs_dap] calculando pra {len(datas)} datas (projecao={projecao_pct}%)...")
    rs_map = {}
    for d in datas:
        try:
            res = dv.ipca_pro_rata(d, projecao_pct, ni_ipca, feriados)
            rs_map[d] = res['rs_dap']
        except Exception:
            try:
                ultimo_ni = float(ni_ipca.iloc[-1]['NI']) if 'NI' in ni_ipca.columns else float(ni_ipca.iloc[-1])
                rs_map[d] = RS_POR_PONTO_DAP_BASE * ultimo_ni
            except Exception:
                rs_map[d] = RS_POR_PONTO_DAP_BASE * 7000.0
    return pd.Series(rs_map, name='rs_dap')


# =====================================================================
# Leitura parquet local (pt-BR strings → float)
# =====================================================================

def _ptbr_to_float(v):
    if v is None:
        return np.nan
    try:
        if pd.isna(v):
            return np.nan
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        pass
    s = str(v).strip()
    if not s or s == '-':
        return np.nan
    s2 = s.replace('.', '').replace(',', '.')
    try:
        return float(s2)
    except Exception:
        return np.nan


def _ler_parquet_b3_long(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        return pd.DataFrame(columns=['Data', 'Assets', 'valor'])
    df = pd.read_parquet(caminho)
    if 'Assets' not in df.columns:
        return pd.DataFrame(columns=['Data', 'Assets', 'valor'])
    df_long = df.melt(id_vars=['Assets'], var_name='Data', value_name='valor_raw')
    df_long['Data'] = pd.to_datetime(df_long['Data'], errors='coerce')
    df_long = df_long.dropna(subset=['Data'])
    df_long['valor'] = df_long['valor_raw'].apply(_ptbr_to_float)
    df_long = df_long.dropna(subset=['valor'])
    return df_long[['Data', 'Assets', 'valor']]


# =====================================================================
# df_inicial (v2.5: SÓ do parquet local)
# =====================================================================

def gerar_df_inicial(caminho_saida: str = "Dados/df_inicial.parquet") -> pd.DataFrame:
    print(f"[gerar_df_inicial] começando (v2.5 — só parquet local)...")

    if not PARQUET_B3.exists():
        print(f"ERRO: {PARQUET_B3} não encontrado.")
        return pd.DataFrame()

    df_local_all = _ler_parquet_b3_long(PARQUET_B3)
    print(f"[local] total: {len(df_local_all)} linhas, {df_local_all['Assets'].nunique()} ativos")

    # -------- (A) DI: valor = PU direto do parquet --------
    mask_di = df_local_all['Assets'].apply(lambda a: bool(_RE_DI_NOVO.match(str(a))))
    df_di = df_local_all[mask_di].copy()
    # Filtra valores razoáveis (PU deve estar entre 1000 e 200000 para DI/DAP)
    df_di = df_di[(df_di['valor'] >= 1000) & (df_di['valor'] <= 200_000)]
    print(f"[A] DI: {len(df_di)} linhas válidas, {df_di['Assets'].nunique()} tickers")

    # -------- (B) DAP: valor = PU × rs_dap(data) --------
    mask_dap = df_local_all['Assets'].apply(lambda a: bool(_RE_DAP_NOVO.match(str(a))))
    df_dap = df_local_all[mask_dap].copy()
    df_dap = df_dap[(df_dap['valor'] >= 1000) & (df_dap['valor'] <= 200_000)]

    if not df_dap.empty:
        client = _get_client()
        projecao_pct = _get_projecao_anbima(client)
        feriados = dv.load_feriados()
        ni_ipca = dv.load_ni_ipca()

        datas_dap = sorted(pd.Timestamp(d) for d in df_dap['Data'].dt.normalize().unique())
        rs_map = _construir_rs_dap_por_data(datas_dap, ni_ipca, feriados, projecao_pct)
        df_dap['rs_dap'] = df_dap['Data'].dt.normalize().map(rs_map)
        df_dap['valor'] = df_dap['valor'] * df_dap['rs_dap']
        df_dap = df_dap[['Data', 'Assets', 'valor']]
    print(f"[B] DAP: {len(df_dap)} linhas convertidas para R$")

    # -------- (C) Spots (WDO1, TREASURY, NTNBs) --------
    mask_spot = (df_local_all['Assets'].isin(ATIVOS_LOCAL_SPOT)
                 | df_local_all['Assets'].apply(
                     lambda a: any(str(a).startswith(p) for p in PREFIXOS_LOCAL_SPOT)
                 ))
    df_spot = df_local_all[mask_spot].copy()
    print(f"[C] spot: {len(df_spot)} linhas, {df_spot['Assets'].nunique()} ativos")

    # -------- (D) Concat + pivot --------
    df_total = pd.concat([df_di, df_dap, df_spot], ignore_index=True)
    if df_total.empty:
        print("[gerar_df_inicial] nada para gerar!")
        return pd.DataFrame()

    wide = df_total.pivot_table(
        index='Data', columns='Assets', values='valor', aggfunc='last'
    ).sort_index()
    wide = wide.reset_index().rename(columns={'Data': 'Date'})

    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(caminho_saida, index=False)
    print(f"[gerar_df_inicial] salvo: {caminho_saida} — shape={wide.shape}")

    # Diagnóstico
    ultimo = wide.tail(1).iloc[0].to_dict()
    amostras = {k: v for k, v in ultimo.items()
                if k in ('DI_F26','DI_F29','DAP_Q26','DAP_K35','WDO1','TREASURY')}
    print(f"     amostras última linha: {amostras}")

    # Sanidade: mediana das colunas principais
    for col in ('DI_F29', 'DAP_Q26', 'WDO1'):
        if col in wide.columns:
            s = wide[col].dropna()
            if not s.empty:
                print(f"     {col}: n={len(s)}, mediana={s.median():.2f}, min={s.min():.2f}, max={s.max():.2f}")
    return wide


# =====================================================================
# df_divone (preserva base + atualiza DI/DAP com dv01_dinamico)
# =====================================================================

def _carregar_df_divone_base() -> pd.DataFrame:
    tentativas = [
        Path("Dados/df_divone.parquet"),
        Path("Dados_backup_pre_naming/df_divone.parquet"),
    ]
    for p in tentativas:
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            if 'TREASURY' in df.columns:
                print(f"[df_divone base] carregado de {p} — {df.shape[1]} colunas")
                return df.copy()
        except Exception as e:
            print(f"[df_divone base] erro lendo {p}: {e}")
    return pd.DataFrame()


def _renomear_dv_colunas_legado(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    novo_map = {}
    for c in df.columns:
        c_s = str(c)
        m = _RE_DI_OLD.match(c_s)
        if m:
            novo_map[c] = f'DI_F{m.group(1)}'
            continue
        m = _RE_DAP_OLD.match(c_s)
        if m:
            yy = int(m.group(1))
            letra = 'Q' if yy % 2 == 0 else 'K'
            novo_map[c] = f'DAP_{letra}{m.group(1)}'
    if novo_map:
        print(f"[df_divone base] renomeando {len(novo_map)} cols legado")
        df = df.rename(columns=novo_map)
    return df


def gerar_df_divone(caminho_saida: str = "Dados/df_divone.parquet",
                    data_ref: date | pd.Timestamp | None = None) -> pd.DataFrame:
    client = _get_client()
    data_ref = pd.Timestamp(data_ref).normalize() if data_ref else None

    # Se data_ref não passada, pega a última data disponível no parquet local
    if data_ref is None:
        df_local = _ler_parquet_b3_long(PARQUET_B3)
        if df_local.empty:
            print("[gerar_df_divone] parquet local vazio.")
            return pd.DataFrame()
        data_ref = pd.Timestamp(df_local['Data'].max()).normalize()

    print(f"[gerar_df_divone] data_ref = {data_ref.date()}")
    df_base = _carregar_df_divone_base()
    df_base = _renomear_dv_colunas_legado(df_base)

    # Deriva taxas do parquet local pra data_ref (últimos 30 dias)
    df_local_all = _ler_parquet_b3_long(PARQUET_B3)
    mask = df_local_all['Assets'].apply(
        lambda a: bool(_RE_DI_NOVO.match(str(a))) or bool(_RE_DAP_NOVO.match(str(a)))
    )
    df_di_dap = df_local_all[mask].copy()
    df_di_dap = df_di_dap[(df_di_dap['valor'] >= 1000) & (df_di_dap['valor'] <= 200_000)]
    df_di_dap = df_di_dap[df_di_dap['Data'] <= data_ref]

    if df_di_dap.empty:
        if df_base.empty:
            return pd.DataFrame()
        Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
        df_base.to_parquet(caminho_saida, index=True)
        return df_base

    # Última linha por ativo
    df_di_dap = df_di_dap.sort_values('Data').groupby('Assets').tail(1)

    projecao_pct = _get_projecao_anbima(client)
    feriados = dv.load_feriados()
    ni_ipca = dv.load_ni_ipca()

    dv01_por_ativo = {}
    for _, row in df_di_dap.iterrows():
        ticker = str(row['Assets'])
        pu = float(row['valor'])
        # Deriva taxa do PU
        try:
            venc = dv.vencimento(ticker, feriados)
        except Exception:
            continue
        du = dv.networkdays(pd.Timestamp(row['Data']).normalize(), venc, feriados)
        if du <= 0 or pu <= 0:
            continue
        try:
            taxa = ((NOMINAL / pu) ** (DIAS_ANO / du) - 1) * 100
            res = dv.calcular_dv01(
                ticker, taxa, data_ref,
                projecao_mensal_pct=projecao_pct,
                ni_ipca=ni_ipca,
                feriados=feriados,
            )
            dv01_por_ativo[ticker] = float(res["dv01"])
        except Exception as e:
            print(f"[gerar_df_divone] {ticker} skip: {e}")

    if df_base.empty:
        df_divone = pd.DataFrame([dv01_por_ativo], index=["FUT_TICK_VAL"])
    else:
        df_divone = df_base.copy()
        row_target = 'FUT_TICK_VAL' if 'FUT_TICK_VAL' in df_divone.index else df_divone.index[0]
        for ativo, v in dv01_por_ativo.items():
            df_divone.loc[row_target, ativo] = v

    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    df_divone.to_parquet(caminho_saida, index=True)
    print(f"[gerar_df_divone] salvo: {caminho_saida} — {df_divone.shape[1]} colunas, "
          f"{len(dv01_por_ativo)} DI/DAP atualizados, projeção={projecao_pct}%")
    return df_divone


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--inicial", action="store_true")
    ap.add_argument("--divone", action="store_true")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    if args.all or args.inicial:
        gerar_df_inicial()
    if args.all or args.divone:
        data = pd.Timestamp(args.data) if args.data else None
        gerar_df_divone(data_ref=data)
    if not (args.all or args.inicial or args.divone):
        print("Uso: python taxas_dinamicas.py --all")
