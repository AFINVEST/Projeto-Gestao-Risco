# -*- coding: utf-8 -*-
"""
Pipeline B3 (BDI) — via ConsolidatedTradesDerivatives
- Baixa UM endpoint: ConsolidatedTradesDerivatives
- Filtra DI1 / DAP / WDO (e DOL como fallback) e T10 (Treasury) e transforma em:
    * wide_preco: "Preço de Ajuste Atual" (coluna "Ajuste")
    * wide_valor: "Variação em pontos" (coluna "Variação")
      - EXCEÇÃO DAP: usa "Valor do ajuste por contrato (R$)" (cash) no lugar de IPCA
- Backfill (se o dia vier vazio)
- Exporta parquet + CSV (pt-BR em texto) + JSON (pt-BR em texto)
- Remove do output final: DI_25, DAP_25, DAP25, DI25
- Substitui "" / "-" por missing (pd.NA) no processamento e no parquet
- TREASURY: pega sempre o contrato "mais próximo" (>= mês de referência), como o WDO
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import time
import unicodedata
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import pandas_market_calendars as mcal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==========================
# Arquivos (bases)
# ==========================

PATH_LONG       = "Dados/df_ajustes_b3.parquet"                  # base longa
PATH_PRECO      = "Dados/df_preco_de_ajuste_atual_completo.parquet"
PATH_VALOR      = "Dados/df_valor_ajuste_contrato.parquet"
PATH_JSON       = "Dados/df_preco_de_ajuste_atual_completo.json" # JSON pt-BR (texto)
PATH_PRECO_CSV  = "Dados/df_preco_de_ajuste_atual_completo.csv"  # CSV pt-BR (texto)
PATH_VALOR_CSV  = "Dados/df_valor_ajuste_contrato.csv"           # CSV pt-BR (texto)
PATH_RUN_LOG    = "atualizacao_b3_log.txt"

# Assets a EXCLUIR do output final (parquet/csv/json)
ASSETS_EXCLUIR = ["DI_25", "DAP_25", "DAP25", "DI25"]

# ==========================
# HTTP — sessão com retries/timeouts
# ==========================

URL = "https://arquivos.b3.com.br/bdi/table/export/csv?lang=pt-BR"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "text/csv, application/json;q=0.9, */*;q=0.8",
    "Origin": "https://arquivos.b3.com.br",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}

# Nome ÚNICO do endpoint
B3_NAME = "ConsolidatedTradesDerivatives"

PAYLOAD_BASE = {
    "Name": B3_NAME,
    "Date": "2025-01-02",
    "FinalDate": "2025-01-02",
    "ClientId": "",
    "Filters": {},
}

HTTP_CONNECT_TIMEOUT = 3.0
HTTP_READ_TIMEOUT    = 20.0
HTTP_TOTAL_BUDGET    = 90.0

RETRY_CFG = Retry(
    total=2,
    backoff_factor=0.6,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset(["POST"])
)

_session = requests.Session()
_adapter = HTTPAdapter(max_retries=RETRY_CFG, pool_connections=10, pool_maxsize=10)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


# ==========================
# Debug / inspeção
# ==========================

DEBUG_MAX_LINES = 40
DEBUG_DUMP_DIR  = "debug_b3_csv"  # None para desativar
BACKOFF_LIM     = 15              # janela backoff (dias ÚTEIS)


try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("America/Sao_Paulo")
except Exception:
    _TZ = None


def _append_log(msg: str):
    ts = dt.datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S") if _TZ else dt.datetime.now().isoformat(sep=" ", timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(PATH_RUN_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _ensure_debug_dir():
    if DEBUG_DUMP_DIR:
        Path(DEBUG_DUMP_DIR).mkdir(parents=True, exist_ok=True)


def _dump_csv(data: dt.date, raw_bytes: bytes):
    if not DEBUG_DUMP_DIR:
        return
    _ensure_debug_dir()
    fn = Path(DEBUG_DUMP_DIR) / f"{B3_NAME}_{data.strftime('%Y-%m-%d')}.csv"
    try:
        fn.write_bytes(raw_bytes)
        print(f"    [dump] CSV bruto salvo em: {fn}")
    except Exception as e:
        print(f"    [dump:fail] {e}")


def _print_snippet(tag: str, text: str, max_lines: int = DEBUG_MAX_LINES):
    lines = (text or "").splitlines()
    header = f"----[ {tag} | primeiras {min(len(lines), max_lines)} de {len(lines)} linhas ]----"
    print(header)
    for ln in lines[:max_lines]:
        print(ln)
    print("-" * len(header))


# ==========================
# Helpers de parsing / missing
# ==========================

def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')


def _normalize_missing_values_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Troca strings vazias/whitespace/"-" por pd.NA.
    """
    if df is None or df.empty:
        return df
    df2 = df.replace(r"^\s*$", pd.NA, regex=True)
    df2 = df2.replace({"-": pd.NA, "–": pd.NA, "—": pd.NA})
    df2 = df2.replace({"nan": pd.NA, "NaN": pd.NA, "none": pd.NA, "None": pd.NA, "null": pd.NA, "NULL": pd.NA})
    return df2


def ptbr_to_float(s):
    """
    Converte string pt-BR para float.
    Retorna None para "", "-", None, NaN etc.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)

    s = str(s).strip()
    if s in {"", "-", "–", "—"}:
        return None

    s = re.sub(r"[^0-9\-,\.]", "", s).strip()
    if s in {"", "-"}:
        return None

    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def remover_assets_indesejados(w: pd.DataFrame) -> pd.DataFrame:
    if w is None or w.empty:
        return w
    return w.drop(index=ASSETS_EXCLUIR, errors="ignore")


def _fmt_ptbr_2dec(x):
    """
    98252.84 -> "98.252,84"
    Missing -> pd.NA
    """
    if x is None or pd.isna(x):
        return pd.NA
    if isinstance(x, str) and x.strip() == "":
        return pd.NA

    try:
        v = float(x)
    except Exception:
        s = str(x)
        return pd.NA if s.strip() == "" else s

    s = f"{v:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def wide_to_ptbr_json_text(wide_df: pd.DataFrame) -> str:
    if wide_df is None or wide_df.empty:
        return "[]"

    cols_norm = []
    for c in wide_df.columns:
        try:
            cols_norm.append(pd.to_datetime(c).strftime("%Y-%m-%d"))
        except Exception:
            cols_norm.append(str(c))

    df = wide_df.copy()
    df.columns = cols_norm
    df.index.name = "Assets"

    df_txt = df.copy()
    for c in df_txt.columns:
        df_txt[c] = df_txt[c].map(_fmt_ptbr_2dec)

    records = []
    for asset, row in df_txt.iterrows():
        rec = {"Assets": str(asset)}
        for col in df_txt.columns:
            val = row[col]
            rec[str(col)] = "" if (val is None or pd.isna(val)) else str(val)
        records.append(rec)

    return json.dumps(records, ensure_ascii=False)


# ==========================
# Payload / download do CSV
# ==========================

def montar_payload(data: dt.date) -> dict:
    p = deepcopy(PAYLOAD_BASE)
    s = data.strftime("%Y-%m-%d")
    p["Date"] = s
    p["FinalDate"] = s
    return p


def baixar_csv_b3(data: dt.date) -> str:
    start = time.monotonic()
    r = _session.post(
        URL,
        headers=HEADERS,
        json=montar_payload(data),
        timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
    )
    elapsed = time.monotonic() - start
    clen = r.headers.get("Content-Length", "?")
    _append_log(f"[HTTP] {B3_NAME} {data} -> status={r.status_code} content-length={clen} t={elapsed:.2f}s")

    if r.status_code != 200:
        raise RuntimeError(f"{data}: HTTP {r.status_code} - {r.text[:300]}")
    if not r.content:
        raise RuntimeError(f"{data}: CSV vazio")

    _dump_csv(data, r.content)
    md5 = hashlib.md5(r.content).hexdigest()
    _append_log(f"[HTTP] md5={md5} bytes={len(r.content)} ({B3_NAME} {data})")

    # decode
    txt = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            txt = r.content.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    if txt is None:
        txt = r.content.decode("utf-8", errors="replace")

    _print_snippet(f"RAW {B3_NAME} {data}", txt)
    return txt


# ==========================
# Mapeamento de ativos
# ==========================

MONTH_CODE = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12
}


def maturity_date_from_venc(venc: str) -> dt.date | None:
    if not venc or len(venc) < 3:
        return None
    letra = venc[0].upper()
    yy = venc[-2:]
    if letra not in MONTH_CODE or not yy.isdigit():
        return None
    return dt.date(2000 + int(yy), MONTH_CODE[letra], 1)


def mapear_asset(name: str, venc: str) -> str | None:
    """
    - DAP (IPCACoupon):
        * ano par  -> manter só Q
        * ano ímpar -> manter só K
        -> Asset "DAPyy"
    - DI (DI1Day): manter ano >= 26
        -> Asset "DI_yy"
    - WDO/DOL -> "WDO1"
    - Treasury (T10 -> USTNOTEFuture) -> "TREASURY"
    """
    if not venc:
        return None

    venc = venc.strip().upper()

    if name == "IPCACoupon":
        if len(venc) < 3:
            return None
        letra = venc[0]
        ano_str = venc[-2:]
        try:
            ano = int(ano_str)
        except ValueError:
            return None

        if ano % 2 == 0:
            if letra != "Q":
                return None
        else:
            if letra != "K":
                return None

        return f"DAP{ano_str}"

    if name == "DI1Day":
        sufixo = venc[-2:]
        try:
            ano = int(sufixo)
        except ValueError:
            return None
        if not (26 <= ano):
            return None
        return f"DI_{sufixo}"

    if name in ("BusinessDollar", "WDOMiniFuture"):
        return "WDO1"

    if name == "USTNOTEFuture":
        return "TREASURY"

    return None


# ==========================
# Parsing do ConsolidatedTradesDerivatives
# ==========================

def parse_consolidated_trades(csv_text: str, data_ref: dt.date) -> pd.DataFrame:
    """
    Lê o CSV do ConsolidatedTradesDerivatives e devolve DF “long” já enxuto.

    Inclui: DI1 / DAP / WDO / DOL / T10 (Treasury)

    Colunas finais:
      - Instrumento
      - Vencimento (ex.: F26 / H26 / M26)
      - Name (DI1Day / IPCACoupon / WDOMiniFuture / BusinessDollar / USTNOTEFuture)
      - PrecoAjusteAtual (float)  <- coluna "Ajuste"
      - Pontos (float)            <- coluna "Variação"
      - ValorAjusteR$ (float)     <- coluna "Valor do ajuste por contrato (R$)"
      - Data_Referencia (datetime)
      - ValorIndiceDia (NA)
    """
    df_raw = pd.read_csv(
        io.StringIO(csv_text),
        sep=";",
        skiprows=2,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=[
            "Instrumento","Vencimento","Name",
            "PrecoAjusteAtual","Pontos","ValorAjusteR$",
            "Data_Referencia","ValorIndiceDia"
        ])

    df_raw = _normalize_missing_values_df(df_raw)

    colmap = {c: _strip_accents(c).lower() for c in df_raw.columns}

    def _find_col(*must_have):
        for orig, norm in colmap.items():
            if all(x in norm for x in must_have):
                return orig
        return None

    c_inst = _find_col("instrumento", "financeiro") or _find_col("instrumento")
    c_ajuste = _find_col("ajuste")
    c_var = _find_col("variacao", "ponto") or _find_col("variacao")
    c_val_adj = _find_col("valor", "ajuste", "contrato")

    if not all([c_inst, c_ajuste, c_var, c_val_adj]):
        _append_log(f"[parse] Colunas esperadas não encontradas. Vistas: {list(df_raw.columns)}")
        return pd.DataFrame(columns=[
            "Instrumento","Vencimento","Name",
            "PrecoAjusteAtual","Pontos","ValorAjusteR$",
            "Data_Referencia","ValorIndiceDia"
        ])

    df = df_raw[[c_inst, c_ajuste, c_var, c_val_adj]].copy()
    df.columns = ["Instrumento", "PrecoAjusteAtual", "Pontos", "ValorAjusteR$"]

    # Mantém apenas tickers “curtos” (6 chars) e prefixos relevantes (inclui T10)
    s = df["Instrumento"].astype(str)
    df = df[s.str.len().eq(6) & s.str.match(r"^(DI1|DAP|WDO|DOL|T10)", na=False)].copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "Instrumento","Vencimento","Name",
            "PrecoAjusteAtual","Pontos","ValorAjusteR$",
            "Data_Referencia","ValorIndiceDia"
        ])

    # Vencimento = parte após o prefixo (3 chars): ex DI1F26 -> F26 / T10H26 -> H26
    df["Vencimento"] = df["Instrumento"].astype(str).str[3:]

    def _name_from_inst(x: str) -> str:
        if x.startswith("DI1"):
            return "DI1Day"
        if x.startswith("DAP"):
            return "IPCACoupon"
        if x.startswith("WDO"):
            return "WDOMiniFuture"
        if x.startswith("DOL"):
            return "BusinessDollar"
        if x.startswith("T10"):
            return "USTNOTEFuture"
        return "Other"

    df["Name"] = df["Instrumento"].astype(str).map(_name_from_inst)

    df["PrecoAjusteAtual"] = df["PrecoAjusteAtual"].map(ptbr_to_float)
    df["Pontos"] = df["Pontos"].map(ptbr_to_float)
    df["ValorAjusteR$"] = df["ValorAjusteR$"].map(ptbr_to_float)

    df["Data_Referencia"] = pd.to_datetime(data_ref)
    df["ValorIndiceDia"] = pd.NA

    df = df[(pd.notna(df["PrecoAjusteAtual"])) | (pd.notna(df["Pontos"])) | (pd.notna(df["ValorAjusteR$"]))].copy()

    return df[[
        "Instrumento","Vencimento","Name",
        "PrecoAjusteAtual","Pontos","ValorAjusteR$",
        "Data_Referencia","ValorIndiceDia"
    ]].reset_index(drop=True)


def selecionar_vertices(df_day: pd.DataFrame, data_ref: dt.date) -> pd.DataFrame:
    """
    - DI: mantém só DI1Fyy (Jan) por ano -> DI_yy
    - DAP: regra par/ímpar (Q/K) -> DAPyy
    - WDO1: escolhe 1 contrato (prefere WDO, senão DOL) com vencimento mais próximo (>= mês ref)
    - TREASURY: escolhe 1 contrato T10 com vencimento mais próximo (>= mês ref), igual WDO
    """
    if df_day is None or df_day.empty:
        return df_day

    df = df_day.copy()

    # DI: keep apenas mês "F" (Jan)
    di_mask = df["Name"].eq("DI1Day")
    df = df[~di_mask | df["Vencimento"].astype(str).str.startswith("F", na=False)].copy()

    # Mapeia Asset
    df["Asset"] = [mapear_asset(n, v) for n, v in zip(df["Name"], df["Vencimento"])]
    df = df[df["Asset"].notna()].copy()

    ref_month = dt.date(data_ref.year, data_ref.month, 1)

    # Seleção do WDO1
    w = df[df["Asset"].eq("WDO1")].copy()
    if not w.empty:
        w["Prefix"] = w["Instrumento"].astype(str).str[:3]
        w["MatDate"] = w["Vencimento"].map(maturity_date_from_venc)

        w_pref = w[w["Prefix"].eq("WDO")].copy()
        if w_pref.empty:
            w_pref = w.copy()

        w_pref["MatDate2"] = w_pref["MatDate"].fillna(dt.date(2099, 1, 1))
        after = w_pref[w_pref["MatDate2"] >= ref_month]
        use = after if not after.empty else w_pref

        best = use.sort_values("MatDate2").head(1)
        df = pd.concat([df[df["Asset"].ne("WDO1")], best], ignore_index=True)

    # Seleção do TREASURY (T10)
    t = df[df["Asset"].eq("TREASURY")].copy()
    if not t.empty:
        t["MatDate"] = t["Vencimento"].map(maturity_date_from_venc)
        t["MatDate2"] = t["MatDate"].fillna(dt.date(2099, 1, 1))
        after = t[t["MatDate2"] >= ref_month]
        use = after if not after.empty else t

        best = use.sort_values("MatDate2").head(1)
        df = pd.concat([df[df["Asset"].ne("TREASURY")], best], ignore_index=True)

    return df.reset_index(drop=True)


# ==========================
# Base longa
# ==========================

LONG_COLS = [
    "Instrumento","Vencimento","Name",
    "PrecoAjusteAtual","Pontos","ValorAjusteR$",
    "Data_Referencia","ValorIndiceDia"
]


def carregar_base_parquet_long(path_parquet: str) -> pd.DataFrame:
    p = Path(path_parquet)
    if p.exists():
        base = pd.read_parquet(p)
        for c in LONG_COLS:
            if c not in base.columns:
                base[c] = pd.NA
        base = base[LONG_COLS].copy()
        return base
    return pd.DataFrame(columns=LONG_COLS)


def incrementar_base_ajuste(path_parquet: str, df_novo: pd.DataFrame,
                           chaves=("Data_Referencia","Name","Vencimento")) -> pd.DataFrame:
    df_base = carregar_base_parquet_long(path_parquet)
    df_novo2 = df_novo.copy()
    for c in LONG_COLS:
        if c not in df_novo2.columns:
            df_novo2[c] = pd.NA
    df_novo2 = df_novo2[LONG_COLS].copy()

    df_comb = pd.concat([df_base, df_novo2], ignore_index=True) if not df_base.empty else df_novo2.copy()
    df_comb = df_comb.drop_duplicates(subset=list(chaves), keep="last")
    df_comb.to_parquet(path_parquet, index=False)
    return df_comb


# ==========================
# Wides (preço e valor)
# ==========================

def b3_calendar():
    return mcal.get_calendar("B3")


def b3_valid_days(start: dt.date, end: dt.date) -> list[dt.date]:
    v = b3_calendar().valid_days(start, end)
    return [d.date() for d in v]


def drop_tail_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Se a última coluna for apenas uma cópia da penúltima e representar
    exatamente o próximo dia útil B3, remove essa coluna.
    """
    if df is None or df.empty or df.shape[1] < 2:
        return df

    try:
        cols_dt = sorted(pd.to_datetime(df.columns))
    except Exception:
        return df

    last_dt = cols_dt[-1].date()
    prev_dt = cols_dt[-2].date()
    last_col = cols_dt[-1].strftime("%Y-%m-%d")
    prev_col = cols_dt[-2].strftime("%Y-%m-%d")

    prox_list = b3_valid_days(prev_dt, prev_dt + dt.timedelta(days=10))
    prox_list = [d for d in prox_list if d > prev_dt]
    if not prox_list:
        return df

    prox_dt = prox_list[0]
    if prox_dt != last_dt:
        return df

    try:
        if df[last_col].equals(df[prev_col]):
            _append_log(f"[drop-dup] Removendo coluna duplicada {last_col} (cópia de {prev_col})")
            return df.drop(columns=[last_col])
    except Exception:
        return df

    return df


def ler_wide(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame().rename_axis("Assets")

    df = pd.read_parquet(path)

    if "Assets" in df.columns:
        df = df.set_index("Assets")
    df.index.name = "Assets"

    try:
        cols_dt = sorted(pd.to_datetime(df.columns))
        df = df[[c.strftime("%Y-%m-%d") for c in cols_dt]]
    except Exception:
        pass

    df = df[~df.index.duplicated(keep="last")]
    df = drop_tail_duplicate(df)
    df = _normalize_missing_values_df(df)
    df = remover_assets_indesejados(df)
    return df


def salvar_wide(df: pd.DataFrame, path_parquet: str, path_csv: str, csv_ptbr_text: bool = True):
    df2 = df.copy()
    df2.index.name = "Assets"
    base = df2.reset_index().rename(columns={df2.reset_index().columns[0]: "Assets"})

    if csv_ptbr_text:
        out_txt = base.copy()

        cols_norm = []
        for c in out_txt.columns:
            if c == "Assets":
                cols_norm.append(c)
            else:
                try:
                    cols_norm.append(pd.to_datetime(c).strftime("%Y-%m-%d"))
                except Exception:
                    cols_norm.append(str(c))
        out_txt.columns = cols_norm

        for c in out_txt.columns:
            if c == "Assets":
                continue
            out_txt[c] = out_txt[c].map(_fmt_ptbr_2dec)

        out_txt.to_parquet(path_parquet, index=False)
        out_txt.to_csv(path_csv, index=False, encoding="utf-8")
    else:
        base.to_parquet(path_parquet, index=False)
        base.to_csv(path_csv, index=False, encoding="utf-8")


def adicionar_coluna_duplicada_final(wp: pd.DataFrame, wv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    col_names = []
    if wp is not None and wp.shape[1] > 0:
        col_names.extend(list(wp.columns))
    if wv is not None and wv.shape[1] > 0:
        col_names.extend(list(wv.columns))

    if not col_names:
        _append_log("[dup-final] wp/wv vazios → nada para duplicar.")
        return wp, wv

    try:
        cols_dt = sorted(pd.to_datetime(col_names))
    except Exception:
        _append_log("[dup-final] Não consegui interpretar colunas como datas → não duplica.")
        return wp, wv

    last_dt = cols_dt[-1].date()
    last_col = cols_dt[-1].strftime("%Y-%m-%d")

    prox_list = b3_valid_days(last_dt, last_dt + dt.timedelta(days=10))
    prox_list = [d for d in prox_list if d > last_dt]
    if not prox_list:
        _append_log(f"[dup-final] Não há próximo dia útil após {last_dt} → não duplica.")
        return wp, wv

    prox_dt = prox_list[0]
    prox_col = prox_dt.strftime("%Y-%m-%d")

    already_p = wp is not None and wp.shape[1] > 0 and prox_col in wp.columns
    already_v = wv is not None and wv.shape[1] > 0 and prox_col in wv.columns

    if already_p and already_v:
        _append_log(f"[dup-final] Coluna {prox_col} já existe em preços e valores → nenhuma duplicação feita.")
        return wp, wv

    if wp is not None and wp.shape[1] > 0 and not already_p and last_col in wp.columns:
        wp[prox_col] = wp[last_col]
        _append_log(f"[dup-final] (preço) Duplicado {last_col} -> {prox_col}")

    if wv is not None and wv.shape[1] > 0 and not already_v and last_col in wv.columns:
        wv[prox_col] = wv[last_col]
        _append_log(f"[dup-final] (valor) Duplicado {last_col} -> {prox_col}")

    return wp, wv


def construir_colunas_wide_duplas(df_long_dia: pd.DataFrame, data_ref: dt.date) -> tuple[pd.Series, pd.Series]:
    """
    - s_preco: PrecoAjusteAtual (Ajuste)
    - s_valor: Pontos (Variação) para DI/WDO/TREASURY
      - DAP: usa ValorAjusteR$ (Valor do ajuste por contrato (R$))
    """
    df = selecionar_vertices(df_long_dia, data_ref)
    if df is None or df.empty:
        col_name = pd.to_datetime(data_ref).strftime("%Y-%m-%d")
        s_preco = pd.Series(dtype="float64", name=col_name)
        s_valor = pd.Series(dtype="float64", name=col_name)
        return s_preco, s_valor

    if "Asset" not in df.columns:
        df["Asset"] = [mapear_asset(n, v) for n, v in zip(df["Name"], df["Vencimento"])]
        df = df[df["Asset"].notna()].copy()

    df["MatDate"] = df["Vencimento"].map(maturity_date_from_venc)
    df = df.sort_values(["Asset", "MatDate", "Instrumento"], na_position="last").copy()

    s_preco = df.groupby("Asset")["PrecoAjusteAtual"].first()
    s_valor = df.groupby("Asset")["Pontos"].first()

    dap_mask = df["Asset"].astype(str).str.startswith("DAP", na=False)
    if dap_mask.any():
        s_cash = df.loc[dap_mask].groupby("Asset")["ValorAjusteR$"].first()
        for a, v in s_cash.items():
            if pd.notna(v):
                s_valor.loc[a] = v

    col_name = pd.to_datetime(data_ref).strftime("%Y-%m-%d")
    s_preco.name = col_name
    s_valor.name = col_name
    return s_preco, s_valor


# ==========================
# Fetch com backoff (histórico) + “hoje” exato
# ==========================

def buscar_dia_com_backoff(target_d: dt.date) -> pd.DataFrame:
    validos_back = b3_valid_days(target_d - dt.timedelta(days=60), target_d)[::-1]
    tentativas_max = max(BACKOFF_LIM, 15)

    tentativa = 0
    for prev_d in [target_d] + validos_back:
        try:
            csv_text = baixar_csv_b3(prev_d)
            df_n = parse_consolidated_trades(csv_text, prev_d)
            if df_n is not None and not df_n.empty:
                df_n["Data_Referencia"] = pd.to_datetime(target_d)
                if prev_d != target_d:
                    _append_log(f"• {B3_NAME}: {target_d} vazio → usando {prev_d} (backfill)")
                return df_n
        except Exception as e:
            _append_log(f"! {B3_NAME} @ {prev_d}: falhou parse ({str(e)[:120]})")

        tentativa += 1
        if tentativa > tentativas_max:
            break

    _append_log(f"! {B3_NAME}: sem dados até {tentativas_max} DUs atrás para {target_d}")
    return pd.DataFrame(columns=LONG_COLS)


def buscar_dia_EXATO_sem_backfill(target_d: dt.date) -> pd.DataFrame:
    _append_log(f"[today-check] Coleta EXATA do dia {target_d} (sem backfill).")
    try:
        csv_text = baixar_csv_b3(target_d)
        df_n = parse_consolidated_trades(csv_text, target_d)
        if df_n is not None and not df_n.empty:
            df_n["Data_Referencia"] = pd.to_datetime(target_d)
            _append_log(f"[today-ok] Dados encontrados para {target_d}: {len(df_n)} linhas.")
            return df_n
    except Exception as e:
        _append_log(f"[today-empty] {target_d}: {e}")

    _append_log(f"[today-nodata] NENHUM dado em {target_d}.")
    return pd.DataFrame(columns=LONG_COLS)


# ==========================
# Calendário / range
# ==========================

def ultimo_dia_util_ANTES_de_hoje() -> dt.date:
    today = dt.date.today()
    v = b3_calendar().valid_days(today - dt.timedelta(days=20), today - dt.timedelta(days=1))
    return v[-1].date()


# ==========================
# Pipeline principal
# ==========================

def main():
    wide_preco = ler_wide(PATH_PRECO)
    wide_valor = ler_wide(PATH_VALOR)

    wide_preco = remover_assets_indesejados(_normalize_missing_values_df(wide_preco))
    wide_valor = remover_assets_indesejados(_normalize_missing_values_df(wide_valor))

    def _last_col_date(df):
        if df is None or df.empty or len(df.columns) == 0:
            return None
        try:
            return sorted(pd.to_datetime(df.columns))[-1].date()
        except Exception:
            return None

    last_preco = _last_col_date(wide_preco)
    last_valor = _last_col_date(wide_valor)

    if last_preco is None and last_valor is None:
        start_dt = dt.date(2025, 1, 2)
    else:
        candidates = [d for d in [last_preco, last_valor] if d is not None]
        start_dt = max(candidates)

    end_dt_hist = ultimo_dia_util_ANTES_de_hoje()
    dias_util = b3_valid_days(start_dt, end_dt_hist)

    _append_log(f"Atualizando de {start_dt} até {end_dt_hist} (DU B3: {len(dias_util)})")

    # 2) histórico (com backfill)
    for dref in dias_util:
        df_dia_all = buscar_dia_com_backoff(dref)
        if df_dia_all.empty:
            _append_log(f"{dref}: nenhum dado disponível — mantendo (sem atualização).")
            continue

        df_long = incrementar_base_ajuste(PATH_LONG, df_dia_all)
        _append_log(f"{dref}: base longa atualizada; total linhas = {len(df_long)}.")

        s_preco, s_valor = construir_colunas_wide_duplas(df_dia_all, dref)

        if wide_preco.empty:
            wide_preco = pd.DataFrame(s_preco)
        else:
            wide_preco = wide_preco.reindex(wide_preco.index.union(s_preco.index))
            wide_preco[s_preco.name] = s_preco

        if wide_valor.empty:
            wide_valor = pd.DataFrame(s_valor)
        else:
            wide_valor = wide_valor.reindex(wide_valor.index.union(s_valor.index))
            wide_valor[s_valor.name] = s_valor

        wide_preco = remover_assets_indesejados(_normalize_missing_values_df(wide_preco))
        wide_valor = remover_assets_indesejados(_normalize_missing_values_df(wide_valor))

        _append_log(f"{dref}: preços/valores atualizados.")

    # 3) tentar “hoje” (mantive sua lógica: pega ontem)
    today = dt.date.today() - dt.timedelta(days=1)
    is_today_du = today in set(b3_valid_days(today, today))

    if is_today_du:
        df_today = buscar_dia_EXATO_sem_backfill(today)
        if not df_today.empty:
            df_long = incrementar_base_ajuste(PATH_LONG, df_today)
            _append_log(f"{today}: base longa atualizada (hoje); total linhas = {len(df_long)}.")

            s_preco, s_valor = construir_colunas_wide_duplas(df_today, today)

            wide_preco = wide_preco.reindex(wide_preco.index.union(s_preco.index))
            wide_valor = wide_valor.reindex(wide_valor.index.union(s_valor.index))
            wide_preco[s_preco.name] = s_preco
            wide_valor[s_valor.name] = s_valor

            wide_preco = remover_assets_indesejados(_normalize_missing_values_df(wide_preco))
            wide_valor = remover_assets_indesejados(_normalize_missing_values_df(wide_valor))

            _append_log(f"{today}: preços/valores de HOJE adicionados.")
        else:
            _append_log(f"{today}: sem dados de HOJE — segue sem coluna de hoje (duplicação será feita no export final).")
    else:
        _append_log(f"{today}: não é dia útil B3 — não tenta hoje.")

    # 4) ordenar colunas
    def _order(df):
        if df is None or df.empty:
            return df
        try:
            cols_dt = sorted(pd.to_datetime(df.columns))
            return df[[c.strftime("%Y-%m-%d") for c in cols_dt]]
        except Exception:
            return df

    wide_preco = _order(wide_preco)
    wide_valor = _order(wide_valor)

    # 5) cria a coluna do próximo DU como cópia da última real, se ainda não existir
    wide_preco, wide_valor = adicionar_coluna_duplicada_final(wide_preco, wide_valor)

    # 6) normalize missing e remove assets indesejados no output FINAL
    wide_preco = remover_assets_indesejados(_normalize_missing_values_df(wide_preco))
    wide_valor = remover_assets_indesejados(_normalize_missing_values_df(wide_valor))

    wide_preco = _order(wide_preco)
    wide_valor = _order(wide_valor)

    salvar_wide(wide_preco, PATH_PRECO, PATH_PRECO_CSV, csv_ptbr_text=True)
    salvar_wide(wide_valor, PATH_VALOR, PATH_VALOR_CSV, csv_ptbr_text=True)
    _append_log(f"Salvos: {PATH_PRECO} {wide_preco.shape} | {PATH_VALOR} {wide_valor.shape}")

    # 7) JSON pt-BR (texto garantido) — missing vira ""
    try:
        json_text = wide_to_ptbr_json_text(wide_preco)
        with open(PATH_JSON, "w", encoding="utf-8") as f:
            f.write(json_text)
        _append_log(f"Salvo JSON pt-BR de preços (texto): {PATH_JSON}")
    except Exception as e:
        _append_log(f"[warn] Falha ao gerar JSON pt-BR: {e}")

if __name__ == "__main__":
    main()