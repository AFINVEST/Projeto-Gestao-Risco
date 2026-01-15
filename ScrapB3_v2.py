# -*- coding: utf-8 -*-
"""
Pipeline B3 (BDI) — via ConsolidatedTradesDerivatives
- Baixa UM endpoint: ConsolidatedTradesDerivatives
- Backfill (se o dia vier vazio)
- Exporta SOMENTE PARQUET (nada de CSV/JSON)
- Remove do output final: DI_25, DAP_25, DAP25, DI25
- Substitui "" / "-" por missing (pd.NA) no processamento e no parquet
- TREASURY: pega sempre o contrato "mais próximo" (>= mês de referência), como o WDO

NOVO LAYOUT DE DIRETÓRIO:
- Todos os parquets ficam em ./Dados (pasta "Dados" no mesmo nível do .py)

FIX (2026-01-13 bug):
- Garante que WIDE (preço/valor) fique NUMÉRICO antes do to_parquet
- Converte strings pt-BR tipo "100.000,00" -> 100000.00
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import os
import re
import time
import unicodedata
from copy import deepcopy
from pathlib import Path

import pandas as pd
import requests
import pandas_market_calendars as mcal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# Diretórios / Paths (NOVO)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "Dados"
DADOS_DIR.mkdir(parents=True, exist_ok=True)

PATH_LONG  = str(DADOS_DIR / "df_ajustes_b3.parquet")  # base longa
PATH_PRECO = str(DADOS_DIR / "df_preco_de_ajuste_atual_completo.parquet")
PATH_VALOR = str(DADOS_DIR / "df_valor_ajuste_contrato.parquet")

PATH_RUN_LOG = str(DADOS_DIR / "atualizacao_b3_log.txt")  # log simples (txt)

# Assets a EXCLUIR do output final (parquet)
ASSETS_EXCLUIR = ["DI_25", "DAP_25", "DAP25", "DI25"]


# ============================================================
# HTTP — sessão com retries/timeouts
# ============================================================

URL = "https://arquivos.b3.com.br/bdi/table/export/csv?lang=pt-BR"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "text/csv, application/json;q=0.9, */*;q=0.8",
    "Origin": "https://arquivos.b3.com.br",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}

B3_NAME = "ConsolidatedTradesDerivatives"

PAYLOAD_BASE = {
    "Name": B3_NAME,
    "Date": "2025-01-02",
    "FinalDate": "2025-01-02",
    "ClientId": "",
    "Filters": {},
}

HTTP_CONNECT_TIMEOUT = 3.0
HTTP_READ_TIMEOUT = 20.0
BACKOFF_LIM = 15  # janela backoff (dias úteis) para backfill

RETRY_CFG = Retry(
    total=2,
    backoff_factor=0.6,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset(["POST"]),
)

_session = requests.Session()
_adapter = HTTPAdapter(max_retries=RETRY_CFG, pool_connections=10, pool_maxsize=10)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("America/Sao_Paulo")
except Exception:
    _TZ = None


# ============================================================
# Logging
# ============================================================

def _append_log(msg: str) -> None:
    ts = (
        dt.datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S")
        if _TZ else dt.datetime.now().isoformat(sep=" ", timespec="seconds")
    )
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(PATH_RUN_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============================================================
# Helpers de parsing / missing / NUMERICALIZE (FIX)
# ============================================================

def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )


def _normalize_missing_values_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Troca strings vazias/whitespace/"-" por pd.NA.
    (mantém números intactos)
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
    Exemplos:
      "100.000,00" -> 100000.0
      "13,20"      -> 13.2
    Retorna None para "", "-", None, NaN etc.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
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


def _ptbr_series_to_float(s: pd.Series) -> pd.Series:
    """
    Vetorizado e seguro:
    - Se já é numérico: retorna como float
    - Se é object/string: converte pt-BR (milhar '.' / decimal ',')
    - Mantém NA como NA
    """
    if s is None:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")

    # strings/objects
    x = s.astype("string")
    x = x.str.strip()
    x = x.replace({"": pd.NA, "-": pd.NA, "–": pd.NA, "—": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "null": pd.NA, "NULL": pd.NA})

    # remove qualquer coisa que não seja dígito/sinal/ponto/vírgula
    x = x.str.replace(r"[^0-9\-,\.]", "", regex=True)

    # remove milhar e ajusta decimal
    x = x.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    return pd.to_numeric(x, errors="coerce")


def _ensure_wide_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX PRINCIPAL:
    Garante que TODAS as colunas de datas no wide sejam float64,
    evitando ArrowInvalid ao salvar Parquet (pyarrow não aceita "100.000,00" como double).
    """
    if df is None or df.empty:
        return df

    df2 = df.copy()

    # tenta identificar colunas que são datas (YYYY-MM-DD etc.)
    date_cols = []
    for c in df2.columns:
        try:
            pd.to_datetime(c)
            date_cols.append(c)
        except Exception:
            pass

    # se não conseguir identificar (colunas já podem estar como datetime), tenta assim:
    if not date_cols:
        try:
            date_cols = [c for c in df2.columns if pd.api.types.is_datetime64_any_dtype(pd.Index([c]))]
        except Exception:
            date_cols = []

    # fallback: se ainda nada, assume que todas as colunas são "valor" (wide puro)
    if not date_cols:
        date_cols = list(df2.columns)

    for c in date_cols:
        df2[c] = _ptbr_series_to_float(df2[c])

    return df2


def remover_assets_indesejados(w: pd.DataFrame) -> pd.DataFrame:
    if w is None or w.empty:
        return w
    return w.drop(index=ASSETS_EXCLUIR, errors="ignore")


# ============================================================
# Payload / download do CSV
# ============================================================

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

    md5 = hashlib.md5(r.content).hexdigest()
    _append_log(f"[HTTP] md5={md5} bytes={len(r.content)} ({B3_NAME} {data})")

    txt = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            txt = r.content.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    if txt is None:
        txt = r.content.decode("utf-8", errors="replace")

    return txt


# ============================================================
# Mapeamento de ativos
# ============================================================

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


# ============================================================
# Parsing do ConsolidatedTradesDerivatives
# ============================================================

def parse_consolidated_trades(csv_text: str, data_ref: dt.date) -> pd.DataFrame:
    """
    Lê o CSV do ConsolidatedTradesDerivatives e devolve DF “long” já enxuto.
    """
    df_raw = pd.read_csv(
        io.StringIO(csv_text),
        sep=";",
        skiprows=2,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )

    cols_out = [
        "Instrumento", "Vencimento", "Name",
        "PrecoAjusteAtual", "Pontos", "ValorAjusteR$",
        "Data_Referencia", "ValorIndiceDia",
    ]

    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=cols_out)

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
        return pd.DataFrame(columns=cols_out)

    df = df_raw[[c_inst, c_ajuste, c_var, c_val_adj]].copy()
    df.columns = ["Instrumento", "PrecoAjusteAtual", "Pontos", "ValorAjusteR$"]

    s = df["Instrumento"].astype(str)
    df = df[s.str.len().eq(6) & s.str.match(r"^(DI1|DAP|WDO|DOL|T10)", na=False)].copy()

    if df.empty:
        return pd.DataFrame(columns=cols_out)

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

    # long já nasce numérico (evita lixo virar string)
    df["PrecoAjusteAtual"] = df["PrecoAjusteAtual"].map(ptbr_to_float)
    df["Pontos"] = df["Pontos"].map(ptbr_to_float)
    df["ValorAjusteR$"] = df["ValorAjusteR$"].map(ptbr_to_float)

    df["Data_Referencia"] = pd.to_datetime(data_ref)
    df["ValorIndiceDia"] = pd.NA

    df = df[
        (pd.notna(df["PrecoAjusteAtual"]))
        | (pd.notna(df["Pontos"]))
        | (pd.notna(df["ValorAjusteR$"]))
    ].copy()

    return df[cols_out].reset_index(drop=True)


def selecionar_vertices(df_day: pd.DataFrame, data_ref: dt.date) -> pd.DataFrame:
    if df_day is None or df_day.empty:
        return df_day

    df = df_day.copy()

    di_mask = df["Name"].eq("DI1Day")
    df = df[~di_mask | df["Vencimento"].astype(str).str.startswith("F", na=False)].copy()

    df["Asset"] = [mapear_asset(n, v) for n, v in zip(df["Name"], df["Vencimento"])]
    df = df[df["Asset"].notna()].copy()

    ref_month = dt.date(data_ref.year, data_ref.month, 1)

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

    t = df[df["Asset"].eq("TREASURY")].copy()
    if not t.empty:
        t["MatDate"] = t["Vencimento"].map(maturity_date_from_venc)
        t["MatDate2"] = t["MatDate"].fillna(dt.date(2099, 1, 1))
        after = t[t["MatDate2"] >= ref_month]
        use = after if not after.empty else t

        best = use.sort_values("MatDate2").head(1)
        df = pd.concat([df[df["Asset"].ne("TREASURY")], best], ignore_index=True)

    return df.reset_index(drop=True)


# ============================================================
# Base longa
# ============================================================

LONG_COLS = [
    "Instrumento", "Vencimento", "Name",
    "PrecoAjusteAtual", "Pontos", "ValorAjusteR$",
    "Data_Referencia", "ValorIndiceDia",
]


def carregar_base_parquet_long(path_parquet: str) -> pd.DataFrame:
    p = Path(path_parquet)
    if p.exists():
        base = pd.read_parquet(p)
        for c in LONG_COLS:
            if c not in base.columns:
                base[c] = pd.NA
        return base[LONG_COLS].copy()
    return pd.DataFrame(columns=LONG_COLS)


def incrementar_base_ajuste(
    path_parquet: str,
    df_novo: pd.DataFrame,
    chaves=("Data_Referencia", "Name", "Vencimento"),
) -> pd.DataFrame:
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


# ============================================================
# Wides (preço e valor)
# ============================================================

def b3_calendar():
    return mcal.get_calendar("B3")


def b3_valid_days(start: dt.date, end: dt.date) -> list[dt.date]:
    v = b3_calendar().valid_days(start, end)
    return [d.date() for d in v]


def drop_tail_duplicate(df: pd.DataFrame) -> pd.DataFrame:
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

    # FIX: garante numérico já na leitura
    df = _ensure_wide_numeric(df)

    return df


def salvar_wide_parquet_only(df: pd.DataFrame, path_parquet: str) -> None:
    """
    Salva SOMENTE parquet, no formato: coluna "Assets" + colunas de datas.
    (FIX: força wide numérico antes de salvar)
    """
    df2 = df.copy()
    df2.index.name = "Assets"

    cols_norm = []
    for c in df2.columns:
        try:
            cols_norm.append(pd.to_datetime(c).strftime("%Y-%m-%d"))
        except Exception:
            cols_norm.append(str(c))
    df2.columns = cols_norm

    # FIX: garante numérico antes do to_parquet (evita ArrowInvalid)
    df2 = _ensure_wide_numeric(df2)

    out = df2.reset_index()
    out.to_parquet(path_parquet, index=False)


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

    if wp is not None and wp.shape[1] > 0 and (not already_p) and (last_col in wp.columns):
        wp[prox_col] = wp[last_col]
        _append_log(f"[dup-final] (preço) Duplicado {last_col} -> {prox_col}")

    if wv is not None and wv.shape[1] > 0 and (not already_v) and (last_col in wv.columns):
        wv[prox_col] = wv[last_col]
        _append_log(f"[dup-final] (valor) Duplicado {last_col} -> {prox_col}")

    return wp, wv


def construir_colunas_wide_duplas(df_long_dia: pd.DataFrame, data_ref: dt.date) -> tuple[pd.Series, pd.Series]:
    df = selecionar_vertices(df_long_dia, data_ref)
    col_name = pd.to_datetime(data_ref).strftime("%Y-%m-%d")

    if df is None or df.empty:
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

    # FIX: garante que séries sejam float64
    s_preco = pd.to_numeric(s_preco, errors="coerce").astype("float64")
    s_valor = pd.to_numeric(s_valor, errors="coerce").astype("float64")

    s_preco.name = col_name
    s_valor.name = col_name
    return s_preco, s_valor


# ============================================================
# Fetch com backoff (histórico) + “hoje” exato
# ============================================================

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


# ============================================================
# Calendário / range
# ============================================================

def ultimo_dia_util_ANTES_de_hoje() -> dt.date:
    today = dt.date.today()
    v = b3_calendar().valid_days(today - dt.timedelta(days=20), today - dt.timedelta(days=1))
    return v[-1].date()


# ============================================================
# Pipeline principal
# ============================================================

def main():
    _append_log(f"[BOOT] BASE_DIR={BASE_DIR}")
    _append_log(f"[BOOT] DADOS_DIR={DADOS_DIR}")

    wide_preco = ler_wide(PATH_PRECO)
    wide_valor = ler_wide(PATH_VALOR)

    wide_preco = remover_assets_indesejados(_normalize_missing_values_df(wide_preco))
    wide_valor = remover_assets_indesejados(_normalize_missing_values_df(wide_valor))

    # FIX: garante que wide lido já esteja numérico
    wide_preco = _ensure_wide_numeric(wide_preco)
    wide_valor = _ensure_wide_numeric(wide_valor)

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

        # FIX: a cada loop, garante wide numérico (mata "100.000,00" antes de acumular)
        wide_preco = _ensure_wide_numeric(wide_preco)
        wide_valor = _ensure_wide_numeric(wide_valor)

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

            # FIX: garante wide numérico pós-hoje
            wide_preco = _ensure_wide_numeric(wide_preco)
            wide_valor = _ensure_wide_numeric(wide_valor)

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

    # FIX FINAL: garante wide numérico antes de salvar
    wide_preco = _ensure_wide_numeric(wide_preco)
    wide_valor = _ensure_wide_numeric(wide_valor)

    wide_preco = _order(wide_preco)
    wide_valor = _order(wide_valor)

    # 7) salva SOMENTE PARQUET (overwrite)
    salvar_wide_parquet_only(wide_preco, PATH_PRECO)
    salvar_wide_parquet_only(wide_valor, PATH_VALOR)

    _append_log(f"[SAVE] {PATH_PRECO} shape={wide_preco.shape}")
    _append_log(f"[SAVE] {PATH_VALOR} shape={wide_valor.shape}")
    _append_log("[DONE] atualização concluída.")


if __name__ == "__main__":
    main()
