"""
cota_portfolio_core.py  (v1.2 — Supabase primeiro, fallback local)
===================================================================

Replica EXATAMENTE `simulate_nav_cota()` do app4.py, agora com a mesma
fonte de posições (Supabase `posicoes_por_fundo`, com fallback local).

MUDANÇAS v1.2:
  - `load_basefundos()` agora tenta Supabase primeiro (via
    `load_basefundos_supabase()` — cópia da lógica do app4). Se vazio,
    cai para local (`_load_basefundos_local()`).
  - Alias antigo->novo continua aplicado em ambos os caminhos.
  - Requer variáveis de ambiente SUPABASE_URL/SUPABASE_KEY para o
    caminho Supabase; sem elas, silenciosamente cai pra local.

Uso típico:
    from cota_portfolio_core import computar_cota_serie
    res = computar_cota_serie(pct=0.01, base_dir=r"Z:/...")
    df  = res["df"]
"""
from __future__ import annotations
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constantes (copiadas de app4.py)
# ─────────────────────────────────────────────────────────────────────────────

FUNDOS_ALLOW_LIST = {
    "AF DEB INCENTIVADAS", "BH FIRF INFRA", "BORDEAUX INFRA",
    "GLOBAL BONDS", "HORIZONTE", "JERA2026",
    "MANACA INFRA FIRF", "REAL FIM", "TOPAZIO INFRA",
}

TAXA_MAP = {
    "GLOBAL BONDS":            0.0050,
    "HORIZONTE":               0.0100,
    "JERA2026":                0.0028,
    "REAL FIM":                0.0033,
    "BH FIRF INFRA":           0.0020,
    "BORDEAUX INFRA":          0.0005,
    "TOPAZIO INFRA":           0.0054,
    "MANACA INFRA FIRF":       0.0005,
    "AF DEB INCENTIVADAS":     0.0100,
}

_TABELA_PPF = "posicoes_por_fundo"


# ─────────────────────────────────────────────────────────────────────────────
# Alias antigo -> novo (idempotente)
# ─────────────────────────────────────────────────────────────────────────────
_RE_DAP_NOVO    = re.compile(r"^DAP_[KQ]\d{2}$")
_RE_DAP_ANTIGO  = re.compile(r"^DAP(\d{2})$")
_RE_DI_NOVO     = re.compile(r"^DI_[FJNV]\d{2}$")
_RE_DI_ANTIGO   = re.compile(r"^DI_(\d{2})$")


def _alias_ativo_novo(ativo: str) -> str:
    s = str(ativo).strip()
    if _RE_DAP_NOVO.match(s) or _RE_DI_NOVO.match(s):
        return s
    m = _RE_DAP_ANTIGO.match(s)
    if m:
        yy = int(m.group(1))
        letra = "Q" if yy % 2 == 0 else "K"
        return f"DAP_{letra}{yy:02d}"
    m = _RE_DI_ANTIGO.match(s)
    if m:
        return f"DI_F{m.group(1)}"
    return s


def _migrate_basefundos_names(bf: dict, verbose: bool = True) -> dict:
    """Aplica alias novo aos indices de Ativo de cada fundo."""
    renames = {}
    out = {}
    for fundo, df in bf.items():
        if not hasattr(df, "index"):
            out[fundo] = df
            continue
        novo_index = df.index.map(_alias_ativo_novo)
        for a, b in zip(df.index, novo_index):
            if a != b:
                renames.setdefault(str(a), str(b))
        df2 = df.copy()
        df2.index = novo_index
        out[fundo] = df2
    if verbose and renames:
        print(f"[core] Naming migrado (in-memory): {renames}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# chdir contextual
# ─────────────────────────────────────────────────────────────────────────────
class _chdir_ctx:
    def __init__(self, novo):
        self.novo = str(novo) if novo else None
        self.antigo = None
    def __enter__(self):
        if self.novo:
            self.antigo = os.getcwd()
            os.chdir(self.novo)
        return self
    def __exit__(self, *args):
        if self.antigo:
            os.chdir(self.antigo)


# ─────────────────────────────────────────────────────────────────────────────
# Supabase — cliente lazy
# ─────────────────────────────────────────────────────────────────────────────
_SUPABASE_CLIENT = None

def _get_supabase():
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
    except ImportError:
        return None
    try:
        _SUPABASE_CLIENT = create_client(url, key)
        return _SUPABASE_CLIENT
    except Exception:
        return None


def _load_posicoes_por_fundo() -> pd.DataFrame:
    """Lê toda a tabela posicoes_por_fundo do Supabase — cópia de app4."""
    client = _get_supabase()
    if client is None:
        return pd.DataFrame()
    try:
        rows = []
        offset = 0
        page = 1000
        while True:
            resp = (
                client.table(_TABELA_PPF)
                .select("*")
                .range(offset, offset + page - 1)
                .execute()
            )
            data = resp.data or []
            if not data:
                break
            rows.extend(data)
            if len(data) < page:
                break
            offset += page
        if not rows:
            return pd.DataFrame(
                columns=["Fundo", "Ativo", "Data", "Quantidade",
                         "Preco_Compra", "Preco_Fechamento", "PL", "Rendimento"]
            )
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[core] Erro lendo Supabase posicoes_por_fundo: {e}")
        return pd.DataFrame()


def load_basefundos_supabase() -> dict:
    """Reconstrói formato wide a partir de posicoes_por_fundo. Cópia de app4."""
    df = _load_posicoes_por_fundo()
    if df is None or df.empty:
        return {}
    out = {}
    df["Data"] = pd.to_datetime(df["Data"]).dt.strftime("%Y-%m-%d")
    for fundo, sub in df.groupby("Fundo"):
        sub = sub.copy()
        partes = []
        for metrica in ("PL", "Preco_Fechamento", "Preco_Compra", "Quantidade", "Rendimento"):
            if metrica not in sub.columns:
                continue
            piv = sub.pivot_table(
                index="Ativo", columns="Data", values=metrica, aggfunc="last", dropna=False
            )
            piv.columns = [f"{c} - {metrica}" for c in piv.columns]
            partes.append(piv)
        if not partes:
            continue
        wide = pd.concat(partes, axis=1).sort_index(axis=1)
        wide = wide.reset_index()
        out[fundo] = wide
    return out


def _load_basefundos_local() -> dict[str, pd.DataFrame]:
    """Fallback: lê BaseFundos/*.parquet do disco."""
    out = {}
    for f in os.listdir("BaseFundos"):
        if f.startswith("_") or not f.lower().endswith(".parquet"):
            continue
        nome = f.rsplit(".", 1)[0]
        df = pd.read_parquet(f"BaseFundos/{f}").set_index("Ativo")
        out[nome] = df
    return out


_LOAD_BF_COUNT = 0

def load_basefundos(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Supabase primeiro, fallback local. Espelha o app4.
    Log so na primeira chamada (evita spam em loops)."""
    global _LOAD_BF_COUNT
    out = load_basefundos_supabase()
    fonte = "supabase"
    if not out:
        out = _load_basefundos_local()
        fonte = "local"
    fixed = {}
    for nome, df in out.items():
        if "Ativo" in df.columns:
            fixed[nome] = df.set_index("Ativo")
        else:
            fixed[nome] = df
    _LOAD_BF_COUNT += 1
    if verbose and _LOAD_BF_COUNT == 1:
        print(f"[core] load_basefundos: {len(fixed)} fundos (fonte={fonte})")
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# Outros loaders — cópias do app4.py
# ─────────────────────────────────────────────────────────────────────────────

def load_b3_prices() -> pd.DataFrame:
    df = pd.read_parquet("Dados/df_preco_de_ajuste_atual_completo.parquet")
    cols = df.columns[1:]
    converted = (
        df[cols]
          .replace(r"^\s*$", np.nan, regex=True)
          .apply(
              lambda col: pd.to_numeric(
                  col.astype(str)
                     .str.strip()
                     .str.replace(".", "", regex=False)
                     .str.replace(",", ".", regex=False),
                  errors="coerce"
              )
          )
    )
    df = pd.concat([df.iloc[:, :1], converted], axis=1)
    df.loc[df["Assets"] == "TREASURY", df.columns != "Assets"] *= 1000
    df.loc[df["Assets"] == "WDO1",     df.columns != "Assets"] *= 10
    return df


def load_ajustes() -> pd.DataFrame:
    df = pd.read_parquet("Dados/df_valor_ajuste_contrato.parquet")
    if df.shape[1] >= 3 and df.iloc[:, -1].equals(df.iloc[:, -2]):
        df = df.iloc[:, :-1].copy()
    cols = df.columns[1:]
    converted = (
        df[cols]
        .replace(r"^\s*$", np.nan, regex=True)
        .apply(
            lambda col: pd.to_numeric(
                col.astype(str)
                   .str.strip()
                   .str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False),
                errors="coerce",
            )
        )
    )
    df = pd.concat([df.iloc[:, :1], converted], axis=1)
    return df


def load_pl_series(return_fee: bool = True, dias_ano: int = 252):
    df = pd.read_parquet("Dados/pl_fundos_teste.parquet")
    df = df[df["Fundos/Carteiras Adm"].isin(FUNDOS_ALLOW_LIST)].copy()
    cols_data = [c for c in df.columns if c not in ("Fundos/Carteiras Adm", "Último Valor")]

    def _to_float(s):
        return (s.astype(str)
                 .str.replace(r"[R$\s\.]", "", regex=True)
                 .str.replace(",", ".", regex=False)
                 .replace({"": np.nan, "--": np.nan})
                 .astype(float)
                 .fillna(0.0))

    for c in cols_data:
        df[c] = _to_float(df[c])

    pl_by_fund = df.set_index("Fundos/Carteiras Adm")[cols_data]
    pl_by_fund.columns = pd.to_datetime(pl_by_fund.columns)

    pl_series = pl_by_fund.sum(axis=0).sort_index()
    pl_series.name = "PL_total"

    if not return_fee:
        return pl_series

    taxas_anual = pd.Series(TAXA_MAP, name="taxa_anual").reindex(pl_by_fund.index).fillna(0.0)
    rate_adm_dia = (1 + taxas_anual) ** (1 / dias_ano) - 1
    pl_by_fund = pl_by_fund * 0.01
    fee_by_fund = pl_by_fund.mul(rate_adm_dia, axis=0)
    fee_series = fee_by_fund.sum(axis=0).sort_index()
    fee_series.name = "taxa_total_dia"
    return pl_series, fee_series


def load_lft_series() -> pd.Series:
    df = (pd.read_csv("Dados/dados_lft.csv", parse_dates=["Data"])
            .rename(columns={"RetornoLFT": "preco"}))
    df.dropna(inplace=True)
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")
    df = df.sort_values("Data")
    df["lft_ret"] = df["preco"].pct_change().fillna(0.0)
    df["lft_ret"] = (df["lft_ret"]
                       .replace([np.inf, -np.inf], np.nan)
                       .fillna(0.0))
    return df.set_index("Data")["lft_ret"].astype(float)


def load_cdi_series(cache_csv: str = "Dados/cdi_cached.csv") -> pd.Series:
    """Le o CDI cacheado. SEM FALLBACK: se csv nao existir ou vazio, erro."""
    p = Path(cache_csv)
    if not p.exists():
        raise FileNotFoundError(
            f"{cache_csv} nao existe. Rode 'atualizar_cdi_lft.py' primeiro."
        )
    s = (pd.read_csv(cache_csv, parse_dates=["Data"])
            .set_index("Data")["cdi"]
            .astype(float)
            .sort_index())
    if s.empty:
        raise ValueError(f"{cache_csv} esta vazio.")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# analisar_dados_fundos2 — cópia literal do app4.py
# ─────────────────────────────────────────────────────────────────────────────

def analisar_dados_fundos2(
        soma_pl_sem_pesos: float,
        df_b3_fechamento: pd.DataFrame | None = None,
        df_ajuste: pd.DataFrame | None = None,
        basefundos: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:

    if df_b3_fechamento is None:
        df_b3_fechamento = load_b3_prices()
    if df_ajuste is None:
        df_ajuste = load_ajustes()
    if basefundos is None:
        basefundos = load_basefundos()

    df_b3 = df_b3_fechamento.copy()
    date_cols_b3 = pd.to_datetime(df_b3.columns[1:], errors="coerce")
    df_b3.columns = ["Assets"] + list(date_cols_b3)
    df_b3 = df_b3.loc[:, ["Assets"] + sorted([c for c in df_b3.columns[1:] if pd.notna(c)])]

    prices_only = df_b3.columns[1:]
    if df_b3[prices_only].select_dtypes(include="object").size > 0:
        df_b3[prices_only] = (df_b3[prices_only]
            .replace({r"\.": "", ",": "."}, regex=True)
            .apply(pd.to_numeric, errors="coerce"))

    preco_lookup = (df_b3.set_index("Assets").sort_index(axis=1).ffill(axis=1))

    df_ajuste = df_ajuste.copy()
    df_ajuste.columns = (["Assets"] +
                         [pd.to_datetime(c, errors="coerce").strftime("%Y-%m-%d")
                          for c in df_ajuste.columns[1:]])
    df_ajuste.set_index("Assets", inplace=True)

    pnl_cash, pnl_bps = {}, {}
    despesas_cash: dict[str, pd.Series] = {}
    financeiro_ntnb_daily = defaultdict(float)

    PU_FINAL = 100_000
    DESPESAS_FIXAS = {"DAP": 3.0, "DI": 3.0, "TREASURY": 10.0, "WDO1": 1.0}
    dolar = preco_lookup.loc["WDO1", preco_lookup.columns[-1]] if "WDO1" in preco_lookup.index else 1.0

    for fundo, df_f in basefundos.items():
        if fundo.upper() == "TOTAL":
            continue
        if "Ativo" in df_f.columns:
            df_f = df_f.set_index("Ativo")
        elif df_f.index.name is None:
            df_f.index.name = "Ativo"

        cols_qtd = [c for c in df_f.columns if c.endswith("Quantidade")]

        for ativo, linha in df_f.iterrows():
            for col_q in cols_qtd:
                qtd = float(linha[col_q]) if pd.notna(linha[col_q]) else 0.0
                if qtd == 0: continue
                data_op = pd.to_datetime(col_q.split()[0], errors="coerce")
                if pd.isna(data_op): continue
                p_compra = linha.get(col_q.replace("Quantidade", "Preco_Compra"), np.nan)
                pl_op    = linha.get(col_q.replace("Quantidade", "PL"), np.nan)

                if fundo.upper() != "TOTAL":
                    raiz = ativo.split("_")[0]
                    if raiz.startswith(("DAP", "DI")): raiz = raiz[:3]
                    if raiz == "DI":
                        try: PU_atual = float(preco_lookup.at[ativo, data_op])
                        except Exception: PU_atual = float(preco_lookup.at[ativo, preco_lookup.columns[-1]])
                        custo_op = (PU_FINAL - PU_atual) * 0.03 * abs(qtd) * 0.005
                    elif raiz == "WDO1":
                        try: PU_atual = float(preco_lookup.at[ativo, data_op])
                        except Exception: PU_atual = float(preco_lookup.at[ativo, preco_lookup.columns[-1]])
                        custo_op = (PU_atual) * 0.02 * abs(qtd) * 0.005
                    else:
                        custo_op = DESPESAS_FIXAS.get(raiz, 0.0) * abs(qtd)
                    if custo_op:
                        dkey = data_op.strftime("%Y-%m-%d")
                        despesas_cash.setdefault("Despesas", pd.Series()).at[dkey] = \
                            despesas_cash.get("Despesas", pd.Series()).get(dkey, 0.0) + custo_op

                usa_bps = not pd.isna(pl_op) and pl_op != 0
                p_ant   = p_compra

                for data_fech in preco_lookup.columns:
                    if data_fech < data_op: continue
                    p_fech = preco_lookup.get(data_fech, pd.Series()).get(ativo, np.nan)
                    if not np.isfinite(p_fech): continue

                    if ativo == "TREASURY":
                        rend = (p_fech - (p_ant if np.isfinite(p_ant) else p_fech)) * qtd * dolar / 10_000
                    elif "DAP" in ativo:
                        if data_fech == data_op:
                            rend = 0.0
                        else:
                            ajuste = df_ajuste.get(data_fech.strftime("%Y-%m-%d"), pd.Series()).get(ativo, 0.0)
                            rend   = ajuste * qtd
                    elif "DI" in ativo:
                        if data_fech == data_op:
                            rend = (p_fech - (p_ant if np.isfinite(p_ant) else p_fech)) * qtd
                        else:
                            ajuste = df_ajuste.get(data_fech.strftime("%Y-%m-%d"), pd.Series()).get(ativo, 0.0)
                            rend   = ajuste * qtd
                    elif "NTNB" in ativo:
                        rend = (p_fech - (p_ant if np.isfinite(p_ant) else p_fech)) * qtd
                        financeiro_ntnb_daily[data_fech] += float(p_fech) * float(qtd)
                    else:
                        rend = (p_fech - (p_ant if np.isfinite(p_ant) else p_fech)) * qtd

                    chave = f"{ativo} - {fundo} - P&L"
                    pnl_cash.setdefault(chave, pd.Series()).at[data_fech] = \
                        pnl_cash.get(chave, pd.Series()).get(data_fech, 0.0) + rend
                    if usa_bps:
                        pnl_bps.setdefault(chave, pd.Series()).at[data_fech] = \
                            pnl_bps.get(chave, pd.Series()).get(data_fech, 0.0) + rend / pl_op * 10_000
                    p_ant = p_fech

    df_final    = pd.DataFrame(pnl_cash).T.fillna(0.0)
    df_final_pl = pd.DataFrame(pnl_bps ).T.fillna(0.0)
    df_despesas = (pd.DataFrame(despesas_cash).T.rename_axis("Conta"))
    df_despesas.columns = pd.to_datetime(df_despesas.columns, errors="coerce")
    df_despesas = df_despesas.loc[:, ~df_despesas.columns.isna()].fillna(0.0)

    ZERO_EPS = 1e-9
    df_final_pl = df_final_pl.mask(df_final.abs() < ZERO_EPS, 0.0)

    df_final["Total"]    = df_final.sum(axis=1)
    df_final_pl["Total"] = df_final_pl.sum(axis=1)
    df_final    = df_final[~df_final.index.str.contains("Total")]
    df_final_pl = df_final_pl[~df_final_pl.index.str.contains("Total")]

    if len(financeiro_ntnb_daily):
        serie_ntnb = pd.Series(financeiro_ntnb_daily, dtype=float)
        serie_ntnb.index = pd.to_datetime(serie_ntnb.index)
        serie_ntnb = serie_ntnb.sort_index()
        dict_ntnb  = {d.strftime("%Y-%m-%d"): float(v) for d, v in serie_ntnb.items()}
    else:
        dict_ntnb = {}

    return df_final, df_final_pl, df_despesas, dict_ntnb


# ─────────────────────────────────────────────────────────────────────────────
# computar_cota_serie
# ─────────────────────────────────────────────────────────────────────────────

def computar_cota_serie(
    pct: float = 0.01,
    data_ini=None,
    data_fim=None,
    taxa_adm_on: bool = False,
    custo_pct_aa: float = 0.0,
    custo_fixo_rs: float = 0.0,
    perf_on: bool = False,
    perf_pct: float = 0.20,
    base_dir=None,
    migrate_naming: bool = True,
) -> dict:
    with _chdir_ctx(base_dir):
        pl_series, taxa_adm_off = load_pl_series()
        lft_series = load_lft_series()

        pl_total_ref = float(pl_series.iloc[-1])

        bf = load_basefundos()
        if migrate_naming:
            bf = _migrate_basefundos_names(bf, verbose=True)

        df_pnl, _, df_despesas, financeiro_ntnb = analisar_dados_fundos2(
            soma_pl_sem_pesos=pl_total_ref,
            df_b3_fechamento=load_b3_prices(),
            df_ajuste=load_ajustes(),
            basefundos=bf,
        )

        serie_ntnb = pd.Series(financeiro_ntnb, dtype=float)
        if len(serie_ntnb):
            serie_ntnb.index = pd.to_datetime(serie_ntnb.index)

        pnl = (df_pnl.drop(columns="Total", errors="ignore")
               .apply(pd.to_numeric, errors="coerce")
               .sum(axis=0).replace([np.inf, -np.inf], np.nan).dropna())
        pnl.index = pd.to_datetime(pnl.index)
        pnl = pnl.sort_index()

        if "Despesas" in df_despesas.index:
            desp_series = df_despesas.loc["Despesas"].rename("desp_op")
            desp_series.index = pd.to_datetime(desp_series.index)
            desp_series = desp_series.sort_index()
        else:
            desp_series = pd.Series(dtype=float)

        common = (pnl.index
                  .intersection(pl_series.index)
                  .intersection(lft_series.index))

        d_ini = pd.to_datetime(data_ini) if data_ini is not None else pl_series.index.min()
        d_fim = pd.to_datetime(data_fim) if data_fim is not None else pl_series.index.max()
        if d_ini > d_fim:
            raise ValueError(f"data_ini ({d_ini}) > data_fim ({d_fim})")

        mask = (common >= d_ini) & (common <= d_fim)
        common = common[mask]
        if len(common) == 0:
            raise ValueError("Interseção vazia.")

        pnl_c        = pnl.loc[common]
        pl_series_c  = pl_series.loc[common]
        lft_series_c = lft_series.loc[common]
        taxa_adm_off_c = taxa_adm_off.reindex(common).fillna(0.0)
        # SEM ffill: se CDI faltar para alguma data, mantem NaN — sinaliza falha visivel
        cdi_full = load_cdi_series()
        cdi_series_c = cdi_full.reindex(common)
        n_missing = cdi_series_c.isna().sum()
        if n_missing > 0:
            faltantes = cdi_series_c[cdi_series_c.isna()].index[:5].tolist()
            print(f"[core] AVISO: CDI FALTANDO em {n_missing} dias. Primeiros: {faltantes}. Rode atualizar_cdi_lft.py.")
        desp_series_c = desp_series.reindex(common).fillna(0.0) if len(desp_series) else pd.Series(0.0, index=common)

        rate_adm_dia  = (1.02 ** (1 / 252) - 1) if taxa_adm_on else 0.0
        rate_extra_dia = ((1 + custo_pct_aa) ** (1 / 252) - 1) if custo_pct_aa else 0.0

        capital_dia = pct * pl_series_c
        if taxa_adm_on:
            custo_adm = capital_dia * rate_adm_dia
        else:
            custo_adm = taxa_adm_off_c
        custo_extra = capital_dia * rate_extra_dia
        custo_fixo  = pd.Series(custo_fixo_rs, index=capital_dia.index)
        custo_total_sem_perf = custo_adm + custo_extra + custo_fixo + desp_series_c

        serie_ntnb_c = serie_ntnb.reindex(common).fillna(0) if len(serie_ntnb) else pd.Series(0.0, index=common)
        capital_ajustado = capital_dia - serie_ntnb_c

        ganho_lft = capital_ajustado * lft_series_c
        ganho_total_pre_perf = pnl_c + ganho_lft - custo_total_sem_perf

        capital_ini_dia = capital_dia.shift(1)
        capital_ini_dia.iloc[0] = float(capital_dia.iloc[0])

        ret_preperf = ganho_total_pre_perf / capital_ini_dia

        if perf_on and perf_pct > 0:
            perf_fee = []
            prov_acum = 0.0
            for d in common:
                excess_day = ret_preperf.loc[d] - cdi_series_c.loc[d]
                base_r = capital_ini_dia.loc[d]
                if excess_day > 0:
                    fee_day = perf_pct * excess_day * base_r
                    prov_acum += fee_day
                    perf_fee.append(fee_day)
                else:
                    estorno_teorico = perf_pct * (-excess_day) * base_r
                    release = min(prov_acum, estorno_teorico)
                    prov_acum -= release
                    perf_fee.append(-release)
            perf_fee = pd.Series(perf_fee, index=common, name="perf_fee$")
        else:
            perf_fee = pd.Series(0.0, index=common, name="perf_fee$")

        custo_total = custo_total_sem_perf + perf_fee
        ganho_total = pnl_c + ganho_lft - custo_total
        ret_total   = ganho_total / capital_ini_dia
        cota        = (1 + ret_total).cumprod()

        ret_acum = float(cota.iloc[-1] - 1)
        vol_anual = float(ret_total.std() * np.sqrt(252)) if len(ret_total) >= 2 else np.nan
        max_dd = float((cota / cota.cummax() - 1).min()) if len(cota) else np.nan
        excesso = (ret_total - cdi_series_c).dropna()
        sharpe_cdi = float((excesso.mean() / excesso.std()) * np.sqrt(252)) if len(excesso) >= 5 and excesso.std() > 0 else np.nan

        df_out = pd.DataFrame({
            "pl_series":        pl_series_c,
            "capital_dia":      capital_dia,
            "capital_ini_dia":  capital_ini_dia,
            "pnl":              pnl_c,
            "serie_ntnb":       serie_ntnb_c,
            "capital_ajustado": capital_ajustado,
            "ganho_lft":        ganho_lft,
            "custo_total":      custo_total,
            "ganho_total":      ganho_total,
            "ret_total":        ret_total,
            "cota":             cota,
            "cdi_series":       cdi_series_c,
            "lft_series":       lft_series_c,
        })

        return {
            "df": df_out, "pct": float(pct),
            "data_ini": common[0], "data_fim": common[-1],
            "ret_acum": ret_acum, "vol_anual": vol_anual,
            "max_dd": max_dd, "sharpe_cdi": sharpe_cdi,
        }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--pct", type=float, default=0.01)
    ap.add_argument("--no-migrate", action="store_true")
    args = ap.parse_args()

    res = computar_cota_serie(pct=args.pct, base_dir=args.base_dir,
                              migrate_naming=not args.no_migrate)
    df = res["df"]
    print("-" * 78)
    print(f"Periodo: {res['data_ini'].date()}  ->  {res['data_fim'].date()}  ({len(df)} dias)")
    print(f"pct                = {res['pct']*100:.2f}%")
    print(f"Retorno acumulado  = {res['ret_acum']*100:+.2f}%")
    print(f"Vol anual          = {res['vol_anual']*100:.2f}%")
    print(f"Max DD             = {res['max_dd']*100:.2f}%")
    print(f"Sharpe (vs CDI)    = {res['sharpe_cdi']:.2f}")
    print(f"Cota (base 1):  primeira={df['cota'].iloc[0]:.6f}   ultima={df['cota'].iloc[-1]:.6f}")
    print("-" * 78)
    print(df[["cota", "ret_total", "pnl", "ganho_lft", "custo_total"]].tail())
