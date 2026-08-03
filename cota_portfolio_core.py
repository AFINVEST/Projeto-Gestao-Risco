"""
cota_portfolio_core.py  (v1.1 — com alias de naming antigo -> novo)
===================================================================

Módulo standalone (SEM streamlit) que replica EXATAMENTE o cálculo de
`simulate_nav_cota()` do app4.py.

MUDANÇA v1.1:
  - Adicionado `_alias_ativo_novo` + `_migrate_basefundos_names` que traduz
    DAP30 -> DAP_Q30 / DAP35 -> DAP_K35 / DI_27 -> DI_F27 (idempotente).
  - `computar_cota_serie` chama a migração in-memory antes do
    `analisar_dados_fundos2`, garantindo que os ativos do BaseFundos com
    naming antigo encontrem os preços/ajustes populados sob o naming novo.

Uso típico:
    from cota_portfolio_core import computar_cota_serie
    res = computar_cota_serie(pct=0.01, base_dir=r"Z:\...")
    df  = res["df"]
    print(df.tail())

Requisitos: pandas, numpy, pyarrow. NÃO importa streamlit em nenhum lugar.
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


# ─────────────────────────────────────────────────────────────────────────────
# Alias de naming: antigo -> novo (idempotente)
#   DAP{YY} ano par   -> DAP_Q{YY}   (Q = agosto)
#   DAP{YY} ano impar -> DAP_K{YY}   (K = maio)
#   DI_{YY}           -> DI_F{YY}    (F = janeiro)
#   Nomes já novos (DAP_K27, DI_F27), NTNB*, TREASURY, WDO1 — passam direto.
# ─────────────────────────────────────────────────────────────────────────────
_RE_DAP_NOVO    = re.compile(r"^DAP_[KQ]\d{2}$")
_RE_DAP_ANTIGO  = re.compile(r"^DAP(\d{2})$")
_RE_DI_NOVO     = re.compile(r"^DI_[FJNV]\d{2}$")
_RE_DI_ANTIGO   = re.compile(r"^DI_(\d{2})$")


def _alias_ativo_novo(ativo: str) -> str:
    """Traduz naming antigo -> novo. Idempotente para nomes já novos."""
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
    """Aplica alias novo aos indices de Ativo de cada fundo do BaseFundos.

    Retorna um NOVO dict com DataFrames re-indexados. Não modifica o original.
    """
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
# Utilitário: chdir contextual (para arquivos relativos "Dados/…", "BaseFundos/…")
# ─────────────────────────────────────────────────────────────────────────────

class _chdir_ctx:
    """chdir temporário — restaura no exit."""
    def __init__(self, novo: Optional[str | Path]):
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
# Loaders — cópias literais do app4.py (sem @st.cache_data)
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


def load_basefundos() -> dict[str, pd.DataFrame]:
    out = {}
    for f in os.listdir("BaseFundos"):
        if not f.lower().endswith(".parquet"):
            continue
        nome = f.rsplit(".", 1)[0]
        df = pd.read_parquet(f"BaseFundos/{f}").set_index("Ativo")
        out[nome] = df
    return out


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
    """Lê o CDI cacheado em CSV; se não existir, cai para LFT como proxy."""
    p = Path(cache_csv)
    if not p.exists():
        return load_lft_series().rename("cdi_proxy")
    s = (pd.read_csv(cache_csv, parse_dates=["Data"])
            .set_index("Data")["cdi"]
            .astype(float)
            .sort_index())
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
    assert "Assets" in df_b3.columns, "df_b3_fechamento precisa ter coluna 'Assets'."

    date_cols_b3 = pd.to_datetime(df_b3.columns[1:], errors="coerce")
    df_b3.columns = ["Assets"] + list(date_cols_b3)
    df_b3 = df_b3.loc[:, ["Assets"] + sorted([c for c in df_b3.columns[1:] if pd.notna(c)])]

    prices_only = df_b3.columns[1:]
    if df_b3[prices_only].select_dtypes(include="object").size > 0:
        df_b3[prices_only] = (df_b3[prices_only]
            .replace({r"\.": "", ",": "."}, regex=True)
            .apply(pd.to_numeric, errors="coerce"))

    preco_lookup = (df_b3.set_index("Assets")
                        .sort_index(axis=1)
                        .ffill(axis=1))

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
                if qtd == 0:
                    continue

                data_op = pd.to_datetime(col_q.split()[0], errors="coerce")
                if pd.isna(data_op):
                    continue

                p_compra = linha.get(col_q.replace("Quantidade", "Preco_Compra"), np.nan)
                pl_op    = linha.get(col_q.replace("Quantidade", "PL"), np.nan)

                if fundo.upper() != "TOTAL":
                    raiz = ativo.split("_")[0]
                    if raiz.startswith(("DAP", "DI")):
                        raiz = raiz[:3]
                    if raiz == "DI":
                        try:
                            PU_atual = float(preco_lookup.at[ativo, data_op])
                        except Exception:
                            PU_atual = float(preco_lookup.at[ativo, preco_lookup.columns[-1]])
                        custo_op = (PU_FINAL - PU_atual) * 0.03 * abs(qtd) * 0.005
                    elif raiz == "WDO1":
                        try:
                            PU_atual = float(preco_lookup.at[ativo, data_op])
                        except Exception:
                            PU_atual = float(preco_lookup.at[ativo, preco_lookup.columns[-1]])
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
                    if data_fech < data_op:
                        continue

                    p_fech = preco_lookup.get(data_fech, pd.Series()).get(ativo, np.nan)
                    if not np.isfinite(p_fech):
                        continue

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
# NÚCLEO: computar_cota_serie — replica simulate_nav_cota EXATAMENTE
# ─────────────────────────────────────────────────────────────────────────────

def computar_cota_serie(
    pct: float = 0.01,
    data_ini: Optional[str | pd.Timestamp] = None,
    data_fim: Optional[str | pd.Timestamp] = None,
    taxa_adm_on: bool = False,
    custo_pct_aa: float = 0.0,
    custo_fixo_rs: float = 0.0,
    perf_on: bool = False,
    perf_pct: float = 0.20,
    base_dir: Optional[str | Path] = None,
    migrate_naming: bool = True,
) -> dict:
    """Replica EXATAMENTE `simulate_nav_cota()` do app4.py."""
    with _chdir_ctx(base_dir):
        pl_series, taxa_adm_off = load_pl_series()
        lft_series = load_lft_series()

        pl_total_ref = float(pl_series.iloc[-1])

        # Carrega BaseFundos e (opcionalmente) migra naming antigo -> novo
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

        pnl = (df_pnl
               .drop(columns="Total", errors="ignore")
               .apply(pd.to_numeric, errors="coerce")
               .sum(axis=0)
               .replace([np.inf, -np.inf], np.nan)
               .dropna())
        pnl.index = pd.to_datetime(pnl.index)
        pnl = pnl.sort_index()

        if "Despesas" in df_despesas.index:
            desp_series = (df_despesas.loc["Despesas"].rename("desp_op"))
            desp_series.index = pd.to_datetime(desp_series.index)
            desp_series = desp_series.sort_index()
        else:
            desp_series = pd.Series(dtype=float)

        common = (
            pnl.index
            .intersection(pl_series.index)
            .intersection(lft_series.index)
        )

        data_min = pl_series.index.min()
        data_max = pl_series.index.max()
        d_ini = pd.to_datetime(data_ini) if data_ini is not None else data_min
        d_fim = pd.to_datetime(data_fim) if data_fim is not None else data_max
        if d_ini > d_fim:
            raise ValueError(f"data_ini ({d_ini}) > data_fim ({d_fim})")

        mask = (common >= d_ini) & (common <= d_fim)
        common = common[mask]
        if len(common) == 0:
            raise ValueError("Interseção vazia entre pnl, pl_series e lft_series.")

        pnl_c        = pnl.loc[common]
        pl_series_c  = pl_series.loc[common]
        lft_series_c = lft_series.loc[common]
        taxa_adm_off_c = taxa_adm_off.reindex(common).fillna(0.0)
        cdi_series_c = load_cdi_series().reindex(common).ffill().fillna(0.0)
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
        if len(excesso) >= 5 and excesso.std() > 0:
            sharpe_cdi = float((excesso.mean() / excesso.std()) * np.sqrt(252))
        else:
            sharpe_cdi = np.nan

        df_out = pd.DataFrame({
            "pl_series":       pl_series_c,
            "capital_dia":     capital_dia,
            "capital_ini_dia": capital_ini_dia,
            "pnl":             pnl_c,
            "serie_ntnb":      serie_ntnb_c,
            "capital_ajustado": capital_ajustado,
            "ganho_lft":       ganho_lft,
            "custo_total":     custo_total,
            "ganho_total":     ganho_total,
            "ret_total":       ret_total,
            "cota":            cota,
            "cdi_series":      cdi_series_c,
            "lft_series":      lft_series_c,
        })

        return {
            "df": df_out,
            "pct": float(pct),
            "data_ini": common[0],
            "data_fim": common[-1],
            "ret_acum": ret_acum,
            "vol_anual": vol_anual,
            "max_dd": max_dd,
            "sharpe_cdi": sharpe_cdi,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI — smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Smoke test do cota_portfolio_core")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--pct", type=float, default=0.01)
    ap.add_argument("--no-migrate", action="store_true",
                    help="Não aplica migracao de naming antigo->novo (para debug).")
    args = ap.parse_args()

    res = computar_cota_serie(pct=args.pct, base_dir=args.base_dir,
                              migrate_naming=not args.no_migrate)
    df = res["df"]
    print("-" * 78)
    print(f"Periodo: {res['data_ini'].date()}  ->  {res['data_fim'].date()}  ({len(df)} dias)")
    print(f"pct                = {res['pct']*100:.2f}%")
    print(f"Retorno acumulado  = {res['ret_acum']*100:+.2f}%")
    print(f"Vol anual (ret_tot)= {res['vol_anual']*100:.2f}%")
    print(f"Max DD             = {res['max_dd']*100:.2f}%")
    print(f"Sharpe (vs CDI)    = {res['sharpe_cdi']:.2f}")
    print(f"Cota (base 1):  primeira={df['cota'].iloc[0]:.6f}   ultima={df['cota'].iloc[-1]:.6f}")
    print("-" * 78)
    print("Ultimas 5 linhas:")
    print(df[["cota", "ret_total", "pnl", "ganho_lft", "custo_total"]].tail())
