"""
Helpers de posicoes_por_fundo — para colar dentro de app4.py
============================================================

Estas funções substituem/complementam o fluxo de leitura/escrita
de BaseFundos/*.parquet, tirando essa responsabilidade do
filesystem efêmero do Streamlit Cloud e jogando tudo no Supabase.

Cole o bloco inteiro abaixo dentro de app4.py, logo depois da
definição de `def load_data(): ...` (perto da linha 3897). NÃO
remova `load_data` — só adicione estas funções junto.

Ao final, há um pequeno bloco "PATCH POINTS" listando as edições
pontuais que você precisa fazer em outras partes do app4.py.
"""

# ====== INÍCIO DO BLOCO PARA COLAR EM app4.py =================
import math
import pandas as pd
import numpy as np


_TABELA_PPF = "posicoes_por_fundo"


def _ppf_safe_float(v):
    """Converte para float ou None (NaN/Inf -> None)."""
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


def load_posicoes_por_fundo() -> pd.DataFrame:
    """Lê TODA a tabela posicoes_por_fundo do Supabase.
    Retorna DataFrame longo com colunas:
        Fundo, Ativo, Data, Quantidade, Preco_Compra, Preco_Fechamento, PL, Rendimento
    Em caso de falha, retorna DataFrame vazio (caller decide o fallback).
    """
    try:
        # Paginação simples para tabelas grandes (Supabase limita ~1000 por chamada)
        rows: list[dict] = []
        offset = 0
        page = 1000
        while True:
            resp = (
                supabase.table(_TABELA_PPF)
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
        df = pd.DataFrame(rows)
        # Tipagem
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        for col in ("Quantidade", "Preco_Compra", "Preco_Fechamento", "PL", "Rendimento"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        print(f"[posicoes_por_fundo] erro de leitura: {e}")
        return pd.DataFrame()


def upsert_posicoes_por_fundo(registros: list[dict]) -> bool:
    """Upsert idempotente em lotes; chave única (Fundo, Ativo, Data).
    `registros` deve ser lista de dicts com pelo menos
    Fundo, Ativo, Data, Quantidade.
    """
    if not registros:
        return True

    # Sanitiza payload: garante tipos JSON-compatíveis
    payload: list[dict] = []
    for r in registros:
        try:
            data_v = r["Data"]
            if isinstance(data_v, (pd.Timestamp,)):
                data_iso = data_v.date().isoformat()
            elif isinstance(data_v, str):
                data_iso = pd.to_datetime(data_v).date().isoformat()
            else:
                data_iso = pd.to_datetime(data_v).date().isoformat()
        except Exception:
            continue

        payload.append({
            "Fundo": str(r["Fundo"]),
            "Ativo": str(r["Ativo"]),
            "Data":  data_iso,
            "Quantidade":       _ppf_safe_float(r.get("Quantidade")) or 0.0,
            "Preco_Compra":     _ppf_safe_float(r.get("Preco_Compra")),
            "Preco_Fechamento": _ppf_safe_float(r.get("Preco_Fechamento")),
            "PL":               _ppf_safe_float(r.get("PL")),
            "Rendimento":       _ppf_safe_float(r.get("Rendimento")),
        })

    try:
        BATCH = 500
        for i in range(0, len(payload), BATCH):
            lote = payload[i : i + BATCH]
            (supabase.table(_TABELA_PPF)
                     .upsert(lote, on_conflict="Fundo,Ativo,Data")
                     .execute())
        # Invalida caches dependentes
        invalidar_caches_posicoes()
        return True
    except Exception as e:
        print(f"[posicoes_por_fundo] erro de upsert: {e}")
        try:
            st.error(f"Erro ao gravar posicoes_por_fundo: {e}")
        except Exception:
            pass
        return False


def delete_posicao_por_fundo(ativo: str, data: str, fundo: str | None = None) -> bool:
    """Apaga linha(s) da tabela.
    - Se `fundo` for fornecido, apaga só naquele fundo.
    - Caso contrário, apaga todas as linhas (Ativo, Data) em todos os fundos.
    """
    try:
        data_iso = pd.to_datetime(data).date().isoformat()
        q = supabase.table(_TABELA_PPF).delete().eq("Ativo", ativo).eq("Data", data_iso)
        if fundo is not None:
            q = q.eq("Fundo", fundo)
        q.execute()
        invalidar_caches_posicoes()
        return True
    except Exception as e:
        print(f"[posicoes_por_fundo] erro de delete: {e}")
        return False


def read_atual_contratos_supabase() -> pd.DataFrame:
    """Equivalente ao antigo `read_atual_contratos()`, porém lendo
    do Supabase. Retorna DataFrame com Fundo no índice, Ativo nas
    colunas, e a soma da Quantidade nos valores. Linhas zeradas
    são preservadas (fillna(0)).
    """
    df = load_posicoes_por_fundo()
    if df is None or df.empty:
        # Fallback: lê parquets locais (backwards-compat enquanto migra)
        try:
            return read_atual_contratos()  # função original que lê BaseFundos
        except Exception:
            return pd.DataFrame()

    # Soma Quantidade por (Fundo, Ativo)
    agrupado = (df.groupby(["Fundo", "Ativo"], as_index=False)["Quantidade"]
                  .sum())
    pivot = agrupado.pivot(index="Fundo", columns="Ativo", values="Quantidade").fillna(0)
    return pivot


def load_basefundos_supabase() -> dict[str, pd.DataFrame]:
    """Equivalente ao antigo `load_basefundos()`, devolvendo
    um dict {fundo: DataFrame_wide} compatível com o restante do
    código (ex.: `analisar_dados_fundos2` espera colunas tipo
    '<data> - Quantidade'). Reconstrói o formato wide a partir do
    formato longo do Supabase.
    """
    df = load_posicoes_por_fundo()
    if df is None or df.empty:
        # Fallback: lê parquets locais
        try:
            return load_basefundos()  # função original
        except Exception:
            return {}

    out: dict[str, pd.DataFrame] = {}
    df["Data"] = pd.to_datetime(df["Data"]).dt.strftime("%Y-%m-%d")

    for fundo, sub in df.groupby("Fundo"):
        sub = sub.copy()

        # monta as colunas wide
        partes = []
        for metrica in ("PL", "Preco_Fechamento", "Preco_Compra", "Quantidade", "Rendimento"):
            if metrica not in sub.columns:
                continue
            piv = sub.pivot_table(
                index="Ativo", columns="Data", values=metrica, aggfunc="last"
            )
            piv.columns = [f"{c} - {metrica}" for c in piv.columns]
            partes.append(piv)

        if not partes:
            continue
        wide = pd.concat(partes, axis=1).sort_index(axis=1)

        # Mantém compat com o resto do código que faz `set_index('Ativo')`
        wide = wide.reset_index()  # 'Ativo' vira coluna
        out[fundo] = wide

    return out


def invalidar_caches_posicoes() -> None:
    """Limpa caches do Streamlit que dependem das posições, e força
    o próximo recálculo do bundle de risco."""
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    try:
        st.session_state.pop("_risk_bundle", None)
        st.session_state.pop("df_precos_ajustados_base", None)
    except Exception:
        pass


# ====== FIM DO BLOCO PARA COLAR EM app4.py ====================


# =====================================================================
# PATCH POINTS — edições pontuais a fazer em app4.py
# =====================================================================
#
# Veja o arquivo 04_app4_edits.md para o passo a passo com before/after.
# Resumo:
#   1) Trocar `read_atual_contratos_cached()` (linha ~5152) para devolver
#      `read_atual_contratos_supabase()` em vez do read antigo. Trocar
#      ttl=60 para ter um teto de segurança.
#   2) Trocar `load_basefundos()` (linha ~5186) similarmente.
#   3) Em `add_data()` (linha ~3908), `delete_data()` e `update_data()`,
#      chamar `invalidar_caches_posicoes()` antes do return.
#   4) Em `update_base_fundos()` (linha ~3780), depois de mexer no
#      parquet, espelhar a remoção: `delete_posicao_por_fundo(ativo, dia)`.
#   5) Em `atualizar_parquet_fundos()` (linha ~3979), no fim do loop por
#      (fundo, asset) coletar um dict com a foto daquele dia e enviar
#      via `upsert_posicoes_por_fundo([...])`.
#   6) Em `get_risk_static_bundle()` (linha ~6545), apagar as linhas:
#         if '_risk_bundle' in st.session_state:
#             return st.session_state['_risk_bundle']
#      e mudar a chave de cache para incluir um max(Id) do Supabase.
#
# =====================================================================
