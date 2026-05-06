"""
Script de migração ÚNICA: lê todos os arquivos em BaseFundos/*.parquet
e popula a tabela posicoes_por_fundo no Supabase.

USO:
    1. Garanta que SUPABASE_URL e SUPABASE_KEY estejam disponíveis no
       ambiente (.env, st.secrets equivalente local, ou export).
    2. Execute uma única vez no MESMO ambiente onde os parquets de
       BaseFundos/ estão atualizados.
    3. Confira no Supabase que a tabela foi populada.

Importante:
- Upsert idempotente baseado em (Fundo, Ativo, Data) -> pode rodar várias vezes.
- Linhas totalmente vazias são puladas.
- NaN/Inf que escapam do round-trip DataFrame->dict são limpos antes do envio.
"""

from __future__ import annotations

import os
import sys
import math
import pandas as pd
from pathlib import Path

try:
    from supabase import create_client
except ImportError:
    print("ERRO: pacote 'supabase' não está instalado. Rode: pip install supabase", file=sys.stderr)
    sys.exit(1)


# ----------------------- Configuração -----------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print(
        "ERRO: defina as variáveis de ambiente SUPABASE_URL e SUPABASE_KEY (ou SUPABASE_SERVICE_ROLE_KEY).",
        file=sys.stderr,
    )
    sys.exit(1)

BASEFUNDOS_DIR = Path(os.environ.get("BASEFUNDOS_DIR", "BaseFundos"))
TABLE = "posicoes_por_fundo"
BATCH_SIZE = 500


def _is_nanlike(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _to_float_or_none(v):
    if _is_nanlike(v):
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def parse_basefundos_to_long(diretorio: Path) -> pd.DataFrame:
    """Converte BaseFundos/<fundo>.parquet (formato wide) num DataFrame longo:
    Fundo, Ativo, Data, Quantidade, Preco_Compra, Preco_Fechamento, PL, Rendimento.
    """
    if not diretorio.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {diretorio.resolve()}")

    METRICAS = ["PL", "Preco_Fechamento", "Preco_Compra", "Quantidade", "Rendimento"]
    linhas = []

    for arquivo in sorted(diretorio.iterdir()):
        if arquivo.suffix.lower() != ".parquet":
            continue
        fundo = arquivo.stem
        if fundo.upper() == "TOTAL":
            print(f"  - {fundo}: pulando (fundo agregado 'Total')")
            continue

        print(f"  - lendo {arquivo.name} ...")
        df = pd.read_parquet(arquivo)

        if "Ativo" not in df.columns:
            if df.index.name == "Ativo":
                df = df.reset_index()
            else:
                print(f"    AVISO: {arquivo.name} sem coluna 'Ativo'. Pulando.")
                continue

        datas = set()
        for col in df.columns:
            for m in METRICAS:
                sufixo = f" - {m}"
                if str(col).endswith(sufixo):
                    data_str = str(col)[: -len(sufixo)].strip()
                    datas.add(data_str)
                    break

        for _, row in df.iterrows():
            ativo = row["Ativo"]
            if _is_nanlike(ativo):
                continue
            ativo = str(ativo)

            for data_str in datas:
                try:
                    data_dt = pd.to_datetime(data_str, errors="raise").date()
                except Exception:
                    continue

                payload = {
                    "Fundo": fundo,
                    "Ativo": ativo,
                    "Data": data_dt.isoformat(),
                    "Quantidade":       _to_float_or_none(row.get(f"{data_str} - Quantidade")),
                    "Preco_Compra":     _to_float_or_none(row.get(f"{data_str} - Preco_Compra")),
                    "Preco_Fechamento": _to_float_or_none(row.get(f"{data_str} - Preco_Fechamento")),
                    "PL":               _to_float_or_none(row.get(f"{data_str} - PL")),
                    "Rendimento":       _to_float_or_none(row.get(f"{data_str} - Rendimento")),
                }

                if (
                    (payload["Quantidade"] in (None, 0))
                    and payload["Preco_Compra"] is None
                    and payload["Preco_Fechamento"] is None
                    and payload["PL"] is None
                    and payload["Rendimento"] is None
                ):
                    continue

                if payload["Quantidade"] is None:
                    payload["Quantidade"] = 0.0

                linhas.append(payload)

    if not linhas:
        return pd.DataFrame(columns=["Fundo", "Ativo", "Data"])

    long_df = pd.DataFrame(linhas)
    long_df = long_df.drop_duplicates(
        subset=["Fundo", "Ativo", "Data"], keep="last"
    )
    return long_df


def _scrub_nan_in_records(registros: list[dict]) -> list[dict]:
    """Substitui NaN/Inf por None nos dicts. Necessário porque o round-trip
    DataFrame -> to_dict reintroduz NaN onde tinha None (em colunas numéricas)."""
    saida = []
    for r in registros:
        novo = {}
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                novo[k] = None
            elif v is pd.NaT:
                novo[k] = None
            else:
                # numpy.float64 também passa em isinstance(float) mas só por garantia
                try:
                    if pd.isna(v):
                        novo[k] = None
                        continue
                except (TypeError, ValueError):
                    pass
                novo[k] = v
        # Quantidade nunca pode ser NULL no banco
        if novo.get("Quantidade") is None:
            novo["Quantidade"] = 0.0
        saida.append(novo)
    return saida


def upsert_em_lotes(supabase, registros: list[dict], batch_size: int = BATCH_SIZE) -> int:
    registros = _scrub_nan_in_records(registros)
    total = 0
    for i in range(0, len(registros), batch_size):
        lote = registros[i : i + batch_size]
        resp = (
            supabase.table(TABLE)
            .upsert(lote, on_conflict="Fundo,Ativo,Data")
            .execute()
        )
        if getattr(resp, "data", None) is None:
            raise RuntimeError(f"Falha no upsert do lote {i // batch_size}: {resp}")
        total += len(lote)
        print(f"    lote {i // batch_size + 1}: {len(lote)} linhas (total acumulado: {total})")
    return total


def main():
    print(f"Diretório fonte: {BASEFUNDOS_DIR.resolve()}")
    print("Lendo parquets ...")
    long_df = parse_basefundos_to_long(BASEFUNDOS_DIR)
    print(f"Total de linhas a enviar: {len(long_df)}")

    if long_df.empty:
        print("Nada para migrar. Verifique se BASEFUNDOS_DIR está apontando para a pasta certa.")
        return

    registros = long_df.to_dict(orient="records")

    print("Conectando no Supabase ...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    print(f"Iniciando upsert em lotes de {BATCH_SIZE} ...")
    total = upsert_em_lotes(supabase, registros)
    print(f"Migração concluída. Linhas enviadas: {total}")

    print("\nDicas de verificação no SQL Editor do Supabase:")
    print('  select count(*) from posicoes_por_fundo;')
    print('  select "Fundo", sum("Quantidade")::int as qtd')
    print('  from posicoes_por_fundo where "Ativo" = \'DI_29\' group by 1 order by 1;')


if __name__ == "__main__":
    main()
