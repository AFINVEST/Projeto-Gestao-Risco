"""
Migração de naming dos ativos DI/DAP em TODOS os arquivos parquet
locais que o app4.py consome.

REGRA:
    DI_YY  → DI_F<YY>
    DAPYY  → DAP_Q<YY>  (YY par)
    DAPYY  → DAP_K<YY>  (YY ímpar)

Arquivos afetados:
    Dados/df_inicial.parquet                          — colunas (universo)
    Dados/df_divone.parquet                           — colunas (DV01)
    Dados/df_preco_de_ajuste_atual_completo.parquet   — linhas em 'Assets'
    Dados/df_valor_ajuste_contrato.parquet            — linhas em 'Assets'
    Dados/df_ajustes_b3.parquet                       — linhas (se existir)
    Dados/portifolio_posições.parquet                 — coluna 'Ativo'
    BaseFundos/*.parquet                              — coluna/index 'Ativo'

Comportamento:
    - Idempotente (pode rodar 2x)
    - Cria backup antes: Dados_backup_pre_naming/  e  BaseFundos_backup_pre_naming/
    - Se um arquivo não existir, pula silenciosamente
    - Log detalhado no console

USO:
    Rode DENTRO da pasta do projeto (onde Dados/ e BaseFundos/ existem).
    python 03_migrar_naming_parquets.py

REQUISITO: pip install pandas pyarrow
"""

from __future__ import annotations
import shutil
import sys
from pathlib import Path
import re
import pandas as pd


ROOT = Path(".")
DADOS_DIR = ROOT / "Dados"
BASEFUNDOS_DIR = ROOT / "BaseFundos"
BKP_DADOS = ROOT / "Dados_backup_pre_naming"
BKP_BASEFUNDOS = ROOT / "BaseFundos_backup_pre_naming"


RE_DI_ANTIGO  = re.compile(r"^DI_(\d{2})$")
RE_DAP_ANTIGO = re.compile(r"^DAP(\d{2})$")


def novo_nome(s: str) -> str:
    """Aplica o mapeamento. Retorna a string original se não bater com o padrão antigo."""
    if not isinstance(s, str):
        return s
    m = RE_DI_ANTIGO.match(s)
    if m:
        return f"DI_F{m.group(1)}"
    m = RE_DAP_ANTIGO.match(s)
    if m:
        yy = int(m.group(1))
        letra = "Q" if (yy % 2) == 0 else "K"
        return f"DAP_{letra}{m.group(1)}"
    return s


def backup(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        print(f"  [backup] já existe {dst}, pulando cópia")
        return
    print(f"  [backup] copiando {src} → {dst}")
    shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)


def renomear_coluna(df: pd.DataFrame, coluna: str) -> tuple[pd.DataFrame, int]:
    """Renomeia valores na coluna. Retorna (df, num_renamed)."""
    if coluna not in df.columns:
        return df, 0
    mask = df[coluna].apply(lambda x: isinstance(x, str) and (RE_DI_ANTIGO.match(x) or RE_DAP_ANTIGO.match(x)) is not None)
    n = int(mask.sum())
    if n > 0:
        df[coluna] = df[coluna].apply(novo_nome)
    return df, n


def renomear_index(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Renomeia valores no index. Retorna (df, num_renamed)."""
    idx_novo = [novo_nome(x) for x in df.index]
    n = sum(1 for a, b in zip(df.index, idx_novo) if a != b)
    if n > 0:
        df.index = idx_novo
    return df, n


def renomear_colunas_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Renomeia nomes de colunas (para arquivos onde ativo é coluna, ex: df_inicial)."""
    cols_novas = [novo_nome(c) for c in df.columns]
    n = sum(1 for a, b in zip(df.columns, cols_novas) if a != b)
    if n > 0:
        df.columns = cols_novas
    return df, n


def processar_wide(path: Path):
    """Ativos como colunas (df_inicial, df_divone)."""
    if not path.exists():
        print(f"  [skip] não existe: {path}")
        return
    print(f"  [processando] {path}")
    df = pd.read_parquet(path)
    df, n = renomear_colunas_wide(df)
    print(f"    colunas renomeadas: {n}")
    if n > 0:
        df.to_parquet(path, index=False if df.index.name is None else True)


def processar_longo(path: Path, coluna: str = "Assets"):
    """Ativo em coluna nomeada 'Assets' ou 'Ativo'."""
    if not path.exists():
        print(f"  [skip] não existe: {path}")
        return
    print(f"  [processando] {path}")
    df = pd.read_parquet(path)
    if coluna not in df.columns:
        # tenta 'Ativo' como fallback
        alternativa = "Ativo" if coluna != "Ativo" else "Assets"
        if alternativa in df.columns:
            coluna = alternativa
        else:
            print(f"    aviso: coluna '{coluna}' nem '{alternativa}' encontrada. Colunas: {list(df.columns)[:5]}...")
            return
    df, n = renomear_coluna(df, coluna)
    print(f"    linhas renomeadas em coluna '{coluna}': {n}")
    if n > 0:
        df.to_parquet(path, index=False)


def processar_basefundos(dir_path: Path):
    """Cada parquet tem coluna/index 'Ativo'."""
    if not dir_path.exists():
        print(f"  [skip] não existe: {dir_path}")
        return
    for parquet in sorted(dir_path.iterdir()):
        if parquet.suffix.lower() != ".parquet":
            continue
        print(f"  [processando] {parquet}")
        df = pd.read_parquet(parquet)
        n_tot = 0
        if "Ativo" in df.columns:
            df, n = renomear_coluna(df, "Ativo")
            n_tot += n
        if df.index.name == "Ativo":
            df, n = renomear_index(df)
            n_tot += n
        elif "Ativo" not in df.columns:
            # tenta index sem name
            df, n = renomear_index(df)
            n_tot += n
        print(f"    renomeados: {n_tot}")
        if n_tot > 0:
            df.to_parquet(parquet, index=(df.index.name is not None))


def main():
    print("=" * 60)
    print("MIGRAÇÃO DE NAMING — DI_YY→DI_F<YY> / DAPYY→DAP_K/Q<YY>")
    print("=" * 60)

    if not DADOS_DIR.exists():
        print(f"ERRO: não achei {DADOS_DIR.resolve()}. Rode este script na raiz do projeto.", file=sys.stderr)
        sys.exit(1)

    print(f"\nBackup dos arquivos originais...")
    backup(DADOS_DIR, BKP_DADOS)
    backup(BASEFUNDOS_DIR, BKP_BASEFUNDOS)

    print(f"\n--- Dados/*.parquet (formato wide: ativos como colunas) ---")
    processar_wide(DADOS_DIR / "df_inicial.parquet")
    processar_wide(DADOS_DIR / "df_divone.parquet")

    print(f"\n--- Dados/*.parquet (formato longo: ativos em coluna 'Assets') ---")
    processar_longo(DADOS_DIR / "df_preco_de_ajuste_atual_completo.parquet", "Assets")
    processar_longo(DADOS_DIR / "df_valor_ajuste_contrato.parquet",           "Assets")
    processar_longo(DADOS_DIR / "df_ajustes_b3.parquet",                      "Instrumento")

    print(f"\n--- Dados/portifolio_posições.parquet (coluna 'Ativo') ---")
    processar_longo(DADOS_DIR / "portifolio_posições.parquet", "Ativo")

    print(f"\n--- BaseFundos/*.parquet ---")
    processar_basefundos(BASEFUNDOS_DIR)

    print("\n" + "=" * 60)
    print("Concluído.")
    print(f"Backups em: {BKP_DADOS} e {BKP_BASEFUNDOS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
