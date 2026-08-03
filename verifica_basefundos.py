"""
verifica_basefundos.py
=======================
Lê cada parquet de BaseFundos/ e imprime os valores da coluna 'Ativo'.
Também mostra o que load_basefundos() (do cota_portfolio_core) enxerga.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cota_portfolio_core import load_basefundos

BF = Path("BaseFundos")
print(f"CWD: {os.getcwd()}")
print(f"BaseFundos existe? {BF.exists()}  is_dir? {BF.is_dir()}")
print(f"Arquivos em {BF}:")
for p in sorted(BF.iterdir()):
    stat = p.stat()
    marker = " <BKP>" if p.is_dir() else ""
    print(f"  {p.name:<40s}  size={stat.st_size:>8}  mtime={pd.Timestamp(stat.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')}{marker}")

print()
print("=" * 88)
print("LEITURA DIRETA DE CADA PARQUET (pd.read_parquet -> coluna 'Ativo')")
print("=" * 88)

for p in sorted(BF.iterdir()):
    if not p.is_file() or p.suffix.lower() != ".parquet":
        continue
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"[erro] {p.name}: {e}")
        continue
    if "Ativo" not in df.columns:
        print(f"[warn] {p.name}: sem coluna 'Ativo'; colunas={list(df.columns)[:5]}...")
        continue
    ativos = df["Ativo"].astype(str).tolist()
    print(f"{p.name:<35s}  ativos={ativos}")

print()
print("=" * 88)
print("O QUE load_basefundos() ENXERGA (via cota_portfolio_core)")
print("=" * 88)

bf = load_basefundos()
for fundo, dff in bf.items():
    print(f"  {fundo:<35s}  index={list(dff.index)}")
