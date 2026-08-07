"""
patch_scrapb3_backfill.py
==========================

Corrige bug de backfill do ScrapB3_v2.py:
  start_dt = max(candidates)   →   start_dt = min(candidates)

Motivo: quando last_preco != last_valor (uma parquet mais atualizada que outra),
usar MAX pula os dias faltantes na parquet mais atrasada. MIN garante que
todos os dias pendentes sao puxados.

USO:
    python patch_scrapb3_backfill.py           # dry-run
    python patch_scrapb3_backfill.py --apply   # aplica com backup

Rollback: copia de volta ScrapB3_v2.py.bak_YYYYMMDD_HHMMSS
"""
from __future__ import annotations
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


ORIGINAL = """    if last_preco is None and last_valor is None:
        start_dt = dt.date(2025, 1, 2)
    else:
        candidates = [d for d in [last_preco, last_valor] if d is not None]
        start_dt = max(candidates)"""

PATCHED = """    if last_preco is None and last_valor is None:
        start_dt = dt.date(2025, 1, 2)
    else:
        candidates = [d for d in [last_preco, last_valor] if d is not None]
        start_dt = min(candidates)   # patched: min garante backfill quando parquets desalinhadas"""


def processar(app_path: Path, apply: bool = False) -> int:
    if not app_path.exists():
        print(f"[erro] {app_path} nao existe.")
        return 2

    content = app_path.read_text(encoding="utf-8")

    if "patched: min garante backfill" in content:
        print(f"[ok] {app_path.name} ja foi patchado. Nada a fazer.")
        return 0

    if ORIGINAL not in content:
        print(f"[erro] Bloco original nao encontrado em {app_path.name}.")
        print("Bloco procurado:")
        print("---")
        print(ORIGINAL)
        print("---")
        return 3

    novo = content.replace(ORIGINAL, PATCHED)
    print(f"[info] Bloco encontrado. Substitui max por min.")
    print(f"[info] Modo: {'APLICAR' if apply else 'DRY-RUN'}")

    if apply:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = app_path.with_name(f"{app_path.name}.bak_backfill_{ts}")
        shutil.copy2(app_path, backup)
        print(f"[backup] {backup}")
        app_path.write_text(novo, encoding="utf-8")
        print(f"[ok] {app_path} atualizado.")
    else:
        print("[dry-run] Nada foi escrito.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--file", default="ScrapB3_v2.py")
    args = ap.parse_args()
    sys.exit(processar(Path(args.file), apply=args.apply))
