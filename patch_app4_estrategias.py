"""
patch_app4_estrategias.py
=========================

Atualiza o dict `estrategias` no app4.py para aceitar AMBOS os namings
(antigo e novo). Faz backup antes de sobrescrever. Idempotente.

USO:
    python patch_app4_estrategias.py               # dry-run
    python patch_app4_estrategias.py --apply

Rollback: copiar de volta app4.py.bak_estrategias_YYYYMMDD_HHMMSS
"""
from __future__ import annotations
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


ORIGINAL = """        estrategias = {
            "Juros nominais": ['DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30', 'DI_31', 'DI_32', 'DI_33', 'DI_35'],
            "Juros reais": ['DAP26', 'DAP27', 'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40', 'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40', 'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60'],
            "Moeda": ['WDO1'],
            "Juros US": ['TREASURY']
        }"""


PATCHED = """        estrategias = {
            "Juros nominais": [
                # naming novo (F=jan)
                'DI_F26', 'DI_F27', 'DI_F28', 'DI_F29', 'DI_F30', 'DI_F31', 'DI_F32', 'DI_F33', 'DI_F35',
                # naming antigo (retrocompat)
                'DI_26', 'DI_27', 'DI_28', 'DI_29', 'DI_30', 'DI_31', 'DI_32', 'DI_33', 'DI_35',
            ],
            "Juros reais": [
                # DAP naming novo — par -> Q, impar -> K
                'DAP_Q26', 'DAP_K27', 'DAP_Q28', 'DAP_K29', 'DAP_Q30', 'DAP_K31', 'DAP_Q32', 'DAP_K33', 'DAP_K35', 'DAP_Q40',
                # DAP naming antigo (retrocompat)
                'DAP26', 'DAP27', 'DAP28', 'DAP30', 'DAP32', 'DAP35', 'DAP40',
                # NTNB (naming inalterado)
                'NTNB26', 'NTNB27', 'NTNB28', 'NTNB30', 'NTNB32', 'NTNB35', 'NTNB40', 'NTNB45', 'NTNB50', 'NTNB55', 'NTNB60',
            ],
            "Moeda": ['WDO1'],
            "Juros US": ['TREASURY']
        }"""


SENTINEL = "DAP_Q30"  # se aparece dentro do dict estrategias, ja foi patchado


def processar(app4_path: Path, apply: bool = False) -> int:
    if not app4_path.exists():
        print(f"[erro] {app4_path} nao existe.")
        return 2

    content = app4_path.read_text(encoding="utf-8")

    # Verifica idempotencia: procura DAP_Q30 dentro do dict estrategias
    if "'DAP_Q30'" in content and ORIGINAL not in content:
        print(f"[ok] {app4_path.name} ja foi patchado anteriormente. Nada a fazer.")
        return 0

    if ORIGINAL not in content:
        print(f"[erro] Bloco 'estrategias' NAO encontrado literalmente em {app4_path.name}.")
        print("[erro] Bloco procurado:")
        print("---")
        print(ORIGINAL)
        print("---")
        return 3

    novo = content.replace(ORIGINAL, PATCHED)
    if novo == content:
        print("[erro] Substituicao resultou em conteudo identico.")
        return 4

    delta = PATCHED.count("\n") - ORIGINAL.count("\n")
    print(f"[info] Bloco encontrado. Substituicao adiciona +{delta} linhas.")
    print(f"[info] Modo: {'APLICAR (com backup)' if apply else 'DRY-RUN'}")

    if apply:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = app4_path.with_name(f"{app4_path.name}.bak_estrategias_{ts}")
        shutil.copy2(app4_path, backup)
        print(f"[backup] {backup}")
        app4_path.write_text(novo, encoding="utf-8")
        print(f"[ok] {app4_path} atualizado.")
    else:
        print("[dry-run] Nada foi escrito. Rode com --apply para efetivar.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--file", default="app4.py")
    args = ap.parse_args()
    sys.exit(processar(Path(args.file), apply=args.apply))
