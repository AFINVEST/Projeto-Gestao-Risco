"""
08_patch_app4_odf.py
====================

Patch idempotente: corrige o _rate_col_for_di_F() em app4.py pra
funcionar com o naming novo dos DIs (DI_F29, DI_J29, DI_N29, DI_V29)
E manter compatibilidade com o naming legado (DI_29 = janeiro).

USO:
    cd Z:\\...\\Projeto-Gestao-Risco
    python dash_risco_v2\\08_patch_app4_odf.py
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path


ARQUIVO = Path("app4.py")
BKP     = Path("app4.py.bak_pre_odf_patch")


OLD_BLOCK = '''        def _rate_col_for_di_F(cols, di_code: str) -> str | None:
            yy = str(di_code).split("_")[-1]
            col = f"ODF{yy} Comdty"
            return col if col in cols else None'''

NEW_BLOCK = '''        def _rate_col_for_di_F(cols, di_code: str) -> str | None:
            # v2b: aceita naming novo (DI_F29, DI_J29, DI_N29, DI_V29)
            # e legado (DI_29 = janeiro). Extrai letra do mês e yy corretamente.
            parts = str(di_code).split("_")
            if len(parts) < 2:
                return None
            tail = parts[-1]
            if len(tail) >= 3 and tail[0].isalpha() and tail[1:].isdigit():
                letter, yy = tail[0], tail[1:3]
            else:
                yy = ''.join(c for c in tail if c.isdigit())[:2]
                letter = 'F'
            col = f"OD{letter}{yy} Comdty"
            return col if col in cols else None'''


def main():
    if not ARQUIVO.exists():
        print(f"ERRO: {ARQUIVO} não encontrado.", file=sys.stderr)
        sys.exit(1)

    if not BKP.exists():
        shutil.copy2(ARQUIVO, BKP)
        print(f"Backup criado: {BKP}")
    else:
        print(f"Backup já existe: {BKP}")

    text = ARQUIVO.read_text(encoding="utf-8")

    # Detecta se já foi aplicado
    if "aceita naming novo (DI_F29, DI_J29" in text:
        print("Patch já aplicado (idempotente). Nada a fazer.")
        return

    n = text.count(OLD_BLOCK)
    if n == 0:
        print("BLOCO ANTIGO NÃO ENCONTRADO. Talvez a versão do app4 seja diferente.")
        print("Confira manualmente a função _rate_col_for_di_F em app4.py.")
        sys.exit(1)
    if n != 1:
        print(f"Esperado 1 match, encontrei {n}. Aborto pra segurança.")
        sys.exit(1)

    text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    ARQUIVO.write_text(text, encoding="utf-8")
    print(f"OK — patch aplicado em {ARQUIVO}. Backup em {BKP}.")


if __name__ == "__main__":
    main()
