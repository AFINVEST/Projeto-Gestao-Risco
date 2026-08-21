"""patch_delete_covar_antigo.py - deleta range exato do CoVaR bar+donut antigo"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_delcovar_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
lines = s.split("\n")

# Delete linhas 9420-9579 (1-indexed) = index 9419-9578 (0-indexed)
DEL_START = 9419  # index 0-based da 1a linha a deletar
DEL_END   = 9579  # index 0-based da 1a linha a MANTER (exclusive)

# Preserva contexto antes/depois pra verificar
print(f"Linha antes ({DEL_START}): {lines[DEL_START-1][:80]}")
print(f"1a a deletar ({DEL_START+1}): {lines[DEL_START][:80]}")
print(f"Ultima a deletar ({DEL_END}): {lines[DEL_END-1][:80]}")
print(f"Linha depois ({DEL_END+1}): {lines[DEL_END][:80]}")

# Substitui range por uma linha de comentario
novo_lines = lines[:DEL_START] + [
    "            # DELETED Fase 3: bloco antigo CoVaR bar+donut removido (substituido por composicao nova acima)",
] + lines[DEL_END:]

n_deletadas = (DEL_END - DEL_START) - 1
print(f"\nDeletadas {n_deletadas} linhas ({DEL_START+1}-{DEL_END})")

f.write_text("\n".join(novo_lines), encoding="utf-8")
print("[done]")
