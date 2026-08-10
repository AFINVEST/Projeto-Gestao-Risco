"""patch_scrapb3_no_dup.py - desliga dup-final e aplica max->min"""
from pathlib import Path
import shutil, datetime as dt, sys

f = Path("ScrapB3_v2.py")
if not f.exists():
    sys.exit("ScrapB3_v2.py nao encontrado")

bak = f.with_suffix(f".py.bak_{dt.datetime.now():%Y%m%d_%H%M%S}")
shutil.copy2(f, bak)
print(f"[backup] {bak.name}")

src = f.read_text(encoding="utf-8")
changed = 0

# --- Fix 1: max -> min no start_dt ---
old1 = "        candidates = [d for d in [last_preco, last_valor] if d is not None]\n        start_dt = max(candidates)"
new1 = "        candidates = [d for d in [last_preco, last_valor] if d is not None]\n        start_dt = min(candidates)   # patched: min garante backfill quando parquets desalinhadas"
if old1 in src:
    src = src.replace(old1, new1); changed += 1
    print("[ok] Fix 1: start_dt = min(candidates)")
elif "start_dt = min(candidates)" in src:
    print("[skip] Fix 1: ja aplicado")
else:
    print("[warn] Fix 1: bloco nao encontrado")

# --- Fix 2: desliga adicionar_coluna_duplicada_final ---
# Substitui a chamada por um no-op comentado
old2 = "    # 5) cria a coluna do próximo DU como cópia da última real, se ainda não existir\n    wide_preco, wide_valor = adicionar_coluna_duplicada_final(wide_preco, wide_valor)"
new2 = "    # 5) DESLIGADO: adicionar_coluna_duplicada_final criava coluna fake pro proximo DU,\n    #    o que quebrava snapshot/cota downstream (retorno=0, email do dia errado).\n    #    Se B3 nao tem dado real, a coluna simplesmente nao existe.\n    # wide_preco, wide_valor = adicionar_coluna_duplicada_final(wide_preco, wide_valor)"
if old2 in src:
    src = src.replace(old2, new2); changed += 1
    print("[ok] Fix 2: dup-final desligado")
elif "# wide_preco, wide_valor = adicionar_coluna_duplicada_final" in src:
    print("[skip] Fix 2: ja aplicado")
else:
    print("[warn] Fix 2: chamada nao encontrada")

f.write_text(src, encoding="utf-8")
print(f"[done] {changed} mudancas aplicadas em {f}")
