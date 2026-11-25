from pathlib import Path
import shutil, datetime as dt, sys
f = Path("ScrapB3_v2.py"); s = f.read_text(encoding="utf-8")
if "[b3-sem-backfill]" in s: print("[skip] ja aplicado"); sys.exit(0)
old = "    for prev_d in [target_d] + validos_back:\n"
if s.count(old) != 1: print(f"[ERRO] ancora {s.count(old)}x - nada alterado"); sys.exit(1)
shutil.copy2(f, f"{f}.bak_semfill_{dt.datetime.now():%Y%m%d_%H%M%S}")
f.write_text(s.replace(old, "    for prev_d in [target_d]:  # [b3-sem-backfill] nunca gravar pregao de outro dia com esta data\n"), encoding="utf-8")
print("[ok] ScrapB3: backfill com dia anterior desativado (dia que falhar fica sem atualizar e o email trava)")
