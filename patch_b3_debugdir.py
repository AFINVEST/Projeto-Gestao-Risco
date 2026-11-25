from pathlib import Path
import shutil, datetime as dt, sys
f = Path("ScrapB3_v2.py"); s = f.read_text(encoding="utf-8")
if "[b3-debugdir]" in s: print("[skip] ja aplicado"); sys.exit(0)
old = "        Path(DEBUG_DUMP_DIR).mkdir(parents=True, exist_ok=True)\n"
if s.count(old) != 1: print(f"[ERRO] ancora {s.count(old)}x - nada alterado"); sys.exit(1)
new = ("        try:  # [b3-debugdir] drive de rede as vezes acusa WinError 183; o dump e opcional\n"
       "            Path(DEBUG_DUMP_DIR).mkdir(parents=True, exist_ok=True)\n"
       "        except OSError:\n"
       "            pass\n")
shutil.copy2(f, f"{f}.bak_debugdir_{dt.datetime.now():%Y%m%d_%H%M%S}")
f.write_text(s.replace(old, new), encoding="utf-8")
print("[ok] ScrapB3: falha na pasta de debug nao afeta mais a leitura")
