"""patch_dv01_return_final.py - corrige return com abs() + dv01_por_ativo"""
from pathlib import Path
import shutil, datetime as dt

f = Path("gravar_snapshot_diario.py")
shutil.copy2(f, f"{f}.bak_dvret_final_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

old = '''    return {
        "dv01_juros_nom":  buckets["juros_nom"]  if buckets["juros_nom"]  > 0 else None,
        "dv01_juros_real": buckets["juros_real"] if buckets["juros_real"] > 0 else None,
        "dv01_treasury":   None,   # não suportado por dv01_dinamico
        "dv01_ntnb":       None,   # não suportado por dv01_dinamico
        "dv01_total":      total if total > 0 else None,
    }'''

new = '''    return {
        "dv01_juros_nom":  buckets["juros_nom"]  if abs(buckets["juros_nom"])  > 1e-6 else None,
        "dv01_juros_real": buckets["juros_real"] if abs(buckets["juros_real"]) > 1e-6 else None,
        "dv01_treasury":   None,   # não suportado por dv01_dinamico
        "dv01_ntnb":       None,   # não suportado por dv01_dinamico
        "dv01_total":      total if abs(total) > 1e-6 else None,
        "dv01_por_ativo":  dict(dv_por_ativo) if dv_por_ativo else None,
    }'''

if old in s:
    s = s.replace(old, new)
    f.write_text(s, encoding="utf-8")
    print("[ok] return corrigido: abs() + dv01_por_ativo")
else:
    print("[warn] bloco nao encontrado")

# Confere
s2 = f.read_text(encoding="utf-8")
print("Tem dv01_por_ativo no return:", "dict(dv_por_ativo)" in s2)
