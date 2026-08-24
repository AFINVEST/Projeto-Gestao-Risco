"""patch_dv01_por_ativo_return.py - _dv01_hoje passa a retornar per-ativo tambem"""
from pathlib import Path
import shutil, datetime as dt

f = Path("gravar_snapshot_diario.py")
shutil.copy2(f, f"{f}.bak_dvperat_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Adiciona per_ativo no return do _dv01_hoje
old = '''    return {
        "dv01_juros_nom":  buckets["juros_nom"]  if abs(buckets["juros_nom"])  > 1e-6 else None,
        "dv01_juros_real": buckets["juros_real"] if abs(buckets["juros_real"]) > 1e-6 else None,
        "dv01_treasury":   None,
        "dv01_ntnb":       None,
        "dv01_total":      total if abs(total) > 1e-6 else None,
    }'''

new = '''    return {
        "dv01_juros_nom":  buckets["juros_nom"]  if abs(buckets["juros_nom"])  > 1e-6 else None,
        "dv01_juros_real": buckets["juros_real"] if abs(buckets["juros_real"]) > 1e-6 else None,
        "dv01_treasury":   None,
        "dv01_ntnb":       None,
        "dv01_total":      total if abs(total) > 1e-6 else None,
        "dv01_por_ativo":  dv_por_ativo,   # dict signed em R$/bp
    }'''
if old in s: s = s.replace(old, new); print("[ok] return dict + dv01_por_ativo")

# Inicializa o dict antes do loop
old_init = '    buckets = {"juros_nom": 0.0, "juros_real": 0.0}\n    n_ok, n_skip = 0, 0'
new_init = '    buckets = {"juros_nom": 0.0, "juros_real": 0.0}\n    dv_por_ativo = {}   # ativo -> dv01_signed R$/bp\n    n_ok, n_skip = 0, 0'
if old_init in s: s = s.replace(old_init, new_init); print("[ok] dict dv_por_ativo inicializado")

# Popula o dict dentro do loop apos calcular dv_signed
old_signed = '            print(f"[dv01] {ativo}: PU={pu:.2f} du={du} taxa={taxa_pct:.4f}% dv_contrato={dv_contrato:.4f} qty={qty:.1f} dv_signed={dv_signed:.2f}")'
new_signed = '            print(f"[dv01] {ativo}: PU={pu:.2f} du={du} taxa={taxa_pct:.4f}% dv_contrato={dv_contrato:.4f} qty={qty:.1f} dv_signed={dv_signed:.2f}")\n            dv_por_ativo[ativo] = dv_signed'
if old_signed in s: s = s.replace(old_signed, new_signed); print("[ok] dv_por_ativo populado no loop")

f.write_text(s, encoding="utf-8")
print("[done]")
