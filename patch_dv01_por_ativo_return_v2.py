"""patch_dv01_por_ativo_return_v2.py - reaplica patch correto"""
from pathlib import Path
import shutil, datetime as dt

f = Path("gravar_snapshot_diario.py")
shutil.copy2(f, f"{f}.bak_dvret_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
n = 0

# 1) Inicializa dv_por_ativo antes do loop
old1 = '    buckets = {"juros_nom": 0.0, "juros_real": 0.0}\n    n_ok, n_skip = 0, 0'
new1 = '    buckets = {"juros_nom": 0.0, "juros_real": 0.0}\n    dv_por_ativo = {}   # ativo -> dv01 signed em R$/bp\n    n_ok, n_skip = 0, 0'
if old1 in s: s = s.replace(old1, new1); n += 1; print("[ok] dv_por_ativo inicializado")

# 2) Popula dv_por_ativo dentro do loop
old2 = '            print(f"[dv01] {ativo}: PU={pu:.2f} du={du} taxa={taxa_pct:.4f}% dv_contrato={dv_contrato:.4f} qty={qty:.1f} dv_signed={dv_signed:.2f}")'
new2 = old2 + '\n            dv_por_ativo[ativo] = float(dv_signed)'
if old2 in s: s = s.replace(old2, new2); n += 1; print("[ok] dv_por_ativo populado no loop")

# 3) Adiciona no return do _dv01_hoje
old3 = '''    return {
        "dv01_juros_nom":  buckets["juros_nom"]  if abs(buckets["juros_nom"])  > 1e-6 else None,
        "dv01_juros_real": buckets["juros_real"] if abs(buckets["juros_real"]) > 1e-6 else None,
        "dv01_treasury":   None,
        "dv01_ntnb":       None,
        "dv01_total":      total if abs(total) > 1e-6 else None,
    }'''
new3 = '''    return {
        "dv01_juros_nom":  buckets["juros_nom"]  if abs(buckets["juros_nom"])  > 1e-6 else None,
        "dv01_juros_real": buckets["juros_real"] if abs(buckets["juros_real"]) > 1e-6 else None,
        "dv01_treasury":   None,
        "dv01_ntnb":       None,
        "dv01_total":      total if abs(total) > 1e-6 else None,
        "dv01_por_ativo":  dict(dv_por_ativo) if dv_por_ativo else None,
    }'''
if old3 in s: s = s.replace(old3, new3); n += 1; print("[ok] dv01_por_ativo no return")

# 4) Extrai _dv_por_at no _compute_snapshot
old4 = '    dv01_dict = dv01_dict or {}'
new4 = '    dv01_dict = dv01_dict or {}\n    _dv_por_at = dv01_dict.get("dv01_por_ativo") or {}'
if old4 in s: s = s.replace(old4, new4, 1); n += 1; print("[ok] _dv_por_at extraido")

# 5) Adiciona campo no snap dict
old5 = '        "dv01_ntnb":        dv01_dict.get("dv01_ntnb"),'
new5 = '        "dv01_ntnb":        dv01_dict.get("dv01_ntnb"),\n        "dv01_por_ativo":   _dv_por_at if _dv_por_at else None,'
if old5 in s: s = s.replace(old5, new5); n += 1; print("[ok] dv01_por_ativo no snap dict")

f.write_text(s, encoding="utf-8")
print(f"[done] {n}/5 fixes aplicados")

# Verifica
s2 = f.read_text(encoding="utf-8")
print("Confirmacao:")
print("  dv_por_ativo init:", "dv_por_ativo = {}" in s2)
print("  populado no loop:", "dv_por_ativo[ativo] = float(dv_signed)" in s2)
print("  no return:", '"dv01_por_ativo":  dict(dv_por_ativo)' in s2)
print("  no snap dict:", '"dv01_por_ativo":   _dv_por_at' in s2)
