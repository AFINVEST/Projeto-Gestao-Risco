"""patch_isfinite_num_forte.py - _is_finite_num que tambem coerce"""
from pathlib import Path
import shutil, datetime as dt
f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_isfnv2_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Envolve o subtracao inteira num try/except no ponto de rend
# Estrategia: substituir p_fech - p_ant pela versao safe usando funcao auxiliar

# Melhor: cria funcao _diff_safe(a, b) que faz float(a) - float(b) com fallback
if "def _diff_safe" not in s:
    helper = '''
def _diff_safe(a, b):
    """(float(a) - float(b)) com fallback 0.0 se qualquer for nao-numerico."""
    try:
        fa = float(a); fb = float(b)
        import math
        if not math.isfinite(fa) or not math.isfinite(fb):
            return 0.0
        return fa - fb
    except (TypeError, ValueError):
        return 0.0

'''
    marker = "def _is_finite_num"
    if marker in s:
        s = s.replace(marker, helper + marker, 1)
        print("[ok] _diff_safe adicionado")

# Substitui o padrao "(p_fech - (p_ant if _is_finite_num(p_ant) else p_fech))" 
# por "_diff_safe(p_fech, p_ant)"
padroes = [
    "(p_fech - (p_ant if _is_finite_num(p_ant) else p_fech))",
    "(p_fech - (p_ant if np.isfinite(p_ant) else p_fech))",
]
for p in padroes:
    cnt = s.count(p)
    if cnt:
        s = s.replace(p, "_diff_safe(p_fech, p_ant)")
        print(f"[ok] {cnt}x '{p[:50]}...' substituido")

f.write_text(s, encoding="utf-8")
print("[done]")
