"""patch_pfech_coerce.py - garante p_fech/p_ant como float via _safe_num"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_pfsafe_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Adiciona helper _safe_num  
if "def _safe_num" not in s:
    helper = '''
def _safe_num(x):
    """Converte pra float retornando np.nan se falhar."""
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")

'''
    if "def _is_finite_num" in s:
        s = s.replace("def _is_finite_num", helper + "def _is_finite_num", 1)
    else:
        s = s.replace("def analisar_dados_fundos2(", helper + "def analisar_dados_fundos2(", 1)
    print("[ok] _safe_num helper adicionado")

# Substitui as 4 linhas problematicas
# padrao: rend = (p_fech - (p_ant if _is_finite_num(p_ant) else p_fech)) * qtd [...]
# novo:   _pf = _safe_num(p_fech); _pa = _safe_num(p_ant); rend = (_pf - (_pa if _is_finite_num(_pa) else _pf)) * qtd [...]

replacements = [
    ('rend = (p_fech - (p_ant if _is_finite_num(p_ant) else p_fech)) * qtd * dolar / 10_000',
     '_pf, _pa = _safe_num(p_fech), _safe_num(p_ant)\n                        rend = (_pf - (_pa if _is_finite_num(_pa) else _pf)) * qtd * dolar / 10_000'),
    ('rend = (p_fech - (p_ant if _is_finite_num(p_ant) else p_fech)) * qtd',
     '_pf, _pa = _safe_num(p_fech), _safe_num(p_ant)\n                        rend = (_pf - (_pa if _is_finite_num(_pa) else _pf)) * qtd'),
]
for old, new in replacements:
    cnt = s.count(old)
    if cnt > 0:
        s = s.replace(old, new)
        print(f"[ok] {cnt}x substituicoes de '{old[:60]}...'")

f.write_text(s, encoding="utf-8")
print("[done]")
