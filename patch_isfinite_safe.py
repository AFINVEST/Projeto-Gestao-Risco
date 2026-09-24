"""patch_isfinite_safe.py - torna np.isfinite(p_ant) seguro contra tipos nao-numericos"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_isfinite_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Adiciona helper _is_finite_num logo antes da funcao analisar_dados_fundos2 ou no topo
if "_is_finite_num" not in s:
    helper = '''
def _is_finite_num(x):
    """isfinite seguro: retorna True se x eh um float/int finito, False pra string/None/NaN."""
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False

'''
    # Insere antes de "def analisar_dados_fundos2"
    marker = "def analisar_dados_fundos2("
    if marker in s:
        s = s.replace(marker, helper + marker, 1)
        print("[ok] helper _is_finite_num adicionado")

# Substitui todas as ocorrencias de np.isfinite(p_ant) por _is_finite_num(p_ant)
count = s.count("np.isfinite(p_ant)")
s = s.replace("np.isfinite(p_ant)", "_is_finite_num(p_ant)")
print(f"[ok] {count} ocorrencias de np.isfinite(p_ant) substituidas")

# Tambem protege np.isfinite(p_fech)
count2 = s.count("np.isfinite(p_fech)")
s = s.replace("np.isfinite(p_fech)", "_is_finite_num(p_fech)")
print(f"[ok] {count2} ocorrencias de np.isfinite(p_fech) substituidas")

f.write_text(s, encoding="utf-8")
print("[done]")
