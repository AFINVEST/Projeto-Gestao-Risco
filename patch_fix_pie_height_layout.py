"""patch_fix_pie_height_layout.py"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_pie_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
n = 0

# Fix 1: adiciona PIE_HEIGHT global proximo do "if True: _dv_dict_old = {}"
old_true = '            if True:\n                _dv_dict_old = {}'
new_true = '            if True:\n                _dv_dict_old = {}\n                PIE_HEIGHT = 340  # restaurada pra evitar UnboundLocalError'
if old_true in s:
    s = s.replace(old_true, new_true); n += 1
    print("[ok] PIE_HEIGHT redefinido")

# Fix 2: colapsa COL1/COLmeio/COL2 (COL2 esta vazia apos hide do DV01 hist)
old_cols = 'COL1, COLmeio, COL2 = st.columns([4.8, 0.2, 4.8])'
new_cols = 'COL1, COLmeio, COL2 = st.columns([100, 0.001, 0.001])  # COL2/meio invisiveis (conteudo hidden Fase 2)'
if old_cols in s:
    s = s.replace(old_cols, new_cols); n += 1
    print("[ok] COL1/COL2 colapsados")

# Fix 3: colapsa tambem a divisao do CoVaR (colll1, colllmeio, colll2)
old_covar = 'colll1, colllmeio, colll2 = st.columns([4.8, 0.2, 4.8])'
new_covar = 'colll1, colllmeio, colll2 = st.columns([100, 0.001, 0.001])  # CoVaR historico oculto'
if old_covar in s:
    s = s.replace(old_covar, new_covar); n += 1
    print("[ok] CoVaR cols colapsados")

f.write_text(s, encoding="utf-8")
print(f"[done] {n} fixes")
