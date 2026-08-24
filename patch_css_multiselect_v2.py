"""patch_css_multiselect_v2.py - CSS mais forte pra dropdown options"""
from pathlib import Path
import shutil, datetime as dt
f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_css2_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Substitui bloco CSS existente por versao mais forte
old = '''<style>
[data-baseweb="popover"] [role="option"] { color: #111 !important; }
[data-baseweb="popover"] li { color: #111 !important; }
[data-baseweb="select"] span { color: #fff !important; }
div[role="listbox"] * { color: #111 !important; }
</style>'''
new = '''<style>
/* Dropdown options — texto escuro em fundo branco */
[data-baseweb="popover"] { background: white !important; }
[data-baseweb="popover"] * { color: #111 !important; }
[data-baseweb="popover"] [role="option"] { color: #111 !important; background: white !important; }
[data-baseweb="popover"] [role="option"]:hover { background: #e5e7eb !important; color: #111 !important; }
[data-baseweb="popover"] li { color: #111 !important; }
[data-baseweb="popover"] ul { background: white !important; }
div[role="listbox"] { background: white !important; }
div[role="listbox"] * { color: #111 !important; }
[role="option"] { color: #111 !important; background: white !important; }
[role="option"] * { color: #111 !important; }
/* Chips selecionados: fundo azul escuro, texto branco (mantém como está) */
[data-baseweb="tag"] { background: #1a3a6c !important; }
[data-baseweb="tag"] * { color: white !important; }
</style>'''
if old in s: s = s.replace(old, new); print("[ok] CSS multiselect reforcado")

f.write_text(s, encoding="utf-8")
print("[done]")
