"""patch_css_multiselect_final.py"""
from pathlib import Path
import shutil, datetime as dt
f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_cssfin_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# CSS a inserir apos 'import streamlit as st'
css_block = """

# ─── CSS Fix multiselect (dropdown options visiveis) ───
st.markdown('''<style>
[data-baseweb="popover"] { background: white !important; }
[data-baseweb="popover"] * { color: #111 !important; }
[data-baseweb="popover"] [role="option"] { color: #111 !important; background: white !important; }
[data-baseweb="popover"] [role="option"]:hover { background: #e5e7eb !important; }
[data-baseweb="popover"] li { color: #111 !important; background: white !important; }
[data-baseweb="popover"] ul { background: white !important; }
div[role="listbox"] { background: white !important; }
div[role="listbox"] * { color: #111 !important; }
[role="option"] { color: #111 !important; background: white !important; }
[data-baseweb="tag"] { background: #1a3a6c !important; }
[data-baseweb="tag"] * { color: white !important; }
</style>''', unsafe_allow_html=True)
"""

marker = "import streamlit as st"
if "data-baseweb=\"popover\"" not in s:
    idx = s.find(marker) + len(marker)
    nl = s.find("\n", idx)
    s = s[:nl+1] + css_block + s[nl+1:]
    f.write_text(s, encoding="utf-8")
    print("[ok] CSS multiselect inserido apos import streamlit")
else:
    print("[skip] CSS ja aplicado")
