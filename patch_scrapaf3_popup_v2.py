"""patch_scrapaf3_popup_v2.py - fecha popup + JS click definitivamente"""
from pathlib import Path
import shutil, datetime as dt

f = Path("ScrapAF3.py")
shutil.copy2(f, f"{f}.bak_popupv2_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

Q3 = chr(34)*3   # """

old = """            (By.CSS_SELECTOR, "button.btn.btn-outline-primary[data-type='custom']"))).click()"""

new = """            (By.CSS_SELECTOR, "button.btn.btn-outline-primary[data-type='custom']")))
        # Fecha popup/announcement antes de clicar (evita element click intercepted)
        try:
            driver.execute_script(_JS_DISMISS_POPUP)
            import time; time.sleep(0.5)
        except Exception:
            pass
        # JS click bypassa qualquer overlay residual
        _btn_custom = driver.find_element(By.CSS_SELECTOR, "button.btn.btn-outline-primary[data-type='custom']")
        driver.execute_script("arguments[0].click();", _btn_custom)"""

# Adiciona a constante _JS_DISMISS_POPUP no topo do arquivo (apos imports)
JS_CONST = Q3 + """
var banners = document.querySelectorAll('.ihub-announcement-header, .ihub-announcement, [class*="announcement"]');
banners.forEach(function(b){ b.style.display='none'; });
var closeBtns = document.querySelectorAll('.close, [aria-label="Close"], .ihub-announcement-close');
closeBtns.forEach(function(b){ try{ b.click(); }catch(e){} });
""" + Q3

if "_JS_DISMISS_POPUP" not in s:
    # Insere no topo, apos os imports (procura primeira linha com "from selenium")
    lines = s.split(chr(10))
    inserted = False
    for i, l in enumerate(lines):
        if l.strip().startswith("from selenium") or l.strip().startswith("import selenium"):
            # Insere DEPOIS de todos os imports
            j = i
            while j < len(lines) and (lines[j].startswith("from ") or lines[j].startswith("import ") or lines[j].strip() == ""):
                j += 1
            lines.insert(j, "_JS_DISMISS_POPUP = " + JS_CONST + chr(10))
            inserted = True
            break
    if inserted:
        s = chr(10).join(lines)
        print("[ok] _JS_DISMISS_POPUP adicionado")

if old in s:
    s = s.replace(old, new)
    print("[ok] click substituido por popup dismiss + JS click")
elif "arguments[0].click()" in s:
    print("[skip] JS click ja aplicado")
else:
    print("[warn] linha do click nao encontrada")

f.write_text(s, encoding="utf-8")
print("[done]")
