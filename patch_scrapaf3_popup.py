"""patch_scrapaf3_popup.py - fecha popup antes de clicar em data-type=custom"""
from pathlib import Path
import shutil, datetime as dt

f = Path("ScrapAF3.py")
shutil.copy2(f, f"{f}.bak_popup_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

old = '''            (By.CSS_SELECTOR, "button.btn.btn-outline-primary[data-type='custom']"))).click()'''

new = '''            (By.CSS_SELECTOR, "button.btn.btn-outline-primary[data-type='custom']")))
        # Fecha popup/announcement que sobrepoe o botao antes de clicar
        try:
            driver.execute_script('''
                + chr(34) + chr(34) + chr(34) + '''
                var banners = document.querySelectorAll('.ihub-announcement-header, .ihub-announcement, [class*="announcement"]');
                banners.forEach(function(b){ b.style.display='none'; });
                var closeBtns = document.querySelectorAll('.close, [aria-label="Close"], .ihub-announcement-close');
                closeBtns.forEach(function(b){ try{ b.click(); }catch(e){} });
            ''' + chr(34) + chr(34) + chr(34) + ''')
            import time; time.sleep(0.5)
        except Exception:
            pass
        # Click via JS (bypassa overlay se ainda restar)
        _btn_custom = driver.find_element(By.CSS_SELECTOR, "button.btn.btn-outline-primary[data-type='custom']")
        driver.execute_script("arguments[0].click();", _btn_custom)'''

if old in s:
    s = s.replace(old, new)
    f.write_text(s, encoding="utf-8")
    print("[ok] popup dismiss + JS click aplicado no botao data-type=custom")
else:
    print("[warn] linha nao encontrada")
