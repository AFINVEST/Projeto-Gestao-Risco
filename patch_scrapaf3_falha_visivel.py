"""patch_scrapaf3_falha_visivel.py - ScrapAF3 mostra onde falhou, salva print e sai com erro."""
from pathlib import Path
import shutil, datetime as dt, sys

f = Path("ScrapAF3.py")
s = f.read_text(encoding="utf-8")
if "[af3-falha]" in s:
    print("[skip] ja aplicado"); sys.exit(0)
old = '        print(f"Erro durante a execução: {str(e)}")\n'
if s.count(old) != 1:
    print(f"[ERRO] ancora encontrada {s.count(old)}x - nada alterado"); sys.exit(1)
new = '''        # [af3-falha] diagnostico + sai com erro para o .bat parar
        import traceback as _tb, sys as _sys
        print(f"Erro durante a execução: {type(e).__name__}: {str(e).strip()[:300]}")
        _fr = [x for x in _tb.extract_tb(e.__traceback__) if x.filename.endswith("ScrapAF3.py")]
        if _fr:
            print(f"[ScrapAF3] falhou na linha {_fr[-1].lineno}: {(_fr[-1].line or '').strip()}")
        try:
            print(f"[ScrapAF3] URL no momento do erro: {driver.current_url}")
            driver.save_screenshot("debug_scrapaf3.png")
            with open("debug_scrapaf3.html", "w", encoding="utf-8") as _fh:
                _fh.write(driver.page_source)
            print("[ScrapAF3] print salvo em debug_scrapaf3.png (HTML em debug_scrapaf3.html)")
        except Exception as _e2:
            print(f"[ScrapAF3] nao consegui salvar o print: {_e2}")
        driver.quit()
        _sys.exit(1)
'''
shutil.copy2(f, f"{f}.bak_af3falha_{dt.datetime.now():%Y%m%d_%H%M%S}")
f.write_text(s.replace(old, new), encoding="utf-8")
print("[ok] ScrapAF3 agora mostra a etapa da falha e sai com erro")
