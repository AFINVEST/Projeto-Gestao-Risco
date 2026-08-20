"""patch_covarhist_lower_hide.py"""
from pathlib import Path
import shutil, datetime as dt
f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_covlow_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Fix nomes lowercase no chart historico
old = '''_rh = supabase.table("snapshot_diario").select(
                            "Data,covar_juros_nom_pct,covar_juros_real_pct,covar_moeda_pct,covar_juros_us_pct,covar_outros_pct"
                        ).order("Data").range(_offset_h, _offset_h+999).execute()'''
new = '''_rh = supabase.table("snapshot_diario").select(
                            "Data,covar_juros_nom_pct,covar_juros_real_pct,covar_moeda_pct,covar_juros_us_pct,covar_outros_pct,covar_total_r"
                        ).order("Data").range(_offset_h, _offset_h+999).execute()'''
if old in s: s = s.replace(old, new); print("[ok] query lowercase")

# Hide chart antigo "CoVaR por estratégia — área empilhada" (linha ~9587-9605)
old_area = '''            with colll2:
                #st.caption("CoVaR por estratégia — área empilhada")'''
new_area = '''            with colll2:
              if False:  # HIDDEN Fase 3: substituido pelo novo chart historico CoVaR
                #st.caption("CoVaR por estratégia — área empilhada")'''
if old_area in s: s = s.replace(old_area, new_area); print("[ok] area empilhada antiga oculta (subheader)")

f.write_text(s, encoding="utf-8")
print("[done]")
