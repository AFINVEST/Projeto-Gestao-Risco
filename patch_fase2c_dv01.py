"""patch_fase2c_dv01.py - Resumo de DV01 no formato do email diario"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_fase2c_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# --- Replace bloco "Resumo de DV01" ---
old = '''    with tab_orcamento:
        with COL1:
            st.subheader("Resumo de DV01")
            c4, c5= st.columns(2)
            c4.metric("DV01 Port (R$/bp / bps)", dv01_port_display)
            c5.metric("DV01 Stress (R$ / bps)",  dv01_strss_display)
            #c6.metric("CoVaR Total (R$ / bps)",  covar_tot_display)'''

novo = '''    with tab_orcamento:
        with COL1:
            st.subheader("Risco atual da carteira")
            # Puxa DV01 do ultimo snapshot
            _snap_dv = supabase.table("snapshot_diario").select(
                "Data,pl_total,dv01_juros_nom,dv01_juros_real,dv01_treasury,dv01_ntnb,dv01_total"
            ).order("Data", desc=True).limit(1).execute().data
            if _snap_dv:
                _sd = _snap_dv[0]
                _pl_tot = float(_sd.get("pl_total") or 0)
                _pl_risco = _pl_tot * 0.01
                _dv_nom  = float(_sd.get("dv01_juros_nom") or 0)
                _dv_real = float(_sd.get("dv01_juros_real") or 0)
                _dv_ntnb = float(_sd.get("dv01_ntnb") or 0)
                _dv_real_total = _dv_real + _dv_ntnb
                _dv_total = _dv_nom + _dv_real_total

                _STRESS_NOM  = 100
                _STRESS_REAL = 50
                _str_nom  = _dv_nom * _STRESS_NOM
                _str_real = _dv_real_total * _STRESS_REAL
                _str_tot  = _str_nom + _str_real

                def _bps_risco(v):
                    return (v / _pl_risco) * 10_000 if _pl_risco else 0

                _html_dv = """
<div style='background:#f8f9fa;border:1px solid #ddd;border-radius:6px;padding:16px;font-size:14px;line-height:1.8;'>
  <div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;font-weight:600;'>DV01 e Stress do dia</div>
  <b>Juros Nominais<sup>*</sup>:</b><br>
  &nbsp;&nbsp;DV01 = R$ {dv_nom:,.0f} ({bps_nom:.2f} bps)<br>
  &nbsp;&nbsp;Stress = R$ {str_nom:,.0f} ({str_bps_nom:.1f} bps)<br><br>
  <b>Juros Reais<sup>**</sup>:</b><br>
  &nbsp;&nbsp;DV01 = R$ {dv_real:,.0f} ({bps_real:.2f} bps)<br>
  &nbsp;&nbsp;Stress = R$ {str_real:,.0f} ({str_bps_real:.1f} bps)<br><br>
  <b style='color:#1a3a6c;'>TOTAL:</b><br>
  &nbsp;&nbsp;DV01 = R$ {dv_tot:,.0f} ({bps_tot:.2f} bps)<br>
  &nbsp;&nbsp;Stress = R$ {str_tot:,.0f} ({str_bps_tot:.1f} bps)
  <div style='font-size:10px;color:#666;margin-top:12px;border-top:1px solid #ddd;padding-top:6px;'>
    <sup>*</sup>Juros nominais (DI): choque de 100 bps<br>
    <sup>**</sup>Juros reais (DAP + NTNB): choque de 50 bps<br>
    bps calculados sobre PL Risco (1% do PL total).
  </div>
</div>
""".format(
                    dv_nom=_dv_nom, bps_nom=_bps_risco(_dv_nom),
                    str_nom=_str_nom, str_bps_nom=_bps_risco(_str_nom),
                    dv_real=_dv_real_total, bps_real=_bps_risco(_dv_real_total),
                    str_real=_str_real, str_bps_real=_bps_risco(_str_real),
                    dv_tot=_dv_total, bps_tot=_bps_risco(_dv_total),
                    str_tot=_str_tot, str_bps_tot=_bps_risco(_str_tot),
                )
                st.markdown(_html_dv, unsafe_allow_html=True)
            else:
                st.info("Sem snapshot_diario disponivel.")'''

if old in s:
    s = s.replace(old, novo)
    f.write_text(s, encoding="utf-8")
    print("[ok] Resumo de DV01 substituido pelo formato do email")
else:
    print("[warn] bloco nao encontrado")
