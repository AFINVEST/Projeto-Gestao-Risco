"""patch_fase2c_v2.py - DV01 ao lado dos donuts + hide historico + treemap"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_fase2c_v2_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
n = 0

# --- Fix 1: Layout donuts + DV01 side-by-side (3 colunas) ---
old_layout = '''                _c1, _c2 = st.columns(2)
                with _c1:
                    st.plotly_chart(_donut(_cons_hist, _tit_hist), use_container_width=True)
                with _c2:
                    st.plotly_chart(_donut(_cons_ewma, _tit_ewma), use_container_width=True)

                st.caption("Limite: R$ {:,.0f} ({:.2f} bps do PL) — snapshot {}".format(_limite_R, _lim_bps, _s.get("Data")))'''

novo_layout = '''                # Puxa tambem DV01 pro mesmo snapshot
                _snap_dv = supabase.table("snapshot_diario").select(
                    "dv01_juros_nom,dv01_juros_real,dv01_ntnb,dv01_total"
                ).eq("Data", _s.get("Data")).execute().data
                _sd = _snap_dv[0] if _snap_dv else {}
                _pl_risco = _pl * 0.01
                _dv_nom = float(_sd.get("dv01_juros_nom") or 0)
                _dv_real = float(_sd.get("dv01_juros_real") or 0) + float(_sd.get("dv01_ntnb") or 0)
                _dv_total = _dv_nom + _dv_real
                _str_nom = _dv_nom * 100
                _str_real = _dv_real * 50
                _str_tot = _str_nom + _str_real
                def _bps_r(v): return (v/_pl_risco*10_000) if _pl_risco else 0

                _c1, _c2, _c3 = st.columns([1, 1, 1.3])
                with _c1:
                    st.plotly_chart(_donut(_cons_hist, _tit_hist), use_container_width=True)
                with _c2:
                    st.plotly_chart(_donut(_cons_ewma, _tit_ewma), use_container_width=True)
                with _c3:
                    _dv_html = """<div style='background:#f8f9fa;border:1px solid #ddd;border-radius:6px;padding:14px;font-size:13px;line-height:1.7;margin-top:30px;'>
<div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;font-weight:600;'>DV01 e Stress do dia</div>
<b>Juros Nominais<sup>*</sup>:</b><br>
&nbsp;&nbsp;DV01 = R$ {n_dv:,.0f} ({n_bps:.2f} bps)<br>
&nbsp;&nbsp;Stress = R$ {n_st:,.0f} ({n_sb:.1f} bps)<br><br>
<b>Juros Reais<sup>**</sup>:</b><br>
&nbsp;&nbsp;DV01 = R$ {r_dv:,.0f} ({r_bps:.2f} bps)<br>
&nbsp;&nbsp;Stress = R$ {r_st:,.0f} ({r_sb:.1f} bps)<br><br>
<b style='color:#1a3a6c;'>TOTAL:</b><br>
&nbsp;&nbsp;DV01 = R$ {t_dv:,.0f} ({t_bps:.2f} bps)<br>
&nbsp;&nbsp;Stress = R$ {t_st:,.0f} ({t_sb:.1f} bps)
<div style='font-size:10px;color:#666;margin-top:10px;border-top:1px solid #ddd;padding-top:6px;'>
<sup>*</sup>Juros nominais (DI): choque de 100 bps<br>
<sup>**</sup>Juros reais (DAP + NTNB): choque de 50 bps<br>
bps calculados sobre PL Risco (1% do PL total).
</div>
</div>""".format(
                        n_dv=_dv_nom, n_bps=_bps_r(_dv_nom), n_st=_str_nom, n_sb=_bps_r(_str_nom),
                        r_dv=_dv_real, r_bps=_bps_r(_dv_real), r_st=_str_real, r_sb=_bps_r(_str_real),
                        t_dv=_dv_total, t_bps=_bps_r(_dv_total), t_st=_str_tot, t_sb=_bps_r(_str_tot),
                    )
                    st.markdown(_dv_html, unsafe_allow_html=True)

                st.caption("Limite: R$ {:,.0f} ({:.2f} bps do PL) — snapshot {}".format(_limite_R, _lim_bps, _s.get("Data")))'''

if old_layout in s:
    s = s.replace(old_layout, novo_layout); n += 1
    print("[ok] Layout donuts + DV01 side-by-side")

# --- Fix 2: Remove o bloco antigo do DV01 (Risco atual da carteira) que Fase 2c v1 tinha criado ---
# Ele criou um bloco com "Risco atual da carteira" — apaga isso pra evitar duplicacao
old_risco = '''    with tab_orcamento:
        with COL1:
            st.subheader("Risco atual da carteira")'''
new_risco = '''    with tab_orcamento:
        with COL1:
            if False:  # DESATIVADO: DV01 agora aparece ao lado dos donuts (col11)
                st.subheader("Risco atual da carteira")'''
if old_risco in s:
    s = s.replace(old_risco, new_risco); n += 1
    print("[ok] Bloco duplicado 'Risco atual da carteira' desativado")

# --- Fix 3: Hide DV01 historico area chart ---
old_hist_dv = '''            with COL2:
                st.caption("DV01 por estratégia (área empilhada)")'''
new_hist_dv = '''            with COL2:
              if False:  # HIDDEN Fase 2: DV01 historico area chart ocultado
                st.caption("DV01 por estratégia (área empilhada)")'''
if old_hist_dv in s:
    s = s.replace(old_hist_dv, new_hist_dv); n += 1
    print("[ok] DV01 historico area chart oculto")

# --- Fix 4: Substituir "DV01 por classe" (bar+donut) por TREEMAP unificado ---
old_dv_classe_start = '''        # ===================== DV01 por CLASSE com CATEGORIAS empilhadas =====================
            st.subheader("DV01 por classe")

            dv01_asset_rs_dict  = risco.get("DV01 por ativo (bps)", {}) or {}
            if not dv01_asset_rs_dict:
                st.info("DV01 por ativo indisponível para este portfólio.")'''

novo_dv_classe = '''        # ===================== DV01 por ATIVO x CLASSE (TREEMAP) =====================
            st.subheader("Composição do DV01")

            dv01_asset_rs_dict  = risco.get("DV01 por ativo (bps)", {}) or {}
            if not dv01_asset_rs_dict:
                st.info("DV01 por ativo indisponível para este portfólio.")
            else:
                import plotly.express as _px
                import pandas as _pd
                def _mc(a):
                    au = str(a).upper()
                    if au.startswith("DI"): return "Juros Nominais BR"
                    if au.startswith(("DAP","NTNB")): return "Juros Reais BR"
                    if "TREASURY" in au: return "Juros US"
                    if au.startswith("WDO"): return "Moeda"
                    return "Outros"
                _cores_cls = {"Juros Nominais BR":"#1565c0","Juros Reais BR":"#dc3545","Juros US":"#0d47a1","Moeda":"#e57373","Outros":"#7f7f7f"}
                _df_tm = _pd.DataFrame([
                    {"Ativo": a, "Classe": _mc(a), "DV01_abs": abs(float(v)), "DV01_signed": float(v), "Sinal": "Long" if float(v)>=0 else "Short"}
                    for a, v in dv01_asset_rs_dict.items() if abs(float(v)) > 1e-9
                ])
                if _df_tm.empty:
                    st.info("Todos os DV01 zerados.")
                else:
                    _fig_tm = _px.treemap(
                        _df_tm, path=["Classe","Ativo"], values="DV01_abs",
                        color="Classe", color_discrete_map=_cores_cls,
                        custom_data=["DV01_signed","Sinal"]
                    )
                    _fig_tm.update_traces(
                        hovertemplate="<b>%{label}</b><br>DV01: R$ %{customdata[0]:,.0f}<br>%{customdata[1]}<extra></extra>",
                        textinfo="label+value", texttemplate="<b>%{label}</b><br>R$ %{value:,.0f}"
                    )
                    _fig_tm.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=380)
                    st.plotly_chart(_fig_tm, use_container_width=True)

            # Bloco antigo de DV01 por classe (bar + donut) desativado
            if False:
                _dv_dict_old = {}'''

if old_dv_classe_start in s:
    s = s.replace(old_dv_classe_start, novo_dv_classe); n += 1
    print("[ok] DV01 por classe substituido por TREEMAP")

f.write_text(s, encoding="utf-8")
print(f"[done] {n} mudancas Fase 2c v2")
