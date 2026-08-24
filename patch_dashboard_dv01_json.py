"""patch_dashboard_dv01_json.py - dashboard le dv01_por_ativo do Supabase (JSON)"""
from pathlib import Path
import shutil, datetime as dt
f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_dvjson_dash_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

old = '''        # Fonte primaria: metodo novo (dv01_dinamico) via _dv01_hoje
        # Fallback: risco dict (BBG method)
        dv01_asset_bps_dict = {}
        try:
            from gravar_snapshot_diario import _dv01_hoje as _dv_calc
            from cota_portfolio_core import load_basefundos as _load_bf_dv
            import pandas as _pd_dv
            # Usa data do ultimo snapshot (mais confiavel que today() em hospedado)
            _snap_ult = supabase.table("snapshot_diario").select("Data,pl_total").order("Data", desc=True).limit(1).execute().data
            if _snap_ult:
                _ref_dt = _pd_dv.Timestamp(_snap_ult[0]["Data"])
                _pl_tot_dv = float(_snap_ult[0].get("pl_total") or 0)
                _pl_risco_dv = _pl_tot_dv * 0.01
                _dv_hoje_dict = _dv_calc(_load_bf_dv(), _ref_dt)
                _dv_por_at = _dv_hoje_dict.get("dv01_por_ativo") or {}
                if _dv_por_at and _pl_risco_dv > 0:
                    for a, v_R in _dv_por_at.items():
                        dv01_asset_bps_dict[a] = (v_R / _pl_risco_dv) * 10_000
        except Exception as _edv:
            st.caption(f"[debug DV01] excecao no metodo novo: {_edv}")

        # Fallback: se metodo novo nao populou, usa risco dict
        if not dv01_asset_bps_dict:
            dv01_asset_bps_dict = risco.get("DV01 por ativo (bps)", {}) or {}
            if dv01_asset_bps_dict:
                st.caption("[debug DV01] usando fonte legado (risco dict). Total pode divergir do summary.")'''

novo = '''        # Le dv01_por_ativo (JSON) direto do snapshot_diario — bate 100% com o summary
        dv01_asset_bps_dict = {}
        try:
            _snap_dv_at = supabase.table("snapshot_diario").select("Data,pl_total,dv01_por_ativo").order("Data", desc=True).limit(1).execute().data
            if _snap_dv_at:
                _pl_tot_dv = float(_snap_dv_at[0].get("pl_total") or 0)
                _pl_risco_dv = _pl_tot_dv * 0.01
                _dv_json = _snap_dv_at[0].get("dv01_por_ativo") or {}
                if _dv_json and _pl_risco_dv > 0:
                    for a, v_R in _dv_json.items():
                        dv01_asset_bps_dict[a] = (float(v_R) / _pl_risco_dv) * 10_000
        except Exception as _edv:
            st.caption(f"[debug DV01] erro lendo Supabase: {_edv}")

        # Fallback: risco dict (BBG method) se snapshot nao tem
        if not dv01_asset_bps_dict:
            dv01_asset_bps_dict = risco.get("DV01 por ativo (bps)", {}) or {}
            if dv01_asset_bps_dict:
                st.caption("[debug DV01] usando fonte legado (risco). Rode gravar_snapshot_diario.py pra popular dv01_por_ativo no Supabase.")'''

if old in s:
    s = s.replace(old, novo)
    f.write_text(s, encoding="utf-8")
    print("[ok] dashboard le dv01_por_ativo do Supabase")
else:
    print("[warn] bloco nao encontrado")
