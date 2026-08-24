"""patch_dv01_fallback_debug.py - fallback melhor + info debug"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_dvfb_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

old = '''        # Usa o mesmo metodo do snapshot (dv01_dinamico + PU B3) — bate com o summary
        try:
            from gravar_snapshot_diario import _dv01_hoje as _dv_calc
            from cota_portfolio_core import load_basefundos as _load_bf_dv
            import pandas as _pd_dv
            _dv_hoje_dict = _dv_calc(_load_bf_dv(), _pd_dv.Timestamp.today().normalize())
            _dv_por_at = _dv_hoje_dict.get("dv01_por_ativo", {}) or {}
            # PL total pra converter R$ em bps sobre PL Risco
            _snap_pl = supabase.table("snapshot_diario").select("pl_total").order("Data", desc=True).limit(1).execute().data
            _pl_tot_dv = float(_snap_pl[0].get("pl_total") or 0) if _snap_pl else 0
            _pl_risco_dv = _pl_tot_dv * 0.01
            dv01_asset_bps_dict = {}
            for a, v_R in _dv_por_at.items():
                if _pl_risco_dv > 0:
                    dv01_asset_bps_dict[a] = (v_R / _pl_risco_dv) * 10_000
        except Exception as _edv:
            st.warning(f"Fallback DV01 (risco dict): {_edv}")
            dv01_asset_bps_dict = risco.get("DV01 por ativo (bps)", {}) or {}'''

novo = '''        # Fonte primaria: metodo novo (dv01_dinamico) via _dv01_hoje
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

if old in s:
    s = s.replace(old, novo)
    f.write_text(s, encoding="utf-8")
    print("[ok] fallback melhorado + debug messages")
else:
    print("[warn] bloco nao encontrado")
