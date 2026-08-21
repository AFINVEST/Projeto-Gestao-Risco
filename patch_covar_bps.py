"""patch_covar_bps.py - adiciona bps ao lado do R$ no bar chart CoVaR"""
from pathlib import Path
import shutil, datetime as dt
f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_covbps_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Modifica text no bar chart do CoVaR por ativo
old = '''_df_at = _pd_c.DataFrame([
                            {"Ativo": a, "Classe": _mc2(a), "CoVaR_R": v}
                            for a, v in _covar_ativo.items() if abs(v) > 1e-6
                        ]).sort_values("CoVaR_R", ascending=True)'''
new = '''_covar_bps_by_ativo = _covar_res.get("covar_por_ativo_bps", {})
                        # ajusta signal: se CoVaR_R for negativo (hedge), bps tb negativo
                        _df_at = _pd_c.DataFrame([
                            {"Ativo": a, "Classe": _mc2(a), "CoVaR_R": v,
                             "CoVaR_bps": _covar_bps_by_ativo.get(a, 0) * (1 if v >= 0 else -1)}
                            for a, v in _covar_ativo.items() if abs(v) > 1e-6
                        ]).sort_values("CoVaR_R", ascending=True)'''
if old in s: s = s.replace(old, new); print("[ok] bps col adicionada em _df_at")

# Ajusta o text do bar pra mostrar R$ e bps
old_txt = 'text=[f"R$ {v:,.0f}" for v in _sub["CoVaR_R"]],'
new_txt = 'text=[f"R$ {r:,.0f} ({b:+.2f} bps)" for r, b in zip(_sub["CoVaR_R"], _sub["CoVaR_bps"])],'
if old_txt in s: s = s.replace(old_txt, new_txt); print("[ok] text bar mostra R$ + bps")

# Adiciona bps na donut composicao (por classe)
old_pie = '''_fig_pie = _pgo_c.Figure(_pgo_c.Pie(
                            labels=_labels, values=_vals, hole=0.55,
                            marker=dict(colors=_cores),
                            textinfo="label+percent", sort=False,
                        ))'''
new_pie = '''# Agrega bps por classe
                    _bps_por_classe = {}
                    for _a, _bp in _covar_bps_by_ativo.items():
                        _cl = _mc2(_a) if callable(_mc2) else None
                        if _cl:
                            _bps_por_classe[_cl] = _bps_por_classe.get(_cl, 0) + abs(_bp)
                    _cl_labels_bps = [f"{l}<br>{_bps_por_classe.get(l, 0):.2f} bps" for l in _labels]
                    _fig_pie = _pgo_c.Figure(_pgo_c.Pie(
                        labels=_labels, values=_vals, hole=0.55,
                        marker=dict(colors=_cores),
                        text=_cl_labels_bps, textinfo="text+percent", sort=False,
                    ))'''
# Nao vou aplicar essa parte agora (precisa mover _mc2 pra cima da _fig_pie) — deixo pro proximo patch se necessario

f.write_text(s, encoding="utf-8")
print("[done]")
