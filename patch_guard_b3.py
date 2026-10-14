from pathlib import Path
import shutil, datetime as dt
f = Path("enviar_email_diario.py")
shutil.copy2(f, f"{f}.bak_guardb3_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

old = """    _hoje = _dt_g.date.today()
    _esp = _hoje - _dt_g.timedelta(days=1)
    while _esp.weekday() >= 5:  # pula sab/dom
        _esp -= _dt_g.timedelta(days=1)"""

new = """    # Esperado = ultimo pregao publicado pela B3 (respeita feriados)
    try:
        import pandas as _pd_g
        _dfp = _pd_g.read_parquet("Dados/df_preco_de_ajuste_atual_completo.parquet")
        _datas_b3 = sorted(c for c in _dfp.columns if c != "Assets")
        _esp = _dt_g.date.fromisoformat(_datas_b3[-1])
    except Exception:
        _esp = _dt_g.date.today() - _dt_g.timedelta(days=1)
        while _esp.weekday() >= 5:
            _esp -= _dt_g.timedelta(days=1)"""

if old in s:
    s = s.replace(old, new); f.write_text(s, encoding="utf-8")
    print("[ok] guard agora usa calendario B3 (ultimo pregao do df_preco)")
else:
    print("[warn] trecho do guard nao encontrado")
