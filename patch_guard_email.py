"""patch_guard_email.py - aborta envio se snapshot estiver atrasado"""
from pathlib import Path
import shutil, datetime as dt

f = Path("enviar_email_diario.py")
shutil.copy2(f, f"{f}.bak_guard_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Insere guard logo apos "snapshots_all = _load_snapshot_serie_completa(client)"
marker = 'snapshots_all = _load_snapshot_serie_completa(client)'
if marker in s and 'GUARD SNAPSHOT ATRASADO' not in s:
    guard = marker + '''

    # ─── GUARD: aborta se snapshot estiver atrasado (previne re-envio de dia velho) ───
    # GUARD SNAPSHOT ATRASADO
    import datetime as _dt_g
    _hoje = _dt_g.date.today()
    _esp = _hoje - _dt_g.timedelta(days=1)
    while _esp.weekday() >= 5:  # pula sab/dom
        _esp -= _dt_g.timedelta(days=1)
    if snapshots_all:
        _ult = _dt_g.date.fromisoformat(snapshots_all[-1]["Data"])
        if _ult < _esp:
            _dias = (_esp - _ult).days
            print(f"[email] ABORTADO: snapshot mais recente ({_ult}) < esperado ({_esp}). Atraso: {_dias}d.")
            print(f"[email] Motivo: pipeline nao atualizou ate {_esp}. Verifique ScrapAF3/ScrapB3/snapshot.")
            print(f"[email] Email NAO enviado pra evitar re-envio de dado velho.")
            return
    # ────────────────────────────────────────────────────────────────────────'''
    s = s.replace(marker, guard)
    f.write_text(s, encoding="utf-8")
    print("[ok] guard adicionado")
else:
    print("[skip] guard ja existe ou marker nao encontrado")
