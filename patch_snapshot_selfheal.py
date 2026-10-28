"""patch_snapshot_selfheal.py - reprocessa ultimos 3 dias e preserva governance"""
from pathlib import Path
import shutil, datetime as dt
f = Path("gravar_snapshot_diario.py")
shutil.copy2(f, f"{f}.bak_selfheal_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
n = 0

# 1) incremental + reprocessa ultimos 3 dias ja gravados
old1 = "        datas_a_processar = [d for d in cotas.index if d.date().isoformat() not in datas_existentes]"
new1 = old1 + """
        # self-heal: reprocessa os ultimos 3 dias ja gravados (trade/preco que chegou atrasado)
        _ja_gravados = [d for d in cotas.index if d.date().isoformat() in datas_existentes][-3:]
        datas_a_processar = sorted(set(datas_a_processar) | set(_ja_gravados))"""
if old1 in s: s = s.replace(old1, new1); n += 1; print("[ok] reprocessa ultimos 3 dias")

# 2) dia ja existente: preserva campos de governance e DV01 de dias anteriores
old2 = "            registros.append(snap)"
new2 = """            if snap["Data"] in datas_existentes:
                # nao sobrescreve o que o governance gravou (stop/limite efetivo)
                for _k in ("var_limite_base_bps", "var_limite_efet_bps", "stop_loss_ativo", "fator_governance"):
                    snap.pop(_k, None)
                # DV01 so e calculado para a ultima data; nao apaga o dos dias anteriores
                if data != data_ultima:
                    for _k in ("dv01_total", "dv01_juros_nom", "dv01_juros_real", "dv01_treasury", "dv01_ntnb", "dv01_por_ativo"):
                        snap.pop(_k, None)
            registros.append(snap)"""
if old2 in s: s = s.replace(old2, new2); n += 1; print("[ok] preserva governance/DV01 em dias existentes")

# 3) upsert agrupado por conjunto de colunas (PostgREST exige chaves homogeneas no lote)
old3 = """    print(f"[snapshot] enviando {len(registros)} snapshots...")
    n = _upsert_snapshots(client, registros)"""
new3 = """    print(f"[snapshot] enviando {len(registros)} snapshots...")
    _grupos = {}
    for _r in registros:
        _grupos.setdefault(tuple(sorted(_r.keys())), []).append(_r)
    n = sum(_upsert_snapshots(client, _g) for _g in _grupos.values())"""
if old3 in s: s = s.replace(old3, new3); n += 1; print("[ok] upsert agrupado")

f.write_text(s, encoding="utf-8")
print(f"[done] {n}/3")
