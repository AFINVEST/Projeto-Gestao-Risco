"""
aplicar_governance.py
======================

Aplica a regra de stop-loss vol-normalizado com base no snapshot mais
recente de snapshot_diario.

Regra (config_risco):
    vol_target             = 0.04     (4% a.a.)
    n_vols_stop_gatilho    = 1.0
    n_vols_stop_liberacao  = 0.5
    fator_reducao_stop     = 0.5

    Gatilho (aciona): DD ≤ -n_vols_gatilho × vol_target  (=  -4% com defaults)
    Liberação:        DD ≥ -n_vols_liberacao × vol_target  (= -2% com defaults)
    Histerese: uma vez acionado, o stop só libera quando DD volta ao gatilho de liberação

Ações:
    1. Lê snapshot mais recente (dd_atual)
    2. Lê governance_state atual (em_stop_loss)
    3. Aplica regra + histerese → decide novo estado
    4. Se estado mudou:
       - Atualiza governance_state
       - Grava eventos_risco (tipo=stop_ativado/stop_liberado)
       - Atualiza a linha de hoje em snapshot_diario com stop_loss_ativo e fator_governance
    5. Se não mudou: só atualiza o snapshot de hoje com o estado corrente

Rodar depois do gravar_snapshot_diario.py no .bat.

USO:
    python aplicar_governance.py
"""
from __future__ import annotations
import os
import sys
import json
from datetime import datetime, timezone

try:
    from supabase import create_client
except ImportError:
    print("ERRO: pip install supabase", file=sys.stderr)
    sys.exit(1)


DEFAULT_VOL_TARGET = 0.04
DEFAULT_N_VOLS_GATILHO = 1.0
DEFAULT_N_VOLS_LIBERACAO = 0.5
DEFAULT_FATOR_REDUCAO = 0.5


def _get_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Defina SUPABASE_URL e SUPABASE_KEY.")
    return create_client(url, key)


def _get_config(client) -> dict:
    resp = client.table("config_risco").select("parametro,valor").execute()
    out = {}
    for row in resp.data or []:
        v = row["valor"]
        if isinstance(v, str):
            v = v.strip('"')
        try:
            out[row["parametro"]] = float(v)
        except Exception:
            out[row["parametro"]] = v
    return out


def _load_last_snapshot(client) -> dict | None:
    resp = (client.table("snapshot_diario")
                  .select("Data,dd_atual,vol_60d,cota,pl_total,var_limite_base_bps")
                  .order("Data", desc=True)
                  .limit(1)
                  .execute())
    return resp.data[0] if resp.data else None


def _load_governance_state(client, regra: str = "stop_loss_dd") -> dict | None:
    resp = (client.table("governance_state")
                  .select("*")
                  .eq("regra", regra)
                  .execute())
    return resp.data[0] if resp.data else None


def _upsert_governance_state(client, regra, ativo, ativado_em, liberado_em, snapshot_disparo):
    payload = {
        "regra": regra,
        "ativo": ativo,
        "ativado_em": ativado_em,
        "liberado_em": liberado_em,
        "snapshot_disparo": snapshot_disparo,
        "atualizado_em": datetime.now(timezone.utc).isoformat(),
    }
    client.table("governance_state").upsert(payload, on_conflict="regra").execute()


def _insert_evento(client, tipo, severidade, titulo, payload):
    client.table("eventos_risco").insert({
        "tipo": tipo,
        "severidade": severidade,
        "titulo": titulo,
        "payload": payload,
    }).execute()


def _atualizar_snapshot_hoje(client, data_iso, stop_loss_ativo, fator_governance, var_limite_base_bps):
    payload = {
        "stop_loss_ativo": stop_loss_ativo,
        "fator_governance": float(fator_governance),
        "var_limite_efet_bps": float(var_limite_base_bps) * float(fator_governance),
    }
    (client.table("snapshot_diario")
           .update(payload)
           .eq("Data", data_iso)
           .execute())


def decidir_estado(dd_atual: float,
                   vol_target: float,
                   n_vols_gatilho: float,
                   n_vols_liberacao: float,
                   estado_atual: bool) -> tuple[bool, str | None]:
    """Aplica a regra + histerese. Retorna (novo_estado, motivo_mudanca)."""
    gatilho = -n_vols_gatilho * vol_target      # ex: -0.04
    libera = -n_vols_liberacao * vol_target      # ex: -0.02

    if not estado_atual:
        # Não estamos em stop — vê se aciona
        if dd_atual <= gatilho:
            return True, f"DD {dd_atual*100:.2f}% ≤ gatilho {gatilho*100:.2f}%"
        return False, None
    else:
        # Já em stop — vê se libera
        if dd_atual >= libera:
            return False, f"DD {dd_atual*100:.2f}% ≥ liberação {libera*100:.2f}%"
        return True, None


def run():
    client = _get_client()
    config = _get_config(client)

    vol_target = float(config.get("vol_target", DEFAULT_VOL_TARGET))
    n_vols_gatilho = float(config.get("n_vols_stop_gatilho", DEFAULT_N_VOLS_GATILHO))
    n_vols_liberacao = float(config.get("n_vols_stop_liberacao", DEFAULT_N_VOLS_LIBERACAO))
    fator_reducao = float(config.get("fator_reducao_stop", DEFAULT_FATOR_REDUCAO))
    var_base = float(config.get("var_limite_base_bps", 1.0))

    print(f"[governance] parâmetros: vol_target={vol_target}, "
          f"gatilho={-n_vols_gatilho*vol_target*100:.2f}%, "
          f"liberação={-n_vols_liberacao*vol_target*100:.2f}%, "
          f"fator_reducao={fator_reducao}")

    snap = _load_last_snapshot(client)
    if not snap:
        print("[governance] snapshot_diario vazio. Nada a fazer.")
        return

    data_iso = snap["Data"]
    dd_atual = float(snap.get("dd_atual") or 0.0)
    cota = snap.get("cota")
    print(f"[governance] snapshot: Data={data_iso}, dd_atual={dd_atual*100:.2f}%, cota={cota}")

    state = _load_governance_state(client) or {
        "regra": "stop_loss_dd", "ativo": False,
        "ativado_em": None, "liberado_em": None, "snapshot_disparo": None,
    }
    estado_atual = bool(state.get("ativo", False))
    print(f"[governance] estado atual: {'ATIVO' if estado_atual else 'inativo'}")

    novo_estado, motivo = decidir_estado(
        dd_atual, vol_target, n_vols_gatilho, n_vols_liberacao, estado_atual
    )
    fator_novo = fator_reducao if novo_estado else 1.0

    mudou = novo_estado != estado_atual

    if mudou:
        agora_iso = datetime.now(timezone.utc).isoformat()
        snapshot_disparo = {
            "dd_atual": dd_atual,
            "vol_target": vol_target,
            "gatilho": -n_vols_gatilho * vol_target,
            "liberacao": -n_vols_liberacao * vol_target,
            "cota": cota,
            "data_snapshot": data_iso,
            "motivo": motivo,
        }

        if novo_estado:
            # Acionou
            _upsert_governance_state(
                client, "stop_loss_dd", True, agora_iso, None, snapshot_disparo
            )
            _insert_evento(
                client, "stop_ativado", "critical",
                f"Stop-loss ativado — DD {dd_atual*100:.2f}%",
                snapshot_disparo
            )
            print(f"[governance] *** STOP-LOSS ATIVADO *** motivo: {motivo}")
        else:
            # Liberou
            _upsert_governance_state(
                client, "stop_loss_dd", False, state.get("ativado_em"), agora_iso, snapshot_disparo
            )
            _insert_evento(
                client, "stop_liberado", "warn",
                f"Stop-loss liberado — DD {dd_atual*100:.2f}%",
                snapshot_disparo
            )
            print(f"[governance] Stop-loss liberado — {motivo}")
    else:
        print(f"[governance] sem mudança de estado")

    # Sempre atualiza o snapshot de hoje com o estado corrente
    _atualizar_snapshot_hoje(client, data_iso, novo_estado, fator_novo, var_base)
    print(f"[governance] snapshot_diario atualizado: stop_loss_ativo={novo_estado}, "
          f"fator={fator_novo}, var_efet_bps={var_base*fator_novo}")


if __name__ == "__main__":
    run()
