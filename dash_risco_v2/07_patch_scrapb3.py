"""
07_patch_scrapb3.py
====================

Patcheador idempotente do ScrapB3_v2.py existente. Faz 3 mudanças:

1) mapear_asset:
   - DAPyy → DAP_Qyy (par) ou DAP_Kyy (ímpar) [antes retornava só DAPyy]
   - DI_yy → DI_Fyy / DI_Jyy / DI_Nyy / DI_Vyy conforme letra [antes só DI_yy]

2) selecionar_vertices:
   - Aceita todos os vencimentos F, J, N, V do DI (antes filtrava só F)

3) main() → append: escreve em precos_diarios (Supabase) após salvar os parquets
   locais. Se falhar (sem env, offline), grava warning no log mas não quebra.

USO:
    cd Z:\\...\\Projeto-Gestao-Risco
    python dash_risco_v2\\07_patch_scrapb3.py

    # Cria backup ScrapB3_v2.py.bak_pre_v2b antes de qualquer edição.
    # Rodar 2x = no-op (procura por marcadores).
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path


ARQUIVO = Path("ScrapB3_v2.py")
BKP     = Path("ScrapB3_v2.py.bak_pre_v2b")
MARCADOR_FIM_MAIN = "# =====  Envio para Supabase (precos_diarios) — v2b  ====="


def apply_edit(text: str, label: str, old: str, new: str, esperado: int = 1) -> str:
    n = text.count(old)
    if n == 0:
        print(f"  [{label}] BLOCO ANTIGO NÃO ENCONTRADO — pode já ter sido aplicado.")
        return text
    if n != esperado:
        raise RuntimeError(f"[{label}] esperado {esperado} matches, encontrado {n}")
    print(f"  [{label}] aplicando substituição")
    return text.replace(old, new, esperado)


def main():
    if not ARQUIVO.exists():
        print(f"ERRO: {ARQUIVO} não encontrado. Rode na raiz do projeto.", file=sys.stderr)
        sys.exit(1)

    if not BKP.exists():
        print(f"Backup criado: {BKP}")
        shutil.copy2(ARQUIVO, BKP)
    else:
        print(f"Backup já existe (não sobrescrevendo): {BKP}")

    text = ARQUIVO.read_text(encoding="utf-8")
    orig_len = len(text)

    # ----- Patch 1: mapear_asset (DAP + DI naming novo) -----
    text = apply_edit(
        text, "PATCH 1a (mapear_asset DAP)",
        old='''        if ano % 2 == 0:
            if letra != "Q":
                return None
        else:
            if letra != "K":
                return None

        return f"DAP{ano_str}"''',
        new='''        if ano % 2 == 0:
            if letra != "Q":
                return None
        else:
            if letra != "K":
                return None

        return f"DAP_{letra}{ano_str}"    # v2b naming: DAP_Q26, DAP_K27, ...''',
    )

    text = apply_edit(
        text, "PATCH 1b (mapear_asset DI)",
        old='''    if name == "DI1Day":
        sufixo = venc[-2:]
        try:
            ano = int(sufixo)
        except ValueError:
            return None
        if not (26 <= ano):
            return None
        return f"DI_{sufixo}"''',
        new='''    if name == "DI1Day":
        if len(venc) < 3:
            return None
        letra_di = venc[0]
        sufixo   = venc[-2:]
        try:
            ano = int(sufixo)
        except ValueError:
            return None
        if not (26 <= ano):
            return None
        # v2b: aceita F, J, N, V (jan, abr, jul, out) e nomeia com letra
        if letra_di not in ("F", "J", "N", "V"):
            return None
        return f"DI_{letra_di}{sufixo}"''',
    )

    # ----- Patch 2: selecionar_vertices (aceitar F/J/N/V para DI) -----
    text = apply_edit(
        text, "PATCH 2 (selecionar_vertices DI)",
        old='''    # DI: keep apenas mês "F" (Jan)
    di_mask = df["Name"].eq("DI1Day")
    df = df[~di_mask | df["Vencimento"].astype(str).str.startswith("F", na=False)].copy()''',
        new='''    # v2b: DI aceita F, J, N, V (jan, abr, jul, out)
    di_mask = df["Name"].eq("DI1Day")
    def _di_letra_ok(v):
        v = str(v)
        return len(v) >= 3 and v[0] in ("F", "J", "N", "V")
    df = df[~di_mask | df["Vencimento"].map(_di_letra_ok)].copy()''',
    )

    # ----- Patch 3: adicionar bloco de envio pro Supabase no fim do main() -----
    if MARCADOR_FIM_MAIN in text:
        print("  [PATCH 3 (supabase upsert)] JÁ APLICADO — pulando")
    else:
        bloco_novo = f'''
    {MARCADOR_FIM_MAIN}
    try:
        _enviar_para_supabase(wide_preco, wide_valor)
    except Exception as _e_supa:
        _append_log(f"[warn] envio Supabase falhou: {{_e_supa}}")
'''
        marcador_main = '''    # 7) JSON pt-BR (texto garantido) — missing vira ""'''
        text = apply_edit(
            text, "PATCH 3 (append no main)",
            old=marcador_main,
            new=bloco_novo + "\n" + marcador_main,
        )

        # Adiciona a função _enviar_para_supabase ao final do arquivo
        func_supabase = '''

# =============================================================
# v2b: espelho no Supabase da tabela precos_diarios
# =============================================================
def _enviar_para_supabase(wide_preco, wide_valor):
    """Faz upsert em precos_diarios das colunas de data recentes.
    Não falha se supabase-py não estiver instalado ou env não configurada."""
    import math
    import os as _os
    try:
        from supabase import create_client
    except ImportError:
        _append_log("[supabase] pacote supabase não instalado — pulando upsert.")
        return

    url = _os.environ.get("SUPABASE_URL")
    key = _os.environ.get("SUPABASE_KEY") or _os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        _append_log("[supabase] SUPABASE_URL/KEY não configuradas — pulando upsert.")
        return

    NOMINAL = 100_000.0
    DIAS_ANO = 252

    # Carrega tabela de vencimentos padrão pra derivar DU/taxa dos DIs/DAPs
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path("dash_risco_v2")))
        import dv01_dinamico as _dv
        _feriados = _dv.load_feriados()
    except Exception as _e:
        _append_log(f"[supabase] dv01_dinamico indisponível ({_e}) — usando PU só (sem Taxa derivada).")
        _dv = None
        _feriados = None

    sb = create_client(url, key)

    # Só envia as últimas ~30 colunas (backfill inicial já foi feito por script à parte)
    def _colunas_recentes(df, n=30):
        if df is None or df.empty:
            return []
        cols = sorted(df.columns, key=lambda c: pd.to_datetime(c))
        return cols[-n:]

    registros = []
    for col_data in _colunas_recentes(wide_preco, n=30):
        data_iso = pd.to_datetime(col_data).date().isoformat()
        for ativo, pu in wide_preco[col_data].items():
            if pd.isna(pu):
                continue
            try:
                pu_f = float(pu)
            except Exception:
                continue
            if pu_f <= 0 or math.isnan(pu_f) or math.isinf(pu_f):
                continue

            taxa = None
            if _dv is not None and str(ativo).startswith(("DI_", "DAP_")):
                try:
                    venc = _dv.vencimento(str(ativo), _feriados)
                    du   = _dv.networkdays(pd.Timestamp(data_iso), venc, _feriados)
                    if du > 0:
                        taxa = ((NOMINAL / pu_f) ** (DIAS_ANO / du) - 1) * 100
                except Exception:
                    taxa = None

            registros.append({
                "Data":      data_iso,
                "Ativo":     str(ativo),
                "PU_ajuste": pu_f,
                "Taxa":      taxa,
                "Fonte":     "b3",
            })

    if not registros:
        _append_log("[supabase] nenhum registro para enviar.")
        return

    BATCH = 500
    total = 0
    for i in range(0, len(registros), BATCH):
        lote = registros[i:i + BATCH]
        (sb.table("precos_diarios")
           .upsert(lote, on_conflict="Data,Ativo")
           .execute())
        total += len(lote)
    _append_log(f"[supabase] enviados {total} registros em precos_diarios.")
'''
        text = text.rstrip() + func_supabase + "\n"

    if len(text) == orig_len:
        print("Nenhuma mudança aplicada (talvez o patch já estivesse instalado).")
    else:
        ARQUIVO.write_text(text, encoding="utf-8")
        delta = len(text) - orig_len
        print(f"\nOK — arquivo atualizado. Delta: +{delta} bytes.")
        print(f"Backup original em: {BKP}")


if __name__ == "__main__":
    main()
