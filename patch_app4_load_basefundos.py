"""
patch_app4_load_basefundos.py  (v2 — reconhece assinatura atual com ttl=120)
============================================================================

Adiciona alias antigo->novo em `load_basefundos` do app4.py.
Idempotente: se já foi aplicado, não faz nada.
Faz backup antes de sobrescrever.

USO:
    python patch_app4_load_basefundos.py            # dry-run
    python patch_app4_load_basefundos.py --apply
    python patch_app4_load_basefundos.py --apply --file app4_atual.py

Rollback: copiar de volta app4.py.bak_YYYYMMDD_HHMMSS
"""
from __future__ import annotations
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


# Bloco atual do app4.py (linhas ~5463–5475)
ORIGINAL_FUNC = '''@st.cache_data(show_spinner=False, ttl=120)
def load_basefundos() -> dict[str, pd.DataFrame]:
    """Lê o estado dos fundos primariamente do Supabase (fallback local)."""
    out = load_basefundos_supabase()
    if not out:
        out = _load_basefundos_local()
    fixed = {}
    for nome, df in out.items():
        if "Ativo" in df.columns:
            fixed[nome] = df.set_index("Ativo")
        else:
            fixed[nome] = df
    return fixed'''


PATCHED_FUNC = '''# ── alias antigo -> novo (idempotente) — patch fase2c ───────────────────────
import re as _re_bf
_RE_DAP_NOVO_BF   = _re_bf.compile(r"^DAP_[KQ]\\d{2}$")
_RE_DAP_ANTIGO_BF = _re_bf.compile(r"^DAP(\\d{2})$")
_RE_DI_NOVO_BF    = _re_bf.compile(r"^DI_[FJNV]\\d{2}$")
_RE_DI_ANTIGO_BF  = _re_bf.compile(r"^DI_(\\d{2})$")

def _alias_ativo_novo_bf(a) -> str:
    s = str(a).strip()
    if _RE_DAP_NOVO_BF.match(s) or _RE_DI_NOVO_BF.match(s):
        return s
    m = _RE_DAP_ANTIGO_BF.match(s)
    if m:
        yy = int(m.group(1))
        return f"DAP_{'Q' if yy % 2 == 0 else 'K'}{yy:02d}"
    m = _RE_DI_ANTIGO_BF.match(s)
    if m:
        return f"DI_F{m.group(1)}"
    return s
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=120)
def load_basefundos() -> dict[str, pd.DataFrame]:
    """Lê o estado dos fundos primariamente do Supabase (fallback local).
    Aplica alias antigo->novo (fase2c) — idempotente."""
    out = load_basefundos_supabase()
    if not out:
        out = _load_basefundos_local()
    fixed = {}
    for nome, df in out.items():
        if "Ativo" in df.columns:
            fixed[nome] = df.set_index("Ativo")
        else:
            fixed[nome] = df
        # alias in-memory (blindagem contra naming antigo em qualquer fonte)
        try:
            fixed[nome].index = fixed[nome].index.map(_alias_ativo_novo_bf)
        except Exception:
            pass
    return fixed'''


SENTINEL = "_alias_ativo_novo_bf"


def processar(app4_path: Path, apply: bool = False) -> int:
    if not app4_path.exists():
        print(f"[erro] {app4_path} não existe.")
        return 2

    content = app4_path.read_text(encoding="utf-8")

    if SENTINEL in content:
        print(f"[ok] {app4_path.name} já foi patchado anteriormente. Nada a fazer.")
        return 0

    if ORIGINAL_FUNC not in content:
        print(f"[erro] Bloco original de load_basefundos NÃO encontrado em {app4_path.name}.")
        print("[erro] Assinatura procurada:")
        print("---")
        print(ORIGINAL_FUNC)
        print("---")
        print("Rode 'Select-String -Path app4.py -Pattern \"def load_basefundos\" -Context 0,15'")
        print("para me mostrar o bloco atual.")
        return 3

    novo_content = content.replace(ORIGINAL_FUNC, PATCHED_FUNC)
    if novo_content == content:
        print("[erro] Substituição resultou em conteúdo idêntico. Aborta.")
        return 4

    delta = PATCHED_FUNC.count("\n") - ORIGINAL_FUNC.count("\n")

    print(f"[info] Bloco encontrado. Substituição adiciona +{delta} linhas.")
    print(f"[info] Modo: {'APLICAR (com backup)' if apply else 'DRY-RUN'}")

    if apply:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = app4_path.with_name(f"{app4_path.name}.bak_{ts}")
        shutil.copy2(app4_path, backup)
        print(f"[backup] {backup}")
        app4_path.write_text(novo_content, encoding="utf-8")
        print(f"[ok] {app4_path} atualizado.")
    else:
        print("[dry-run] Nada foi escrito. Rode com --apply para efetivar.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--file", default="app4.py",
                    help="Nome do arquivo app4 (default: app4.py). Use 'app4_atual.py' para o backup.")
    args = ap.parse_args()

    sys.exit(processar(Path(args.file), apply=args.apply))
