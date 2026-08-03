"""
migrar_basefundos_naming.py
============================

Reescreve TODOS os parquets de BaseFundos/ com naming novo:
   DAP{YY} par   -> DAP_Q{YY}    (2030 -> DAP_Q30)
   DAP{YY} impar -> DAP_K{YY}    (2035 -> DAP_K35)
   DI_{YY}       -> DI_F{YY}     (todos janeiro)
   TREASURY, WDO1, NTNB*         (passam direto)

FAZ BACKUP AUTOMÁTICO antes de sobrescrever.

USO:
    python migrar_basefundos_naming.py               # dry-run (default): só lista mudanças
    python migrar_basefundos_naming.py --apply       # executa (com backup)
    python migrar_basefundos_naming.py --apply --base-dir "Z:\...\Projeto-Gestao-Risco"

Rollback:
    Basta copiar de volta o conteúdo da pasta BaseFundos/_backup_YYYYMMDD_HHMMSS/
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Mapa antigo -> novo (mesmo do cota_portfolio_core)
# ─────────────────────────────────────────────────────────────────────────────
_RE_DAP_NOVO   = re.compile(r"^DAP_[KQ]\d{2}$")
_RE_DAP_ANTIGO = re.compile(r"^DAP(\d{2})$")
_RE_DI_NOVO    = re.compile(r"^DI_[FJNV]\d{2}$")
_RE_DI_ANTIGO  = re.compile(r"^DI_(\d{2})$")


def alias_ativo_novo(ativo: str) -> str:
    s = str(ativo).strip()
    if _RE_DAP_NOVO.match(s) or _RE_DI_NOVO.match(s):
        return s
    m = _RE_DAP_ANTIGO.match(s)
    if m:
        yy = int(m.group(1))
        letra = "Q" if yy % 2 == 0 else "K"
        return f"DAP_{letra}{yy:02d}"
    m = _RE_DI_ANTIGO.match(s)
    if m:
        return f"DI_F{m.group(1)}"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def processar(base_dir: Path, apply: bool = False) -> int:
    """Retorna código de saída (0 = sucesso)."""
    bf_dir = base_dir / "BaseFundos"
    if not bf_dir.exists():
        print(f"[erro] Diretório não encontrado: {bf_dir}")
        return 2

    parquets = sorted([p for p in bf_dir.iterdir()
                       if p.is_file() and p.suffix.lower() == ".parquet"])
    if not parquets:
        print(f"[erro] Nenhum .parquet em {bf_dir}")
        return 2

    print(f"[info] {len(parquets)} parquets em {bf_dir}")
    print(f"[info] Modo: {'APLICAR (com backup)' if apply else 'DRY-RUN (nada será escrito)'}")
    print()

    if apply:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = bf_dir / f"_backup_{ts}"
        backup_dir.mkdir(exist_ok=False)
        print(f"[backup] Criado: {backup_dir}")
        print()

    total_renames = 0
    total_arquivos_afetados = 0

    for p in parquets:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"[skip] {p.name}: erro ao ler -> {e}")
            continue

        # Detecta a coluna de ativo — geralmente "Ativo"
        col_ativo = None
        for cand in ("Ativo", "ativo", "ATIVO"):
            if cand in df.columns:
                col_ativo = cand
                break

        if col_ativo is None:
            # sem coluna Ativo — pode ser algum arquivo especial; deixa
            print(f"[skip] {p.name}: sem coluna 'Ativo'")
            continue

        antigos = df[col_ativo].astype(str).tolist()
        novos   = [alias_ativo_novo(a) for a in antigos]

        diffs = [(a, b) for a, b in zip(antigos, novos) if a != b]
        if not diffs:
            print(f"[ok  ] {p.name}: 0 renames (já novo ou não aplicável)")
            continue

        total_arquivos_afetados += 1
        total_renames += len(diffs)

        print(f"[chg ] {p.name}: {len(diffs)} renames")
        for a, b in diffs[:10]:
            print(f"          {a:<12} -> {b}")
        if len(diffs) > 10:
            print(f"          ... (+{len(diffs)-10} mais)")

        if apply:
            # Backup do arquivo original
            shutil.copy2(p, backup_dir / p.name)
            # Escreve com naming novo
            df2 = df.copy()
            df2[col_ativo] = novos
            df2.to_parquet(p, index=False)
            print(f"          -> escrito: {p}")

    print()
    print(f"[resumo] {total_arquivos_afetados} arquivos alterados, {total_renames} renames no total.")
    if not apply:
        print(f"[resumo] Dry-run — nada foi escrito. Rode com --apply para efetivar.")
    else:
        print(f"[resumo] Backup em: {backup_dir}")
        print(f"[resumo] Rollback: Copy-Item \"{backup_dir}\\*.parquet\" \"{bf_dir}\" -Force")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="Aplica a migração (com backup). Sem essa flag, é dry-run.")
    ap.add_argument("--base-dir", default=None,
                    help="Diretório do dashboard (contém BaseFundos/). Default = CWD.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else Path.cwd()
    sys.exit(processar(base_dir, apply=args.apply))
