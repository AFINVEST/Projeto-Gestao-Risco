"""patch_move_covar_hist.py - move CoVaR historico Portfolio Atual -> Historico Carteira"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_moveh_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
lines = s.split("\n")

# 1) Extrai bloco (linhas 9364-9413, 1-indexed = index 9363-9412 exclusive 9413)
SRC_START = 9363
SRC_END = 9414   # exclusive (deleta ate a linha 9413 inclusive)
bloco = lines[SRC_START:SRC_END]

# 2) Reindent: bloco esta com ~20 spaces, alvo tem ~12 spaces => remove 8 spaces
def dedent(l, n=8):
    if l.strip() == "": return l
    prefix = l[:n]
    if prefix == " " * n:
        return l[n:]
    return l   # se nao tem indent suficiente, mantem
bloco_reind = [dedent(l, 8) for l in bloco]

# 3) Deleta do local original (Portfolio Atual)
del lines[SRC_START:SRC_END]

# 4) Reencontra linha de insercao (indice muda apos delete)
INS_IDX = None
for i, l in enumerate(lines):
    if "st.plotly_chart(_fig_var, use_container_width=True)" in l:
        INS_IDX = i + 1   # inserir DEPOIS da linha
        break

if INS_IDX is None:
    print("[erro] linha de insercao nao encontrada")
else:
    # Insere bloco reindentado
    lines[INS_IDX:INS_IDX] = [""] + bloco_reind
    print(f"[ok] {len(bloco_reind)} linhas movidas para apos linha {INS_IDX} (Historico Carteira)")
    print(f"     Preview inicio:")
    for l in bloco_reind[:5]: print(f"       | {l[:80]}")

f.write_text("\n".join(lines), encoding="utf-8")
print("[done]")
