"""patch_fase2c_cleanup.py - cleanup final DV01 duplicado"""
from pathlib import Path
import shutil, datetime as dt

f = Path("app4.py")
shutil.copy2(f, f"{f}.bak_cleanup_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")
lines = s.split("\n")

# --- Fix 1: Deletar bloco DV01 duplicado (linhas 8793-8848) ---
# Substituir por comentario simples
inicio_bloco1 = 8793 - 1  # index 0-based
fim_bloco1 = 8848  # exclusive
novo_bloco1 = [
    "    # REMOVED Fase 2c cleanup: DV01 duplicado (agora so aparece ao lado dos donuts em col11)",
    "    with tab_orcamento:",
    "        with COL1:",
    "            pass  # bloco anterior removido",
]
# Substitui as linhas
lines_novo = lines[:inicio_bloco1] + novo_bloco1 + lines[fim_bloco1:]

# --- Fix 2: Corrigir o `if False: _dv_dict_old = {} else:` (linha 9206) ---
# Mudar `if False:` para `if True:` — assim a True branch executa (nao faz nada) e o else e pulado
# Ajusta o indice pela reducao de linhas do Fix 1
delta = len(novo_bloco1) - (fim_bloco1 - inicio_bloco1)
linha_if_false = 9206 - 1 + delta   # novo indice

if 0 <= linha_if_false < len(lines_novo):
    if "if False:" in lines_novo[linha_if_false]:
        lines_novo[linha_if_false] = lines_novo[linha_if_false].replace("if False:", "if True:")
        print(f"[ok] linha {linha_if_false+1}: if False -> if True (skip antigo DV01)")
    else:
        print(f"[warn] linha {linha_if_false+1} nao contem 'if False:' — conteudo: {lines_novo[linha_if_false][:80]}")

s_novo = "\n".join(lines_novo)
f.write_text(s_novo, encoding="utf-8")
print(f"[done] Reduzido {fim_bloco1 - inicio_bloco1} linhas -> {len(novo_bloco1)} linhas (delta {delta})")
