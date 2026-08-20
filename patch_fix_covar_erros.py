"""patch_fix_covar_erros.py - remove tail_mask ref + hide antigo CoVaR completo"""
from pathlib import Path
import shutil, datetime as dt

# --- Fix 1: tail_mask no risco_carteira_core ---
f1 = Path("risco_carteira_core.py")
shutil.copy2(f1, f"{f1}.bak_tail_{dt.datetime.now():%Y%m%d_%H%M%S}")
s1 = f1.read_text(encoding="utf-8")
old = '        "n_scenarios_tail":      int(tail_mask.sum()),'
new = '        "n_scenarios_tail":      len(scenarios),'
if old in s1:
    s1 = s1.replace(old, new); f1.write_text(s1, encoding="utf-8")
    print("[ok] tail_mask -> len(scenarios) no return dict")

# --- Fix 2: encontrar range do bloco antigo CoVaR e fazer hide completo ---
f2 = Path("app4.py")
shutil.copy2(f2, f"{f2}.bak_covarhide_{dt.datetime.now():%Y%m%d_%H%M%S}")
s2 = f2.read_text(encoding="utf-8")
lines = s2.split("\n")

# Encontra "if False:" com "st.subheader(\"CoVaR por classe (legado BBG 5y)\")"
inicio = None
for i, l in enumerate(lines):
    if 'st.subheader("CoVaR por classe (legado BBG 5y)")' in l:
        # linha ANTERIOR deve ser "if False:"
        for j in range(i-1, max(0, i-5), -1):
            if "if False:" in lines[j]:
                inicio = j
                break
        break

if inicio is None:
    print("[warn] marker CoVaR legado nao encontrado")
else:
    # Localiza fim: proximo st.subheader que NAO seja o legado
    fim = None
    for i in range(inicio + 3, min(len(lines), inicio + 300)):
        l = lines[i]
        if 'st.subheader("Histórico de DV01 & CoVaR")' in l:
            fim = i
            break
    if fim is None:
        # tenta outro marker
        for i in range(inicio + 3, min(len(lines), inicio + 300)):
            if 'st.subheader(' in lines[i] and 'legado' not in lines[i]:
                fim = i
                break
    if fim is None:
        print("[warn] fim do bloco CoVaR antigo nao encontrado")
    else:
        # Reindenta linhas entre inicio+1 e fim (exclusive) pra ficar dentro do if False
        # Adiciona 4 espacos de indent nas linhas de codigo (que ja estao com 12 spaces)
        # Mais seguro: colocar tudo comentado ou envelopar em try/except
        # Solucao: adicionar linhas "if False:" que absorvam
        # Melhor: contar indent das linhas afetadas e reindentar
        alterado = 0
        for i in range(inicio + 1, fim):
            l = lines[i]
            # linha vazia mantem
            if not l.strip():
                continue
            # ja tem 12+ spaces? adiciona 4 pra ficar dentro do if False (16 spaces)
            # verificar indent atual
            stripped = l.lstrip()
            indent = len(l) - len(stripped)
            if indent >= 12:
                # esta no nivel do bloco - reindenta pra dentro do if False
                lines[i] = " " * 4 + l
                alterado += 1
        print(f"[ok] {alterado} linhas reindentadas dentro do 'if False:' (linhas {inicio+1}-{fim-1})")

s2_novo = "\n".join(lines)
f2.write_text(s2_novo, encoding="utf-8")
print(f"[done] app4.py atualizado")
