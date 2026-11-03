"""patch_wdo_fase2b_app4.py - rolagem do WDO dentro de 'Adicionar Ativos' (item 'ROLAGEM WDO')."""
from pathlib import Path
import shutil, datetime as dt, sys

ARQ = Path("app4.py")
MARK = "# [wdo-rolagem-adic]"
raw = ARQ.read_bytes()
crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")
if MARK in s:
    print("[skip] ja aplicado"); sys.exit(0)
if "# [wdo-venc]" not in s:
    print("[ERRO] rode antes o patch_wdo_fase2_app4.py"); sys.exit(1)

TROCAS = [
    ("radio", '"Simular Cota", "Rolar WDO"], index=3', '"Simular Cota"], index=3', 1),
    ("branch", '    elif opti == "Rolar WDO":  # [wdo-venc]\n        render_rolagem_wdo()\n\n', '', 1),
    ("lista", "        lista_todos_ativos = list(df.columns)\n",
     "        lista_todos_ativos = list(df.columns)\n"
     "        lista_todos_ativos.append(\"ROLAGEM WDO\")  " + MARK + "\n", 1),
    ("fluxo",
     "        if len(assets) > 0:\n            df_precos, df_completo = load_and_process_excel(df, assets)\n",
     "        if \"ROLAGEM WDO\" in assets:  " + MARK + "\n"
     "            if len(assets) > 1:\n"
     "                st.sidebar.info(\"A rolagem é lançada sozinha: os demais ativos selecionados foram ignorados.\")\n"
     "            render_rolagem_wdo()\n"
     "            st.stop()\n\n"
     "        if len(assets) > 0:\n            df_precos, df_completo = load_and_process_excel(df, assets)\n", 1),
    ("alerta", "Lance a rolagem (opção 'Rolar WDO') ou a liquidação.",
     "Lance a rolagem ('ROLAGEM WDO' em Adicionar Ativos) ou a liquidação.", 1),
]
erros = []
for nome, old, new, n in TROCAS:
    c = s.count(old)
    if c != n:
        erros.append(f"  - '{nome}': esperado {n}, encontrado {c}"); continue
    s = s.replace(old, new)
if erros:
    print("[ERRO] app4.py NAO alterado:\n" + "\n".join(erros)); sys.exit(1)
if sys.version_info >= (3, 12):
    compile(s, "app4.py", "exec")
bak = f"app4.py.bak_wdo2b_{dt.datetime.now():%Y%m%d_%H%M%S}"
shutil.copy2(ARQ, bak)
with open(ARQ, "w", encoding="utf-8", newline="\r\n" if crlf else "\n") as f:
    f.write(s)
print(f"[ok] app4.py: {len(TROCAS)} alteracoes (backup {bak})")
