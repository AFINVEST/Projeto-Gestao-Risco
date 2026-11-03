"""patch_wdo_fase1.py - WDO por vencimento (pipeline + core)."""
from pathlib import Path
import re, shutil, datetime as dt

TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
MARK = "# [wdo-venc]"

def patch(arquivo, trocas):
    f = Path(arquivo)
    s = f.read_text(encoding="utf-8")
    if MARK in s:
        print(f"[skip] {arquivo}: ja aplicado"); return
    for nome, padrao, novo in trocas:
        s2, n = re.subn(padrao, novo, s, count=1, flags=re.M)
        if n == 0:
            print(f"[ERRO] {arquivo}: ancora '{nome}' nao encontrada. Arquivo NAO alterado."); return
        s = s2
    shutil.copy2(f, f"{f}.bak_wdo_{TS}")
    f.write_text(s, encoding="utf-8")
    print(f"[ok] {arquivo}: {len(trocas)} alteracoes")

patch("ScrapB3_v2.py", [
    ("concat WDO1",
     r'^(\s*)df = pd\.concat\(\[df\[df\["Asset"\]\.ne\("WDO1"\)\], best\], ignore_index=True\)\n',
     r'\1' + MARK + r' contratos especificos p/ operacao e P&L (WDO_X26, WDO_Z26, ...)\n'
     r'\1espec = use[use["Prefix"].eq("WDO")].sort_values("MatDate2").head(3).copy()\n'
     r'\1espec["Asset"] = "WDO_" + espec["Vencimento"].astype(str).str.upper()\n'
     r'\1df = pd.concat([df[df["Asset"].ne("WDO1")], best, espec], ignore_index=True)\n'),
])

patch("cota_portfolio_core.py", [
    ("x10 WDO1",
     r'^(\s*)df\.loc\[df\["Assets"\] == "WDO1",\s*df\.columns != "Assets"\] \*= 10\n',
     r'\1df.loc[df["Assets"].astype(str).str.startswith("WDO"), df.columns != "Assets"] *= 10  ' + MARK + r'\n'),
    ("custo WDO1", r'elif raiz == "WDO1":', r'elif raiz in ("WDO1", "WDO"):'),
])

patch("risco_carteira_core.py", [
    ("def carregar_precos_atuais",
     r'^def carregar_precos_atuais\(',
     MARK + '\ndef ativo_risco(ativo) -> str:\n'
     '    """Contrato especifico de dolar (WDO_X26) -> serie generica WDO1 para risco."""\n'
     '    a = str(ativo).strip()\n'
     '    return "WDO1" if a.upper().startswith("WDO") else a\n\n\n'
     'def carregar_precos_atuais('),
    ("preco WDO x10",
     r'^(\s*)out\[ativo\] = v\n',
     r'\1out[ativo] = v * 10 if ativo.upper().startswith("WDO") else v\n'),
    ("qty por ativo_risco",
     r'qty\[ativo\] = qty\.get\(ativo, 0\.0\) \+ float\(v\)',
     r'qty[ativo_risco(ativo)] = qty.get(ativo_risco(ativo), 0.0) + float(v)'),
])

patch("atualizar_retornos_diarios.py", [
    ("rename Ativo",
     r'^(\s*)df_long = df_long\.rename\(columns=\{"Assets": "Ativo"\}\)\n',
     r'\1df_long = df_long.rename(columns={"Assets": "Ativo"})\n'
     r'\1df_long = df_long[~df_long["Ativo"].astype(str).str.startswith("WDO_")]  ' + MARK + r'\n'),
])
