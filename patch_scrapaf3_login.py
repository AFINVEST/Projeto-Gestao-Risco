"""patch_scrapaf3_login.py - espera o login concluir e le credenciais do .env."""
from pathlib import Path
import re, shutil, datetime as dt, sys

f = Path("ScrapAF3.py")
s = f.read_text(encoding="utf-8")
if "[af3-login]" in s:
    print("[skip] ja aplicado"); sys.exit(0)

trocas = [
    ("usuario",
     r'(\(By\.ID, "atributo"\)\)\)\.send_keys\()"[^"]*"(\))',
     r'\1os.environ["AF_USUARIO"]\2'),
    ("senha",
     r'(find_element\(By\.ID, "passwordLogin"\)\.send_keys\()"[^"]*"(\))',
     r'\1os.environ["AF_SENHA"]\2'),
    ("espera login",
     r'^(\s*)(driver\.find_element\(By\.ID, "loginInterno"\)\.click\(\))\n',
     r'\1\2\n'
     r'\1# [af3-login] espera sair da tela de login antes de navegar\n'
     r'\1try:\n'
     r'\1    WebDriverWait(driver, 30).until(lambda drv: "/login" not in drv.current_url)\n'
     r'\1except Exception:\n'
     r'\1    raise RuntimeError("Login no portal AF falhou (continua na tela de login) - verificar AF_USUARIO/AF_SENHA no .env")\n'),
]
for nome, pad, rep in trocas:
    s, n = re.subn(pad, rep, s, count=1, flags=re.M)
    if n != 1:
        print(f"[ERRO] ancora '{nome}' nao encontrada - nada alterado"); sys.exit(1)
if not re.search(r"^import os\b", s, flags=re.M):
    s = "import os\n" + s
shutil.copy2(f, f"{f}.bak_af3login_{dt.datetime.now():%Y%m%d_%H%M%S}")
f.write_text(s, encoding="utf-8")
print("[ok] ScrapAF3: espera do login + credenciais via .env")
