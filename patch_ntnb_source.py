from pathlib import Path
import shutil, datetime as dt
f = Path('TransformarRetornosParquet.py')
shutil.copy2(f, f'{f}.bak_ntnb_{dt.datetime.now():%Y%m%d_%H%M%S}')
s = f.read_text(encoding='utf-8')
old = "df_ntnb = pd.read_excel('Dados/FechamentoNTNBs.xlsx')"
new = "df_ntnb = pd.read_excel('Dados/FechamentoNTNBs.xlsx', sheet_name='Planilha1')  # PUs (Planilha2 tem yields)"
if old in s:
    s = s.replace(old, new)
    f.write_text(s, encoding='utf-8')
    print('OK: agora le Planilha1 (PUs)')
else:
    print('WARN: linha nao encontrada')
