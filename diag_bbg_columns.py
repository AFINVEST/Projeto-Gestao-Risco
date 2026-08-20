"""diag_bbg_columns.py - lista colunas BBG e mapeia por NOME para nomes canonicos"""
import pandas as pd
import re

df = pd.read_excel('Dados/BBG - ECO DASH.xlsx', sheet_name='BZ RATES',
                   skiprows=1, thousands='.', decimal=',')
df.drop(['Unnamed: 0','Unnamed: 1','Unnamed: 2','Unnamed: 3','Unnamed: 25'],
        axis=1, inplace=True, errors='ignore')
df = df.drop([0], errors='ignore')

print(f"Total colunas apos drops: {len(df.columns)}")
print()
print("COLUNAS RAW + MAPEAMENTO PROPOSTO:")
print(f"{'idx':>3}  {'raw':<38}  {'canonical':<12}  {'sample':>15}")
print("-" * 78)

# Mapeamento por regex do codigo BBG -> nome canonico
def mapear(raw_col):
    s = str(raw_col).strip().upper()
    # DIs: ODF{yy} Comdty, ODJ, ODN, ODV
    m = re.match(r"^OD([FJNV])(\d{2})\s+COMDTY", s)
    if m:
        return f"DI_{m.group(1)}{m.group(2)}"
    # DAPs: WLQ{yy} Index (Q=agosto par), WLK{yy} Index (K=maio impar)
    m = re.match(r"^WL([QK])(\d{2})\s+INDEX", s)
    if m:
        return f"DAP_{m.group(1)}{m.group(2)}"
    # Outros
    if s.startswith("WDO1"): return "WDO1"
    if s.startswith("OI1"):  return "TREASURY"
    if s.startswith("BZ1"):  return "IBOV"
    if s.startswith("WSP1"): return None  # descarta
    # NTNBs: BNTNB 6 MM/DD/YYYY Govt -> extrai ano
    m = re.match(r"^BNTNB\s+\d+\s+\d+/\d+/(\d{4})", s)
    if m:
        yy = int(m.group(1)) % 100
        return f"NTNB{yy:02d}"
    return None

for i, raw in enumerate(df.columns):
    if i == 0:
        print(f"{i:>3}  {str(raw):<38}  {'Date':<12}")
        continue
    canonical = mapear(raw)
    v = pd.to_numeric(df[raw], errors='coerce').dropna()
    sample = f"{v.iloc[0]:.2f}" if len(v) > 0 else "VAZIA"
    print(f"{i:>3}  {str(raw):<38}  {str(canonical or '(descarta)'):<12}  {sample:>15}")

print()
print("SUMARIO:")
canonicals = [mapear(c) for c in df.columns[1:]]
tem = [c for c in canonicals if c is not None]
descarta = sum(1 for c in canonicals if c is None)
print(f"  Ativos mapeaveis: {len(tem)}")
print(f"  Colunas descartadas: {descarta}")
print(f"  Ativos por classe:")
from collections import Counter
def classe(a):
    if a.startswith("DI_"): return "DI"
    if a.startswith("DAP"): return "DAP"
    if a.startswith("NTNB"): return "NTNB"
    return "OUTROS"
cnt = Counter([classe(c) for c in tem])
for k, v in sorted(cnt.items()):
    print(f"    {k}: {v}")
