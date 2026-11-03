"""patch_wdo_fase2_app4.py - WDO por vencimento no app4.
- Risco (VaR/CoVaR/vol/DV01/stress): WDO_* agregado em WDO1
- Adicionar Ativos: escolhe o contrato exato (WDO_X26...) e grava a operacao nele
- Nova opcao 'Rolar WDO': rolagem por fundo (sai do contrato atual, entra no novo)
- Alerta de vencimento em Ver Portfolio / Rolar WDO
"""
from pathlib import Path
import shutil, datetime as dt, sys

ARQ = Path("app4.py")
MARK = "# [wdo-venc]"
raw = ARQ.read_bytes()
crlf = b"\r\n" in raw
s = raw.decode("utf-8").replace("\r\n", "\n")
if MARK in s:
    print("[skip] app4.py: patch WDO ja aplicado"); sys.exit(0)

HELPERS = r'''
# [wdo-venc] ---------------------------------------------------------------
# Risco usa a serie generica WDO1; operacao/P&L usam o contrato exato WDO_<venc>.
_WDO_MES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
            "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


def ativo_risco(ativo) -> str:
    """WDO_X26 / WDO_Z26 ... -> WDO1 (serie usada no risco). Demais ativos inalterados."""
    a = str(ativo).strip()
    return "WDO1" if a.upper().startswith("WDO") else a


def wdo_vencimento(ativo):
    """WDO_X26 -> date(2026, 11, 1). None se nao for contrato especifico."""
    import re as _re
    m = _re.match(r"^WDO_([FGHJKMNQUVXZ])(\d{2})$", str(ativo).strip().upper())
    if not m:
        return None
    return datetime.date(2000 + int(m.group(2)), _WDO_MES[m.group(1)], 1)


def wdo_ultimo_pregao(ativo):
    """Ultimo dia de negociacao: ultimo dia util do mes anterior ao vencimento (sem feriados)."""
    v = wdo_vencimento(ativo)
    if v is None:
        return None
    d = v - datetime.timedelta(days=1)
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d


def wdo_contratos_disponiveis(data_ref) -> list:
    """Contratos WDO_* com ajuste B3 na data e ainda negociaveis nela (ordenados por vencimento)."""
    try:
        df = pd.read_parquet("Dados/df_preco_de_ajuste_atual_completo.parquet")
    except Exception:
        return []
    d = str(data_ref)[:10]
    if d not in df.columns:
        return []
    d_dt = pd.to_datetime(d).date()
    out = []
    for _, r in df[df["Assets"].astype(str).str.startswith("WDO_")].iterrows():
        v = r[d]
        if v is None or str(v).strip() in ("", "nan", "None", "<NA>"):
            continue
        lp = wdo_ultimo_pregao(r["Assets"])
        if lp is not None and lp >= d_dt:
            out.append(str(r["Assets"]))
    return sorted(set(out), key=wdo_vencimento)


def _agrupar_wdo_pivot(p):
    """Pivot Fundo x Ativo: soma as colunas WDO_* dentro de WDO1."""
    if p is None or getattr(p, "empty", True):
        return p
    cols = [c for c in p.columns if str(c).upper().startswith("WDO")]
    if not cols or cols == ["WDO1"]:
        return p
    p = p.copy()
    tot = p[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    p = p.drop(columns=cols)
    p["WDO1"] = tot
    return p


def _agrupar_wdo_contratos(df):
    """Colunas 'Contratos WDO_*' somadas em 'Contratos WDO1'."""
    cols = [c for c in df.columns if str(c).startswith("Contratos WDO")]
    if not cols or cols == ["Contratos WDO1"]:
        return df
    df = df.copy()
    tot = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    df = df.drop(columns=cols)
    df["Contratos WDO1"] = tot
    return df
# [wdo-venc] fim helpers ---------------------------------------------------


def load_and_process_excel('''

ROLAGEM = r'''
# [wdo-venc] ---------------------------------------------------------------
def alerta_vencimento_wdo(dias_alerta: int = 3):
    """Avisa posicoes em WDO_* perto do ultimo pregao (ou ja vencidas)."""
    try:
        df = load_posicoes_por_fundo()
        if df is None or df.empty:
            return
        df = df.copy()
        df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0)
        pos = df[df["Ativo"].astype(str).str.startswith("WDO_")].groupby("Ativo")["Quantidade"].sum()
        pos = pos[pos.abs() > 1e-9]
        if pos.empty:
            return
        hoje = pd.to_datetime([c for c in load_b3_prices().columns if c != "Assets"][-1]).date()
        for a, q in pos.items():
            lp = wdo_ultimo_pregao(a)
            if lp is None:
                continue
            if lp < hoje:
                st.error(f"⚠️ {a}: {q:+.0f} contratos em aberto, mas o último pregão foi {lp:%d/%m/%Y}. "
                         f"Lance a rolagem (opção 'Rolar WDO') ou a liquidação.")
            else:
                du = int(np.busday_count(hoje, lp))
                if du <= dias_alerta:
                    st.warning(f"{a}: {q:+.0f} contratos em aberto. Último pregão em {lp:%d/%m/%Y} "
                               f"({du} dia(s) útil(eis)). Programe a rolagem.")
    except Exception as e:
        print(f"[alerta_vencimento_wdo] {e}")


def render_rolagem_wdo():
    """Rolagem de WDO por fundo: -Q no contrato atual e +Q no novo, na mesma data."""
    st.title("Rolagem de WDO")
    alerta_vencimento_wdo()

    df = load_posicoes_por_fundo()
    if df is None or df.empty:
        st.info("Sem posições cadastradas.")
        return
    df = df.copy()
    df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce").fillna(0)
    wdo = df[df["Ativo"].astype(str).str.upper().str.startswith("WDO")]
    pos = wdo.groupby(["Ativo", "Fundo"])["Quantidade"].sum()
    pos = pos[pos.abs() > 1e-9]
    tot = pos.groupby(level=0).sum()
    tot = tot[tot.abs() > 1e-9]
    if tot.empty:
        st.info("Nenhuma posição aberta em WDO.")
        return

    b3 = load_b3_prices()
    datas = [c for c in b3.columns if c != "Assets"]
    ult = pd.to_datetime(datas[-1]).date()
    px_tab = b3.set_index("Assets")

    c1, c2, c3 = st.columns(3)
    with c1:
        de = st.selectbox("Contrato atual (sai)", list(tot.index),
                          format_func=lambda a: f"{a}  ({tot[a]:+.0f})", key="rol_de")
    with c2:
        data_op = st.date_input("Data da rolagem", value=ult, max_value=ult, key="rol_data")
    d = str(data_op)
    if d not in px_tab.columns:
        st.error("Data sem pregão B3 (fim de semana/feriado) ou ainda sem preço. Rode o atualizar_precos.bat.")
        return
    candidatos = [c for c in wdo_contratos_disponiveis(d) if c != de]
    if not candidatos:
        st.error("Nenhum contrato de destino com ajuste B3 nessa data.")
        return
    with c3:
        para = st.selectbox("Contrato novo (entra)", candidatos, key="rol_para")

    def _pts(a):
        try:
            v = float(px_tab.at[a, d]) / 10.0
            return 0.0 if np.isnan(v) else v
        except Exception:
            return 0.0

    c4, c5, c6 = st.columns(3)
    with c4:
        p_de = st.number_input(f"Preço de saída {de} (pontos)", value=_pts(de), step=0.5,
                               format="%.2f", key="rol_pde")
    with c5:
        p_para = st.number_input(f"Preço de entrada {para} (pontos)", value=_pts(para), step=0.5,
                                 format="%.2f", key="rol_ppara")
    with c6:
        st.metric("Spread da rolagem (pontos)", f"{p_para - p_de:,.2f}")
    st.caption("Preços na cotação B3 (R$ por US$ 1.000). O sistema grava × 10 (R$ por contrato), "
               "mesmo padrão do ajuste usado no P&L. Default = ajuste B3 do dia.")

    tab = pos.loc[de].rename("Posição atual").to_frame()
    tab["Rolar"] = tab["Posição atual"].round().astype(int)
    st.write("Quantidade a rolar por fundo (edite se for rolagem parcial):")
    tab_ed = st.data_editor(
        tab, key="rol_editor",
        column_config={"Posição atual": st.column_config.NumberColumn(disabled=True, format="%.0f"),
                       "Rolar": st.column_config.NumberColumn(step=1, format="%d")})
    rolar = pd.to_numeric(tab_ed["Rolar"], errors="coerce").fillna(0).astype(int)
    rolar = rolar[rolar != 0]
    if rolar.empty:
        st.info("Nada a rolar.")
        return
    atual = tab_ed.loc[rolar.index, "Posição atual"]
    ruins = rolar[(rolar.abs() > atual.abs() + 1e-9) | (np.sign(rolar) != np.sign(atual))]
    if not ruins.empty:
        st.error("Quantidade a rolar maior que a posição ou com sinal trocado: " + ", ".join(map(str, ruins.index)))
        return

    Q = int(rolar.sum())
    p_ult_de = float(px_tab.at[de, datas[-1]]) if de in px_tab.index else np.nan
    p_ult_para = float(px_tab.at[para, datas[-1]]) if para in px_tab.index else np.nan
    df_info = pd.DataFrame([
        {"Ativo": de, "Quantidade": -Q, "Dia de Compra": d, "Preço de Compra": p_de * 10,
         "Preço de Ajuste Atual": p_ult_de, "Rendimento": -Q * (p_ult_de - p_de * 10)},
        {"Ativo": para, "Quantidade": Q, "Dia de Compra": d, "Preço de Compra": p_para * 10,
         "Preço de Ajuste Atual": p_ult_para, "Rendimento": Q * (p_ult_para - p_para * 10)},
    ])
    st.write("Operações que serão lançadas (consolidado):")
    st.table(df_info.set_index("Ativo")[["Quantidade", "Dia de Compra", "Preço de Compra"]])

    if not st.button(f"Confirmar rolagem {de} → {para} ({Q:+d} contratos)", type="primary", key="rol_ok"):
        return

    # 1) BaseFundos + posicoes_por_fundo (mesmo caminho do Adicionar Ativos)
    df_cur = pd.DataFrame({f"Contratos {de}": -rolar, f"Contratos {para}": rolar})
    df_cur.loc["Total"] = df_cur.sum()
    atualizar_parquet_fundos(df_cur, d, df_info, {de: -Q, para: Q})

    # 2) portfolio_posicoes (mesmo padrão do 'Salvar novo portfólio')
    _supa = load_data()
    base = pd.DataFrame(_supa) if _supa else pd.DataFrame(columns=list(df_info.columns) + ["Id"])
    if "Id" not in base.columns:
        base["Id"] = np.nan
    novo = pd.concat([base, df_info], ignore_index=True)
    max_id = pd.to_numeric(novo["Id"], errors="coerce").max()
    max_id = 0 if pd.isna(max_id) else int(max_id)
    falt = novo["Id"].isna()
    novo.loc[falt, "Id"] = list(range(max_id + 1, max_id + 1 + int(falt.sum())))
    novo["Id"] = novo["Id"].astype(int)
    novo["Dia de Compra"] = pd.to_datetime(novo["Dia de Compra"], errors="coerce").dt.strftime("%Y-%m-%d")
    res = add_data(novo.replace({np.nan: None}).to_dict(orient="records"))
    if res is None:
        st.error("Erro ao salvar portfolio_posicoes no Supabase (BaseFundos já foi atualizado).")
    else:
        st.success(f"Rolagem lançada: {de} {-Q:+d} @ {p_de:,.2f} | {para} {Q:+d} @ {p_para:,.2f} em {d}.")
    invalidar_caches_posicoes()
# [wdo-venc] fim rolagem ---------------------------------------------------


def main_page():'''

ADIC_OLD = '''            df_port, key, soma_pl_sem_pesos = checkar_portifolio(
                assets, quantidade_nomes, precos_user, data_compra, filtered_df, df_view)
'''
ADIC_NEW = '''            # [wdo-venc] WDO1 (risco) -> contrato exato para gravar a operacao e o P&L
            if "WDO1" in assets:
                _contratos_wdo = wdo_contratos_disponiveis(data_compra_todos)
                if not _contratos_wdo:
                    st.sidebar.error(f"Nenhum contrato WDO com ajuste B3 em {data_compra_todos}. "
                                     "Rode o atualizar_precos.bat.")
                    st.stop()
                _csv_wdo = st.session_state.get("wdo_contrato_csv")
                _idx_wdo = _contratos_wdo.index(_csv_wdo) if _csv_wdo in _contratos_wdo else 0
                st.sidebar.write("---")
                _contrato_wdo = st.sidebar.selectbox("Contrato WDO operado:", _contratos_wdo,
                                                     index=_idx_wdo, key="contrato_wdo")
                st.sidebar.caption("Preço do WDO em R$ por contrato (pontos × 10). "
                                   "O risco continua calculado pelo WDO1.")
                _ren = {"WDO1": _contrato_wdo}
                assets = [_ren.get(a, a) for a in assets]
                quantidade_nomes = {_ren.get(k, k): v for k, v in quantidade_nomes.items()}
                precos_user = {_ren.get(k, k): v for k, v in precos_user.items()}
                data_compra = {_ren.get(k, k): v for k, v in data_compra.items()}
                filtered_df = filtered_df.rename(columns={"Contratos WDO1": f"Contratos {_contrato_wdo}"})
                columns = [f"Contratos {_contrato_wdo}" if c == "Contratos WDO1" else c for c in columns]
            df_port, key, soma_pl_sem_pesos = checkar_portifolio(
                assets, quantidade_nomes, precos_user, data_compra, filtered_df, df_view)
'''

TROCAS = [
    # (nome, antigo, novo, n_esperado)
    ("helpers", "\ndef load_and_process_excel(", HELPERS, 1),
    ("rolagem", "\ndef main_page():", ROLAGEM, 1),
    ("processar_dados_port",
     "    df_assets = df_assets.groupby('Ativo').sum().reset_index()\n",
     "    df_assets['Ativo'] = df_assets['Ativo'].map(ativo_risco)  # [wdo-venc] risco agrega WDO_* em WDO1\n"
     "    df_assets = df_assets.groupby('Ativo').sum().reset_index()\n", 1),
    ("pivot supabase def",
     "def read_atual_contratos_supabase() -> pd.DataFrame:",
     "def read_atual_contratos_supabase(agregar_wdo: bool = True) -> pd.DataFrame:", 1),
    ("pivot fallback",
     "            return read_atual_contratos()\n",
     "            _p = read_atual_contratos()\n"
     "            return _agrupar_wdo_pivot(_p) if agregar_wdo else _p  # [wdo-venc]\n", 1),
    ("pivot agrega",
     '    agrupado = (df.groupby(["Fundo", "Ativo"], as_index=False)["Quantidade"].sum())\n',
     '    if agregar_wdo:  # [wdo-venc] risco: WDO_* somado em WDO1\n'
     '        df = df.copy()\n'
     '        df["Ativo"] = df["Ativo"].map(ativo_risco)\n'
     '    agrupado = (df.groupby(["Fundo", "Ativo"], as_index=False)["Quantidade"].sum())\n', 1),
    ("checkar atual",
     "        df_atual = df_portifolio_salvo.groupby('Ativo', as_index=False)[\n            'Quantidade'].sum()\n",
     "        df_atual = (df_portifolio_salvo.assign(Ativo=df_portifolio_salvo['Ativo'].map(ativo_risco))\n"
     "                    .groupby('Ativo', as_index=False)['Quantidade'].sum())  # [wdo-venc]\n", 1),
    ("checkar contratos",
     "    df_contratos = df_contratos.add(df_contratos_2, fill_value=0)\n",
     "    df_contratos = df_contratos.add(df_contratos_2, fill_value=0)\n"
     "    df_contratos = _agrupar_wdo_contratos(df_contratos)  # [wdo-venc]\n", 1),
    ("checkar novo",
     "            df_final_group = df_compilado.groupby('Ativo', as_index=False)[\n                'Quantidade'].sum()\n",
     "            df_final_group = (df_compilado.assign(Ativo=df_compilado['Ativo'].map(ativo_risco))\n"
     "                              .groupby('Ativo', as_index=False)['Quantidade'].sum())  # [wdo-venc]\n", 2),
    ("x10 processar_b3_portifolio",
     '    mask_wdo = df["Assets"] == "WDO1"\n',
     '    mask_wdo = df["Assets"].astype(str).str.startswith("WDO")  # [wdo-venc]\n', 1),
    ("x10 atualizar_parquet_fundos",
     "    df_fechamento_b3.loc[df_fechamento_b3['Assets'] == 'WDO1',\n                         df_fechamento_b3.columns != 'Assets'] *= 10\n",
     "    df_fechamento_b3.loc[df_fechamento_b3['Assets'].astype(str).str.startswith('WDO'),\n                         df_fechamento_b3.columns != 'Assets'] *= 10  # [wdo-venc]\n", 1),
    ("x10 load_b3_prices",
     "    df.loc[df['Assets'] == 'WDO1',     df.columns != 'Assets'] *= 10\n",
     "    df.loc[df['Assets'].astype(str).str.startswith('WDO'), df.columns != 'Assets'] *= 10  # [wdo-venc]\n", 1),
    ("custo WDO", 'elif raiz == "WDO1":', 'elif raiz in ("WDO1", "WDO"):', 1),
    ("is_fx", 'is_fx      = lambda a: "WDO1" in a', 'is_fx      = lambda a: str(a).startswith("WDO")', 1),
    ("estrategia dict", "'WDO1': 'Moedas'", "'WDO': 'Moedas'", 2),
    ("estrategia idx", "elif 'WDO1' in idx:", "elif 'WDO' in idx:", 2),
    ("csv WDO",
     '            if m: return f"DAP{m.group(1)}"\n',
     '            if m: return f"DAP{m.group(1)}"\n'
     '            m = re.match(r"^WDO([FGHJKMNQUVXZ]\\d{2})$", s)  # [wdo-venc]\n'
     '            if m:\n'
     '                st.session_state["wdo_contrato_csv"] = f"WDO_{m.group(1)}"\n'
     '                return "WDO1"\n', 1),
    ("radio", '["Ver Portfólio", "Adicionar Ativos", "Remover Ativos", "Simular Cota"], index=3',
     '["Ver Portfólio", "Adicionar Ativos", "Remover Ativos", "Simular Cota", "Rolar WDO"], index=3', 1),
    ("ver portfolio alerta", '    elif opti == "Ver Portfólio":\n',
     '    elif opti == "Ver Portfólio":\n        alerta_vencimento_wdo()  # [wdo-venc]\n', 1),
    ("branch rolar", '    elif opti == "Simular Cota":\n',
     '    elif opti == "Rolar WDO":  # [wdo-venc]\n        render_rolagem_wdo()\n\n    elif opti == "Simular Cota":\n', 1),
    ("adicionar contrato", ADIC_OLD, ADIC_NEW, 1),
]

erros = []
for nome, old, new, n in TROCAS:
    c = s.count(old)
    if c != n:
        erros.append(f"  - '{nome}': esperado {n}, encontrado {c}")
        continue
    s = s.replace(old, new)
if erros:
    print("[ERRO] app4.py NAO alterado. Ancoras com problema:\n" + "\n".join(erros)); sys.exit(1)

if sys.version_info >= (3, 12):
    compile(s, "app4.py", "exec")   # garante que nao quebrou a sintaxe
bak = f"app4.py.bak_wdo_{dt.datetime.now():%Y%m%d_%H%M%S}"
shutil.copy2(ARQ, bak)
with open(ARQ, "w", encoding="utf-8", newline="\r\n" if crlf else "\n") as f:
    f.write(s)
print(f"[ok] app4.py: {len(TROCAS)} alteracoes (backup {bak})")
