# Edições pontuais em `app4.py`

Aplique na ordem. Os números de linha são aproximações baseadas no estado atual
do arquivo (12039 linhas). Use a string única antes/depois como âncora se a
linha tiver deslocado.

---

## 1) Colar bloco novo de helpers

Abra `03_app4_helpers.py`, copie todo o conteúdo entre os marcadores
`====== INÍCIO DO BLOCO PARA COLAR EM app4.py =====` e
`====== FIM DO BLOCO PARA COLAR EM app4.py ======` e cole no `app4.py`
**logo após** a função `load_data()` (~linha 3903).

> Observação: o bloco usa `supabase`, `st`, `pd`, `np`, `math`. Esses símbolos já
> estão importados no topo do `app4.py`. Os `import` redundantes do bloco são
> inofensivos.

---

## 2) Substituir `read_atual_contratos_cached()` (~linha 5152)

**Antes:**
```python
@st.cache_data(show_spinner=False)
def read_atual_contratos_cached():
    return read_atual_contratos()
```

**Depois:**
```python
@st.cache_data(show_spinner=False, ttl=120)
def read_atual_contratos_cached():
    # Fonte primária: tabela posicoes_por_fundo no Supabase
    # (read_atual_contratos antigo continua sendo fallback dentro da função)
    return read_atual_contratos_supabase()
```

---

## 3) Substituir `load_basefundos()` (~linha 5186)

**Antes:**
```python
@st.cache_data(show_spinner=False)
def load_basefundos() -> dict[str, pd.DataFrame]:
    """Lê todos os arquivos de BaseFundos apenas 1x por sessão."""
    out = {}
    for f in os.listdir("BaseFundos"):
        nome = f.rsplit(".", 1)[0]
        df   = pd.read_parquet(f"BaseFundos/{f}").set_index("Ativo")
        out[nome] = df
    return out
```

**Depois:**
```python
def _load_basefundos_local() -> dict[str, pd.DataFrame]:
    """Versão antiga, usada apenas como fallback se o Supabase
    estiver indisponível ou ainda não tiver sido populado."""
    out = {}
    for f in os.listdir("BaseFundos"):
        nome = f.rsplit(".", 1)[0]
        df   = pd.read_parquet(f"BaseFundos/{f}").set_index("Ativo")
        out[nome] = df
    return out


@st.cache_data(show_spinner=False, ttl=120)
def load_basefundos() -> dict[str, pd.DataFrame]:
    """Lê o estado dos fundos a partir do Supabase (com fallback local)."""
    out = load_basefundos_supabase()
    if not out:
        out = _load_basefundos_local()
    # Compatibilidade: o resto do código espera df.set_index('Ativo')
    fixed = {}
    for nome, df in out.items():
        if "Ativo" in df.columns:
            fixed[nome] = df.set_index("Ativo")
        else:
            fixed[nome] = df
    return fixed
```

> Importante: a chamada a `load_basefundos_supabase()` requer o bloco do passo 1
> já colado.

---

## 4) Cache invalidation em `add_data()`, `delete_data()`, `update_data()`

### 4.1) `add_data` (~linha 3908)

**Antes:**
```python
def add_data(data):
    try:
        supabase.table('portfolio_posicoes').delete().neq('Ativo', 0).execute()
        response = supabase.table('portfolio_posicoes').insert(data).execute()
        print("Resposta do Supabase:", response)
        return response.data
    except Exception as e:
        print(f"Erro detalhado add_data: {e}")
        st.error(f"Erro ao salvar no Supabase: {e}")
        return None
```

**Depois:**
```python
def add_data(data):
    try:
        supabase.table('portfolio_posicoes').delete().neq('Ativo', 0).execute()
        response = supabase.table('portfolio_posicoes').insert(data).execute()
        print("Resposta do Supabase:", response)
        invalidar_caches_posicoes()
        return response.data
    except Exception as e:
        print(f"Erro detalhado add_data: {e}")
        st.error(f"Erro ao salvar no Supabase: {e}")
        return None
```

### 4.2) `delete_data()` e `update_data()`
Procure as definições no arquivo e adicione `invalidar_caches_posicoes()` logo
antes do `return` em ambos. Mesmo padrão do passo 4.1.

---

## 5) Espelhar `update_base_fundos()` no Supabase (~linha 3780)

A função `update_base_fundos(ativo_escolhido, dia_compra_escolhido)` apaga
colunas de uma operação nos parquets de `BaseFundos`. Adicione, depois do bloco
que mexe nos parquets e antes do `return`:

```python
    # Espelha o delete na tabela do Supabase
    delete_posicao_por_fundo(ativo_escolhido, dia_compra_escolhido)
    invalidar_caches_posicoes()
```

---

## 6) Espelhar `atualizar_parquet_fundos()` no Supabase (~linha 3979)

Esta é a função maior. A ideia é coletar a foto que está sendo gravada no
parquet e enviar **a mesma foto** para o Supabase. No fim de cada iteração do
loop interno (depois de gravar `df_fundo.loc[asset, ...]`), acumule um dict.
No fim da função (depois do último `df_fundo.to_parquet(...)`), faça um único
upsert.

Localize o final da função (depois do for principal sobre `df_current`) e
adicione, antes do `return`:

```python
    # ---------- Espelho no Supabase (posicoes_por_fundo) ----------
    registros_ppf: list[dict] = []
    for arquivo in os.listdir("BaseFundos"):
        if not arquivo.endswith(".parquet"):
            continue
        nome_fundo = arquivo.rsplit(".", 1)[0]
        if nome_fundo.upper() == "TOTAL":
            continue
        df_f = pd.read_parquet(os.path.join("BaseFundos", arquivo))
        if "Ativo" not in df_f.columns:
            if df_f.index.name == "Ativo":
                df_f = df_f.reset_index()
            else:
                continue
        # Filtra colunas do dia
        col_PL    = f"{dia_operacao} - PL"
        col_Pf    = f"{dia_operacao} - Preco_Fechamento"
        col_Pc    = f"{dia_operacao} - Preco_Compra"
        col_Q     = f"{dia_operacao} - Quantidade"
        col_R     = f"{dia_operacao} - Rendimento"
        for _, row in df_f.iterrows():
            ativo = row.get("Ativo")
            if pd.isna(ativo):
                continue
            registros_ppf.append({
                "Fundo": nome_fundo,
                "Ativo": str(ativo),
                "Data":  dia_operacao,
                "Quantidade":       row.get(col_Q),
                "Preco_Compra":     row.get(col_Pc),
                "Preco_Fechamento": row.get(col_Pf),
                "PL":               row.get(col_PL),
                "Rendimento":       row.get(col_R),
            })
    if registros_ppf:
        upsert_posicoes_por_fundo(registros_ppf)
```

> Por que reler todos os parquets em vez de coletar dentro do loop existente?
> A função original tem várias ramificações (asset novo vs. existente; fundo
> "Total" tratado diferente). Reler garante que o Supabase fica idêntico ao
> que acabou de ir para o disco, sem se preocupar com regalias internas.

---

## 7) Remover atalho problemático de `get_risk_static_bundle()` (~linha 6545)

**Antes:**
```python
@st.cache_resource(show_spinner=False)
def get_risk_static_bundle(pl_signature: tuple, interpret_quantities: str = "delta"):
    """
    Atualiza a função para garantir o uso correto dos dados de fechamento e correção das posições.
    """
    # Verificar se o bundle já está no session_state
    if '_risk_bundle' in st.session_state:
        return st.session_state['_risk_bundle']  # Se já estiver no session_state, apenas retorna

    # Caso não esteja no session_state, cria o bundle
    b = {}
    ...
```

**Depois:**
```python
def _ppf_signature() -> tuple:
    """Assinatura leve da carteira para invalidar o cache do bundle."""
    try:
        # Faz um SELECT simples só para hashear o estado mais recente.
        resp = (supabase.table('portfolio_posicoes')
                        .select('"Id","Quantidade"')
                        .order('"Id"', desc=True)
                        .limit(1)
                        .execute())
        max_id = resp.data[0]["Id"] if resp.data else 0
        # Conta total como segundo eixo da assinatura
        cnt = (supabase.table('portfolio_posicoes')
                       .select("Id", count="exact")
                       .limit(1)
                       .execute())
        total = cnt.count or 0
        return (int(max_id), int(total))
    except Exception:
        # se falhar, devolve algo dependente do tempo p/ derrubar o cache em ~minutos
        return (int(pd.Timestamp.utcnow().value // 60_000_000_000), 0)


@st.cache_resource(show_spinner=False)
def get_risk_static_bundle(pl_signature: tuple,
                           interpret_quantities: str = "delta",
                           positions_signature: tuple = ()):
    """
    Bundle de risco. A chave de cache passa a depender também das posições.
    """
    b = {}
    ...
```

E **no caller** (`simulate_nav_cota`, ~linha 6843), troque:

```python
bundle = get_risk_static_bundle(pl_sig, interpret_quantities="delta")
```

por:

```python
bundle = get_risk_static_bundle(
    pl_sig,
    interpret_quantities="delta",
    positions_signature=_ppf_signature(),
)
```

---

## 8) (Opcional, mas recomendado) Avisar quando o cache foi invalidado

Logo após `add_data(...)` na aba "Adicionar Ativos" (~linha 940), adicione:

```python
st.toast("Posições atualizadas. Recarregue 'Ver Portfólio' para ver os novos números.", icon="✅")
```

---

## 9) Limpeza dos `SyntaxWarning`

Se quiser de quebra silenciar os warnings que aparecem no log, troque os escapes
literais por raw strings:

| Linha aprox. | De                                         | Para                                       |
|--------------|--------------------------------------------|--------------------------------------------|
| 452          | `'\.'`                                     | `r'\.'`                                    |
| 504          | `'\.'`                                     | `r'\.'`                                    |
| 1040         | `'\.'`                                     | `r'\.'`                                    |
| 4011         | `'\.'`                                     | `r'\.'`                                    |
| 4034         | `'\.'`                                     | `r'\.'`                                    |
| 4887         | `"\$"`                                     | `r"\$"`                                    |
| 7664         | `"\d"`                                     | `r"\d"`                                    |
| 8280         | `"\d"`                                     | `r"\d"`                                    |
