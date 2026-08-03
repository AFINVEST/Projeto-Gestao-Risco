# Dash Risco v2 — Fase 1: Fundação

Este pacote entrega a **infraestrutura de persistência histórica** + **governance** +
**migração de naming** dos ativos DI/DAP. Não altera nenhuma linha do `app4.py`
ainda — isso vem na Fase 2. O objetivo é destravar tudo que depende de dados
persistidos e de um naming consistente.

## Conteúdo do pacote

| Arquivo | O que é |
|---|---|
| `01_schema_snapshot.sql` | Cria 5 tabelas novas no Supabase + seed de `config_risco` |
| `02_migrar_naming_supabase.sql` | Renomeia `DI_YY`→`DI_F<YY>`, `DAPYY`→`DAP_K/Q<YY>` no Supabase |
| `03_migrar_naming_parquets.py` | Renomeia o mesmo em todos os parquets locais + backup |
| `Dados/feriados_anbima.parquet` | 1263 feriados Anbima (2001–2099) |
| `Dados/ni_ipca.parquet` | 557 meses de NI IPCA (1979–2026) |
| `README.md` | Este arquivo |

## Ordem de aplicação

### Passo 1 — Rodar o SQL de tabelas no Supabase

Abra o SQL Editor do projeto Supabase e cole `01_schema_snapshot.sql`. Rode.

Deve aparecer `Success. No rows returned`. Se o Supabase avisar sobre RLS,
clique **Run with RLS** (o script já inclui as policies de leitura para bater
com o padrão da `portfolio_posicoes`).

Confira:
```sql
select count(*) from config_risco;           -- deve dar 11
select * from config_risco order by "parametro";
```

### Passo 2 — Migrar o naming no Supabase

No mesmo SQL Editor, cole `02_migrar_naming_supabase.sql`. Rode.

Ele:
1. Cria tabelas de backup (`portfolio_posicoes_backup_pre_naming` e
   `posicoes_por_fundo_backup_pre_naming`) com o estado atual
2. Renomeia `DI_29` → `DI_F29`, `DAP26` → `DAP_Q26`, `DAP27` → `DAP_K27`, etc.
3. É idempotente (rodar 2x não quebra nem duplica)

Verifique com a query no final do script:
```sql
select "Ativo", count(*) from portfolio_posicoes group by "Ativo" order by "Ativo";
```

Todos os DIs devem ter aparecido como `DI_F<YY>` e todos os DAPs como
`DAP_Q<YY>` (par) ou `DAP_K<YY>` (ímpar).

### Passo 3 — Migrar o naming nos parquets locais

Na sua máquina, dentro da pasta raiz do projeto (`Z:\...\Projeto-Gestao-Risco`),
rode:
```powershell
python 03_migrar_naming_parquets.py
```

O script:
1. Cria backups: `Dados_backup_pre_naming/` e `BaseFundos_backup_pre_naming/`
2. Renomeia colunas em `df_inicial.parquet` e `df_divone.parquet`
3. Renomeia linhas em `df_preco_de_ajuste_atual_completo.parquet`, `df_valor_ajuste_contrato.parquet`, etc
4. Renomeia coluna/index 'Ativo' em cada parquet de `BaseFundos/`

O output vai mostrar quantos ativos foram renomeados em cada arquivo.

### Passo 4 — Copiar os arquivos de referência

Copie os dois parquets deste pacote para o `Dados/` do seu projeto:
```powershell
Copy-Item feriados_anbima.parquet Dados\
Copy-Item ni_ipca.parquet Dados\
```

Esses arquivos vão ser usados pelo `dv01_dinamico.py` na Fase 2.

## Verificação de sanidade

Depois de rodar tudo:

**No Supabase:**
```sql
-- Nenhuma linha antiga deve sobrar
select "Ativo" from portfolio_posicoes where "Ativo" ~ '^(DI_[0-9]{2}|DAP[0-9]{2})$';
select "Ativo" from posicoes_por_fundo   where "Ativo" ~ '^(DI_[0-9]{2}|DAP[0-9]{2})$';
```
Ambas devem vir vazias.

**No dashboard:**
- Abra o app (modo anônimo pra evitar cache de sessão)
- Vai em **Ver Portfólio** — os ativos DI/DAP agora aparecem com o naming novo
- Pode ser que algum lugar do código ainda tenha string hardcoded do naming
  antigo — isso é esperado e vai ser tratado na Fase 2 (patches em app4.py)

**Se algo estiver estranho:**
- Rollback do Supabase está descrito no rodapé do `02_migrar_naming_supabase.sql`
- Rollback dos parquets: `Remove-Item -Recurse Dados; Rename-Item Dados_backup_pre_naming Dados` (mesmo padrão para BaseFundos)

## O que muda no app após a Fase 1

**Nada visível** — o app4.py ainda espera o naming antigo em várias funções
hardcoded, então provavelmente alguma tela vai quebrar até aplicarmos a Fase 2.

Especificamente, o `ScrapB3_v2.py` ainda vai continuar produzindo `DI_29` em vez
de `DI_F29`. Isso significa que **na próxima execução do scraping ele vai
sobrescrever os nomes migrados de volta pro formato antigo**. Portanto:

**IMPORTANTE**: **NÃO rode o `ScrapB3_v2.py` (nem o `update_dash_risco.bat`)
entre a Fase 1 e a Fase 2**. Se você rodar, terá que rodar o
`03_migrar_naming_parquets.py` de novo pra corrigir.

## Próximos passos (Fase 2)

Na próxima leva vem:
- `dv01_dinamico.py` — funções de DV01 usando `feriados_anbima.parquet` e `ni_ipca.parquet`
- `atualiza_projecao_anbima.py` — scraper da projeção IPCA mensal
- Patches em `app4.py` para consumir `read_atual_contratos_supabase` corrigido + naming novo
- Patches em `ScrapB3_v2.py` para produzir naming novo por padrão
- `gravar_snapshot_diario.py` — job que popula `snapshot_diario` a cada rodada do .bat
- `aplicar_governance.py` — job do stop-loss
- Wire dos jobs no `update_dash_risco.bat`

Rode a Fase 1, me avisa que tá tudo verde, e eu mando a Fase 2.
