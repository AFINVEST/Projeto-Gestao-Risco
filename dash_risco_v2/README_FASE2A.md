# Fase 2a — Fundação de Dados

Depois da Fase 1 (tabelas de snapshot + migração de naming), esta fase entrega:

- Tabela `precos_diarios` no Supabase (histórico persistente de PU/taxa)
- Módulo `dv01_dinamico.py` (cálculo de DV01 sem depender do AF_Trading.xlsm)
- Módulo `taxas_dinamicas.py` (reconstrói o antigo df_inicial a partir do Supabase)
- Script de backfill (popula precos_diarios com BBG histórico + B3 recente)
- Scraper ANBIMA (atualiza projeção IPCA mensal)
- `teste_dv01.py` que valida contra a planilha (**14/14 DIs e 10/10 DAPs com 0.00% de diferença**)

## Arquivos

| Arquivo | Onde vai | O que faz |
|---|---|---|
| `04_schema_precos_diarios.sql` | SQL Editor Supabase | Cria tabela precos_diarios + RLS |
| `dv01_dinamico.py` | `Z:\...\dash_risco_v2\` | Módulo core (75 vencimentos, DV01 DI/DAP) |
| `taxas_dinamicas.py` | `Z:\...\dash_risco_v2\` | Lê Supabase → devolve DataFrame wide de taxas |
| `05_backfill_precos_supabase.py` | Rodar 1x na Z:\ | Popula precos_diarios (BBG + B3) |
| `atualiza_projecao_anbima.py` | Rodar mensal (ou via .bat) | Scraper ANBIMA → config_risco |
| `teste_dv01.py` | Rodar pra validar | Compara com planilha em 29/07/26 |

## Ordem de execução

### Passo 1 — SQL no Supabase

Cola `04_schema_precos_diarios.sql` no SQL Editor, Run com RLS. Deve aparecer `Success. No rows returned`.

Confirma:
```sql
select count(*) from precos_diarios;   -- 0 nesse momento
```

### Passo 2 — Copia os módulos e valida DV01

Copia esses 4 arquivos pra `Z:\...\dash_risco_v2\`:
- `dv01_dinamico.py`
- `taxas_dinamicas.py`
- `teste_dv01.py`
- `05_backfill_precos_supabase.py`
- `atualiza_projecao_anbima.py`

**IMPORTANTE**: `dv01_dinamico.py` importa dos parquets em `Dados/` (raiz do projeto):
- `Dados/feriados_anbima.parquet`
- `Dados/ni_ipca.parquet`

(Esses dois já foram copiados na Fase 1)

Roda o teste:
```powershell
cd "Z:\Asset Management\Equipe\Marcos\Risco\Projeto-Gestao-Risco"
python dash_risco_v2\teste_dv01.py
```

Espera ver:
```
TESTE 1 — Calendário: 24 ok / 0 divergências
TESTE 2 — IPCA pro-rata: 5 ok / 0 divergências
TESTE 3 — DV01 DI: 14 ok / 0 divergências
TESTE 4 — DV01 DAP: 10 ok / 0 divergências
*** TODOS OS TESTES PASSARAM ***
```

Se algum falhar, me manda o output. Se passar, tá tudo certo pra seguir.

### Passo 3 — Backfill do Supabase

```powershell
$env:SUPABASE_URL='https://clyyhlibvmnavdpfsbtr.supabase.co'
$env:SUPABASE_KEY='<sua service_role key>'
python dash_risco_v2\05_backfill_precos_supabase.py
```

Vai popular a tabela `precos_diarios` com:
- Tudo que tem em `df_inicial.parquet` (BBG, história longa) → fonte='bbg'
- Tudo que tem em `df_preco_de_ajuste_atual_completo.parquet` (B3, mais recente) → fonte='b3'

Onde tiver conflito (mesma data+ativo), B3 sobrescreve BBG (mais confiável e mais recente).

Verifica:
```sql
select "Fonte", count(*) from precos_diarios group by "Fonte";
select "Ativo", count(*), min("Data"), max("Data")
from precos_diarios where "Ativo" like 'DI_%'
group by "Ativo" order by "Ativo";
```

Deve ver algo como `bbg: X mil linhas | b3: Y mil linhas`.

### Passo 4 — Scraper ANBIMA (opcional agora, obrigatório pra produção)

```powershell
# Primeiro testa em dry-run
python dash_risco_v2\atualiza_projecao_anbima.py --dry-run

# Se identificou a projeção correta, roda de verdade
python dash_risco_v2\atualiza_projecao_anbima.py
```

Se o site da ANBIMA mudar estrutura ou o scraping não funcionar, tem fallback manual:
```powershell
python dash_risco_v2\atualiza_projecao_anbima.py --manual 0.05
```

Verifica:
```sql
select * from config_risco where "parametro" like '%ipca%';
```

## O que essa fase ainda NÃO faz

- **Não altera `app4.py`** — o dashboard continua rodando exatamente como está, ainda dependendo do `df_inicial.parquet` local e do `df_divone.parquet`
- **Não altera `ScrapB3_v2.py`** — o scraping continua gerando parquets locais
- **Não altera `update_dash_risco.bat`** — pipeline diário continua o mesmo

Ou seja, o dashboard não vai "quebrar" nem "mudar" só pela Fase 2a. Ela cria infraestrutura pra Fase 2b consumir.

## O que vem na Fase 2b

- Patch em `app4.py` que troca `load_and_process_divone` (estático) por `dv01_dinamico` (dinâmico)
- Patch em `app4.py` pra ler taxas via `taxas_dinamicas.carregar_taxas_historicas` em vez de `df_inicial.parquet`
- Patch em `ScrapB3_v2.py` pra escrever em `precos_diarios` além do parquet
- Modificação do fluxo pra usar o naming novo (DI_F29 etc)
- Integração com a config `ipca_projecao_anbima_pct` para o DV01 do DAP

## Validação já realizada aqui na sandbox

Rodei `teste_dv01.py` contra os valores da sua planilha `di_curvab3vsimplicita.xlsx` para a data 29/07/2026:

- **24 vencimentos** — todos os DUs batendo (F27..F37, Q26..K45)
- **5 constantes IPCA** — batendo até a 6ª casa decimal
- **14 DVs** DIs — batendo até a 4ª casa decimal (diff 0.00%)
- **10 DVs** DAPs — batendo até a 4ª casa decimal (diff 0.00%)

Descobri e documentei uma **inconsistência interna da sua planilha**: o named range `rs_dap` (Base de dados J5) usa data ref 28/10/2025 mas a aba PnL usa 29/07/2026. Isso não afeta produção — nossa implementação usa um único `rs_dap` consistente com a data ref do dia.

Bônus: também tem os parâmetros `ni_ref_override` e `rs_dap_override` no `calcular_dv01` pra facilitar backtests históricos (você pode passar valores específicos pra recomputar DV01 em datas passadas com as constantes daquela data).
