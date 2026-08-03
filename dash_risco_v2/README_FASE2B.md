# Fase 2b — Pipeline atualizado

Muda o produtor de `df_inicial.parquet` e `df_divone.parquet` de "BBG manual"
pra "Supabase dinâmico". **O `app4.py` não é tocado** — os arquivos que ele lê
continuam existindo com o mesmo formato; só o processo que os gera muda.

## O que muda no fluxo diário

**Antes:**
```
BBG - ECO DASH.xlsx  →  TransformarRetornosParquet.py  →  df_inicial + df_divone
AF_Trading.xlsm      →  app4.py (load_and_process_divone2)  →  df_divone atualizado
```

**Depois:**
```
B3 endpoint  →  ScrapB3_v2.py (patched)                                  →  parquets + precos_diarios (Supabase)
Supabase     →  taxas_dinamicas.gerar_df_inicial()                       →  df_inicial.parquet
Supabase     →  taxas_dinamicas.gerar_df_divone() + dv01_dinamico        →  df_divone.parquet
```

`AF_Trading.xlsm` e `BBG - ECO DASH.xlsx` **não são mais lidos** pela pipeline
diária (podem ficar arquivados como referência).

## Arquivos entregues

| Arquivo | Onde vai |
|---|---|
| `taxas_dinamicas.py` | `dash_risco_v2\taxas_dinamicas.py` (substitui o placeholder) |
| `07_patch_scrapb3.py` | `dash_risco_v2\07_patch_scrapb3.py` — script de patch |
| `TransformarRetornosParquet_v2.py` | Copia pra raiz do projeto como `TransformarRetornosParquet.py` (substitui o antigo) |

## Ordem de execução

### Passo 1 — Copia os 3 arquivos pra Z:

```powershell
cd "Z:\Asset Management\Equipe\Marcos\Risco\Projeto-Gestao-Risco"

# Cards clicáveis embaixo desta mensagem — salva:
# - taxas_dinamicas.py  →  dash_risco_v2\ (substitui)
# - 07_patch_scrapb3.py →  dash_risco_v2\
# - TransformarRetornosParquet_v2.py → raiz (SALVA COMO 'TransformarRetornosParquet.py' SUBSTITUINDO O ANTIGO)
```

**Antes de substituir o `TransformarRetornosParquet.py` antigo**, faça:
```powershell
Copy-Item TransformarRetornosParquet.py TransformarRetornosParquet.py.bak_pre_v2b
```

### Passo 2 — Patchear o ScrapB3_v2

```powershell
cd "Z:\Asset Management\Equipe\Marcos\Risco\Projeto-Gestao-Risco"
python dash_risco_v2\07_patch_scrapb3.py
```

O script:
- Cria `ScrapB3_v2.py.bak_pre_v2b` (backup)
- Aplica 3 patches:
  - `mapear_asset` → DAP_Q/K e DI_F/J/N/V
  - `selecionar_vertices` → aceita F/J/N/V do DI
  - Adiciona `_enviar_para_supabase` + chamada no fim do main
- É idempotente (rodar 2x = no-op)

Confirma com `git diff ScrapB3_v2.py` que só 3 blocos mudaram (mais a função nova no fim).

### Passo 3 — Testar geração de df_inicial e df_divone

```powershell
$env:SUPABASE_URL='https://clyyhlibvmnavdpfsbtr.supabase.co'
$env:SUPABASE_KEY='<sua service_role key>'
python dash_risco_v2\taxas_dinamicas.py --all
```

Deve imprimir:
```
[gerar_df_inicial] carregando precos_diarios...
[gerar_df_inicial] salvo: Dados/df_inicial.parquet — shape=(N linhas, M colunas)
[gerar_df_divone] data_ref = 2026-05-04 (ou última data disponível)
[gerar_df_divone] salvo: Dados/df_divone.parquet — X ativos, projeção=0.05%
```

### Passo 4 — Testar update_dash_risco.bat completo

Antes de rodar o .bat completo, testa cada etapa isoladamente:

```powershell
# 1. Rode o ScrapB3 patched (não precisa alterar o .bat ainda)
python ScrapB3_v2.py
# Verifica que os parquets locais foram atualizados COM O NOVO NAMING (DI_F26 etc)
# Verifica no Supabase: select "Fonte", count(*) from precos_diarios group by "Fonte";
# (a "b3" deve ter aumentado)

# 2. Rode o TransformarRetornosParquet (versão v2)
python TransformarRetornosParquet.py
# Verifica que Dados/df_inicial.parquet foi regerado
# Verifica que Dados/df_divone.parquet foi regerado

# 3. Rode o Streamlit local pra ver se não quebrou nada
streamlit run app4.py
```

Se tudo funcionar, agora pode rodar o `.bat` normalmente:

```powershell
.\update_dash_risco.bat
```

## Rollback

Se algo quebrar:
```powershell
Copy-Item ScrapB3_v2.py.bak_pre_v2b ScrapB3_v2.py -Force
Copy-Item TransformarRetornosParquet.py.bak_pre_v2b TransformarRetornosParquet.py -Force
```

E se os parquets `Dados/*.parquet` ficaram corruptos:
```powershell
Remove-Item Dados\df_inicial.parquet, Dados\df_divone.parquet
# Restaure do último backup (o Dados_backup_pre_naming que a Fase 1 criou é BEM antigo,
# mas serve como ponto de partida — depois rode o backfill de novo)
```

## O que essa fase NÃO faz

- Não gera snapshots diários (isso vem na Fase 2c)
- Não implementa stop-loss (Fase 2c)
- Não envia emails (Fase 2c)
- Não muda os gráficos históricos do dashboard (Fase 2c)

Essas coisas dependem do pipeline base estar sólido, que é o que essa fase entrega.

## Validação sugerida

Depois de rodar o `.bat` uma vez:

1. **Ver Portfólio**: quantidades e P&L devem estar corretos (como antes)
2. **Simular Cota**: métricas de risco (VaR, Vol, Sharpe) devem estar corretas
3. **Análise por Fundo**: quantidades por fundo devem estar corretas
4. **DV01**: entra em Adicionar Ativos e verifica se os DV01 mostrados agora estão coerentes com o cálculo dinâmico (é o momento de rodar em paralelo com o valor da sua planilha `di_curvab3vsimplicita.xlsx` pra confirmar consistência)

Se algum número parecer estranho, me manda print + qual data/ativo, que eu diagnóstico.

## Próxima fase (2c)

Vai vir:
- `gravar_snapshot_diario.py` — popula `snapshot_diario` toda rodada
- `aplicar_governance.py` — stop-loss vol-normalizado
- `enviar_email_diario.py` — Outlook COM
- `enviar_email_mensal.py`
- `atualiza_ni_ipca.py` — script mensal do IPEADATA
- Migração dos gráficos históricos do dashboard pra ler de `snapshot_diario`
- Novo `update_dash_risco.bat` com wire completo

Manda a saída dos 4 passos acima que eu ataco a 2c.
