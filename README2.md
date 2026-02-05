**Sistema de Risco, Portfólio e Stress**

Este documento responde às principais perguntas que um **analista de risco ou gestor** pode ter ao operar e manter este aplicativo, mesmo sem saber programar.

* * * * *

📑 Índice
=========

1.  Como o VaR é calculado?

2.  Como mudar o nível de confiança (95%, 99%)

3.  Como mudar a janela histórica (quantos dias entram no VaR)

4.  Como mudar a metodologia de VaR

5.  Como funciona o CVaR

6.  Onde o VaR vira dinheiro e bps

7.  Onde definir o limite de risco do fundo

8.  Como o risco é distribuído por ativo (CoVaR)

9.  Onde entra o Stress Test e DV01

10. Como mudar os ativos do portfólio

11. Como mudar as quantidades dos ativos

12. Como adicionar ou remover fundos

13. Como mudar os pesos dos fundos

14. Como mudar o que entra no cálculo de risco

* * * * *

1\. Como o VaR é calculado?
===========================

O sistema usa **VaR histórico não-paramétrico**:

`def var_not_parametric(data, alpha=0.05):
    return data.quantile(alpha)`

Ou seja:

> pega os piores 5% dos retornos e usa o percentil.
>
> app4

* * * * *

2\. Como mudar o nível de confiança?
====================================

Aqui:

`alpha=0.05`

| Valor | Confiança |
| --- | --- |
| 0.10 | 90% |
| 0.05 | 95% |
| 0.01 | 99% |

app4

* * * * *

3\. Como mudar a janela histórica?
==================================

Aqui:

`df_retorno = df_retorno.tail(1260)`

1260 ≈ 5 anos (252 dias por ano)

| Janela | Valor |
| --- | --- |
| 1 ano | 252 |
| 2 anos | 504 |
| 3 anos | 756 |
| 5 anos | 1260 |

app4

* * * * *

4\. Como mudar a metodologia de VaR?
====================================

Troque apenas esta função:

`def var_not_parametric(...)`

Por EWMA, Paramétrico Normal, Cornish-Fisher etc.\
Todas as métricas do sistema usam essa função.

* * * * *

5\. Como funciona o CVaR?
=========================

`cvar = abs(
    df_retorno[df_retorno['Portifolio'] < VaR]['Portifolio'].mean()
)`

Isso mede:

> A perda média quando o VaR é violado.
>
> app4

* * * * *

6\. Onde o VaR vira dinheiro e bps?
===================================

`var_port_dinheiro = vp_soma * var_port
pl_ref = soma_pl_sem_pesos * 0.01
VaR_bps = var_port_dinheiro / pl_ref * 10000`

O sistema mede risco como **bps sobre 1% do PL do fundo**.

app4

* * * * *

7\. Onde definir o limite de risco?
===================================

`var_bps = 1/10000
var_limite = 1.0`

Hoje o limite é **1 bp do PL**.\
Para 50 bps:

`var_limite = 50`

app4

* * * * *

8\. Como o risco é distribuído por ativo?
=========================================

Aqui:

`df_beta = cov_port / vol_port_retornos**2
df_mvar = df_beta * var_port
covar = df_mvar * pesos * vp_soma`

Isso gera:

-   Beta

-   Marginal VaR

-   CoVaR

-   % do risco total

    app4

* * * * *

9\. Onde entra Stress Test e DV01?
==================================

Os DV01 vêm de:

`df_divone = pd.read_parquet('Dados/df_divone.parquet')`

E o stress é calculado por:

`stress_test_juros_interno = df_divone * 100
stress_test_juros_real    = df_divone * 50`

E depois normalizado em bps do PL.

app4

* * * * *

10\. Como mudar os ativos do portfólio?
=======================================

O portfólio default vem de:

`Dados/portifolio_posições.parquet`

E é lido por:

`processar_dados_port()`

Você pode:

-   Adicionar linhas nesse parquet

-   Ou usar a tela do app (que grava nesse arquivo)

    app4

* * * * *

11\. Como mudar as quantidades dos ativos?
==========================================

Mesma fonte:

`Dados/portifolio_posições.parquet`

Coluna:

`Quantidade`

app4

* * * * *

12\. Como adicionar ou remover fundos?
======================================

Os fundos são os arquivos dentro da pasta:

`BaseFundos/`

Cada arquivo = 1 fundo.

Eles são lidos por:

`read_atual_contratos()`

Apagar um arquivo = remove um fundo.

app4

* * * * *

13\. Como mudar os pesos dos fundos?
====================================

Aqui:

`dict_pesos = {
    'GLOBAL BONDS': 4,
    'HORIZONTE': 1,
    'JERA2026': 1,
    ...
}`

Esses pesos multiplicam o PL no cálculo de risco.

app4

* * * * *

14\. Como mudar o que entra no risco?
=====================================

Os ativos considerados são:

`assets = assets_atual`

Que vem do portfólio salvo.

Se um ativo não estiver no parquet → ele **não entra no VaR, Stress, DV01 ou CoVaR**.

app4