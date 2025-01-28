import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import os


def atualizar_csv_fundos(
    df_current,         # DataFrame do dia atual (1 linha por Fundo)
    dia_operacao,       # Exemplo: "2025-01-20"
    # DF de transações: [Ativo, Quantidade, Dia de Compra, Preço de Compra, ...]
    df_info,
    # DF de preços de fechamento: colunas ["Assets", <data1>, <data2>, ...]
):
    """
    - O CSV final de cada fundo terá linhas = cada ativo (ou "PL") e
      colunas = ["Ativo","Preco_Compra","Preco_Fechamento_Atual", <data1>, <data2>, ...],
      onde cada dataX é a posição final do ativo naquele dia.
    - Sempre que chamamos a função (inclusive mais de uma vez no mesmo dia),
      se houver aumento de posição (quantidade final maior que a anterior naquele dia),
      recalculamos o preço médio.
    - A coluna "Preco_Fechamento_Atual" vem do df_fechamento_b3 para a
      última data disponível (ex: "2025-01-20").
    """
    df_b3_fechamento = processar_b3_portifolio()
    # ------------------------------------------------------------------
    # 1) Identificar qual é a última data disponível em df_fechamento_b3
    # ------------------------------------------------------------------
    # Supondo que a 1ª coluna é "Assets" e as demais são datas
    colunas_b3 = list(df_b3_fechamento.columns)
    colunas_b3.remove("Assets")  # agora só temos as datas
    colunas_b3_ordenadas = sorted(colunas_b3)
    ultima_data_fechamento = colunas_b3_ordenadas[-1]  # ex: "2025-01-20"

    # ------------------------------------------------------------------
    # Itera cada FONDO (linha) em df_current
    # ------------------------------------------------------------------
    for fundo, row_fundo in df_current.iterrows():
        # Caminho do CSV do Fundo
        nome_arquivo_csv = os.path.join("BaseFundos", f"{fundo}.csv")

        # ----------------------------------------------------------------
        # 2) Carregar (ou criar) o DataFrame histórico do Fundo (df_fundo)
        # ----------------------------------------------------------------
        if os.path.exists(nome_arquivo_csv):
            df_fundo = pd.read_csv(nome_arquivo_csv, index_col=None)
        else:
            df_fundo = pd.DataFrame(
                columns=["Ativo", "Preco_Compra", "Preco_Fechamento_Atual"])

        # Garante que "Ativo" seja índice
        if "Ativo" in df_fundo.columns:
            df_fundo.set_index("Ativo", inplace=True, drop=False)

        # ----------------------------------------------------------------
        # 3) Verifica PL (se existir)
        # ----------------------------------------------------------------
        valor_pl = None
        if "PL" in df_current.columns:
            valor_pl = row_fundo["PL"]

        # ----------------------------------------------------------------
        # 4) Identificar colunas de CONTRATOS no df_current
        # ----------------------------------------------------------------
        colunas_contratos = [
            c for c in df_current.columns
            if c.startswith("Contratos ")
        ]
        # Exemplo: {"Contratos WDO1": 10, "Contratos DI_33": -25, ...}
        serie_contratos = row_fundo[colunas_contratos]

        # ----------------------------------------------------------------
        # 5) Garante a coluna do dia_operacao em df_fundo
        # ----------------------------------------------------------------
        if dia_operacao not in df_fundo.columns:
            df_fundo[dia_operacao] = 0.0

        # Se existir "PL", cria/atualiza linha
        if valor_pl is not None:
            if "PL" not in df_fundo.index:
                df_fundo.loc["PL", "Ativo"] = "PL"
                df_fundo.loc["PL", "Preco_Compra"] = np.nan
                df_fundo.loc["PL", "Preco_Fechamento_Atual"] = np.nan
            df_fundo.loc["PL", dia_operacao] = valor_pl

        # ----------------------------------------------------------------
        # 6) Para cada contrato (ativo), atualizar posição e preço médio
        # ----------------------------------------------------------------
        for col_contrato, qtd_dia_raw in serie_contratos.items():
            ativo = col_contrato.replace("Contratos ", "").strip()

            # Converte a quantidade para float
            try:
                qtd_dia_nova = float(qtd_dia_raw)
            except:
                qtd_dia_nova = 0.0

            # Se o ativo ainda não existe no df_fundo, cria a linha
            if ativo not in df_fundo.index:
                df_fundo.loc[ativo, "Ativo"] = ativo
                df_fundo.loc[ativo, "Preco_Compra"] = 0.0
                df_fundo.loc[ativo, "Preco_Fechamento_Atual"] = 0.0
                df_fundo.loc[ativo, dia_operacao] = 0.0

            # Carrega o preço médio atual (se for NaN, substitui por 0.0)
            preco_medio_atual = df_fundo.loc[ativo, "Preco_Compra"]
            if pd.isna(preco_medio_atual):
                preco_medio_atual = 0.0

            # ----------------------------------------------------------------
            # 6.1) Descobrir a quantidade que já estava nesse MESMO dia
            #      (caso tenha sido atualizada em chamadas anteriores hoje)
            # ----------------------------------------------------------------
            qtd_dia_anterior = 0.0
            try:
                qtd_dia_anterior = float(df_fundo.loc[ativo, dia_operacao])
            except:
                # Se não existia nada, fica 0
                pass

            # ----------------------------------------------------------------
            # 6.2) Calcula a diferença no MESMO dia
            #      Se > 0 => houve compra adicional;
            #      se < 0 => houve venda adicional.
            # ----------------------------------------------------------------
            diff = qtd_dia_nova - qtd_dia_anterior

            if diff > 0:
                # Houve compra adicional no MESMO dia_operacao
                # Precisamos achar no df_info as linhas do DIA + ATIVO
                subset = df_info[
                    (df_info["Ativo"] == ativo) &
                    (df_info["Dia de Compra"] == dia_operacao)
                ]
                if len(subset) == 0:
                    # Se não encontrou nada em df_info, mantemos o preço
                    preco_compra_dia = preco_medio_atual
                else:
                    # Soma ponderada das quantidades positivas
                    qtd_total_dia = 0.0
                    valor_total_dia = 0.0
                    for i, r in subset.iterrows():
                        q_linha = float(str(r["Quantidade"]).replace(",", "."))
                        p_linha = float(
                            str(r["Preço de Compra"]).replace(",", "."))
                        if q_linha > 0:
                            qtd_total_dia += q_linha
                            valor_total_dia += (q_linha * p_linha)

                    if qtd_total_dia == 0:
                        preco_compra_dia = preco_medio_atual
                    else:
                        preco_compra_dia = valor_total_dia / qtd_total_dia

                # Novo preço médio = ponderação da posição anterior do DIA + compra nova
                # Posição anterior do dia (qtd_dia_anterior) * preco_medio_atual
                # + diff * preco_compra_dia
                # -----------------------------------------
                # Nova posição final = qtd_dia_nova
                # (que é qtd_dia_anterior + diff)
                # -----------------------------------------
                qtd_comprada = diff
                qtd_final = qtd_dia_nova  # que é qtd_dia_anterior + diff

                novo_preco_medio = (
                    (qtd_dia_anterior * preco_medio_atual) +
                    (qtd_comprada * preco_compra_dia)
                ) / qtd_final

                df_fundo.loc[ativo, "Preco_Compra"] = novo_preco_medio

            elif diff < 0:
                # Houve venda no mesmo dia.
                # Nesse caso, normalmente não mexemos no preço médio se for apenas redução parcial.
                # Mas se a posição final ficar 0 ou negativa, podemos ter regras específicas.

                # Exemplo de lógica:

                qtd_final = qtd_dia_nova  # lembrando que qtd_dia_nova = qtd_dia_anterior + diff

                if qtd_final == 0:
                    #
                    # ZEROU posição -> podemos "resetar" o preço médio
                    #
                    # ou np.nan, se preferir
                    df_fundo.loc[ativo, "Preco_Compra"] = 0.0
                    # Obs: Aqui não apuramos P&L. Se quiser apurar ganho/perda realizado,
                    # você precisaria saber o preço de venda (vem do df_info, por ex.)
                    # e comparar com o preço médio antigo.
                    pass

                elif qtd_final > 0:
                    #
                    # Vendeu parcialmente, mas não zerou. Mantemos o mesmo preço médio.
                    #
                    # Se desejar, poderia apurar P&L parcial aqui também.
                    pass

                else:
                    #
                    # Se chegou aqui, significa que a posição final ficou NEGATIVA.
                    # Exemplo: passou de +10 para -5.
                    #
                    # 1) Se quiser, pode dividir a lógica em:
                    #    - "Zerar" a parte que estava positiva (apurar P&L)
                    #    - "Abrir" a nova posição short com preço médio próprio.
                    #
                    # 2) A maneira mais simples (exemplo) é:
                    #    - Sempre que virar posição (de + p/ -), redefinimos o preço médio
                    #      com base no valor de 'venda' da quantidade que 'excedeu' a posição antiga.
                    #
                    #    - Caso já estivéssemos short e só aumentou o short, poderíamos
                    #      recalcular via média ponderada também. (depende do seu controle)
                    #
                    # Para ilustrar, vamos ver se a posição anterior era >= 0 e agora é < 0.
                    # Precisamos identificar quantas unidades "excederam":
                    # quantas unidades ficaram short
                    qtd_excedida = abs(qtd_final)

                    # Buscamos o preço de venda no df_info (quantidade negativa e data = dia_operacao)
                    subset_venda = df_info[
                        (df_info["Ativo"] == ativo) &
                        (df_info["Dia de Compra"] == dia_operacao) &
                        (df_info["Quantidade"] < 0)
                    ]
                    # Caso tenha várias linhas de venda, você pode somar/ponderar.
                    # Neste exemplo, assumimos que existe apenas uma linha ou pegamos a média:
                    if len(subset_venda) > 0:
                        # Exemplo simples: pegamos a 1ª linha
                        p_venda = float(
                            str(subset_venda.iloc[0]["Preço de Compra"]).replace(
                                ",", ".")
                        )
                    else:
                        # Se não encontrou info, use o preço médio anterior
                        p_venda = preco_medio_atual

                    # (Opcional) "resetar" o preço médio
                    #           definindo como o preço da 'venda' para a posição short
                    df_fundo.loc[ativo, "Preco_Compra"] = p_venda

                    # Em muitos sistemas, preço médio de posição short é controlado de forma diferente,
                    # pois, na verdade, esse "preço médio" significaria quanto se espera recomprar no futuro.
                    # Vai depender de como você quer registrar.
                    pass

                # Por fim, independemente do caso, atualizamos a coluna do dia com a nova quantidade final:
                df_fundo.loc[ativo, dia_operacao] = qtd_final

            # Por fim, atualizamos a coluna do dia com a nova quantidade final
            df_fundo.loc[ativo, dia_operacao] = qtd_dia_nova

            # ----------------------------------------------------------------
            # 6.3) Atualizar "Preço de Fechamento Atual" com base em df_fechamento_b3
            # ----------------------------------------------------------------
            # Localiza o ativo em df_fechamento_b3["Assets"] == ativo
            row_fech = df_b3_fechamento[df_b3_fechamento["Assets"] == ativo]
            if not row_fech.empty:
                valor_fechamento_str = row_fech[ultima_data_fechamento].values[0]
                # Converter de string (com vírgula) para float:
                # Exemplo: "6.081,5680" => "6081.5680"
                valor_fechamento_str = valor_fechamento_str.replace(
                    ".", "").replace(",", ".")
                try:
                    valor_fechamento_float = float(valor_fechamento_str)
                except:
                    valor_fechamento_float = 0.0
                df_fundo.loc[ativo,
                             "Preco_Fechamento_Atual"] = valor_fechamento_float
            else:
                df_fundo.loc[ativo, "Preco_Fechamento_Atual"] = np.nan

        # ----------------------------------------------------------------
        # 7) Salvar o CSV do fundo
        # ----------------------------------------------------------------
        df_fundo.reset_index(drop=True, inplace=True)
        df_fundo.to_csv(nome_arquivo_csv, index=False, encoding="utf-8")
        print(f"[{fundo}] -> CSV atualizado: {nome_arquivo_csv}")


# segundo


def processar_b3_portifolio():
    """
    Exemplo fictício: Retorna um df de fechamento com colunas:
    ["Assets", "2025-01-19", "2025-01-20", ...]
    Você deve adaptar para seu caso real.
    """
    data = {
        "Assets": ["WDO1", "DI_33", "PETR4", "VALE3"],
        "2025-01-19": ["5.000,00", "1.000,00", "27,00", "70,00"],
        "2025-01-20": ["5.100,00", "1.050,00", "28,00", "72,00"],
    }
    df = pd.DataFrame(data)
    return df


def atualizar_csv_fundos(
    df_current,         # DataFrame do dia atual (1 linha por Fundo)
    dia_operacao,       # Exemplo: "2025-01-20"
    # DF de transações: [Ativo, Quantidade, Dia de Compra, Preço de Compra, ...]
    df_info,
):
    """
    - O CSV final de cada fundo terá linhas = cada ativo (ou "PL") e
      colunas = ["Ativo","Preco_Compra","Preco_Fechamento_Atual",
                 "PnL_Realizado", "PnL_Atual", <datas...>],
      onde cada dataX é a posição final do ativo naquele dia (quantidade).
    - Apuramos o PnL realizado nas operações de venda (encerrando ou reduzindo long)
      ou de compra (encerrando ou reduzindo short).
    - Calculamos PnL_Atual (não realizado) = (Preco_Fechamento_Atual - Preco_Compra)*Quantidade
      (ajustando sinal se for short).
    """

    df_b3_fechamento = processar_b3_portifolio()

    # ------------------------------------------------------------------
    # 1) Identificar a última data disponível em df_b3_fechamento
    # ------------------------------------------------------------------
    colunas_b3 = list(df_b3_fechamento.columns)
    colunas_b3.remove("Assets")  # agora só temos as datas
    colunas_b3_ordenadas = sorted(colunas_b3)
    ultima_data_fechamento = colunas_b3_ordenadas[-1]  # ex: "2025-01-20"

    # ------------------------------------------------------------------
    # Itera cada FUNDO (linha) em df_current
    # ------------------------------------------------------------------
    for fundo, row_fundo in df_current.iterrows():
        # Caminho do CSV do Fundo
        nome_arquivo_csv = os.path.join("BaseFundos", f"{fundo}.csv")

        # 2) Carregar (ou criar) o DataFrame histórico do Fundo (df_fundo)
        if os.path.exists(nome_arquivo_csv):
            df_fundo = pd.read_csv(nome_arquivo_csv, index_col=None)
        else:
            df_fundo = pd.DataFrame(
                columns=[
                    "Ativo",
                    "Preco_Compra",
                    "Preco_Fechamento_Atual",
                    "PnL_Realizado",   # P&L realizado acumulado
                    "PnL_Atual"        # P&L não realizado (marcado a mercado)
                ]
            )

        # Garante que "Ativo" seja índice
        if "Ativo" in df_fundo.columns:
            df_fundo.set_index("Ativo", inplace=True, drop=False)

        # Se colunas de PnL não existirem, cria
        if "PnL_Realizado" not in df_fundo.columns:
            df_fundo["PnL_Realizado"] = 0.0
        if "PnL_Atual" not in df_fundo.columns:
            df_fundo["PnL_Atual"] = 0.0

        # ----------------------------------------------------------------
        # 3) Verifica PL do Fundo (se existir) e colunas Contratos
        # ----------------------------------------------------------------
        valor_pl = None
        if "PL" in df_current.columns:
            valor_pl = row_fundo["PL"]

        colunas_contratos = [
            c for c in df_current.columns if c.startswith("Contratos ")]
        serie_contratos = row_fundo[colunas_contratos]

        # ----------------------------------------------------------------
        # 4) Garante a coluna do dia_operacao em df_fundo (p/ posição)
        # ----------------------------------------------------------------
        if dia_operacao not in df_fundo.columns:
            df_fundo[dia_operacao] = 0.0

        # Se existir "PL", cria/atualiza linha
        if valor_pl is not None:
            if "PL" not in df_fundo.index:
                df_fundo.loc["PL", "Ativo"] = "PL"
                df_fundo.loc["PL", "Preco_Compra"] = np.nan
                df_fundo.loc["PL", "Preco_Fechamento_Atual"] = np.nan
                df_fundo.loc["PL", "PnL_Realizado"] = 0.0
                df_fundo.loc["PL", "PnL_Atual"] = 0.0
            df_fundo.loc["PL", dia_operacao] = valor_pl

        # ----------------------------------------------------------------
        # 5) Para cada contrato (ativo), atualizar posição e preço médio
        # ----------------------------------------------------------------
        for col_contrato, qtd_dia_raw in serie_contratos.items():
            ativo = col_contrato.replace("Contratos ", "").strip()

            # Converte a quantidade para float
            try:
                qtd_dia_nova = float(qtd_dia_raw)
            except:
                qtd_dia_nova = 0.0

            # Se o ativo ainda não existe no df_fundo, cria a linha
            if ativo not in df_fundo.index:
                df_fundo.loc[ativo, "Ativo"] = ativo
                df_fundo.loc[ativo, "Preco_Compra"] = 0.0
                df_fundo.loc[ativo, "Preco_Fechamento_Atual"] = 0.0
                df_fundo.loc[ativo, "PnL_Realizado"] = 0.0
                df_fundo.loc[ativo, "PnL_Atual"] = 0.0
                df_fundo.loc[ativo, dia_operacao] = 0.0

            # Carrega variáveis
            preco_medio_atual = df_fundo.loc[ativo, "Preco_Compra"]
            if pd.isna(preco_medio_atual):
                preco_medio_atual = 0.0

            pnl_realizado = df_fundo.loc[ativo, "PnL_Realizado"]  # acumulado
            # (PnL_Atual será recalculado ao final, não precisamos carregar)

            qtd_dia_anterior = float(df_fundo.loc[ativo, dia_operacao]) if not pd.isna(
                df_fundo.loc[ativo, dia_operacao]) else 0.0

            # Diferença no mesmo dia
            diff = qtd_dia_nova - qtd_dia_anterior

            # ------------------------------------------------------------
            # 5.1) Se houve compra (diff > 0)
            # ------------------------------------------------------------
            if diff > 0:
                # Buscar preço de compra no df_info
                subset_compra = df_info[
                    (df_info["Ativo"] == ativo) &
                    (df_info["Dia de Compra"] == dia_operacao) &
                    (df_info["Quantidade"] > 0)
                ]
                if len(subset_compra) > 0:
                    p_compra = float(
                        str(subset_compra.iloc[0]["Preço de Compra"]).replace(",", "."))
                else:
                    p_compra = preco_medio_atual

                # Se a posição anterior era negativa, parte ou toda essa compra
                # reduz o short => apuramos PnL realizado na parte encerrada
                if qtd_dia_anterior < 0:
                    qtd_cobrindo_short = min(abs(qtd_dia_anterior), diff)
                    # PnL short = (preco_medio_short - p_compra) * quantidade_coberta
                    realized_pnl = (preco_medio_atual -
                                    p_compra) * qtd_cobrindo_short
                    pnl_realizado += realized_pnl

                    # Se "virou" para positiva, parte do diff abre nova posição long
                    qtd_excesso = diff - qtd_cobrindo_short
                    if (qtd_dia_anterior + diff) > 0:
                        # Ex.: -10 + 15 = +5 -> abrimos 5 de posição long
                        if qtd_excesso > 0:
                            # Novo preço médio para o bloco que excedeu (poderia ponderar)
                            preco_medio_atual = p_compra
                        else:
                            # Não abriu long, só reduziu short (ou zerou).
                            if (qtd_dia_anterior + diff) == 0:
                                preco_medio_atual = 0.0
                    else:
                        # Ainda fica short, mas menos negativa (ex.: -10 + 5 = -5)
                        # Em geral, não mexemos no preco médio do short ou poderíamos recalcular.
                        pass
                else:
                    # Posição anterior >= 0 => compra normal => recalcula preço médio
                    qtd_comprada = diff
                    qtd_final = qtd_dia_anterior + diff
                    novo_preco = (
                        (qtd_dia_anterior * preco_medio_atual) +
                        (qtd_comprada * p_compra)
                    ) / qtd_final
                    preco_medio_atual = novo_preco

            # ------------------------------------------------------------
            # 5.2) Se houve venda (diff < 0)
            # ------------------------------------------------------------
            elif diff < 0:
                subset_venda = df_info[
                    (df_info["Ativo"] == ativo) &
                    (df_info["Dia de Compra"] == dia_operacao) &
                    (df_info["Quantidade"] < 0)
                ]
                if len(subset_venda) > 0:
                    p_venda = float(
                        str(subset_venda.iloc[0]["Preço de Compra"]).replace(",", "."))
                else:
                    p_venda = preco_medio_atual

                qtd_vendida = abs(diff)

                # Se posição anterior era > 0, vendemos parte ou tudo => apuramos PnL
                if qtd_dia_anterior > 0:
                    qtd_encerrada_long = min(qtd_vendida, qtd_dia_anterior)
                    realized_pnl = (p_venda - preco_medio_atual) * \
                        qtd_encerrada_long
                    pnl_realizado += realized_pnl

                    # Se excedeu a quantidade (virou short):
                    excedente_short = qtd_vendida - qtd_encerrada_long
                    if excedente_short > 0:
                        # Passou a posição para negativa
                        preco_medio_atual = p_venda
                    else:
                        # Continua com posição long ou zerou
                        if (qtd_dia_anterior + diff) == 0:  # zerou
                            preco_medio_atual = 0.0

                else:
                    # Posição anterior <= 0 => estamos aumentando short
                    # Ex.: -5 -> -10
                    # Podemos recalcular o preco médio do short via média ponderada:
                    qtd_antiga_short = abs(qtd_dia_anterior)
                    qtd_nova_short = qtd_vendida  # vendemos mais
                    qtd_total_short = qtd_antiga_short + qtd_nova_short
                    # média ponderada
                    novo_preco_short = (
                        (qtd_antiga_short * preco_medio_atual) +
                        (qtd_nova_short * p_venda)
                    ) / qtd_total_short
                    preco_medio_atual = novo_preco_short

            # ------------------------------------------------------------
            # 5.3) Atualiza posição final e colunas
            # ------------------------------------------------------------
            qtd_final = qtd_dia_nova
            df_fundo.loc[ativo, dia_operacao] = qtd_final
            df_fundo.loc[ativo, "Preco_Compra"] = preco_medio_atual
            df_fundo.loc[ativo, "PnL_Realizado"] = pnl_realizado

            # ------------------------------------------------------------
            # 5.4) Atualiza Preço de Fechamento Atual (para PnL_Atual)
            # ------------------------------------------------------------
            row_fech = df_b3_fechamento[df_b3_fechamento["Assets"] == ativo]
            if not row_fech.empty:
                valor_fechamento_str = row_fech[ultima_data_fechamento].values[0]
                # Ex.: "5.100,00" -> "5100.00"
                valor_fechamento_str = valor_fechamento_str.replace(
                    ".", "").replace(",", ".")
                try:
                    valor_fechamento_float = float(valor_fechamento_str)
                except:
                    valor_fechamento_float = 0.0
            else:
                valor_fechamento_float = 0.0

            df_fundo.loc[ativo,
                         "Preco_Fechamento_Atual"] = valor_fechamento_float

            # ------------------------------------------------------------
            # 5.5) Calcula PnL_Atual (Não Realizado) = mark-to-market
            #      com base em (preço fechamento - preco médio) * qtd


# ==========================================================
#               FUNÇÕES AUXILIARES
# ==========================================================

def processar_b3_portifolio():
    """
    Exemplo de função que carrega dois dataframes de CSV:
     - df_preco_de_ajuste_atual.csv : preços de fechamento (colunas de datas)
     - df_variacao.csv : variação diária dos ativos (colunas de datas)
    Retorna df_b3_fechamento e df_b3_variacao já tratados.
    """
    df_b3_fechamento = pd.read_csv("df_preco_de_ajuste_atual.csv")
    # df_b3_fechamento possui colunas: ['Assets', '2025-01-17', '2025-01-18', ...] (exemplo)

    df_b3_variacao = pd.read_csv("df_variacao.csv")
    # df_b3_variacao possui colunas: ['Assets', '2025-01-17', '2025-01-18', ...]

    # Tirar pontos das casas de milhar
    df_b3_fechamento = df_b3_fechamento.replace('\.', '', regex=True)
    df_b3_variacao = df_b3_variacao.replace('\.', '', regex=True)

    # Trocar vírgula por ponto
    df_b3_fechamento = df_b3_fechamento.replace(',', '.', regex=True)
    df_b3_variacao = df_b3_variacao.replace(',', '.', regex=True)

    # Converter para float (exceto a primeira coluna, que é 'Assets')
    df_b3_fechamento.iloc[:, 1:] = df_b3_fechamento.iloc[:, 1:].astype(float)
    df_b3_variacao.iloc[:, 1:] = df_b3_variacao.iloc[:, 1:].astype(float)

    return df_b3_fechamento, df_b3_variacao


def processar_dados_port():
    """
    Lê o arquivo CSV 'portifolio_posições.csv' e retorna a lista de ativos existentes.
    Também poderíamos retornar diretamente as quantidades.
    """
    df_assets = pd.read_csv(
        "portifolio_posições.csv")  # Espera colunas: ['Ativo','Quantidade', 'Dia de Compra', 'Preço de Compra', ...]
    if 'Unnamed: 0' in df_assets.columns:
        df_assets.rename(columns={'Unnamed: 0': 'Ativo'}, inplace=True)
    assets_iniciais = df_assets['Ativo'].tolist()
    return assets_iniciais


def checkar_portifolio(assets,
                       quantidades,
                       # dict { 'PETR4': 100.0, ... } se usuário inserir valor de compra
                       compra_especifica,
                       # dict { 'PETR4': '2025-01-17', ... } se usuário inserir data de compra
                       dia_compra,
                       df_b3_fechamento,
                       df_b3_variacao):
    """
    Atualiza (ou cria) o CSV 'portifolio_posições.csv' com as novas posições, datas e preços de compra específicos.
    Em seguida, calcula colunas como: 'Preço de Ajuste Atual', 'Variação de Taxa' e 'Rendimento', caso deseje.
    """

    nome_arquivo_portifolio = 'portifolio_posições.csv'

    # 1) Carrega portfólio existente (se existir)
    if os.path.exists(nome_arquivo_portifolio):
        df_portifolio = pd.read_csv(nome_arquivo_portifolio, index_col=None)
        # Caso a 1ª coluna seja "Unnamed: 0", ajusta:
        if 'Unnamed: 0' in df_portifolio.columns:
            df_portifolio.drop(columns=['Unnamed: 0'], inplace=True)
    else:
        # Se não existe, cria um DataFrame vazio com colunas padronizadas
        df_portifolio = pd.DataFrame(columns=[
            'Ativo',
            'Quantidade',
            'Dia de Compra',
            'Preço de Compra',
            'Preço de Ajuste Atual',
            'Variação de Taxa',
            'Rendimento'
        ])

    # Garantir que 'Ativo' será a chave principal
    if 'Ativo' not in df_portifolio.columns:
        df_portifolio['Ativo'] = []

    # 2) Remover do DataFrame os ativos que NÃO estão na nova lista
    #    (significa que o usuário removeu aquele ativo do portfólio)
    ativos_existentes_csv = df_portifolio['Ativo'].tolist()
    for ativo_em_port in ativos_existentes_csv:
        if ativo_em_port not in assets:
            df_portifolio = df_portifolio[df_portifolio['Ativo']
                                          != ativo_em_port]

    # 3) Adicionar ou atualizar as quantidades dos ativos que estão na nova lista
    #    Se há compra_especifica, ela sobrescreve a coluna 'Preço de Compra'
    #    Também atualizamos o 'Dia de Compra' se fornecido
    for i, asset in enumerate(assets):
        qtd_final = quantidades[i]  # Quantidade informada pelo usuário

        # Verifica se esse ativo existe no df_portifolio
        if asset not in df_portifolio['Ativo'].values:
            # Cria uma linha nova
            new_row = {
                'Ativo': asset,
                'Quantidade': qtd_final,
                'Dia de Compra': dia_compra.get(asset, None) if isinstance(dia_compra, dict) else dia_compra,
                'Preço de Compra': compra_especifica.get(asset, np.nan) if compra_especifica else np.nan,
                'Preço de Ajuste Atual': np.nan,
                'Variação de Taxa': np.nan,
                'Rendimento': np.nan
            }
            df_portifolio = pd.concat(
                [df_portifolio, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Se já existe, apenas atualiza
            df_portifolio.loc[df_portifolio['Ativo']
                              == asset, 'Quantidade'] = qtd_final

            # Atualiza dia_compra, se houver
            if isinstance(dia_compra, dict):
                if asset in dia_compra and dia_compra[asset] is not None:
                    df_portifolio.loc[df_portifolio['Ativo'] ==
                                      asset, 'Dia de Compra'] = dia_compra[asset]
            else:
                # Se dia_compra for só uma string ou None
                if dia_compra is not None:
                    df_portifolio.loc[df_portifolio['Ativo']
                                      == asset, 'Dia de Compra'] = dia_compra

            # Atualiza preço de compra específico, se houver
            if compra_especifica and (asset in compra_especifica):
                df_portifolio.loc[df_portifolio['Ativo'] == asset,
                                  'Preço de Compra'] = compra_especifica[asset]

    # 4) Calcular colunas de 'Preço de Ajuste Atual', 'Variação de Taxa' e 'Rendimento'
    #    Precisamos de data_hoje_str e df_b3_fechamento/df_b3_variacao
    data_hoje_str = datetime.date.today().strftime('%Y-%m-%d')

    for index, row in df_portifolio.iterrows():
        asset = row['Ativo']

        # Localiza no df_b3_fechamento
        if asset not in df_b3_fechamento['Assets'].values:
            # Se não encontrar no df, pula ou zera
            continue

        # Tenta ler a data de compra
        dia_compra_ativo = row['Dia de Compra']
        # Se dia_compra_ativo é NaN ou None, não conseguimos calcular Preço de Compra pelo CSV B3
        # Mas se o usuário definiu compra_especifica, já temos algo na coluna.

        # PREÇO DE COMPRA (se não tiver na coluna, podemos buscar no CSV b3 se a data existir)
        preco_compra_atual = row['Preço de Compra']

        # PREÇO DE AJUSTE ATUAL = valor na coluna de data_hoje_str, se existir
        try:
            mask_ativo = (df_b3_fechamento['Assets'] == asset)
            if data_hoje_str in df_b3_fechamento.columns:
                preco_ajuste_atual = df_b3_fechamento.loc[mask_ativo,
                                                          data_hoje_str].values[0]
            else:
                # Caso não exista a coluna com data_hoje_str no CSV, pega a última coluna
                ult_col = df_b3_fechamento.columns[-1]
                if ult_col != 'Assets':
                    preco_ajuste_atual = df_b3_fechamento.loc[mask_ativo,
                                                              ult_col].values[0]
                else:
                    preco_ajuste_atual = np.nan
        except:
            preco_ajuste_atual = np.nan

        df_portifolio.loc[index, 'Preço de Ajuste Atual'] = preco_ajuste_atual

        # VARIAÇÃO DE TAXA
        # Pode ser diária ou acumulada, depende do que você deseja.
        # Para simplificar, supomos que a variação é na data de compra, mas se não tiver data de compra, pula.
        try:
            if isinstance(dia_compra_ativo, str) and dia_compra_ativo in df_b3_variacao.columns:
                variacao_taxa = df_b3_variacao.loc[df_b3_variacao['Assets']
                                                   == asset, dia_compra_ativo].values[0]
            else:
                # Se não tiver a data, pode ser NaN ou a última data
                variacao_taxa = np.nan
            df_portifolio.loc[index, 'Variação de Taxa'] = variacao_taxa
        except:
            df_portifolio.loc[index, 'Variação de Taxa'] = np.nan

        # RENDIMENTO = Quantidade * (Preço Ajuste Atual - Preço de Compra)
        # Se não tiver Preço de Compra, fica NaN
        if not pd.isna(preco_compra_atual) and not pd.isna(preco_ajuste_atual):
            df_portifolio.loc[index, 'Rendimento'] = row['Quantidade'] * \
                (preco_ajuste_atual - preco_compra_atual)
        else:
            df_portifolio.loc[index, 'Rendimento'] = np.nan

    # 5) Salva o DataFrame atualizado de volta no CSV
    df_portifolio.reset_index(drop=True, inplace=True)
    df_portifolio.to_csv(nome_arquivo_portifolio, index=False)

    return df_portifolio


# ------------------------------------------------
# Exemplo de outras funções de análise/risco (opcional)
# ------------------------------------------------
def process_portfolio(df_pl, Weights):
    df_pl['PL'] = (
        df_pl['PL']
        .str.replace('R$', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .replace('--', np.nan)
        .astype(float)
    )
    df_pl = df_pl.iloc[[5, 9, 10, 11, 17, 18, 19, 20, 22]]
    soma_sem_pesos = df_pl['PL'].sum()
    df_pl['Weights'] = Weights
    df_pl['Weights'] = df_pl['Weights'].astype(float)
    df_pl['PL_atualizado'] = df_pl['PL'] * df_pl['Weights']
    df_pl['Adm'] = ['SANTANDER', 'BTG', 'SANTANDER',
                    'SANTANDER', 'BTG', 'BTG', 'BTG', 'BTG', 'BTG']
    return df_pl, df_pl['PL_atualizado'].sum(), soma_sem_pesos


# ------------------------------------------------
#   Configurações do Streamlit
# ------------------------------------------------
st.set_page_config(page_title="Dashboard de Análise", layout="wide")

# CSS customizado (exemplo simplificado)


def add_custom_css():
    st.markdown(
        """
        <style>
         section[data-testid="stSidebar"] * {
            color: White;
         }
         /* Exemplo: cor de fundo do sidebar */
         div[data-testid="stSidebar"] {
            background-color: #2F4F4F;
         }
        </style>
        """,
        unsafe_allow_html=True,
    )


add_custom_css()


# ------------------------------------------------
#   Definição das "páginas" do app
# ------------------------------------------------
def main_page():
    st.title("Página Principal - Edição de Portfólio")

    st.write("""
    Nesta página, você pode:
    1. Ler seu arquivo `portifolio_posições.csv` atual.
    2. Selecionar quais ativos quer manter no portfólio ou adicionar novos.
    3. Ajustar as quantidades.
    4. Salvar/Atualizar o CSV.
    """)

    # Carrega arquivo CSV com as posições
    assets_iniciais = processar_dados_port()  # lista de ativos
    df_positions = pd.read_csv("portifolio_posições.csv")

    # Garante colunas mínimas
    if 'Quantidade' not in df_positions.columns:
        df_positions['Quantidade'] = 0

    # Lista de ativos disponíveis (carregada do CSV) --
    # Se quiser permitir o usuário digitar ativos novos, podemos permitir via text_input
    st.subheader("Selecione ou adicione novos ativos")
    all_assets_currently = list(df_positions['Ativo'].unique())

    # Cria um multiselect com os ativos existentes do CSV
    selected_assets = st.multiselect(
        "Ativos do Portfólio",
        options=all_assets_currently,
        default=assets_iniciais  # Seleciona por padrão os que já estavam
    )

    # Campo para inserir um novo ativo, caso não exista
    novo_ativo = st.text_input(
        "Adicionar Ativo Manualmente (se não estiver na lista):", "")
    if novo_ativo:
        if novo_ativo not in all_assets_currently:
            all_assets_currently.append(novo_ativo)
            selected_assets.append(novo_ativo)

    # Monta um dicionário {ativo: quantidade} para exibir e editar
    quantidades_dict = {}
    for ativo in selected_assets:
        # Pega quantidade atual do CSV (se existir)
        mask = (df_positions['Ativo'] == ativo)
        if mask.any():
            qtd_atual = float(df_positions.loc[mask, 'Quantidade'].values[0])
        else:
            qtd_atual = 0
        new_qtd = st.number_input(
            f"Quantidade para {ativo}:", value=qtd_atual, step=1.0)
        quantidades_dict[ativo] = new_qtd

    # Botão para atualizar o CSV
    if st.button("Atualizar/Salvar Novo Portfólio"):
        # Precisamos chamar checkar_portifolio
        # Mas checkar_portifolio espera listas e dicts bem definidos
        assets_list = list(quantidades_dict.keys())
        qtd_list = list(quantidades_dict.values())

        # Para agora, não passaremos dia_compra e compra_especifica (passaremos {}), pois isso fica na página 2
        compra_especifica = {}
        dia_compra = {}

        # Precisamos do df_b3_fechamento e df_b3_variacao:
        df_b3_fechamento, df_b3_variacao = processar_b3_portifolio()

        df_port_updated = checkar_portifolio(
            assets=assets_list,
            quantidades=qtd_list,
            compra_especifica=compra_especifica,
            dia_compra=dia_compra,
            df_b3_fechamento=df_b3_fechamento,
            df_b3_variacao=df_b3_variacao
        )
        st.success("Portfólio atualizado com sucesso!")
        st.write(df_port_updated)

    st.write("---")
    st.write("Visualização atual do CSV `portifolio_posições.csv`:")
    st.dataframe(df_positions)


def second_page():
    st.title("Página 2 - Dados de Compra do Portfólio")

    st.write("""
    Nesta página, você pode:
    1. Ler o portfólio já atualizado (CSV).
    2. Definir a data de compra de cada ativo.
    3. Inserir um preço de compra específico (se desejar).
    4. Salvar novamente no CSV.
    """)

    # Lê novamente as posições do CSV atualizado
    df_positions = pd.read_csv("portifolio_posições.csv")

    if df_positions.empty:
        st.warning(
            "O portfólio está vazio. Volte à Página Principal para adicionar ativos.")
        return

    # Lê df de B3
    df_b3_fechamento, df_b3_variacao = processar_b3_portifolio()

    # Vamos construir dois dicionários para passar ao checkar_portifolio:
    #  - dia_compra = { 'AtivoX': 'YYYY-MM-DD', ... }
    #  - compra_especifica = { 'AtivoX': valor, ... }
    dia_compra_dict = {}
    compra_especifica_dict = {}

    st.write("#### Edite os campos abaixo para cada Ativo:")
    for idx, row in df_positions.iterrows():
        ativo = row['Ativo']
        col1, col2, col3 = st.columns([3, 3, 3])
        with col1:
            st.markdown(f"**{ativo}**")

        with col2:
            # Dia de Compra
            # Se já estiver no CSV, converta em datetime. Se não, use today como default
            default_date = datetime.date.today()
            if not pd.isna(row.get('Dia de Compra', None)):
                try:
                    default_date = datetime.datetime.strptime(
                        str(row['Dia de Compra']), "%Y-%m-%d").date()
                except:
                    pass

            new_date = st.date_input(
                f"Dia de Compra - {ativo}", value=default_date, key=f"dia_{ativo}")
            dia_compra_dict[ativo] = str(new_date)  # salva string YYYY-MM-DD

        with col3:
            # Preço de Compra Específico
            default_compra = 0.0
            if not pd.isna(row.get('Preço de Compra', None)):
                default_compra = float(row['Preço de Compra'])

            new_compra = st.number_input(
                f"Preço Compra - {ativo}", value=default_compra, step=1.0, key=f"compra_{ativo}")
            # Se o usuário deixar 0, interpretamos como "não quero customizar"
            if new_compra > 0:
                compra_especifica_dict[ativo] = new_compra

        st.markdown("---")

    # Botão para salvar
    if st.button("Salvar Dados de Compra"):
        # Montar os assets e quantidades
        assets_list = df_positions['Ativo'].tolist()
        qtd_list = df_positions['Quantidade'].tolist()

        # Chama novamente checkar_portifolio
        updated_port = checkar_portifolio(
            assets=assets_list,
            quantidades=qtd_list,
            compra_especifica=compra_especifica_dict,
            dia_compra=dia_compra_dict,
            df_b3_fechamento=df_b3_fechamento,
            df_b3_variacao=df_b3_variacao
        )
        st.success("Dados de compra salvos com sucesso!")
        st.write(updated_port)

    st.write("### Portfólio Atualizado (visualização):")
    st.dataframe(df_positions)


# ------------------------------------------------
#   Lógica de Navegação
# ------------------------------------------------
PAGINAS = {
    "Página Principal": main_page,
    "Página 2 (Compras)": second_page
}

# Sidebar para navegar
st.sidebar.title("Navegação")
choice = st.sidebar.radio("Ir para:", list(PAGINAS.keys()))
PAGINAS[choice]()


def atualizar_csv_fundos(
    df_current,         # DataFrame do dia atual (1 linha por Fundo)
    dia_operacao,       # Exemplo: "2025-01-20"
    # DF de transações: [Ativo, Quantidade, Dia de Compra, Preço de Compra, ...]
    df_info,
    # DF de preços de fechamento B3: colunas ["Assets", <data1>, <data2>, ...]
):
    """
    - O CSV final de cada fundo terá linhas = cada ativo (ou "PL"),
      e colunas básicas = ["Ativo","Preco_Compra","Preco_Fechamento_Atual", ...],
      além das colunas diárias geradas a cada dia de operação:
         <dia_operacao> - Quantidade
         <dia_operacao> - Preço Pago
         <dia_operacao> - Rendimento
    - Lógica geral:
       1) Se houver aumento de posição no mesmo dia (quantidade final > anterior),
          recalculamos o preço médio (ponderado).
       2) A coluna "Preco_Fechamento_Atual" vem do CSV de preços B3 para a
          última data disponível.
       3) Tratamento especial da linha "PL":
          - Se não existir, criamos com quantidade definida (ex.: 1.0).
          - Armazenamos o PL do dia em "Preco_Fechamento_Atual".
    """
    # ------------------------------------------------------------------
    # 1) Carregar (ou processar) DF de preços de fechamento
    # ------------------------------------------------------------------
    # Supondo que o arquivo local seja "df_preco_de_ajuste_atual.csv"
    df_fechamento_b3 = pd.read_csv("df_preco_de_ajuste_atual.csv")
    colunas_b3 = list(df_fechamento_b3.columns)
    colunas_b3.remove("Assets")
    colunas_b3_ordenadas = sorted(colunas_b3)
    ultima_data_fechamento = colunas_b3_ordenadas[-1]  # ex: "2025-01-20"

    # ------------------------------------------------------------------
    # 2) Iterar cada fundo (linha) em df_current
    # ------------------------------------------------------------------
    for fundo, row_fundo in df_current.iterrows():
        # Caminho do CSV do Fundo
        nome_arquivo_csv = os.path.join("BaseFundos", f"{fundo}.csv")

        # 2.1) Carregar (ou criar) o DataFrame histórico do Fundo (df_fundo)
        if os.path.exists(nome_arquivo_csv):
            df_fundo = pd.read_csv(nome_arquivo_csv, index_col=None)
        else:
            df_fundo = pd.DataFrame(columns=["Ativo",
                                             "Preco_Compra",
                                             "Preco_Fechamento_Atual"])

        # 2.2) Garante que "Ativo" seja índice (mas mantendo a coluna)
        if "Ativo" in df_fundo.columns:
            df_fundo.set_index("Ativo", inplace=True, drop=False)

        # ----------------------------------------------------------------
        # 3) Verifica se existe coluna "PL" em df_current
        # ----------------------------------------------------------------
        valor_pl = None
        if "PL" in df_current.columns:
            valor_pl = row_fundo["PL"]  # PL atual do fundo

        # ----------------------------------------------------------------
        # 4) Identificar colunas de CONTRATOS no df_current
        # ----------------------------------------------------------------
        colunas_contratos = [
            c for c in df_current.columns
            if c.startswith("Contratos ")
        ]
        # Exemplo: {"Contratos WDO1": 10, "Contratos DI_33": -25, ...}
        serie_contratos = row_fundo[colunas_contratos]

        # ----------------------------------------------------------------
        # 5) Para cada ativo do df_current, precisamos garantir 3 colunas novas:
        #    <dia_operacao> - Quantidade
        #    <dia_operacao> - Preço Pago
        #    <dia_operacao> - Rendimento
        # ----------------------------------------------------------------
        col_qtd = f"{dia_operacao} - Quantidade"
        col_preco_pago = f"{dia_operacao} - Preço Pago"
        col_rendimento = f"{dia_operacao} - Rendimento"

        for c_ in [col_qtd, col_preco_pago, col_rendimento]:
            if c_ not in df_fundo.columns:
                df_fundo[c_] = 0.0

        # ----------------------------------------------------------------
        # 6) Se existir PL, criar/atualizar linha "PL"
        # ----------------------------------------------------------------
        if valor_pl is not None:
            if "PL" not in df_fundo.index:
                # Linha PL não existe: criamos com alguma quantidade específica
                df_fundo.loc["PL", "Ativo"] = "PL"
                df_fundo.loc["PL", "Preco_Compra"] = np.nan
                # Aqui, assumimos que você queira dar "1.0" de quantidade para PL,
                # mas pode alterar para outro valor, se preferir.
                df_fundo.loc["PL", col_qtd] = 1.0
            # Sempre que tiver PL, jogamos esse valor em Preco_Fechamento_Atual
            df_fundo.loc["PL", "Preco_Fechamento_Atual"] = valor_pl
            # Se quiser calcular "rendimento" do PL, defina a lógica.
            # Por default, deixaremos 0.0
            df_fundo.loc["PL", col_rendimento] = 0.0

        # ----------------------------------------------------------------
        # 7) Processar cada ativo (contratos)
        # ----------------------------------------------------------------
        for col_contrato, qtd_dia_raw in serie_contratos.items():
            ativo = col_contrato.replace("Contratos ", "").strip()

            # Converte a quantidade para float
            try:
                qtd_dia_nova = float(qtd_dia_raw)
            except:
                qtd_dia_nova = 0.0

            # Se o ativo ainda não existe no df_fundo, cria a linha
            if ativo not in df_fundo.index:
                df_fundo.loc[ativo, "Ativo"] = ativo
                df_fundo.loc[ativo, "Preco_Compra"] = 0.0
                df_fundo.loc[ativo, "Preco_Fechamento_Atual"] = 0.0

            # Carrega o preço médio atual (se for NaN, vira 0.0)
            preco_medio_atual = df_fundo.loc[ativo, "Preco_Compra"]
            if pd.isna(preco_medio_atual):
                preco_medio_atual = 0.0

            # 7.1) Descobre a quantidade que já estava neste MESMO dia
            qtd_dia_anterior = df_fundo.loc[ativo, col_qtd]
            if pd.isna(qtd_dia_anterior):
                qtd_dia_anterior = 0.0

            diff = qtd_dia_nova - qtd_dia_anterior
            # 7.2) Se houve compra adicional (diff > 0), recalcular preço médio
            if diff > 0:
                # Localizar no df_info as linhas para este dia/ativo
                subset = df_info[
                    (df_info["Ativo"] == ativo) &
                    (df_info["Dia de Compra"] == dia_operacao)
                ]
                if len(subset) == 0:
                    # Se não encontrou nada em df_info, mantemos o preço anterior
                    preco_compra_dia = preco_medio_atual
                else:
                    # Faz soma ponderada das quantidades positivas
                    qtd_total_dia = 0.0
                    valor_total_dia = 0.0
                    for i, r in subset.iterrows():
                        q_linha = float(str(r["Quantidade"]).replace(",", "."))
                        p_linha = float(
                            str(r["Preço de Compra"]).replace(",", "."))
                        if q_linha > 0:
                            qtd_total_dia += q_linha
                            valor_total_dia += (q_linha * p_linha)
                    if qtd_total_dia == 0:
                        preco_compra_dia = preco_medio_atual
                    else:
                        preco_compra_dia = valor_total_dia / qtd_total_dia

                # Novo preço médio = ponderação do que havia até qtd_dia_anterior + diff
                qtd_comprada = diff
                qtd_final = qtd_dia_nova
                novo_preco_medio = (
                    (qtd_dia_anterior * preco_medio_atual) +
                    (qtd_comprada * preco_compra_dia)
                ) / qtd_final

                df_fundo.loc[ativo, "Preco_Compra"] = novo_preco_medio
                # Também vamos registrar em "<dia> - Preço Pago" a média do dia
                df_fundo.loc[ativo, col_preco_pago] = preco_compra_dia

            elif diff < 0:
                # Se houve venda, normalmente não alteramos o preço médio
                # (a menos que queira zerar posição ou recalcular).
                # No caso de venda, se quiser, pode registrar "Preço Pago" do dia
                # com o mesmo valor do preço médio anterior, ou outro.
                df_fundo.loc[ativo, col_preco_pago] = preco_medio_atual
                subset = df_info[
                    (df_info["Ativo"] == ativo) &
                    (df_info["Dia de Compra"] == dia_operacao)
                ]
                for i, r in subset.iterrows():
                    p_linha = float(
                        str(r["Preço de Compra"]).replace(",", "."))
                    df_fundo.loc[ativo, col_preco_pago] = p_linha

            else:
                # Se não houve alteração de posição neste dia, mantemos o Preço Pago
                # para o que já estava (caso já tenha sido setado acima).
                if pd.isna(df_fundo.loc[ativo, col_preco_pago]):
                    df_fundo.loc[ativo, col_preco_pago] = 0.0

            # Atualiza a coluna de quantidade do dia
            df_fundo.loc[ativo, col_qtd] = qtd_dia_nova

            # 7.3) Atualizar Preço de Fechamento Atual com base em df_fechamento_b3
            row_fech = df_fechamento_b3[df_fechamento_b3["Assets"] == ativo]
            if not row_fech.empty:
                valor_fechamento_str = str(
                    row_fech[ultima_data_fechamento].values[0])
                # Exemplo de conversão "6.081,5680" -> "6081.5680"
                valor_fechamento_str = valor_fechamento_str.replace(
                    ".", "").replace(",", ".")
                try:
                    valor_fechamento_float = float(valor_fechamento_str)
                except:
                    valor_fechamento_float = 0.0
                df_fundo.loc[ativo,
                             "Preco_Fechamento_Atual"] = valor_fechamento_float
            else:
                df_fundo.loc[ativo, "Preco_Fechamento_Atual"] = np.nan

        # ----------------------------------------------------------------
        # 8) Calcular "Rendimento" do dia para cada ativo
        #    (exemplo: (Preco_Fechamento_Atual - Preco_Compra) * Quantidade_dia)
        # ----------------------------------------------------------------
        for ativo_idx in df_fundo.index:
            # Se for "PL", já tratamos lá em cima, mas se quiser,
            # podemos garantir zero ou outro valor aqui
            if ativo_idx == "PL":
                # Se desejar, pode redefinir aqui a fórmula de rendimento do PL
                continue

            preco_fech = df_fundo.loc[ativo_idx, "Preco_Fechamento_Atual"]
            preco_med = df_fundo.loc[ativo_idx, "Preco_Compra"]
            qtd_hoje = df_fundo.loc[ativo_idx, col_qtd]

            if (not pd.isna(preco_fech)) and (not pd.isna(preco_med)) and (not pd.isna(qtd_hoje)):
                rendimento_dia = (preco_fech - p_linha) * qtd_hoje
            else:
                rendimento_dia = 0.0

            df_fundo.loc[ativo_idx, col_rendimento] = rendimento_dia

        # ----------------------------------------------------------------
        # 9) Salvar o CSV do fundo
        # ----------------------------------------------------------------
        df_fundo.reset_index(drop=True, inplace=True)
        df_fundo.to_csv(nome_arquivo_csv, index=False, encoding="utf-8")

        print(f"[{fundo}] -> CSV atualizado: {nome_arquivo_csv}")


df_stress_div01 = pd.DataFrame({
    'DIV01': [
        f"R${df_divone_juros_nominais.iloc[0]:,.2f}",
        f"R${df_divone_juros_real.iloc[0]:,.2f}",
        f"R${df_divone_juros_externo.iloc[0]:,.2f}" if lista_juros_externo else f"R${df_divone_juros_externo.iloc[0]:,.2f}",
        ''
    ],
    'Stress (R$)': [
        f"R${stress_test_juros_interno_Nominais['FUT_TICK_VAL']:,.2f}",
        f"R${stress_test_juros_interno_Reais['FUT_TICK_VAL']:,.2f}",
        f"R${stress_test_juros_externo:,.2f}" if lista_juros_externo else stress_test_juros_externo[
            'FUT_TICK_VAL'],
        f"R${stress_dolar:.2f}"
    ],
    'Stress (bps)': [
        f"{stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL']:,.2f}bps",
        f"{stress_test_juros_interno_Reais_percent['FUT_TICK_VAL']:,.2f}bps",
        f"{stress_test_juros_externo_percent:,.2f}bps" if lista_juros_externo else f"{stress_test_juros_externo_percent['FUT_TICK_VAL']:,.2f}bps",
        f"{stress_dolar_percent:,.2f}bps"
    ]
}, index=['Juros Nominais Brasil', 'Juros Reais Brasil', 'Juros US', 'Moedas'])

sum_row = pd.DataFrame({
    'DIV01': [f"R${df_divone_juros_nominais.iloc[0] + df_divone_juros_real[0] + df_divone_juros_externo:,.2f}" if lista_juros_externo else f"R${df_divone_juros_nominais.iloc[0] + df_divone_juros_real.iloc[0] + df_divone_juros_externo.iloc[0]:,.2f}"],
    'Stress (R$)': [f"R${stress_test_juros_interno_Nominais['FUT_TICK_VAL'] + stress_test_juros_interno_Reais['FUT_TICK_VAL'] + stress_test_juros_externo + stress_dolar:,.2f}"] if lista_juros_externo else [f"R${stress_test_juros_interno_Nominais['FUT_TICK_VAL'] + stress_test_juros_interno_Reais['FUT_TICK_VAL'] + stress_test_juros_externo['FUT_TICK_VAL'] + stress_dolar:,.2f}"],
    'Stress (bps)': [f"{stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL'] + stress_test_juros_interno_Reais_percent['FUT_TICK_VAL'] + stress_test_juros_externo_percent + stress_dolar_percent:,.2f}bps" if lista_juros_externo else f"{stress_test_juros_interno_Nominais_percent['FUT_TICK_VAL'] + stress_test_juros_interno_Reais_percent['FUT_TICK_VAL'] + stress_test_juros_externo_percent['FUT_TICK_VAL'] + stress_dolar_percent:,.2f}bps"]
}, index=['Total'])
df_stress_div01 = pd.concat([df_stress_div01, sum_row])

df_precos_ajustados = calculate_portfolio_values(
    df_precos_ajustados, df_pl_processado, var_bps)
df_pl_processado = calculate_contracts_per_fund(
    df_pl_processado, df_precos_ajustados)





























df_ativo = dict_result["df_diario_fundo_ativo"]
            df_estr = dict_result["df_diario_fundo_estrategia"]
            df_fundo = dict_result["df_diario_fundo_total"]

            # REMOVER linhas onde Ativo == "PL"
            df_ativo = df_ativo[df_ativo["Ativo"] != "PL"]

            df_teste_fundos =df_ativo[df_ativo['fundo'] != 'Total']
            df_teste = df_ativo[df_ativo['fundo'] == 'Total']

            #Agrupar por dia
            df_teste_dia = df_teste.groupby(['date']).sum().reset_index()
            df_teste_dia.drop(columns=['fundo','Ativo','Estratégia'], inplace=True)
            df_teste_estrategia = df_teste[['Estratégia','Rendimento_diario']]
            df_teste_estrategia = df_teste_estrategia.groupby(['Estratégia']).sum().reset_index()

            df_teste_estrategia_data = df_teste.groupby(['date','Estratégia']).sum().reset_index()
            df_teste_estrategia_data.drop(columns=['fundo','Ativo'], inplace=True)
            st.write("## Performance Diária Total")               
            col1, col2 = st.columns([7, 3])
            st.write('---')
            st.write("## Performance Diária Total")
            #Criar Multiselect para escolher os fundos ativos e estratégias

            st.table(df_final)

            st.write('---')
            st.write("## Performance Diária por PL")
            st.table(df_final_pl)
            #Criar gráficos de barra para performance diária
            with col1:
                sns.set_theme(style="whitegrid")
                fig, ax = plt.subplots(figsize=(15, 6))
                sns.barplot(x='date', y='Rendimento_diario', data=df_teste_dia, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            with col2:
                st.table(df_teste_dia)

            if tipo_visao == "Estratégia":
                st.write('---')
                st.write("## Performance Diária por Estratégia")
                coll1, coll2 = st.columns([1, 1])

                with coll1:
                    fig, ax = plt.subplots(figsize=(15, 6))
                    sns.barplot(x='Estratégia', y='Rendimento_diario', data=df_teste_estrategia, ax=ax)
                    st.pyplot(fig)
                with coll2:
                    st.table(df_teste_estrategia)
                    st.table(df_teste_estrategia_data)
            
            if tipo_visao == "Fundo":
                st.write('---')
                st.write("## Performance Diária por Fundo")
                fig, ax = plt.subplots(figsize=(15, 6))
                sns.barplot(x='date', y='Rendimento_diario', hue='fundo', data=df_teste_fundos, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            if tipo_visao == "Ativo":                
                st.write('---')
                st.write("## Performance Diária por Ativo")
                fig, ax = plt.subplots(figsize=(15, 6))
                sns.lineplot(x='date', y='Rendimento_diario', hue='Ativo', data=df_teste, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
