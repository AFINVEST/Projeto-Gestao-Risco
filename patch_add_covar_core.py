"""patch_add_covar_core.py - adiciona calcular_covar_ativo em risco_carteira_core.py"""
from pathlib import Path
import shutil, datetime as dt

f = Path("risco_carteira_core.py")
shutil.copy2(f, f"{f}.bak_covar_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

nova_funcao = '''

# ─────────────────────────────────────────────────────────────────────────────
# CoVaR por ativo (Euler / Component VaR via historical decomposition)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_covar_ativo(retornos_df: pd.DataFrame,
                          qty_dict: dict,
                          precos_dict: dict,
                          alpha: float = ALPHA_DEFAULT,
                          metodo: str = "hist") -> dict:
    """Decompoe VaR de carteira em contribuicao por ativo (component VaR).

    Metodo historical: pega os cenarios da cauda alpha% e computa a contribuicao
    media de cada ativo pra perda observada nesses dias.
      CoVaR_i = mean over tail of (w_i * r_i * MV_total)

    Retorna dict:
      {
        "covar_por_ativo_R": {ativo: contribuicao_media_R$},
        "covar_por_ativo_bps": {ativo: bps sobre MV},
        "covar_por_classe_R": {classe: sum_R$},
        "covar_por_classe_pct": {classe: fracao do CoVaR total},
        "var_estimado_R": soma_das_contribuicoes,
      }
    """
    ativos_com_pos = [a for a in qty_dict.keys() if a in retornos_df.columns and a in precos_dict]
    if not ativos_com_pos:
        return {"erro": "sem_posicoes"}

    mv = pd.Series({a: qty_dict[a] * precos_dict[a] for a in ativos_com_pos})
    mv_total = float(mv.abs().sum())
    if mv_total == 0:
        return {"erro": "mv_zero"}
    pesos = (mv / mv_total).astype(float)

    ret_hist = retornos_df[ativos_com_pos].dropna(how="all").fillna(0.0)
    # P&L simulado do portfolio por dia (fracao do MV)
    port_ret = (ret_hist * pesos).sum(axis=1)

    n = len(port_ret)
    if n < 30:
        return {"erro": "historico_insuficiente"}

    # tail: piores alpha% dias (cauda esquerda)
    threshold = float(port_ret.quantile(alpha))
    tail_mask = port_ret <= threshold
    if tail_mask.sum() == 0:
        return {"erro": "tail_vazia"}
    tail_days = port_ret.index[tail_mask]

    # Contribuicao de cada ativo: w_i * r_i(day) * MV_total (em R$)
    # Media sobre os dias da cauda
    contrib_R = {}
    for a in ativos_com_pos:
        # sinal do peso (long/short) preserva direcao correta
        r_i_tail = ret_hist.loc[tail_days, a]
        pnl_i = (pesos[a] * r_i_tail * mv_total).mean()
        contrib_R[a] = float(pnl_i)

    # Como estamos na cauda esquerda, contribuicoes esperadas sao negativas
    # Reportamos como VALOR ABSOLUTO (perda esperada)
    contrib_R_abs = {a: abs(v) for a, v in contrib_R.items()}
    var_estimado = sum(contrib_R_abs.values())

    # Por classe
    def _classe(a):
        au = str(a).upper()
        if au.startswith("DI_") or au.startswith("DI"): return "Juros Nominais BR"
        if au.startswith(("DAP", "NTNB")): return "Juros Reais BR"
        if "TREASURY" in au: return "Juros US"
        if au.startswith("WDO"): return "Moeda"
        return "Outros"

    classe_R = {}
    for a, v in contrib_R_abs.items():
        c = _classe(a)
        classe_R[c] = classe_R.get(c, 0.0) + v

    total = sum(classe_R.values())
    classe_pct = {c: (v/total if total else 0) for c, v in classe_R.items()}
    covar_bps = {a: (v/mv_total*10_000) for a, v in contrib_R_abs.items()}

    return {
        "covar_por_ativo_R":     contrib_R_abs,
        "covar_por_ativo_bps":   covar_bps,
        "covar_por_classe_R":    classe_R,
        "covar_por_classe_pct":  classe_pct,
        "var_estimado_R":        var_estimado,
        "n_scenarios_tail":      int(tail_mask.sum()),
        "mv_total":              mv_total,
        "metodo":                metodo,
    }


def calcular_covar_completo(data_ref: pd.Timestamp,
                             basefundos: dict | None = None,
                             janela_dias: int = JANELA_DIAS_DEFAULT,
                             alpha: float = ALPHA_DEFAULT,
                             base_dir: str | None = None) -> dict:
    """Orquestrador — carrega posicoes/precos/retornos e chama calcular_covar_ativo."""
    if base_dir:
        cwd0 = os.getcwd(); os.chdir(base_dir)
    else:
        cwd0 = None
    try:
        qty_dict = carregar_posicoes_atuais(data_ref, basefundos=basefundos)
        if not qty_dict: return {"erro": "sem_posicoes"}
        precos_dict = carregar_precos_atuais(data_ref)
        if not precos_dict: return {"erro": "sem_precos"}
        retornos_df = carregar_retornos_historicos(data_ref,
                                                     ativos=list(qty_dict.keys()),
                                                     janela_dias=janela_dias)
        if retornos_df.empty: return {"erro": "sem_retornos"}
        return calcular_covar_ativo(retornos_df, qty_dict, precos_dict, alpha=alpha)
    finally:
        if cwd0: os.chdir(cwd0)


'''

# Insere ANTES da linha "# ─── CLI ─── teste rapido"
marker = "# CLI — teste rapido"
if marker in s:
    s = s.replace(marker, nova_funcao + "\n# CLI — teste rapido")
    f.write_text(s, encoding="utf-8")
    print("[ok] calcular_covar_ativo + calcular_covar_completo adicionadas")
else:
    # fallback: append at end
    s += nova_funcao
    f.write_text(s, encoding="utf-8")
    print("[ok] Funcoes CoVaR anexadas ao fim do arquivo")
