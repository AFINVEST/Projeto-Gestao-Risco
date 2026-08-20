"""patch_fix_covar_calc.py - corrige calculo CoVaR (Euler no quantil VaR)"""
from pathlib import Path
import shutil, datetime as dt

f = Path("risco_carteira_core.py")
shutil.copy2(f, f"{f}.bak_covarfix_{dt.datetime.now():%Y%m%d_%H%M%S}")
s = f.read_text(encoding="utf-8")

# Substitui o corpo do calcular_covar_ativo pelo metodo correto (Euler window at VaR quantile)
old = '''    n = len(port_ret)
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
    var_estimado = sum(contrib_R_abs.values())'''

new = '''    n = len(port_ret)
    if n < 30:
        return {"erro": "historico_insuficiente"}

    # Component VaR via Euler: usa janela de cenarios em torno do quantil VaR
    # (nao a cauda inteira, que seria CVaR/ES). Assim sum(CoVaR_i) ~= VaR.
    sorted_ret = port_ret.sort_values()
    idx_var = int(len(sorted_ret) * alpha)
    window = 3   # cenarios em torno do quantil VaR pra suavizar ruido
    lo = max(0, idx_var - window)
    hi = min(len(sorted_ret), idx_var + window + 1)
    scenarios = sorted_ret.iloc[lo:hi].index

    # Contribuicao SIGNED por ativo: w_i * E[r_i | port ~ VaR] * MV_total
    # (long+short netta corretamente; sum ~= -VaR)
    contrib_signed = {}
    for a in ativos_com_pos:
        r_i_win = ret_hist.loc[scenarios, a].mean()
        contrib_signed[a] = float(pesos[a] * r_i_win * mv_total)

    # Total signed ~= -VaR (perda esperada no quantil)
    sum_signed = sum(contrib_signed.values())
    var_estimado = abs(sum_signed)   # em R$

    # Reporta CoVaR por ativo como valor SIGNED (positivo = adiciona a perda; negativo = hedge)
    # Convencao: perda = valor positivo (multiplico por -1 do signed original)
    contrib_R_abs = {a: -v for a, v in contrib_signed.items()}'''

if old in s:
    s = s.replace(old, new)
    f.write_text(s, encoding="utf-8")
    print("[ok] Formula CoVaR corrigida (Euler no quantil VaR, sum ~= VaR)")
else:
    print("[warn] bloco original nao encontrado")
