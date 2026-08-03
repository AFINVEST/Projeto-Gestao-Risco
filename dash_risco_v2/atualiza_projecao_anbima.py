"""
atualiza_projecao_anbima.py
============================

Scraper da projeção IPCA mensal ANBIMA. Atualiza o parâmetro
`ipca_projecao_anbima_pct` na tabela config_risco do Supabase.

Regra de escolha da projeção:
    A ANBIMA publica 2 coletas por mês (dia ~10 e dia ~28), cada uma
    com uma "Data de Validade". Escolhemos a projeção mais recente
    cuja Data de Validade ainda não passou (i.e., ainda vigente hoje).
    Se todas as validades passaram, usa a última publicada.

URL: https://www.anbima.com.br/pt_br/informar/estatisticas/precos-e-indices/projecao-de-inflacao-gp-m.htm

USO:
    export SUPABASE_URL='...'
    export SUPABASE_KEY='...'
    python atualiza_projecao_anbima.py

    # Ou passa o valor manualmente sem scraping:
    python atualiza_projecao_anbima.py --manual 0.32

Requisitos: pip install supabase requests beautifulsoup4
"""
from __future__ import annotations
import os
import sys
import re
import argparse
from datetime import datetime, date
import pandas as pd

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("ERRO: pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)

try:
    from supabase import create_client
except ImportError:
    print("ERRO: pip install supabase", file=sys.stderr)
    sys.exit(1)


URL_ANBIMA = ("https://www.anbima.com.br/pt_br/informar/"
              "estatisticas/precos-e-indices/projecao-de-inflacao-gp-m.htm")

HEADERS_HTTP = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}


def _parse_data_br(s: str) -> date | None:
    """Aceita 10/07/26 ou 10/07/2026."""
    s = s.strip()
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{2,4})$', s)
    if not m:
        return None
    d, mo, y = m.group(1), m.group(2), m.group(3)
    y = int(y)
    if y < 100:
        y += 2000
    try:
        return date(y, int(mo), int(d))
    except Exception:
        return None


def _parse_pct_br(s: str) -> float | None:
    """Aceita '0,23' ou '0.23' ou '0,23%' etc."""
    s = s.strip().replace('%', '').replace(',', '.').strip()
    try:
        return float(s)
    except Exception:
        return None


def buscar_projecoes_anbima(url: str = URL_ANBIMA) -> list[dict]:
    """Faz scraping da página ANBIMA. Devolve lista de dicts com
    keys: mes_ref, data_coleta, projecao_pct, data_validade, tipo."""
    print(f"Fetching {url}...")
    r = requests.get(url, headers=HEADERS_HTTP, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # A página tem uma aba "IPCA" com tabelas "Projeções para o mês"
    # e "Projeções para o mês posterior". Cada tabela tem colunas:
    # Mês de Coleta | Data | Projeção (%) | Data de Validade

    projecoes = []
    for tabela in soup.find_all("table"):
        rows = tabela.find_all("tr")
        if len(rows) < 2:
            continue
        # Detecta se cabeçalho contém "Projeção" e "Validade"
        header = " ".join(rows[0].get_text(" ", strip=True).lower().split())
        if 'projeção' not in header and 'projecao' not in header:
            continue
        if 'validade' not in header:
            continue

        # Tenta descobrir o "tipo" (mês corrente ou próximo) do contexto
        # anterior à tabela
        tipo_hint = ""
        prev = tabela.find_previous(["h2", "h3", "h4", "strong", "p", "caption"])
        if prev:
            tipo_hint = prev.get_text(" ", strip=True).lower()
        tipo = "posterior" if "posterior" in tipo_hint else "corrente"

        for tr in rows[1:]:
            cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if len(cols) < 4:
                continue
            mes_ref, data_coleta_s, proj_s, data_valid_s = cols[0], cols[1], cols[2], cols[3]
            d_coleta = _parse_data_br(data_coleta_s)
            d_valid  = _parse_data_br(data_valid_s)
            proj     = _parse_pct_br(proj_s)
            if proj is None:
                continue
            projecoes.append({
                'mes_ref': mes_ref,
                'data_coleta': d_coleta,
                'projecao_pct': proj,
                'data_validade': d_valid,
                'tipo': tipo,
            })

    return projecoes


def escolher_projecao_vigente(projecoes: list[dict], hoje: date | None = None) -> dict | None:
    """Escolhe a projeção do MÊS CORRENTE cuja validade ainda não passou
    (ou a mais recente se todas passaram)."""
    hoje = hoje or date.today()
    corrente = [p for p in projecoes if p['tipo'] == 'corrente']
    if not corrente:
        corrente = projecoes  # fallback: usa todas
    if not corrente:
        return None

    # Ordena por data_coleta desc
    corrente.sort(key=lambda x: x['data_coleta'] or date.min, reverse=True)

    # Preferência: projeção mais recente cuja validade ainda não passou
    for p in corrente:
        if p['data_validade'] and p['data_validade'] >= hoje:
            return p
    # Fallback: a mais recente publicada
    return corrente[0]


def atualizar_supabase(sb, projecao_pct: float, data_validade: date | None,
                       fonte_info: dict) -> None:
    valor_valida = f'"{data_validade.isoformat()}"' if data_validade else '"2099-12-31"'
    # Atualiza os 2 parâmetros
    sb.table("config_risco").update({
        "valor": str(projecao_pct),
        "atualizado_em": datetime.now().astimezone().isoformat(),
        "atualizado_por": "atualiza_projecao_anbima.py",
    }).eq("parametro", "ipca_projecao_anbima_pct").execute()

    if data_validade:
        sb.table("config_risco").update({
            "valor": f'"{data_validade.isoformat()}"',
            "atualizado_em": datetime.now().astimezone().isoformat(),
            "atualizado_por": "atualiza_projecao_anbima.py",
        }).eq("parametro", "ipca_projecao_valida_ate").execute()

    # Grava evento
    sb.table("eventos_risco").insert({
        "tipo": "projecao_atualizada",
        "severidade": "info",
        "titulo": f"Projeção IPCA ANBIMA atualizada: {projecao_pct}%",
        "payload": {
            "projecao_pct": projecao_pct,
            "data_validade": data_validade.isoformat() if data_validade else None,
            "mes_ref": fonte_info.get("mes_ref"),
            "data_coleta": (fonte_info.get("data_coleta") or "").isoformat() if fonte_info.get("data_coleta") else None,
        },
    }).execute()


def main():
    ap = argparse.ArgumentParser(description="Atualiza projeção IPCA ANBIMA em config_risco.")
    ap.add_argument("--manual", type=float, default=None,
                    help="Passa valor manual em %% (ex: 0.32). Pula o scraping.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Só imprime, não escreve no Supabase.")
    args = ap.parse_args()

    supa_url = os.environ.get("SUPABASE_URL")
    supa_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (supa_url and supa_key) and not args.dry_run:
        print("ERRO: SUPABASE_URL e SUPABASE_KEY obrigatórios (ou use --dry-run)", file=sys.stderr)
        sys.exit(1)

    if args.manual is not None:
        projecao = args.manual
        info = {"mes_ref": "manual", "data_coleta": date.today()}
        validade = None
        print(f"Modo manual: projeção = {projecao}%")
    else:
        try:
            todas = buscar_projecoes_anbima()
        except Exception as e:
            print(f"ERRO ao buscar ANBIMA: {e}", file=sys.stderr)
            print("Dica: use --manual <valor> para inserir manualmente.")
            sys.exit(1)

        print(f"\nProjeções extraídas ({len(todas)}):")
        for p in todas:
            print(f"  {p['tipo']:<10} mes={p['mes_ref']:<15} coleta={p['data_coleta']} "
                  f"proj={p['projecao_pct']}% valid={p['data_validade']}")

        vigente = escolher_projecao_vigente(todas)
        if vigente is None:
            print("Nenhuma projeção 'corrente' encontrada.", file=sys.stderr)
            sys.exit(1)
        projecao = vigente["projecao_pct"]
        info = vigente
        validade = vigente["data_validade"]
        print(f"\nProjeção vigente escolhida: {projecao}% (validade: {validade})")

    if args.dry_run:
        print("\n[dry-run] não atualizando Supabase.")
        return

    print(f"\nAtualizando config_risco...")
    sb = create_client(supa_url, supa_key)
    atualizar_supabase(sb, projecao, validade, info)
    print(f"OK. Verifica com:")
    print(f'  select * from config_risco where "parametro" like \'%ipca%\';')


if __name__ == "__main__":
    main()
