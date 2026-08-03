-- =============================================================
-- Fase 1 do Dash Risco v2 — Persistência histórica + Governance
--
-- Cria 5 tabelas novas + faz seed dos parâmetros de risco em
-- config_risco. Idempotente (create if not exists / on conflict).
--
-- Rode no SQL Editor do Supabase. Como a service_role bypassa RLS,
-- não precisa de policies pra escrever.
-- =============================================================


-- =============================================================
-- 1) config_risco — parâmetros gerenciáveis via SQL
-- =============================================================
create table if not exists public.config_risco (
    "parametro"       text primary key,
    "valor"           jsonb not null,
    "descricao"       text,
    "atualizado_em"   timestamptz not null default now(),
    "atualizado_por"  text
);

-- Seed dos defaults acordados
insert into public.config_risco ("parametro", "valor", "descricao") values
    ('vol_target',               '0.04',      '4% a.a. — vol target de referência para calibração de stop-loss'),
    ('var_limite_base_bps',      '1.0',       'Orçamento base de VaR em bps (aplicado sobre PL)'),
    ('lambda_var_brw',           '0.99',      'Lambda do age-weighted historical VaR (BRW). Meia-vida ~69 dias.'),
    ('n_vols_stop_gatilho',      '1.0',       'Nº de vol_target para acionar stop-loss (DD gatilho = -N × vol_target)'),
    ('n_vols_stop_liberacao',    '0.5',       'Nº de vol_target para liberar stop-loss (histerese assimétrica)'),
    ('fator_reducao_stop',       '0.5',       'Fator aplicado ao VaR limite quando stop-loss ativo (0.5 = metade)'),
    ('ipca_projecao_anbima_pct', '0.05',      'Projeção IPCA mensal ANBIMA em % (valor placeholder até 1º scraping)'),
    ('ipca_projecao_valida_ate', '"2099-12-31"',      'Data até quando a projeção acima é válida (ISO)'),
    ('emails_diario',            '["marcos.freitas@afinvest.com.br"]', 'Destinatários do email diário'),
    ('emails_mensal',            '["marcos.freitas@afinvest.com.br"]', 'Destinatários do email mensal'),
    ('emails_alertas',           '["marcos.freitas@afinvest.com.br"]', 'Destinatários de alertas (breach, stop)')
on conflict ("parametro") do nothing;


-- =============================================================
-- 2) snapshot_diario — foto do portfólio inteiro por dia
-- =============================================================
create table if not exists public.snapshot_diario (
    "Data"                    date primary key,

    -- Cota e retorno
    "cota"                    numeric,
    "pl_total"                numeric,
    "retorno_dtd"             numeric,     -- decimal (0.0012 = 12bps)
    "retorno_mtd"             numeric,
    "retorno_ytd"             numeric,
    "cdi_dtd"                 numeric,
    "cdi_mtd"                 numeric,
    "cdi_ytd"                 numeric,

    -- Volatilidade (anualizada)
    "vol_20d"                 numeric,
    "vol_60d"                 numeric,
    "vol_252d"                numeric,
    "vol_ewma"                numeric,

    -- Drawdown
    "dd_atual"                numeric,     -- decimal negativo (-0.02 = -2%)
    "dd_max_hist"             numeric,
    "dias_em_dd"              int,

    -- VaR (3 variantes lado a lado)
    "var_hist_ew_bps"         numeric,     -- historical equal-weighted
    "var_hist_bw_bps"         numeric,     -- BRW ponderado (λ=0.99)
    "var_param_bps"           numeric,     -- paramétrico normal
    "cvar_bps"                numeric,
    "var_hist_ew_reais"       numeric,
    "var_hist_bw_reais"       numeric,

    -- DV01 / Duration (todos em R$)
    "dv01_total"              numeric,
    "dv01_juros_nom"          numeric,     -- soma dos DIs
    "dv01_juros_real"         numeric,     -- soma dos DAPs
    "dv01_treasury"           numeric,
    "dv01_ntnb"               numeric,

    -- Parâmetros VIGENTES no dia — chave pra fix do gráfico "utilização"
    "var_limite_base_bps"     numeric,
    "var_limite_efet_bps"     numeric,     -- base × fator_governance
    "stop_loss_ativo"         boolean not null default false,
    "fator_governance"        numeric not null default 1.0,

    -- Metadata
    "computado_em"            timestamptz not null default now(),
    "fonte"                   text not null default 'batch_diario'
);

create index if not exists idx_snap_data on public.snapshot_diario ("Data" desc);


-- =============================================================
-- 3) snapshot_diario_por_ativo — quebra ativo-a-ativo
-- =============================================================
create table if not exists public.snapshot_diario_por_ativo (
    "Data"                date not null,
    "Ativo"               text not null,
    "quantidade_total"    numeric,     -- soma através de fundos
    "preco_fechamento"    numeric,
    "mtm_reais"           numeric,     -- qtd × preço
    "dv01_reais"          numeric,     -- DV01 unitário × qtd
    "var_hist_bw_bps"     numeric,     -- VaR do ativo standalone
    "component_var_bps"   numeric,     -- contribuição ao VaR do portfólio
    "computado_em"        timestamptz not null default now(),
    primary key ("Data", "Ativo")
);

create index if not exists idx_snap_ativo on public.snapshot_diario_por_ativo ("Ativo", "Data" desc);


-- =============================================================
-- 4) governance_state — estado corrente das regras
-- =============================================================
create table if not exists public.governance_state (
    "regra"          text primary key,
    "ativo"          boolean not null default false,
    "ativado_em"     timestamptz,
    "liberado_em"    timestamptz,
    "snapshot_disparo" jsonb,               -- ex: {dd_atual: -0.045, vol_target: 0.04, gatilho: -0.04}
    "atualizado_em"  timestamptz not null default now()
);

insert into public.governance_state ("regra", "ativo") values
    ('stop_loss_dd', false)
on conflict ("regra") do nothing;


-- =============================================================
-- 5) eventos_risco — audit trail (append-only)
-- =============================================================
create table if not exists public.eventos_risco (
    "id"              bigserial primary key,
    "timestamp"       timestamptz not null default now(),
    "tipo"            text not null,      -- 'breach_var', 'stop_ativado', 'stop_liberado', 'override_manual', 'projecao_atualizada'
    "severidade"      text not null,      -- 'info', 'warn', 'critical'
    "titulo"          text,
    "payload"         jsonb,
    "notificado"      boolean not null default false,
    "canal"           text,
    "notificado_em"   timestamptz
);

create index if not exists idx_evt_data on public.eventos_risco ("timestamp" desc);
create index if not exists idx_evt_tipo on public.eventos_risco ("tipo", "timestamp" desc);
create index if not exists idx_evt_nao_notif on public.eventos_risco ("notificado") where "notificado" = false;


-- =============================================================
-- Policies mínimas para as tabelas novas — espelha a política de
-- portfolio_posicoes / posicoes_por_fundo (só SELECT liberado; escrita
-- via service_role que bypassa RLS).
-- =============================================================
alter table public.snapshot_diario           enable row level security;
alter table public.snapshot_diario_por_ativo enable row level security;
alter table public.config_risco              enable row level security;
alter table public.governance_state          enable row level security;
alter table public.eventos_risco             enable row level security;

do $$
begin
    if not exists (select 1 from pg_policies where tablename = 'snapshot_diario' and policyname = 'read_all_temp') then
        create policy "read_all_temp" on public.snapshot_diario           for select to anon, authenticated using (true);
    end if;
    if not exists (select 1 from pg_policies where tablename = 'snapshot_diario_por_ativo' and policyname = 'read_all_temp') then
        create policy "read_all_temp" on public.snapshot_diario_por_ativo for select to anon, authenticated using (true);
    end if;
    if not exists (select 1 from pg_policies where tablename = 'config_risco' and policyname = 'read_all_temp') then
        create policy "read_all_temp" on public.config_risco              for select to anon, authenticated using (true);
    end if;
    if not exists (select 1 from pg_policies where tablename = 'governance_state' and policyname = 'read_all_temp') then
        create policy "read_all_temp" on public.governance_state          for select to anon, authenticated using (true);
    end if;
    if not exists (select 1 from pg_policies where tablename = 'eventos_risco' and policyname = 'read_all_temp') then
        create policy "read_all_temp" on public.eventos_risco             for select to anon, authenticated using (true);
    end if;
end$$;


-- =============================================================
-- Verificação
-- =============================================================
-- select count(*) from config_risco;                 -- deve dar 11
-- select * from config_risco order by "parametro";   -- inspeção dos defaults
-- select * from governance_state;                    -- deve ter 1 linha (stop_loss_dd, ativo=false)
