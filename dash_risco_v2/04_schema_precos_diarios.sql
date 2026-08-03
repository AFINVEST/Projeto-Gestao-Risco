-- =============================================================
-- Fase 2a — Tabela precos_diarios
--
-- História persistente de preços/taxas para DI, DAP, WDO, TREASURY.
-- Elimina a dependência de parquet local sobreviver entre sessões.
--
-- Chave composta (Data, Ativo). Upsert idempotente.
-- =============================================================

create table if not exists public.precos_diarios (
    "Data"        date not null,
    "Ativo"       text not null,
    "PU_ajuste"   numeric,        -- preço de ajuste (para DI/DAP: PU do contrato; para outros: preço observado)
    "Taxa"        numeric,        -- taxa em % (para DI/DAP; NULL para futuros de moeda/treasury)
    "Fonte"       text not null default 'b3',   -- 'b3' | 'bbg' | 'manual' | 'derived'
    "created_at"  timestamptz not null default now(),
    "updated_at"  timestamptz not null default now(),
    primary key ("Data", "Ativo")
);

create index if not exists idx_precos_data  on public.precos_diarios ("Data" desc);
create index if not exists idx_precos_ativo on public.precos_diarios ("Ativo", "Data" desc);

-- Trigger de updated_at (reutiliza a fn_set_updated_at criada na Fase 1)
drop trigger if exists trg_precos_updated_at on public.precos_diarios;
create trigger trg_precos_updated_at
    before update on public.precos_diarios
    for each row execute function public.fn_set_updated_at();

-- RLS espelhando o padrão das outras tabelas
alter table public.precos_diarios enable row level security;

do $$
begin
    if not exists (
        select 1 from pg_policies
        where tablename = 'precos_diarios' and policyname = 'read_all_temp'
    ) then
        create policy "read_all_temp" on public.precos_diarios
            for select to anon, authenticated using (true);
    end if;
end$$;

-- Verificação
-- select count(*) from precos_diarios;   -- 0 imediatamente após criação
