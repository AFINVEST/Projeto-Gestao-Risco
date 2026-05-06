-- =============================================================
-- Tabela: posicoes_por_fundo
-- Substitui a função do diretório BaseFundos/ que armazena
-- a atribuição quantidade-por-fundo-por-ativo-por-data.
--
-- Convenção de nomes em PT-BR (com aspas) para casar com a
-- tabela existente "portfolio_posicoes".
-- =============================================================

create table if not exists public.posicoes_por_fundo (
    "Id"               bigserial primary key,
    "Fundo"            text     not null,
    "Ativo"            text     not null,
    "Data"             date     not null,
    "Quantidade"       numeric  not null default 0,
    "Preco_Compra"     numeric,
    "Preco_Fechamento" numeric,
    "PL"               numeric,
    "Rendimento"       numeric,
    "created_at"       timestamptz not null default now(),
    "updated_at"       timestamptz not null default now(),
    constraint posicoes_por_fundo_unico unique ("Fundo", "Ativo", "Data")
);

-- Índices para acelerar as consultas mais comuns
create index if not exists idx_ppf_fundo_ativo
    on public.posicoes_por_fundo ("Fundo", "Ativo");

create index if not exists idx_ppf_ativo
    on public.posicoes_por_fundo ("Ativo");

create index if not exists idx_ppf_data
    on public.posicoes_por_fundo ("Data");

-- Trigger para manter "updated_at" sempre coerente
create or replace function public.fn_set_updated_at()
returns trigger language plpgsql as $$
begin
    new."updated_at" = now();
    return new;
end;
$$;

drop trigger if exists trg_ppf_updated_at on public.posicoes_por_fundo;
create trigger trg_ppf_updated_at
    before update on public.posicoes_por_fundo
    for each row execute function public.fn_set_updated_at();

-- =============================================================
-- (Opcional) Permissões: ajuste conforme RLS do seu projeto.
-- Caso você use service_role no app, não precisa habilitar RLS.
-- =============================================================
-- alter table public.posicoes_por_fundo enable row level security;
-- create policy "service can do anything" on public.posicoes_por_fundo
--     for all using (true) with check (true);
