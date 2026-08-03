-- =============================================================
-- Migração de naming dos ativos DI/DAP no Supabase
--
-- REGRA:
--   DI_YY  → DI_F<YY>   (todos os DIs existentes = janeiro)
--   DAPYY  → DAP_Q<YY>  se YY par     (agosto)
--   DAPYY  → DAP_K<YY>  se YY ímpar   (maio)
--
-- Idempotente: pode rodar 2x sem quebrar (só renomeia se ainda
-- estiver com nome antigo).
-- =============================================================


-- Backup rápido antes da migração (opcional mas recomendado)
-- Se algo der errado, `insert into portfolio_posicoes select * from portfolio_posicoes_backup_pre_naming`
create table if not exists public.portfolio_posicoes_backup_pre_naming as
    select * from public.portfolio_posicoes where 1=0;

insert into public.portfolio_posicoes_backup_pre_naming
    select * from public.portfolio_posicoes
    where "Ativo" ~ '^(DI_[0-9]+|DAP[0-9]+)$'
    and not exists (
        select 1 from public.portfolio_posicoes_backup_pre_naming p2
        where p2."Id" = portfolio_posicoes."Id"
    );

create table if not exists public.posicoes_por_fundo_backup_pre_naming as
    select * from public.posicoes_por_fundo where 1=0;

insert into public.posicoes_por_fundo_backup_pre_naming
    select * from public.posicoes_por_fundo
    where "Ativo" ~ '^(DI_[0-9]+|DAP[0-9]+)$'
    and not exists (
        select 1 from public.posicoes_por_fundo_backup_pre_naming p2
        where p2."Id" = posicoes_por_fundo."Id"
    );


-- =============================================================
-- portfolio_posicoes
-- =============================================================

-- DI_YY -> DI_F<YY>
update public.portfolio_posicoes
   set "Ativo" = 'DI_F' || substring("Ativo" from 4)
 where "Ativo" ~ '^DI_[0-9]{2}$';

-- DAPYY -> DAP_Q<YY> (par) ou DAP_K<YY> (ímpar)
update public.portfolio_posicoes
   set "Ativo" = case
       when (substring("Ativo" from 4)::int % 2) = 0 then 'DAP_Q' || substring("Ativo" from 4)
       else                                                'DAP_K' || substring("Ativo" from 4)
   end
 where "Ativo" ~ '^DAP[0-9]{2}$';


-- =============================================================
-- posicoes_por_fundo (mesma regra)
-- =============================================================

update public.posicoes_por_fundo
   set "Ativo" = 'DI_F' || substring("Ativo" from 4)
 where "Ativo" ~ '^DI_[0-9]{2}$';

update public.posicoes_por_fundo
   set "Ativo" = case
       when (substring("Ativo" from 4)::int % 2) = 0 then 'DAP_Q' || substring("Ativo" from 4)
       else                                                'DAP_K' || substring("Ativo" from 4)
   end
 where "Ativo" ~ '^DAP[0-9]{2}$';


-- =============================================================
-- Verificação
-- =============================================================
-- Nada abaixo deve retornar linhas se a migração deu certo:
--
-- select "Ativo", count(*) from portfolio_posicoes
-- where "Ativo" ~ '^(DI_[0-9]{2}|DAP[0-9]{2})$'
-- group by "Ativo";
--
-- select "Ativo", count(*) from posicoes_por_fundo
-- where "Ativo" ~ '^(DI_[0-9]{2}|DAP[0-9]{2})$'
-- group by "Ativo";
--
-- Contagem por nova nomenclatura:
-- select "Ativo", count(*) from portfolio_posicoes group by "Ativo" order by "Ativo";


-- =============================================================
-- Rollback (SE precisar reverter — mas backup já foi feito):
--
-- truncate public.portfolio_posicoes;
-- insert into public.portfolio_posicoes select * from public.portfolio_posicoes_backup_pre_naming;
-- truncate public.posicoes_por_fundo;
-- insert into public.posicoes_por_fundo select * from public.posicoes_por_fundo_backup_pre_naming;
-- =============================================================
