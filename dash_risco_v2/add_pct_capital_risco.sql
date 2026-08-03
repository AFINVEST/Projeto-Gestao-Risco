-- Adiciona parâmetro pct_capital_risco em config_risco
-- (fatia do PL total que é investida em estratégia de risco, mesmo default do app4 sidebar = 1%)

insert into public.config_risco ("parametro", "valor", "descricao") values
    ('pct_capital_risco', '0.01',
     'Fatia do PL total investida em risco (1%% default = mesmo pct do simulador de cota do app4). '
     'Retorno diário = PnL / (pct × PL). Vol/VaR/DD são calculados sobre essa cota sintética.')
on conflict ("parametro") do nothing;

-- Verificação
select * from config_risco where parametro = 'pct_capital_risco';
