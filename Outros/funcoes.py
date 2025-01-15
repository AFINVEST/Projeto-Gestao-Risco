# %% [markdown]
# <h3>Bibliotecas</h3>

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import holidays as hd

# %% [markdown]
# <h3>Buscando dados</h3>

# %%
#import data from excel economatica.xlsx

pd.options.display.float_format = '{:.2f}'.format

df = pd.read_excel('BBG - ECO DASH.xlsx', sheet_name='BZ RATES', skiprows=1 , thousands='.', decimal=',')
df.head()

#drop column unnamed
df.drop('Unnamed: 1', axis=1, inplace=True)
df.drop('Unnamed: 2', axis=1, inplace=True)
df.drop('Unnamed: 3', axis=1, inplace=True)
df.drop('ODF25 Comdty', axis=1, inplace=True)
df.head()
df.rename(columns={'Unnamed: 4':'Date'}, inplace=True)

#remover primeira linha
df = df.drop([0])
df.head()

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.rename(columns={'ODF25 Comdty.1' : 'ODF25 Comdty'}, inplace=True)

df.head()
df.drop('OI1 Comdty', axis=1, inplace=True)
df.drop('WSP1 Index', axis=1, inplace=True)
df.columns = ['Date','DI_25','DI_26', 'DI_27','DI_28','DI_29', 'DI_30', 'DI_31', 'DI_32','DI_33','DI_35','DAP25','DAP26','DAP27','DAP28','DAP30','DAP32','DAP35','DAP40','WDO1','IBOV','TREASURY','S&P']
df.head()


# %% [markdown]
# <h3>Tratando</h3>

# %%
#drop nan values and the values with '-' in the dataframe
df = df.replace('-', np.nan)
df = df.replace(' ', np.nan)
df = df.replace('', np.nan)
#df.dropna(inplace=True)
df.head(10)

# %%
#info of the dataframe
df.info()

# %%
# Transform the columns DI and DAP in float 
df['DI_25'] = df['DI_25'].astype(float)
df['DI_26'] = df['DI_26'].astype(float)
df['DI_27'] = df['DI_27'].astype(float)
df['DI_28'] = df['DI_28'].astype(float)
df['DI_29'] = df['DI_29'].astype(float)
df['DI_30'] = df['DI_30'].astype(float)
df['DI_31'] = df['DI_31'].astype(float)
df['DI_32'] = df['DI_32'].astype(float)
df['DI_33'] = df['DI_33'].astype(float)
df['DI_35'] = df['DI_35'].astype(float)

df['DAP25'] = df['DAP25'].astype(float)
df['DAP26'] = df['DAP26'].astype(float)
df['DAP27'] = df['DAP27'].astype(float)
df['DAP28'] = df['DAP28'].astype(float)
df['DAP30'] = df['DAP30'].astype(float)
df['DAP32'] = df['DAP32'].astype(float)
df['DAP35'] = df['DAP35'].astype(float)
df['DAP40'] = df['DAP40'].astype(float)
df['WDO1'] = df['WDO1'].astype(float)
df['IBOV'] = df['IBOV'].astype(float)
df['TREASURY'] = df['TREASURY'].astype(float)
df['S&P'] = df['S&P'].astype(float)

df.info()

# %%
from datetime import datetime, timedelta

def dias_uteis_ate_ano_referencia(ano_referencia):
    # Obter a data atual
    data_atual = datetime.now()
    
    # Data alvo: Primeiro dia do ano de referência
    primeiro_dia = datetime(ano_referencia, 1, 1)
    
    # Obter feriados do Brasil para o intervalo de datas
    feriados = hd.Brazil(years=range(data_atual.year, ano_referencia + 1))

    #Adicionar feriado do Carnaval e Consciencia Negra
    feriados.append(datetime(data_atual.year, 2, 15))
    feriados.append(datetime(data_atual.year, 11, 20))
    
    # Encontrar o primeiro dia útil no ano de referência
    while primeiro_dia.weekday() >= 5 or primeiro_dia in feriados:  # Fim de semana ou feriado
        primeiro_dia += timedelta(days=1)
    
    # Contar os dias úteis entre a data atual e o primeiro dia útil do ano de referência
    dias_uteis = 0
    while data_atual < primeiro_dia:
        if data_atual.weekday() < 5 and data_atual not in feriados:  # Segunda a sexta e não feriado
            dias_uteis += 1
        data_atual += timedelta(days=1)
    
    return dias_uteis

# Exemplo de uso
print(dias_uteis_ate_ano_referencia(2026))
data_atual = datetime.now()
print(data_atual)
feriados = hd.Brazil(years=range(data_atual.year, 2027))
print(feriados)



# %%
#Calculate the return of the DI_25
df['DI_25_Return'] = df['DI_25'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_25_Return'] = df['DI_25_Return'].replace(0, np.nan)
df['DI_25_Return']

#Calculate the return of the DI_26
df['DI_26_Return'] = df['DI_26'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_26_Return'] = df['DI_26_Return'].replace(0, np.nan)
df['DI_26_Return']

#Calculate the return of the DI_27
df['DI_27_Return'] = df['DI_27'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_27_Return'] = df['DI_27_Return'].replace(0, np.nan)

#Calculate the return of the DI_28
df['DI_28_Return'] = df['DI_28'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_28_Return'] = df['DI_28_Return'].replace(0, np.nan)

#Calculate the return of the DI_29
df['DI_29_Return'] = df['DI_29'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_29_Return'] = df['DI_29_Return'].replace(0, np.nan)

#Calculate the return of the DI_30
df['DI_30_Return'] = df['DI_30'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_30_Return'] = df['DI_30_Return'].replace(0, np.nan)

#Calculate the return of the DI_31
df['DI_31_Return'] = df['DI_31'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_31_Return'] = df['DI_31_Return'].replace(0, np.nan)

#Calculate the return of the DI_32
df['DI_32_Return'] = df['DI_32'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_32_Return'] = df['DI_32_Return'].replace(0, np.nan)

#Calculate the return of the DI_33
df['DI_33_Return'] = df['DI_33'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_33_Return'] = df['DI_33_Return'].replace(0, np.nan)

#Calculate the return of the DI_35
df['DI_35_Return'] = df['DI_35'].pct_change()
#Pegar os zeros e adicionar Nan
df['DI_35_Return'] = df['DI_35_Return'].replace(0, np.nan)

#Calculate the return of the DAP25
df['DAP25_Return'] = df['DAP25'].pct_change()
#Pegar os zeros e adicionar Nan
df['DAP25_Return'] = df['DAP25_Return'].replace(0, np.nan)

#Calculate the return of the DAP26
df['DAP26_Return'] = df['DAP26'].pct_change()
#Pegar os zeros e adicionar Nan
df['DAP26_Return'] = df['DAP26_Return'].replace(0, np.nan)


# %%
#Calculate the return of each column in a new column

df['DAP27_Return'] = df['DAP27'].pct_change()
df['DAP27_Return'] = df['DAP27_Return'].replace(0, np.nan)

df['DAP28_Return'] = df['DAP28'].pct_change()
df['DAP28_Return'] = df['DAP28_Return'].replace(0, np.nan)

df['DAP30_Return'] = df['DAP30'].pct_change()
df['DAP30_Return'] = df['DAP30_Return'].replace(0, np.nan)

df['DAP32_Return'] = df['DAP32'].pct_change()
df['DAP32_Return'] = df['DAP32_Return'].replace(0, np.nan)

df['DAP35_Return'] = df['DAP35'].pct_change()
df['DAP35_Return'] = df['DAP35_Return'].replace(0, np.nan)

df['DAP40_Return'] = df['DAP40'].pct_change()
df['DAP40_Return'] = df['DAP40_Return'].replace(0, np.nan)

df['WDO1_Return'] = df['WDO1'].pct_change()
df['WDO1_Return'] = df['WDO1_Return'].replace(0, np.nan)

df['IBOV_Return'] = df['IBOV'].pct_change()
df['IBOV_Return'] = df['IBOV_Return'].replace(0, np.nan)

df['TREASURY_Return'] = df['TREASURY'].pct_change()
df['TREASURY_Return'] = df['TREASURY_Return'].replace(0, np.nan)

df['S&P_Return'] = df['S&P'].pct_change()
df['S&P_Return'] = df['S&P_Return'].replace(0, np.nan)

df.head()


# %%
#Create a new dataframe with the returns
df_returns = df[['Date','DI_25_Return','DI_26_Return','DI_27_Return','DI_28_Return','DI_29_Return','DI_30_Return','DI_31_Return','DI_32_Return','DI_33_Return','DI_35_Return','DAP25_Return','DAP26_Return','DAP27_Return','DAP28_Return','DAP30_Return','DAP32_Return','DAP35_Return','DAP40_Return','WDO1_Return','IBOV_Return','TREASURY_Return','S&P_Return']]
df_returns.head(10)


# %%
#Calulate the matrix of correlation
correlation_matrix = df_returns.corr()

#plot the heatmap of the correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of the Returns')
plt.show()


# %%
def calcular_retornos_log(df):
    # Cria um novo DataFrame para armazenar os retornos logarítmicos
    df_returns = pd.DataFrame()

    # Itera sobre as colunas do DataFrame
    for col in df.columns:
        # Verifica se a coluna é do tipo float
        if pd.api.types.is_float_dtype(df[col]):
            # Calcula o retorno logarítmico e adiciona no DataFrame de retornos
            df_returns[f'{col}_Return_Log'] = np.log(df[col] / df[col].shift(1))

    return df_returns

# %%
df_retorno = calcular_retornos_log(df)
df_retorno.tail()

# %%
#Calculate the return log of all the columns in the dataframe
df_returns['DI_25_Return_Log'] = np.log(df['DI_25']/df['DI_25'].shift(1))
df_returns['DI_26_Return_Log'] = np.log(df['DI_26']/df['DI_26'].shift(1))
df_returns['DI_27_Return_Log'] = np.log(df['DI_27']/df['DI_27'].shift(1))
df_returns['DI_28_Return_Log'] = np.log(df['DI_28']/df['DI_28'].shift(1))
df_returns['DI_29_Return_Log'] = np.log(df['DI_29']/df['DI_29'].shift(1))
df_returns['DI_30_Return_Log'] = np.log(df['DI_30']/df['DI_30'].shift(1))
df_returns['DI_31_Return_Log'] = np.log(df['DI_31']/df['DI_31'].shift(1))
df_returns['DI_32_Return_Log'] = np.log(df['DI_32']/df['DI_32'].shift(1))
df_returns['DI_33_Return_Log'] = np.log(df['DI_33']/df['DI_33'].shift(1))
df_returns['DI_35_Return_Log'] = np.log(df['DI_35']/df['DI_35'].shift(1))

df_returns['DAP25_Return_Log'] = np.log(df['DAP25']/df['DAP25'].shift(1))
df_returns['DAP26_Return_Log'] = np.log(df['DAP26']/df['DAP26'].shift(1))
df_returns['DAP27_Return_Log'] = np.log(df['DAP27']/df['DAP27'].shift(1))
df_returns['DAP28_Return_Log'] = np.log(df['DAP28']/df['DAP28'].shift(1))
df_returns['DAP30_Return_Log'] = np.log(df['DAP30']/df['DAP30'].shift(1))
df_returns['DAP32_Return_Log'] = np.log(df['DAP32']/df['DAP32'].shift(1))
df_returns['DAP35_Return_Log'] = np.log(df['DAP35']/df['DAP35'].shift(1))
df_returns['DAP40_Return_Log'] = np.log(df['DAP40']/df['DAP40'].shift(1))

df_returns['WDO1_Return_Log'] = np.log(df['WDO1']/df['WDO1'].shift(1))
df_returns['IBOV_Return_Log'] = np.log(df['IBOV']/df['IBOV'].shift(1))
df_returns['TREASURY_Return_Log'] = np.log(df['TREASURY']/df['TREASURY'].shift(1))
df_returns['S&P_Return_Log'] = np.log(df['S&P']/df['S&P'].shift(1))


df_returns['DI_25_Return_Log'] = np.log(df['DI_25']/df['DI_25'].shift(1))
df_returns['DI_26_Return_Log'] = np.log(df['DI_26']/df['DI_26'].shift(1))
df_returns['DI_27_Return_Log'] = np.log(df['DI_27']/df['DI_27'].shift(1))
df_returns['DI_28_Return_Log'] = np.log(df['DI_28']/df['DI_28'].shift(1))
df_returns['DI_29_Return_Log'] = np.log(df['DI_29']/df['DI_29'].shift(1))
df_returns['DI_30_Return_Log'] = np.log(df['DI_30']/df['DI_30'].shift(1))
df_returns['DI_31_Return_Log'] = np.log(df['DI_31']/df['DI_31'].shift(1))
df_returns['DI_32_Return_Log'] = np.log(df['DI_32']/df['DI_32'].shift(1))
df_returns['DI_33_Return_Log'] = np.log(df['DI_33']/df['DI_33'].shift(1))
df_returns['DI_35_Return_Log'] = np.log(df['DI_35']/df['DI_35'].shift(1))

df_returns['DAP25_Return_Log'] = np.log(df['DAP25']/df['DAP25'].shift(1))
df_returns['DAP26_Return_Log'] = np.log(df['DAP26']/df['DAP26'].shift(1))
df_returns['DAP27_Return_Log'] = np.log(df['DAP27']/df['DAP27'].shift(1))
df_returns['DAP28_Return_Log'] = np.log(df['DAP28']/df['DAP28'].shift(1))
df_returns['DAP30_Return_Log'] = np.log(df['DAP30']/df['DAP30'].shift(1))
df_returns['DAP32_Return_Log'] = np.log(df['DAP32']/df['DAP32'].shift(1))
df_returns['DAP35_Return_Log'] = np.log(df['DAP35']/df['DAP35'].shift(1))
df_returns['DAP40_Return_Log'] = np.log(df['DAP40']/df['DAP40'].shift(1))

df_returns['WDO1_Return_Log'] = np.log(df['WDO1']/df['WDO1'].shift(1))
df_returns['IBOV_Return_Log'] = np.log(df['IBOV']/df['IBOV'].shift(1))
df_returns['TREASURY_Return_Log'] = np.log(df['TREASURY']/df['TREASURY'].shift(1))
df_returns['S&P_Return_Log'] = np.log(df['S&P']/df['S&P'].shift(1))
# Remover as colunas especificadas do DataFrame
df_returns = df_returns.drop(
    ['DI_25_Return', 'DI_26_Return', 'DI_27_Return', 'DI_28_Return', 'DI_29_Return',
     'DI_30_Return', 'DI_31_Return', 'DI_32_Return', 'DI_33_Return', 'DI_35_Return',
     'DAP25_Return', 'DAP26_Return', 'DAP27_Return', 'DAP28_Return', 'DAP30_Return',
     'DAP32_Return', 'DAP35_Return', 'DAP40_Return', 'WDO1_Return', 'IBOV_Return',
     'TREASURY_Return', 'S&P_Return'],
    axis=1,   # Remove colunas (axis=1)
    errors='ignore'  # Evita erro se alguma coluna não existir no DataFrame
)
df_returns.tail(10)

# %% [markdown]
# <H1>Calculo da Matrix de Cor</H1>

# %%
#PLOT THE MATRIX OF CORRELATION OF THE RETURNS LOG WITHOUT DATE

plt.figure(figsize=(10,10))
sns.heatmap(df_returns.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of the Returns Log')
plt.show()

# %%
#PLOT THE MATRIX OF THE DI
df_DI = df_returns[['DI_25_Return_Log','DI_26_Return_Log','DI_27_Return_Log','DI_28_Return_Log','DI_29_Return_Log','DI_30_Return_Log','DI_31_Return_Log','DI_32_Return_Log','DI_33_Return_Log','DI_35_Return_Log']]
plt.figure(figsize=(10,10))
sns.heatmap(df_DI.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of the DI Returns Log')
plt.show()

#PLOT THE MATRIX OF THE DAP
df_DAP = df_returns[['DAP25_Return_Log','DAP26_Return_Log','DAP27_Return_Log','DAP28_Return_Log','DAP30_Return_Log','DAP32_Return_Log','DAP35_Return_Log','DAP40_Return_Log']]
plt.figure(figsize=(10,10))
sns.heatmap(df_DAP.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of the DAP Returns Log')
plt.show()

#PLOT THE MATRIX OF THE OTHERS
df_OTHERS = df_returns[['WDO1_Return_Log','IBOV_Return_Log','TREASURY_Return_Log','S&P_Return_Log']]
plt.figure(figsize=(10,10))
sns.heatmap(df_OTHERS.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of the Others Returns Log')
plt.show()

# %%
#PLOT THE HISTOGRAM OF THE RETURNS
plt.figure(figsize=(10,10))
df_returns.hist(bins=50, figsize=(20,15))
plt.suptitle('Histogram of the Returns')
plt.show()



# %%
df_volatilidade_DI25 = df_returns['DI_25_Return_Log'].std()

# Calcular a volatilidade anualizada (volatilidade * sqrt(252))
volatilidade_anualizada = df_volatilidade_DI25 * np.sqrt(252)
print(f'Volatilidade Anualizada: {volatilidade_anualizada:.2%}')

#Fazer o mesmo para os outros ativos
df_volatilidade_DI25 = df_returns['DI_25_Return_Log'].std()
df_volatilidade_DI26 = df_returns['DI_26_Return_Log'].std()
df_volatilidade_DI27 = df_returns['DI_27_Return_Log'].std()
df_volatilidade_DI28 = df_returns['DI_28_Return_Log'].std()
df_volatilidade_DI29 = df_returns['DI_29_Return_Log'].std()
df_volatilidade_DI30 = df_returns['DI_30_Return_Log'].std()
df_volatilidade_DI31 = df_returns['DI_31_Return_Log'].std()
df_volatilidade_DI32 = df_returns['DI_32_Return_Log'].std()
df_volatilidade_DI33 = df_returns['DI_33_Return_Log'].std()
df_volatilidade_DI35 = df_returns['DI_35_Return_Log'].std()
df_volatilidade_DAP25 = df_returns['DAP25_Return_Log'].std()
df_volatilidade_DAP26 = df_returns['DAP26_Return_Log'].std()
df_volatilidade_DAP27 = df_returns['DAP27_Return_Log'].std()
df_volatilidade_DAP28 = df_returns['DAP28_Return_Log'].std()
df_volatilidade_DAP30 = df_returns['DAP30_Return_Log'].std()
df_volatilidade_DAP32 = df_returns['DAP32_Return_Log'].std()
df_volatilidade_DAP35 = df_returns['DAP35_Return_Log'].std()
df_volatilidade_DAP40 = df_returns['DAP40_Return_Log'].std()
df_volatilidade_WDO1 = df_returns['WDO1_Return_Log'].std()
df_volatilidade_IBOV = df_returns['IBOV_Return_Log'].std()
df_volatilidade_TREASURY = df_returns['TREASURY_Return_Log'].std()
df_volatilidade_SP = df_returns['S&P_Return_Log'].std()

volatilidade_anualizada_DI25 = df_volatilidade_DI25 * np.sqrt(252)
volatilidade_anualizada_DI26 = df_volatilidade_DI26 * np.sqrt(252)
volatilidade_anualizada_DI27 = df_volatilidade_DI27 * np.sqrt(252)
volatilidade_anualizada_DI28 = df_volatilidade_DI28 * np.sqrt(252)
volatilidade_anualizada_DI29 = df_volatilidade_DI29 * np.sqrt(252)
volatilidade_anualizada_DI30 = df_volatilidade_DI30 * np.sqrt(252)
volatilidade_anualizada_DI31 = df_volatilidade_DI31 * np.sqrt(252)
volatilidade_anualizada_DI32 = df_volatilidade_DI32 * np.sqrt(252)
volatilidade_anualizada_DI33 = df_volatilidade_DI33 * np.sqrt(252)
volatilidade_anualizada_DI35 = df_volatilidade_DI35 * np.sqrt(252)
volatilidade_anualizada_DAP25 = df_volatilidade_DAP25 * np.sqrt(252)
volatilidade_anualizada_DAP26 = df_volatilidade_DAP26 * np.sqrt(252)
volatilidade_anualizada_DAP27 = df_volatilidade_DAP27 * np.sqrt(252)
volatilidade_anualizada_DAP28 = df_volatilidade_DAP28 * np.sqrt(252)
volatilidade_anualizada_DAP30 = df_volatilidade_DAP30 * np.sqrt(252)
volatilidade_anualizada_DAP32 = df_volatilidade_DAP32 * np.sqrt(252)
volatilidade_anualizada_DAP35 = df_volatilidade_DAP35 * np.sqrt(252)
volatilidade_anualizada_DAP40 = df_volatilidade_DAP40 * np.sqrt(252)
volatilidade_anualizada_WDO1 = df_volatilidade_WDO1 * np.sqrt(252)
volatilidade_anualizada_IBOV = df_volatilidade_IBOV * np.sqrt(252)
volatilidade_anualizada_TREASURY = df_volatilidade_TREASURY * np.sqrt(252)
volatilidade_anualizada_SP = df_volatilidade_SP * np.sqrt(252)


Volatilidade = {'DI_25': [volatilidade_anualizada] , 'DI_26' : [volatilidade_anualizada_DI26], 'DI_27' : [volatilidade_anualizada_DI27], 'DI_28' : [volatilidade_anualizada_DI28], 'DI_29' : [volatilidade_anualizada_DI29], 'DI_30' : [volatilidade_anualizada_DI30], 'DI_31' : [volatilidade_anualizada_DI31], 'DI_32' : [volatilidade_anualizada_DI32], 'DI_33' : [volatilidade_anualizada_DI33], 'DI_35' : [volatilidade_anualizada_DI35], 'DAP25' : [volatilidade_anualizada_DAP25], 'DAP26' : [volatilidade_anualizada_DAP26], 'DAP27' : [volatilidade_anualizada_DAP27], 'DAP28' : [volatilidade_anualizada_DAP28], 'DAP30' : [volatilidade_anualizada_DAP30], 'DAP32' : [volatilidade_anualizada_DAP32], 'DAP35' : [volatilidade_anualizada_DAP35], 'DAP40' : [volatilidade_anualizada_DAP40], 'WDO1' : [volatilidade_anualizada_WDO1], 'IBOV' : [volatilidade_anualizada_IBOV], 'TREASURY' : [volatilidade_anualizada_TREASURY], 'S&P' : [volatilidade_anualizada_SP]}
df_volatilidade = pd.DataFrame(Volatilidade)
df_volatilidade


# %%
#Matrix de Covariancia dos retornos
df_returns.drop('Date', axis=1, inplace=True)
covariance_matrix = df_returns.cov()
covariance_matrix

# %%
#Plot the histogram of the returns of the DI_25
plt.figure(figsize=(10,6))
plt.hist(df_returns['DI_25_Return_Log'].dropna(), bins=100, color='blue')
plt.title('Histogram of the Returns of DI_25')
plt.show()


# %%
#Risco de Mercado - VAR - Value at Risk not parametric - Historical Simulation

def var_not_parametric(df_returns, alpha=0.05):
    # Calcular o VAR
    var = df_returns.quantile(alpha)
    return var


#Calcular o VAR para os outros ativos
for col in df_returns.columns:
    if pd.api.types.is_float_dtype(df_returns[col]):
        var_95 = var_not_parametric(df_returns[col], alpha=0.05)
        print(f'VAR 95% - {col}: {var_95:.2%}')



# %%
#Calculate the CVAR
def cvar_not_parametric(df_returns, alpha=0.05):
    # Calcular o VAR
    var = df_returns.quantile(alpha)
    
    # Filtrar os retornos menores que o VAR
    returns_below_var = df_returns[df_returns < var]

    # Calcular o CVAR
    cvar = returns_below_var.mean()
    return cvar

#Calcular o CVAR para os outros ativos
for col in df_returns.columns:
    if pd.api.types.is_float_dtype(df_returns[col]):
        cvar_95 = cvar_not_parametric(df_returns[col], alpha=0.05)
        print(f'CVAR 95% - {col}: {cvar_95:.2%}')





