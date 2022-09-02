import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import snowflake.connector
from statsmodels.tsa.filters.hp_filter import hpfilter
import warnings

warnings.filterwarnings("ignore")

#### Detrending data using the HP Filter----------------------------------------------------------------------------
df = pd.read_excel(r'C:\Users\DFerra538\Downloads\India_Exchange_Rate_Dataset.xls')
EXINUS_cycle,EXINUS_trend = hpfilter(df['EXINUS'], lamb=1600) #What is this lamb
EXINUS_trend.plot(figsize=(15,6)).autoscale(axis='x',tight=True)
EXINUS_cycle.plot(figsize=(15,6)).autoscale(axis='x',tight=True)

df['detrended'] = df.EXINUS - EXINUS_trend
plt.plot(df['detrended'])
plt.show()
#-------------------------------------------------------------------------------------------------------------------

### Detrending using Pandas Differencing----------------------------------------------------------------------------
df = pd.read_excel(r'C:\Users\DFerra538\Downloads\India_Exchange_Rate_Dataset.xls')
diff = df.EXINUS.diff()  #diff(is the built in function)

plt.plot(diff)
plt.show()

#-------------------------------------------------------------------------------------------------------------------


### Detrending Using a SciPy Signal---------------------------------------------------------------------------------
df = pd.read_excel(r'C:\Users\DFerra538\Downloads\India_Exchange_Rate_Dataset.xls')
detrended = signal.detrend(df.EXINUS.values)

plt.plot(detrended)
plt.show()

#-------------------------------------------------------------------------------------------------------------------

### Seasonality = Seasonality is a periodical fluctuation where the same pattern occurs at a regular interval of time.

import seaborn as sns
df['month'] = df['observation_date'].dt.strftime('%b') #Converting date into month values
df['year'] = [d.year for d in df.observation_date] #Another version
df['month'] = [d.strftime('%b') for d in df.observation_date] #Another version
years = df['year'].unique()

plt.figure(figsize=(15,6))
sns.boxplot(x='month', y='EXINUS', data=df).set_
title("Multi Month-wise Box Plot")
plt.show()

### Auto Correlation-------------------------------------------------------------------------------------------------
#Autocorrelation is used to check randomness in data. It helps to identify types of data where the period is not known.
from pandas.plotting import autocorrelation_plot

df = pd.read_excel(r'C:\Users\DFerra538\Downloads\India_Exchange_Rate_Dataset.xls')
plt.rcParams.update({'figure.figsize':(15,6), 'figure.dpi':220}) #runtime configuration parameters
autocorrelation_plot(df['EXINUS'].tolist())

#Note Sometimes identifying seasonality is not easy; in that case, we need to evaluate other plots such as sequence or seasonal subseries plot

#--------------------------------------------------------------------------------------------------------------------

###Deseasoning of Time-Series Data ----------------------------------------------------------------------------------

#Deseasoning means to remove seasonality from time-series data. It is stripped of the pattern of seasonal effect to deseasonalize the impact.
        #Level
        #Trend
        #Seasonality
        #Noise

#Note An additive model is when time-series data combines these four components for linear trend and seasonality, and a multiplicative
             #model is when components are multiplied to gather for nonlinear
                      #trends and seasonality.


### Seasonal Decomposition ------------------------------------------------------------------------------------------
#Decomposition is the process of understanding generalizations and problems related to time-series forecasting.


from statsmodels.tsa.seasonal import seasonal_decompose
df = pd.read_excel(r'C:\Users\DFerra538\Downloads\India_Exchange_Rate_Dataset.xls', index_col =0, parse_dates =True)
result_mul = seasonal_decompose(df['EXINUS'], model = 'multiplicative', extrapolate_trend ='freq')
deseason = df['EXINUS'] - result_mul.seasonal


plt.plot(deseason)
plt.show()

#---------------------------------------------------------------------------------------------------------------------

###Cyclic Variations -------------------------------------------------------------------------------------------------
# Cyclical componentes are flucuations around a long trend observed every units of time; this behavior is less frquented compared to seasonality.
   #Prosperity
   #Depression
   #Accesibility

###Detecting Cyclical Variations------------------------------------------------------------------------------------





