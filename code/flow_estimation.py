#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Load Data
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

Estimate missing flow data in Scott Creek from USGS regional gages


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

folder = '../data/'  # Data folder

### Load Flow Datasets

## USGS regional stations
# Pescadero Creek
flowp = pd.read_csv(os.path.join(folder,'flow','pescadero_creek_usgs_flow_20100101_20201231.csv'), parse_dates=['date'], index_col=['date'])
flowp['stage'] = np.sqrt((flowp['flow'] / 19.7514))  
flowp.columns = [c+'_p' for c in flowp.columns]
## from quadratic regression on Pescadero stage-discharge rating (Rsq = .994, underest low flow)

# # Soquel Creek
# flows = pd.read_csv(os.path.join(folder,'flow','soquel_creek_usgs_flow_20100101_20201231.csv'), parse_dates=['date'], index_col=['date'])
# flows['stage'] = np.sqrt((flows['flow'] / 25.5105))  
# flows.columns = [c+'_s' for c in flows.columns]
## from quadratic regression on Soquel stage-discharge rating (Rsq = .995, underest low flow)


## Scott Creek (from NOAA)
# Flow from stage:
# IF Stage <1.75, Q = 90.756*Stage^2 - 181.68*Stage + 93.047
# Else: Q = 30.72*Stage^2 + 82.575*Stage-186.57
flow = pd.read_csv(os.path.join(folder,'flow','flow_NOAA_raw.csv'), parse_dates=['dt'], index_col=['dt'])
flow = flow.resample('D').mean()  # turn to daily dataset
flow['logflow'] = np.log10(flow.flow)

print('Missing Days of Data:')
print(flow.isna().sum())

#%% Regression to fill missing days

### Regress on regional flow data and time of year data
df = pd.concat([flow,flowp],axis=1)  # met['rain','rain30T','dry','dry_days']]
df['month'] = df.index.month
df = pd.concat([df,pd.get_dummies(df.month,prefix='month')],axis=1)  # add month categorical vars

#missing = df['2020':]  # to backfill
missing = df[df.logflow.isna()]  
reg = df['2010':'2019'].dropna() # to regress

## logflow model
#lm = sm.OLS(df.logflow, df[['logflow_p','logflow_s']]).fit()
reg_vars = ['logflow_p', 
            'month_1', 'month_2', 'month_3', 'month_4',
            'month_5', 'month_6', 'month_7', 'month_8', 
            'month_9', 'month_10','month_11', 'month_12']
lm = sm.OLS(reg.logflow, reg[reg_vars]).fit()

print(lm.summary())

## Rsq = .700, RMSE = 0.31 log10(flow), AIC = 1717 (lowest)

## Notes: multicolinnearity when both stations used
## Day of year/month continuous variables made good models by Rsq
## Categorical month variables made the best models by RMSE and AIC


### Backfill flow data
backfill = lm.predict(missing[reg_vars])
idx = flow[flow.logflow.isna()].index  # Missing Days index
flow['regressed'] = 0  # indicate which had filled in data
flow.loc[idx,'regressed'] = 1

flow.loc[idx,'logflow'] = backfill[idx]
flow.loc[idx, 'flow'] = 10**flow.loc[idx,'logflow']


### Stage from Flow:
# IF Stage <1.75, Q = 90.756*Stage^2 - 181.68*Stage + 93.047
# Else: Q = 30.72*Stage^2 + 82.575*Stage-186.57
# Split point Q is about 52.5 ft3/s
min_stage = flow.stage.min()

def quad_eq(y,a,b,c):
    d = c-y
    x1 = (-b + (b*b - 4*a*d)**.5)/(2*a)
    x2 = (-b - (b*b - 4*a*d)**.5)/(2*a)
    return max([x1, x2])

temp = []
for i in idx:
    y = flow.loc[i, 'flow']
    if y > 52.5:
        a = 30.72
        b = 82.575
        c = -186.57
    else:
        a = 90.756
        b = -181.68
        c = 93.047
        
    x = quad_eq(y,a,b,c)
    temp += [x]

flow.loc[idx, 'stage'] = temp
flow.loc[flow.stage.isna(), 'stage'] = min_stage # fill in remaining missing with minimum stage 

## Plot
# plt.figure()
# plt.plot(df.resample('D').first().logflow)
# plt.plot(lm.predict(test[reg_vars]).resample('D').first())

#%% Save new data
flow.index.name = 'date'
flow.to_csv(os.path.join(folder,'flow','scott_creek_daily_flow.csv'))


