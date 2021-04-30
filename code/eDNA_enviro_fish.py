#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Load Data / Create Variables
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

Compare eDNA data to environmental and fishcount data


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats, signal
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.tsatools import detrend
import os
import datetime
from eDNA_corr import eDNA_corr

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 8)
# pd.set_option('display.width', 175)


folder = '../data/'  # Data folder

### eDNA Data
# Contains target, replicate #,dilution level, concentration (post-dilution factor)
eDNA = pd.read_csv(os.path.join(folder,'eDNA','eDNA.csv'), parse_dates=['dt','date'])

dr = pd.date_range(min(eDNA.date),max(eDNA.date))  # project date range

### Set BLOD to 0 for stats?
# Note: setting to NAN biases the data (excludes many samples where we know conc < lod)
eDNA.loc[eDNA.BLOD == 1,'eDNA'] = 0
eDNA.loc[eDNA.BLOD == 1,'log10eDNA'] = 0 

## Separate targets
trout = eDNA[eDNA.target=='trout'].sort_values('dt') 
coho = eDNA[eDNA.target=='coho'].sort_values('dt')



### ESP logs
# Contains sampling times, volumes, ESP name
ESP = pd.read_csv(os.path.join(folder,'ESP_logs','ESP_logs_combined.csv'), 
                  parse_dates = ['sample_wake','sample_start','sample_mid','sample_end','date'], 
                  index_col=['sample_mid'])  # can also use sample_start, but probably little diff.

# ESP.dropna(inplace=True, subset=['sample_wake', 'sample_start', 'sample_end', 'sample_duration',
#        'vol_target', 'vol_actual', 'vol_diff',])  # Drop samples with no time or volume data 

ESP = ESP[~ESP.lab_field.isin(['lab','control', 'control '])]  # Keep deployed/field/nana; Drop control/lab samples
ESP.drop(ESP[ESP.id.isin(['SCr-181', 'SCr-286', 'SCr-479', 'SCr-549'])].index, inplace=True) # Drop duplicate (no qPCR IDs)

## Lists of days with sampling frequencies of 1x, 2x, and 3x per day
three_per_day = [d for d in ESP['date'].unique() if (ESP.groupby('date').count() == 3)['id'].loc[d]]
# morning, afternoon, and evening samples
mae = [d for d in three_per_day if (ESP.groupby('date').sum()['morn_midday_eve']==3).loc[d]] 
two_per_day = [d for d in ESP['date'].unique() if (ESP.groupby('date').count() == 2)['id'].loc[d]]
one_per_day = [d for d in ESP['date'].unique() if (ESP.groupby('date').count() == 1)['id'].loc[d]]


###  Hatchery Data
# For plots
# Adult/Juvenile Steelhead and Coho counts
hatch = pd.read_csv(os.path.join(folder,'NOAA_data', 'hatchery_releases.csv'), 
                 parse_dates = ['date'], index_col=['date'])

hatch_vars = pd.DataFrame(index=pd.date_range(hatch.index[0], hatch.index[-1]))

## Hatch variables
hatch_vars['release'] = 0 # Release binary
hatch_vars.loc[hatch.index, 'release'] = 1

hatch_vars['tr'] = 0  # Total releases in watershed (0-3d, 7d lag)
hatch_vars.loc[hatch.index, 'tr'] = hatch.sum(axis=1)

for i in [1,2,3]:  # hatchery release i days before
    hatch_vars['release'+str(i)] = hatch_vars['release'].shift(i)
    hatch_vars['tr'+str(i)] = hatch_vars['tr'].shift(1)
    
for i in [2,3,7,14,30]:  # hatchery release totals over X days binary
    hatch_vars['release'+str(i) + 'T'] = 0
    hatch_vars['tr'+ str(i) + 'T'] = 0
    for d in range(1,i+1):
        hatch_vars['release'+ str(i) + 'T'] += hatch_vars['release'].shift(d)
        hatch_vars['tr'+ str(i) + 'T'] += hatch_vars['tr'].shift(d)

# Near weir
hatch_vars['S0'] = 0  
hatch_vars.loc[hatch.index, 'S0'] = hatch['S0'].fillna(0)


### Trap Data (Fish Counts/Water Temp)
## Fish counts
fish = pd.read_csv(os.path.join(folder,'NOAA_data', 'fish_vars.csv'), 
                 parse_dates = ['date'], index_col=['date'])

fish['logbiomass'] = np.log10(fish['biomass_total'])


## Water temperature at the weir
wtemp = pd.read_csv(os.path.join(folder,'NOAA_data', 'weir_wtemp.csv'), 
                    parse_dates = ['dt'], index_col=['dt'])
wtemp = wtemp.resample('D').mean()

## Lagoon WQ
lagoon = pd.read_csv(os.path.join(folder,'NOAA_data', 'lagoon_wq.csv'), parse_dates = ['dt'], index_col=['dt'])
lagoon = lagoon.resample('D').mean()
# wtemp in lagoon and weir - PCC of 0.86


### Flow Data
## Scott Creek (from NOAA), daily TS (with regressed missing data)
flow = pd.read_csv(os.path.join(folder,'flow','scott_creek_daily_flow.csv'), parse_dates=['date'], index_col=['date'])

flow['quartile'] = 4  # Indicate which quantile of the record the flow data are in
flow.loc[flow.logflow < flow.logflow.quantile(.75),'quartile'] = 3
flow.loc[flow.logflow < flow.logflow.quantile(.5),'quartile'] = 2
flow.loc[flow.logflow < flow.logflow.quantile(.25),'quartile'] = 1

for i in [1,2,3]:  # previous days flow
    flow['logflow' + str(i)] = flow.logflow.shift(i)


###  Met/Rain/Rad
## Daily Rain/Temp Data
met = pd.read_csv(os.path.join(folder,'met','LD_swanton_daily_met.csv'), parse_dates=['date'], index_col=['date'])
met = met.dropna(how='all')

## Rain vars
for i in [1,2,3]:  # previous days rainfall
    met['rain' + str(i)] = met.rain.shift(i)
    
for i in [2,3,7,14,30]:  # rainfall total over X days
    met['rain'+str(i) + 'T'] = 0
    for d in range(1,i+1):
        met['rain'+str(i) + 'T'] += met['rain'].shift(d)

met['dry'] = 1  # Dry day (no rain)
met.loc[met.rain > 0, 'dry'] = 0

met['dry_days'] = 0  # consecutive days since significant rain
met.loc[(met.rain < 0.05), 'dry_days'] = np.nan
met['dry_days'] = met.dry_days.ffill() + met.groupby(met.dry_days.notnull().cumsum()).cumcount()

## Hourly Air Temp Data
met_h = pd.read_csv(os.path.join(folder,'met','LD_swanton_hourly_airtemp.csv'), parse_dates=['dt'], index_col=['dt'])
met_h = met_h.dropna(how='all')
#met_h.resample('D').mean().plot()

## Rad data (Pescadero Station, with other met)
rad = pd.read_csv(os.path.join(folder,'met','Pescadero_CIMIS_day_met_20190101_20200501.csv'), parse_dates=['date'], index_col=['date'])


### Moon Phases / Photo Period
moon = pd.read_csv(os.path.join(folder,'moon_phases_2019_2020.csv'), parse_dates=['date'], index_col=['date'])


#%% Plot parameters / functions
params = {
   'axes.labelsize': 11,
   'font.size': 11,
   'legend.fontsize': 10.5,
   'xtick.labelsize': 10,
   'ytick.labelsize': 11,
   'font.family'  : 'sans-serif',
   'font.sans-serif':'Helvetica',
   'axes.axisbelow': True
   }
plt.rcParams.update(params)


### Colors
#pal = ['#969696','#525252']  # grey, black
#pal = ['#ca0020', '#f4a582'] # salmon colors
#pal = ['#253494','#41b6c4'] # light, dark blue
#pal = ['#39778d', '#de787c'] # 
pal = ['#de425b','#2c8380']
#pal = sns.color_palette(pal)

pal4c = ['#253494','#2c7fb8','#41b6c4','#a1dab4'] # 4 color blue tone

# other pinks: '#de787c'
# other blue/green/silver: 

### Functions
def caps_off(axx): ## Turn off caps on boxplots
    lines = axx.lines
    for i in range(0, int(len(lines)/6)):
        lines[(i*6)+2].set_color('none')
        lines[(i*6)+3].set_color('none')

def flier_shape(axx, shape='.'):  ## Set flier shape on boxplots
    lines = axx.lines
    for i in range(0, int(len(lines)/6)):
        lines[(i*6)+5].set_marker(shape)

def plot_spines(axx, offset=8): # Offset position, Hide the right and top spines
    axx.spines['left'].set_position(('outward', offset))
    axx.spines['bottom'].set_position(('outward', offset))
    
    axx.spines['right'].set_visible(False)
    axx.spines['top'].set_visible(False)


#%% Environmental Data Time Series (temp, flow, rain)

### Flow / Rain
plt.figure(figsize=(10,5))

plt.subplot(2,1,1)
ax1 = plt.gca()
plt.plot(flow.index, flow.logflow, lw=1.75, color='#00264d', label='Discharge', zorder=2)  ## Scott
plt.bar([],[],width=1, color = '#1a8cff', linewidth=0, alpha=.7, label='Rain')
#plt.ylabel('Stream Discharge (log$_{10}$ft$^3$/s)')
plt.ylabel('Stream Stage (ft)')
plt.ylim(1,4.5)
#plt.yscale('log')
#plt.ylim(1,300)
plt.legend(frameon=False, loc='upper right', ncol=2)

ax2 = plt.twinx(plt.gca())
plt.bar(met.index, met.rain * 25.4, width=1, color = '#1a8cff', ec='#1a8cff', alpha=.7, label='Rain', zorder=1)
plt.ylabel('Total Rainfall (mm)', rotation=270, ha='center', va='baseline', rotation_mode='anchor')
plt.ylim(0,45)

plt.xlim(dr[0],dr[-1])
offset = 7
for a in [ax1,ax2]:
    a.spines['left'].set_position(('outward', offset))
    a.spines['right'].set_position(('outward', offset))
    a.spines['top'].set_visible(False)
    
    a.set_xticklabels([])

plt.tight_layout()


### Air/Water Temp
#plt.figure(figsize=(10,4))
plt.subplot(2,1,2)

ax1 = plt.gca()
wtemp_plot = wtemp.resample('D').mean()
plt.plot(wtemp_plot.index, wtemp_plot.wtemp, lw=1.75, color='#004080', label='Water', zorder=2)
#plt.plot([], lw=1.5, ls='--', color = '#ff6666', label='Air')
plt.plot(met.index, (5/9)*(met.temp - 32), lw=1.5, ls='--', color = '#ff6666', label='Air', zorder=1)
plt.ylabel('Temperature (°C)')
plt.ylim(3,30)
plt.legend(frameon=False, ncol=2)

ax2 = plt.twinx(plt.gca())
#plt.ylabel('Air Temperature (°C)', rotation=270, ha='center', va='baseline', rotation_mode='anchor')
plt.ylim(3,30)
ax2.set_yticklabels([])

plt.xlim(dr[0],dr[-1])
offset = 7
for a in [ax1, ax2]:
    a.spines['left'].set_position(('outward', offset))
    a.spines['right'].set_position(('outward', offset))
    a.spines['top'].set_visible(False)
    #a.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(folder.replace('data','figures'),'temp_flow_rain_TS.png'),dpi=300)

#%% Lagoon WQ Time Series

lagoon_plot = lagoon.resample('D').mean()

plt.figure(figsize=(10,5))

## Wtemp / pH
plt.subplot(2,1,1)
ax1 = plt.gca()
plt.plot(lagoon_plot.index, lagoon_plot.wtemp, lw=1.75, color='#00264d', label='Water Temp.', zorder=2)
plt.plot([], lw=1.5, ls='--', color = '#669900', label='pH')
plt.ylabel('Water Temperature (°C)')
#plt.yscale('log')
#plt.ylim(1,300)
plt.legend(frameon=False, loc='upper left', ncol=2)

ax2 = plt.twinx(plt.gca())
plt.plot(lagoon_plot.index, lagoon_plot.pH, lw=1.5, ls='--', color = '#669900', label='pH', zorder=1)
plt.ylabel('pH', rotation=270, ha='center', va='baseline', rotation_mode='anchor')
#plt.ylim(0,45)

plt.xlim(dr[0],dr[-1])
offset = 7
for a in [ax1,ax2]:
    a.spines['left'].set_position(('outward', offset))
    a.spines['right'].set_position(('outward', offset))
    a.spines['top'].set_visible(False)
    
    a.set_xticklabels([])

plt.tight_layout()

## Turbidity / DO
plt.subplot(2,1,2)
ax1 = plt.gca()
plt.plot(lagoon_plot.index, lagoon_plot.turb, lw=1.75, color='#ff6600', label='Turbidity', zorder=2)
plt.plot([], lw=1.5, ls='--', color = '#6666ff', label='DO')
plt.ylabel('Turbidity (NTU)')
plt.yscale('log')
#plt.ylim(1,300)
plt.legend(frameon=False, loc='upper left', ncol=2)

ax2 = plt.twinx(plt.gca())
plt.plot(lagoon_plot.index, lagoon_plot.DO, lw=1.5, ls='--', color = '#6666ff', label='DO', zorder=1)
plt.ylabel('Dissolved Oxygem (mg/l)', rotation=270, ha='center', va='baseline', rotation_mode='anchor')
#plt.ylim(0,45)

plt.xlim(dr[0],dr[-1])
offset = 7
for a in [ax1,ax2]:
    a.spines['left'].set_position(('outward', offset))
    a.spines['right'].set_position(('outward', offset))
    a.spines['top'].set_visible(False)

plt.tight_layout()

plt.suptitle('Lagoon Water Quality')


#%% Regression on eDNA
#y = trout.set_index('date').log10eDNA  # all data
dep_var = 'BLOQ'
y = coho.groupby('date').first()

X = pd.concat([
    y[['season','morn_midday_eve']],
    fish[fish.species=='coho'],
    flow[['stage','logflow','logflow1','quartile']], 
    wtemp,
    met[['temp', 'rain', 'rain3T', 'rain30T','dry_days','dry']], 
    moon.photoperiod, moon.moon_phase,
    hatch_vars[['release','release1','release3T','release7T','release14T']],
    lagoon[['pH','turb','DO']]
    ], axis=1)
X = X.loc[y.index]


## Correlation
print(pd.concat([y[dep_var],X], axis=1).corr()[dep_var].sort_values())

#%%
## Categories
for c in ['moon_phase','quartile','dry', 'season']:
    X = pd.concat([X, pd.get_dummies(X[c], prefix=c, drop_first=True)], axis=1)
    X = X.drop(c, axis=1)
#X = X.loc[y.index]
    
## Logistic Regression
lm = sm.Logit(y[dep_var],X[['stage','rain30T','wtemp','photoperiod']], missing='drop').fit_regularized()
print(lm.summary())
