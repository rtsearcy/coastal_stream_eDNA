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
import statsmodels.formula.api as smf
from scipy import stats, signal
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.tsatools import detrend
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import statsmodels.stats.contingency_tables as ct
import os
import datetime
from eDNA_corr import eDNA_corr

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 8)
# pd.set_option('display.width', 175)
np.seterr(divide='ignore')

folder = '../data/'  # Data folder

### eDNA Data
# Contains target, replicate #,dilution level, concentration (post-dilution factor)
eDNA = pd.read_csv(os.path.join(folder,'eDNA','eDNA.csv'), parse_dates=['dt','date'])

dr = pd.date_range('2019-03-25', '2020-04-04')  # project date range
eDNA = eDNA[eDNA.date.isin(dr)]  

### Set BLOD to 0 for stats?
# Note: setting to NAN biases the data (excludes many samples where we know conc < lod)
# eDNA.loc[eDNA.BLOD == 1,'eDNA'] = 0
# eDNA.loc[eDNA.BLOD == 1,'log10eDNA'] = 0 

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

hatch_vars['tr'] = 0  # Total fish released in watershed (0-3d, 7d lag)
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

# Fish released Near weir
hatch_vars['S0'] = 0  
hatch_vars.loc[hatch.index, 'S0'] = hatch['S0'].fillna(0)


### Trap Data (Fish Counts/Water Temp)
## Fish counts
fish = pd.read_csv(os.path.join(folder,'NOAA_data', 'fish_vars.csv'), 
                 parse_dates = ['date'], index_col=['date'])

fish['logbiomass'] = np.log10(fish['biomass'])
fish['logN_fish'] = np.log10(fish['N_fish'] + 1)

fish['fish_present'] = 1
fish.loc[fish.N_fish == 0,'fish_present'] = 0

## Water temperature at the weir
wtemp = pd.read_csv(os.path.join(folder,'NOAA_data', 'weir_wtemp.csv'), 
                    parse_dates = ['dt'], index_col=['dt'])
wtemp = pd.concat([wtemp.resample('D').mean().round(3), 
                   wtemp.resample('D').min(), 
                   wtemp.resample('D').max()],axis=1)
wtemp.columns = ['wtemp', 'wtemp_min', 'wtemp_max']
wtemp.index.name = 'date'

## Lagoon WQ
lagoon = pd.read_csv(os.path.join(folder,'NOAA_data', 'lagoon_wq.csv'), parse_dates = ['dt'], index_col=['dt'])
lagoon = lagoon.resample('D').mean()
lagoon.columns = ['lagoon_'+c for c in lagoon.columns]
lagoon.index.name = 'date'
# wtemp in lagoon and weir - PCC of 0.86


### Flow Data
## Scott Creek (from NOAA), daily TS (with regressed missing data)
flow_all = pd.read_csv(os.path.join(folder,'flow','scott_creek_daily_flow.csv'), parse_dates=['date'], index_col=['date'])
flow = flow_all[dr[0]:dr[-1]].copy()
flow.drop('regressed', axis=1, inplace=True)

for i in [1,2,3]:  # previous days flow
    flow['logflow' + str(i)] = flow.logflow.shift(i)
    
# Flow difference / Spikes
flow['flow_diff'] = np.log10(flow.flow.diff().abs().round(3))
flow.loc[flow['flow_diff']==-np.inf, 'flow_diff'] = np.log10(0.001)

flow['flow_spike'] = 0
flow.loc[flow['flow_diff'] >= 1, 'flow_spike'] = 1

flow['flow_quartile'] = 4  # Indicate which quantile of the record the flow data are in
flow.loc[flow.flow < flow_all.flow.quantile(.75),'flow_quartile'] = 3
flow.loc[flow.flow < flow_all.flow.quantile(.5),'flow_quartile'] = 2
flow.loc[flow.flow <= flow_all.flow.quantile(.25),'flow_quartile'] = 1

flow['flow_high'] = 0  # Flow greater, lower than median for the year
flow.loc[flow.flow_quartile.isin([3,4]),'flow_high'] = 1

###  Met/Rain/Rad
## Daily Rain/Temp Data
met = pd.read_csv(os.path.join(folder,'met','LD_swanton_daily_met.csv'), parse_dates=['date'], index_col=['date'])
met = met.dropna(how='all')
met['rain'] = met.rain * 25.4  # convert to metric
met['temp'] = (5/9)*(met.temp - 32)

## Rain vars
for i in [1]:  # previous days rainfall
    met['rain' + str(i)] = met.rain.shift(i)
    met['lograin' + str(i)] = np.log10(met['rain' + str(i)])
    met.loc[met['lograin' + str(i)]== -np.inf, 'lograin' + str(i)] = np.log10(0.001)
    
for i in [3,7,30]:  # rainfall total over X days
    met['rain'+str(i) + 'T'] = 0
    for d in range(1,i+1):
        met['rain'+str(i) + 'T'] += met['rain'].shift(d)
        met['lograin'+str(i) + 'T'] = np.log10(met['rain'+str(i) + 'T'])
        met.loc[met['lograin' + str(i) + 'T']== -np.inf, 'lograin' + str(i) + 'T'] = np.log10(0.001)

met['dry'] = 1  # Dry day (no rain in past 3 days)
met.loc[met.rain3T >= 1, 'dry'] = 0

met['dry_days'] = 0  # consecutive days since significant rain
met.loc[(met.rain < 1), 'dry_days'] = np.nan
met['dry_days'] = met.dry_days.ffill() + met.groupby(met.dry_days.notnull().cumsum()).cumcount()


## Hourly Air Temp Data
met_h = pd.read_csv(os.path.join(folder,'met','LD_swanton_hourly_airtemp.csv'), parse_dates=['dt'], index_col=['dt'])
met_h = met_h.dropna(how='all')
#met_h.resample('D').mean().plot()

## Rad data (Pescadero Station, with other met)
rad = pd.read_csv(os.path.join(folder,'met','Pescadero_CIMIS_day_met_20190101_20200501.csv'), parse_dates=['date'], index_col=['date'])


### Moon Phases / Photo Period
moon = pd.read_csv(os.path.join(folder,'moon_phases_2019_2020.csv'), parse_dates=['date'], index_col=['date'])
moon['moon_phase_continuous'] = np.cos(2*np.pi * moon.days_since_full_moon / 15)

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
pal_grey = ['#969696','#525252']  # grey, black
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
plt.figure(figsize=(8,5))

plt.subplot(2,1,1)
ax1 = plt.gca()
plt.bar(met.index, met.rain, width=1, color = '#1a8cff', ec='#1a8cff', alpha=.7, label='Rain', zorder=1)
plt.plot([], [], lw=1.75, color='#00264d', label='Stream Discharge', zorder=1)
plt.ylabel('Total Rainfall (mm)')
plt.ylim(0,45)
#plt.yscale('log')
#plt.ylim(1,300)
plt.legend(frameon=False, loc='upper right', ncol=2)

ax2 = plt.twinx(plt.gca())
plt.plot(flow.index, flow.logflow, lw=1.75, color='#00264d', label='Discharge', zorder=1)  ## Scott
plt.ylabel('Discharge (log$_{10}$ft$^3$/s)', rotation=270, ha='center', va='baseline', rotation_mode='anchor')
plt.ylim(1,3)

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
plt.plot(met.index, met.temp, lw=1.5, ls='--', color = '#ff6666', label='Air', zorder=1)
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

plt.figure(figsize=(8,5))

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

#%% Hypothesis tests on eDNA in different Enviro Regimes

# ## Daily Mean
# df_hyp = eDNA.groupby(['target','date']).mean()[['eDNA','log10eDNA']].reset_index().set_index('date')

## All Data
df_hyp = eDNA.groupby(['target','dt']).first()[['eDNA','log10eDNA','detected','BLOQ','date','season']].reset_index().set_index('date')


### Low Medium High Flow
temp = pd.merge(df_hyp, flow['flow_quartile'],how='left', left_index=True, right_index=True)

## Test
print('Flow Regime')
print('Low - < 23 CFS; Medium - 23<flow<52CFS; high - > 52 CFS')
print(temp.groupby(['target','flow_quartile']).describe()['eDNA'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print('Overall')
    print(stats.kruskal(temp[(temp.target==t)&(temp.flow_quartile==2)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.flow_quartile==3)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.flow_quartile==4)]['log10eDNA']))
    print('Low vs. Medium')
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.flow_quartile==2)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.flow_quartile==3)]['log10eDNA']))
    print('Low vs. High')
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.flow_quartile==2)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.flow_quartile==4)]['log10eDNA']))
    print('High vs. Medium')
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.flow_quartile==4)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.flow_quartile==3)]['log10eDNA']))
    
## Plot
plt.figure(figsize=(4,4))
sns.boxplot(x='flow_quartile',y='log10eDNA', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1,2],labels=['Low Flow','Medium Flow', 'High Flow'])
plt.xlabel('')
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend(frameon=False)
plt.tight_layout()

### Wet vs. Dry
temp = pd.merge(df_hyp, met['dry'],how='left', left_index=True, right_index=True)

## Test
print('\nWet vs. Dry Days\n(Wet - > 1 mm over 3 days)')
print(temp.groupby(['target','dry']).describe()['eDNA'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.dry==0)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.dry==1)]['log10eDNA']))
## Plot
plt.figure(figsize=(4,4))
sns.boxplot(x='dry',y='log10eDNA', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1],labels=['Wet Day', 'Dry Day'])
plt.xlabel('')
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend(frameon=False)
plt.tight_layout()

### TIde Phase
temp = pd.merge(df_hyp, moon['tide_phase'],how='left', left_index=True, right_index=True)

## Test
print('\nTide Category (Spring Neap')
print(temp.groupby(['target','tide_phase']).describe()['eDNA'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.tide_phase=='spring')]['log10eDNA'],
                             temp[(temp.target==t)&(temp.tide_phase=='neap')]['log10eDNA']))
## Plot
plt.figure(figsize=(4,4))
sns.boxplot(x='tide_phase',y='log10eDNA', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1],labels=['Neap', 'Spring'])
plt.xlabel('')
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend(frameon=False)
plt.tight_layout()

### Hatchery Releases
temp = pd.merge(df_hyp, hatch_vars['release3T'],how='left', left_index=True, right_index=True)

## Test
print('\nHatchery Releases')
print(temp.groupby(['target','release3T']).describe()['eDNA'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.release3T==0)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.release3T>0)]['log10eDNA']))
## Plot
plt.figure(figsize=(4,4))
sns.boxplot(x='release3T',y='log10eDNA', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1],labels=['Outside Release Window', 'Release 3D'])
plt.xlabel('')
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend(frameon=False)
plt.tight_layout()

#%% BLOQ testing by EV

## All Samples
df_bloq = eDNA.groupby(['target','dt']).first()[['detected','BLOQ','date']].reset_index().set_index('date')

### Flow
temp = pd.merge(df_bloq, flow['logflow'],how='left', left_index=True, right_index=True)

## Test
print('Flow')
print(temp.groupby(['target','BLOQ']).describe()['logflow'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.BLOQ==0)]['logflow'],
                             temp[(temp.target==t)&(temp.BLOQ==1)]['logflow']))
## Plot
plt.figure(figsize=(4,4))
sns.boxplot(x='BLOQ',y='logflow', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1],labels=['Above LOQ', 'BLOQ'])
plt.xlabel('')
plt.ylabel('Discharge (log$_{10}$cfs)')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend(frameon=False)
plt.tight_layout()

### Wtemp
temp = pd.merge(df_bloq, wtemp['wtemp'],how='left', left_index=True, right_index=True)

## Test
print('\nWater Temp.')
print(temp.groupby(['target','BLOQ']).describe()['wtemp'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.BLOQ==0)]['wtemp'],
                             temp[(temp.target==t)&(temp.BLOQ==1)]['wtemp']))
## Plot
plt.figure(figsize=(4,4))
sns.boxplot(x='BLOQ',y='wtemp', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1],labels=['Above LOQ', 'BLOQ'])
plt.xlabel('')
plt.ylabel('Water Temperature (C)')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend(frameon=False)
plt.tight_layout()

# ### Lagoon DO
# temp = pd.merge(df_bloq, lagoon['lagoon_DO'],how='left', left_index=True, right_index=True)

# ## Test
# print('\nDissolved Oxygen')
# print(temp.groupby(['target','BLOQ']).describe()['lagoon_DO'][['count','mean','50%','max']])
# for t in temp.target.unique():
#     print('\n' + t.upper())
#     print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.BLOQ==0)]['lagoon_DO'],
#                              temp[(temp.target==t)&(temp.BLOQ==1)]['lagoon_DO']))
# ## Plot
# plt.figure(figsize=(4,4))
# sns.boxplot(x='BLOQ',y='lagoon_DO', hue='target', data=temp, palette=pal)
# plt.xticks(ticks=[0,1],labels=['Above LOQ', 'BLOQ'])
# plt.xlabel('')
# plt.ylabel('Dissolved Oxygen (mg/l)')
# plot_spines(plt.gca())
# caps_off(plt.gca())
# flier_shape(plt.gca())
# plt.legend(frameon=False)
# plt.tight_layout()

### Photoperiod
temp = pd.merge(df_bloq, moon['photoperiod'],how='left', left_index=True, right_index=True)

## Test
print('\nPhotoperiod')
print(temp.groupby(['target','BLOQ']).describe()['photoperiod'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.BLOQ==0)]['photoperiod'],
                             temp[(temp.target==t)&(temp.BLOQ==1)]['photoperiod']))
## Plot
plt.figure(figsize=(4,4))
sns.boxplot(x='BLOQ',y='photoperiod', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1],labels=['Above LOQ', 'BLOQ'])
plt.xlabel('')
plt.ylabel('Photoperiod (Day)')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend(frameon=False)
plt.tight_layout()


#%% Combine eDNA, Fish, EVs

### Combine datasets
df_combo = pd.DataFrame(index=dr)
df_combo = pd.DataFrame()

## eDNA columns
for t in eDNA.target.unique():  
    
    # # Use all data
    # temp = eDNA[eDNA.target==t].groupby('dt').first().reset_index().set_index('date') 
    
    # # Use first sample of the day
    # temp = eDNA[eDNA.target==t].groupby('date').first()  
    
    #Use mean conc of the day
    temp = eDNA[eDNA.target==t].groupby('date').mean()   
    temp['detected'] = np.ceil(temp['detected']).astype(int) # if any detected, mean detected
    temp['BLOQ'] = np.floor(temp['BLOQ']).astype(int) # if any above LOQ, mean above LOQ
    
    temp = temp[['eDNA','log10eDNA','detected','BLOQ']]
    temp.columns = [t+'_'+c for c in temp.columns]
    df_combo = pd.concat([df_combo,temp],axis=1)
    
## Date variables
date_cols = ['year', 'month', 'year_month', 'season',
             'week', 'year_week','day_of_year', 
             'hour', 'morn_midday_eve'
             ]

# # All samples
# temp = eDNA.groupby('dt').first().reset_index().set_index('date')[date_cols]
# df_combo = pd.concat([df_combo, temp], axis=1)

temp = eDNA.groupby('date').first()[date_cols]  # first of day, daily mean
df_combo = pd.merge(df_combo, temp, how='left', left_index=True, right_index=True)

# Enviro Variables
temp = pd.concat([hatch_vars, 
                  flow, 
                  wtemp,
                  lagoon,
                  met, 
                  moon], axis=1)

df_combo = pd.merge(df_combo, temp, how='left',left_index=True, right_index=True)

## TBD: Fish
temp_fish = pd.DataFrame(index=dr)
for t in fish.species.unique():  
    temp = fish[fish.species==t].copy()
    temp.drop('species',axis=1,inplace=True)
    temp.columns = [t+'_'+c for c in temp.columns]
    temp_fish = pd.concat([temp_fish,temp],axis=1)
df_combo = pd.merge(df_combo, temp_fish, how='left',left_index=True, right_index=True)


#%% Correlations
print('- - CORRELATIONS - - ')

### Correlations
cor_type='spearman'
df_corr = df_combo.corr(method=cor_type)

EV = ['flow',#'flow_quartile','flow_high',
      'wtemp',
      'lagoon_turb', 'lagoon_pH','lagoon_DO',  
      'temp','dry','dry_days','rain1','rain3T','rain7T','rain30T',
      'release','tr','release3T','tr3T',
      'photoperiod','moon_phase_continuous',
      #'coho_N_fish','coho_biomass','trout_N_fish','trout_biomass',
      ]

print(df_corr.loc[['coho_eDNA','trout_eDNA'],EV].T)

eDNA_cor = pd.DataFrame()
for t in ['coho_eDNA','coho_BLOQ',
          #'coho_N_fish','coho_biomass_total',
          'trout_eDNA','trout_BLOQ']:
          #'trout_N_fish','trout_biomass_total']:
    for e in EV:
        r, p = stats.spearmanr(df_combo[[t,e]],nan_policy='omit')
        r = round(r,2)
        if r>0:
            rsign='+'
        else:
            rsign='-'
            
        temp ={'EV': e, 'target':t, 'r':abs(r),'r_sign':rsign, 'p':p}
        eDNA_cor = eDNA_cor.append(temp, ignore_index=True)

print(cor_type.capitalize() + ' Correlations:')
for t in eDNA_cor.target.unique():
    print('\n')
    print(eDNA_cor[eDNA_cor.target==t].sort_values('r', ascending=False))
    
#%% Hatchery effect
print('\nEffect of hatchery releases on coho')
release_vars = ['release','release1','release3T','release7T']
df = df_combo[['coho_BLOQ','coho_log10eDNA',
               'coho_fish_present','coho_N_fish','coho_biomass'
               ]+release_vars]

r='release7T'
print('Release variable: ' + r)
for i in ['coho_log10eDNA','coho_N_fish','coho_biomass']:
    print('\n' + i)
    print(df.groupby(r).describe()[[i]].round(3))
    print(stats.mannwhitneyu(df[(df[r]==0)][i], df[(df[r]==1)][i]))

## Side by Side BOXPLOTS
plt.figure(figsize=(3,6))

plt.subplot(2,1,1)  # eDNA
sns.boxplot(x=r, y='coho_log10eDNA', data=df, palette=pal_grey, width=.5)
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
plt.xlabel('')
plt.gca().set_xticklabels(['',''])
plt.title(r.upper())

plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())

plt.subplot(2,1,2)  # Fish
sns.boxplot(x=r, y='coho_N_fish', data=df, palette=pal_grey, width=.5)
plt.ylabel('N (fish)')
plt.xlabel('')
plt.gca().set_xticklabels(['No Release','Release'])

plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())

plt.tight_layout()





#%% Detection rate (eDNA vs. Fish)
print('\nDetection Rates')
for t in eDNA.target.unique():
    print('\n' + t.upper())
    d = df_combo[[t+'_detected',t+'_BLOQ',t+'_fish_present']].dropna()
    d[t+'_above_LOQ'] = (d[t+'_BLOQ'] - 1).abs()
    print('\n# Days Traps Assessed/eDNA measured: ' + str(len(d)))
    temp = pd.concat([d.sum(),(100*d.sum() / len(d)).round(3)],axis=1)
    temp.columns = ['N','%']
    print(temp)
    print('\n')
    tab = pd.crosstab(d[t+'_above_LOQ'],d[t+'_fish_present'])
    print(tab)
    # print('acc: ' + str(round((tab[1][1] + tab[0][0]) / tab.sum().sum(), 3)))
    # print('sens: ' + str(round(tab[1,1] / (pt[1,1] + pt[1,0]), 3)))
    # print('spec: ' + str(round(tab[0,0] / (pt[0,0] + pt[0,1]), 3)))
    print(ct.mcnemar(tab, exact=False))  # compare 
    
    
    

#%% Simple Regressions
t = 'coho'
## Fish on eDNA
#formula = 'np.sqrt(' + t + '_N_fish) ~ ' + t + '_log10eDNA'    # Fish counts
#formula = 'np.sqrt(' + t + '_biomass) ~ ' + t +'_log10eDNA'  # Biomass

## eDNA on Fish
formula = t +'_log10eDNA ~ np.sqrt(' + t + '_N_adult)'    # Fish counts
#formula = t +'_log10eDNA ~ np.sqrt(' + t + '_biomass)'  # Biomass

## Linear Regression
#lm = smf.ols(formula, data=df_combo).fit()
lm = smf.glsar(formula, rho=2, data=df_combo).iterative_fit(maxiter=10)

print(lm.summary())
print('N - ' + str(len(lm.predict())))
print('\nRMSE: ' + str(round(lm.mse_resid**.5, 3)))  # OLS
print('\nVIFs:')
print(lm.params.index)

#%% Scatterplot - Fish vs. eDNA
df = df_combo[['coho_log10eDNA','trout_log10eDNA',
               'coho_N_fish','coho_biomass','coho_N_adult','coho_biomass_adult',
               'trout_N_fish','trout_biomass','trout_N_adult','trout_biomass_adult',
               'season']]

df_plot = pd.DataFrame()
for t in ['coho','trout']:
    cols = [c for c in df.columns if t in c]
    temp = df[['season'] + cols].copy()
    temp.columns = ['season'] + [c.replace(t+'_','') for c in cols]
    temp['target']=t
    df_plot = df_plot.append(temp)
               
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)  # N
sns.scatterplot(x='log10eDNA',y='N_fish',hue='target',data=df_plot, palette=pal)
plt.xlabel('log$_{10}$(eDNA copies/mL + 1)')
plt.ylabel('N')
plt.legend([],frameon=False)
plot_spines(plt.gca())

plt.subplot(1,2,2)  # N
sns.scatterplot(x='log10eDNA',y='biomass',hue='target',data=df_plot, palette=pal)
plt.xlabel('log$_{10}$(eDNA copies/mL + 1)')
plt.ylabel('Biomass (kg)')
plt.legend(frameon=False)
plot_spines(plt.gca())

plt.tight_layout()

#%% Regressions - eDNA on FIsh / EVs
t = 'trout'
metric = 'N_fish'
#metric = 'biomass'
formula = t + '_log10eDNA ~ np.sqrt(' + t +'_'+metric+') + wtemp + logflow + lograin7T + tide_phase + photoperiod'
if t == 'coho':
    formula = formula + '+ release3T'

## Linear Regression
#lm = smf.ols(formula, data=df_combo).fit()
lm = smf.glsar(formula, rho=2, data=df_combo).iterative_fit(maxiter=10)

## Logistic Regression
#lm = smf.logit(formula, data=df_combo).fit()

print(lm.summary())
print('N - ' + str(len(lm.predict())))

if 'Binary' in str(type(lm)):
    print('\nAIC: ' + str(round(lm.aic, 3)))
    print('\nObserved/Predicted:')
    pt = lm.pred_table()
    print(pt)
    print('acc: ' + str(round((pt[1,1] + pt[0,0]) / pt.sum(), 3)))
    print('sens: ' + str(round(pt[1,1] / (pt[1,1] + pt[1,0]), 3)))
    print('spec: ' + str(round(pt[0,0] / (pt[0,0] + pt[0,1]), 3)))

else:
    print('\nRMSE: ' + str(round(lm.mse_resid**.5, 3)))  # OLS
    print('\nVIFs:')
    print(lm.params.index)
    variables = lm.model.exog
    print([round(VIF(variables, i),3) for i in range(variables.shape[1])])
    
    
# coho_log10eDNA ~  season + wtemp + dry_days + photoperiod + logflow
# trout_log10eDNA ~  logflow + season + wtemp + photoperiod + lagoon_turb

#%% Regressions - Fish on eDNA / EVs
t='trout'
metric = 'N_fish'
#metric = 'biomass'
formula = 'np.sqrt(' + t +'_'+metric+') ~ coho_log10eDNA + wtemp + logflow + lograin7T + tide_phase + photoperiod'
if t == 'coho':
    formula = formula + '+ release3T'

## Linear Regression
#lm = smf.ols(formula, data=df_combo).fit()
lm = smf.glsar(formula, rho=2, data=df_combo).iterative_fit(maxiter=10)

## Logistic Regression
#lm = smf.logit(formula, data=df_combo).fit()

print(lm.summary())
print('N - ' + str(len(lm.predict())))

if 'Binary' in str(type(lm)):
    print('\nAIC: ' + str(round(lm.aic, 3)))
    print('\nObserved/Predicted:')
    pt = lm.pred_table()
    print(pt)
    print('acc: ' + str(round((pt[1,1] + pt[0,0]) / pt.sum(), 3)))
    print('sens: ' + str(round(pt[1,1] / (pt[1,1] + pt[1,0]), 3)))
    print('spec: ' + str(round(pt[0,0] / (pt[0,0] + pt[0,1]), 3)))

else:
    print('\nRMSE: ' + str(round(lm.mse_resid**.5, 3)))  # OLS
    print('\nVIFs:')
    print(lm.params.index)
    variables = lm.model.exog
    print([round(VIF(variables, i),3) for i in range(variables.shape[1])])
    
    
# coho_log10eDNA ~  season + wtemp + dry_days + photoperiod + logflow
# trout_log10eDNA ~  logflow + season + wtemp + photoperiod + lagoon_turb

#%%
#y = trout.set_index('date').log10eDNA  # all data
dep_var = 'BLOQ'
y = trout.groupby('date').first()
X = pd.concat([
    y[['log10eDNA','BLOQ','detected','season','morn_midday_eve']],
    fish[fish.species=='trout'],
    flow[['stage','logflow','logflow1','quartile']], 
    wtemp,
    met[['temp', 'rain', 'rain3T', 'rain30T','dry_days','dry']], 
    moon.photoperiod, moon.moon_phase,
    hatch_vars[['release','release1','release3T','release7T','release14T']],
    lagoon[['pH','turb','DO']]
    ], axis=1)
X = X.loc[y.index]


## Correlation
print(pd.concat([y[dep_var],X], axis=1).corr(method='spearman')[dep_var].sort_values())

#%%
## Categories
for c in ['moon_phase','quartile','dry', 'season']:
    X = pd.concat([X, pd.get_dummies(X[c], prefix=c, drop_first=True)], axis=1)
    X = X.drop(c, axis=1)
#X = X.loc[y.index]
    
## Logistic Regression
lm = sm.Logit(y[dep_var],sm.add_constant(X[['biomass_total']]), missing='drop').fit_regularized()
print(lm.summary())
