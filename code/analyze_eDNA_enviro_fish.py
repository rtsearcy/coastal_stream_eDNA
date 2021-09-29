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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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

## Separate targets
trout = eDNA[eDNA.target=='trout'].sort_values('dt') 
coho = eDNA[eDNA.target=='coho'].sort_values('dt')


### ESP logs
# Contains sampling times, volumes, ESP name
ESP = pd.read_csv(os.path.join(folder,'ESP_logs','ESP_logs_combined.csv'), 
                  parse_dates = ['sample_wake','sample_start','sample_mid','sample_end','date'], 
                  index_col=['sample_mid'])  # can also use sample_start, but probably little diff.

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

fish['coho_fish_present'] = np.nan
fish.loc[fish.coho_N_fish == 0,'coho_fish_present'] = 0
fish.loc[fish.coho_N_fish > 0,'coho_fish_present'] = 1
fish['trout_fish_present'] = np.nan
fish.loc[fish.trout_N_fish == 0,'trout_fish_present'] = 0
fish.loc[fish.trout_N_fish > 0,'trout_fish_present'] = 1

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

# Flow spike (>75% percentile flow [141 cfs])
flow['flow_spike'] = 1
flow.loc[flow.flow < flow_all.flow.quantile(.9), 'flow_spike'] = 0

flow['flow_quartile'] = 4  # Indicate which quantile of the record the flow data are in
flow.loc[flow.flow < flow_all.flow.quantile(.75),'flow_quartile'] = 3
flow.loc[flow.flow < flow_all.flow.quantile(.5),'flow_quartile'] = 2
flow.loc[flow.flow <= flow_all.flow.quantile(.25),'flow_quartile'] = 1

flow['flow_high'] = 0  # Flow greater, lower than median for the year
flow.loc[flow.flow_quartile.isin([3,4]),'flow_high'] = 1

## Berm Status
flow['berm_open'] = 1
flow.loc['2019-09-05':'2019-12-04','berm_open'] = 0

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

met['dry'] = 1  # Dry day (no rain > 5mm in past 3 days)
met.loc[met.rain3T >= 10, 'dry'] = 0
met['wet'] = np.abs(met['dry'] - 1)

met['dry_days'] = 0  # consecutive days since significant rain
met.loc[(met.rain < 1), 'dry_days'] = np.nan
met['dry_days'] = met.dry_days.ffill() + met.groupby(met.dry_days.notnull().cumsum()).cumcount()


## Hourly Air Temp Data
met_h = pd.read_csv(os.path.join(folder,'met','LD_swanton_hourly_airtemp.csv'), parse_dates=['dt'], index_col=['dt'])
met_h = met_h.dropna(how='all')
#met_h.resample('D').mean().plot()


### Moon Phases / Photo Period
moon = pd.read_csv(os.path.join(folder,'met','moon_phases_2019_2020.csv'), parse_dates=['date'], index_col=['date'])
moon['moon_phase_continuous'] = np.cos(2*np.pi * moon.days_since_full_moon / 15)

moon['long_day'] = 1
moon.loc[moon.photoperiod < 0.5,'long_day'] = 0

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


def plot_regression(df,lm):
    ''' df - dataframe of variables; lm - fitted regression model
    '''
    
    ### Plot Regression Output
    f, (a1,a2) = plt.subplots(1, 2,figsize=(10,3.5), gridspec_kw={'width_ratios': [3, 1.5]})
    
    ## Time Series
    plt.sca(a1)
    plt.scatter(df.index, df[t+'_log10eDNA'], s=8, c='r',label='Observed')
    plt.plot(df.index, lm.predict(), color='k',label='Modeled')
    plt.xlabel('')
    plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
    plt.legend(frameon=False)
    plot_spines(a1)
    
    ## Scatter Plot
    plt.sca(a2)
    plt.scatter(df[t+'_log10eDNA'], lm.predict(), s=8, c='k')
    x = np.linspace(plt.gca().get_xlim()[0],plt.gca().get_xlim()[1]) 
    plt.plot(x,x,ls=":",color='k',alpha=0.75)
    plt.xlabel('Observed')
    plt.ylabel('Modeled')
    plot_spines(a2)

    plt.suptitle(lm.model.formula)
    plt.tight_layout()

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

## Berm dates
plt.axvline(datetime.datetime(2019,9,5), color='k', alpha=.8, ls=':')
plt.axvline(x=datetime.datetime(2019,12,4), ymin=0, ymax=.83, color='k', alpha=.8, ls=':')
plt.axvline(x=datetime.datetime(2019,12,4), ymin=0.96, ymax=1, color='k', alpha=.8, ls=':')

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


### Air/Water Temp/photoperiod
#plt.figure(figsize=(10,4))
plt.subplot(2,1,2)

ax1 = plt.gca()
wtemp_plot = wtemp.resample('D').mean()
plt.plot(wtemp_plot.index, wtemp_plot.wtemp, lw=1.75, color='#004080', label='Water', zorder=2)
plt.plot(met.index, met.temp, lw=1.5, ls='--', color = '#ff6666', label='Air', zorder=1)
plt.plot([], lw=1.5, ls=':', color = 'k', label='Photoperiod')
plt.ylabel('Temperature (°C)')
plt.ylim(3,33)
plt.legend(frameon=False, ncol=3, loc='upper right')

ax2 = plt.twinx(plt.gca())
plt.plot(moon.photoperiod, lw=1.5, ls=':', color = 'k', label='Photoperiod')
plt.ylabel('Photoperiod (day)', rotation=270, ha='center', va='baseline', rotation_mode='anchor')
#plt.ylim(3,30)
#ax2.set_yticklabels([])

plt.xlim(dr[0],dr[-1])
offset = 7
for a in [ax1, ax2]:
    a.spines['left'].set_position(('outward', offset))
    a.spines['right'].set_position(('outward', offset))
    a.spines['top'].set_visible(False)
    #a.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(folder.replace('data','figures'),'temp_flow_rain_TS.png'),dpi=300)

#%% Hypothesis tests on eDNA in different Enviro Regimes

# ## Daily Mean
df_hyp = eDNA.groupby(['target','date']).mean()[['eDNA','log10eDNA']].reset_index().set_index('date')

## All Data
#df_hyp = eDNA.groupby(['target','dt']).first()[['eDNA','log10eDNA','detected','BLOQ','date','season']].reset_index().set_index('date')


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
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
sns.boxplot(x='flow_quartile',y='log10eDNA', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1,2],labels=['Low','Medium', 'High'])
plt.title('Creek Discharge', fontsize=11)
plt.xlabel('')
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend([], frameon=False)
#plt.tight_layout()

### Long Day (Photoperiod)
temp = pd.merge(df_hyp, moon['long_day'],how='left', left_index=True, right_index=True)

## Test
print('\nLong Day (Photoperiod < 0.5 day')
print(temp.groupby(['target','long_day']).describe()['eDNA'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.long_day==0)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.long_day==1)]['log10eDNA']))
## Plot
#plt.figure(figsize=(4,4))
plt.subplot(1,3,2)
sns.boxplot(x='long_day',y='log10eDNA', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1],labels=['< 0.5 Day', '> 0.5'])
plt.title('Photoperiod', fontsize=11)
plt.xlabel('')
plt.ylabel('')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend([],frameon=False)
#plt.tight_layout()


### Berm
temp = pd.merge(df_hyp, flow['berm_open'],how='left', left_index=True, right_index=True)

## Test
print('\nBerm Status (Closed/Open)')
print(temp.groupby(['target','berm_open']).describe()['eDNA'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.berm_open==0)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.berm_open==1)]['log10eDNA']))
## Plot
#plt.figure(figsize=(4,4))
plt.subplot(1,3,3)
sns.boxplot(x='berm_open',y='log10eDNA', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1],labels=['Bermed', 'Open'])
plt.title('Creek Mouth', fontsize=11)
plt.xlabel('')
plt.ylabel('')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())

leg = plt.gca().get_legend()
leg.legendHandles[0].set_color(pal[0]) # coho
leg.legendHandles[1].set_color(pal[1]) # trout

legend_elements = [Patch(facecolor=pal[0], edgecolor='k',label='$\it{O. kisutch}$'),
                   Patch(facecolor=pal[1], edgecolor='k',label='$\it{O. mykiss}$')]
plt.gca().legend(handles=legend_elements,frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_by_EV.png'),dpi=300)


### EXTRA: Wet vs. Dry
temp = pd.merge(df_hyp, met['wet'],how='left', left_index=True, right_index=True)

## Test
print('\nWet vs. Dry Days\n(Wet - > 5 mm over 3 days)')
print(temp.groupby(['target','wet']).describe()['eDNA'][['count','mean','50%','max']])
for t in temp.target.unique():
    print('\n' + t.upper())
    print(stats.mannwhitneyu(temp[(temp.target==t)&(temp.wet==0)]['log10eDNA'],
                             temp[(temp.target==t)&(temp.wet==1)]['log10eDNA']))
## Plot
plt.figure(figsize=(4,4))
sns.boxplot(x='wet',y='log10eDNA', hue='target', data=temp, palette=pal, saturation=.9)
plt.xticks(ticks=[0,1],labels=['Dry Day', 'Wet Day'])
plt.xlabel('')
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
plot_spines(plt.gca())
caps_off(plt.gca())
flier_shape(plt.gca())
plt.legend(frameon=False)
plt.tight_layout()

### EXTRA: Hatchery Releases
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


#%% Combine eDNA, Fish, EVs

### Combine datasets
df_combo = pd.DataFrame(index=dr)
df_combo = pd.DataFrame()

## eDNA columns
for t in eDNA.target.unique():  
    
    # # Use all data
    # temp = eDNA[eDNA.target==t].groupby('dt').first().reset_index().set_index('date') 
    
    # # Use first sample of the day
    #temp = eDNA[eDNA.target==t].groupby('date').first()  
    
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

## Fish
df_combo = pd.merge(df_combo, fish, how='left',left_index=True, right_index=True)

#%% Correlations
print('- - CORRELATIONS - - ')

### Correlations
cor_type='spearman'
df_corr = df_combo.corr(method=cor_type)

EV = ['flow',#'flow_quartile','flow_high',
      'berm_open',
      'wtemp',
      #'lagoon_turb', 'lagoon_pH','lagoon_DO',  
      'temp','dry','dry_days','rain1','rain3T','rain7T','rain30T',
      'release','tr','release3T','tr3T',
      'photoperiod','moon_phase_continuous',
      'coho_N_fish','coho_biomass','trout_N_fish','trout_biomass',
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


#%% Regressions - eDNA on EVs
t = 'trout'
formula = t + '_log10eDNA ~ wtemp + logflow + photoperiod + wet + berm_open'

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
    

### Plot Regression Output
plot_regression(df_combo,lm)
#%% Fish detection rate (eDNA vs. Fish)

print('\nFish Stats on Sampling Days')
temp = df_combo[['fish_count','coho_N_fish','coho_N_adult','coho_N_juvenile','coho_biomass','coho_fish_present',
                 'trout_N_fish','trout_N_adult','trout_N_juvenile','trout_biomass','trout_fish_present']]
temp = temp.dropna()
print('# trap days: ' + str(temp['fish_count'].sum()))
print('\nSums:')
print(temp.sum())

print('\nFish detection Rates')
for t in eDNA.target.unique():
    print('\n' + t.upper())
    d = df_combo[[t+'_detected',t+'_BLOQ',t+'_fish_present']].dropna()
    d[t+'_above_LOQ'] = (d[t+'_BLOQ'] - 1).abs()
    print('\n# Days Traps Assessed/eDNA measured: ' + str(len(d)))
    temp = pd.concat([d.sum(),(100*d.sum() / len(d)).round(3)],axis=1)
    temp.columns = ['N','%']
    print(temp)
    print('\n')
    tab = pd.crosstab(d[t+'_detected'],d[t+'_fish_present'])
    #tab = pd.crosstab(d[t+'_above_LOQ'],d[t+'_fish_present'])
    print(tab)
    # print('acc: ' + str(round((tab[1][1] + tab[0][0]) / tab.sum().sum(), 3)))
    # print('sens: ' + str(round(tab[1,1] / (pt[1,1] + pt[1,0]), 3)))
    # print('spec: ' + str(round(tab[0,0] / (pt[0,0] + pt[0,1]), 3)))
    print(ct.mcnemar(tab, exact=False))  # compare 
    
    
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

## One Plot               
plt.figure(figsize=(4.5,7))
plt.subplot(2,1,1)  # N
sns.scatterplot(x='log10eDNA',y='N_fish',hue='target', style='target', data=df_plot, palette=pal, size=[9]*len(df_plot))
#plt.xlabel('log$_{10}$(eDNA copies/mL + 1)')
plt.xlabel('')
plt.gca().set_xticklabels([]) ## turn off yticks
plt.ylabel('N')
plt.ylim(-5,390)
plt.legend([],frameon=False)
plot_spines(plt.gca())

legend_elements = [Line2D([0], [0], marker='o', color='w', label='$\it{O. kisutch}$', markerfacecolor=pal[0], markersize=6),
                   Line2D([0], [0], marker='X', color='w', label='$\it{O. mykiss}$', markerfacecolor=pal[1], markersize=7)
                   ]
leg = plt.gca().legend(handles=legend_elements, loc='upper center', ncol=2)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.75)

plt.subplot(2,1,2)  # N
sns.scatterplot(x='log10eDNA',y='biomass',hue='target',style='target', data=df_plot, palette=pal, size=[9]*len(df_plot))
plt.xlabel('log$_{10}$(eDNA copies/mL + 1)')
plt.ylabel('Biomass (kg)')
plt.legend([],frameon=False)

plot_spines(plt.gca())

plt.tight_layout()

plt.savefig(os.path.join(folder.replace('data','figures'),'fish_eDNA_scatter.png'),dpi=300)


## Two Plots
plt.figure(figsize=(9,7))
plt.subplot(2,2,1)  # N Coho
plt.scatter(x='coho_log10eDNA',y='coho_N_fish',data=df, c=pal[0], s=9)
plt.xlabel('log$_{10}$(eDNA copies/mL + 1)')
plt.xlabel('')
plt.gca().set_xticklabels([]) ## turn off yticks
plt.ylabel('N')
plt.title('$\it{O. kisutch}$')

plot_spines(plt.gca())

plt.subplot(2,2,3)  # Biomass Coho
plt.scatter(x='coho_log10eDNA',y='coho_biomass',data=df, c=pal[0], s=9)
plt.xlabel('log$_{10}$(eDNA copies/mL + 1)')
plt.ylabel('Biomass (kg)')

plt.legend([],frameon=False)
plot_spines(plt.gca())

plt.subplot(2,2,2)  # N Trout
plt.scatter(x='trout_log10eDNA',y='trout_N_fish',data=df, c=pal[1], s=9, marker='^')
plt.xlabel('log$_{10}$(eDNA copies/mL + 1)')
plt.xlabel('')
plt.gca().set_xticklabels([]) ## turn off yticks
plt.ylabel('')
plt.title('$\it{O. mykiss}$')
plot_spines(plt.gca())

plt.subplot(2,2,4)  # Biomass Trout
plt.scatter(x='trout_log10eDNA',y='trout_biomass',data=df, c=pal[1], s=9, marker='^')
plt.xlabel('log$_{10}$(eDNA copies/mL + 1)')
plt.ylabel('')
plt.legend([],frameon=False)
plot_spines(plt.gca())

plt.tight_layout()

#%% Simple Regressions - Fish on eDNA
t = 'coho'
## Fish on eDNA
#formula = 'np.sqrt(' + t + '_N_fish) ~ ' + t + '_log10eDNA'    # Fish counts
formula = 'np.sqrt(' + t + '_biomass) ~ ' + t +'_log10eDNA'  # Biomass

## eDNA on Fish
#formula = t +'_log10eDNA ~ np.sqrt(' + t + '_N_adult)'    # Fish counts
#formula = t +'_log10eDNA ~ np.sqrt(' + t + '_biomass)'  # Biomass

## Linear Regression
#lm = smf.ols(formula, data=df_combo).fit()
lm = smf.glsar(formula, rho=2, data=df_combo).iterative_fit(maxiter=10)

print(lm.summary())
print('N - ' + str(len(lm.predict())))
print('\nRMSE: ' + str(round(lm.mse_resid**.5, 3)))  # OLS
print('\nVIFs:')
print(lm.params.index)

### Plot Regression Output
df_plot = df_combo[[t+'_log10eDNA',t+'_N_fish',t+'_biomass']].dropna()
plot_regression(df_plot,lm)

#%% Regressions - Fish on eDNA / EVs
t='trout'
#metric = 'N_fish'
metric = 'biomass'
formula = 'np.sqrt(' + t +'_'+metric+') ~ ' + t + '_log10eDNA + wtemp + logflow + photoperiod + wet + berm_open'

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
    
#df_plot = df_combo[[i for i in df_combo.columns if i in formula]].dropna()
#plot_regression(df_plot,lm)
