#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Aggregate eDNA and ESP data 
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

Aggregates and analyzes the ESP and eDNA data

ESP Sampling logs
- Sample volumes/rates (distributions, time series)
- Time of day


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import os

folder = '../data/'  # Data folder

### Load Combined ESP logs
# Contains sampling times, volumes, ESP name
ESP = pd.read_csv(os.path.join(folder,'ESP_logs','ESP_logs_combined.csv'), 
                 parse_dates = ['sample_wake','sample_start','sample_mid','sample_end'], 
                 index_col=['sample_mid'])  # can also use sample_start, but probably little diff.

ESP.dropna(inplace=True, subset=['sample_wake', 'sample_start', 'sample_end', 'sample_duration',
       'vol_target', 'vol_actual', 'vol_diff',])  # Drop samples with no time or volume data 

ESP = ESP[~ESP.lab_field.isin(['lab','control', 'control '])]  # Keep deployed/field/nana; Drop control/lab samples


### Load eDNA Data
# Contains mean qPCR concentrations, including target, n_replicates/BLOD, 
# delta_Ct, and dilution level (post-dilution factor)
qPCR = pd.read_csv(os.path.join(folder,'eDNA','qPCR_calculated_mean.csv'))
qPCR.set_index(['id','target','dilution'], inplace=True)


### Combine eDNA and ESP data
df = pd.merge(qPCR.reset_index(), ESP.reset_index(), how='left', on='id')
df = df[~df.sample_mid.isna()]  # Drop samples w/o ESP data


### Convert from copies/rxn to copies/mL filtered
conv_factor = 66.67 / df.vol_actual     # conversion from MBARI
df['eDNA'] = df.conc * conv_factor
df['eDNA_sd'] = df.conc_sd * conv_factor

df['log10eDNA'] = df.log10conc + np.log10(conv_factor)
df['log10eDNA_sd'] = df.log10conc_sd  # stdev values stay same in log transform


### Choose dilution
df = df[df.dilution == '1:5'] 


### Save aggregate dataframe
df['dt'] = df['sample_mid']  # timestamp
df['ESP_file'] = df['log_file']
df['qPCR_file'] = df['source_file']

df = df[['dt',
         'id',
         'target',
         'eDNA',
         'eDNA_sd',
         'log10eDNA',
         'log10eDNA_sd',
         'BLOD',
         'BLOQ',
         'n_replicates',
         'n_amplified',
         'n_BLOD',
         'n_BLOQ',
         'dilution',
         #'delta_Ct',
         'ESP',
         'vol_actual',
         'sample_duration',
         'sample_rate',
         'morn_midday_eve',
         'ESP_file',
         'qPCR_file']]
df.set_index(['id','target'], inplace=True)
df.sort_values(['dt','target'], inplace=True)  # sort by date
df.to_csv(os.path.join(folder,'eDNA','eDNA.csv'))
print('ESP and qPCR data aggregated. eDNA concentrations saved.')
print('  N = ' + str(len(df)))


#%% Plot parameters
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

pal = ['#969696','#525252']  # grey, black
pal = sns.color_palette(pal)

pal4c = ['#253494','#2c7fb8','#41b6c4','#a1dab4']
pal4c = sns.color_palette(pal4c)

#%% eDNA analysis
print('\n- - - eDNA Samples - - -')

### Set BLOD to 0 for stats?
# Note: setting to NAN biases the data (excludes many samples where we know conc < lod)
df.loc[df.BLOD == 1,'eDNA'] = 0
df.loc[df.BLOD == 1,'log10eDNA'] = 0 


### Date variables
df['date'] = df['dt'].dt.date
df['year'] = df['dt'].dt.year
df['month'] = df['dt'].dt.month
df['year_month'] = df.year.astype(str) + '-' + df.month.astype(str)
df['week'] = df['dt'].dt.isocalendar().week

df['wet_season'] = 0  # dry season
df.loc[df.month.isin([10,11,12,1,2,3,4]),'wet_season'] = 1 # wet season
df['season'] = 'winter' # Dec-Feb
df.loc[df.month.isin([3,4,5]),'season'] = 'spring'
df.loc[df.month.isin([6,7,8]),'season'] = 'summer'
df.loc[df.month.isin([9,10,11]),'season'] = 'fall'

# Num. samples per day
n_per_day = df.reset_index()[df.reset_index().target=='trout'].groupby('date').count()['id']
n_per_day.value_counts()


### Separate out targets
for t in df.reset_index().target.unique():  
    print('\n' + t.upper())
    target = df.xs(t,level=1).reset_index().set_index('dt')
    print(target['eDNA'].describe())
    print('N BLOD - ' + str((target['BLOD']==1).sum()))
    print('N > 100 copies/mL - ' + str((target['eDNA']>100).sum()))

trout = df.xs('trout', level=1)
coho = df.xs('coho', level=1)
df_corr = pd.merge(trout.log10eDNA,coho.log10eDNA, how = 'inner', left_index=True,right_index=True).dropna()
stats.pearsonr(df_corr.iloc[:,0], df_corr.iloc[:,1])  
# If BLOD samples = 0 -> r = 0.299, p<0.01; if BLOD samples = np.nan -> r = 0.136, p < 0.05

### Boxplots by year/month, season
plt.figure(figsize=(10,4))  
sns.boxplot(x='year_month',y='log10eDNA', hue='target', data=df.reset_index(), palette=pal4c[0:3:2])

plt.figure()  
sns.boxplot(x='wet_season',y='log10eDNA', hue='target', data=df.reset_index(), palette=pal4c[0:3:2])
stats.mannwhitneyu(
    df.reset_index().loc[(df.reset_index().target=='coho') & (df.reset_index().wet_season==0),'log10eDNA'],
    df.reset_index().loc[(df.reset_index().target=='coho') & (df.reset_index().wet_season==1),'log10eDNA'])
stats.mannwhitneyu(
    df.reset_index().loc[(df.reset_index().target=='trout') & (df.reset_index().wet_season==0),'log10eDNA'],
    df.reset_index().loc[(df.reset_index().target=='trout') & (df.reset_index().wet_season==1),'log10eDNA'])
# Both sig. diff to p<0.01

plt.figure()  
sns.boxplot(x='season',y='log10eDNA', hue='target', data=df.reset_index(), palette=pal4c[0:3:2])

X = df.reset_index().groupby(['target','year','week']).mean()[['eDNA','log10eDNA']]
Xsd =  df.reset_index().astype(float, errors='ignore').groupby(['target','year','week']).std()[['eDNA','log10eDNA']]
#X = X.reset_index()

### Line plot by week or month
plt.figure()
X.xs('trout')['log10eDNA'].plot(marker='.', yerr=Xsd.xs('trout')['log10eDNA'])
X.xs('coho')['log10eDNA'].plot(marker='.', yerr=Xsd.xs('coho')['log10eDNA'])


#%% eDNA - Boxplots / Hist
#    eDNA dataframe -> logeDNA are for individual replicates
#    df dataframe -> logeDNA is mean of replicates
df_plot = df.reset_index()  # eDNA

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
sns.boxplot(x='target',y='log10eDNA', data = df_plot, width=.3, palette=[pal4c[0],pal4c[2]])
plt.xlabel('')
plt.ylabel('log$_{10}$(copies/mL)')
ylim = plt.ylim()

plt.subplot(1,2,2)
plt.hist(df_plot[df_plot.target=='coho']['log10eDNA'],histtype='step',
         orientation='horizontal', color=pal4c[0])
plt.hist(df_plot[df_plot.target=='trout']['log10eDNA'],histtype='step', 
          orientation='horizontal', color=pal4c[2])
#plt.xlabel('log$_{10}$(copies/μL)')
plt.ylim(ylim)

plt.tight_layout()
plt.legend(['coho','trout'], frameon=False, loc='upper right')

#%% eDNA Time Series (log)

### Trout/Coho TS 1:5 Dilutions
A = df.xs('trout',level=1)
A = A.sort_values('dt')
B = df.xs('coho',level=1)
B = B.sort_values('dt')

A.loc[A.BLOD == 1,'log10eDNA'] = 0  # Remove samples BLOQ
B.loc[B.BLOD == 1,'log10eDNA'] = 0

plt.figure(figsize=(10,4))
plt.plot(A['dt'],A['log10eDNA'],marker='.',ms=4, color=pal4c[2])
#plt.fill_between(A.dt, A.log10eDNA - A.log10eDNA_sd, A.log10eDNA + A.log10eDNA_sd,
#                 color=pal4c[0], alpha=0.25)

plt.plot(B['dt'],B['log10eDNA'],marker='.',ms=4, color=pal4c[0])
#plt.fill_between(B.dt, B.log10eDNA - B.log10eDNA_sd, B.log10eDNA + B.log10eDNA_sd,
#                 color=pal4c[2], alpha=0.25)

#plt.xlim(B.dt.iloc[0], B.dt.dropna().iloc[-1])  # Range of samples in hand
plt.xlim(ESP.index[0], ESP.index[-1])   # Range ESP was deployed
plt.ylabel('log$_{10}$(copies/mL)')
plt.legend(['trout', 'coho', 'stdev'], frameon=False)

ax = plt.gca()
ax.spines['left'].set_position(('outward', 8))
ax.spines['bottom'].set_position(('outward', 8))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()

#%% eDNA Time Series (linear)

### Trout/Coho TS 1:5 Dilutions
A = df.xs('trout',level=1)
A = A.sort_values('dt')
B = df.xs('coho',level=1)
B = B.sort_values('dt')

A.loc[A.BLOD == 1,'log10eDNA'] = 0  # Remove samples BLOQ
B.loc[B.BLOD == 1,'log10eDNA'] = 0

plt.figure(figsize=(10,4))
plt.plot(A['dt'],A['eDNA'],marker='.',ms=4, color=pal4c[2])
#plt.fill_between(A.dt, A.eDNA - A.eDNA_sd, A.eDNA + A.eDNA_sd,
 #                color=pal4c[0], alpha=0.25)

plt.plot(B['dt'],B['eDNA'],marker='.',ms=4, color=pal4c[0])
#plt.fill_between(B.dt, B.eDNA - B.eDNA_sd, B.eDNA + B.eDNA_sd,
#                 color=pal4c[2], alpha=0.25)

#plt.xlim(B.dt.iloc[0], B.dt.dropna().iloc[-1])  # Range of samples in hand
plt.xlim(ESP.index[0], ESP.index[-1])   # Range ESP was deployed
plt.ylabel('log$_{10}$(copies/mL)')
plt.legend(['trout', 'coho', 'stdev'], frameon=False)

ax = plt.gca()
ax.spines['left'].set_position(('outward', 8))
ax.spines['bottom'].set_position(('outward', 8))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()