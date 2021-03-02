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
import os

folder = '../data/'  # Data folder

### Load Combined ESP logs
# Contains sampling times, volumes, ESP name
ESP = pd.read_csv(os.path.join(folder,'ESP_logs','ESP_logs_combined.csv'), 
                 parse_dates = ['sample_wake','sample_start','sample_mid','sample_end'], 
                 index_col=['sample_mid'])  # can also use sample_start, but probably little diff.

ESP.dropna(inplace=True, subset=['sample_wake', 'sample_start', 'sample_end', 'sample_duration',
       'vol_target', 'vol_actual', 'vol_diff',])  # Drop samples with not time or volume data 

ESP = ESP[~ESP.lab_field.isin(['lab','control', 'control '])]  # Keep deployed/field/nana; Drop control/lab samples

### Load eDNA Data
# Contains target, replicate #,dilution level, concentration (post-dilution factor)
eDNA = pd.read_csv(os.path.join(folder,'eDNA','qPCR_calculated.csv'))
eDNA.Ct = pd.to_numeric(eDNA.Ct, errors='coerce')

### Combine eDNA and ESP data
df = pd.merge(eDNA, ESP.reset_index(), how='left', on='id')

### Convert from copies/rxn to copies/mL filtered
df['eDNA'] = df.conc * 66.67 / df.vol_actual     # conversion from MBARI
df['logeDNA'] = np.log10(df.eDNA)

### Average Replicates (w/ Error est.)
# TODO: LOGIC FOR REPLICATES/DILUTIONS
# TODO: LOGIC FOR > 3 REPLICATES -> WHICH TO USE?
# Q: drop Undetermined from average, or use 1/2

df['n_replicates'] = 0
df['eDNA_mean'] = np.nan
df['eDNA_sd'] = np.nan    # error = stdev
df['logeDNA_mean'] = np.nan
df['logeDNA_sd'] = np.nan 
#eDNA_means = df.groupby(['id','target','dilution']).mean()  # Doesn't preserve all columns
#eDNA_stds = df.groupby(['id','target','dilution']).std()
print('Averaging replicates (N=' + str(len(eDNA)) + ') ...')
for i in df.id.unique():               # iterate through samples with sample IDs
    for t in df.target.unique():       # iterate through target names
        for d in df.dilution.unique(): # iterate through dilutions
            idx = (df.id == i) & (df.target == t) & (df.dilution == d)
            df.loc[idx, 'n_replicates'] = int(idx.sum())
            df.loc[idx, 'eDNA_mean'] = df.loc[idx,'eDNA'].mean()
            df.loc[idx, 'eDNA_sd'] = df.loc[idx,'eDNA'].std()
            df.loc[idx, 'logeDNA_mean'] = np.log10(df.loc[idx,'eDNA']).mean()
            df.loc[idx, 'logeDNA_sd'] = np.log10(df.loc[idx,'eDNA']).std()
            # could also set error = bootstrap samples
print('Done.')

df = df[~df[['id','target','dilution']].duplicated()] # drop duplicated rows so to keep means
df['dt'] = df['sample_mid']  # timestamp
df[['eDNA','logeDNA']] = df[['eDNA_mean','logeDNA_mean']]

df = df[['dt','id','target','dilution',
         'eDNA','eDNA_sd','logeDNA','logeDNA_sd','BLOD','BLOQ','n_replicates',
         'ESP','vol_actual','sample_duration','sample_rate','morn_midday_eve']]
df.set_index(['id','target','dilution'], inplace=True)
df.sort_values('dt', inplace=True)  # sort by date

### TODO: Daily mean
# T = df.xs(t,level=1).resample('d', on='dt').mean()['logeDNA'] 

### TODO: Save

#%% eDNA analysis
print('- - - eDNA Samples - - -')

### Separate out dilution means
trout = df.xs('trout',level=1)
trout1 = df.xs('trout',level=1).xs('1:1',level=1)  # trout 1:1 dilutions
trout5 = df.xs('trout',level=1).xs('1:5',level=1)  # 1:5 dilutions

coho = df.xs('coho',level=1)
coho1 = df.xs('coho',level=1).xs('1:1',level=1)  # coho 1:1 dilutions
coho5 = df.xs('coho',level=1).xs('1:5',level=1)  # 1:5 dilutions

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

pal4c = ['#253494','#2c7fb8','#41b6c4','#a1dab4'] # grey, black
pal4c = sns.color_palette(pal4c)

# #%% ESP - Sample Volume / Rates

# ### Stats
# print('\n- - - ESP Stats - - -')
# print('Sampling Volumes (mL)')
# print(ESP.vol_actual.describe())

# print('\nDuration (min)')
# print(ESP.sample_duration.describe())
# # Note: 93% of samples took 60m or less to collect 

# print('\nAvg. Sampling Rate (mL/min)')
# print(ESP.sample_rate.describe())      

# ### Time Series
# # Sample volume
# plt.figure(figsize=(10,4))
# plt.plot('vol_actual', data=ESP, color=pal[1])
# plt.plot([],ls=':',color=pal[0])
# #plt.plot('vol_target', data=ESP, ls=':', color=pal[0])
# ax = plt.gca()
# plt.ylabel('Volume (mL)')

# ax2 = ESP['sample_rate'].plot(secondary_y=True, color=pal[0], ls=':')  # secondary axes
# plt.ylabel('Average Rate (mL/min)', rotation=270, ha='center', va='baseline', rotation_mode='anchor')

# plt.sca(ax)  # Switch back to primary axes
# # Fill background by ESP name
# ESPlist = list(ESP.ESP.unique())
# x_ax = ax.lines[0].get_xdata()
# for i in range(1,len(ESP)-1):
#     ESP_idx = ESPlist.index(ESP.ESP.iloc[i])  # which ESP
#     plt.fill_between(y1=3000, y2=-100, x=x_ax[i-1:i+1], facecolor=pal4c[ESP_idx], edgecolor=None, alpha=0.25)

# plt.title('Sampling Volumes/Rates', loc='left', pad=10)
# plt.xlabel('')
# plt.xticks(rotation = 0, ha='center')

# plt.legend(['Total Volume',
#             'Sampling Rate',  #'Target',
#             ESPlist[0],ESPlist[1],ESPlist[2]],frameon=False,  
#            bbox_to_anchor=(0,1.02,1,0.2), loc="lower right", borderaxespad=0, ncol=5)
# leg = ax.get_legend()
# leg.legendHandles[2].set_color(pal4c[0])
# leg.legendHandles[3].set_color(pal4c[1])
# leg.legendHandles[4].set_color(pal4c[2])

# plt.ylim(-10,2050)
# plt.xlim(x_ax[0],x_ax[-1])
# plt.tight_layout()

# ### Histograms
# # Volumes
# plt.figure(figsize=(5,4))

# plt.hist([ESP[ESP.ESP == ESPlist[0]].vol_actual,
#           ESP[ESP.ESP == ESPlist[1]].vol_actual,
#           ESP[ESP.ESP == ESPlist[2]].vol_actual], 
#           bins=20, histtype='barstacked', stacked=True, color=pal4c[0:3], alpha=0.4)
# plt.xlabel('Actual Volume (mL)')
# plt.ylabel('Count')
# plt.xlim(0,2000)
# plt.legend(ESPlist, frameon=False, loc='upper right')
# plt.tight_layout()

# # Duration
# plt.figure(figsize=(5,4))

# plt.hist([ESP[ESP.ESP == ESPlist[0]].sample_duration,
#           ESP[ESP.ESP == ESPlist[1]].sample_duration,
#           ESP[ESP.ESP == ESPlist[2]].sample_duration],
#           bins=40, histtype='barstacked', stacked=True, color=pal4c[0:3], alpha=0.4)
# plt.xlabel('Sampling Duration (min)')
# plt.ylabel('Count')
# plt.yscale('log')
# plt.xlim(0,250)
# plt.legend(ESPlist, frameon=False, loc='upper right')
# plt.tight_layout()

# # ESP
# plt.figure(figsize=(5,4))
# plt.hist(ESP.sample_duration, density=True, cumulative=True,
#          histtype='step', bins=100, color = pal[1])
# plt.xlabel('Sampling Duration (min)')
# plt.ylabel('CDF')
# plt.xlim(0,685)
# plt.tight_layout()

# #%% Time of Day
# print('\n- - Time of Day - -\n')
# N = len(ESP)
# print('# Samples: ' + str(N))

# # Before 6a, 11a, 5p
# print('\n% Samples Collected: \nBefore 06:00 PST: ' + str(round(100*len(ESP[ESP.index.hour < 6])/N,1)))
# print('Before 11:00 PST: ' + str(round(100*len(ESP[ESP.index.hour < 12])/N,1)))
# print('Before 17:00 PST: ' + str(round(100*len(ESP[ESP.index.hour < 18])/N,1)))

# print('\nN per TOD Category (0-morn,1-mid,2-eve)')
# print(ESP.groupby('morn_midday_eve').count()['ESP'])

# plt.figure(figsize=(5,5))
# plt.hist(ESP.index.hour + ESP.index.minute/60, bins=48, color=pal[1])  # Sampling time distribution
# plt.ylabel('N')
# plt.xlabel('Sample Time Distribution \n(Hour of Day)')
# plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
#            labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])
# plt.xlim(0,23)

# plt.axvline(11,color='k', ls='--') #morn/adt divide
# plt.axvline(17,color='k', ls='--') #aft/evening divide

# plt.subplots_adjust(top=0.917, bottom=0.139, left=0.145, right=0.967, hspace=0.2, wspace=0.2)

#%% eDNA - Boxplots / Hist
#    eDNA dataframe -> logeDNA are for individual replicates
#    df dataframe -> logeDNA is mean of replicates
df_plot = df.reset_index()  # eDNA

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
sns.boxplot(x='target',y='logeDNA', data = df_plot, width=.3, palette=[pal4c[2],pal4c[0]])
plt.xlabel('')
plt.ylabel('log$_{10}$(copies/mL)')
ylim = plt.ylim()

plt.subplot(1,2,2)
plt.hist(df_plot[df_plot.target=='coho']['logeDNA'],histtype='step',
         orientation='horizontal', color=pal4c[2])
plt.hist(df_plot[df_plot.target=='trout']['logeDNA'],histtype='step', 
          orientation='horizontal', color=pal4c[0])
#plt.xlabel('log$_{10}$(copies/μL)')
plt.ylim(ylim)

plt.tight_layout()
plt.legend(['coho','trout'], frameon=False, loc='lower right')

#%% eDNA Time Series

### Trout/Coho TS 1:5 Dilutions
A = trout5
A = A.sort_values('dt')
B = coho5
B = B.sort_values('dt')

A = A[A.BLOQ == 0]  # Remove samples BLOQ
B = B[B.BLOQ == 0]

plt.figure(figsize=(10,4))
plt.plot(A['dt'],A['logeDNA'],marker='.',ms=4, color=pal4c[0])
plt.fill_between(A.dt, A.logeDNA - A.logeDNA_sd, A.logeDNA + A.logeDNA_sd,
                 color=pal4c[0], alpha=0.25)

plt.plot(B['dt'],B['logeDNA'],marker='.',ms=4, color=pal4c[2])
plt.fill_between(B.dt, B.logeDNA - B.logeDNA_sd, B.logeDNA + B.logeDNA_sd,
                 color=pal4c[2], alpha=0.25)

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
