#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Load Data
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

Stats and Plot of the ESP data

ESP Sampling logs
- Sample volumes/rates (distributions, time series)
- Time of day


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

folder = '../data/'  # Data folder

### Load Combined ESP logs
# Contains sampling times, volumes, ESP name
ESP = pd.read_csv(os.path.join(folder,'ESP_logs','ESP_logs_combined.csv'), 
                  parse_dates = ['sample_wake','sample_start','sample_mid','sample_end','date'], 
                  index_col=['sample_mid'])  # can also use sample_start, but probably little diff.

# ESP.dropna(inplace=True, subset=['sample_wake', 'sample_start', 'sample_end', 'sample_duration',
#        'vol_target', 'vol_actual', 'vol_diff',])  # Drop samples with no time or volume data 

dr = pd.date_range('2019-03-25', '2020-04-04')  # project dates
ESP = ESP[ESP.date.isin(dr)]  

ESP = ESP[~ESP.lab_field.isin(['lab','control', 'control '])]  # Keep deployed/field/nana; Drop control/lab samples
ESP.drop(ESP[ESP.id.isin(['SCr-181', 'SCr-286', 'SCr-479', 'SCr-549'])].index, inplace=True) # Drop duplicate (no qPCR IDs)

## Lists of days with sampling frequencies of 1x, 2x, and 3x per day
three_per_day = [d for d in ESP['date'].unique() if (ESP.groupby('date').count() == 3)['id'].loc[d]]
# morning, afternoon, and evening samples
mae = [d for d in three_per_day if (ESP.groupby('date').sum()['morn_midday_eve']==3).loc[d]] 
 
two_per_day = [d for d in ESP['date'].unique() if (ESP.groupby('date').count() == 2)['id'].loc[d]]
one_per_day = [d for d in ESP['date'].unique() if (ESP.groupby('date').count() == 1)['id'].loc[d]]

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


#%% ESP - Sample Volume / Rates / Time of Day
### Stats
print('\n- - - ESP Stats - - -')
print('Sampling Volumes (mL)')
print(ESP.vol_actual.describe())

print('\nDuration (min)')
print(ESP.sample_duration.describe())
# Note: 93% of samples took 60m or less to collect 

# print('\nAvg. Sampling Rate (mL/min)')
# print(ESP.sample_rate.describe())


### Deployment table
print('\nProject Period: ' + str(min(ESP.date)) + ' to ' + str(max(ESP.date)) + ' (' + str(len(dr)) + ' days)')
print('Total Days Deployed: ' + str(ESP.date.nunique()))
for e in ESP.ESP.unique():
    print(e + ' - ' + str(len(ESP.loc[ESP.ESP==e,'date'].unique())))    

ESP['date2'] = ESP.date
df_deployment = pd.concat([ESP.groupby('date2').first().groupby('deployment').first()['ESP'],
                            ESP.groupby('date2').first().groupby('deployment').first()['date'].rename('start_date'),
                            ESP.groupby('date2').first().groupby('deployment').last()['date'].rename('end_date'),
                            ESP.groupby('date2').first().groupby('deployment').count()['date'].rename('days')
                            ], axis=1)
df_deployment['gap_days'] = np.nan   # No. gap days prior to deployment where samples were not collected
for i in range(1,len(df_deployment)):
    gap = df_deployment.iloc[i]['start_date'] - df_deployment.iloc[i-1]['end_date']
    ind = df_deployment.index[i]
    df_deployment.loc[ind,'gap_days'] = gap.days - 1

## N samples during each deployment
df_deployment = pd.concat([df_deployment, ESP.groupby('deployment').count().date.rename('samples')], axis=1)
df_deployment.dropna(subset=['start_date'],inplace=True)
print(df_deployment)
df_deployment.to_csv(os.path.join(folder,'ESP_logs','deployments_summary.csv'))

print('\nDays missing samples within project range')
days_missing = [str(d.date()) for d in dr if d not in ESP.date.unique()]
print(days_missing)


### Time Series Plot
## Sample volume
plt.figure(figsize=(11,3))
plt.plot('vol_actual', data=ESP, color='k')
#plt.stem(ESP.index, ESP['vol_actual'], linefmt='k-', markerfmt=' ', basefmt=' ')
plt.plot([],ls=':',color='grey') # for legend
ax = plt.gca()
plt.ylabel('Total Volume (mL)')

# ## Sample Rate
ax2 = ESP['sample_rate'].plot(secondary_y=True, color='grey', ls=':')  # secondary axes
plt.ylabel('Average Rate (mL/min)', rotation=270, ha='center', va='baseline', rotation_mode='anchor')

## Fill background by ESP name
plt.sca(ax)  # Switch back to primary axes
ESPlist = list(ESP.ESP.unique())
x_ax = ax.lines[0].get_xdata()

for i in df_deployment.index:
    ESP_idx = ESPlist.index(df_deployment.loc[i,'ESP'])  # which ESP
    plt.fill_between(y1=3000, 
                      y2=-100, 
                      x=[df_deployment.loc[i,'start_date'], df_deployment.loc[i,'end_date'] + datetime.timedelta(days=1)], 
                      facecolor=pal4c[ESP_idx], 
                      edgecolor=None, 
                      alpha=0.25)

## remove fill for missing days
for i in days_missing:
    plt.fill_between(y1=3000, 
                      y2=-100, 
                      x= [pd.to_datetime(i), pd.to_datetime(i) + datetime.timedelta(days=1)], 
                      facecolor='w', 
                      edgecolor=None)

plt.title('Sampling Volumes/Rates', loc='left', pad=10)
plt.xlabel('')
plt.xticks(rotation = 0, ha='center')
plt.legend(['Total Volume',
            'Sampling Rate',  #'Target',
            ESPlist[0],ESPlist[1],ESPlist[2]],frameon=False,  
            bbox_to_anchor=(0,1.02,1,0.2), loc="lower right", borderaxespad=0, ncol=5)
leg = ax.get_legend()
leg.legendHandles[2].set_color(pal4c[0])
leg.legendHandles[3].set_color(pal4c[1])
leg.legendHandles[4].set_color(pal4c[2])

## Mark Hand sample
plt.scatter(pd.to_datetime('2/11/2020 07:00'), 1000, s=20,c='k',marker='*')

plt.ylim(-10,2050)
plt.xlim(x_ax[0],x_ax[-1])
plt.tight_layout()
plt.savefig(os.path.join(folder.replace('data','figures'),'ESP_volume_time_series.png'),dpi=300)


# ### Histograms
# # Volumes
# plt.figure(figsize=(4,4))

# plt.hist([ESP[ESP.ESP == ESPlist[0]].vol_actual,
#           ESP[ESP.ESP == ESPlist[1]].vol_actual,
#           ESP[ESP.ESP == ESPlist[2]].vol_actual], 
#           bins=20, histtype='barstacked', stacked=True, color=pal4c[0:3], alpha=0.3)
# plt.xlabel('Actual Volume (mL)')
# plt.ylabel('N')
# plt.xlim(0,2000)
# plt.legend(ESPlist, frameon=False, loc='upper right')
# plt.tight_layout()
# plt.savefig(os.path.join(folder.replace('data','figures'),'ESP_volume_histogram.png'),dpi=300)

# # # Duration
# # plt.figure(figsize=(4,4))

# # plt.hist([ESP[ESP.ESP == ESPlist[0]].sample_duration,
# #           ESP[ESP.ESP == ESPlist[1]].sample_duration,
# #           ESP[ESP.ESP == ESPlist[2]].sample_duration],
# #           bins=40, histtype='barstacked', stacked=True, color=pal4c[0:3], alpha=0.4)
# # plt.xlabel('Sampling Duration (min)')
# # plt.ylabel('N')
# # plt.yscale('log')
# # plt.xlim(0,250)
# # plt.legend(ESPlist, frameon=False, loc='upper right')
# # plt.tight_layout()

# # ESP
# plt.figure(figsize=(4,4))
# plt.hist(ESP.sample_duration, density=True, cumulative=True,
#          histtype='step', bins=100, color = 'k')
# plt.xlabel('Sampling Duration (min)')
# plt.ylabel('CDF')
# plt.xlim(0,685)
# plt.tight_layout()
# plt.savefig(os.path.join(folder.replace('data','figures'),'ESP_sample_duration_CDF.png'),dpi=300)


### Time of Day
print('\n- - Time of Day - -\n')
N = len(ESP)
print('# Samples: ' + str(N))

ESP['hour_frac'] = ESP.index.hour + ESP.index.minute/60

# Before 6a, 11a, 5p
print('\n% Samples Collected: \nBefore 06:00 PST: ' + str(round(100*len(ESP[ESP.index.hour < 6])/N,1)))
print('Before 11:00 PST: ' + str(round(100*len(ESP[ESP.index.hour < 12])/N,1)))
print('Before 17:00 PST: ' + str(round(100*len(ESP[ESP.index.hour < 18])/N,1)))

print('\nN per Time of Day \n(0-morn [<11], 1-mid [11-17], 2-eve [>17])')
temp = ESP.groupby('morn_midday_eve').count()['ESP']
temp.index = ['Morning','Midday','Evening']
print(temp)

plt.figure(figsize=(4,4))
plt.hist(ESP.hour_frac, bins=48, color='k')  # Sampling time distribution
plt.ylabel('N')
plt.xlabel('Hour of Day')
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
            labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])
plt.xlim(0,23)

plt.axvline(11,color='k', ls='--') #morn/adt divide
plt.axvline(17,color='k', ls='--') #aft/evening divide

plt.subplots_adjust(top=0.917, bottom=0.139, left=0.145, right=0.967, hspace=0.2, wspace=0.2)
plt.savefig(os.path.join(folder.replace('data','figures'),'ESP_time_of_day_histogram.png'),dpi=300)


### Sample per day Plot
ESP['sample_of_day'] = np.nan  # sample No. for each day
for d in ESP.date.unique():
    idx = ESP.loc[ESP.date==d].index.drop_duplicates()
    ESP.loc[idx,'sample_of_day'] = np.arange(1, len(idx)+1)

samples_of_day = ESP.groupby(['sample_of_day',
                              'date']).max()['hour_frac'].reset_index().pivot(index='date',
                                                                          columns='sample_of_day',
                                                                          values='hour_frac')
                                                                   
print('\nNumber of Samples Per Day: ')
print(samples_of_day.count())

print('\nMedian Sample Time by Sample of Day: ')
print(samples_of_day.median())
      
                                                                 
plt.figure(figsize=(10,5))
# plt.scatter(samples_of_day.index, samples_of_day[1], s=5, label='1st')
# plt.scatter(samples_of_day.index, samples_of_day[2], s=5, label='2nd')
# plt.scatter(samples_of_day.index, samples_of_day[3], s=5, label='3rd')
plt.bar(samples_of_day.index, samples_of_day[3],color=pal4c[0], label='3rd', align='edge')
plt.bar(samples_of_day.index, samples_of_day[2],color=pal4c[1], label='2nd', align='edge')
plt.bar(samples_of_day.index, samples_of_day[1],color=pal4c[2], label='1st', align='edge')
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim(0,24)
plt.ylabel('Hour of Day')
plt.yticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
            labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])

plt.legend(frameon=False, title='Sample of Day:')
plt.title('Sample Time by Day')

# Morn/Mid/Eve distinctions
plt.axhline(11,c='k',ls=':')
plt.axhline(17,c='k',ls=':')

# Fill for ESP
# for i in df_deployment.index:
#     ESP_idx = ESPlist.index(df_deployment.loc[i,'ESP'])  # which ESP
#     plt.fill_between(y1=30, 
#                       y2=-1, 
#                       x=[df_deployment.loc[i,'start_date'], df_deployment.loc[i,'end_date'] + datetime.timedelta(days=1)], 
#                       facecolor=pal4c[ESP_idx], 
#                       edgecolor=None, 
#                       alpha=0.25)

# ## remove fill for missing days
# for i in days_missing:
#     plt.fill_between(y1=30, 
#                       y2=-1, 
#                       x= [pd.to_datetime(i), pd.to_datetime(i) + datetime.timedelta(days=1)], 
#                       facecolor='w', 
#                       edgecolor=None)

plt.tight_layout()
