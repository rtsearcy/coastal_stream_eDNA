#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Load Data
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

Stats and Plot of the ESP and eDNA data

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
from eDNA_corr import eDNA_corr

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
# Contains target, replicate #,dilution level, concentration (post-dilution factor)
eDNA = pd.read_csv(os.path.join(folder,'eDNA','eDNA.csv'), parse_dates=['dt'])


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
pal = ['#969696','#525252']  # grey, black
pal = sns.color_palette(pal)

pal2c = ['#ca0020', '#f4a582'] # salmon colors
pal2c = sns.color_palette(pal2c)


pal4c = ['#253494','#2c7fb8','#41b6c4','#a1dab4'] # 4 color blue tone
pal4c = sns.color_palette(pal4c)

def caps_off(axx): ## Turn off caps on boxplots
    lines = axx.lines
    for i in range(0, int(len(lines)/6)):
        lines[(i*6)+2].set_color('none')
        lines[(i*6)+3].set_color('none')

def flier_shape(axx, shape='.'):  ## Set flier shape on boxplots
    lines = axx.lines
    for i in range(0, int(len(lines)/6)):
        lines[(i*6)+5].set_marker(shape)

def plot_spines(axx): # Offset position, Hide the right and top spines
    axx.spines['left'].set_position(('outward', 8))
    axx.spines['bottom'].set_position(('outward', 8))
    
    axx.spines['right'].set_visible(False)
    axx.spines['top'].set_visible(False)


#%% ESP - Sample Volume / Rates
### Stats
print('\n- - - ESP Stats - - -')
print('Sampling Volumes (mL)')
print(ESP.vol_actual.describe())

print('\nDuration (min)')
print(ESP.sample_duration.describe())
# Note: 93% of samples took 60m or less to collect 

print('\nAvg. Sampling Rate (mL/min)')
print(ESP.sample_rate.describe())      


### Time Series
# Sample volume
plt.figure(figsize=(10,4))
plt.plot('vol_actual', data=ESP, color=pal[1])
plt.plot([],ls=':',color=pal[0])
#plt.plot('vol_target', data=ESP, ls=':', color=pal[0])
ax = plt.gca()
plt.ylabel('Volume (mL)')

ax2 = ESP['sample_rate'].plot(secondary_y=True, color=pal[0], ls=':')  # secondary axes
plt.ylabel('Average Rate (mL/min)', rotation=270, ha='center', va='baseline', rotation_mode='anchor')

plt.sca(ax)  # Switch back to primary axes
# Fill background by ESP name
ESPlist = list(ESP.ESP.unique())
x_ax = ax.lines[0].get_xdata()
for i in range(1,len(ESP)-1):
    ESP_idx = ESPlist.index(ESP.ESP.iloc[i])  # which ESP
    plt.fill_between(y1=3000, y2=-100, x=x_ax[i-1:i+1], facecolor=pal4c[ESP_idx], edgecolor=None, alpha=0.25)

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

plt.ylim(-10,2050)
plt.xlim(x_ax[0],x_ax[-1])
plt.tight_layout()

### Histograms
# Volumes
plt.figure(figsize=(5,4))

plt.hist([ESP[ESP.ESP == ESPlist[0]].vol_actual,
          ESP[ESP.ESP == ESPlist[1]].vol_actual,
          ESP[ESP.ESP == ESPlist[2]].vol_actual], 
          bins=20, histtype='barstacked', stacked=True, color=pal4c[0:3], alpha=0.4)
plt.xlabel('Actual Volume (mL)')
plt.ylabel('Count')
plt.xlim(0,2000)
plt.legend(ESPlist, frameon=False, loc='upper right')
plt.tight_layout()

# Duration
plt.figure(figsize=(5,4))

plt.hist([ESP[ESP.ESP == ESPlist[0]].sample_duration,
          ESP[ESP.ESP == ESPlist[1]].sample_duration,
          ESP[ESP.ESP == ESPlist[2]].sample_duration],
          bins=40, histtype='barstacked', stacked=True, color=pal4c[0:3], alpha=0.4)
plt.xlabel('Sampling Duration (min)')
plt.ylabel('Count')
plt.yscale('log')
plt.xlim(0,250)
plt.legend(ESPlist, frameon=False, loc='upper right')
plt.tight_layout()

# ESP
plt.figure(figsize=(5,4))
plt.hist(ESP.sample_duration, density=True, cumulative=True,
         histtype='step', bins=100, color = pal[1])
plt.xlabel('Sampling Duration (min)')
plt.ylabel('CDF')
plt.xlim(0,685)
plt.tight_layout()

#%% Time of Day
print('\n- - Time of Day - -\n')
N = len(ESP)
print('# Samples: ' + str(N))

# Before 6a, 11a, 5p
print('\n% Samples Collected: \nBefore 06:00 PST: ' + str(round(100*len(ESP[ESP.index.hour < 6])/N,1)))
print('Before 11:00 PST: ' + str(round(100*len(ESP[ESP.index.hour < 12])/N,1)))
print('Before 17:00 PST: ' + str(round(100*len(ESP[ESP.index.hour < 18])/N,1)))

print('\nN per TOD Category (0-morn,1-mid,2-eve)')
print(ESP.groupby('morn_midday_eve').count()['ESP'])

plt.figure(figsize=(5,5))
plt.hist(ESP.index.hour + ESP.index.minute/60, bins=48, color=pal[1])  # Sampling time distribution
plt.ylabel('N')
plt.xlabel('Sample Time Distribution \n(Hour of Day)')
plt.xticks(ticks= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
           labels = [0,'',2,'',4,'',6,'',8,'',10,'',12,'',14,'',16,'',18,'',20,'',22,''])
plt.xlim(0,23)

plt.axvline(11,color='k', ls='--') #morn/adt divide
plt.axvline(17,color='k', ls='--') #aft/evening divide

plt.subplots_adjust(top=0.917, bottom=0.139, left=0.145, right=0.967, hspace=0.2, wspace=0.2)


#%% eDNA - Stats
print('\n- - - eDNA Samples - - -')

### Set BLOD to 0 for stats?
# Note: setting to NAN biases the data (excludes many samples where we know conc < lod)
eDNA.loc[eDNA.BLOD == 1,'eDNA'] = 0
eDNA.loc[eDNA.BLOD == 1,'log10eDNA'] = 0 

### Separate out targets
for t in eDNA.target.unique():  
    print('\n' + t.upper())
    target = eDNA[eDNA.target==t] #.set_index('dt')
    print(target['eDNA'].describe())
    print('N BLOD - ' + str((target['BLOD']==1).sum()))
    print('N > 100 copies/mL - ' + str((target['eDNA']>100).sum()))
    
    ## Num. samples per day
    n_per_day = eDNA[eDNA.target==t].groupby('date').count()['id']
    print('\nSamples per day / # Days')
    print(n_per_day.value_counts())
    
    ## Differences between wet and dry season?
    print('\nDifferences between wet and dry season?')
    print(stats.mannwhitneyu(
        eDNA.loc[(eDNA.target==t) & (eDNA.wet_season==0),'log10eDNA'],
        eDNA.loc[(eDNA.target==t) & (eDNA.wet_season==1),'log10eDNA']))

trout = eDNA[eDNA.target=='trout'].set_index('id').sort_values('dt')
coho = eDNA[eDNA.target=='coho'].set_index('id').sort_values('dt')


### Correlation between signals
print('\nCorrelation between Trout and Coho signals')
print(eDNA_corr(trout.log10eDNA,coho.log10eDNA, corr_type='spearman'))


### Autocorrelation  
T = trout.reset_index().set_index('dt').resample('D').mean()['log10eDNA']
T = T.interpolate('quadratic')
C = coho.reset_index().set_index('dt').resample('D').mean()['log10eDNA']
C = C.interpolate('quadratic')

rho = acf(T, nlags=100, fft=False)
#rho = pacf(T, nlags=100)
plt.stem(range(0,len(rho)), rho, linefmt='k-', markerfmt=' ', basefmt='k-')
plt.axhline(1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)
plt.axhline(-1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)


### Spectrogram
data = T
f, P = signal.welch(data, fs=1, 
                            window='hamming', # boxcar, hann, hamming, 
                            nfft=None,
                            nperseg=len(data) // 2,
                            noverlap=None,
                            detrend='constant', # False, 'constant'
                            scaling='density')  # 'density' [conc^2/Hz] or 'spectrum' [conc^2]
   
# could also use signal.periodogram
# f [1/day] / P [log10 conc^2*day] or [log10 conc^2]
plt.figure(figsize=(6,4))
plt.plot(f,P, color='k',lw=1)
#plt.semilogx(np.flip(1/f),P, color=pal4c[c],lw=1)
#c+=1

#plt.xscale('log')
#plt.xlim(f[1],f[-1])
ax = plt.gca()
plt.xlabel('f [1/day]')
plt.ylabel(r'P$_{xx}$ [$(log_{10}$copies/mL)$^2$*day]')

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
#plt.xscale('log')
def tick_function(X):
    V = 1/(X)
    return ["%.3f" % z for z in V]
ax2.set_xticklabels(tick_function(ax.get_xticks()))
ax2.set_xlabel(r"T [day]")

plt.sca(ax)
#plt.legend(list(df.columns), frameon=False)
plt.tight_layout()

#%% eDNA - Boxplots / Histograms
df_plot = eDNA.reset_index()  # eDNA

### All eDNA Data - Boxplot and Histogram
plt.figure(figsize=(6,4))
plt.subplot(1,2,1)
sns.boxplot(x='target',y='log10eDNA', data = df_plot, width=.5, palette=[pal4c[0],pal4c[2]])
plt.xlabel('')
plt.ylabel('log$_{10}$(copies/mL)')
ylim = plt.ylim()
caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.subplot(1,2,2)
plt.hist(df_plot[df_plot.target=='coho']['log10eDNA'],histtype='step',
         orientation='horizontal', color=pal4c[0])
plt.hist(df_plot[df_plot.target=='trout']['log10eDNA'],histtype='step', 
          orientation='horizontal', color=pal4c[2])
#plt.xlabel('log$_{10}$(copies/μL)')
plt.ylim(ylim)

plt.legend(['coho','trout'], frameon=False, loc='upper right')
plot_spines(plt.gca())
plt.gca().spines['left'].set_position(('outward', 0))

plt.tight_layout()


### Boxplots by year/month, season
plt.figure(figsize=(10,4))  
sns.boxplot(x='year_month',y='log10eDNA', hue='target', data=eDNA, palette=pal4c[0:3:2])
plt.xlabel('')
plt.ylabel('log$_{10}$(copies/mL)')


plt.legend(['coho','trout'], frameon=False, loc='upper left')
leg = plt.gca().get_legend()
leg.legendHandles[0].set_color(pal4c[0]) # coho
leg.legendHandles[1].set_color(pal4c[2]) # trout

caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.tight_layout()


plt.figure(figsize=(4,4))  
sns.boxplot(x='wet_season',y='log10eDNA', hue='target', data=eDNA,  width=.6, palette=pal4c[0:3:2])
plt.xlabel('')
plt.xticks(ticks=[0,1], labels=['Dry Season','Wet Season'])
plt.ylabel('log$_{10}$(copies/mL)')

plt.legend(['coho','trout'], frameon=False, loc='upper left')
leg = plt.gca().get_legend()
leg.legendHandles[0].set_color(pal4c[0]) # coho
leg.legendHandles[1].set_color(pal4c[2]) # trout

caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.tight_layout()


plt.figure(figsize=(6,4))  
sns.boxplot(x='season',y='log10eDNA', hue='target', data=eDNA, palette=pal4c[0:3:2])
plt.xlabel('')
plt.ylabel('log$_{10}$(copies/mL)')

plt.legend(['coho','trout'], frameon=False, loc='upper right')
leg = plt.gca().get_legend()
leg.legendHandles[0].set_color(pal4c[0]) # coho
leg.legendHandles[1].set_color(pal4c[2]) # trout

caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.tight_layout()


### Line plot of data binned by week or month
X = eDNA.groupby(['target','year','week']).mean()[['eDNA','log10eDNA']]
Xsd =  eDNA.astype(float, errors='ignore').groupby(['target','year','week']).std()[['eDNA','log10eDNA']]
#X = X.reset_index()

plt.figure(figsize=(10,4))
X.xs('coho')['log10eDNA'].plot(marker='.', yerr=Xsd.xs('coho')['log10eDNA'], color = pal4c[0])
X.xs('trout')['log10eDNA'].plot(marker='.', yerr=Xsd.xs('trout')['log10eDNA'], color = pal4c[2])
plt.legend(['coho','trout'], frameon=False, loc='upper left')

plot_spines(plt.gca())
plt.tight_layout()

#%% eDNA Time Series (log)

### Trout/Coho TS 
A = trout
A = A.sort_values('dt')
B = coho
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

plot_spines(plt.gca())

plt.tight_layout()

#%% eDNA Time Series (linear)

### Trout/Coho TS 1:5 Dilutions
A = trout
A = A.sort_values('dt')
B = coho
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
plt.ylabel('copies/mL')
plt.legend(['trout', 'coho', 'stdev'], frameon=False)

plot_spines(plt.gca())

plt.tight_layout()
