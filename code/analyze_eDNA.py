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
import datetime
from eDNA_corr import eDNA_corr

folder = '../data/'  # Data folder

# ### Load Combined ESP logs
# FOr TOD analyses
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


### Load eDNA Data
# Contains target, replicate #,dilution level, concentration (post-dilution factor)
eDNA = pd.read_csv(os.path.join(folder,'eDNA','eDNA.csv'), parse_dates=['dt','date'])


### Load Hatchery Data
# For plots
# Adult/Juvenile Steelhead and Coho counts
hatch = pd.read_csv(os.path.join(folder,'NOAA_data', 'hatchery_releases.csv'), 
                 parse_dates = ['date'], index_col=['date'])


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


#%% eDNA - Stats
print('\n- - - eDNA Samples - - -')

### Set BLOD to 0 for stats?
# Note: setting to NAN biases the data (excludes many samples where we know conc < lod)
#eDNA.loc[eDNA.BLOD == 1,'eDNA'] = 0
#eDNA.loc[eDNA.BLOD == 1,'log10eDNA'] = 0 

### Separate out targets
for t in eDNA.target.unique():  
    print('\n' + t.upper())
    target = eDNA[eDNA.target==t] #.set_index('dt')
    print(target['eDNA'].describe())
    print('N BLOD - ' + str((target['BLOD']==1).sum()))
    print('N > 100 copies/mL - ' + str((target['eDNA']>100).sum()))
    
    # ## Num. samples per day
    # n_per_day = eDNA[eDNA.target==t].groupby('date').count()['id']
    # print('\nSamples per day / # Days')
    # print(n_per_day.value_counts())
    
    ## Differences between time of day?
    df_mae = eDNA[eDNA.date.isin(mae)]
    print('\nDifference between time of day?')
    print('N=' + str(len(df_mae[df_mae.target==t])) + ' days with morn/aft/eve samples')
    morn = eDNA.loc[(eDNA.target==t) & (eDNA.morn_midday_eve==0),'log10eDNA']
    midd = eDNA.loc[(eDNA.target==t) & (eDNA.morn_midday_eve==1),'log10eDNA']
    eve = eDNA.loc[(eDNA.target==t) & (eDNA.morn_midday_eve==2),'log10eDNA']
    print('Median (Morning/Midday/Evening): ' + str(round(morn.median(),3)) + 
          '/' + str(round(midd.median(),3)) + 
          '/' + str(round(eve.median(),3)))
    print(stats.kruskal(morn,midd,eve))
    
    # Correlation between morn/midday/eve
    print('\nTime of day correlations')
    mae_corr = pd.concat([
        df_mae[(df_mae.target==t) & (df_mae.morn_midday_eve==0)].set_index('date').sort_index()['log10eDNA'],
        df_mae[(df_mae.target==t) & (df_mae.morn_midday_eve==1)].set_index('date').sort_index()['log10eDNA'],
        df_mae[(df_mae.target==t) & (df_mae.morn_midday_eve==2)].set_index('date').sort_index()['log10eDNA']], axis=1)
    mae_corr.columns = ['morning', 'midday','evening']
    print(mae_corr.corr())
    
    ## Differences between wet and dry season?
    print('\nDifference between Dry and Wet seasons?')
    dry = eDNA.loc[(eDNA.target==t) & (eDNA.wet_season==0),'log10eDNA']
    wet = eDNA.loc[(eDNA.target==t) & (eDNA.wet_season==1),'log10eDNA']
    print('N (Dry/Wet): ' + str(len(dry)) + '/' + str(len(wet)))
    print('Median (Dry/Wet): ' + str(round(dry.median(),3)) + '/' + str(round(wet.median(),3)))
    print(stats.mannwhitneyu(dry,wet))
    
    ## Differences between season?
    print('\nDifference between season?')
    spring = eDNA.loc[(eDNA.target==t) & (eDNA.season=='spring'),'log10eDNA']
    summer = eDNA.loc[(eDNA.target==t) & (eDNA.season=='summer'),'log10eDNA']
    fall = eDNA.loc[(eDNA.target==t) & (eDNA.season=='fall'),'log10eDNA']
    winter = eDNA.loc[(eDNA.target==t) & (eDNA.season=='winter'),'log10eDNA']
    print('Median (Sp/Su/Fa/Wi): ' + str(round(spring.median(),3)) + 
          '/' + str(round(summer.median(),3)) + 
          '/' + str(round(fall.median(),3)) + 
          '/' + str(round(winter.median(),3)))
    print(stats.kruskal(spring,summer,fall,winter))

trout = eDNA[eDNA.target=='trout'].set_index('id').sort_values('dt')
coho = eDNA[eDNA.target=='coho'].set_index('id').sort_values('dt')

### % detect / BLOD / above LOD

## Overall
print('\n\n% non-detect (0 replicates amplified)')
print(100*eDNA[(eDNA.n_amplified==0)].groupby(['target','BLOD']).count()['id'] / eDNA.groupby(['target']).count()['id'])
print('\n% detect but BLOD / above LOD')
print(100*eDNA[(eDNA.n_amplified>0)].groupby(['target','BLOD']).count()['id'] / eDNA.groupby(['target']).count()['id'])

## By TOD
# Is there a differential amplification / quantification rate by TOD?
temp = eDNA[eDNA.date.isin(mae)]
len(mae) # 97 days each with morn, midday, eve samples
temp[(temp.n_amplified>0)].groupby(['target','morn_midday_eve']).count()['id'] # No
temp[(temp.BLOD==0)].groupby(['target','morn_midday_eve']).count()['id']  # Maybe for trout
 

## BY hour of day
# What percentage of samples collected in each hour bin were detected / BLOD / non-detect
temp = pd.concat([eDNA[eDNA.n_amplified>0].groupby(['hour','target']).count()['id'] / eDNA.groupby(['hour','target']).count()['id'].rename('percent detected'),
                  eDNA.groupby(['hour','target']).count()['id'].rename('total samples')], axis=1)


## visualize with a bar chart? line plot? barbell plot between BLOD and detects
## log transform / adjust sizes to account for skewed N samples by hour


### Correlations

## between target's overall signals
print('\n\nCorrelation between Trout and Coho signals')
print('Overall')
print(eDNA_corr(trout.log10eDNA,coho.log10eDNA))
print(eDNA_corr(trout.log10eDNA,coho.log10eDNA, corr_type='spearman'))

## Difference in correlation during different conditions?
for s in eDNA.season.unique():
    print(s.capitalize() + ' samples only')
    print(eDNA_corr(trout[trout.season==s], coho[coho.season==s], x_col='log10eDNA', y_col='log10eDNA'))


### Autocorrelation  
T = trout.reset_index().set_index('dt').resample('D').mean()['log10eDNA']
T = T.interpolate('quadratic')
C = coho.reset_index().set_index('dt').resample('D').mean()['log10eDNA']
C = C.interpolate('quadratic')

plt.figure(figsize=(8,4))

plt.subplot(2,2,1)  # Trout autocorrelation
rhoT = acf(T, nlags=90, fft=False)
plt.stem(range(0,len(rhoT)), rhoT, linefmt='k-', markerfmt=' ', basefmt='k-')
plt.axhline(1.96/(len(T)**.5), ls='--', color='grey', alpha=0.7)
plt.axhline(-1.96/(len(T)**.5), ls='--', color='grey', alpha=0.7)
plt.title('Autocorrelation')
plt.ylabel('TROUT')
plot_spines(plt.gca(), offset=4)

plt.subplot(2,2,2)  # Trout partial
rhoT = pacf(T, nlags=90)
plt.stem(range(0,len(rhoT)), rhoT, linefmt='k-', markerfmt=' ', basefmt='k-')
plt.axhline(1.96/(len(T)**.5), ls='--', color='grey', alpha=0.7)
plt.axhline(-1.96/(len(T)**.5), ls='--', color='grey', alpha=0.7)
plt.title('Partial Autocorrelation')
plot_spines(plt.gca(), offset=4)

plt.subplot(2,2,3)  # Coho autocorrelation
rhoT = acf(C, nlags=90, fft=False)
plt.stem(range(0,len(rhoT)), rhoT, linefmt='k-', markerfmt=' ', basefmt='k-')
plt.axhline(1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)
plt.axhline(-1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)
plt.ylabel('COHO')
plot_spines(plt.gca(), offset=4)

plt.subplot(2,2,4)  # Coho partial
rhoT = pacf(C, nlags=90)
plt.stem(range(0,len(rhoT)), rhoT, linefmt='k-', markerfmt=' ', basefmt='k-')
plt.axhline(1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)
plt.axhline(-1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)
plot_spines(plt.gca(), offset=4)

plt.tight_layout()
plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_autocorrelation.png'),dpi=300)

### Spectra
plt.figure(figsize=(4,4))
data = [C,T]
col = [pal[0],pal[1]]
for i in range(0,len(data)):
    nperseg = len(data[i]) // 4
    f, P = signal.welch(data[i],fs=1,                      # could also use signal.periodogram
                                window='hamming',          # boxcar, hann, hamming, 
                                nfft=None,
                                nperseg=nperseg,           # N data points per segement
                                noverlap=nperseg//2,       # N data points overlapping (50% is common)
                                detrend='constant',        # remove trend, False, 'constant'
                                scaling='density')         # 'density' [conc^2/Hz] or 'spectrum' [conc^2]
    # f [1/day] / P [log10conc^2*day] or [log10 conc^2]

    plt.plot(f,P, color=col[i],lw=1.5)
    
    # Confidence intervals
    alpha = 0.05
    DOF = 5 * len(data[i]) // nperseg                      # Degrees of Freedom, from Emery and Thomson for Hamming window
    chi2_U = stats.chi2.ppf((alpha/2), DOF)
    chi2_L = stats.chi2.ppf((1-alpha/2), DOF)              # Chi-squared for DOF and alpha      
    upperCI = (DOF/chi2_U) * P
    lowerCI = (DOF/chi2_L) * P
    
    #plt.plot(f,upperCI, color=col[i],lw=1, ls=':')         # Upper CI
    #plt.plot(f,lowerCI, color=col[i],lw=1, ls=':')         # Lower CI
    
    plt.fill_between(y1=P, y2=upperCI, x=f, facecolor=col[i], edgecolor=None, alpha=0.2)
    plt.fill_between(y1=lowerCI, y2=P, x=f, facecolor=col[i], edgecolor=None, alpha=0.2)
    
#plt.semilogx(np.flip(1/f),P, color=pal4c[c],lw=1)
#plt.xlim(f[1],f[-1])
ax = plt.gca()
plt.xlabel('f [1/day]')
plt.ylabel(r'P$_{xx}$ [$(log_{10}$copies/mL)$^2$*day]')

# ax2 = ax.twiny()
# ax2.set_xlim(ax.get_xlim())
# #plt.xscale('log')
# def tick_function(X):
#     V = 1/(X)
#     return ["%.3f" % z for z in V]
# ax2.set_xticklabels(tick_function(ax.get_xticks()))
# ax2.set_xlabel(r"T [day]")

plt.sca(ax)
plt.xscale('log')
plt.yscale('log')
plt.legend(['coho','trout'], frameon=False)
plot_spines(ax)
plt.tight_layout()
plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_spectra.png'),dpi=300)


#%% eDNA - Boxplots / Histograms
df_plot = eDNA.reset_index()  # eDNA

### Set BLOD to 0 for plots?
df_plot.loc[df_plot.BLOD == 1,'eDNA'] = 0
df_plot.loc[df_plot.BLOD == 1,'log10eDNA'] = 0 


### All eDNA Data - Boxplot and Histogram
plt.figure(figsize=(6,4))
plt.subplot(1,2,1)
sns.boxplot(x='target',y='log10eDNA', data = df_plot, width=.5, palette=[pal[0],pal[1]])
plt.xlabel('')
plt.ylabel('log$_{10}$(copies/mL)')
ylim = plt.ylim()
caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.subplot(1,2,2)
plt.hist(df_plot[df_plot.target=='coho']['log10eDNA'],histtype='step',
         orientation='horizontal', color=pal[0])
plt.hist(df_plot[df_plot.target=='trout']['log10eDNA'],histtype='step', 
          orientation='horizontal', color=pal[1])
#plt.xlabel('log$_{10}$(copies/μL)')
plt.ylim(ylim)

plt.legend(['coho','trout'], frameon=False, loc='upper right')
plot_spines(plt.gca())
plt.gca().spines['left'].set_position(('outward', 0))

plt.tight_layout()


### Boxplots by time, year/month, season

## Morning, afternoon evening
df_mae = df_plot[df_plot.date.isin(mae)]
plt.figure(figsize=(6,4))  
sns.boxplot(x='morn_midday_eve',y='log10eDNA', hue='target', data=df_mae, palette=pal)
plt.xlabel('')
plt.ylabel('log$_{10}$(copies/mL)')

plt.xticks(ticks=[0,1,2], labels=['Morning','Midday','Evening'])

plt.legend(['coho','trout'], frameon=False, loc='upper left')
leg = plt.gca().get_legend()
leg.legendHandles[0].set_color(pal[0]) # coho
leg.legendHandles[1].set_color(pal[1]) # trout

caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.tight_layout()

## Year and Month
plt.figure(figsize=(10,4))  
sns.boxplot(x='year_month',y='log10eDNA', hue='target', data=df_plot, palette=pal)
plt.xlabel('')
plt.ylabel('log$_{10}$(copies/mL)')

plt.legend(['coho','trout'], frameon=False, loc='upper left')
leg = plt.gca().get_legend()
leg.legendHandles[0].set_color(pal[0]) # coho
leg.legendHandles[1].set_color(pal[1]) # trout

caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.tight_layout()

## Wet Season
plt.figure(figsize=(4,4))  
sns.boxplot(x='wet_season',y='log10eDNA', hue='target', data=df_plot,  width=.6, palette=pal)
plt.xlabel('')
plt.xticks(ticks=[0,1], labels=['Dry Season','Wet Season'])
plt.ylabel('log$_{10}$(copies/mL)')

plt.legend(['coho','trout'], frameon=False, loc='upper left')
leg = plt.gca().get_legend()
leg.legendHandles[0].set_color(pal[0]) # coho
leg.legendHandles[1].set_color(pal[1]) # trout

caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.tight_layout()

## Season
plt.figure(figsize=(6,4))  
sns.boxplot(x='season',y='log10eDNA', hue='target', data=df_plot, palette=pal)
plt.xlabel('')
plt.ylabel('log$_{10}$(copies/mL)')

plt.legend(['coho','trout'], frameon=False, loc='upper right')
leg = plt.gca().get_legend()
leg.legendHandles[0].set_color(pal[0]) # coho
leg.legendHandles[1].set_color(pal[1]) # trout

caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.tight_layout()


### Line plot of data binned by week or month
X = df_plot.groupby(['target','year','week']).mean()[['eDNA','log10eDNA']]
Xsd =  df_plot.astype(float, errors='ignore').groupby(['target','year','week']).std()[['eDNA','log10eDNA']]
#X = X.reset_index()

plt.figure(figsize=(10,4))
X.xs('coho')['log10eDNA'].plot(marker='.', yerr=Xsd.xs('coho')['log10eDNA'], color = pal[0])
X.xs('trout')['log10eDNA'].plot(marker='.', yerr=Xsd.xs('trout')['log10eDNA'], color = pal[1])
plt.legend(['coho','trout'], frameon=False, loc='upper left')

plot_spines(plt.gca())
plt.tight_layout()

#%% eDNA Time Series (log, STEM)

### Trout/Coho TS 
df_plot = eDNA.reset_index()  # eDNA

## All values
A = df_plot[df_plot.target=='trout'].set_index('dt').sort_index()
B = df_plot[df_plot.target=='coho'].set_index('dt').sort_index()

## Daily mean
#A = df_plot[df_plot.target=='trout'].groupby('date').mean().sort_index()
#B = df_plot[df_plot.target=='coho'].groupby('date').mean().sort_index()

# Rolling mean
Arm = A['log10eDNA'].rolling(window=7,center=True).mean()
Brm = B['log10eDNA'].rolling(window=7,center=True).mean()

A.loc[A.BLOD == 1,'log10eDNA'] = 0  # Remove samples BLOQ
B.loc[B.BLOD == 1,'log10eDNA'] = 0

plt.figure(figsize=(10,5))
plt.subplot(211) # coho
plt.bar(B.index,B.log10eDNA + 0.5, bottom = -.5, color=pal[0])
#plt.axhline(0,color=pal[0])
plt.scatter(B.index,B.log10eDNA, color=pal[0], s=5)
plt.plot(Brm,color='k')

## Hatchery releases
plt.scatter(hatch.index.unique(), 3*np.ones(len(hatch.index.unique())),s=18,color='k',marker='$H$') # 'v'

plt.ylim(-.05, 1.05*df_plot.log10eDNA.max())
plt.xlim(ESP.index[0], ESP.index[-1])   # Range ESP was deployed
plt.gca().axes.xaxis.set_ticklabels([])

plt.text(ESP.index[10],.95*plt.ylim()[1], 'COHO')
plt.ylabel('log$_{10}$(copies/mL)')
#plt.legend(['trout', 'coho', 'hatchery release'], frameon=False)
plot_spines(plt.gca(), offset=0)


plt.subplot(212)  # trout
plt.bar(A.index, A.log10eDNA + .5, bottom=-.5, color=pal[1])
#plt.axhline(0,color=pal[1])
plt.scatter(A.index,A.log10eDNA, color=pal[1], s=5)
plt.plot(Arm,color='k')
plt.ylim(-.05, 1.05*df_plot.log10eDNA.max())
plt.xlim(ESP.index[0], ESP.index[-1])   # Range ESP was deployed

plt.text(ESP.index[10],.95*plt.ylim()[1], 'TROUT')
plt.ylabel('log$_{10}$(copies/mL)')
#plt.legend(['trout', 'rolling mean'], frameon=False)
plot_spines(plt.gca(), offset=0)

## Invert?
plt.gca().invert_yaxis()
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['bottom'].set_visible(False) 
plt.gca().xaxis.tick_top()

plt.tight_layout()
plt.subplots_adjust(top=0.961,bottom=0.078,left=0.057,right=0.984,hspace=0.16,wspace=0.2)

plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_time_series_log_stem.png'),dpi=300)


#%% eDNA Time Series (log, one plot)

# ### Trout/Coho TS 
# df_plot = eDNA.reset_index()  # eDNA

# ## All values
# # A = df_plot[df_plot.target=='trout'].set_index('dt').sort_index()
# # B = df_plot[df_plot.target=='coho'].set_index('dt').sort_index()

# ## Daily mean
# A = df_plot[df_plot.target=='trout'].groupby('date').mean().sort_index()
# B = df_plot[df_plot.target=='coho'].groupby('date').mean().sort_index()

# # Rolling mean
# Arm = A['log10eDNA'].rolling(window=7,center=True).mean()
# Brm = B['log10eDNA'].rolling(window=7,center=True).mean()


# #A.loc[A.BLOD == 1,'log10eDNA'] = 0  # Remove samples BLOQ
# #B.loc[B.BLOD == 1,'log10eDNA'] = 0

# plt.figure(figsize=(10,4))
# plt.plot(A['log10eDNA'],marker='.',ms=4, color=pal[1])
# plt.plot(B['log10eDNA'],marker='.',ms=4, color=pal[0])

# ## Hatchery releases
# plt.scatter(hatch.index.unique(), -.15*np.ones(len(hatch.index.unique())),s=18,color='k', marker='^')

# plt.xlim(ESP.index[0], ESP.index[-1])   # Range ESP was deployed
# plt.ylabel('log$_{10}$(copies/mL)')
# plt.legend(['trout', 'coho', 'hatchery release'], frameon=False)

# plot_spines(plt.gca(), offset=4)

# plt.tight_layout()
# plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_time_series_log.png'),dpi=300)



#%% eDNA Time Series (linear)

# ### Trout/Coho TS 
#df_plot = eDNA.reset_index()  # eDNA
# ## All values
# A = df_plot[df_plot.target=='trout'].set_index('dt').sort_index()
# B = df_plot[df_plot.target=='coho'].set_index('dt').sort_index()

# ## Daily mean
# #A = df_plot[df_plot.target=='trout'].groupby('date').mean().sort_index()
# #B = df_plot[df_plot.target=='coho'].groupby('date').mean().sort_index()

# #A.loc[A.BLOD == 1,'log10eDNA'] = 0  # Remove samples BLOQ
# #B.loc[B.BLOD == 1,'log10eDNA'] = 0

# plt.figure(figsize=(10,4))
# plt.plot(A['eDNA'],marker='.',ms=4, color=pal[1])
# plt.plot(B['eDNA'],marker='.',ms=4, color=pal[0])

# ## Hatchery releases
# plt.scatter(hatch.index.unique(), -250*np.ones(len(hatch.index.unique())),s=18,color='k',marker='^')

# plt.xlim(ESP.index[0], ESP.index[-1])   # Range ESP was deployed
# plt.ylabel('copies/mL')
# plt.legend(['trout', 'coho', 'hatchery release'], frameon=False)

# plot_spines(plt.gca(), offset=4)

# plt.tight_layout()
# plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_time_series_linear.png'),dpi=300)