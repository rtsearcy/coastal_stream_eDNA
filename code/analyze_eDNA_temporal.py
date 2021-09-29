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
from matplotlib.patches import Patch
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

### Load eDNA Data
# Contains target, replicate #,dilution level, concentration (post-dilution factor)
eDNA = pd.read_csv(os.path.join(folder,'eDNA','eDNA.csv'), parse_dates=['dt','date'])

# Project date range
dr = pd.date_range('2019-03-25', '2020-04-04')
eDNA = eDNA[eDNA.date.isin(dr)]  


## Separate targets
trout = eDNA[eDNA.target=='trout'].sort_values('dt').set_index('dt') 
coho = eDNA[eDNA.target=='coho'].sort_values('dt').set_index('dt') 


### Load Combined ESP logs
# FOr TOD analyses
# Contains sampling times, volumes, ESP name
ESP = pd.read_csv(os.path.join(folder,'ESP_logs','ESP_logs_combined.csv'), 
                  parse_dates = ['sample_wake','sample_start','sample_mid','sample_end','date'], 
                  index_col=['sample_mid'])  # can also use sample_start, but probably little diff.

ESP = ESP[ESP.date.isin(dr)]
ESP = ESP[~ESP.lab_field.isin(['lab','control', 'control '])]  # Keep deployed/field/nana; Drop control/lab samples
ESP.drop(ESP[ESP.id.isin(['SCr-181', 'SCr-286', 'SCr-479', 'SCr-549'])].index, inplace=True) # Drop duplicate (no qPCR IDs)

## Lists of days with sampling frequencies of 1x, 2x, and 3x per day
three_per_day = [d for d in ESP['date'].unique() if (ESP.groupby('date').count() == 3)['id'].loc[d]]
# morning, afternoon, and evening samples
mae = [d for d in three_per_day if (ESP.groupby('date').sum()['morn_midday_eve']==3).loc[d]] 
two_per_day = [d for d in ESP['date'].unique() if (ESP.groupby('date').count() == 2)['id'].loc[d]]
one_per_day = [d for d in ESP['date'].unique() if (ESP.groupby('date').count() == 1)['id'].loc[d]]


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


#%% General Stats 
print('\n- - - eDNA Samples - - -')

for t in eDNA.target.unique():  
    print('\n' + t.upper())
    target = eDNA[eDNA.target==t] #.set_index('dt')
    print(target['eDNA'].describe().round(2))
    print('N BLOD - ' + str((target['BLOD']==1).sum()))
    print('N > 100 copies/mL - ' + str((target['eDNA']>100).sum()))
    

### % detect / BLOD / above LOD
print('\n\nNon-Detects/BLOD\n% non-detect (0 replicates amplified)')
all_detect = pd.concat([
    eDNA[(eDNA.detected==0)].groupby(['target']).count()['id'].rename('non-detect'),
    eDNA[(eDNA.detected==1) & (eDNA.BLOD==1)].groupby(['target']).count()['id'].rename('detect_but_BLOD'),
    eDNA[(eDNA.BLOD==1)].groupby(['target']).count()['id'].rename('total_BLOD'),
    eDNA[(eDNA.BLOD==0)].groupby(['target']).count()['id'].rename('above_LOD'),
    ], axis=1)
all_detect = 100*(all_detect.T / eDNA.groupby(['target']).count()['id']).T.round(3)
all_detect = pd.concat([all_detect, eDNA.groupby(['target']).count()['id'].rename('N')], axis=1)
print(all_detect)

### Differences
print('\n\nDifferences between Trout and Coho signals')
print(stats.mannwhitneyu(trout.log10eDNA,coho.log10eDNA))

### Correlation
## between target's overall signals
print('\nCorrelation between Trout and Coho signals')
print(eDNA_corr(trout.log10eDNA,coho.log10eDNA))
print('\n')
print(eDNA_corr(trout.log10eDNA,coho.log10eDNA, corr_type='spearman'))


### CV Between sample replicates
print('\nCV between 3 samples replicates')
eDNA['eDNA_CV'] = eDNA['eDNA_sd'] / eDNA['eDNA']
eDNA.loc[eDNA.eDNA_CV == np.inf, 'eDNA_CV'] = np.nan # replace inf with NaN
#trout['repCV'] = trout['eDNA_sd'] / trout['eDNA']
#coho['repCV'] = coho['eDNA_sd'] / coho['eDNA']
group = eDNA.groupby(['target','dilution'])
print('\nTotal by dilution: ')
print(group.count()['id'].rename('count'))
print('CVs of samles above LOQ')
#print(group['eDNA_CV'].describe().round(3))
print(eDNA[(eDNA.BLOQ==0)&(eDNA.n_BLOQ < 2)].groupby(['target','dilution']).describe()['eDNA_CV'].round(3))
print('Overall')
print(eDNA[(eDNA.BLOQ==0)&(eDNA.n_BLOQ < 2)].groupby(['target']).describe()['eDNA_CV'].round(3))


### Plots
df_plot = eDNA.reset_index()  # eDNA

## Set BLOD to 0 for plots?
df_plot.loc[df_plot.BLOD == 1,'eDNA'] = 0
df_plot.loc[df_plot.BLOD == 1,'log10eDNA'] = 0 

## All eDNA Data - Boxplot and Histogram
plt.figure(figsize=(3,4))
#plt.subplot(1,2,1)
sns.boxplot(x='target',y='log10eDNA', data = df_plot, width=.75, palette=pal, saturation=1, linewidth=1.2)
plt.xlabel('')
plt.gca().set_xticklabels(['$\it{O. kisutch}$', '$\it{O. mykiss}$'])
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
ylim = plt.ylim()
caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.tight_layout()

## Scatterplot between trout and coho
temp = pd.concat([trout.log10eDNA,coho.log10eDNA], axis=1).dropna()
temp.columns = ['trout','coho']

plt.figure(figsize=(4,4))
plt.scatter('trout','coho', s = 10, data=temp, color='k')
plt.xlabel('trout')
plt.ylabel('coho')
plt.title('Correlation between targets')

#%% Sub-daily variability
print('\nSub-daily variation')
#temp = eDNA[eDNA.date.isin(two_per_day + three_per_day)]
for t in eDNA.target.unique():
    print('\n' + t)
    temp = eDNA[eDNA.target == t].reset_index()
    ip1 = (((temp.dt.diff().dt.seconds/60/60) <= 12) & \
    ((temp.dt.diff().dt.days < 1)))
    ip1 = ip1[ip1 == 1].index  # consecutive sample
    i = ip1 - 1  # original sample
    print('# samples where consecutive was taken < 12h later: ' + str(len(i)))
    
    delta = pd.Series(100*np.abs(temp.loc[i,'eDNA'].values - temp.loc[ip1,'eDNA'].values)/(temp.loc[i,'eDNA'].values+1))
    print(delta.describe().round(0))
    
    print('# days ND to Above LOQ (or visa versa): ' + str(np.ceil(temp.BLOQ.dropna()).diff().abs().sum()))

#%% Morning, Midday, Evening
print('\n - Time of Day (Morning/Midday/Evening) -')
print('0 - morning (<1100); 1 - midday (1100<=time<1700); 2 - evening (>=1700)')
print('\nN days w/ samples in all 3 periods: ' + str(len(mae))) # 97 days each with morn, midday, eve samples
df_mae = eDNA[eDNA.date.isin(mae)]
mae_block = pd.date_range('4/5/2019','5/14/2019')  # largest period of consecutive days (n=40)

### General stats
print('\n Summary of eDNA concentrations (copies/mL)')
print(df_mae.groupby(['target','morn_midday_eve']).describe()['eDNA'].round(2))


### Detection / BLOD
print('\nIs there a differential amplification / quantification rate by TOD?')
tod_detect = pd.concat([
    df_mae[(df_mae.detected==0)].groupby(['target','morn_midday_eve']).count()['id'].rename('non-detect'),
    df_mae[(df_mae.detected==1) & (df_mae.BLOD==1)].groupby(['target','morn_midday_eve']).count()['id'].rename('detect_but_BLOD'),
    df_mae[(df_mae.BLOD==1)].groupby(['target','morn_midday_eve']).count()['id'].rename('total_BLOD'),
    df_mae[(df_mae.BLOD==0)].groupby(['target','morn_midday_eve']).count()['id'].rename('above_LOD'),
    df_mae.groupby(['target','morn_midday_eve']).count()['id'].rename('N')
    ], axis=1)
print(tod_detect.round(1))
## Detection: no seeming pattern; BLOD: maybe with trout

for t in df_mae.target.unique():
    print('\n' + t.upper())
    morn = df_mae[(df_mae.target==t) & (df_mae.morn_midday_eve==0)].set_index('date').sort_index()['eDNA']
    midd = df_mae[(df_mae.target==t) & (df_mae.morn_midday_eve==1)].set_index('date').sort_index()['eDNA']
    eve = df_mae[(df_mae.target==t) & (df_mae.morn_midday_eve==2)].set_index('date').sort_index()['eDNA']

### Correlation between morn/midday/eve
    mae_corr = pd.concat([morn,midd,eve], axis=1)
    mae_corr.columns = ['morning', 'midday','evening']
    print('\nCorrelation')
    print(mae_corr.corr(method='spearman'))
    
    (mae_corr.sum(axis=1) > 0).sum()
    
### Rank of time of day
    print('\nRank of concentrations by times of day')
    temp = pd.concat([
        (mae_corr.rank(axis=1) == 3).sum().rename('Highest'),
        (mae_corr.rank(axis=1) == 2).sum().rename('Middle'),
        (mae_corr.rank(axis=1) == 1).sum().rename('Lowest')
        ],axis=1)
    print(temp)
    
### Median absolute deviation
    # print('\nMedian absolute deviation (MAD)')    
    # mae_corr['MAD'] = stats.median_abs_deviation(mae_corr,axis=1)
    # print(mae_corr.MAD.describe().round(2))

### Differences between time of day?
    print('\nDifference between time of day')
    mae_corr['diff_morn_midday'] = abs(mae_corr.morning - mae_corr.midday)
    mae_corr['diff_morn_eve'] = abs(mae_corr.morning - mae_corr.evening)
    mae_corr['diff_midday_eve'] = abs(mae_corr.evening - mae_corr.midday)
    print(mae_corr[['diff_morn_midday',  'diff_morn_eve',  'diff_midday_eve']].describe().round(2))

### Difference sample to sample
    print('\nDifference sample to sample (high-freq, consecutive days)')
    temp = df_mae[(df_mae.date.isin(mae_block)) & (df_mae.target==t)][['dt','date','eDNA']]  
    # only samples in consecutive days
    print(abs(temp.eDNA.diff()).describe().round(2)) 

### Hypothesis tests
    print('\nStatistical difference between time of day?')
    print('Median (Morning/Midday/Evening): ' + str(round(morn.median(),3)) + 
          '/' + str(round(midd.median(),3)) + 
          '/' + str(round(eve.median(),3)))
    #print(stats.kruskal(morn,midd,eve))
    print(stats.friedmanchisquare(morn,midd,eve))  # Sign test for repeated measures
    
    print('\nMorning/Midday')
    # print(stats.mannwhitneyu(morn,midd))
    print(stats.wilcoxon(morn,midd))               # Signed rank for paired samples
    print('\nMorning/Evening')
    #print(stats.mannwhitneyu(morn,eve))
    print(stats.wilcoxon(morn,eve))
    print('\nMidday/Evening')
    #print(stats.mannwhitneyu(eve,midd))
    print(stats.wilcoxon(eve,midd))


### Variance in a day's samples
### Spectra on subdaily signal
tod_data = df_mae[(df_mae.date.isin(mae_block))].groupby(['dt',
                                                          'target']).first().log10eDNA.reset_index().pivot(index='dt',                                                                                                      columns='target',
                                                                                                           values='log10eDNA')
data = [tod_data['coho'],tod_data['trout']]
plt.figure(figsize=(4,4))
col = [pal[0],pal[1]]
for i in range(0,len(data)):
    # ## zero padding
    z = 2**np.ceil(np.log2(len(data[i]))) - len(data[i])  # number of zeros needed to get series to length 2^N
    pad = np.log10(.0001)
    if z % 2 == 0:  # eve
        d = np.pad(data[i],(int(z/2), int(z/2)), 'constant', constant_values = (pad,pad))
    else:
        d = np.pad(data[i],(int(z/2 +.5), int(z/2 -.5)), 'constant', constant_values = (pad,pad))
    data[i] = d
    
    nperseg = len(data[i]) // 2
    f, P = signal.welch(data[i],fs=1/8,
                                window='hamming',          # boxcar, hann, hamming, 
                                nfft=None,
                                nperseg=nperseg,           # N data points per segement (bigger windows = less smooth, more freq res)
                                noverlap=nperseg//2,       # N data points overlapping (50% is common)
                                detrend='constant',        # remove trend, False, 'constant'
                                scaling='density')         # 'density' [conc^2/Hz] or 'spectrum' [conc^2]

    plt.plot(f, P, color=col[i],lw=1.5)
    
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
plt.xlabel('f [1/hr]')
plt.ylabel(r'P$_{xx}$ [$(log_{10}$copies/mL)$^2$*hr]')

plt.sca(ax)
plt.xscale('log')
plt.yscale('log')
plt.legend(['coho','trout'], frameon=False)
plot_spines(ax)
plt.tight_layout()
#plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_spectra.png'),dpi=300)


#%% Day to Day
print('\n - Day to Day -')
day_block = pd.date_range('8/25/2019','1/30/2020') # biggest consecutive blocks (159d)

### Daily Mean
T = trout.reset_index().set_index('dt').resample('D').mean()
C = coho.reset_index().set_index('dt').resample('D').mean()

### Downsampled
# T = trout.reset_index().set_index('dt').resample('D').first()
# C = coho.reset_index().set_index('dt').resample('D').first()


### Difference day to day
print('\nDifference sample to sample (high-freq) [COHO/TROUT]')
data = [C,T]
for d in data:
    #temp = d[(d.index.isin(day_block))]
    temp = d
    # only samples in consecutive days
    print('\n')
    delta = 100*abs(temp.eDNA.diff())/(temp.eDNA+1)
    #print(abs(temp.eDNA.diff()).describe().round(2))
    print(delta.describe().round(0))  # normalized by first sample
    #print(abs(temp.log10eDNA.diff()).describe().round(3))
    
    print('\n# days with no change: ' + str((temp.eDNA.diff() == 0).sum()))
    print('# days ND to Above LOQ (or visa versa): ' + str(np.ceil(temp.BLOQ).diff().abs().sum()))
    
print('\n Correlation in these differences between targets:')
print('Linear')
print(eDNA_corr(T.eDNA.diff(), C.eDNA.diff()))
print('Log')
print(eDNA_corr(T.log10eDNA.diff(), C.log10eDNA.diff()))
    
## Interpolate missing days
T[['log10eDNA','eDNA']] = T[['log10eDNA','eDNA']].interpolate('linear')
C[['log10eDNA','eDNA']] = C[['log10eDNA','eDNA']].interpolate('linear')


### Serial Correlation  
plt.figure(figsize=(8,4))

plt.subplot(2,2,1)  # Trout autocorrelation
rhoT = acf(T.log10eDNA, nlags=30, fft=False)
plt.stem(range(0,len(rhoT)), rhoT, linefmt='k-', markerfmt=' ', basefmt='k-')
plt.axhline(1.96/(len(T)**.5), ls='--', color='grey', alpha=0.7)
plt.axhline(-1.96/(len(T)**.5), ls='--', color='grey', alpha=0.7)
plt.title('Autocorrelation')
plt.ylabel('TROUT')
plot_spines(plt.gca(), offset=4)

plt.subplot(2,2,2)  # Trout partial
rhoT = pacf(T.log10eDNA, nlags=30)
plt.stem(range(0,len(rhoT)), rhoT, linefmt='k-', markerfmt=' ', basefmt='k-')
plt.axhline(1.96/(len(T)**.5), ls='--', color='grey', alpha=0.7)
plt.axhline(-1.96/(len(T)**.5), ls='--', color='grey', alpha=0.7)
plt.title('Partial Autocorrelation')
plot_spines(plt.gca(), offset=4)

plt.subplot(2,2,3)  # Coho autocorrelation
rhoT = acf(C.log10eDNA, nlags=30, fft=False)
plt.stem(range(0,len(rhoT)), rhoT, linefmt='k-', markerfmt=' ', basefmt='k-')
plt.axhline(1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)
plt.axhline(-1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)
plt.ylabel('COHO')
plot_spines(plt.gca(), offset=4)

plt.subplot(2,2,4)  # Coho partial
rhoT = pacf(C.log10eDNA, nlags=30)
plt.stem(range(0,len(rhoT)), rhoT, linefmt='k-', markerfmt=' ', basefmt='k-')
plt.axhline(1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)
plt.axhline(-1.96/(len(C)**.5), ls='--', color='grey', alpha=0.7)
plot_spines(plt.gca(), offset=4)

plt.tight_layout()
plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_autocorrelation.png'),dpi=300)


### Spectra on daily signal
data = [C.log10eDNA,T.log10eDNA]
plt.figure(figsize=(4,4))
col = [pal[0],pal[1]]
for i in range(0,len(data)):
    # ## zero padding
    z = 2**np.ceil(np.log2(len(data[i]))) - len(data[i])  # number of zeros needed to get series to length 2^N
    #pad = np.log10(.0001)
    pad = 0
    if z % 2 == 0:  # eve
        d = np.pad(data[i],(int(z/2), int(z/2)), 'constant', constant_values = (pad,pad))
    else:
        d = np.pad(data[i],(int(z/2 +.5), int(z/2 -.5)), 'constant', constant_values = (pad,pad))
    data[i] = d
    
    nperseg = len(data[i]) // 2
    f, P = signal.welch(data[i],fs=1,                      # could also use signal.periodogram
                                window='hamming',          # boxcar, hann, hamming, 
                                nfft=None,
                                nperseg=nperseg,           # N data points per segement (bigger windows = less smooth, more freq res)
                                noverlap=nperseg//2,       # N data points overlapping (50% is common)
                                detrend='constant',        # remove trend, False, 'constant'
                                scaling='density')         # 'density' [conc^2/Hz] or 'spectrum' [conc^2]
    # f [1/day] / P [log10conc^2*day] or [log10 conc^2]

    plt.plot(f, P, color=col[i],lw=1.5)
    
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

plt.sca(ax)
plt.xscale('log')
plt.yscale('log')
plt.legend(['coho','trout'], frameon=False)
plot_spines(ax)
plt.tight_layout()
plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_spectra.png'),dpi=300)

#%% Season (Spring/Summer/Fall/Winter)
print('\n - Season of Year -')
print('Spring - Mar-May; Summer - Jun-Aug; Fall - Sep-Nov; Winter - Dec-Feb')

### General
print(eDNA.groupby(['target','season']).describe()['eDNA'].round(1))

### BLOD / Detection
print('\nIs there a differential amplification / quantification rate by season?')
season_detect = pd.concat([
    eDNA[(eDNA.detected==0)].groupby(['target','season']).count()['id'].rename('non-detect'),
    eDNA[(eDNA.detected==1) & (eDNA.BLOD==1)].groupby(['target','season']).count()['id'].rename('detect_but_BLOD'),
    eDNA[(eDNA.BLOD==1)].groupby(['target','season']).count()['id'].rename('total_BLOD'),
    eDNA[(eDNA.BLOD==0)].groupby(['target','season']).count()['id'].rename('above_LOD'),
    ], axis=1)
season_detect = 100*(season_detect.T / eDNA.groupby(['target','season']).count()['id']).T.round(3)
season_detect = pd.concat([season_detect, eDNA.groupby(['target','season']).count()['id'].rename('N')], axis=1)
season_detect = season_detect.reindex(['spring','summer','fall','winter'], level=1)
print(season_detect.round(2))

### Differences between season?
print('\nDifference between season?')
for t in eDNA.target.unique():
    print('\n' + t.upper())
    spring = eDNA.loc[(eDNA.target==t) & (eDNA.season=='spring'),'eDNA']
    summer = eDNA.loc[(eDNA.target==t) & (eDNA.season=='summer'),'eDNA']
    fall = eDNA.loc[(eDNA.target==t) & (eDNA.season=='fall'),'eDNA']
    winter = eDNA.loc[(eDNA.target==t) & (eDNA.season=='winter'),'eDNA']
    
    print('Median (Sp/Su/Fa/Wi): ' + str(round(spring.median(),3)) + 
          '/' + str(round(summer.median(),3)) + 
          '/' + str(round(fall.median(),3)) + 
          '/' + str(round(winter.median(),3)))
    print(stats.kruskal(spring,summer,fall,winter))
    
    print('\nSpring/Summer')
    print(stats.mannwhitneyu(spring,summer))
    print('\nSpring/Fall')
    print(stats.mannwhitneyu(spring,fall))
    print('\nSpring/Winter')
    print(stats.mannwhitneyu(spring,winter))
    print('\nSummer/Fall')
    print(stats.mannwhitneyu(fall,summer))
    print('\nSummer/Winter')
    print(stats.mannwhitneyu(winter,summer))
    print('\nFall/Winter')
    print(stats.mannwhitneyu(fall,winter))
    

## Differences between wet and dry season?
# print('\nDifference between Dry and Wet seasons?')
# dry = eDNA.loc[(eDNA.target==t) & (eDNA.wet_season==0),'log10eDNA']
# wet = eDNA.loc[(eDNA.target==t) & (eDNA.wet_season==1),'log10eDNA']
# print('N (Dry/Wet): ' + str(len(dry)) + '/' + str(len(wet)))
# print('Median (Dry/Wet): ' + str(round(dry.median(),3)) + '/' + str(round(wet.median(),3)))
# print(stats.mannwhitneyu(dry,wet))

### Difference in correlation during different conditions?
print('\nCorrelation')
for s in eDNA.season.unique():
    print('\n' + s.capitalize() + ' samples')
    
    ## CORRELATION
    print(eDNA_corr(trout[trout.season==s], 
                    coho[coho.season==s], 
                    x_col='log10eDNA',
                    y_col='log10eDNA', corr_type='spearman'))


### Plots
df_plot = eDNA.reset_index()  # eDNA

## Set BLOD to 0 for plots?
df_plot.loc[df_plot.BLOD == 1,'eDNA'] = 0
df_plot.loc[df_plot.BLOD == 1,'log10eDNA'] = 0 

## Season Boxplots
plt.figure(figsize=(5,4))  
sns.boxplot(x='season',y='log10eDNA', hue='target', data=df_plot, palette=pal, saturation=.9, linewidth=1.2)
plt.xlabel('')
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')

plt.legend(['coho','trout'], frameon=False, loc='upper right')
leg = plt.gca().get_legend()
leg.legendHandles[0].set_color(pal[0]) # coho
leg.legendHandles[1].set_color(pal[1]) # trout

caps_off(plt.gca())     # turn off caps
flier_shape(plt.gca())  # fliers to circles
plot_spines(plt.gca())

plt.tight_layout()

# ## Wet Season
# plt.figure(figsize=(4,4))  
# sns.boxplot(x='wet_season',y='log10eDNA', hue='target', data=df_plot,  width=.6, palette=pal)
# plt.xlabel('')
# plt.xticks(ticks=[0,1], labels=['Dry Season','Wet Season'])
# plt.ylabel('log$_{10}$(copies/mL)')

# plt.legend(['coho','trout'], frameon=False, loc='upper left')
# leg = plt.gca().get_legend()
# leg.legendHandles[0].set_color(pal[0]) # coho
# leg.legendHandles[1].set_color(pal[1]) # trout

# caps_off(plt.gca())     # turn off caps
# flier_shape(plt.gca())  # fliers to circles
# plot_spines(plt.gca())

# plt.tight_layout()


#%% eDNA Time Series (log, STEM)

### Trout/Coho TS 
df_plot = eDNA.reset_index()  # eDNA

## All values
A = df_plot[df_plot.target=='trout'].set_index('dt').sort_index()
B = df_plot[df_plot.target=='coho'].set_index('dt').sort_index()

## Daily mean
#A = df_plot[df_plot.target=='trout'].groupby('date').mean().sort_index()
#B = df_plot[df_plot.target=='coho'].groupby('date').mean().sort_index()

A.loc[A.BLOD == 1,'log10eDNA'] = 0  # Remove samples BLOQ
B.loc[B.BLOD == 1,'log10eDNA'] = 0

# A.loc[A.detected == 0,'log10eDNA'] = 0  # Remove samples BLOQ
# B.loc[B.detected == 0,'log10eDNA'] = 0

# Rolling mean
Arm = A['log10eDNA'].resample('D').mean().rolling(window=5,center=False).mean()
Brm = B['log10eDNA'].resample('D').mean().rolling(window=5,center=False).mean()

plt.figure(figsize=(10,6))
plt.subplot(211) # coho
plt.bar(B.index,B.log10eDNA + 0.5, bottom = -.5, width=.5, color=pal[0])
#plt.axhline(0,color=pal[0])
plt.scatter(B.index,B.log10eDNA, color=pal[0], s=5)
#plt.plot(Brm,color='k')

plt.axvline(datetime.datetime(2019,9,5), color='grey', ls=':')
plt.axvline(datetime.datetime(2019,12,4), color='grey', ls=':')

# #Plot BLOQ marker
# Bb = B[B.detected==0]
# plt.scatter(Bb.index,Bb.log10eDNA, color='k', s=16, marker='*', zorder=1)

## Hatchery releases
plt.scatter(hatch.index.unique(), 3*np.ones(len(hatch.index.unique())),s=18,color='k',marker='$H$') # 'v'

plt.ylim(-.05, 1.05*df_plot.log10eDNA.max())
plt.xlim(dr[0], dr[-1])   # Project Date Range
plt.gca().axes.xaxis.set_ticklabels([])

plt.text(ESP.index[10],.95*plt.ylim()[1], '$\it{O. kisutch}$')
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
#plt.legend(['trout', 'coho', 'hatchery release'], frameon=False)
plot_spines(plt.gca(), offset=0)


plt.subplot(212)  # trout
plt.bar(A.index, A.log10eDNA + .5, bottom=-.5, width=.5, color=pal[1])
#plt.axhline(0,color=pal[1])
plt.scatter(A.index,A.log10eDNA, color=pal[1], s=5)

# Berm opening/close
plt.axvline(datetime.datetime(2019,9,5), color='grey', ls=':')
plt.axvline(datetime.datetime(2019,12,4), color='grey', ls=':')

# #Plot BLOQ marker
# Ab = A[A.detected==0]
# plt.scatter(Ab.index,Ab.log10eDNA, color='k', s=16, marker='*', zorder=1)

#plt.plot(Arm,color='k')
plt.ylim(-.05, 1.05*df_plot.log10eDNA.max())
plt.xlim(dr[0], dr[-1])   # Project Date Range

plt.text(ESP.index[10],.95*plt.ylim()[1], '$\it{O. mykiss}$')
plt.ylabel('log$_{10}$(eDNA copies/mL + 1)')
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

#%% eDNA Time Series (linear, STEM)

### Trout/Coho TS 
df_plot = eDNA.reset_index()  # eDNA

## All values
#A = df_plot[df_plot.target=='trout'].set_index('dt').sort_index()
#B = df_plot[df_plot.target=='coho'].set_index('dt').sort_index()

## Daily mean
A = df_plot[df_plot.target=='trout'].groupby('date').mean().sort_index()
B = df_plot[df_plot.target=='coho'].groupby('date').mean().sort_index()

A.loc[A.BLOD == 1,'log10eDNA'] = 0  # Remove samples BLOQ
B.loc[B.BLOD == 1,'log10eDNA'] = 0

# A.loc[A.detected == 0,'log10eDNA'] = 0  # Remove samples BLOQ
# B.loc[B.detected == 0,'log10eDNA'] = 0


plt.figure(figsize=(10,6))
plt.subplot(211) # coho
plt.bar(B.index,B.eDNA + 0.5, bottom = -.5, width=.5, color=pal[0])
#plt.axhline(0,color=pal[0])
plt.scatter(B.index,B.eDNA, color=pal[0], s=5)

## Hatchery releases
#plt.scatter(hatch.index.unique(), 3*np.ones(len(hatch.index.unique())),s=18,color='k',marker='$H$') # 'v'

plt.ylim(-.05, 1.05*B.eDNA.max())
plt.xlim(dr[0], dr[-1])   # Project Date Range
plt.gca().axes.xaxis.set_ticklabels([])

plt.text(ESP.index[10],.95*plt.ylim()[1], '$\it{O. kisutch}$')
plt.ylabel('eDNA (copies/mL)')

#plt.legend(['trout', 'coho', 'hatchery release'], frameon=False)
plot_spines(plt.gca(), offset=0)


plt.subplot(212)  # trout
plt.bar(A.index, A.eDNA + .5, bottom=-.5, width=.5, color=pal[1])
#plt.axhline(0,color=pal[1])
plt.scatter(A.index,A.eDNA, color=pal[1], s=5)

plt.ylim(-.05, 1.05*A.eDNA.max())
plt.xlim(dr[0], dr[-1])   # Project Date Range

plt.text(ESP.index[10],.95*plt.ylim()[1], '$\it{O. mykiss}$')
plt.ylabel('eDNA (copies/mL)')
#plt.legend(['trout', 'rolling mean'], frameon=False)
plot_spines(plt.gca(), offset=0)

## Invert?
plt.gca().invert_yaxis()
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['bottom'].set_visible(False) 
plt.gca().xaxis.tick_top()

plt.tight_layout()
plt.subplots_adjust(top=0.961,bottom=0.078,left=0.057,right=0.984,hspace=0.16,wspace=0.2)

plt.savefig(os.path.join(folder.replace('data','figures'),'eDNA_time_series_linear_stem.png'),dpi=300)
