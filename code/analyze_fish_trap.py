#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Load Data
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

Stats and Plots for Fish Trap data

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

### Load Fish data
# Contains sampling times, volumes, ESP name
df = pd.read_csv(os.path.join(folder,'NOAA_data', 'fish_trap.csv'), 
                 parse_dates = ['date','dt'], index_col=['id'], encoding='latin1')

### Date variables
df = df.sort_values('dt')  # Sort by datetime
df['year'] = df['dt'].dt.year
df['month'] = df['dt'].dt.month
df['year_month'] = df.year.astype(str) + '-' + df.month.astype(str).str.rjust(2,'0')
df['week'] = df['dt'].dt.isocalendar().week
df['year_week'] = df.year.astype(str) + '-' + df.week.astype(str).str.rjust(2,'0')

df['wet_season'] = 0  # dry season
df.loc[df.month.isin([10,11,12,1,2,3,4]),'wet_season'] = 1 # wet season
df['season'] = 'winter' # Dec-Feb
df.loc[df.month.isin([3,4,5]),'season'] = 'spring'
df.loc[df.month.isin([6,7,8]),'season'] = 'summer'
df.loc[df.month.isin([9,10,11]),'season'] = 'fall'

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

def plot_spines(axx, offset=8): # Offset position, Hide the right and top spines
    axx.spines['left'].set_position(('outward', offset))
    axx.spines['bottom'].set_position(('outward', offset))
    
    axx.spines['right'].set_visible(False)
    axx.spines['top'].set_visible(False)

#%% - Stats - 
#%% Counts
print('Date Range: ' + str(df.date.iloc[0].date()) + ' to ' + str(df.date.iloc[-1].date()))
print('Days of counts: ' + str(len(df.date.unique())))

print('\nN fish: ' + str(len(df)))
#print(df.groupby(['species','life_stage','life_stage_cat']).count()['date'].rename('N').sort_values(ascending=False))
print(df.groupby(['species','origin','life_stage','life_stage_cat']).count()['date'].rename('N'))
## Notes: Most captures were juveniles, order of magnitude more trout than coho, 
## only 3 total natural salmon (most from hatchery)

print('\nAdult Live / Dead:')
print(df.groupby(['species','life_stage_cat','adult_live']).count()['date'].rename('N'))
## Notes: Half of adult coho were found dead

### Plots
date_range = pd.date_range(df.date.iloc[0].date(), '05-01-2020')  # date range of datasets
# Counts
A = df.groupby(['species','life_stage','date']).count()['year'].rename('value').reset_index()   # Find counts by species and life stage 
# Mass (not available for every fish)
#A = (df.groupby(['species','life_stage','date']).sum()['mass'].rename('value')/1000).reset_index()

A = A.pivot(index='date',columns=['species','life_stage'],values='value')
A = A.reindex(index=date_range)  # reindex to entire date range
A = A.fillna(value=0)

### Adult / juvenille barcharts
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)  # coho
plt.bar(A.index, A['coho','Juvenile'], label='Juvenile')
plt.bar(A.index, A['coho','Adult'], bottom=A['coho','Juvenile'],label='Adult')
plt.ylabel('COHO')
#plt.ylim(0,1.1*A.max().max())
plt.legend(frameon=False)

# plt.twinx(plt.gca())
# plt.plot(Brm, label='eDNA (7d rolling)',color='k',lw=1)

plt.subplot(2,1,2)  # trout
plt.bar(A.index, A['trout','Juvenile'], label='Juvenile')
plt.bar(A.index, A['trout','Adult'], bottom=A['trout','Juvenile'],label='Adult')
plt.ylabel('TROUT')
#plt.ylim(0,1.1*A.max().max())
plt.legend(frameon=False)

# plt.twinx(plt.gca())
# plt.plot(Arm, label='eDNA (7d rolling)',color='k',lw=1)

plt.tight_layout()

### Adult only
plt.figure(figsize=(10,10))

plt.subplot(2,1,1)  # coho
plt.bar(A.index, A['coho','Adult'], label='Adult')
plt.ylabel('N')
#plt.ylim(0,1.1*A.max().max())
plt.legend(frameon=False)
#plt.twinx(plt.gca())
#plt.plot(Brm, label='eDNA (7d rolling)',color='k',lw=1)

plt.subplot(2,1,2)  # trout
plt.bar(A.index, A['trout','Adult'],label='Adult')
plt.ylabel('N')
#plt.ylim(0,1.1*A.max().max())
plt.legend(frameon=False)
#plt.twinx(plt.gca())
#plt.plot(Arm, label='eDNA (7d rolling)',color='k',lw=1)

plt.tight_layout()


#%% Counts by season
print('\nBy season:')
season_count = df.groupby(['season', 'species','life_stage']).count()['date'].rename('N')
print(season_count.loc[['spring','summer','fall','winter']])
## Notes: Most fish in spring, Adults primarily in winter, spring (wet season); 
##Only adult fish in winter (Dec-Feb), No coho in summer/fall (Jun-Nov)

# by wet_season (1-wet, 0-dry()
df.groupby(['wet_season', 'species','life_stage']).count()['date'].rename('N')
# by month
df.groupby(['year_month', 'species','life_stage']).count()['date'].rename('N')

### Bar plot - season
season_plot = season_count.reset_index().pivot(index='season',columns=['life_stage','species'],values='N')
season_plot = season_plot.loc[['spring','summer','fall','winter']].fillna(0)  # sort by season, fill NAN with 0 count
plt.figure(figsize=(5,5))

plt.subplot(211)  # Adult
plt.bar(1.5*np.arange(0,len(season_plot.index)) + 1/4, season_plot['Adult','coho'], width=.5, color=pal4c[0])
plt.bar(1.5*np.arange(0,len(season_plot.index)) - 1/4, season_plot['Adult','trout'], width=.5, color=pal4c[2])
plt.xticks(ticks=1.5*np.arange(0,len(season_plot.index)), labels=season_plot.index)
plt.title('N Adults')
plt.yscale('log')
plt.legend(['coho','trout'], frameon=False)
plot_spines(plt.gca(), offset=0)

plt.subplot(212)  # Juvenile
plt.bar(1.5*np.arange(0,len(season_plot.index)) + 1/4, season_plot['Juvenile','coho'], width=.5, color=pal4c[0])
plt.bar(1.5*np.arange(0,len(season_plot.index)) - 1/4, season_plot['Juvenile','trout'], width=.5, color=pal4c[2])
plt.xticks(ticks=1.5*np.arange(0,len(season_plot.index)), labels=season_plot.index)
plt.title('N Juveniles')
plt.yscale('log')
#plt.legend(['coho','trout'], frameon=False)
plot_spines(plt.gca(), offset=0)

plt.tight_layout()


#%% Biomass / Length
print('\nTotal Biomass: ' + str(df.mass.sum()/1000) + ' kg')
print('Median mass (g) / length (mm):')
print(pd.concat([
    df.groupby(['species','life_stage']).count()['length'].rename('N_length'),
    df.groupby(['species','life_stage']).median()[['length']],
    df.groupby(['species','life_stage']).count()['mass'].rename('N_mass'),
    df.groupby(['species','life_stage']).median()[['mass']]],
    axis=1))
## Not all fish had mass or length measurements

### Plot
#sns.catplot(x='origin',y='mass',hue='species',row='life_stage',data=df)


# ### Time of count histogram
## Mostly around 10a, many without timestamp (assume morning)
# plt.figure()
# df.dt.dt.hour.hist()
# plt.title('Hour of Count')
