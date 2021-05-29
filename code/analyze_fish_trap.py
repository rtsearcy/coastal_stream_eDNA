#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Initialize
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

Stats and Plots for Fish Trap data; Create daily fish variables

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

### Plot parameters / functions
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
pal = ['#de425b','#2c8380']
pal = sns.color_palette(pal)

# pal2c = ['#ca0020', '#f4a582'] # salmon colors
# pal2c = sns.color_palette(pal2c)


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

#%%Load Data

folder = '../data/'  # Data folder
dr = pd.date_range('2019-03-25', '2020-04-04')

### Load Fish data
# Contains sampling times, volumes, ESP name
df = pd.read_csv(os.path.join(folder,'NOAA_data', 'fish_trap.csv'), 
                 parse_dates = ['date','dt'], index_col=['id'], encoding='latin1')

df['N'] = 1  # count for grouping

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


#%% Biomass estimation

### Summary - Biomass / Length
print('\nTotal Biomass: ' + str(df.mass.sum()/1000) + ' kg')
print('Median mass (g) / length (mm):')
print(pd.concat([
    df.groupby(['species','life_stage']).count()['length'].rename('N_length'),
    df.groupby(['species','life_stage']).median()[['length']],
    df.groupby(['species','life_stage']).count()['mass'].rename('N_mass'),
    df.groupby(['species','life_stage']).median()[['mass']]],
    axis=1))
## Not all fish had mass or length measurements


### Seperate targets and life stage
trout = df[df.species=='trout']
trout_a = trout[trout.life_stage=='Adult']
trout_j = trout[trout.life_stage=='Juvenile']

coho = df[df.species=='coho']
coho_a = coho[coho.life_stage=='Adult']
coho_j = coho[coho.life_stage=='Juvenile']


### Plot length vs. mass
plt.figure(figsize=(8,4))
 
plt.subplot(121) # Adults
plt.scatter('length','mass',s=8,color=pal[1],marker='^',data=trout_a, label='O. mykiss')
plt.scatter('length','mass',s=10,color=pal[0],marker='o',data=coho_a,label='O. kisutch')
plt.ylabel('Mass (g)')
plt.xlabel('Length (mm)')
plt.title('Adults')
plt.legend(frameon=False, loc='upper left')

plot_spines(plt.gca())

plt.subplot(122) # Juveniles
plt.scatter('length','mass',s=8,color=pal[1],marker='^',data=trout_j, label='O. mykiss')
plt.scatter('length','mass',s=10,color=pal[0],marker='o',data=coho_j,label='O. kisutch')
#plt.ylabel('Mass (g)')
plt.xlabel('Length (mm)')
plt.title('Juveniles')

plot_spines(plt.gca())

plt.tight_layout()


### Regression to estimate missing biomass
print('\nRegression mass on length')
df_reg = pd.DataFrame()
c=0
for d in [coho_a,trout_a,coho_j,trout_j]:
    
    ## simple linear regression
    lm1 = sm.OLS(d['mass'],d['length'],missing='drop').fit()
    
    ## 2nd order linear regression
    lm2 = sm.OLS(d['mass'],d['length']*d['length'],missing='drop').fit()
    
    reg_dict = {
        'beta1': round(lm1.params[0],4),  # first order parameter
        'Rsq1':round(lm1.rsquared,3),
        'beta2': round(lm2.params[0],4),  # first order parameter
        'Rsq2':round(lm2.rsquared,3),
        'N reg': len(d[['mass','length']].dropna()),
        'N all': len(d),
        '% Missing': round(100*d.mass.isna().sum()/len(d),3)
        }
    df_reg = df_reg.append(pd.DataFrame(reg_dict, index=[c]))
    c+=1
df_reg['species'] = ['coho','trout','coho','trout']
df_reg['life_stage']= ['Adult','Adult','Juvenile','Juvenile']
df_reg.set_index(['species','life_stage'],inplace=True)
print(df_reg)


### Backfill missing mass from regression
print('\nEstimating mass from length')
df['mass_est'] = df['mass']
for i in df_reg.index:
    print(i)
    idx = df.loc[(df.species==i[0])&(df.life_stage==i[1])&(np.isnan(df.mass))].index
    print(str(len(idx)) + ' missing mass points')
    mass_est = df_reg.loc[i,'beta2'] *  (df.loc[idx,'length']**2)
    df.loc[idx,'mass_est'] = mass_est
    print(str(len(mass_est.dropna())) + ' points backfilled\n')

g = df.groupby(['species','life_stage'])['mass_est']
print(pd.concat([g.count(),g.mean()],axis=1))


### Fill missing with mean of species/life stage
mean_mass = df.groupby(['species','life_stage']).mean()[['mass_est']]
for i in mean_mass.index:
    df.loc[(df.mass_est.isna()) & 
           (df.species == i[0]) & 
           (df.life_stage== i[1]),'mass_est'] = mean_mass.loc[i,'mass_est']

#%% Daily Variables

## pre-process
df = df.drop(df[df.species=='undetermined'].index) # drop few undetermined species
df.loc[df.life_stage == 'Unknown', 'life_stage'] = 'Juvenile'
df['count_hour'] = df.dt.dt.round('H').dt.hour

## Daily Variables
group = df.groupby(['date','species'])
df_vars = pd.DataFrame(index=group.first().index)

df_vars['N_fish'] = group.count()['N'] # total fish counted

# life stage count
ls_count = df.groupby(['date','species','life_stage']).count()['N']
ls_count = ls_count.reset_index().pivot(index=['date','species'], columns='life_stage',values='N').fillna(0)
ls_count.columns = ['N_adult','N_juvenile']
ls_count = ls_count.astype(int)
df_vars = pd.concat([df_vars,ls_count], axis=1)

# N adults live/dead
nlive = df.groupby(['date','species','adult_live']).count()['N']
nlive = nlive.reset_index().pivot(index=['date','species'], columns='adult_live',values='N').fillna(0)
nlive.columns = ['N_adult_dead', 'N_adult_live']
nlive = nlive.astype(int)
df_vars = pd.concat([df_vars, nlive], axis=1)

## Mass
df_vars['biomass'] = round(group.sum()['mass_est'] / 1000, 3) # total biomass (kg)

ls_mass = df.groupby(['date','species','life_stage']).sum()['mass_est'] / 1000
ls_mass = ls_mass.reset_index().pivot(index=['date','species'], columns='life_stage',values='mass_est').fillna(0)
ls_mass.columns = ['biomass_adult','biomass_juvenile']
df_vars = pd.concat([df_vars, ls_mass], axis=1)

# Live/Dead Adults
blive = round(df.groupby(['date','species','adult_live']).sum()['mass_est'] / 1000, 3) # biomass of live/dead adults
blive = blive.reset_index().pivot(index=['date','species'], columns='adult_live',values='mass_est').fillna(0)
blive.columns = ['biomass_adult_dead', 'biomass_adult_live']
df_vars = pd.concat([df_vars, blive], axis=1)

df_vars['count_hour'] = group.median()['count_hour']                # median hour of fish counting

## Fill in 0 counts for species
for i in df.date.unique():
    if len(df_vars.loc[i]) == 2:  # if observation for both trout and coho
        continue
    elif len(df_vars.loc[i]) == 0:
        print('No obs for ' + str(i))
    else:
        s = [x for x in df.species.unique() if x != df_vars.loc[i].index[0]][0] # missing species
        temp = pd.DataFrame({'N_fish':0},index=[(i,s)])
        df_vars = df_vars.append(temp)
df_vars.sort_index(inplace=True)

df_vars = df_vars.fillna(0)

## Save
df_vars.to_csv(os.path.join(folder,'NOAA_data', 'fish_vars.csv'))

#%% Counts
print('Date Range: ' + str(df.date.iloc[0].date()) + ' to ' + str(df.date.iloc[-1].date()))
print(str(df.date.iloc[-1].date() - df.date.iloc[0].date()))
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
A = A.pivot(index='date',columns=['species','life_stage'],values='value')
A = A.reindex(index=date_range)  # reindex to entire date range
A = A.fillna(value=0)

### Adult / juvenille barcharts
plt.figure(figsize=(10,5))

plt.subplot(2,1,1)  # coho
plt.bar(A.index, A['coho','Juvenile'], label='Juvenile', color=pal[0])
plt.bar(A.index, A['coho','Adult'], bottom=A['coho','Juvenile'],label='Adult', color='k')

plt.ylabel('COHO')
plot_spines(plt.gca(),0)
plt.legend(frameon=False)

plt.yscale('log')
plt.ylim(0.5,1.1*A.max().max())

plt.subplot(2,1,2)  # trout
plt.bar(A.index, A['trout','Juvenile'], label='Juvenile', color=pal[1])
plt.bar(A.index, A['trout','Adult'], bottom=A['trout','Juvenile'],label='Adult', color='k')
plt.ylabel('TROUT')

plt.legend(frameon=False)

plt.yscale('log')
plt.ylim(0.5,1.1*A.max().max())
plot_spines(plt.gca(),0)
plt.tight_layout()

plt.savefig(os.path.join(folder.replace('data','figures'),'fish_counts_time_series.png'),dpi=300)


### Adult only
plt.figure(figsize=(10,5))

plt.subplot(2,1,1)  # coho
plt.bar(A.index, A['coho','Adult'], label='Adult', color=pal[0])
plt.ylabel('N')
#plt.ylim(0,1.1*A.max().max())
plt.legend(frameon=False)


plt.subplot(2,1,2)  # trout
plt.bar(A.index, A['trout','Adult'],label='Adult', color=pal[1])
plt.ylabel('N')
#plt.ylim(0,1.1*A.max().max())
plt.legend(frameon=False)

plt.tight_layout()

### Time of count histogram
# Mostly around 10a, many without timestamp (assume morning)
plt.figure()
df.dt.dt.hour.hist()
plt.title('Hour of Count')


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
plt.figure(figsize=(4,5))

plt.subplot(211)  # Adult
plt.bar(1.5*np.arange(0,len(season_plot.index)) + 1/4, season_plot['Adult','coho'], width=.5, color=pal[0])
plt.bar(1.5*np.arange(0,len(season_plot.index)) - 1/4, season_plot['Adult','trout'], width=.5, color=pal[1])
plt.xticks(ticks=1.5*np.arange(0,len(season_plot.index)), labels=season_plot.index)
plt.title('N Adults')
plt.yscale('log')
plt.ylim(1,season_plot.max().max())
#plt.legend(['coho','trout'], frameon=False)
plot_spines(plt.gca(), offset=0)

plt.subplot(212)  # Juvenile
plt.bar(1.5*np.arange(0,len(season_plot.index)) + 1/4, season_plot['Juvenile','coho'], width=.5, color=pal[0])
plt.bar(1.5*np.arange(0,len(season_plot.index)) - 1/4, season_plot['Juvenile','trout'], width=.5, color=pal[1])
plt.xticks(ticks=1.5*np.arange(0,len(season_plot.index)), labels=season_plot.index)
plt.title('N Juveniles')
plt.yscale('log')
plt.ylim(1,season_plot.max().max())
plt.legend(['coho','trout'], frameon=False)
plot_spines(plt.gca(), offset=0)

plt.tight_layout()

#%% Biomass time series
date_range = pd.date_range(df.date.iloc[0].date(), '05-01-2020')  # date range of datasets

# Mass (not available for every fish)
#A = (df.groupby(['species','life_stage','date']).sum()['mass'].rename('value')/1000).reset_index()

# Estimated mass (from regressions)
A = (df.groupby(['species','life_stage','date']).sum()['mass_est'].rename('value')/1000).reset_index()

A = A.pivot(index='date',columns=['species','life_stage'],values='value')
A = A.reindex(index=date_range)  # reindex to entire date range
A = A.fillna(value=0)

plt.figure(figsize=(10,5))

plt.subplot(2,1,1)  # coho
plt.bar(A.index, A['coho','Juvenile'], label='Juvenile', color=pal[0])
plt.bar(A.index, A['coho','Adult'], bottom=A['coho','Juvenile'],label='Adult', color='k')
#plt.plot([],color='b',lw=2,ls=':') # if loaded from analyze_ESP_eDNA.py
plt.ylabel('Mass (kg)')
plt.legend(frameon=False, loc = 'upper right')

## Hatchery releases
plt.scatter(hatch.index.unique(), 
            .25*A.max().max()*np.ones(len(hatch.index.unique())),
            s=18,color='k',marker='$H$') # 'v'

plt.yscale('log')
plt.ylim(0.5,1.1*A.max().max())

# ## eDNA overlay
# ax2 = plt.twinx()
# C.loc[C<0] = 0
# plt.plot(C,color='b',lw=2,ls=':') # if loaded from analyze_ESP_eDNA.py
# plt.ylabel('log$_{10}$(copies/mL)')

plt.xlim(date_range[0],date_range[-1])
plt.text(date_range[5], .75*A.max().max(),'O. kisutch')
plt.gca().set_xticklabels([])
plot_spines(plt.gca(),0)

plt.subplot(2,1,2)  # trout
plt.bar(A.index, A['trout','Juvenile'], label='Juvenile', color=pal[1])
plt.bar(A.index, A['trout','Adult'], bottom=A['trout','Juvenile'],label='Adult', color='k')
#plt.plot([],color='b',lw=2,ls=':') # if loaded from analyze_ESP_eDNA.py
plt.ylabel('Mass (kg)')
plt.legend(frameon=False, loc = 'upper right')

plt.yscale('log')
plt.ylim(0.5,1.1*A.max().max())

# ## eDNA overlay
# ax2 = plt.twinx()
# T.loc[T<0] = 0
# plt.plot(T,color='b',lw=2,ls=':') # if loaded from analyze_ESP_eDNA.py
# plt.ylabel('log$_{10}$(copies/mL)')

plt.xlim(date_range[0],date_range[-1])
plt.text(date_range[5], .75*A.max().max(),'O. mykiss')

plot_spines(plt.gca(),0)
plt.tight_layout()

plt.savefig(os.path.join(folder.replace('data','figures'),'fish_biomass_time_series.png'),dpi=300)
