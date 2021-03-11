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


### Load qPCR Data
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
#df = df[df.dilution == '1:5'] # Use 1:5 diluted samples only
#df = df[df.dilution == '1:1'] # Use undiluted samples only

# Check inhibition in each sample to decide which dilution to use
for i in df.id.unique():
    for t in df.target.unique():
        idx = (df.id==i) & (df.target==t)
        inh = df.loc[idx,'inhibition'].max()
        if inh == 1:            # If inhibition, use 1:5 dilution
            df.drop(df.loc[idx & (df.dilution=='1:1')].index, inplace=True)
        elif inh in [0,-1]:     # If in range or overdiluted, use 1:1 dilution
            df.drop(df.loc[idx & (df.dilution=='1:5')].index, inplace=True)
        else:  # Inh. = NAN
            if idx.sum() == 2:  # If both samples undetected, use 1:1
                  df.drop(df.loc[idx & (df.dilution=='1:5')].index, inplace=True)
        
        
### Create date variables
df['dt'] = df['sample_mid']  # timestamp
df['date'] = df['dt'].dt.date
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

        
### Save aggregate dataframe
df['ESP_file'] = df['log_file']
df['qPCR_file'] = df['source_file']

df = df[['dt',
         'date',
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
         'inhibition',
         #'delta_Ct',
         'ESP',
         'vol_actual',
         'sample_duration',
         'sample_rate',
         'year',
         'month',
         'year_month',
         'season',
         'wet_season',
         'week',
         'year_week',
         'morn_midday_eve',
         'ESP_file',
         'qPCR_file']]
df.set_index(['id','target'], inplace=True)
df.sort_values(['dt','target'], inplace=True)  # sort by date
df.to_csv(os.path.join(folder,'eDNA','eDNA.csv'))
print('ESP and qPCR data aggregated. eDNA concentrations saved.')
print('\nN = ' + str(len(df)))
print(df.reset_index().groupby('target').count()['id'])

print('\nN (BLOD)')
print(df.reset_index().groupby('target').sum()['BLOD'])
