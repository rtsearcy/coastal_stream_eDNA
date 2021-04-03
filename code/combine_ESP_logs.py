#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

Reads individual ESP sampling logs and aggregates data into a single spreadsheet
"""

import pandas as pd
import numpy as np
import os

folder = '../data/ESP_logs'  # Location of ESP logs

### Aggregate logs into one df
df = pd.DataFrame()
for f in os.listdir(folder):
    if 'Scott_Creek' not in f:  # skip Readme file
        continue

    print(f)
    ESP = f.split('_')[0]  # which ESP was sampling

    df_temp = pd.read_csv(os.path.join(folder,f))  # Open file
    print('Rows: ' + str(len(df_temp)))
    df_temp['ESP'] = ESP
    df_temp['log_file'] = f

    df = df.append(df_temp)  # Add to agg. df

print('\nRows in Combined: ' + str(len(df)))
df.reset_index(inplace=True, drop = True)  # Reset index


### Drop columns we aren't using 
df.drop([w for w in df.columns if 'WCR' in w], axis=1, inplace=True)  # Drop WCR columns (no data)
df.drop([j for j in df.columns if 'Julian' in j], axis=1, inplace=True)  # Drop Julian columns (not precise)
df.drop(['Protocol','Extract_No','Extract_Abbr'], axis = 1, inplace = True)


### Rename columns
df.rename(columns = {'Start Date':'sample_wake',   # dt for Datetime
                     'Sample Start Date':'sample_start',
                     'Sample End Date':'sample_end',
                     'Time To Sample (hh:mm:ss)':'sample_duration',
                     'Target Volume (ml)':'vol_target', 
                     'Actual Volume (ml)':'vol_actual',
                     'Difference (ml)':'vol_diff',
                     'Lab_Field':'lab_field',
                     'Extract_ID':'id'
                     }, inplace=True)


### Convert times, account for time zone
for c in ['sample_wake','sample_start','sample_end']:  # convert to Datetime, account for DST
    df[c] = pd.to_datetime(df[c], utc=True)   # converts to UTC, 
    df[c] = df[c].dt.tz_convert('Etc/GMT+8')  # then subtract 8hrs to get to PST
    df[c] = df[c].dt.tz_localize(None)          # drop TZ info


### Create Midpoint time (for timestamping samples)
df['sample_mid'] = df.sample_start + (pd.to_timedelta(df['sample_duration'])/2)  # half way between start and end
df['sample_mid'] = df.sample_mid.dt.round('s')  # round to nearest second

# Set mid time for samples with only wake time
df.loc[df.sample_mid.isna(),'sample_mid'] = df.loc[df.sample_mid.isna(),'sample_wake'] 

### Sample Duration and Rate
df['sample_duration'] = round(pd.to_timedelta(df['sample_duration']).dt.seconds/60, 2)  # convert sample duration (minutes)
df['sample_rate'] = df.vol_actual / df.sample_duration  # avg. sampling rate

df.set_index('sample_mid', inplace = True)      # Sort by start date
df.sort_index(inplace=True)


### Identify Time of Day (Morn/Midday/Eve)
df['date'] = df.index.date                      # create date column
df['hour'] = df.index.round('H').hour           # create hour column (round to nearest hour)
df['morn_midday_eve'] = 2 # Sample collected morning/afternoon/evening; Evening > 5p
df.loc[df.index.hour<17,'morn_midday_eve'] = 1  # Midday - 11a - 5p
df.loc[df.index.hour<11,'morn_midday_eve'] = 0  # Morn < 11a


### Deployment index
c = 1
df['deployment'] = 1
for i in range(1,len(df)):
    if df.ESP.iloc[i] != df.ESP.iloc[i-1]:
        c += 1
    df.loc[df.index[i],'deployment'] = c

print('\n')    
print(df.groupby('deployment').agg(pd.Series.mode)[['ESP','lab_field']])  # Deployment type


### Save to new CSV
lab_control = df[df.lab_field.isin(['control ','control','lab'])]
print('\n# Lab/Control: ' + str(len(lab_control)))
print('\nMissing:')
print(df.isnull().sum())  # print missing 

df.to_csv(os.path.join(folder,'ESP_logs_combined.csv'))  
          