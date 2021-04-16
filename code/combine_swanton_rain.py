#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregates swanton ranch rain gage data into a single file. Data downloaded
from HOBO tipping bucket rain gages using the HOBOware software

Created on Thu Apr  8 09:53:48 2021

@author: rtsearcy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

folder = '../data/swanton_rain/'

dfd = pd.DataFrame()  # Daily time series
dft = pd.DataFrame()  # All temp data

for f in os.listdir(os.path.join(folder,'gage_data','raw_csv')):
    if not f.endswith('.csv'):
        continue
    print(f)
    
### Identify station
    if 'Water' in f:
        station = 'Waterlab'
    elif 'LQC' in f:
        station = 'LQC'
    elif 'Landing' in f:
        station = 'LD'
    
### Read File    

    temp = pd.read_csv(os.path.join(folder,'gage_data','raw_csv',f), skiprows=1)
    
    date_col = [c for c in temp.columns if 'Date' in c][0]
    temp_col = [c for c in temp.columns if 'Temp' in c][0]
    rain_col = [c for c in temp.columns if 'Event' in c][0]
    
    temp = temp[[date_col,temp_col,rain_col]]
    temp.columns = ['dt','temp','rain']
    temp['dt'] = pd.to_datetime(temp.dt)
    temp.set_index('dt',inplace=True)
    
### Daily TS
    mean_temp = temp.temp.resample('D').mean()  # Mean temperature by day
    rain_total = temp.rain.resample('D').max().dropna().diff()  # Total rainfall each day
    temp_d = pd.concat([mean_temp, rain_total], axis=1).fillna(0)
    
    date_range = pd.date_range(mean_temp.index[0], mean_temp.index[-1]) # Ensure all days in date range present
    temp_d = temp_d.reindex(index=date_range)
    
    temp_d['station'] = station
    temp_d['file'] = f
    
    dfd = dfd.append(temp_d)
    
### Raw temp TS
    temp_t = temp.temp.dropna().to_frame()
    temp_t['station'] = station
    temp_t['file'] = f
    
    dft = dft.append(temp_t)
    

### Adjust, plot and save outputs

## Daily
dfd.index.name = 'dt'
dfd = dfd.reset_index().drop_duplicates(subset=['dt','station'])
dfd = dfd.reset_index().pivot(index='dt', columns = 'station', values=['temp','rain','file'])
dfd.sort_index(inplace=True)
dfd = dfd.reindex(pd.date_range(dfd.index[0],dfd.index[-1]))
dfd.index.name = 'date'

# Temperature TS
dfd.temp.rolling(7, center=True).mean().plot()
plt.legend(frameon=False)
plt.ylabel('Temperature (F)')
plt.tight_layout()
plt.savefig(os.path.join(folder,'swanton_temp_time_series.png'),dpi=300)


# Rain TS
dfd.rain.plot()
plt.legend(frameon=False)
plt.ylabel('Total Rain (in.)')
plt.tight_layout()
plt.savefig(os.path.join(folder,'swanton_rain_time_series.png'),dpi=300)

# Save
for s in dfd.temp.columns:
    dfd.xs(s,level=1,axis=1).to_csv(os.path.join(folder,s+'_swanton_daily_met.csv'))
    
    
## Hourly temp
dft.index.name = 'dt'
dft = dft.reset_index().drop_duplicates(subset=['dt','station'])
dft = dft.pivot(index='dt', columns = 'station', values=['temp','file'])
dft.sort_index(inplace=True)
dft.index.name = 'dt'

# Save
for s in dft.temp.columns:
    dft.xs(s,level=1,axis=1).to_csv(os.path.join(folder,s+'_swanton_hourly_airtemp.csv'))
    