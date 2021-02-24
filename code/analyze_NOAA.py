#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Parameters / Load Data
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

NOAA fish count / gage data

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os


folder = '../data/NOAA_data'  # Data folder

### Load

# Gage Data - Water temp, creek stage
df_gage = pd.read_csv(os.path.join(folder,'ScottCreek_WY2019_GageData_101618_093019.csv'), 
                 parse_dates = ['dt'], index_col=['dt'])

gage_means = df_gage.reset_index().resample('d', on='dt').mean()

plt.figure()
plt.plot(gage_means.temp)
plt.ylabel('Temperature (C)')
ax = plt.gca()
ax2 = plt.twinx(ax)
plt.plot(gage_means.stage_ft, color='r')
plt.ylabel('Stage (ft)', color='r')
plt.yticks(color='r')

#%% Trap Data - Adult/Juvenile Steelhead and Coho counts
df_trap = pd.read_csv(os.path.join(folder,'ScottCreek_TrapSummary_100118_053119.csv'), 
                 parse_dates = ['date'], index_col=['date'])

# Weir
plt.figure()
plt.title('Weir Trap')
df_trap.Adult_Coho_Weir.plot()
df_trap.Adult_Sthd_Weir.plot()
plt.legend(['coho','trout'])

# Smolt Trap
plt.figure()
plt.title('Juvenilles - Smolt Trap')
JC = df_trap.Juv_Coho_SmltTrap
JS = df_trap.Juv_Sthd_SmltTrap
JC.dropna()[JC.str.isnumeric().dropna()].astype(int).plot(marker='.')
JS.dropna()[JS.str.isnumeric().dropna()].astype(int).plot()
plt.legend(['coho','trout'])

#%% Hatchery Data - Adult/Juvenile Steelhead and Coho counts
df_hatch = pd.read_csv(os.path.join(folder,'Coho_Releases_NOAA_Mar_Dec2019.csv'), 
                 parse_dates = ['date'], index_col=['date'])
plt.figure()
for s in df_hatch.site.unique():
    temp = df_hatch[df_hatch.site==s]
    plt.semilogy(temp['count'], marker = '.', label=s)
    
plt.ylabel('N')
plt.title('Hatchery releases')
plt.legend(frameon=False)
    

