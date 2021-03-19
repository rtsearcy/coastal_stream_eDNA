#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% Parameters / Load Data
"""
Created on Jan 29 2021

@author: rtsearcy

Description: Creates a data inventory that documents current data availability
for the project

Outputs:
    - Bar chart-like graphic (see NOAA CO-OPS)
    - Table w/ # Missing days per category

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import os
import datetime
from eDNA_corr import eDNA_corr

def print_df(df, date_range):
    print('  N obs.: ' + str(len(df)))
    print('  Range: ' + str(df.index[0].date()) + ' to ' + str(df.index[-1].date()))
    
    df['date'] = df.index.date
    miss = len(date_range) - len([i for i in date_range if i in list(df['date'])])
    
    # Days from project range that are missing
    print('  Missing Project Days: ' + str(miss) + ' (' + str(round(100*miss/len(date_range),2)) + '%)\n')
    
    # Missing observations within record
    tempc = df.isnull().sum()
    tempp = round(100*tempc/len(df),1)  # % missing
    tempc.name = 'Missing'
    tempp.name = '%'
    print(pd.merge(tempc,tempp,left_index=True,right_index=True))
    
def fill_plot(df, plot_range, y, c):
    # Plots a filled area plot indicating data availability
    # across a range (plot_range) for a dataframe (df)
    # y - y axis location (bottom of bar)
    # c - color
    
    for i in plot_range:
        if i.date() in list(df.date):
            s = i.date()
            e = s + datetime.timedelta(1)
            plt.fill_between(x=[s,e], y1=y, y2=y+0.9, facecolor=c, edgecolor=None, alpha=0.3)   
        

### Plot parameters
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

pal = ['#969696','#525252']  # grey, black
pal = sns.color_palette(pal)

pal4c = ['#253494','#2c7fb8','#41b6c4','#a1dab4'] # grey, black
pal4c = sns.color_palette(pal4c)


### Inputs
folder = '../data/'  # Data base folder

today = str(datetime.datetime.today())[0:10]
print('- - - DATA INVENTORY (AS OF ' + today + ') - - -')

## Project Start/End
ps = '2019-03-25' # Project start
pe = '2020-04-16' # end
date_range = pd.date_range(ps,pe)
plot_range = pd.date_range('2019-01-01','2020-07-01')
print('\nProject Dates: ' + ps + ' to ' + pe + \
      ' (' + str(len(date_range)) + ' d)')

#%% Load Data

### Combined ESP logs
# Contains sampling times, volumes, ESP name
ESP = pd.read_csv(os.path.join(folder,'ESP_logs','ESP_logs_combined.csv'), 
                 parse_dates = ['sample_wake','sample_start','sample_mid','sample_end','date'])  # can also use sample_start, but probably little diff.

ESP.dropna(inplace=True, subset=['sample_wake', 'sample_start', 'sample_end', 'sample_duration',
        'vol_target', 'vol_actual', 'vol_diff',])  # Drop samples with not time or volume data 

ESP.set_index('sample_wake',inplace=True)

#ESP = ESP[~ESP.lab_field.isin(['lab','control', 'control '])]  # Keep deployed/field/nana; Drop control/lab samples

print('\nESP\n  Freq: 1-3x daily')
print_df(ESP, date_range)


### eDNA Data
# Contains target, replicate #,dilution level, concentration (post-dilution factor)
eDNA = pd.read_csv(os.path.join(folder,'eDNA','eDNA.csv'), parse_dates=['dt','date'])
#eDNA['eDNA'] = eDNA.conc
#eDNA.drop('conc', axis=1, inplace=True)

print('\neDNA Samples\n')
print('   Unique Sample IDs: ' + str(len(eDNA.id.unique())))
eDNA_miss = [i for i in ESP.id.unique() if i not in eDNA.id.unique()]  # List of missing sample IDs
ESP_miss =  [i for i in eDNA.id.unique() if i not in ESP.id.unique()]  # IDs in qPCR data not in ESP data
print('   Missing Sample IDs: ' + str(len(eDNA_miss)))


### NOAA Data
# Contains daily fish count, NOAA stream gage, hatchery data
print('\n\nNOAA Data')

# Trap Data - Adult/Juvenile Steelhead and Coho counts
trap = pd.read_csv(os.path.join(folder,'NOAA_data', 'ScottCreek_TrapSummary_100118_053119.csv'), 
                 parse_dates = ['date'], index_col=['date'])
print('\nTrap Counts\n  Freq: daily')
print_df(trap, date_range)

# Hatchery Data - Adult/Juvenile Steelhead and Coho counts
hatch = pd.read_csv(os.path.join(folder,'NOAA_data', 'Coho_Releases_NOAA_Mar_Dec2019.csv'), 
                 parse_dates = ['date'], index_col=['date']).dropna()

print('\nHatchery Releases \n   Freq: Occasional')
print('   Releases: ' + str(len(hatch)))
print('   Sites: ' + str(hatch.site.unique()))
print('   Days: ' + str(len(hatch.index.unique())))
print(hatch.groupby(hatch.index).sum())

# Gage Data - Water temp, creek stage
gage = pd.read_csv(os.path.join(folder,'NOAA_data', 'ScottCreek_WY2019_GageData_101618_093019.csv'), 
                 parse_dates = ['dt'], index_col=['dt'])
print('\nStream Gage\n  Freq: 1 / 15 min')
print_df(gage, date_range)


### Load Met Data
# Daily AND Hourly:  Air/dew temp, wind speed, solar rad, precip
print('\n\nMeteorological')
daily_file = 'Pescadero_CIMIS_day_met_20190101_20200501.csv'
hourly_file = 'Pescadero_CIMIS_hourly_met_20190101_20200501.csv'

# Daily Data
met_d = pd.read_csv(os.path.join(folder,'met',daily_file), parse_dates=['date'], index_col=['date'])
met_d = met_d.dropna(how='all')
print('\nDaily')
print_df(met_d, date_range)

# Hourly Data
met_h = pd.read_csv(os.path.join(folder,'met',hourly_file), parse_dates=['dt'], index_col=['dt'])
met_h = met_h.dropna(how='all')
print('\nHourly')
print_df(met_h, date_range)


### Load Sonde Data
# Water temp and other WQ parameters
sonde = pd.read_csv(os.path.join(folder,'sonde','YSI_6000_20191218_Scott_Creek.csv'))
sonde = sonde.iloc[1:]
sonde['dt'] = pd.to_datetime(sonde['dt'])
sonde.set_index('dt',inplace=True)
sonde = sonde.astype(float)  # Skip units header

print('\n\nYSI Sonde\n  Freq: 1 / 15 min')
print_df(sonde, date_range)


#%% Data Inventory Plot
plt.figure(figsize=(12,6))
# 3 subplots: EPS/eDNA data, Fish data, Enviro data

### ESP/EDNA
# ESP data
fill_plot(ESP,plot_range,6,'b')
# eDNA plot points
have_ID = [i for i in eDNA.id.unique() if i  in ESP.id.unique()]  # IDs in qPCR data and in ESP data
have_date = list(ESP.loc[ESP.id.isin(have_ID),'date'])
plt.scatter(x = have_date , y = len(have_date)*[6.45], s = 8, c = 'k', marker = '.')

### Fish
# trap, hatch
fill_plot(trap,plot_range,5,'g')
# hatch plot points
hatch_date = hatch.index.unique()
plt.scatter(x = hatch_date , y = len(hatch_date)*[5.45], s = 24, c = 'k', marker = '*')

### Environmental Variables
# gage, sonde, met_d, met_h
fill_plot(gage,plot_range,4,'r')
fill_plot(sonde,plot_range,3,'k')
fill_plot(met_d,plot_range,2,'orange')

plt.yticks(ticks=[],labels=[])
plt.suptitle('Data Inventory for Coastal Stream eDNA Project (as of ' + \
              today + ')', fontsize=18)

plt.axvline(date_range[0],color='k', ls='--',lw=1) #Project Start
plt.axvline(date_range[-1],color='k', ls='--',lw=1) #Project End

plt.xlim(plot_range[0],plot_range[-1])

# Hide the right, left, and top spines
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)

# Legend
legend_elements = [
    Patch(facecolor='b', alpha=0.3, edgecolor=None,label='ESP'),
    Patch(facecolor='g', alpha=0.3, edgecolor=None,label='Trap'),
    Patch(facecolor='r', alpha=0.3, edgecolor=None,label='NOAA Gage'),
    Patch(facecolor='k', alpha=0.3, edgecolor=None,label='YSI Sonde'),
    Patch(facecolor='orange', alpha=0.3, edgecolor=None,label='Met Data'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=6, label='qPCR Done'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=10, label='Hatchery Release'),
    Line2D([0], [0], marker='', color='k', linestyle='--', linewidth=1, label='Project Start/End'),
                   ]
plt.legend(handles=legend_elements,
            frameon=False,  
            #bbox_to_anchor=(0,1.02,1,1.02),
            bbox_to_anchor=(0,-0.1,1,-0.1),
            loc="lower center", borderaxespad=0, ncol=8)

plt.tight_layout()
plt.subplots_adjust(
top=0.934,
bottom=0.098,
left=0.044,
right=0.954,
hspace=0.2,
wspace=0.2)


#%% Stats
trout = eDNA[eDNA.target=='trout'].set_index('id').sort_values('dt')
coho = eDNA[eDNA.target=='coho'].set_index('id').sort_values('dt')

### Correlation between eDNA and other vars
print('\nTROUT')
print(eDNA_corr(trout, met_d, x_col='log10eDNA', y_col='temp_avg', on='date',corr_type='spearman'))
print(eDNA_corr(trout, met_d, x_col='log10eDNA', y_col='rain_total', on='date',corr_type='spearman'))
print('\nCOHO')
print(eDNA_corr(coho, met_d, x_col='log10eDNA', y_col='temp_avg', on='date',corr_type='spearman'))
print(eDNA_corr(coho, met_d, x_col='log10eDNA', y_col='rain_total', on='date',corr_type='spearman'))
