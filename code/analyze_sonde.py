#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 22:13:00 2021

@author: rtsearcy

EDA on Met Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

folder = '../data/sonde/'
sonde = pd.read_csv(os.path.join(folder,'sonde','YSI_6000_20191218_Scott_Creek.csv'))
units = sonde.iloc[0]  # Units
sonde = sonde.iloc[1:]
sonde['dt'] = pd.to_datetime(sonde['dt'])
sonde.set_index('dt',inplace=True)
sonde = sonde.astype(float)  # Skip units header

#%% Plots

# Plot parameters
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

df[['Temp','Cond','Sal','Turbid+','Depth','Chl',]].plot(subplots=True)

# # Temp, Rad, Rain Time Series
# fig = plt.figure(figsize=(10,5.5))

# plt.subplot(2,1,1)  # Avg Temp/Rad (7-day rolling average to smooth)
# plt.plot(dfd[['temp_avg']].rolling(window=7, center=True).mean(),color='k')
# plt.ylabel('Avg. Temperature (C)')
# ax2 = plt.gca().twinx()
# ax2.plot(dfd[['rad_avg']].rolling(window=7, center=True).mean(),color='r',alpha=0.75,ls='--')
# plt.ylabel(r'Avg. Solar Irradiance ($W/m^2$)',color='r',alpha=0.75, 
#            rotation=270, ha='center', va='baseline', rotation_mode='anchor' )
# plt.setp(ax2.get_yticklabels(), color="red", alpha=0.75)
# ax2.set_xticklabels([])

# plt.subplot(2,1,2)
# plt.plot(dfd['rain_total'],alpha=0.75)
# plt.ylabel('Precipitation (mm)')

# plt.tight_layout()