#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 11:34:10 2021

@author: rtsearcy

Using Master Standard Curves, convert Ct values to concentration (copies/rxn),
accounts for dilution

Output: df - dataframe of eDNA concentrations with sample ID, target, dilution
replicate, and Ct data; saved to qPCR_calculated.csv
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

folder = '../data/eDNA/'

### Load Standard Curves
df_stand = pd.read_csv(os.path.join(folder,'standard_fits.csv'))
df_stand = df_stand[df_stand.source == 'combined']  # Master curve

### Load qPCR Data
df = pd.read_csv(os.path.join(folder,'qPCR_raw_combined.csv'), index_col=['index'])
df = df[(df.task != 'STANDARD') & (df.task != 'NTC')]  # remove standard and NTC data

print('qPCR data and standards loaded')
print('\n- - Replicates - -')
print('N (samples) - ' + str(len(df)))
df.Ct = pd.to_numeric(df.Ct, errors = 'coerce')  # convert Undetermined to NaN
df = df[~df.id.isna()]                           # drop samples with no id/dilution


### Calculate reaction concentration (copies / rxn)
df['conc'] = np.nan  # preload column
for t in df.target.unique():
    data = df[df.target==t].dropna(subset=['Ct']) # Drop Undetermined
    
    slope = float(df_stand[df_stand.target==t]['slope'])
    intercept = float(df_stand[df_stand.target==t]['intercept'])
    x = (data.Ct.astype(float) - intercept) / slope  # y = mx + b

    df.loc[x.index, 'conc'] = 10**x  # copies/rxn
    
    
### Drop High Conc. Outliers 
# Samples with only 1 amplified at a high conc.(> 10,000 copies/rxn)
print('\nOutliers (single reps > 10,000 copies/rxn')
outlier = (df[df.conc > 10000].groupby(['id','target','dilution']).count().replicate == 1)
outlier_ids = outlier[outlier == True].index
for o in outlier_ids:
    print(o)
    df.loc[(df.id == o[0]) & (df.target == o[1]) & (df.dilution == o[2]) & (df.conc > 10000),'conc'] = np.nan


### Check replicates amplification
und = df[np.isnan(df.Ct)]
print('\nN (%) Undetermined - ' + str(len(und)) + ' ('+ str(round(100*len(und)/len(df),1)) +'%)')
for t in df.target.unique():
    print('   ' + t + ' - ' + str(len(und[und.target==t])))
df['amplified'] = 1
df.loc[und.index,'amplified'] = 0

reps_ampd = df.groupby(['id','target','dilution']).count()['conc']  # Num reps/sample amplified
reps_ampd.name = 'replicates'
print('\nNumber of replicates amplified per sample')
print(reps_ampd.reset_index().groupby(['dilution','replicates']).count()['id'])  # all samples
# reps_ampd.reset_index().groupby(['target','dilution','replicates']).count() # by target/dilution

rep_check = reps_ampd[reps_ampd == 1].reset_index()  # Check mean concentration of remaining detected reps
rep_check['mean_conc']= np.nan
for r in range(0,len(rep_check)):
    R = rep_check.loc[r]
    mc = df[(df.id == R.id) & (df.target == R.target) & (df.dilution == R.dilution)]['conc'].mean()
    rep_check.loc[r,'mean_conc'] = mc
rep_check.groupby(['target','dilution']).mean()
rep_check.groupby(['target','dilution']).quantile(.95)

# Dilution pairs amplification
A = reps_ampd.reset_index().set_index(['id','target'])
B = A[A.dilution=='1:1']['replicates']
C = A[A.dilution=='1:5']['replicates']
D = pd.merge(B,C, left_index=True,right_index=True)
D.columns = ['1:1','1:5']
E = D.reset_index()
F = E[E.target=='coho']
G = E[E.target=='trout']
print('\nDilution pair amplification rates')
print(pd.crosstab(D['1:1'], D['1:5']))


### LOD/LOQ
print('\nLOD/LOQ')
df['BLOD'] = 0
df['BLOQ'] = 0
for t in df.target.unique():
    lod = float(df_stand[df_stand.target==t]['LOD'])
    loq = float(df_stand[df_stand.target==t]['LOQ'])
    
    # Replace non-detects and detected samples below the LOD: Use half the LOD
    df.loc[(np.isnan(df.conc)) & (df.target==t),'conc'] = 0.5 * lod  
    df.loc[(df.conc < lod)  & (df.target==t),'conc'] = 0.5 * lod

    # Index LOD/LOQ
    print('\n  ' + t.upper() + ' (' + str(int(lod)) + '/' + str(int(loq)) +')')
    df.loc[(df.target==t) & (df.conc < lod),'BLOD'] = 1
    print('  BLOD: ' + str(df.loc[(df.target==t),'BLOD'].sum()))
    df.loc[(df.target==t) & (df.conc < loq),'BLOQ'] = 1
    print('  BLOQ: ' + str(df.loc[(df.target==t),'BLOQ'].sum()))

### Factor Dilutions
# df.loc[df.dilution == '1:5','conc'] *= 5
df.loc[(df.dilution == '1:5') & (df.BLOD == 0 ),'conc'] *= 5  # convert conc > LOD
df['log10conc'] = np.log10(df.conc)

### Save and Plot All Replicate Concentrations
plt.figure()
plt.subplot(1,2,1)  # Ct
for t in df.target.unique():
    plt.hist(df[df.target==t].Ct,bins=100, histtype='step')
plt.xlabel(r'C$_t$')
plt.legend(list(df.target.unique()), frameon=False, loc='upper left')

plt.subplot(1,2,2)  # log(copies/rxn)
sns.boxplot(x='target', y='log10conc', data=df)
plt.ylabel(r'log$_{10}$(copies/rxn)')
plt.xlabel('')

plt.tight_layout()

print('\nCt converted to concentrations')
df = df[['id', 
         'target', 
         'dilution', 
         'replicate',
         'conc',       # Calculated from Ct and regression
         'log10conc',
         'amplified',
         'BLOD',
         'BLOQ',
         'Ct', 
         'Ct_Mean',
         'Ct_sd',
         #'quantity',    # From PCR machine
         'well',
         'source_file']]

df.to_csv(os.path.join(folder,'qPCR_calculated_all_reps.csv'),index=False)  # Save to same CSV


### Calculate Replicate Means
print('\n- - Sample Mean - -')
df_mean = df.copy()
df_mean['n_replicates'] = 0
df_mean['n_amplified'] = 0
df_mean['n_BLOD'] = 0
df_mean['n_BLOQ'] = 0
df_mean['conc_mean'] = np.nan
df_mean['conc_sd'] = np.nan   # replicate error = std
df_mean['log10conc_mean'] = np.nan
df_mean['log10conc_sd'] = np.nan 
print('\nAveraging replicates (N=' + str(len(df)) + ') ...')
for i in df_mean.id.unique():               # iterate through samples with sample IDs
    for t in df_mean.target.unique():       # iterate through target names
        for d in df_mean.dilution.unique(): # iterate through dilutions
            idx = (df_mean.id == i) & (df_mean.target == t) & (df_mean.dilution == d)
            df_mean.loc[idx, 'conc_mean'] = df_mean.loc[idx,'conc'].mean()
            df_mean.loc[idx, 'conc_sd'] = df_mean.loc[idx,'conc'].std()
            df_mean.loc[idx, 'log10conc_mean'] = df_mean.loc[idx,'log10conc'].mean()
            df_mean.loc[idx, 'log10conc_sd'] = df_mean.loc[idx,'log10conc'].std()
            df_mean.loc[idx, 'n_replicates'] = int(idx.sum())
            df_mean.loc[idx, 'n_amplified'] = int(df_mean.loc[idx,'amplified'].sum())
            df_mean.loc[idx, 'n_BLOD'] = int(df_mean.loc[idx,'BLOD'].sum())
            df_mean.loc[idx, 'n_BLOQ'] = int(df_mean.loc[idx,'BLOQ'].sum())
            
df_mean = df_mean[~df_mean[['id','target','dilution']].duplicated()] # drop duplicated rows so to keep means
df_mean.drop(['conc','log10conc'], axis=1,inplace=True)
df_mean.rename(columns={'conc_mean':'conc','log10conc_mean':'log10conc'}, inplace=True) # rename conc_mean

df_mean['BLOD'] = 0   # Declare BLOD/BLOQ for sample mean
df_mean['BLOQ'] = 0
for t in df_mean.target.unique():  
    for d  in df_mean.dilution.unique():
        dil_factor = int(d[-1])
        lod = float(df_stand[df_stand.target==t]['LOD']) * dil_factor
        loq = float(df_stand[df_stand.target==t]['LOQ']) * dil_factor
        df_mean.loc[(df_mean.target==t) & (df_mean.conc < lod),'BLOD'] = 1
        df_mean.loc[(df_mean.target==t) & (df_mean.conc < loq),'BLOQ'] = 1
    

### Check ΔCt between dilution pairs
# Theory:
# Ct1 = slope*np.log10(c1) + intercept
# Ct5 = slope*np.log10(c5) + intercept
# Ct1 = Ct5 + log(5**slope) => Ct1 - Ct5 = delta_Ct  ~= -2.31
# In other words: 1:1 dilution meets threshold 2.3 cycles before 1:5

# Acceptable range: -2.8 < delta_Ct < -1.8  (Yamahara, Sassoubre, Boehm??)
# Inhibition: delta_Ct > -1.8  (1:5 amplified earlier than expected)
# Pippetting error: detla_Ct < -2.8 (over dilution)

inh_thresh = -1.8
over_dil_thresh = -2.8 

print('\nΔCt Dilutions Check:')
Ct = df_mean.groupby(['id','target','dilution']).mean().Ct_Mean.to_frame()
print('Total Samples: ' + str(len(Ct.dropna())))

plt.figure(figsize=(7,7))
c=1
df_delta = pd.DataFrame()  # to save delta_Ct values
for t in df_mean.target.unique():
    print('\n' + t.upper())
    ### Separate out dilution means
    targ = Ct.xs(t,level=1).dropna()
    print('  # Unique IDs: ' + str(len(targ.reset_index().id.unique())))
    fish1 = targ.xs('1:1',level=1)  # 1:1 dilutions
    fish5 = targ.xs('1:5',level=1)  # 1:5 dilutions
    fish = pd.merge(fish1,fish5, left_index=True, right_index=True).dropna()
    print('  # Dilution Pairs: ' + str(len(fish)))  # Some samples had 1 or both dilutions undetected
    fish.columns = ['1:1','1:5']
    fish['delta_Ct'] = fish['1:1'] - fish['1:5']  # ideally -2.3 (-1.8 to -2.8 OK)
    
    ### Num samples within acceptable range
    accept = (fish.delta_Ct >= -2.8) & (fish.delta_Ct <= -1.8)
    print('     Expected:  ' + str(accept.sum()) + ' (' + str(round(100*accept.sum()/len(fish),2))+'%)')
    
    ### Samples Inhibited
    inhibited = (fish.delta_Ct > inh_thresh)
    print('     Inhibited:   ' + str(inhibited.sum()) + ' (' + str(round(100*inhibited.sum()/len(fish),2))+'%)')
    
    ### Samples Overdiluted
    overdiluted = (fish.delta_Ct < over_dil_thresh)
    print('     Overdiluted: ' + str(overdiluted.sum()) + ' (' + str(round(100*overdiluted.sum()/len(fish),2))+'%)')
    
    ### Delta_Ct Distributions
    fish_id = (fish.delta_Ct >= -100) & (fish.delta_Ct <= 100)
    #fish_id = (fish.delta_Ct >= -2.8) & (fish.delta_Ct <= -1.8)
    #print('\ndropped: ' + str(len(fish) - fish_id.sum()))
    fish = fish[fish_id]  # Keep only delta_Ct between these values
    #print(fish.delta_Ct.describe())
    
    ### Regression between dilutions for all pairs
    X = fish['1:5']
    X = sm.add_constant(X)
    lm = sm.OLS(fish['1:1'],X).fit()
    print('\nRegression on all pairs:')
    print(lm.params)
    m = lm.params['1:5']
    b = lm.params['const']
    r2 = lm.rsquared
    print('Rsq - ' + str(round(r2,3)))
    
    ### Scatterplot
    plt.subplot(2, 2, 2*c-1)
    plt.scatter(fish['1:5'], fish['1:1'],s=3, color='k')
    x = np.arange(25,50)
    plt.plot(x, m*x + b,c='k', lw=1)       # regression line 
    plt.plot(x,x - 2.32,c='g',ls=':',lw=1.5) # Ideal line (ideal: m=1,b = -2.32)
    plt.fill_between(x, y1=x + over_dil_thresh, y2=x - 2.32, color='g', edgecolor='w', alpha=0.3) # Acceptable range
    plt.fill_between(x, y1=x + inh_thresh, y2=x - 2.32, color='g', edgecolor='w', alpha=0.3)
    plt.ylim(25,45)
    plt.xlim(25,45)
    plt.ylabel('C$_t$ (1:1)')
    plt.xlabel('C$_t$ (1:5)')
    
    plt.text(25.5,43.75,t.upper() + ' (N = ' + str(len(fish))+')') # target / N pair
    # reg_text = 'Slope: ' + str(round(m,3)) + \
    #                     '\nInt: ' + str(round(b,3)) + \
    #                     '\n$R^2$: ' + str(round(r2,3))
    reg_text = 'In Range: ' + str(accept.sum()) + \
                        '\nInhibited: ' + str(inhibited.sum()) + \
                        '\nOverdiluted: ' + str(overdiluted.sum())
    plt.text(25.7,40.5,reg_text, fontsize=9.5) # add regression text
    
    ### Histogram
    plt.subplot(2,2,2*c)
    plt.hist(fish.delta_Ct, bins=50, color='k',alpha=.4)
    plt.axvline(-2.32,c='k', ls=':', lw=1.5) # Ideal line and acceptable range
    plt.axvline(over_dil_thresh,c='g', lw=1)
    plt.axvline(inh_thresh,c='g', lw=1)
    
    plt.xlabel(r'ΔC$_t$ [(1:1) - (1:5)]')
    
    fish['target'] = t
    df_delta = df_delta.append(fish)
    c+=1
    
plt.tight_layout()
plt.savefig(os.path.join(folder,'delta_Ct.png'),dpi=500)


### Use ΔCt to Indicate Inhibition
df_delta['dilution'] = '1:1'
df_delta = df_delta.reset_index().set_index(['id','target','dilution']).sort_index()
df_delta = pd.merge(Ct,df_delta,how='left',left_index=True,right_index=True)
df_delta = df_delta.reset_index().set_index(['id','target'])

df_mean['delta_Ct'] = np.nan
df_mean['inhibition'] = np.nan
for i in df_delta.index.unique():  
    dct = df_delta.loc[i,'delta_Ct'].max()
    df_delta.loc[i,'delta_Ct'] = dct   # Save delta_Ct to both dilutions
    
    df_mean.loc[(df_mean.id == i[0]) & (df_mean.target == i[1]),'delta_Ct'] = dct
    if dct > inh_thresh:
        df_mean.loc[(df_mean.id == i[0]) & (df_mean.target == i[1]),'inhibition'] = 1  # inhibited
    elif dct < over_dil_thresh:
        df_mean.loc[(df_mean.id == i[0]) & (df_mean.target == i[1]),'inhibition'] = -1 # overdiluted
    elif (dct >= -2.8) & (dct <= -1.8):
        df_mean.loc[(df_mean.id == i[0]) & (df_mean.target == i[1]),'inhibition'] = 0  # in range
    
df_delta[['Ct_Mean','delta_Ct']].to_csv(os.path.join(folder,'delta_Ct.csv'))  # separate dataframe with delta_Ct

### Save reaction concentration means
df_mean = df_mean[[
         'id', 
         'target', 
         'conc',            # Mean of replicates
         'conc_sd',
         'log10conc',       
         'log10conc_sd',
         'BLOD',            # for mean
         'BLOQ',
         'n_replicates',
         'n_amplified',
         'n_BLOD',          # replicates BLOD/BLOQ
         'n_BLOQ',
         'Ct_Mean',
         'Ct_sd',
         'delta_Ct',
         'inhibition',
         'dilution', 
         'source_file']]
df_mean = df_mean.sort_values(['id','target','dilution'])
df_mean.to_csv(os.path.join(folder,'qPCR_calculated_mean.csv'),index=False)  # Save
print('\nMean concentrations saved')
