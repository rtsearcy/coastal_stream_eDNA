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

print('qPCR data loaded')
print('  N (samples) - ' + str(len(df)))

### Calculate copies / rxn
df['conc'] = np.nan  # preload column
for t in df.target.unique():
    data = df[df.target==t].dropna(subset=['Ct'])
    data = data[data.Ct != 'Undetermined']
    
    slope = float(df_stand[df_stand.target==t]['slope'])
    intercept = float(df_stand[df_stand.target==t]['intercept'])
    x = (data.Ct.astype(float) - intercept) / slope  # y = mx + b

    df.loc[x.index, 'conc'] = 10**x  # copies/rxn
    

### Outlier IDs
# Samples with only 1 amplified at a high conc.(> 10,000 copies/rxn)
(df[df.conc > 10000].groupby(['id','target','dilution']).count().replicate == 1)
outlier_ids = ['SCr-24','SCr-133','SCr-478','SCr-378']


### Check amplified vs. non-detects
reps_ampd = df.groupby(['id','target','dilution']).count()['conc']  # Num reps/sample amplified
reps_ampd.reset_index().groupby('conc').count()['id']  # all samples
reps_ampd.reset_index().groupby(['target','dilution','conc']).count() # by target/dilution

# Check concentration of remaining detected reps
rep_check = reps_ampd[reps_ampd == 2].reset_index()  
rep_check['mean_conc']= np.nan
for r in range(0,len(rep_check)):
    R = rep_check.loc[r]
    mc = df[(df.id == R.id) & (df.target == R.target) & (df.dilution == R.dilution)]['conc'].mean()
    rep_check.loc[r,'mean_conc'] = mc
rep_check.groupby(['target','dilution']).median()

# Dilution pairs amplification
A = reps_ampd.reset_index().set_index(['id','target'])
B = A[A.dilution=='1:1']['conc']
C = A[A.dilution=='1:5']['conc']
D = pd.merge(B,C, left_index=True,right_index=True)
D.columns = ['1:1','1:5']
E = D.reset_index()
F = E[E.target=='coho']
G = E[E.target=='trout']
pd.crosstab(D['1:1'],D['1:5'])

# Nondetects vs. amplified?
und = df[df.Ct=='Undetermined']  
print('  N (%) Undetermined - ' + str(len(und)) + ' ('+ str(round(100*len(und)/len(df),1)) +'%)')
df['amplified'] = 1
df.loc[und.index,'amplified'] = 0


### LOD/LOQ
df['BLOD'] = 0
df['BLOQ'] = 0
for t in df.target.unique():
    lod = float(df_stand[df_stand.target==t]['LOD'])
    loq = float(df_stand[df_stand.target==t]['LOQ'])
    
    # Replace Undetermined and samples under the LOD: Use half the LOD for non-amplified
    df.loc[(df['Ct']=='Undetermined') & (df.target==t),'conc'] = 0.5 * lod  
    df.loc[(df.conc < lod)  & (df.target==t),'conc'] = 0.5 * lod

    # Index LOD/LOQ
    print('  ' + t)
    df.loc[(df.target==t) & (df.conc < lod),'BLOD'] = 1
    print('   BLOD: ' + str(df.loc[(df.target==t),'BLOD'].sum()))
    df.loc[(df.target==t) & (df.conc < loq),'BLOQ'] = 1
    print('   BLOQ: ' + str(df.loc[(df.target==t),'BLOQ'].sum()))

### Factor Dilutions
df.loc[df.dilution == '1:5','conc'] *= 5

### Boxplots
plt.figure()
plt.subplot(1,2,1)  # Ct
for t in df.target.unique():
    plt.hist(pd.to_numeric(df[df.target==t].Ct, errors='coerce'),bins=100, histtype='step')
plt.xlabel(r'C$_t$')
plt.legend(list(df.target.unique()), frameon=False, loc='upper left')

plt.subplot(1,2,2)  # log(copies/rxn)
df['logconc'] = np.log10(df.conc)
sns.boxplot(x='target', y='logconc', data=df)
plt.ylabel(r'log$_{10}$(copies/rxn)')
plt.xlabel('')

plt.tight_layout()

### Save
print('\nCt converted to eDNA concentration')
df = df[['id', 
         'target', 
         'dilution', 
         'replicate',
         'conc',       # Calculated from Ct and regression
         'amplified',
         'BLOD',
         'BLOQ',
         'Ct', 
         #'Ct_Mean',
         #'Ct_sd',
         #'quantity',    # From PCR machine
         'well',
         'source_file']]
df.to_csv(os.path.join(folder,'qPCR_calculated.csv'),index=False)  # Save to same CSV

#%% delta_Ct between 1 and 5 dilutions
# Theory:
# Ct1 = slope*np.log10(c1) + intercept
# Ct5 = slope*np.log10(c5) + intercept
# Ct1 = Ct5 + log(5**slope) => Ct1 - Ct5  ~= -2.31
# In other words: 1:1 dilution meets threshold 2.3 cycles before 1:5


print('\n\nDilutions:')
df.Ct = pd.to_numeric(df.Ct, errors = 'coerce')
Ct = df.groupby(['id','target','dilution']).mean().Ct

plt.figure(figsize=(10,10))
c=1
for t in df.target.unique():
    print('\n' + t)
    ### Separate out dilution means
    fish1 = Ct.xs(t,level=1).xs('1:1',level=1)  # trout 1:1 dilutions
    fish5 = Ct.xs(t,level=1).xs('1:5',level=1)  # 1:5 dilutions
    fish = pd.merge(fish1,fish5, left_index=True, right_index=True).dropna()
    fish.columns = ['1:1','1:5']
    fish['delta'] = fish['1:1'] - fish['1:5']  # ideally -2.3
    
    fish_id = (fish.delta >= -10.75) & (fish.delta <= 10)
    print('dropped: ' + str(len(fish) - fish_id.sum()))
    fish = fish[fish_id]  # Keep only delta_Ct between these values
    print(fish.delta.describe())
    
    X = fish['1:5']
    X = sm.add_constant(X)
    lm = sm.OLS(fish['1:1'],X).fit()
    print('\nRegression:')
    print(lm.params)
    m = lm.params['1:5']
    b = lm.params['const']
    r2 = lm.rsquared
    print('Rsq - ' + str(round(r2,3)))
    
    x = np.arange(25,50)
    
    # Scatterplot
    plt.subplot(2, 2, 2*c-1)
    plt.scatter(fish['1:5'], fish['1:1'],s=3, color='k')
    plt.plot(x, m*x + b,c='k', lw=1)  # regression line (ideal: m=1,b = -2.32)
    plt.ylim(25,45)
    plt.xlim(25,45)
    plt.ylabel('1:1 dilution')
    plt.xlabel('1:5 dilution')
    
    plt.text(26,44,t)  # target
    reg_text = 'Slope: ' + str(round(m,3)) + \
                        '\nInt: ' + str(round(b,3)) + \
                        '\n$R^2$: ' + str(round(r2,3))
    plt.text(26,40,reg_text) # add regression text
    
    # Histogram
    plt.subplot(2,2,2*c)
    plt.hist(fish.delta, bins=30)
    plt.text(.95*plt.xlim()[0],.95*plt.ylim()[1],'N - ' + str(len(fish)))  # N samples
    plt.xlabel('C$_t$ (1:1) - C$_t$ (1:5)')
    c+=1
    
plt.tight_layout()
