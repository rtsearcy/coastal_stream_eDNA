#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:24:10 2021

@author: rtsearcy

Description: Create Master Standard Curves from combined qPCR standard results from 
all plates. Standard curves calculated from a regression between 
log10(quantity) and Ct value. 'Undetermined' standards values dropped before 
regression.

Curves for Individual plates and all STANDARD data combined (Master Curve)

Output: df - dataframe with only the raw STANDARD data from the combined qPCR results
        - Saved to standard_data.csv   
        
        df_stand - dataframe with standard curve slopes, intercepts, Rsq, and efficiency values/%s
        - Saved to standard_fits.csv
        
        Plots: trout and coho standard curves
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.interpolate as interpolate

def do_regression(data):                  
    ### Do OLS on log(Quantity) vs. Ct
    data = data[['Ct','logQ']].dropna()   # Drop rows with missing values
    data = data[data.Ct!='Undetermined']  # Remove 'Undetermined' Ct values
    Y = data['Ct'].astype('float')
    X = data['logQ']
    X = sm.add_constant(X)
    model = sm.OLS(Y,X).fit()
    r2 = round(model.rsquared, 5)
    slope = round(model.params.logQ, 8)   # Should be close to -3.3
    slope_LC = round(model.conf_int().loc['logQ'][0],8)  # Upper 95% CI
    slope_UC = round(model.conf_int().loc['logQ'][1],8)  # Lower 95% CI
    
    yint = round(model.params.const, 8)   # Expected value for a quantity of 1
    yint_LC = round(model.conf_int().loc['const'][0],8)  # Upper 95% CI
    yint_UC = round(model.conf_int().loc['const'][1],8)  # Lower 95% CI
    
    E = round(10**(-1/slope),3)           # Efficiency
    Ep = round(100*(E - 1),1)             # % Efficiency
    poor_E = 0
    if (Ep > 110) | (Ep < 90):
        poor_E = 1
    
    reg_dict = {'slope':slope, 'slope_LC':slope_LC, 'slope_UC': slope_UC,
                'intercept':yint, 'intercept_LC':yint_LC, 'intercept_UC': yint_UC,
                'Rsq':r2, 'E':E, '%E':Ep, 'E_Poor':poor_E}
    
    return reg_dict

def get_lod_loq(df, t, method):
    if method == 'klymus':
        print('Using Klymus et al. 2020 for LOD/LOQ Calculation')
        dfk = pd.read_csv('../data/eDNA/LOD_LOQ_R_script/Assay summary.csv')
        lod = float(dfk[dfk['Assay']==t]['LOD'])
        loq = float(dfk[dfk['Assay']==t]['LOQ'])
        
    else:
        df_lod = df.loc[df.target==t,['detect','Ct','logQ']].dropna()
        df_lod.Ct = pd.to_numeric(df_lod.Ct, errors='coerce')  # Set Undetermined to NaNs
        
        # med_stand = df_lod.groupby('logQ').median()  # Detect values far from median
        # for s in med_stand.index:
        #     med = med_stand.loc[s,'Ct']
        #     drop_ind = df_lod.loc[(df_lod.logQ == s) & ((df_lod.Ct>1.1*med)|(df_lod.Ct<0.9*med))].index
        #     df_lod = df_lod.drop(drop_ind)
        
        lm = sm.Logit(df_lod.detect,df_lod.logQ).fit(disp=False)
        print('Psuedo-R2: ' + str(round(lm.prsquared,3)))
        q = np.arange(0.01,3,0.01)
        idx = 1 + np.where(lm.predict(q) <= 0.95)[-1][-1]  # min logQ value where %prob of detection is 95%
        lod = 10**q[idx]  # copies/rxn
        
        ### Level of Quantification (LOQ)
        # df_loq = df_lod[df_lod.detect == 1].astype(float)
        # #CV = df_loq.groupby('logQ').std()/df_loq.groupby('logQ').mean()  # Coeffivient of variation (CV)
        # SD = df_loq.groupby('logQ').std().Ct
        # E = 1
        # CV = np.sqrt(((1+E)**((SD**2)*np.log(1+E))) - 1)
        # plt.figure()
        # plt.scatter(CV.index,CV.values)
        
        ### TODO: NEED TO FIT A SPLINE TO FIND LOQ (see wileys SM)
        # spline = interpolate.UnivariateSpline(CV.index,CV.Ct,k=4)
        # x = np.arange(0,6,0.01)
        # plt.plot(x,spline(x),'k',ls="--")
        # plt.xlabel('log(copies/rxn)')
        # plt.ylabel('CV')
        # plt.title(t)
        loq = lod
        
    print('\n' + t.upper() + ' LOD: ' + str(round(lod,2))  + ' copies/rxn' )
    print(t.upper() + ' LOQ: ' + str(round(loq,2))  + ' copies/rxn' )
    
    return lod, loq

### Inputs
folder = '../data/eDNA/'
df = pd.read_csv(os.path.join(folder,'qPCR_raw_combined.csv'), index_col=['index'])

drop_poor = False  # Drop Poor Efficiency Data from Plots and Master Curve

df = df[df.task == 'STANDARD']
df['logQ'] = np.log10(df['quantity'])   # log(copies/rxn)
df['detect'] = 1                        # detect/nondetect
df.loc[df.Ct=='Undetermined','detect'] = 0

df.to_csv(os.path.join(folder,'standard_data.csv')) # Save standard data separately

### Iterate through Targets
df_stand = pd.DataFrame()
plt.figure(figsize=(10,4)) # Plot of individual standard runs and combined
c=1  # subplot index
i=0  # standard plate index

for t in ['coho','trout']:
    print('\n- - Standards for ' + t + ' - -')
    poor_sum = 0  # Poor efficiency curve counter
    
    plt.subplot(1,2,c)
    files = df[df.target == t].source_file.unique()  # list of source files
    print('N (plates) - ' + str(len(files)))
    for f in files:  # Iterate individual standard runs
        data = df[(df.target == t) & (df.source_file==f)]
        
        if (data.Ct == 'Undetermined').sum() == len(data): # Check if all Undet.
            print('  skipping ' + f)
            continue
        
        ### Regression on individual standards
        reg = do_regression(data)
        reg['target'] = t
        reg['source'] = f
        df_stand = df_stand.append(pd.DataFrame(reg, index=[i]))
        i+=1
        
        df_plot = data.groupby('quantity').mean()
        
        if (reg['E_Poor']==1):  # Check if Poor Efficiency
            poor_sum += 1
            
        if (drop_poor) & (reg['E_Poor']==1): # Plot
            continue
        else:
            plt.semilogx(df_plot['Ct_Mean'], color='k',marker ='.', alpha=0.3, lw=0.75)
        
    ### Regression on combined standards
    if drop_poor:   # Drop data from poor eff. STANDARDS
        poor_source = df_stand[(df_stand.target==t) & (df_stand.E_Poor == 1)].source.unique()
        target_df = df[(df.target==t) & (~df.source_file.isin(poor_source))]
    else:
        target_df = df[df.target==t]
    
    miss_target = (target_df.Ct == 'Undetermined').sum()
    print('N (wells) - ' + str(len(target_df)))
    print('# Undetermined - ' + str(miss_target) + '\n')
    reg = do_regression(target_df)
    reg['target'] = t
    reg['source'] = 'combined'
    print(reg)    

    print('\nNo. Poor Efficiency: ' + str(poor_sum) + ' (' + 
          str(round(100*poor_sum/len(files),2)) + '%)')
    
    ### LOD/LOQ
    print('\nLOD/LOQ')
    lod, loq = get_lod_loq(target_df, t, method='klymus')  
    reg['LOD'] = lod
    reg['LOQ'] = loq
    df_stand = df_stand.append(pd.DataFrame(reg, index=[i])) # add to save file
    i+=1
    
    #### Plot combined fit (Master Curve)
    x = df.quantity.dropna().sort_values().unique()   # dilutions
    y = reg['slope']*np.log10(x) + reg['intercept']    
    plt.semilogx(x,y,color='r',alpha=.75,lw=2.5)
    # Note CIs too small to see on plot
    
    # Plot LOD
    plt.axvline(x=lod, color='k',alpha=0.5,ls=':')
    
    plt.xlim(min(x),1000000)
    plt.ylim(15,40)
    plt.xlabel('log$_{10}$copies/rxn')
    plt.ylabel('$C_T$')
    plt.title('Target = ' + t.upper())
    
    reg_text = 'Slope: ' + str(round(reg['slope'],3)) + \
                        ' [' + str(round(reg['slope_LC'],3)) + ', ' + str(round(reg['slope_UC'],3))  + ']' \
                        '\nInt: ' + str(round(reg['intercept'],3)) + \
                        ' [' + str(round(reg['intercept_LC'],3)) + ', ' + str(round(reg['intercept_UC'],3))  + ']' \
                        '\n$R^2$: ' + str(reg['Rsq']) + \
                        '\nEff: ' + str(reg['E']) + ' (' + str(reg['%E']) + '%)'
    plt.text(8,18,reg_text, bbox=dict(facecolor='white', lw = 1)) # add regression text
    
    c += 1
    
plt.tight_layout()
plt.savefig(os.path.join(folder,'standard_curves_combined.png'),dpi=300)

### Save
df_stand.to_csv(os.path.join(folder,'standard_fits.csv'),index=False)# Save standard fits
    