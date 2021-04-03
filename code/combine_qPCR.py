#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:03:18 2021

@author: rtsearcy

Reads results from individual 96-well qPCR plates; aggregates data into a single spreadsheet

Output: df - dataframe with info on sample id, target, dilutions, replicate #,
        Ct values, system estimated quantities, and STANDARD/UNKNOWN classification
        
        Saves dataframe to qPCR_combined.csv
"""

import pandas as pd
import numpy as np
import os

folder = '../data/eDNA/'  # Location of qPCR files folder and output location
subfolder = 'qpcr_csv_results_040121'                 # location of up to date qPCR spreadsheets (from Google Drive)


### Aggregate individual qPCR result files
df = pd.DataFrame()
for f in os.listdir(os.path.join(folder,subfolder)):
    if 'readme' in f:  # skip Readme file
        continue
    
    print(f)
    if '.csv' in f:  # Open file
        df_temp = pd.read_csv(os.path.join(folder,subfolder,f), header=7)  
    elif '.xls' in f:
        df_temp = pd.read_excel(os.path.join(folder,subfolder,f), header=7)  
            
    print('Rows: ' + str(len(df_temp)) + '; Cols: ' + str(len(df_temp.columns)))
    df_temp['source_file'] = f
    df = df.append(df_temp)  # Add to agg. df

print('\nRows in Combined: ' + str(len(df)))
df.reset_index(inplace=True, drop=True)  # Reset index


### Drop columns we aren't using 
df.drop(['Reporter','Quencher', 'Automatic Ct Threshold',
         'Ct Threshold','Automatic Baseline', 'Comments',
         'Baseline Start', 'Baseline End', 'Comments', 'HIGHSD', 'NOAMP',
       'EXPFAIL','OUTLIERRG'], axis = 1, inplace = True)


#### Parse Sample ID and Dilutions
df[['id','dilution']] = df['Sample Name'].str.split('-1:', expand=True)
df.drop('Sample Name', axis=1, inplace=True)
df.loc[df.dilution.values == None,'dilution'] = '1:1'
df.loc[df.dilution.values == '5','dilution'] = '1:5'
                

### Rename targets and other columns
df.loc[df['Target Name'] == 'O.kitsuch','Target Name'] = 'O.kisutch'   # Some misspelling
df.loc[df['Target Name'].isin(['O.kisutch','Okisutch','Okitsuch']),'Target Name'] = 'coho'  # Coho Salmon
df.loc[df['Target Name'].isin(['O.mykiss','Omykiss']),'Target Name'] = 'trout'  # Rainbow/Steelhead Trout

df.rename(columns = {'Cт': 'Ct',                      
                     'Cт Mean': 'Ct_Mean',            
                     'Cт SD': 'Ct_sd',
                     'Target Name':'target',          
                     'Task':'task',                   
                     'Quantity':'quantity',           
                     'Quantity Mean':'quantity_mean', # Mead/SD of replicates
                     'Quantity SD':'quantity_sd',
                     'Well':'well'
                      }, inplace=True)


### Index Replicates
print('\nIndexing replicates, dropping repeated samples...')
df['replicate'] = np.nan
for i in df.id.unique():               # iterate through samples with sample IDs
    for t in df.target.unique():       # iterate through target names
        for d in df.dilution.unique(): # iterate through dilutions
            idx = df.loc[(df.id == i) & (df.target == t) & (df.dilution == d)]
            if len(idx)>3:             # remove duplicate samples
                new_source = idx.source_file.unique()[-1]  # identify the source file with newest data
                idx = idx[idx.source_file == new_source]   
            if len(idx)>0:             # add replicate #s
                df.loc[idx.index,'replicate'] = np.arange(1,len(idx)+1)

df = df.drop(df[(df.task=='UNKNOWN') & (df.replicate.isna())].index) # Drop duplicate ID replicates

# Check more than 3 replicates
rgt3 = len(df.loc[df.replicate > 3,'id'].unique())  
print('   # samples w > 3 replicates: ' + str(rgt3))


### Check NTC
NTC = df[df.task == 'NTC']
NTC_amp = NTC[NTC.Ct != 'Undetermined']
print('\n# NTC samples amplified: ' + str(len(NTC_amp)))


### Manually adjust IDs
df['id'] = df['id'].str.replace('--','-')
df['id'] = df['id'].str.replace('Scr','SCr')
df['id'] =  df.id.str.replace('-00','-')  # remove zero padding
df['id'] =  df.id.str.replace('-0','-')
               
 
### Save file
df.index.name = 'index'
# column order
df = df[['id',          # Sample ID
         'target',      # coho or trout
         'dilution',    # 1:1 or 1:5
         'replicate',   # integer (1:3 usually, sometimes more)
         'task',        # UNKNOWN, STANDARD, NTC
         'Ct',          # Threshold cycle 
         'Ct_Mean',     # Mean/SD of replicates  
         'Ct_sd', 
         'quantity',    # Calculated by PCR machine
         'quantity_mean',
         'quantity_sd',
         'well',
         'source_file']]  
df.to_csv(os.path.join(folder,'qPCR_raw_combined.csv'))  # Save to new CSV
print('\nSaved.')
          