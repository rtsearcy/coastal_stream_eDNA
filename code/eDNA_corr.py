#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates correlation coefficients between two vectors

Created on Thu Mar 11 11:45:36 2021

@author: rtsearcy
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

def eDNA_corr(x, y, x_col=None, y_col=None, on=None, subset=None, corr_type='pearson'):
    ''' x, y - 1D vectors or 2D dataframes
            
        x_col, y_col - str, columns in the vectors to correlate
        
        on - str, the index or column to join and run the correlation on. 
        Often this is a date
            - If no entry, index of the vectors will be used
        
        subset - list of indices to test correlation on
        
        corr_type - 'pearson','spearman', or 'kendall', the type of
        correlation coefficient to calculate
        '''
    
    assert corr_type in ['pearson','spearman','kendall'], \
    'corr_type must be either pearson, spearmann, or kendall'
    
    
    ### Select columns to correlate
    if x_col is not None:                    # x is a dataframe, not a vector
        if on is None:                       # Use x index to join
            x = x[x_col]
        else:                                # Set x_col to index
            if x.index.name in x.columns:    # Drop index if shared column
                x = x.reset_index(drop=True)
            x = x[[x_col, on]]
            #x[on] = x[on].astype(str)
            x = x.set_index(on)
            x = x.iloc[:,0]                  # to series
        
    if y_col is not None:                    # y is a dataframe, not a vector
        if on is None:                       # Use y index to join
            y = y[y_col]
        else:                                # Set y_col to index
            if y.index.name in y.columns:    # Drop index if shared column
                y = y.reset_index(drop=True)
            y = y[[y_col, on]]
            #y[on] = y[on].astype(str)
            y = y.set_index(on)
            y = y.iloc[:,0]
        
    ### Align vectors
    x.index = x.index.astype(str)
    y.index = y.index.astype(str)
    df_corr = pd.merge(x, y, how = 'inner', left_index=True, right_index=True)
    df_corr = df_corr.dropna()  # Drop NaNs
    df_corr.sort_index(inplace=True)
    
    
    ### Grab subset
    if subset is not None:
        df_corr = df_corr[df_corr.index.isin(subset)]
    
    if x.name is None:
        x.name = x_col
    if y.name is None:
        y.name = y_col
    print(corr_type.capitalize() + ' correlation between ' + x.name + ' and ' + y.name)
    print(str(df_corr.index[0]) + ' to ' + str(df_corr.index[-1]) + ' (N=' + str(len(df_corr)) + ')')
    
    
    ### Calculate correlation coefficient
    if corr_type == 'spearman':
        rho, p = stats.spearmanr(df_corr.iloc[:,0], df_corr.iloc[:,1])
    elif corr_type == 'kendall':
        rho, p = stats.kendalltau(df_corr.iloc[:,0], df_corr.iloc[:,1])
    else:
        rho, p = stats.pearsonr(df_corr.iloc[:,0], df_corr.iloc[:,1])


    return round(rho, 5), p
