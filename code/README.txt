README - code directory - Coastal Streams HF eDNA Project
Last update: RTS - September 2021

This directory contains the scripts used for analysis of the eDNA data collected by MBARI in Scott Creek.

- - - 

AGGREGATE DATA

- combine_ESP_logs.py
Reads individual ESP sampling logs and aggregates data into a single spreadsheet

- combine_qPCR_files.py
Reads results from individual 96-well qPCR runs; aggregates data into a single spreadsheet		
- get_standard_curve.py
Create Individual and Master Standard Curves from combined qPCR control results from all sample runs. Standard curves calculated from a regression between log10(quantity) and Ct value.

- qPCR_to_rxn_conc.py
Using Master Standard Curves, convert Ct values to quantities (copies/microliter)

- calculate_eDNA.py
Aggregates the ESP and qPCR data into eDNA concentration
- Uses sample volumes to convert from copy/rxn to cop
- Accounts for dilution/replicate logic
- Accounts for LOD/LOQ



eDNA DATA ANALYSIS

- analyze_ESP.py
Stats and Plot of the ESP data
	- Time of day, sampling volumes/rates, analysis by ESP

- analyze_eDNA_temporal.py
Temporal behaviour of eDNA data

- analyze_eDNA_enviro_fish.py
Compare eDNA data to environmental and fish count data


- eDNA_corr.py
Function to correlate two time series using Pearson, Spearman, or Kendall correlations

ENVIRONMENTAL/FISH DATA

- analyze_fish_trap.py
Stats and Plots for Fish Trap data; Create daily fish variables

- get_usgs_flow.py
Obtain USGS streamflow data from online	
	
