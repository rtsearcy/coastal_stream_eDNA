README - code directory - Coastal Streams HF eDNA Project
Last update: RTS - January 2021

This directory contains the scripts used for analysis of the eDNA data collected by MBARI in Scott Creek.

- - - 

AGGREGATE DATA
- combine_ESP_logs.py
Reads individual ESP sampling logs and aggregates data into a single spreadsheet

- combine_qPCR.py
Reads results from individual 96-well qPCR runs; aggregates data into a single spreadsheet		
- get_standard_curve.py
Create Individual and Master Standard Curves from combined qPCR control results from all sample runs. Standard curves calculated from a regression between log10(quantity) and Ct value.

- Ct_to_quantity.py
Using Master Standard Curves, convert Ct values to quantities (copies/microliter)

- combine_eDNA.py
Aggregates the ESP and eDNA data
- Uses sample volumes to convert from copy/rxn to cop
- Accounts for dilution/replicate logic
- Accounts for LOD/LOQ

eDNA DATA ANALYSIS
- analyze_ESP_eDNA.py
Analyzes the ESP and eDNA data
	- ESP: Time of day, sampling volumes/rates, analysis by ESP
	- eDNA: Time Series, TBD


ENVIRONMENTAL/FISH DATA
analyze_NOAA.py
analyze_sonde.py	
get_met_CIMIS.py	
analyze_met.py		
	
