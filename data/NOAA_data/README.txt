Readme
------

This document provides information about the  data files provided by NOAA  


I. File List
------------

fish_trap.csv
All fish handling records proximate to the Scott Creek weir. Includes data from the adult trap (weir) and the smelt trap (upstream). Record from 1/8/2019 to 5/21/2020 [N=8974 records] (wildfires occurred soon thereafter, but some counts from 6/1/2020 and 10/21/2020). Majority (92%) of records from smolt trap
Variables:
site - site of trap/capture 
species - coho, trout, or undetermined
origin - natural, hatchery, or capture (for broodstock)
sex - male, female, undetermined
life_stage - life stage of capture fish
live_adult - if adult, was the fish alive?
length - body length (mm)
mass - body mass (g)
body_depth - body depth (cm)
Other vars: latitude, longitude, id, event, condition_code, notes  


weir_wtemp.csv
15 min water temperature at the Scott Creek Weir. Temperature (in Celsius) recorded using a Hobo data logger. Record from 1/1/2019 - 12/2/2020. No missing data


lagoon_wq.csv
15m water quality data collected from the Scott Creek lagoon. Record from 1/1/2019 to 8/21/20. 
No data from 7/6/2019-7/24/2019, 10/15/2019-10/16/2019, 11/16/2019-12/9/2019, 3/11/2020-4/3/2020 (71 days total). 
Variables:
wtemp - water temperature (Celsius)
pH - pH
depth - water depth (m)
sp_cond - specific conductance (microS/cm)
turb - turbidity (NTU, assumed)
DO - dissolved oxygen (mg/L)
DO% - dissolved oxygen %. 


hatchery_releases.csv
Coho smolt and parr release counts by the Monterey Bay Trout and Salmon Project between 1/1/2019 and 8/15/2020 (N=12 releases).
Note: release numbers differ from log received in early 2020 (see old files). Seem to be biomass based (not integers).

Release site locations:
Site	Site ID	Latitude	Longitude	Site Description/Notes				
Lower Scott	S0	37.047039	-122.226319	Lower Scott below NOAA weir				
Release Site 1	S1	37.080614	-122.246964	150 m upstream of Swanton Bridge (Cal Poly Apple Orchard)		
Release Site 2	S2	37.083081	-122.248275	At gate near CalPoly/Big Creek Lumber boundary				
Release Site 3	S3	37.095717	-122.251819	Coho release site #3				
Release Site 4	S4	37.087364	-122.249156	0.25 miles downstream of Bettencourt Creek Bridge			
Release Site 5	S5	37.099619	-122.252378	At Bettencourt Creek Bridge ( Purdy Ranch)				
N/A	SC	---	---	Various locations 				
Big Creek	BC	37.07457	-122.221611	Big Creek adjacent to Kingfisher Flat hatchery				


adult_trap_status.csv
juv_trap_status.csv
trap_status_processed.csv

Data indicating the condition of the weir and juvenile traps during the 2019/2020 trapping seasons
