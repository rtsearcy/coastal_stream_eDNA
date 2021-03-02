Readme
------

This document provides information about the raw data files downloaded from the ESP logs.  



I. File List
------------

Gordon_2019_Scott_Creek.csv			CSV file containing ESP Gordon sampling data from ESP logs. Four instrument deployments - (1) 6/25/19 to 8/22/19, (2) 11/21/19 to 11/25/19 and (3) 12/03/19 to 1/30/20, and (4) 2/10/20 to 4/04/20 
Moe_2019_Scott_Creek_1.csv 			CSV file containing ESP Moe sampling data from ESP logs. One instrument deployments - (1)  8/25/19 to 12/02/19
Moe_2019_Scott_Creek_2.csv 			CSV file containing ESP Moe sampling data from ESP logs. One instrument deployments - (1) 5/01/19 to 6/26/19
Waldo_2019_Scott_Creek.csv 			CSV file containing ESP Waldo sampling data from ESP logs. One instrument deployments - (1) 2/22/2019 - 5/06/2020

II. Variables
-------------

Protocol						ESP sampling method
Start dates 					ESP start up (wake up) time - Format "YYYY-MM-DD HH:MM:SS - GMT Offset" 
Start Date (Julian)				ESP start up (wake up) time - Format Julian with origin at January 1 of current year
Sample Start Date				ESP sampling start time - SP sampling start time - Format "YYYY-MM-DD HH:MM:SS - GMT Offset"  
Sample Start Date (Julian)		ESP sampling start time - Format Julian with origin at January 1 of current year
Sample End Date 				ESP sampling end time - FFormat "YYYY-MM-DD HH:MM:SS - GMT Offset" 
Sample End Date (Julian)		ESP sampling end time - Format Julian with origin at January 1 of current year
Time To Sample (hh:mm:ss)		ESP sampling time duration - SP sampling start time - Format "HH:MM:SS" 
Target Volume (ml)				ESP target sample volume in milliliters; preset sampling volume defined in the mission
Actual Volume (ml)				ESP actual sample volume in milliliters
Difference (ml)					ESP difference between the target sample volume and the actual sample volume


III. Observation Notes
----------------------

A. Target Volumes = 2000 mL are generally deployed sample volumes
B. Target Volumes <= 1000 mL are generally control or test samples performed in the lab (not during the deployment)
C. Actual Volumes < 25 mL are generally samples that failed during the sampling protocol



IV. Instrument Deployment Notes
-------------------------------

Deployment Date 		Recovery Date 			Instrument Name				Notes
---------------			-------------			---------------				-----	
03/25/19				05/06/2019				Waldo
05/06/19				06/27/19				Moe
06/27/19				08/22/19				Gordon
08/25/19				12/04/19				Moe
12/04/19				02/04/20				Gordon
02/04/20				02/10/20				Waldo 						Instrument failure 
02/12/20				04/04/20				Gordon





