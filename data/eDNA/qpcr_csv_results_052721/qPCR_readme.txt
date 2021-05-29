# Scott Creek qPCR
***
Files in this directory are result outputs for qPCR runs performed on a StepOnePlus and in csv format. 
Each files represents a different qPCR plate/run.

## File Structure
***
The first 6 rows (lines) are run specific parameters associated with the StepOnePlus
The 8 row (line) is the header for the variables
There are generally 22 variables reported for each qPCR run and each row of the variables represents a PCR well or reaction.
The most important are the following:
1. Sample Name - is the sample identifier
1a. For samples that end in 1:5, these are 1:5 dilutions of the original sample
2. Target Name - defines the assay or target gene. In this case, O.kisutch = Coho Salmon, O.mykiss = Rainbow Trout
3. Task - defines the pcr reaction as either a STANDARD (for generating standard curves) or UNKNOWN (samples)
4. Ct - provides the cycle threshold for each reaction
5. Quantity - provides the concentration of DNA in copies/reaction (in these files this is copies/uL). 
5a. For STANDARDS these are user input values to generate the standard curve
5b. For UNKNOWN these are computed based on the standard curve for each plate by the StepOnePlus Software

### Notes
***
Generally, we utilize only the CT and STANDARD Quanities for each plate. The standards for all plates are combined to generate a combined standard curve which is then utilized to compute the quantities for all of the UNKNOWNS. 