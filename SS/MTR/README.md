# Social Security MTR Calculator Suite

Our python scripts containing our Social Security calculator, and our use of the Anypiab applet are contained in this suite. 

The following are descriptions of the different files contained in this suite:
- We include the average wages index, the bend points, the SSA trustees report intermediate assumptions for CPI index, and maximum earnings for all possible years for the 2014 CPS. In addition, we include scripts that calculate SS_MTR with all dollar amounts in 2014 terms, and thus assume no inflation/wage increases. For these scripts that don't make indexing assumptions (ending in _noindex.py), we include bend points and maximum earnings that aren't indexed (in 2014 terms).

- MTR_Anypiab_CPS: This script uses The Social Security Administration's Anypiab Social Security calculator to estimate the marginal tax rates for all working individuals of the cpsmar2014t.csv file. Creates .pia files that contain individual information, which are then input into the calculator and the results are extracted from the output file.
	- anypiab.exe is the applet file that must be downloaded in order for these scripts to work. We access this program to calculate SS benefits.
	- CPS_keep_SSVars.py cleans the cpsmar2014t.csv data file that is used in our MTR calculation.
	- SS_MTR_ConstantFuture_benefit.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings.
	- SS_MTR_FutureReg_benefit.py calculates MTRs using future earnings that are calculated via Mincer's earnings function.
	- see_outputCPS.py allows for analysis of the output of these files.

- Income_Rules_CPSnormal: Uses our Social Security calculator to estimate the marginal tax rates for all working individuals of the regular cpsmar2014t.csv file. 
	- CPS_keep_SSVars.py cleans the cpsmar2014t.csv data file that is used in our MTR calculation.
	- SS_MTR_constant.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings, but are indexed by CPI and wage index.
	- SS_MTR_futureReg_RETS.py calculates MTRs using future earnings that are calculated via Mincer's earnings functio, and are indexed by CPI and wage index.
	- SS_MTR_constant_noindex.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings. All earnings and benefit amounts are in 2014 dollars.
	- SS_MTR_future_noindex.py calculates MTRs using future earnings that are calculated via Mincer's earnings function. All earnings and benefit amounts are in 2014 dollars.
	- arma_projections contains the code (arma.ipynb) that we used to calculate our future projections (post-2025) for bend points and CPI. 
	- see_output.py allows for analysis of the output of these files.
	
- CPS_RETS: Uses our Social Security calculator to estimate the marginal tax rates for all working individuals of the CPSRETS.csv file. 
	- CPS_keep_SSVars.py cleans the cps_age_fix.csv data file that is used in our MTR calculation.
	- SS_MTR_constant.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings, but are indexed by CPI and wage index.
	- SS_MTR_FutureReg.py calculates MTRs using future earnings that are calculated via Mincer's earnings function, and are indexed by CPI and wage index
	- SS_MTR_constant_noindex.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings. All earnings and benefit amounts are in 2014 dollars.
	- SS_MTR_future_noindex.py calculates MTRs using future earnings that are calculated via Mincer's earnings function. All earnings and benefit amounts are in 2014 dollars.
	- see_outputCPS.py allows for analysis of the output of these files.

- PUF: Uses our Social Security calculator to estimate the marginal tax rates for all working individuals of the puf.csv file. 
	- CPS_keep_SSVars.py cleans the puf.csv data file that is used in our MTR calculation.
	- SS_MTR_constantfuture_PUF.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings.
	- SS_MTR_futurereg_PUF.py calculates MTRs using future earnings that are calculated via Mincer's earnings function.
	- SS_MTR_nofuture_PUF.py calculates MTRs under the assumption that there are no earnings past 2014 for all individuals.



