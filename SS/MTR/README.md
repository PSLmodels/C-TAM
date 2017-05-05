# Social Security MTR Calculator Suite

Our python scripts containing our Social Security calculator, and our use of the Anypiab applet are contained in this suite. 

The following are descriptions of the different files contained in this suite:

- MTR_Anypiab_CPS: This script uses The Social Security Administration's Anypiab Social Security calculator to estimate the marginal tax rates for all working individuals of the cpsmar2014t.csv file. Creates .pia files that contain individual information, which are then input into the calculator and the results are extracted from the output file.
	- anypiab.exe is the applet file that must be downloaded in order for these scripts to work. We access this program to calculate SS benefits.
	- CPS_keep_SSVars.py cleans the cpsmar2014t.csv data file that is used in our MTR calculation.
	- We include the average wages index, the bend points, the SSA trustees report intermediate assumptions for CPI index, and maximum earnings for all possible years for the 2014 CPS.
	- SS_MTR_ConstantFuture_benefit.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings.
	- SS_MTR_FutureReg_benefit.py calculates MTRs using future earnings that are calculated via Mincer's earnings function.
	- see_outputCPS.py allows for analysis of the output of these files.

- CPS_age_fix: Uses our Social Security calculator to estimate the marginal tax rates for all working individuals of the cps_age_fixed.csv file. 
	- CPS_keep_SSVars.py cleans the cps_age_fix.csv data file that is used in our MTR calculation.
	- We include the average wages index, the bend points, the SSA trustees report intermediate assumptions for CPI index, and maximum earnings for all possible years for the 2014 CPS.
	- SS_MTR_ConstantFuture_benefit.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings.
	- SS_MTR_FutureReg_benefit.py calculates MTRs using future earnings that are calculated via Mincer's earnings function.
	- see_outputCPS.py allows for analysis of the output of these files.

- CPS_RETS: Uses our Social Security calculator to estimate the marginal tax rates for all working individuals of the CPSRETS.csv file. 
	- CPS_keep_SSVars.py cleans the CPSRETS.csv data file that is used in our MTR calculation.
	- We include the average wages index, the bend points, the SSA trustees report intermediate assumptions for CPI index, and maximum earnings for all possible years for the 2014 CPS.
	- SS_MTR_Constant_best_rets.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings.
	- SS_MTR_futureReg_RETS.py calculates MTRs using future earnings that are calculated via Mincer's earnings function.
	
- PUF: Uses our Social Security calculator to estimate the marginal tax rates for all working individuals of the puf.csv file. 
	- CPS_keep_SSVars.py cleans the puf.csv data file that is used in our MTR calculation.
	- We include the average wages index, the bend points, the SSA trustees report intermediate assumptions for CPI index, and maximum earnings for all possible years for the 2014 CPS.
	- SS_MTR_constantfuture_PUF.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings.
	- SS_MTR_futurereg_PUF.py calculates MTRs using future earnings that are calculated via Mincer's earnings function.
	- SS_MTR_nofuture_PUF.py calculates MTRs under the assumption that there are no earnings past 2014 for all individuals.

- Income_Rules_CPSnormal: Uses our Social Security calculator to estimate the marginal tax rates for all working individuals of the regular cpsmar2014t.csv file. 
	- CPS_keep_SSVars.py cleans the CPSRETS.csv data file that is used in our MTR calculation.
	- We include the average wages index, the bend points, the SSA trustees report intermediate assumptions for CPI index, and maximum earnings for all possible years for the 2014 CPS.
	- SS_MTR_Constant_best_rets.py calculates MTRs under the assumption that future earnings for all individuals remains constant from their 2014 earnings.
	- SS_MTR_futureReg_RETS.py calculates MTRs using future earnings that are calculated via Mincer's earnings function.
	- see_output.py allows for analysis of the output of these files.
	


