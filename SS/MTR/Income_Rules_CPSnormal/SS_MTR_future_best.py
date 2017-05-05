import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import seaborn
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
import sys, os
from subprocess import Popen, PIPE
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline
import subprocess

# [TODO]: add both constant, and constant indexed
# 1. AIME: the sum of 35 highest years of earnings
# 2. 3 bend point calculations: given by each year. 
# 3. Maximum earnings to be considered for SS calculation
# 4. Must have at least 10 years of earnings to qualify
# 6. I should index past and future earnings by SS index vector
'''This script calculates the Social Security Marginal Tax Rates for 
individuals in the 2014 CPS. We use our regression to calculate future earnings 
here for SS anypiab calculator to calculate future earnings after the year
2014.

Refer to SS_MTR_nofuture.py for a more detailed step-by-step documentation

The differences between the three SS_MTR files are found in the functions
get_SS_MTR, and get_txt  '''

def get_SS_MTR(YrsPstHS, Reg_YrsPstHS, age, wages, adjustment, bendpoints, max_earnings, CPI, boost_futurereg, earning,\
		in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, AI_HP_in, Asian_only,\
		 Black_AI_Asian, Hawaiian_in, White_only, a_sex, age_child, child):
	'''
	Creates the Lifetime Earnings vector with and without adjustment

	inputs:   
		x:  	    scalar, the number of post-secondary years of education.
		age: 	    scalar, age of individual.
		wages:      vector, wage inflation rates since 1950. Used to adjust
				    for wage inflation.
		adjustment: scalar, the amount that we adjust the year 2014 earnings
					to calculate MTRs.
	outputs:

	'''
	years_worked = age - (17 + YrsPstHS)

	if years_worked < 0:
		years_worked = 0
	years_to_work = 65 - (17 + YrsPstHS) 

	start_yr = sim_year - years_worked
	end_yr = start_yr + years_to_work
	
	# ---- Creating index for pre-retirment earnings. We don't increase index before 2014 because we use 2014 earnings
	index = wages.loc[wages['Year'] == end_yr - 6].index.values
	wages = wages.loc[wages['Year'] == end_yr - 6, 'Avg_Wage'].values[0] / wages['Avg_Wage'].values
	wages = wages[: index+1]
	wages = np.append(wages, np.ones(6) * wages[-1])

	# -----Creating regression variables
	experience = np.arange(0, years_to_work + 1)
	experienceSquared = experience*experience
	ones = np.ones(len(experience))
	educ_level = ones * Reg_YrsPstHS
	gender = ones * a_sex
	industry1 = ones * in1
	industry2 = ones * in2
	industry3 = ones * in3
	industry4 = ones * in4
	industry5 = ones * in5
	industry6 = ones * in6
	industry7 = ones * in7
	industry8 = ones * in8
	industry9 = ones * in9
	industry10 = ones * in10
	industry11 = ones * in11
	industry12 = ones * in12
	industry13 = ones * in13
	AI_HP = ones * AI_HP_in
	Asian = ones * Asian_only
	Black_AI = ones * Black_AI_Asian
	Hawaiian = ones * Hawaiian_in
	White = ones * White_only

	child = np.ones(len(experience))

	if age_child < len(experience):
	    child = np.concatenate((np.zeros(len(experience) - int(age_child)), (np.ones(int(age_child)))))

	if age_child == 99:
	    child = np.zeros(len(experience))

	gender_child = a_sex * child

	LE = np.exp(
		     educ_level * params[0] + experience * params[1] + experienceSquared * params[2]
		    + gender * params[3] + child * params[4] + gender_child * params[5] + industry1 * params[6]
		    + industry2 * params[7] + industry3 * params[8] + industry4 * params[9] + industry5 * params[10]
		    + industry6 * params[11] + industry7 * params[12] + industry8 * params[13] + industry9 * params[14]
		    + industry10 * params[15] + industry11 * params[16] + industry12 * params[17] + industry13 * params[18]
		    + AI_HP * params[19] + Asian * params[20] + Black_AI * params[21] + Hawaiian * params[22] + White * params[23]).astype(int)
	if len(LE) == 0:
		pass
	
	else:
		LE = (LE * boost_futurereg[63-years_worked:64+(years_to_work - years_worked)]).astype(int)
		scale = earning / LE[years_worked]
		LE = LE * scale
		LE = (LE * wages[63-years_worked:64+(years_to_work - years_worked)]).astype(int)
	
	LE_adjusted = LE.copy()
	
	max_earnings_use = max_earnings.loc[(max_earnings['Year'] >= start_yr) & (max_earnings['Year'] <= end_yr), 'Max_Earnings']

	within_threshold = False
	# ---------Max earnings check--------------
	if LE[years_worked] > max_earnings.loc[max_earnings['Year'] == sim_year, 'Max_Earnings'].values - adjustment:
		within_threshold = True
	if within_threshold == True: #If within max earnings threshold, make current earnings equal to max earnings
		LE_adjusted[years_worked] = max_earnings.loc[max_earnings['Year'] == sim_year, 'Max_Earnings'].values
	else:
		LE_adjusted[years_worked] += adjustment # Else, add the adjustment
	
	LE = np.where(LE > max_earnings_use, max_earnings_use, LE) #Correcting for max earnings threshold for all years
	LE_adjusted = np.where(LE_adjusted > max_earnings_use, max_earnings_use, LE_adjusted)

	# 10 year minimum contribution eligibility rule
	# if len(LE) < 10:
	# 	return 0

	# Taking top 35 earnings years
	top35 = np.argpartition(-LE, 35)
	result_args = top35[:35]
	top35 = np.partition(-LE, 35)
	LE = -top35[:35]

	if (np.sum(LE) / (35.* 12) - int(np.sum(LE) / (35.* 12))) >= .9999: # Correcting round-down errors from int(.)
		AIME_before = np.sum(LE) / (35.* 12)
	else: # Correcting round-down errors from int(.)
		AIME_before = int(np.sum(LE) / (35.* 12))

	PIA = 0

	# Bend points for year of retirement:
	bend_pt1 = bendpoints.loc[bendpoints['Year'] == end_yr - 4, 'Bend_pt1'].values[0]
	bend_pt2 = bendpoints.loc[bendpoints['Year'] == end_yr - 4, 'Bend_pt2'].values[0]
	effective_bendpt2 = bend_pt2 - bend_pt1

	#First bend-point
	if AIME_before <= 0:
		pass
	elif (AIME_before > 0) & (AIME_before < bend_pt1):
		PIA += AIME_before * .9
		AIME_before = 0
	else :
		PIA += bend_pt1 * .9
		AIME_before -= bend_pt1

	# Second bend-point
	if AIME_before <= 0:
		pass
	elif (AIME_before > 0) & (AIME_before < effective_bendpt2):
		PIA += AIME_before * .32
		AIME_before = 0
	else :
		PIA += effective_bendpt2 * .32
		AIME_before -= effective_bendpt2

	# Rest
	if AIME_before <= 0:
		pass
	else: 
		PIA += AIME_before * .15

	top35 = np.argpartition(-LE_adjusted, 35)
	result_args = top35[:35]
	top35 = np.partition(-LE_adjusted, 35)
	LE_adjusted = -top35[:35]

	if (np.sum(LE_adjusted) / (35.* 12) - int(np.sum(LE_adjusted) / (35.* 12))) >= .9999: # Correcting round-down errors from int(.)
		AIME_after = np.sum(LE_adjusted) / (35.* 12)
	else: # Correcting round-down errors from int(.)
		AIME_after = int(np.sum(LE_adjusted) / (35.* 12))
	PIA_after = 0

	#First bend-point
	if AIME_after <= 0:
		pass
	elif (AIME_after > 0) & (AIME_after < bend_pt1):
		PIA_after += AIME_after * .9
		AIME_after = 0
	else :
		PIA_after += bend_pt1 * .9
		AIME_after -= bend_pt1

	# Second bend-point
	if AIME_after <= 0:
		pass
	elif (AIME_after > 0) & (AIME_after < effective_bendpt2):
		PIA_after += AIME_after * .32
		AIME_after = 0
	else :
		PIA_after += effective_bendpt2 * .32
		AIME_after -= effective_bendpt2

	# Rest
	if AIME_after <= 0:
		pass
	else :
		PIA_after += AIME_after * .15

	# ------CPI adjustment for benefit--------

	# Adjusting for years after 62 years old (default retirement)
	adjust_from = end_yr - 4
	CPI_adjust = CPI.loc[(CPI['Year'] >= adjust_from) & (CPI['Year'] < adjust_from + 4), "CPI"].as_matrix()
	CPI_adjust_scale = np.prod(CPI_adjust)

	# Adjusting the benefit amount upon retirement (65) 

	PIA *= CPI_adjust_scale
	PIA_after *= CPI_adjust_scale

	# Adjusting all benefit amounts after retirment year, up to death (78) for pre-adjustment PIA
	PIA_vec = np.ones(13) * CPI.loc[(CPI['Year'] >= end_yr) & (CPI['Year'] < end_yr + 13), 'CPI'].values
	PIA_vec[0] = PIA
	PIA_vec = np.cumprod(PIA_vec) * 12.
	#Rounding like in the calculator
	# PIA_vec = np.floor(PIA_vec * 10) / 10.
	PIA = np.sum(PIA_vec)

	# Adjusting all benefit amounts after retirment year, up to death (78) for post-adjustment PIA
	PIA_vec_after = np.ones(13) * CPI.loc[(CPI['Year'] >= end_yr) & (CPI['Year'] < end_yr + 13), 'CPI'].values
	PIA_vec_after[0] = PIA_after
	PIA_vec_after = np.cumprod(PIA_vec_after) * 12.
	#Rounding like in the calculator
	# PIA_vec_after = np.floor(PIA_vec_after * 10) / 10.
	PIA_after = np.sum(PIA_vec_after)

	# Taking different between pre- and post-adjustment PIA and dividing by adjustment for MTR
	SS_MTR = ((PIA_after - PIA) / adjustment)
	SS_MTR = np.floor(SS_MTR * 100) / 100.
	if SS_MTR < 0:
		SS_MTR = 0

	return SS_MTR


def LE_reg(CPS, plot = False):
	'''
	Uses a linear regression to approximate coefficient to Mincer's earnings equation 
	which approximates Lifetime Earnings 

	Mincers: ln(earnings) = beta_0 + beta_1 * education + beta_2 * work_experience + beta_3 * work_experience^2 

	returns: array, the fitted parameters of the regression.
	'''
	
	sample = CPS.copy()[

	    (CPS['a_age'] > 16) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]

	earned_income = sample['earned_income']
	indep_vars = ['Reg_YrsPstHS', 'experience', 'experienceSquared', 'a_sex', 'child', 'a_sex_child',
	             'Agriculture, forestry,', 'Construction', 'Educational and health services', 'Financial activities',
	             'Information', 'Leisure and hospitality', 'Manufacturing', 'Mining', 'Other services','Professional and business',
	             'Public administration', 'Transportation and utilities', 'Wholesale and retail trade',
	              'AI-HP','Asian only','Black-AI-Asian', 'Hawaiian/Pacific Islander','White only']

	X = sample[indep_vars]
	model = sm.OLS(np.log(sample['earned_income']), X)
	results = model.fit()
	params = results.params
	if plot == True:
		x = np.linspace(0,np.max(CPS['earned_income']),15000)
		y = 0
		for i in xrange(len(params)):
			y += sample[indep_vars[i]] * params[i]
		# Cross Validation:
		fig, ax  = plt.subplots()
		plt.scatter( np.exp(y) , sample['earned_income'], label = 'earned_income vs. predicted earned_income')
		plt.plot(x, x, label = 'perfect fit', c = 'black', linewidth = 5)
		legend = ax.legend(loc = "upper right", shadow = True, title = 'earned_income vs. predicted earned_income')
		plt.xlabel('Predicted earned_income')
		plt.ylabel('Actual earned_income Amount')
		# plt.ylim(0,7000)
		plt.title('Accuracy of Linear Regresssion When Predicting earned_income')
		plt.show()
	return params

adjustment = 500
sim_year = 2014

bendpoints = pd.read_csv('Bendpoints.csv', dtype = {"Year": np.int32, "Bend_pt1": np.int32, "Bend_pt2": np.int32})
max_earnings = pd.read_csv('Max_Earnings.csv', dtype = {"Year": np.int32, "Max_Earnings": np.float64})
wages = pd.read_csv('averagewages.csv', dtype = {"Year": np.int32, "Avg_Wage": np.float64})
CPI = pd.read_csv('CPI_Intermediate.csv', dtype = {"Year": np.int32, "Max_Earnings": np.float64})
	# Below makes it so the earnings calculated via the regression (in 2014 terms) for previous years 
	# are indexed in 2014 terms later
wages.loc[wages['Year'] < sim_year, 'Avg_Wage'] = wages.loc[wages['Year'] == sim_year, 'Avg_Wage'].values[0]
boost_futurereg = wages['Avg_Wage'].values / wages['Avg_Wage'][wages['Year'] == sim_year].values[0] 

CPS = pd.read_csv('CPS_SS.csv')
CPS['a_age'] = CPS['a_age'].astype(int)
CPS['a_mjind'] = CPS['a_mjind'].astype(str)
dummies = pd.get_dummies(CPS['a_mjind'], drop_first=True)
CPS = pd.concat([CPS, dummies], axis=1)

df = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 7, 11, 11], \
	index=["Children", "Less than 1st grade", "1st,2nd,3rd,or 4th grade",\
	"5th or 6th grade", "7th and 8th grade", "9th grade",\
	"10th grade", "11th grade", "12th grade no diploma",\
	"High school graduate - high", "Some college but no degree",\
	"Associate degree in college -",\
	"Bachelor's degree (for", "Master's degree (for", "Professional school degree (for",\
	"Doctorate degree (for"])
CPS['Reg_YrsPstHS'] = CPS['a_hga'].map(df)

df = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 7, 10, 10], \
	index=["Children", "Less than 1st grade", "1st,2nd,3rd,or 4th grade",\
	"5th or 6th grade", "7th and 8th grade", "9th grade",\
	"10th grade", "11th grade", "12th grade no diploma",\
	"High school graduate - high", "Some college but no degree",\
	"Associate degree in college -",\
	"Bachelor's degree (for", "Master's degree (for", "Professional school degree (for",\
	"Doctorate degree (for"])
CPS['YrsPstHS'] = CPS['a_hga'].map(df)
CPS_child = CPS[CPS['a_famrel'] == 1]
CPS = CPS.join(CPS_child.groupby('fh_seq')['a_age'].max(), on='fh_seq', rsuffix='_child')
CPS['a_age_child'] = CPS['a_age_child'].fillna(99)
CPS['child'] = np.where(CPS['a_age_child'] == 99, 0, 1)
CPS['experience'] = CPS['a_age'] - CPS['YrsPstHS'] - 17  
# CPS['experience'][CPS['experience'] < 0] = 1
CPS.loc[CPS['experience'] < 0, 'experience'] = 1
CPS['experienceSquared'] = CPS['experience'] * CPS['experience']
# CPS['prdtrace'] = np.where(CPS['prdtrace'] == "White only", 1, 0)
dummies = pd.get_dummies(CPS['prdtrace'], drop_first=True)
CPS = pd.concat([CPS, dummies], axis=1)


CPS['a_sex_child'] = CPS['a_sex'] * CPS['child']

params = LE_reg(CPS)

CPS['SS_MTR'] = 0
CPS_laborforce = CPS[(CPS['a_age'] >17) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]
# CPS_laborforce = CPS_laborforce.loc[[60, 2],:]
# CPS_laborforce.to_pickle('use.pickle')
# CPS_laborforce = CPS_laborforce.iloc[:10]
# CPS_laborforce = pd.read_pickle('use.pickle')
CPS_laborforce['SS_MTR'] = CPS_laborforce.apply(lambda x: get_SS_MTR(x['YrsPstHS'], x['Reg_YrsPstHS'], x['a_age'],\
	 wages, adjustment, bendpoints, max_earnings, CPI, boost_futurereg, x['earned_income'],
	 x['Agriculture, forestry,'], x['Construction'], x['Educational and health services'],
     x['Financial activities'], x['Information'], x['Leisure and hospitality'],
     x['Manufacturing'], x['Mining'], x['Other services'],x['Professional and business'],
     x['Public administration'], x['Transportation and utilities'],
     x['Wholesale and retail trade'],  x['AI-HP'], x['Asian only'], x['Black-AI-Asian'], 
     x['Hawaiian/Pacific Islander'], x['White only'], x['a_sex'], x['a_age_child'], x['child']), axis=1)



final = pd.concat([CPS, CPS_laborforce["SS_MTR"]], axis = 1).fillna(0)
final[['SS_MTR', 'peridnum', 'earned_income']].to_csv('SS_MTR_WithFutureEarningsReg.csv', index = None)

# I believe that the 6 different types of MTR values are due to rounding errors (since the calculator rounds up 
# 	or down)
