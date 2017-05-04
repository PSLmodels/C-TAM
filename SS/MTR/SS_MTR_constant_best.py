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
import pickle
from scipy.interpolate import InterpolatedUnivariateSpline
import subprocess
from sklearn.ensemble  import RandomForestRegressor as Rf


# 1. AIME: the sum of 35 highest years of earnings
# 2. 3 bend point calculations: given by each year. 
# 3. Maximum earnings to be considered for SS calculation
# 4. Must have at least 10 years of earnings to qualify
# 5. (I will fix the lifetime earnings vector, and incorporate the new regression model and maybe try random forests if that's okay with you)
# 6. I should index past and future earnings by SS index vector
'''This script calculates the Social Security Marginal Tax Rates for 
individuals in the 2014 CPS. We use our regression to calculate future earnings 
here for SS anypiab calculator to calculate future earnings after the year
2014.

Refer to SS_MTR_nofuture.py for a more detailed step-by-step documentation

The differences between the three SS_MTR files are found in the functions
get_LE, and get_txt  '''

def get_LE(YrsPstHS, Reg_YrsPstHS, age, wages, adjustment, bend_pt1, bend_pt2, wage_index, max_earnings, earning):
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
		years_worked = 1
	years_to_work = 65 - (17 + YrsPstHS) #maybe 64
	experience = np.arange(0, years_worked + 1)
	experienceSquared = experience*experience
	ones = np.ones(len(experience))
	educ_level = ones * Reg_YrsPstHS
	LE = np.exp(ones * params[0] + educ_level * params[1] + experience * params[2] + experienceSquared * params[3]).astype(int)
	if len(LE) == 0:
		LE = np.append(LE, np.zeros(65 - age)).astype(int)
	else:
		LE = (LE * wages[-len(LE):]).astype(int)
		LE = np.append(LE, (np.ones(65 - age) * LE[-1])).astype(int)
	scale = earning / LE[years_worked]
	LE = LE * scale
	LE_adjusted = LE.copy()
	within_threshold = False
	if LE[years_worked] > max_earnings['Max_Earnings'][max_earnings['Year'] == 2014].values - adjustment:
		within_threshold = True
	if within_threshold == True:
		LE_adjusted[years_worked] = max_earnings['Max_Earnings'][max_earnings['Year'] == 2014].values
	else:
		LE_adjusted[years_worked] += adjustment

	start_year = 2014 - years_worked 
	end_year = 2014 + (65 - age)
	# print max_earnings
	# print wage_index
	#No SS_MTR if current years earnings is above threshold. I should just change their 
	# np.where(A > B, B, A )

	# # No SS_MTR if less than 10 years of earnings
	# if len(LE) < 10:
	# 	return 0

	# Taking top 35 earnings years
	top35 = np.argpartition(-LE, 35)
	result_args = top35[:35]
	top35 = np.partition(-LE, 35)
	LE = -top35[:35]
	AIME_before = int(np.sum(LE) / (35.* 12))
	PIA = 0
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
	elif (AIME_before > 0) & (AIME_before < bend_pt2):
		PIA += AIME_before * .32
		AIME_before = 0
	else :
		PIA += bend_pt2 * .32
		AIME_before -= bend_pt2

	# Rest
	if AIME_before <= 0:
		pass
	else: 
		PIA += AIME_before * .15

	top35 = np.argpartition(-LE_adjusted, 35)
	result_args = top35[:35]
	top35 = np.partition(-LE_adjusted, 35)
	LE_adjusted = -top35[:35]
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
	elif (AIME_after > 0) & (AIME_after < bend_pt2):
		PIA_after += AIME_after * .32
		AIME_after = 0
	else :
		PIA_after += bend_pt2 * .32
		AIME_after -= bend_pt2

	# Rest
	if AIME_after <= 0:
		pass
	else :
		PIA_after += AIME_after * .15


	SS_MTR = ((PIA_after - PIA) / adjustment)*12.*13.
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
	sample = CPS.copy()[(CPS['a_age'] >16) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]
	earned_income = sample['earned_income']
	indep_vars = ['const','Reg_YrsPstHS', 'experience', 'experienceSquared']
	sample['const'] = 1.
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

def RF_reg(CPS, plot = False):
	'''
	Uses a linear regression to approximate coefficient to Mincer's earnings equation 
	which approximates Lifetime Earnings 

	Mincers: ln(earnings) = beta_0 + beta_1 * education + beta_2 * work_experience + beta_3 * work_experience^2 

	returns: array, the fitted parameters of the regression.
	'''
	CPS_new = CPS.copy()[(CPS['a_age'] >16) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]
	train = CPS_new.sample(frac=0.8, random_state=1)
	train_x = train.drop('earned_income', 1)
	train_x = train_x[['Reg_YrsPstHS', 'experience', 'experienceSquared']]
	test_x = CPS_new.loc[~CPS_new.index.isin(train_x.index)]
	test_y = test_x['earned_income']
	test_x = test_x[['Reg_YrsPstHS', 'experience', 'experienceSquared']]
	train_y = train["earned_income"]
	# earned_income = sample['earned_income']
	rforest = Rf(n_estimators = 200)
	rforest.fit(train_x , train_y)
	print rforest.score(test_x, test_y)

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




adjustment = 500
# Bend points for 2014:
bend_pt1 = 816.
bend_pt2 = 4917.
effective_bendpt2 = bend_pt2 - bend_pt1
wage_index = pd.read_csv("wage_index.csv", dtype = {"Year": np.int32, "Index": np.float64})
max_earnings = pd.read_csv('Max_Earnings.csv', dtype = {"Year": np.int32, "Max_Earnings": np.float64})
wages = np.array(pd.read_csv('averagewages.csv')["Avg_Wage"]).astype(float)

len_wage_old = len(wages)
x = np.linspace(0, len(wages), len(wages))
x2 = np.linspace(0, len(wages) * 2, len(wages) * 2 - 1)
order = 1
s = InterpolatedUnivariateSpline(x, wages, k = order)
wages = s(x2)
wages = wages / wages[len_wage_old - 1]
CPS = pd.read_csv('CPS_SS.csv')
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
CPS['experience'] = CPS['a_age'] - CPS['YrsPstHS'] - 17  
CPS['experience'][CPS['experience'] < 0] = 1
CPS['experienceSquared'] = CPS['experience'] * CPS['experience']
params = LE_reg(CPS)

CPS['SS_MTR'] = 0
CPS_laborforce = CPS[(CPS['a_age'] >17) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]
# CPS_laborforce = CPS_laborforce.loc[[2, 125],:]
# CPS_laborforce.to_pickle('use.pickle')
# CPS_laborforce = CPS_laborforce.iloc[:10]
# CPS_laborforce = pd.read_pickle('use.pickle')
CPS_laborforce['SS_MTR'] = CPS_laborforce.apply(lambda x: get_LE(x['YrsPstHS'], x['Reg_YrsPstHS'], x['a_age'],\
	 wages, adjustment, bend_pt1, effective_bendpt2, wage_index, max_earnings, x['earned_income']), axis=1)

final = pd.concat([CPS, CPS_laborforce["SS_MTR"]], axis = 1).fillna(0)
final[['SS_MTR', 'peridnum', 'earned_income']].to_csv('SS_MTR_ConstantFuture.csv', index = None)