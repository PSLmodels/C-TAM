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
import time

'''This script calculates the Social Security Marginal Tax Rates for 
individuals in the 2014 CPS. We use no future earnings here for  
SS anypiab calculator to calculate future earnings after the year
2014.

The differences between the three SS_MTR files are found in the functions
get_LE, and get_txt'''

def get_LE(x, age, wages, adjustment):
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
	years_worked = age - (17 + x)
	if years_worked < 0:
		years_worked = 0

	experience = np.arange(0, years_worked + 1)
	experienceSquared = experience*experience
	ones = np.ones(len(experience))
	educ_level = ones * x
	#Using regression to determine LE
	LE = np.exp(ones * params[0] + educ_level * params[1] + experience * params[2] + experienceSquared * params[3]).astype(int)
	if len(LE) == 0:
		LE_new = LE
		LE_adjusted = LE
	else:
		LE = (LE * wages[-len(LE):]).astype(int)#adjusting for wage inflation
		LE_new = LE.astype(int)
		LE_adjusted = LE_new.copy()
		LE_adjusted[-1] += adjustment#adding adjustment on 2014 earnings
	return pd.Series({'LE': LE_new, 'LE_adjusted': LE_adjusted})


def LE_reg(CPS, plot = False):
	'''
	Uses a linear regression to approximate coefficient to Mincer's earnings equation 
	which approximates Lifetime Earnings 

	Mincers: ln(earnings) = beta_0 + beta_1 * education + beta_2 * work_experience + beta_3 * work_experience^2 

	returns: array, the fitted parameters of the regression.
	'''
	sample = CPS.copy()[(CPS['a_age'] >16) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]
	earned_income = sample['earned_income']
	indep_vars = ['const','YrsPstHS', 'experience', 'experienceSquared']
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

adjustment = 500
cwd = os.getcwd()
wages = np.array(pd.read_csv('averagewages.csv')["Avg_Wage"]).astype(float)
wages = wages / wages[-1]
CPS = pd.read_csv('CPS_SS.csv')
#The following creates our YrsPstHS variable, which represents the amount of education that one receives past high school.
df = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 7, 10, 10], \
index=["Children", "Less than 1st grade", "1st,2nd,3rd,or 4th grade",\
"5th or 6th grade", "7th and 8th grade", "9th grade",\
"10th grade", "11th grade", "12th grade no diploma",\
"High school graduate - high", "Some college but no degree",\
"Associate degree in college -",\
"Bachelor's degree (for", "Master's degree (for", "Professional school degree (for",\
"Doctorate degree (for"])
CPS['YrsPstHS'] = CPS['a_hga'].map(df)
CPS['experience'] = CPS['a_age'] - CPS['YrsPstHS'] - 17  
CPS['experienceSquared'] = CPS['experience'] * CPS['experience']
params = LE_reg(CPS)
CPS['SS_MTR'] = 0
CPS_laborforce = CPS[(CPS['a_age'] >17) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]
ind = CPS_laborforce.index.values
df_LE = CPS_laborforce.apply(lambda x: get_LE(x['YrsPstHS'], x['a_age'], wages, adjustment), axis=1)
CPS_laborforce = pd.concat([CPS_laborforce, df_LE],  axis = 1)


def get_txt(sex, age, experience, peridnum, LE):
	'''
	This function creates a usable .pia entry for each individual
	in the CPS that will be used in the anypiab calculator.

	inputs:   
		sex:  	      scalar
		age: 	      scalar, age of individual.
		experience:   scalar, the amount of years has been in the work force.
		peridnum:     scalar, CPS identification number
		LE:			  vector, calculated lifetime earnings of respondent.
	
	outputs:
		entry:        string, a usable entry for the anypiab calcuator.

	'''
	if experience < 0:
		experience = 0
	counter = 0
	#First line must contain a 9 letter identifier, their gender (0 or 1) and birthday year
	line1 = "01{}{}0101{}".format(str(peridnum)[-9:], sex, 2014 - age)
	#Second line contains the date of retirement
	line3 = "03101{}".format(2014 + (65 - age))
	# Third line contains date of first earnings and last earnings
	line6 = "06{}{}".format(2014 - experience, 2014)
	#Fifth line contains identifier
	line16 = "16{}".format(peridnum)
	#These lines contain the earnings for each individual
	line22on = "22"
	linelast = "402017551"
	j = 1
	for i, earning in enumerate(LE):
		new = str(earning).rjust(8)
		
		if i % 10 == 0 and i>0:
			line22on += "\n"
			line22on += str(j + 22)
			j+=1

		line22on += "{}.00".format(new)
	entry = line1 + "\n" + line3  + "\n" + line6 + "\n" + line16 + '\n' + line22on + '\n' + linelast
	return entry

# We create the entries for anypiab.exe in the following code
CPS_laborforce['entries'] = CPS_laborforce.apply(lambda x: get_txt(x['a_sex'], x['a_age'],  x['experience'], x['peridnum'], x['LE']), axis=1)
CPS_laborforce['entries_adjusted'] = CPS_laborforce.apply(lambda x: get_txt(x['a_sex'], x['a_age'],  x['experience'], x['peridnum'], x['LE_adjusted']), axis=1)

CPS_laborforce['anypiabID'] = CPS_laborforce['peridnum'].apply(lambda row: str(row)[-9:])

# These lists will be populated with the id numbers and SS benefit amounts of each individual
piab_id_list_adjusted = []
SS_list_adjusted = []
piab_id_list = []
SS_list = []

# Here we iterate through all of the individuals in the CPS and put their
# entry into the anypiab.exe SS benefit calculator. We then extract the results
# from the file called 'output', and use the 
for i,indiv in CPS_laborforce.iterrows():
	thefile = open('CPS.pia', 'w')
	thefile.write("%s\n" % indiv['entries'])
	p = Popen(cwd +'/anypiab.exe', stdin = PIPE) 
	p.communicate('CPS')
	results = open('output')

	for counter, line in enumerate(results):
		piab_id_list.append(indiv.name)
		SS_list.append(line.split()[2])

# Here we iterate through all of the individuals in the CPS and put their
# entry into the anypiab.exe SS benefit calculator with their new adjusted 2014 earnings.
for i,indiv in CPS_laborforce.iterrows():
	thefile = open('CPS.pia', 'w')
	thefile.write("%s\n" % indiv['entries_adjusted'])
	p = Popen(cwd +'/anypiab.exe', stdin=PIPE)
	p.communicate('CPS')
	results = open('output')

	for counter, line in enumerate(results):
		piab_id_list_adjusted.append(indiv.name)
		SS_list_adjusted.append(line.split()[2])
	
df = pd.DataFrame()
df_adjust= pd.DataFrame()
df['SS'] = SS_list
df['ID'] = piab_id_list
df_adjust['SS_adjust'] = SS_list_adjusted
df_adjust['ID'] = piab_id_list_adjusted

df.SS = df.SS.astype(float)
df_adjust.SS_adjust = df_adjust.SS_adjust.astype(float)
df = df.merge(df_adjust, on = "ID")
# We subtract the new SS benefit amount (after the adjustment) by the old SS benefit
# amount, then we divide by the adjusment. After we multipy by 12 then 13 to represent
# the change in lifetime benefit for $1 earned
df['SS_MTR'] = ((df.ix[1:,'SS_adjust'] - df['SS']) / adjustment)*12.*13.
df = df.set_index('ID', drop=True, append=False, inplace=False, verify_integrity=False)
final = pd.concat([CPS, df['SS_MTR']], axis = 1).fillna(0)
final[['SS_MTR', 'peridnum']].to_csv('SS_MTR_nofuture.csv', index = None)