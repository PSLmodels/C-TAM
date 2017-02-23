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


def get_LE(x, age, wages, adjustment):
	'''
	Creates the Lifetime Earnings vector before adjustment

	inputs:   
		x:  	scalar, the number of post-secondary years of education.
		age: 	scalar, age of individual.
		wages:  scalar 
	'''
	years_worked = age - (17 + x)
	years_to_work = 65 - (17 + x) #maybe 64
	experience = np.arange(0, years_to_work + 1)
	experienceSquared = experience*experience
	ones = np.ones(len(experience))
	educ_level = ones * x
	LE = np.exp(ones * params[0] + educ_level * params[1] + experience * params[2] + experienceSquared * params[3]).astype(int)
	if len(LE) == 0:
		pass
	else:
		LE = (LE * wages[63-years_worked:64+(years_to_work - years_worked)]).astype(int)
	LE_adjusted = LE.copy()
	LE_adjusted[years_worked] += adjustment
	return pd.Series({'LE': LE, 'LE_adjusted': LE_adjusted})

def get_txt(sex, age, experience, peridnum, LE):
	'''
	This function formats the income information correctly for the
	anypiab.exe program 
	'''
	counter = 0
	line1 = "01{}{}0101{}".format(str(peridnum)[-9:], sex, 2014 - age)
	line3 = "03101{}".format(2014 + (65 - age))
	line6 = "06{}{}".format(2014 - experience, 2014 + (65 - age) )
	line16 = "16{}".format(peridnum)
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


def LE_reg(CPS, plot = False):
	'''
	Uses a linear regression to approximate coefficient to Mincer's earnings equation 
	which approximates Lifetime Earnings 
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
wages = np.array(pd.read_csv('averagewages.csv')["Avg_Wage"]).astype(float)
len_wage_old = len(wages)
x = np.linspace(0, len(wages), len(wages))
x2 = np.linspace(0, len(wages) * 2, len(wages) * 2 - 1)
order = 1
s = InterpolatedUnivariateSpline(x, wages, k = order)
wages = s(x2)
wages = wages / wages[len_wage_old - 1]
CPS = pd.read_csv('CPS_SS.csv')
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




CPS_laborforce['entries'] = CPS_laborforce.apply(lambda x: get_txt(x['a_sex'], x['a_age'],  x['experience'], x['peridnum'], x['LE']), axis=1)
CPS_laborforce['entries_adjusted'] = CPS_laborforce.apply(lambda x: get_txt(x['a_sex'], x['a_age'],  x['experience'], x['peridnum'], x['LE_adjusted']), axis=1)

CPS_laborforce['anypiabID'] = CPS_laborforce['peridnum'].apply(lambda row: str(row)[-9:])

# pickled = CPS_laborforce.to_pickle("CPS.pickle")
# pickled1 = CPS.to_pickle("CPS_full.pickle")

# CPS = pd.read_pickle("CPS_full.pickle")
# CPS_laborforce = pd.read_pickle("CPS.pickle")
# CPS_laborforce['entries'].to_frame().to_csv('CPS_anypiab.pia', index = None, header= None)
# CPS_laborforce = CPS_laborforce.iloc[:10]

piab_id_list_adjusted = []
SS_list_adjusted = []
piab_id_list = []
SS_list = []


for i,indiv in CPS_laborforce.iterrows():
	thefile = open('CPS.pia', 'w')
	thefile.write("%s\n" % indiv['entries'])
	p = Popen('/home/parker/Documents/AEI/Benefits/SS/MTR/anypiab.exe', stdin = PIPE) #NOTE: no shell=True here
	p.communicate('CPS')
	results = open('output')

	for counter, line in enumerate(results):
		piab_id_list.append(indiv.name)
		SS_list.append(line.split()[2])

for i,indiv in CPS_laborforce.iterrows():
	thefile = open('CPS.pia', 'w')
	thefile.write("%s\n" % indiv['entries_adjusted'])
	p = Popen('/home/parker/Documents/AEI/Benefits/SS/MTR/anypiab.exe', stdin=PIPE)
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

# df.to_csv(path_or_buf='SS.csv', sep=',', na_rep='0')
# df_adjust.to_csv(path_or_buf='SS_adjust.csv', sep=',', na_rep='0')
# df_adjust = df_adjust.ix[1:].reset_index()
df.SS = df.SS.astype(float)
df_adjust.SS_adjust = df_adjust.SS_adjust.astype(float)
df = df.merge(df_adjust, on = "ID")
df['SS_MTR'] = ((df['SS_adjust'] - df['SS']) / adjustment)*12.*13.
df = df.set_index('ID', drop=True, append=False, inplace=False, verify_integrity=False)
final = pd.concat([CPS, df['SS_MTR']], axis = 1).fillna(0)
final[['SS_MTR', 'peridnum']].to_csv('SS_MTR_WithFutureEarningsReg.csv', index = None)