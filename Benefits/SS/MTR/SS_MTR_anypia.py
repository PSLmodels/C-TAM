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


def get_LE(x, age, wages):
	'''
	Creates the Lifetime Earnings vector before adjustment
	'''
	years_worked = age - (17 + x)
	experience = np.arange(0, years_worked + 1)
	experienceSquared = experience*experience
	ones = np.ones(len(experience))
	educ_level = ones * x
	LE = np.exp(ones * params[0] + educ_level * params[1] + experience * params[2] + experienceSquared * params[3]).astype(int)
	if len(LE) == 0:
		return list(LE)
	else:
		LE = (LE * wages[-len(LE):]).astype(int)
	return list(LE)

def get_LE2(x, age, wages, adjustment):
	'''
	Creates the Lifetime Earnings vector after adjustment
	'''
	years_worked = age - (17 + x)
	experience = np.arange(0, years_worked + 1)
	experienceSquared = experience*experience
	ones = np.ones(len(experience))
	educ_level = ones * x
	LE = np.exp(ones * params[0] + educ_level * params[1] + experience * params[2] + experienceSquared * params[3]).astype(int)
	if len(LE) == 0:
		return list(LE)
	else:
		LE = (LE * wages[-len(LE):]).astype(int)
		LE[-1] += adjustment #ADJUSTING THE FIRST YEAR OF EARNINGS INSTEAD OF 2013 FOR TESTING PURPOSES
	return list(LE)

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

wages = np.array(pd.read_csv('averagewages.csv')["Avg_Wage"]).astype(float)
wages = wages / wages[-1]
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
# SS_plot(CPS)
params = LE_reg(CPS)
CPS['SS_MTR'] = 0
CPS['LE'] = 0
CPS_laborforce = CPS[(CPS['a_age'] >17) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]
ind = CPS_laborforce.index.values
CPS_laborforce['LE'] = CPS_laborforce.apply(lambda x: get_LE(x['YrsPstHS'], x['a_age'], wages), axis=1)
adjustment = 20000
CPS_laborforce['LE_adjusted'] = CPS_laborforce.apply(lambda x: get_LE2(x['YrsPstHS'], x['a_age'], wages, adjustment), axis=1)



def get_txt(sex, age, experience, peridnum, LE):
	'''
	This function formats the income information correctly for the
	anypiab.exe program 
	'''
	counter = 0
	line1 = "01{}{}01011950".format(str(peridnum)[-9:], sex)
	line3 = "031012014"
	line6 = "06{}2014".format(2014-experience)
	line16 = "16{}".format(peridnum)
	line20 = "20{}".format(  "0"*len(LE)  )
	line22on = "22"
	j = 1
	for i, earning in enumerate(LE):
		new = str(earning).rjust(8)
		
		if i % 10 == 0 and i>0:
			line22on += "\n"
			line22on += str(j + 22)
			j+=1

		line22on += "{}.00".format(new)
	entry = line1 + "\n" + line3  + "\n" + line6 +"\n" + line16 + "\n" + line20 + '\n' + line22on
	return entry

CPS_laborforce['entries'] = CPS_laborforce.apply(lambda x: get_txt(x['a_sex'], x['a_age'],  x['experience'], x['peridnum'], x['LE']), axis=1)
CPS_laborforce['entries_adjusted'] = CPS_laborforce.apply(lambda x: get_txt(x['a_sex'], x['a_age'],  x['experience'], x['peridnum'], x['LE_adjusted']), axis=1)

CPS_laborforce['anypiabID'] = CPS_laborforce['peridnum'].apply(lambda row: str(row)[-9:])

pickled = CPS_laborforce.to_pickle("CPS.pickle")

# CPS_laborforce = pd.read_pickle("CPS.pickle")

df_partitions = np.array_split(CPS_laborforce, 2900)

# CPS_laborforce['entries'].to_frame().to_csv('CPS_anypiab.pia', index = None, header= None)


piab_id_list_adjusted = []
SS_list_adjusted = []
piab_id_list = []
SS_list = []


for i,indiv in CPS_laborforce.iterrows():
	thefile = open('CPS.pia', 'w')
	thefile.write("%s\n" % indiv['entries'])
	p = Popen('anypiab', stdin=PIPE) #NOTE: no shell=True here

	p.communicate('CPS')
	results = open('output')

	for counter, line in enumerate(results):
		piab_id_list.append(indiv.name)
		SS_list.append(line.split()[2])

for i,indiv in CPS_laborforce.iterrows():

	thefile = open('CPS.pia', 'w')
	thefile.write("%s\n" % indiv['entries_adjusted'])
	p = Popen('anypiab', stdin=PIPE) #NOTE: no shell=True here

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
df_adjust['ID_adjust'] = piab_id_list_adjusted

df.to_csv(path_or_buf='SS.csv', sep=',', na_rep='0')
df_adjust.to_csv(path_or_buf='SS_adjust.csv', sep=',', na_rep='0')

df['SS_MTR'] = (df_adjust['SS_adjust'] - df['SS']) / adjustment
df = df.set_index('ID', drop=True, append=False, inplace=False, verify_integrity=False)
final = pd.concat([CPS, df['SS_MTR']], axis = 1).fillna(0)
final[['SS_MTR', 'peridnum']].to_csv('SS_MTR.csv', index = None)