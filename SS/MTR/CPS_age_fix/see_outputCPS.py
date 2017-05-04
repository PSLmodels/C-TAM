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
import ast
import csv

def get_LE_total(x):
	x = x.replace("\n", "")	
	x = x.replace("[ ", "[")
	x = x.replace(" ]", "]")
	x = x.replace("  ", " ")
	x = x.replace("''", " ")
	x = x.replace("[", "")
	x = x.replace("['', ", "")
	x = x.replace("]", "")	
	x = x.split(' ')
	if x[0] == '':
		x[0] = 0
	x = map(int, x)
	return np.sum(np.array(x, dtype = float))

def get_LE_vec(x):
	x = x.replace("\n", "")	
	x = x.replace("[ ", "[")
	x = x.replace(" ]", "]")
	x = x.replace("  ", " ")
	x = x.replace("[", "")
	x = x.replace("]", "")	
	x = x.split(' ')
	if x[0] == '':
		x[0] = 0
	x = map(int, x)
	return np.max(np.array(x, dtype = float))

def get_LE_year(x):
	x = x.replace("\n", "")	
	x = x.replace("[ ", "[")
	x = x.replace(" ]", "]")
	x = x.replace("  ", " ")
	x = x.replace("[", "")
	x = x.replace("]", "")	
	x = x.split(' ')
	if x[0] == '':
		x[0] = 0
	x = map(int, x)
	return np.argmax(np.array(x, dtype = float))

def get_LE_len(x):
	x = x.replace("\n", "")	
	x = x.replace("[ ", "[")
	x = x.replace(" ]", "]")
	x = x.replace("  ", " ")
	x = x.replace("[", "")
	x = x.replace("]", "")	
	x = x.split(' ')
	if x[0] == '':
		x[0] = 0
	x = map(int, x)
	return len(np.array(x, dtype = float))

def get_current_earnings(x, years):
	x = x.replace("\n", "")	
	x = x.replace("[ ", "[")
	x = x.replace(" ]", "]")
	x = x.replace("  ", " ")
	x = x.replace("[", "")
	x = x.replace("]", "")	
	x = x.split(' ')
	if x[0] == '':
		x[0] = 0
	x = map(int, x)
	return 
''' I should not be subtracting years past high school for experience!!! That subtracts too much, and makes the age- experience
so that makes it so my experience is much too long for the actual age. There are people who are 49 who have worked for 
39 years'''

'''Make sure to define a variable YrsPstHS for the regression , and another one that corresponds to actual years for the 
regression'''



# SS_MTR = pd.read_csv('SS_MTR_FutureReg_RETS_age_fixed.csv')
SS_MTR = pd.read_csv('SS_MTR_ConstantFuture_RETS_age_fixed.csv')
plt.scatter(SS_MTR["AIME_head"], SS_MTR['SS_MTR_head.1'])
plt.xlim(-1000, 15000)
plt.xlabel("AIME head")
plt.ylabel("SS MTR head")

plt.show()

# print SS_MTR['SS_MTR_head.1'][SS_MTR['SS_MTR_head.1'] == 0.56159999999999999]
# print list(np.unique(SS_MTR['SS_MTR_head.1']))
print SS_MTR['AIME_head'][(SS_MTR['AIME_head'] > 0) & (SS_MTR['SS_MTR_head.1'] == 0) & (SS_MTR['AIME_head'] < 1000)]
print SS_MTR.loc[15376]


# print list(np.unique(SS_MTR['SS_MTR_spouse.1']))
# print SS_MTR['SS_MTR_head'][(SS_MTR['earned_income_head'] > 116500) & (SS_MTR['earned_income_head'] < 117001)]
# print SS_MTR[(SS_MTR['SS_MTR.1'] < .56) & (SS_MTR['earned_income'] < 10000)]
# plt.scatter(SS_MTR['SS_MTR.1'], SS_MTR["earned_income"])
# plt.show()
# print SS_MTR
# SS_MTR = SS_MTR.merge(CPS_use, on = "peridnum")
# # print SS_MTR["LE"]
# print SS_MTR.iloc[3462]
# print SS_MTR.iloc[3462]['LE']
# print SS_MTR['a_age'] - (17 + SS_MTR['YrsPstHS'])
# print 65 - (17 + SS_MTR['YrsPstHS'])
# print SS_MTR['Current_LE'].describe()

# years_worked = age - (17 + x)
# 	years_to_work = 65 - (17 + x) #maybe 64
# 	experience = np.arange(0, years_to_work + 1)
# 	experienceSquared = experience*experience
# print CPS_use
# peridnum = 7205099270127649100103

# print LE.iloc[125]['LE'], LE.iloc[125], CPS_use.iloc[125]
# LE = head.merge(spouse,how = 'left', suffixes = ('_head', '_spouse'), on = 'RECID').fillna(0)
# CPS = CPS.merge(LE, how = 'left', on = "RECID").fillna(0)
# CPS = CPS[[ 'e00200p' ,'e00200s' ,'e00900p', 'e00900s','e02100p' ,'e02100s', 'age_head', 'age_spouse', 'hga_head', 'hga_spouse', 'ftpt_head', 'ftpt_spouse', 'gender_head', 'gender_spouse','peridnum', 'RECID']]
# CPS['earned_income_spouse'] = CPS[['e00200s','e00900s','e02100s']].sum(axis = 1)
# CPS['earned_income_head'] = CPS[['e00200p','e00900p','e02100p']].sum(axis = 1)
quants =  SS_MTR['AIME_head'].quantile(np.linspace(0,1,100)).as_matrix()
quants_s =  SS_MTR['AIME_spouse'].quantile(np.linspace(0,1,100)).as_matrix()
SS_MTR['decile_head'] = 0.
SS_MTR['decile_spouse'] = 0.
# plt.scatter(SS_MTR["AIME_head"], SS_MTR['SS_MTR_head.1'])
# plt.show()


# for i in xrange(len(quants)- 1):
# 	SS_MTR['decile_head'][(SS_MTR['AIME_head'] > quants[i]) & (SS_MTR['AIME_head'] <= quants[i + 1])] = i
# for i in xrange(len(quants_s)- 1):
# 	SS_MTR['decile_spouse'][(SS_MTR['AIME_spouse'] > quants_s[i]) & (SS_MTR['AIME_spouse'] <= quants_s[i + 1])] = i
# grouped = SS_MTR.groupby('decile_head').mean()
# groupeds = SS_MTR.groupby('decile_spouse').mean()


# x_vals = np.linspace(1, len(groupeds['SS_MTR_spouse.1']), len(groupeds['SS_MTR_spouse.1']))
# plt.plot(x_vals/float(len(x_vals)), groupeds["SS_MTR_spouse.1"])
# plt.ylabel("SS MTR spouse")
# plt.show()
# x_vals = np.linspace(1, len(grouped['SS_MTR_head.1']), len(grouped['SS_MTR_head.1']))
# plt.plot(x_vals/float(len(x_vals)), grouped['SS_MTR_head.1'])
# plt.ylabel("SS MTR head")
# plt.show()