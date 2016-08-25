import pandas 						as pd
import numpy 						as np
from matplotlib 					import pyplot as plt
from mpl_toolkits.mplot3d   		import Axes3D
from scipy 							import stats
import seaborn
import statsmodels.formula.api  	as sm
from statsmodels.api 				import add_constant
from selenium 						import webdriver
import numpy 						as np
from selenium.webdriver.common.keys import Keys
import multiprocessing 				as mp
from multiprocessing 				import Pool, Process, Manager, cpu_count

'''This script calculates the SS MTRs for each individual in the March
2014 CPS file. Must run CPS_keep_SSVars.py first before running this script
to be able to read the .csv file. Writes to .csv file a dataframe with 
SS_MTR and peridnum (personal id number)'''

def get_SS_MTR(df_list):
	'''This function takes in a list of dataframes and computes
	SS_MTR for each of them in parallel.

	Inputs lifetime earnings vector for each individual in CPS
	into Social Security's online calculator found at 
	https://www.ssa.gov/planners/retire/AnypiaApplet.html,
	and gathers their respective SS benefit amount that
	to what the calculator outputs'''

	df = df_list
	# An adjustment amount of this size makes a noticable marginal
	# differece in lifetime SS benefit amount. Small Amounts
	# don't as well:
	adjustment = 10000
	ind = df.index.values
	MTR_list = []
	for i in ind:
		# Goes through each individual in dataframe
		# and inputs their info into online SS
		# calculator:
		ageC = df['a_age'][i]
		experience = df['experience'][i]
		LE = df['LE'][i]
		# Creates a web browser:
		browser = webdriver.PhantomJS("/usr/bin/phantomjs", service_args=['--ssl-protocol=any'])
		browser.get('https://www.ssa.gov/planners/retire/AnypiaApplet.html')

		bday = browser.find_element_by_id("bday")
		bday.clear()
		bday.send_keys("1/1/" + str(2016 - ageC)) # inputs birthday

		retire = browser.find_element_by_id("ageYears")
		retire.clear()
		retire.send_keys("66") # inputs retirement age
		months = browser.find_element_by_id("ageMonths")
		months.clear()
		months.send_keys("0")
		# Sets variable of all of the years an individual
		# has worked in past:
		years_worked_past = experience - 1
		j = 0
		for i in np.arange(68 - years_worked_past, 68):	
			# Inputs all of the earnings for
			# the individual of previous years:
			earning = browser.find_element_by_id("Text"+ str(i))
			earning.send_keys(Keys.CONTROL + "a")
			earning.send_keys(Keys.DELETE)
			earning.send_keys(str(LE[j]))
			j += 1
		if len(LE) >= 2:
			# Inputs the current year's earnings for
			# older individuals:	
			current = browser.find_element_by_id("currentEarning")
			current.send_keys(Keys.CONTROL + "a")
			current.send_keys(Keys.DELETE)
			current.send_keys(str(LE[j]))
			j+=1 
			# Inputs the future year's earnings for
			# older individuals:
			future = browser.find_element_by_id("futureEarning")
			future.send_keys(Keys.CONTROL + "a")
			future.send_keys(Keys.DELETE)
			future.send_keys(str(LE[j]))
		elif len(LE) == 1:	
			# Inputs only the future year's earnings for
			# individuals who will just start working:	
			future = browser.find_element_by_id("futureEarning")
			future.send_keys(Keys.CONTROL + "a")
			future.send_keys(Keys.DELETE)
			future.send_keys(str(LE[j]))
		# Clicks the "Calculate Benefit" button:
		browser.find_element_by_id("Button1").click()
		# Extracts the "monthly calculated benefit amount":
		SSamount = browser.find_element_by_id("Text105")
		SSamount = SSamount.get_attribute('value')
		SSamount = float(SSamount[1:])
		# Scales it to be the lifetime SS benefit amount
		# (Assuming individuals die at age 85):
		SSamount *= (12 * 20)

		j = 0
		# Now we add an income adjustment and recalculate SS amount
		# to find marginal tax rates:
		for i in np.arange(68 - years_worked_past, 68):	
			earning = browser.find_element_by_id("Text"+ str(i))
			earning.send_keys(Keys.CONTROL + "a")
			earning.send_keys(Keys.DELETE)
			earning.send_keys(str(LE[j]))
			j += 1
		if len(LE) >= 2:	
			current = browser.find_element_by_id("currentEarning")
			current.send_keys(Keys.CONTROL + "a")
			current.send_keys(Keys.DELETE)
			current.send_keys(str(LE[j] + adjustment))# We add the adjustment to current earnings
			j+=1 
			future = browser.find_element_by_id("futureEarning")
			future.send_keys(Keys.CONTROL + "a")
			future.send_keys(Keys.DELETE)
			future.send_keys(str(LE[j]))
		elif len(LE) == 1:	
			future = browser.find_element_by_id("futureEarning")
			future.send_keys(Keys.CONTROL + "a")
			future.send_keys(Keys.DELETE)
			future.send_keys(str(LE[j] + adjustment))# We add the adjustment to current earnings
		browser.find_element_by_id("Button1").click()
		newSSamount = browser.find_element_by_id("Text105")
		newSSamount = newSSamount.get_attribute('value')
		newSSamount = float(newSSamount[1:])
		newSSamount *= (12 * 20)
		# We take the differece between the old SS benefit amount
		# and the new SS benefit and then divide out the adjustment 
		# amount, since adjustment amounts are greater than 1. This
		# is because smaller adjustments don't make a difference generally
		# in a person's monthly SS benefit
		MTR = (newSSamount - SSamount)/adjustment
		MTR_list.append(MTR)
	df.loc[ind, 'SS_MTR'] = MTR_list

	return df

def get_LE(x, age):
	'''This function creates a vector of earnings
	over each individual's lifetime. Uses parameters
	extracted from Mincer's Earnings Function Regresssion
	to create the appropriate earnings amounts for each 
	year'''
	# How long of a vector of earnings we 
	# create depends on age  years spent 
	# in school:
	years_worked = age - (17 + x)
	experience = np.arange(0, years_worked + 1)
	experienceSquared = experience*experience
	ones = np.ones(len(experience))
	educ_level = ones * x
	LE = np.exp(ones * params[0] + educ_level * params[1]\
		+ experience * params[2] + experienceSquared * params[3]).astype(int)
	return list(LE)

def SS_plot(CPS):
	'''This plots age against SS benefit amount in CPS'''
	age = CPS['a_age']
	SS = CPS['ss_val']
	fig, ax  = plt.subplots()
	plt.scatter(age, SS, label = None)
	legend = ax.legend(loc = "upper right", shadow = True, title = 'SSI for Unearned Income')
	plt.xlabel('Unearned Income')
	plt.ylabel('SSI Amount')
	plt.title('Supplemental Security Income for Unearned Income Amounts')
	plt.show()

def LE_reg(CPS, plot = False):
	'''This function uses Mincer's Earnings Function, which is a Regresssion
	equation that determines earnings based on an individual's experience, and 
	education level. The resulting parameters are used to calculate the individual's 
	lifetime earnings vector'''
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

CPS = pd.read_csv('CPS_SS.csv')
# We create a new variable called 'YrsPstHS', which is the number
# of years an individual has studied past High school minus 1.
df = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 7, 10, 10], \
index=["Children", "Less than 1st grade", "1st,2nd,3rd,or 4th grade",\
"5th or 6th grade", "7th and 8th grade", "9th grade",\
"10th grade", "11th grade", "12th grade no diploma",\
"High school graduate - high", "Some college but no degree",\
"Associate degree in college -",\
"Bachelor's degree (for", "Master's degree (for", "Professional school degree (for",\
"Doctorate degree (for"])
CPS['YrsPstHS'] = CPS['a_hga'].map(df)
# Experience in the workforce is equal to age minus years in school minus years
# in K-12:
CPS['experience'] = CPS['a_age'] - CPS['YrsPstHS'] - 17  
CPS['experienceSquared'] = CPS['experience'] * CPS['experience']
params = LE_reg(CPS)
CPS['SS_MTR'] = 0
CPS['LE'] = 0
# Using only individuals who are oof working age, not in school, and who have a positive income
# (We assume individuals' MTR who are 65+ or less than 17 yrs old, or not working
# is automatically 0 since they aren't going to be earning anything):
CPS_laborforce = CPS[(CPS['a_age'] >17) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]

# Calculating workers' lifetime earnings vector:
CPS_laborforce['LE'] = CPS_laborforce.apply(lambda x: get_LE(x['YrsPstHS'], x['a_age']), axis=1)

# Starting multiprocessing equal to the number of processors in computer:
pool = Pool(processes = cpu_count())
# Splitting dataframe among different processors:
df_split = np.array_split(CPS_laborforce, cpu_count())
# Each processor takes its respective dataframe and plugs in each individuals'
# information into online SS calculator to extract SS MTR:
laborforce_MTR_list = (pool.map(get_SS_MTR, df_split))
# Combines results from each proccesor into one dataframe:
laborforce_MTR = pd.concat(laborforce_MTR_list)
# Puts these individuals with positive SS MTR back into original
# CPS dataframe:
CPS = CPS.loc[~CPS.index.isin(laborforce_MTR.index)]
CPS = pd.concat([laborforce_MTR, CPS], axis = 0, join = 'outer').sort_index()
# Outputs SS_MTR and personal id number to .csv file:
CPS[['SS_MTR', 'peridnum']].to_csv('SS_MTR.csv', header = 'SS_MTR', index=False)







