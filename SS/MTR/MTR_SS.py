import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import seaborn
import statsmodels.formula.api as sm
import tabulate
from statsmodels.api import add_constant
from sklearn.ensemble import RandomForestRegressor
from treeinterpreter import treeinterpreter as ti
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from selenium import webdriver
import numpy as np
from selenium.webdriver.common.keys import Keys
import multiprocessing as mp
from multiprocessing import Pool, Process, Manager, cpu_count


def get_SS_MTR(df_list):
	df = df_list
	adjustment = 20000
	ind = df.index.values
	MTR_list = []
	for i in ind:
		ageC = df['a_age'][i]
		experience = df['experience'][i]
		LE = df['LE'][i]
		# browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
		browser = webdriver.PhantomJS("/usr/bin/phantomjs", service_args=['--ssl-protocol=any'])
		browser.get('https://www.ssa.gov/planners/retire/AnypiaApplet.html')

		bday = browser.find_element_by_id("bday")
		bday.clear()
		bday.send_keys("1/1/" + str(2016 - ageC))
		retire = browser.find_element_by_id("ageYears")
		retire.clear()
		retire.send_keys("66")
		months = browser.find_element_by_id("ageMonths")
		months.clear()
		months.send_keys("0")
		years_worked_past = experience - 1
		j = 0
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
			current.send_keys(str(LE[j]))
			j+=1 
			future = browser.find_element_by_id("futureEarning")
			future.send_keys(Keys.CONTROL + "a")
			future.send_keys(Keys.DELETE)
			future.send_keys(str(LE[j]))
		elif len(LE) == 1:	
			future = browser.find_element_by_id("futureEarning")
			future.send_keys(Keys.CONTROL + "a")
			future.send_keys(Keys.DELETE)
			future.send_keys(str(LE[j]))
		browser.find_element_by_id("Button1").click()
		SSamount = browser.find_element_by_id("Text105")
		SSamount = SSamount.get_attribute('value')
		SSamount = float(SSamount[1:])
		SSamount *= (12 * 20)

		j = 0
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
			current.send_keys(str(LE[j] + adjustment))
			j+=1 
			future = browser.find_element_by_id("futureEarning")
			future.send_keys(Keys.CONTROL + "a")
			future.send_keys(Keys.DELETE)
			future.send_keys(str(LE[j]))
		elif len(LE) == 1:	
			future = browser.find_element_by_id("futureEarning")
			future.send_keys(Keys.CONTROL + "a")
			future.send_keys(Keys.DELETE)
			future.send_keys(str(LE[j] + adjustment))
		browser.find_element_by_id("Button1").click()
		newSSamount = browser.find_element_by_id("Text105")
		newSSamount = newSSamount.get_attribute('value')
		newSSamount = float(newSSamount[1:])
		newSSamount *= (12 * 20)
		MTR = (newSSamount - SSamount)/adjustment
		MTR_list.append(MTR)
	df.loc[ind, 'SS_MTR'] = MTR_list

	return df

def get_LE(x, age):
	years_worked = age - (17 + x)
	experience = np.arange(0, years_worked + 1)
	experienceSquared = experience*experience
	ones = np.ones(len(experience))
	educ_level = ones * x
	LE = np.exp(ones * params[0] + educ_level * params[1] + experience * params[2] + experienceSquared * params[3]).astype(int)
	return list(LE)

def SS_plot(CPS):
    age = CPS['a_age']
    SS = CPS['ss_val']
    fig, ax  = plt.subplots()
    plt.scatter(age, SS, label = None)
    legend = ax.legend(loc = "upper right", shadow = True, title = 'SSI for Unearned Income')
    plt.xlabel('Unearned Income')
    plt.ylabel('SSI Amount')
    # plt.ylim(0,7000)
    plt.title('Supplemental Security Income for Unearned Income Amounts')
    plt.show()

def RandomForest(CPS, plot = True, sanity_check = True, MTR_Calc = False):
    string_list = ['experience', 'experienceSquared', 'YrsPstHS']
    predictor_columns = string_list
    rf = RandomForestRegressor(n_estimators=100 ,max_features = 'sqrt', min_samples_leaf=15, oob_score = True)
    if MTR_Calc == False:
        train = CPS.sample(frac=0.8, random_state=1)
        test = CPS.loc[~CPS.index.isin(train.index)]    
        rf.fit(train[predictor_columns], train['earned_income'])
        print rf.feature_importances_
        print 'here is out-of-bag score: ' , rf.oob_score_
        if sanity_check:
            # Accuracy Diagnosis
            predictions = rf.predict(train[predictor_columns])
            print 'R^2 for %80 of training data = ', 1 - np.sum((train['earned_income'].as_matrix()-predictions)**2)/np.sum((train['earned_income'].as_matrix()-np.mean(train['earned_income'].as_matrix()))**2)
            print (np.linalg.norm(train['marsupwt'].as_matrix()*(pd.Series(predictions).as_matrix()-train['earned_income'].as_matrix())))/(train['marsupwt'].as_matrix().sum())
            if plot == True:
                x = np.linspace(np.min(predictions),np.max(predictions),15000)
                fig, ax  = plt.subplots()
                plt.scatter(predictions, train['earned_income'], label = 'data')
                plt.plot(x, x, label = 'perfect fit', c = 'black', linewidth = 5)
                legend = ax.legend(loc = "upper right", shadow = True, title = 'earned_income against earned_income predicted')
                plt.xlabel('Predicted earned_income')
                plt.ylabel('earned_income Actual')
                # plt.ylim(0,7000)
                plt.title('Accuracy of earned_income Predicted Using Random Forests (using %80 train set sanity check)')
                plt.show()
        else:
            # Accuracy Diagnosis
            predictions = rf.predict(test[predictor_columns])
            print 'score:', rf.score(test[predictor_columns], test['earned_income'], test['marsupwt'])
            print 'R^2 for %20 = ', 1 - np.sum((test['earned_income'].as_matrix()-predictions)**2)/np.sum((test['earned_income'].as_matrix()-np.mean(test['earned_income'].as_matrix()))**2)
            print (np.linalg.norm(test['marsupwt'].as_matrix()*(pd.Series(predictions).as_matrix()-test['earned_income'].as_matrix())))/(test['marsupwt'].as_matrix().sum())
            if plot == True:
                x = np.linspace(np.min(predictions),np.max(predictions),15000)
                fig, ax  = plt.subplots()
                plt.scatter(predictions, test['earned_income'], label = 'data')
                plt.plot(x, x, label = 'perfect fit', c = 'black', linewidth = 5)
                legend = ax.legend(loc = "upper right", shadow = True, title = 'earned_income against earned_income predicted')
                plt.xlabel('Predicted earned_income')
                plt.ylabel('earned_income Actual')
                plt.title('Accuracy of earned_income Predicted Using Random Forests (using %20 test set)')
                plt.show()
        return predictions
    else:
        rf.fit(CPS[predictor_columns], SSI_use)
        print 'here is out-of-bag score: ' , rf.oob_score_
        predict_SSI = rf.predict(CPS[predictor_columns])
        CPS['earned_income'] += adjustment
        NewSSI = rf.predict(CPS[predictor_columns])
        CPS['earned_income'] -= adjustment
        plt.xlabel('Earned Income')
        plt.ylabel('New SSI Amount')
        plt.xlim(-1000,58000)
        plt.ylim(np.min(NewSSI), np.max(NewSSI))
        plt.title('Predicted SSI Amounts Using Random Forests on test set')
        plt.scatter(CPS['earned_income'], NewSSI, c = 'black')
        plt.show()
        MTR = (NewSSI -predict_SSI)/adjustment
        CPS['MTR_RF'] = pd.Series(np.zeros(len(CPS['earned_income'])))
        CPS.loc[ind, 'MTR_RF'] = MTR
        return MTR

def LE_reg(CPS, plot = False):
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
CPS['LE'] = 0
CPS_laborforce = CPS[(CPS['a_age'] >17) & (CPS['a_age'] < 66) & (CPS['a_ftpt'] == str(0.0)) & (CPS['earned_income'] > 0)]

	
ind = CPS_laborforce.index.values
CPS_laborforce = CPS_laborforce.iloc[ind[:10]]

CPS_laborforce['LE'] = CPS_laborforce.apply(lambda x: get_LE(x['YrsPstHS'], x['a_age']), axis=1)

pool = Pool(processes = cpu_count())
df_split = np.array_split(CPS_laborforce, cpu_count())
laborforce_MTR_list = (pool.map(get_SS_MTR, df_split))
laborforce_MTR = pd.concat(laborforce_MTR_list)
CPS = CPS.loc[~CPS.index.isin(laborforce_MTR.index)]
CPS = pd.concat([laborforce_MTR, CPS], axis = 0, join = 'outer').sort_index()
CPS['SS_MTR'].to_csv('SS_MTR.csv', header = 'SS_MTR', index=False)







