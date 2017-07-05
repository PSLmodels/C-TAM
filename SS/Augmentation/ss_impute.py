# coding: utf-8
import pandas as pd
import numpy as np
import csv
import statsmodels.formula.api as sm
import statsmodels

"""
Description:
	This script generates imputed recipients to receive Social Security Benefits 
	to match the CPS 2014 March totals with SSA December 2014 totals. We use the dataset cpsmar2014t.csv


Output:
	The output comes in 2 csv files: 
		1. ByState.csv, which summarizes the results for each state
		2. SS_augmentation.csv, which has the ID for each individual (PERIDNUM), their imputed/actual SS received, 
			and a number indicating Social Security participation status:
			0 - Not Participating
			1 - Original Recipient
			2 - Imputed Recipient


Functions:
	read_data: Reads in the cps dataset and ssa_dataset
	addnewvars: Adds and modifies a few variables to data to use in the regression
	regression: Performs ols regression to get probability to recieve social security
	impute: Imputes high-probabiliity non-recipients to recipient pool from CPS for each state to match SSA totals
"""


#Average of all months in 2014 at https://www.ssa.gov/cgi-bin/payment.cgi
AveBen2014_MonthlySSA = 70703666667


def read_data():

	cps_alldata = pd.read_csv("../../../Dropbox/asec2014_pubuse.csv", header=0, \
							usecols=["peridnum","gestcen", "gestfips", "marsupwt", "ss_yn", "sskidyn", "ss_val", "dis_hp", "a_maritl", "a_age", "a_hga"])
	cps_alldata.rename(columns = {'gestcen':'State'}, inplace = True)

	ssa_data = pd.read_csv("SSA_Compiled.csv", header=0).set_index("State")

	print cps_alldata.shape

	return cps_alldata, ssa_data

def addnewvars(cps_alldata):
	"""
	We create 4 binary (0 or 1) variables to use in the regression:
		1. ss_indicator (Dependent variable): Indicates whether or not the individual in the CPS had received 
											  Social Security benefits (ss_yn or sskid_yn was “Yes”)
		2. disabled_yn: CPS variable was dis_hp was “Yes”. 1-disabled, 0-not diabled
		3. aged_yn: CPS variable a_aged was greater than 65. 1-aged, 0-not aged
		4. widowed_yn = CPS variable a_marital indicated 'widowed'. 1-widowed, 0-anything else
	
	Note: Because the variable a_age had categories for ’80-84 years of age’ and ‘85+ years of age’ we assigned 
		  all of those between 80-84 a random uniformly distributed age and assumed everybody in the 85+ category 
		  to be exactly 85 years. 
	"""

	cps_alldata = cps_alldata.replace({"None or not in universe" : 0.}, regex=True)
	cps_alldata['ss_val'] = cps_alldata['ss_val'].astype(float)
	cps_alldata['ss_wtt'] = cps_alldata['marsupwt']*cps_alldata['ss_val']


	#CHANGE THIS SO NOT RANDOM OR FIXED AT 85
	cps_alldata.loc[cps_alldata['a_age'] == "80-84 years of age", 'a_age'] = np.random.randint(low = 80, high = 85)
	cps_alldata.loc[cps_alldata['a_age'] == "85+ years of age", 'a_age'] = 85
	cps_alldata['a_age'] = cps_alldata['a_age'].astype(int)

	#Creating the independent variable for the regression
	cps_alldata['ss_indicator'] = 0
	condition = (cps_alldata['ss_yn'] == "Yes") | (cps_alldata['sskidyn'] == "Received SS")
	cps_alldata.loc[condition, 'ss_indicator'] = 1


	#Creating new binary variable for over 65 or not
	cps_alldata['Aged_yn'] = 0
	condition = cps_alldata['a_age'].astype(int) >= 65
	cps_alldata.loc[condition, 'Aged_yn'] = 1


	#Creating a new binary variable for disabled or not
	cps_alldata['Disabled_yn'] = 0
	condition = cps_alldata['dis_hp'] == 'Yes'
	cps_alldata.loc[condition, 'Disabled_yn'] = 1


	#Creating a new binary variable for widowed or not
	cps_alldata['Widowed_yn'] = 0
	condition = cps_alldata['a_maritl'] == 'Widowed'
	cps_alldata.loc[condition, 'Widowed_yn'] = 1


	return cps_alldata

def regression(cps_alldata):

	model = sm.ols(formula='ss_indicator ~ Aged_yn + Disabled_yn + Widowed_yn', data=cps_alldata)
	#model = sm.ols(formula='ss_indicator ~ Aged_yn + Disabled_yn + Widowed_yn', data=cps_alldata)
	results = model.fit()
	print results.summary()
	ypred = results.predict()
	cps_alldata['Prob_Received'] = ypred

	return cps_alldata

def impute(cps_alldata, ssa_data):

	#Deleting unnecessary variables from this point on
	cps_trimmed = cps_alldata.drop(['gestfips', 'a_age', 'dis_hp', 'a_maritl',\
									 'Aged_yn', 'Disabled_yn', 'Widowed_yn'], axis=1)


	#Getting only those who received SS from cps
	cps_recipients = cps_trimmed[cps_alldata["ss_yn"] == "Yes"]


	#Gets nonrecipients and sorts them by state and likelihood of getting ss
	nonrecipients = (cps_trimmed[cps_trimmed["ss_yn"] != "Yes"]).sort_values(by=['State', 'Prob_Received'], ascending=[True,False])


	#Converting to monthly values and summing across states
	cps_recipients['ss_wtt'] /= 12
	cps_grouped = cps_recipients.groupby('State').sum()


	#Combining cps and ssa totals (monthly) into one dataframe
	combined_data = cps_grouped.join(ssa_data)
	combined_data = combined_data.drop(['ss_indicator', 'Prob_Received'], axis=1)


	#Scaling to account for December being higher than rest of year
	combined_data['SSA_Benefit'] *= float(combined_data['SSA_Benefit'].sum())/AveBen2014_MonthlySSA


	#Getting the needed new recipients and benefits to impute
	combined_data['rec_diff'] = pd.Series(combined_data['marsupwt']-combined_data['SSA_Recipients'], index=combined_data.index)
	combined_data['ben_diff'] = pd.Series(combined_data['ss_wtt']-combined_data['SSA_Benefit'], index=combined_data.index)
	combined_data['avemonben_ssa'] = pd.Series(combined_data['SSA_Benefit']/combined_data['SSA_Recipients'], index=combined_data.index)


	#Imputing benefits
	imputed = pd.DataFrame(columns = cps_recipients.columns.values)
	nonimputed = pd.DataFrame(columns = cps_recipients.columns.values)
	avemonbensimputed = []

	for state in ssa_data.index:
			print state
			imputed_state = pd.DataFrame(columns = cps_recipients.columns.values)
			recips_state = cps_recipients[cps_recipients["State"] == state]
			nonrecips_state = nonrecipients[nonrecipients["State"] == state]
			gap_rec = combined_data.at[state,'rec_diff']


			#Matching recipient totals by adding one at a time
			iter = 0
			for i,recip in nonrecips_state.iterrows():
				#print i
				recip = recip.copy()
				gap_rec += recip['marsupwt']

				if gap_rec > 0:
					nonimputed_state = nonrecips_state[iter:]
					break


				imputed_state = imputed_state.append(recip)
				iter += 1

			#Imputing the benefits for given state
			avemonbensimputed.append(-combined_data.loc[state,'ben_diff']/imputed_state['marsupwt'].sum())
			imputed_state['ss_val'] = -combined_data.loc[state,'ben_diff']/imputed_state['marsupwt'].sum() * 12
			imputed_state['ss_wtt'] = imputed_state['marsupwt']*imputed_state['ss_val']
			imputed = imputed.append(imputed_state)
			nonimputed = nonimputed.append(nonimputed_state)

	#Assigning the categorical variable for participation
	cps_recipients['Participation'] = 1
	imputed['Participation'] = 2
	nonimputed['Participation'] = 0


	#Combining the imputed recipients with cps recipients
	imputed_combined = cps_recipients.append(imputed)
	imputed_combined = imputed_combined.append(nonimputed)
	imputed_combined = imputed_combined.sort_values(by='Prob_Received')

	#Getting final results and exporting to csv
	imputed_grouped = imputed_combined.groupby('State').sum()
	combined_data.columns = ['CPS Weighted Recipients', 'CPS Benefit per Recipient', 'Weighted Benefits', 'SSA_Recipients',\
	 						 'SSA_Benefit', 'Recipients Gap', 'Benefits Gap', 'Ajusted monthly benefit']	
	combined_data['Imputed Monthly Benefit'] = avemonbensimputed
	combined_data['CPS + Imputed Recipients'] = imputed_grouped['marsupwt']
	combined_data['CPS + Imputed Benefits'] = imputed_grouped['ss_wtt']


	imputed_combined[['peridnum','ss_val', 'Participation']].to_csv("SS_augmentation.csv", sep=',')
	combined_data.to_csv("ByState.csv", sep=',')


cps_alldata, ssa_data = read_data()

cps_alldata = addnewvars(cps_alldata)

cps_alldata = regression(cps_alldata)

impute(cps_alldata, ssa_data)

