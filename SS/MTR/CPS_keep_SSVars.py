import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import csv

'''This code defines and sets the variables that we input into
SS online calculator. You must have the file cpsmar2014t.csv,
which is the 2014 March CPS file.'''

# Reading in 2014 March CPS file:
CPS_dataset = pd.read_csv('cpsmar2014t.csv')

# Defining the colunms to extract from CPS that we'll use in calculation:
columns_to_keep = ['ss_val', 'ss_yn','csp_val', 'rnt_val', 'div_val', 'vet_val', 'a_maritl', 'marsupwt', 'a_age', 'gestfips',
                   'rsnnotw', 'vet_typ1', 'pemlr', 'mcare',
                   'wsal_val', 'semp_val', 'frse_val', 'ss_val', 'rtm_val', 'oi_off', 'oi_val',
                   'uc_yn', 'uc_val', 'int_yn', 'int_val', 
                   'ffpos', 'fh_seq', 'ftot_r', 'ftotval', 'ptot_r', 'ptotval',
                   'peridnum', 'paw_yn', 'filestat', 'a_ftpt', 'a_spouse', 'a_hga', 'a_sex']

CPS = CPS_dataset[columns_to_keep]

def Init_CPS_Vars(CPS):

	'''
	Initializes the variables that we use to calculate SS MTR

				Variables within the CPS that we use:

	    ss_val      Social Security income received
	    ss_yn       If received Social Security Income
	    marsupwt    March supplement final weight
	    a_age       Age
	    gestfips    FIPS State Code
	    vet_typ1    Veterans payments, type 1
	    pemlr       Monthly labor force recode
	    mcare       Medicare coverage
	    wsal_val    Total wage and salary earnings value
	    semp_val    Own business self-employment earnings, total value
	    frse_val    Farm self-employment earnings, total value
	    ss_val      Social Security payments received, value
	    rtm_val     Retirement income received, total amount
	    oi_off      Income sources, other
	    oi_val      Income, other (amount)
	    uc_yn       Unemployment compensation benefits received
	    uc_val      Unemployment compensation benefits value
	    int_yn      Interest received
	    int_val     Interest income received, amount+
	    ffpos       Record type and sequence indicator (similar numbers
	    			indicate in same family)
	    fh_seq      Household sequence number (similar number indicate same Household)
	    finc_ssi    Supplemental Security benefits (Family)
	    ftot_r      Total family income
	    ftotval     Total family income
	    ptot_r      Person income, total
	    ptotval     Person income, total
	    peridnum    Unique Person identifier
	'''
	# Replacing values with numbers so that we can use in LE regression:
	CPS = CPS.replace({'None or not in universe' : 0.}, regex = True)
	CPS = CPS.replace({'Not in universe' : 0.}, regex = True)
	CPS = CPS.replace({'NIU' : 0.}, regex = True)
	CPS = CPS.replace({'Did not receive SSI' : 0.}, regex = True)
	CPS = CPS.replace({'Received SSI' : 1.}, regex = True)
	CPS = CPS.replace({'Female' : 1}, regex = True)
	CPS = CPS.replace({'Male' : 1}, regex = True)


	CPS.a_age = np.where(CPS.a_age == "80-84 years of age",
	                             random.randrange(80, 84),
	                             CPS.a_age)
	CPS.a_age = np.where(CPS.a_age == "85+ years of age",
	                             random.randrange(85, 95),
	                             CPS.a_age)
	CPS.a_age = pd.to_numeric(CPS.a_age)
	CPS_earned = (CPS[['wsal_val' ,'semp_val','frse_val']].astype(float)).copy()
	earned_income = CPS_earned.sum(axis = 1)
	CPS['earned_income'] = earned_income

	return CPS


CPS = Init_CPS_Vars(CPS)
# Saving variables, and smaller, more managable df to .csv file:
CPS.to_csv('CPS_SS.csv', index=False)
