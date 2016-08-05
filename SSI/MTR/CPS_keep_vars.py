import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from patsy import dmatrix
import csv

'''This code is used to define the variables that we use in SSI MTR computation later 
when we use regressions , Income Rules, and machine learning techniques'''

CPS = pd.read_csv('cpsmar2014t.csv')


columns_to_keep = ['ssi_val', 'ssi_yn','csp_val', 'rnt_val', 'div_val', 'vet_val', 'a_maritl', 'ssikidyn', 'resnssi1', 'resnssi2', 'marsupwt', 'a_age', 'gestfips',
                   'pedisdrs', 'pedisear', 'pediseye', 'pedisout', 'pedisphy', 'pedisrem',
                   'dis_hp', 'rsnnotw', 'vet_typ1', 'pemlr', 'mcare',
                   'wsal_val', 'semp_val', 'frse_val', 'ss_val', 'rtm_val', 'oi_off', 'oi_val',
                   'uc_yn', 'uc_val', 'int_yn', 'int_val', 
                   'ffpos', 'fh_seq', 'finc_ssi', 'ftot_r', 'ftotval', 'ptot_r', 'ptotval',
                   'peridnum', 'paw_yn', 'filestat', 'a_ftpt', 'a_spouse']

CPS = CPS[columns_to_keep]

def count_parents(earned_income, unearned_income, deemed = None):
	'''Here we define our countable income variable'''

	SSI_countable = earned_income + unearned_income

	#Allowing only positive values
	SSI_countable = np.where(SSI_countable > 0, SSI_countable, 0)
	# Taking $20 out of most income
	SSI_countable = np.where(SSI_countable > 20 * 12, SSI_countable - 20 *12, 0)
	#Case where unearned income < 20 *12 and earned income + unearned income - 20 * 12 <= 65 * 12
	SSI_countable = np.where((SSI_countable < 65 * 12) & (earned_income >= 65 * 12), 0, SSI_countable)
	#Case where unearned income < 20 * 12, earned income < 65 * 12 but countable income > earned income
	SSI_countable = np.where((SSI_countable < 65 * 12) & (earned_income < 65 * 12) & (SSI_countable > earned_income),  SSI_countable - earned_income, SSI_countable)
	#Case where unearned income < 20 *12, earned income < 65 *12 and countable < earned income
	SSI_countable = np.where((SSI_countable < 65 * 12) & (earned_income < 65 * 12) & (SSI_countable <= earned_income), 0, SSI_countable)
	#Case where unearned income > 20*12 and earned income < 65 * 12
	SSI_countable = np.where((SSI_countable >= 65 * 12) & (earned_income < 65 * 12), SSI_countable - earned_income, SSI_countable)
	#Case where countable is greater than 65 * 12
	SSI_countable = np.where((SSI_countable >= 65 * 12) & (earned_income >= 65 * 12), SSI_countable - 65 * 12, SSI_countable)

	# Taking out half of what's left of earnings
	#Accounting for when the initial $20 is taken out of earned income
	SSI_countable = np.where((earned_income > 65 * 12) & (SSI_countable > 0) & (unearned_income < 20 * 12), SSI_countable - 0.5*(earned_income - (20 * 12 - unearned_income) - 65 * 12), SSI_countable)
    # Case when all initial $20 is taken from unearned income
	SSI_countable = np.where((earned_income > 65 * 12) & (SSI_countable > 0) & (unearned_income >= 20 * 12), SSI_countable - 0.5*(earned_income - 65 * 12), SSI_countable)

	# if deemed is not None:
	# 	SSI_countable += deemed

	return SSI_countable


def Init_CPS_Vars(CPS):

	'''
	Here we initialize the variables that we use in the
	CPS, and clean the data.

				Variables within the CPS that we use:

	    ssi_val     Supplemental Security income amount received
	    ssi_yn      Supplemental Security income received
	    ssikidyn    Supplemental Security income, child received
	    rednssi1    Supplemental Security income, reason 1
	    rednssi2    Supplemental Security income, reason 2
	    marsupwt    March supplement final weight
	    a_age       Age
	    gestfips    FIPS State Code
	    pedisdrs    Disability, dressing or bathing
	    pedisear    Disability, hearing
	    pediseye    Disability, seeing
	    pedisout    Disability, doctor visits, shopping alone
	    pedisphy    Disability, walking, climbing stairs
	    pedisrem    Disability, remembering
	    dis_hp      Health problem or a disability which prevents working
	    rsnnotw     Reason for not working
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
	CPS = CPS.replace({'None or not in universe' : 0.}, regex = True)
	CPS = CPS.replace({'Not in universe' : 0.}, regex = True)
	CPS = CPS.replace({'NIU' : 0.}, regex = True)
	CPS = CPS.replace({'Did not receive SSI' : 0.}, regex = True)
	CPS = CPS.replace({'Received SSI' : 1.}, regex = True)


	CPS.a_age = np.where(CPS.a_age == "80-84 years of age",
	                             random.randrange(80, 84),
	                             CPS.a_age)
	CPS.a_age = np.where(CPS.a_age == "85+ years of age",
	                             random.randrange(85, 95),
	                             CPS.a_age)
	CPS.a_age = pd.to_numeric(CPS.a_age)

	CPS_unearned = (CPS[['ss_val' ,'rtm_val' ,'oi_val' , 'uc_val' , 'int_val', 'rnt_val', 'div_val', 'vet_val', 'csp_val']].astype(float)).copy()
	unearned_income = CPS_unearned.sum(axis = 1)

	CPS_earned = (CPS[['wsal_val' ,'semp_val','frse_val']].astype(float)).copy()
	earned_income = CPS_earned.sum(axis = 1)

	count_income = count_parents(unearned_income, earned_income)

	too_high = np.percentile(count_income[CPS.index[CPS.ssi_yn=='Yes']], 95)
	low_income = (count_income <= too_high)
	SSI_target_pool = np.zeros(len(CPS))
	SSI_target_pool = np.where(low_income, 1, 0)
	CPS['current_recipient'] = np.where((CPS.ssi_yn =='Yes'), 1, 0)
	SSI_target_pool = np.where((CPS.current_recipient==1) & (SSI_target_pool ==1), 1, 0)

	CPS['SSI_target_pool'] = SSI_target_pool
	CPS['Countable_Income'] = count_income
	CPS['ssi_val'] = (CPS['ssi_val'].astype(float)).copy()
	CPS['earned_income'] = earned_income
	CPS['unearned_income'] = unearned_income
	no_public_assistance = (CPS.paw_yn != 'Yes')
	not_married_HH = (CPS.filestat=='Single')|(CPS.filestat=='Nonfiler')
	student_under22 = (CPS.a_ftpt != 'Not in universe or children and')&(CPS.a_age < 22)
	under_18 = (CPS.a_age < 18)
	d1 = (CPS.dis_hp == 'Yes')
	d2 = (CPS.pemlr == 'Not in labor force - disabled')
	d3 = (CPS.rsnnotw == 'Ill or disabled')
	d4 = (CPS.vet_typ1 == 'Yes')
	d5 = (CPS.a_age < 65) & (CPS.mcare == 'Yes')
	work_disability = (d1|d2|d3|d4|d5)
	disability = pd.to_numeric(np.where(CPS.oi_off=='State disability payments', CPS.oi_val, 0))
	# Calculating deemed income:
	ineligible_children = np.where(no_public_assistance & (student_under22|under_18), 1, 0)
	combined = np.where((disability==1)|(work_disability==1), 1, 0)
	aged = np.where(CPS.a_age>=65, 1, 0)
	subfields = {'old_disabled': combined&aged,
	             'young_disabled': combined&(1-aged),
	             'old_not_disabled': (1-combined)&aged,
	             'young_not_disabled': (1-combined)&(1-aged)}
	low_income = np.zeros(len(CPS))
	for key in subfields:
	    benchmark = np.percentile(count_income[CPS.index[(CPS.ssi_yn=='Yes')&subfields[key]]], 95)
	    low_income = np.where((count_income < benchmark) & subfields[key], 1, 0)
	CPS['low_income'] = low_income
	CPS['ineligible_children'] = ineligible_children
	household = CPS.groupby(['fh_seq', 'ffpos'], as_index=False)
	eligible_per_family = household.low_income.sum()
	ineligible_child_per_family = household.ineligible_children.sum()

	parents = household.nth([0,1])
	parents = parents[(parents.a_spouse!='None or children')]

	parents_eligible = pd.merge(parents, eligible_per_family, on=['fh_seq', 'ffpos'])
	parents_eligible = pd.merge(parents_eligible, ineligible_child_per_family, on=['fh_seq', 'ffpos'])
	parents_eligible = parents_eligible[parents_eligible.low_income_y==1]

	parents_countable = count_parents(parents_eligible.earned_income, parents_eligible.unearned_income)
	children_cost = parents_eligible.ineligible_children_y * (1100-733) * 12
	parents_eligible['deemed_income'] = np.where(parents_eligible.low_income_x==0,
	                                             parents_countable - children_cost,
	                                             0)

	parents_eligible = parents_eligible[['peridnum', 'deemed_income']]
	CPS = pd.merge(CPS, parents_eligible, on=['peridnum'], how='left')
	return CPS


CPS = Init_CPS_Vars(CPS)
# Here we used created dummy variables for each state:
# CPS_dummy_states1 = dmatrix('C(gestfips)- 1', CPS, return_type='dataframe')
# i = 0
# for name in CPS_dummy_states1.columns.values:
# 	CPS_dummy_states1.rename(columns={name : str('gestfips: '+ str(i))}, inplace=True)
# 	i+=1
# CPS = pd.concat([CPS, CPS_dummy_states1], axis = 1)
# CPS.to_csv('CPS_SSI_2013.csv', index=False)

CPS['deemed_income'] = CPS['deemed_income'].fillna(0)
deemed_swap = CPS.groupby(['fh_seq', 'ffpos'], as_index = False)
parents = deemed_swap.nth([0,1])
parents = parents[(parents.a_spouse!='None or children')]
house_numbers = np.sort(parents.copy().fh_seq.unique())
family_numbers = np.sort(parents.copy().ffpos.unique())
# Switching deemed income amounts to spouses for computation:
for i in house_numbers.copy():
	for j in family_numbers.copy():
		parents.loc[parents[(parents['fh_seq'] ==i) & (parents['ffpos']==j)].index,'deemed_income'] = parents['deemed_income'][(parents['fh_seq'] ==i) & (parents['ffpos']==j)].values[::-1]

parents = parents[['peridnum', 'deemed_income']]
CPS = pd.merge(CPS, parents, on=['peridnum'], how='left')
eligible_swap = CPS.groupby(['fh_seq', 'ffpos'], as_index = False)
parents = eligible_swap.nth([0,1])
parents = parents[(parents.a_spouse!='None or children')]
house_numbers = np.sort(parents.copy().fh_seq.unique())
family_numbers = np.sort(parents.copy().ffpos.unique())
CPS['eligible_spouse'] = np.zeros((len(CPS['earned_income'])))
# Switching whether or not their spouse is eligible to their spouse
for i in house_numbers.copy():
	for j in family_numbers.copy():
		parents.loc[parents[(parents['fh_seq'] ==i) & (parents['ffpos']==j)].index,'eligible_spouse'] = parents['current_recipient'][(parents['fh_seq'] ==i) & (parents['ffpos']==j)].values[::-1]

parents = parents[['peridnum', 'eligible_spouse']]
CPS = pd.merge(CPS, parents, on=['peridnum'], how='left')
CPS.to_csv('CPS_SSI.csv', index=False)

