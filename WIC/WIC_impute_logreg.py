'''                                 
About: 
    This script imputes Women, Infants and Children (WIC) participation using logistic regression. 
     Please refer to the documentation in the same folder 
    for more details on methodology and assumptions. The output this script is a personal level dataset that contains CPS
    individual participation indicator for women, infants, and children seperately (WIC participationc, 0 - not a recipient, 
    1 - current recipient on file, 2 - imputed recipient), and benefit amount.

Input: 
    2015 CPS (asec2015_pubuse.csv), number of recipients and their benefits amount by state in 2014 (Administrative.csv),

Output: 
    WIC_imputation_{}_logreg.csv.format(either women, infants, or children) for women, infants, and children separately.
 
Additional Source links: 
    USDA FY 2014 administrative data at https://www.fns.usda.gov/pd/wic-program 
    (download FY 2014 (final) from the monthy program and benefit section )
'''
import pandas as pd
from pandas import DataFrame
import numpy as np
import random
import statsmodels.discrete.discrete_model as sm
import matplotlib.pyplot as plt

def accuracy(targets, probabilities):
    '''
    This function takes estimated probabilites and rounds them up to 1 or down to zero in order to produce an accuracy metric for logistic regression commensurable with that of random forests.

    Args:

    targets: the correct answers, i.e. what the model is trying to estimate.
    probabilities: the estimated probabilities

    Returns:

    acc: the percentage of correct guesses after rounding the probabilities.
    '''
    predictions = np.round(probabilities)
    n = len(targets)
    n_wrong = np.sum(np.abs(predictions - targets))
    acc = (n - n_wrong)/n
    return acc

# Whether or not to print accuracy scores for logistic regression estimation
print_acc = True

# Administrative level data. 'Admin_totals_all.csv' obtained from create_admin.py
Admin_totals =  pd.read_csv('Admin_totals_all.csv')
Admin_totals[['Fips','Avg_benefit']].to_csv('avg.csv')



# Variables we use in CPS:
CPS_dataset = pd.read_csv('../asec2015_pubuse.csv')
columns_to_keep = ['frelu6', 'a_pfrel', 'rsnnotw', 'hfoodsp','ch_mc', 'hh5to18', 'fownu6', 'caid', 
'cov_hi', 'fwsval', 'mcaid','hrwicyn','wicyn','oi_off','paw_yn','paw_typ','paw_val', 
  'agi', 'tax_inc', 'peioocc', 'a_wksch', 'wemind', 'hprop_val','housret', 'prop_tax','fhoussub', 'fownu18', 'fpersons','fspouidx', 'prcitshp', 'gestfips','marsupwt','a_age','wsal_val','semp_val','frse_val',
                  'ss_val','rtm_val','div_val','oi_off','oi_val','uc_yn','uc_val', 'int_yn', 'int_val','pedisdrs', 'pedisear', 'pediseye', 
                    'pedisout', 'pedisphy', 'pedisrem','a_sex','peridnum','h_seq','fh_seq', 'ffpos', 'fsup_wgt',
                        'hlorent', 'hpublic', 'hsup_wgt', 'hfdval', 'f_mv_fs', 'a_famrel', 'a_ftpt']
CPS_dataset = CPS_dataset[columns_to_keep]
CPS_dataset = CPS_dataset.replace({'None or not in universe' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'Not in universe' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'NIU' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'None' : 0}, regex = True)
# CPS_dataset.to_csv('CPS_WIC.csv', index=False)
# CPS_dataset = pd.read_csv('CPS_WIC.csv')


#recipient or not of WIC
CPS_dataset.wicyn = np.where(CPS_dataset.wicyn == 'Did not receive WIC', 0, CPS_dataset.wicyn)
CPS_dataset.wicyn = np.where(CPS_dataset.wicyn == 'Received WIC', 1, CPS_dataset.wicyn)
CPS_dataset.wicyn = CPS_dataset.wicyn.astype(int)

CPS_dataset.hrwicyn = np.where(CPS_dataset.hrwicyn == 'NO', 0, CPS_dataset.hrwicyn)
CPS_dataset.hrwicyn = np.where(CPS_dataset.hrwicyn == 'YES', 1, CPS_dataset.hrwicyn)
CPS_dataset.hrwicyn = CPS_dataset.hrwicyn.astype(int)


CPS_dataset.a_age = np.where(CPS_dataset.a_age == "80-84 years of age",
                             random.randrange(80, 84),
                             CPS_dataset.a_age)
CPS_dataset.a_age = np.where(CPS_dataset.a_age == "85+ years of age",
                             random.randrange(85, 95),
                             CPS_dataset.a_age)
CPS_dataset.a_age = pd.to_numeric(CPS_dataset.a_age)

# eliminate errors
CPS_dataset.wicyn = np.where((CPS_dataset.wicyn > 0) & ((CPS_dataset.a_age < 15) | (CPS_dataset.a_age > 44)), 0, CPS_dataset.wicyn)

CPS_dataset.gestfips = pd.to_numeric(CPS_dataset.gestfips)

CPS_dataset.fpersons = pd.to_numeric(CPS_dataset.fpersons)

# Creating child WIC recipients
wic_grouped = CPS_dataset.groupby(['fh_seq', 'ffpos'])['wicyn'].sum()
positive = wic_grouped[(wic_grouped == 1)].index.values
CPS_dataset = CPS_dataset.set_index(['fh_seq', 'ffpos'])
CPS_dataset['fwicyn'] = 0
CPS_dataset.loc[positive, 'fwicyn'] = 1
CPS_dataset = CPS_dataset.reset_index()

#Including only moms with infants 0 years old and pregnant moms (no children under 5 in household)
CPS_dataset['infant'] = np.where(CPS_dataset.a_age < 1,  1, 0)
CPS_dataset['child'] = np.where(CPS_dataset.a_age <= 5, 1 , 0)
infants = CPS_dataset.groupby(['fh_seq', 'ffpos'])['infant'].sum()
children = CPS_dataset.groupby(['fh_seq', 'ffpos'])['child'].sum()
has_infant = infants[(infants > 0)].index.values
no_children = children[(children == 0)].index.values
CPS_dataset = CPS_dataset.set_index(['fh_seq', 'ffpos'])
CPS_dataset['lactating_or_pregnant_mom'] = 0
CPS_dataset.loc[no_children ,'lactating_or_pregnant_mom'] = 1
CPS_dataset.loc[has_infant ,'lactating_or_pregnant_mom'] = 1
CPS_dataset = CPS_dataset.reset_index()

# Giving eligible children the benefit
CPS_dataset['eligible_children'] = np.where((CPS_dataset.a_age <= 4) & (CPS_dataset.fwicyn == 1), 1, 0)
CPS_dataset['old_wic'] = CPS_dataset['wicyn']
CPS_dataset['wicyn'] = np.where((CPS_dataset.eligible_children == 1) | ((CPS_dataset.lactating_or_pregnant_mom == 1) & (CPS_dataset.wicyn == 1)), 1 , 0)
CPS_dataset['WIC_infant'] = np.where((CPS_dataset.wicyn == 1) & (CPS_dataset.a_age == 0), 1, 0)
CPS_dataset['WIC_child'] = np.where((CPS_dataset.wicyn == 1) & (CPS_dataset.a_age <= 4) & (CPS_dataset.a_age >= 1), 1, 0)
CPS_dataset['WIC_woman'] = np.where((CPS_dataset.wicyn == 1) & (CPS_dataset.a_age >= 15), 1, 0)

#Earned income
p_earned = CPS_dataset.wsal_val.astype(int) + CPS_dataset.semp_val.astype(int) + CPS_dataset.frse_val.astype(int) #individual earned income
CPS_dataset['p_earned'] = p_earned

#Unearned income / without uemployment compensation
p_unearned = CPS_dataset.ss_val.astype(int) + CPS_dataset.rtm_val.astype(int) + CPS_dataset.div_val.astype(int) + CPS_dataset.oi_val.astype(int) + CPS_dataset.int_val.astype(int) #individual unearned income
CPS_dataset['p_unearned'] = p_unearned

CPS_dataset['p_total'] = CPS_dataset.p_earned + CPS_dataset.p_unearned

#f_mv_fs
CPS_dataset.f_mv_fs = pd.to_numeric(CPS_dataset.f_mv_fs)

#elderly (age of at least 62) 
CPS_dataset['elderly'] = 0
CPS_dataset.elderly = np.where(CPS_dataset.a_age > 61, 1, CPS_dataset.elderly)

CPS_dataset.ch_mc = np.where((CPS_dataset.ch_mc == 'No') | (CPS_dataset.ch_mc == 'Not child\'s record'), 0, CPS_dataset.ch_mc)
CPS_dataset.ch_mc = np.where(CPS_dataset.ch_mc == 'Yes', 1, CPS_dataset.ch_mc)
CPS_dataset.ch_mc = CPS_dataset.ch_mc.astype(int)

CPS_dataset.cov_hi = np.where((CPS_dataset.cov_hi == 'No'), 0, CPS_dataset.cov_hi)
CPS_dataset.cov_hi = np.where(CPS_dataset.cov_hi == 'Yes', 1, CPS_dataset.cov_hi)
CPS_dataset.cov_hi = CPS_dataset.cov_hi.astype(int)

CPS_dataset.hfoodsp = np.where((CPS_dataset.hfoodsp == 'No'), 0, CPS_dataset.hfoodsp)
CPS_dataset.hfoodsp = np.where(CPS_dataset.hfoodsp == 'Yes', 1, CPS_dataset.hfoodsp)
CPS_dataset.hfoodsp = CPS_dataset.hfoodsp.astype(int)
CPS_dataset.hfdval = np.where((CPS_dataset.hfdval > 0), 1, CPS_dataset.hfdval)

CPS_dataset.a_pfrel = np.where(CPS_dataset.a_pfrel == 'Wife', 1, 0)
CPS_dataset.rsnnotw = np.where((CPS_dataset.rsnnotw == 'Taking care of home or family'), 1, 0)


CPS_dataset.caid = np.where(CPS_dataset.caid == 'Yes', 1, 0)


def income_lim_indicator(income, family_size, gestfips):
    ''''base and step sizes are from USDA's 2013-2014 income eligibility guidelines'''
    if (gestfips != 2) & (gestfips != 15):
        base = 21590
        step = 7511
        if income <= (base + (step * (family_size - 1))):
            return 1
        else:
            return 0
    elif gestfips == 2:
        base = 26973
        step = 9398
        if income <= (base + (step * (family_size - 1))):
            return 1
        else:
            return 0
    else:
        base = 24827
        step = 8640
        if income <= (base + (step * (family_size - 1))):
            return 1
        else:
            return 0

CPS_dataset['income_eligibility'] = CPS_dataset.apply(lambda x: income_lim_indicator(x['p_total'], x['fpersons'],
    x['gestfips']), axis=1)
        
#disabled - must be able to work to receive UI
CPS_dataset['disability'] = np.zeros(len(CPS_dataset))
CPS_dataset.disability = np.where(CPS_dataset.pedisdrs == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisear == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pediseye == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisout == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisphy == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisrem == 'Yes', 1, CPS_dataset.disability)


#Regression
CPS_dataset['intercept'] = np.ones(len(CPS_dataset))
CPS_dataset['indicator'] = CPS_dataset.WIC_infant
CPS_dataset['infant'] = np.where(CPS_dataset.a_age < 1, 1, 0)
model = sm.Logit(endog = CPS_dataset.indicator, exog = CPS_dataset[['intercept','hfdval','cov_hi','ch_mc' ,'infant', 'fwsval' ]].astype(float))
logit_res = model.fit()
probs = logit_res.predict()
targets = CPS_dataset.WIC_infant
print 'Accuracy score for logistic regression estimation (Infants)', accuracy(targets, probs)
CPS_dataset['probs'] = probs

# CPS total benefits and Administrative total benefits
state_benefit = {}
state_recipients = {}
CPS_dataset['WIC_val'] = 0

for fip in Admin_totals.Fips:
    this_state = (CPS_dataset.gestfips == fip)
    CPS_dataset.loc[this_state & (CPS_dataset.indicator == 1),'WIC_val'] = Admin_totals['Avg_benefit'][Admin_totals['Fips'] == fip].values[0]
    CPS_totalb = (CPS_dataset.WIC_val[CPS_dataset.indicator == 1] * CPS_dataset.marsupwt[CPS_dataset.indicator == 1])[this_state].sum() # The CPS subsidy amount is montly 
    admin_totalb =  Admin_totals['tot_infant_benefits'][Admin_totals.Fips == fip].values 
    CPS_totaln = CPS_dataset.marsupwt[this_state & CPS_dataset.indicator==1].sum() 
    admin_totaln =  Admin_totals['total_infants'][Admin_totals.Fips == fip].values
    temp = [Admin_totals.state[Admin_totals['Fips'] == fip].values[0], CPS_totalb, admin_totalb[0], CPS_totaln, admin_totaln[0]]
    state_benefit[fip] = temp

pre_augment_benefit = DataFrame(state_benefit).transpose()
pre_augment_benefit.columns = ['State', 'CPS infant benefits (annually)','Admin infant benefits (annually)',
                               'CPS total infant recipients','Admin total infant recipients']

pre_augment_benefit['Admin infant benefits (annually)'] = pre_augment_benefit['Admin infant benefits (annually)'].astype(int)
pre_augment_benefit['CPS infant benefits (annually)'] = pre_augment_benefit['CPS infant benefits (annually)'].astype(int)
pre_augment_benefit['CPS total infant recipients'] = pre_augment_benefit['CPS total infant recipients'].astype(int)
pre_augment_benefit.to_csv('admin_cps_totals_before.csv')


# caculate difference of UI stats and CPS aggregates on recipients number
# by state
diff = {'Fips':[],'Difference in Population':[],'Mean Benefit':[],'CPS Population':[],'WIC Population':[]}
diff['Fips'] = Admin_totals.Fips
current = (CPS_dataset.indicator==1)
for FIPS in Admin_totals.Fips:
        this_state = (CPS_dataset.gestfips == FIPS)
        current_tots = CPS_dataset.marsupwt[current&this_state].sum()
        valid_num = CPS_dataset.marsupwt[current&this_state].sum() + 0.0000001
        current_mean = ((CPS_dataset.WIC_val * CPS_dataset.marsupwt)[current&this_state].sum())/valid_num
        diff['CPS Population'].append(current_tots)
        diff['WIC Population'].append(float(Admin_totals["total_infants"][Admin_totals.Fips == FIPS]))
        diff['Difference in Population'].append(float(Admin_totals["total_infants"][Admin_totals.Fips == FIPS])- current_tots)
        diff['Mean Benefit'].append(current_mean)



d = DataFrame(diff)
d = d[['Fips', 'Mean Benefit', 'Difference in Population', 'CPS Population', 'WIC Population']]
d.to_csv('recipients_diff.csv')


CPS_dataset['impute'] = np.zeros(len(CPS_dataset))
CPS_dataset['WIC_impute'] = np.zeros(len(CPS_dataset))

non_current = (CPS_dataset.indicator==0)
current = (CPS_dataset.indicator==1)
random.seed()
over_states = 0
over_amount_total = 0

for FIPS in Admin_totals.Fips:
    
        # print ('we need to impute', d['Difference in Population'][d['Fips'] == FIPS].values[0], 'for state', FIPS)
        
        if d['Difference in Population'][d['Fips'] == FIPS].values[0] < 0:
            continue
        else:
            this_state = (CPS_dataset.gestfips==FIPS)
            not_imputed = (CPS_dataset.impute==0)
            pool_index = CPS_dataset[this_state&not_imputed&non_current].index
            pool = DataFrame({'weight': CPS_dataset.marsupwt[pool_index], 'prob': probs[pool_index]},
                            index=pool_index)
            pool = pool.sort_values(by = 'prob', ascending=False)
            pool['cumsum_weight'] = pool['weight'].cumsum()
            pool['distance'] = abs(pool.cumsum_weight-d['Difference in Population'][d['Fips'] == FIPS].values)
            min_index = pool.sort_values(by='distance')[:1].index
            min_weight = int(pool.loc[min_index].cumsum_weight)
            pool['impute'] = np.where(pool.cumsum_weight<=min_weight+10 , 1, 0)
            CPS_dataset.loc[pool.index[pool['impute']==1], 'impute'] = 1
            CPS_dataset.loc[pool.index[pool['impute']==1], 'WIC_impute'] = Admin_totals['Avg_benefit'][Admin_totals['Fips'] ==FIPS].values[0]
            # print ('Method1: regression gives', 
            #     CPS_dataset.marsupwt[(CPS_dataset.impute==1)&this_state].sum()) 

#Adjustment ratio
results = {}

imputed = (CPS_dataset.impute == 1)
has_val = (CPS_dataset.indicator == 1)
no_val = (CPS_dataset.WIC_val == 0)

for FIPS in Admin_totals.Fips:
    this_state = (CPS_dataset.gestfips == FIPS)
    current_total = (CPS_dataset.WIC_val * CPS_dataset.marsupwt)[this_state].sum() 
    imputed_total = (CPS_dataset.WIC_impute * CPS_dataset.marsupwt)[this_state & imputed].sum()
    on_file = current_total + imputed_total
    admin_total = Admin_totals.tot_infant_benefits[Admin_totals.Fips == FIPS]
    adjust_ratio = admin_total / on_file
    this_state_num = [Admin_totals['state'][Admin_totals.Fips == FIPS].values[0], on_file, admin_total.values[0], adjust_ratio.values[0]]
    results[FIPS] = this_state_num
    CPS_dataset.WIC_impute = np.where(has_val & this_state, CPS_dataset.WIC_val * adjust_ratio.values, CPS_dataset.WIC_impute)
    CPS_dataset.WIC_impute = np.where(no_val & this_state, CPS_dataset.WIC_impute * adjust_ratio.values, CPS_dataset.WIC_impute)
CPS_dataset["WIC_participation"] = np.zeros(len(CPS_dataset))
CPS_dataset["WIC_participation"] = np.where(CPS_dataset.impute == 1, 2, 0) #Augmented
CPS_dataset['WIC_participation'] = np.where(has_val, 1, CPS_dataset.WIC_participation)
r = DataFrame(results).transpose()
r.columns = ['State', 'Imputed', 'Admin', 'adjust ratio']
r['Imputed'] = r['Imputed'].astype(int)
r['adjust ratio'] *= 10000
r['adjust ratio'] = r['adjust ratio'].astype(int)
r['adjust ratio'] /= 10000
r.to_csv('amount.csv', index=False)

CPS_dataset.to_csv('WIC_imputation_infants_logreg.csv', 
                   columns=['peridnum','WIC_participation', 'WIC_impute'])


## Checking post-adjustment totals to see if they match admin totals
CPS_dataset.WIC_participation = np.where(CPS_dataset.WIC_participation == 2 , 1, CPS_dataset.WIC_participation)
CPS_dataset['after_totals_reciepts'] = (CPS_dataset.WIC_participation * CPS_dataset.marsupwt)
CPS_dataset['after_totals_outlays'] = (CPS_dataset.WIC_impute * CPS_dataset.marsupwt)
total_outlays = CPS_dataset.groupby(['gestfips'])['after_totals_outlays'].sum().astype(int).reset_index(drop = True)
total_recipients = CPS_dataset.groupby(['gestfips'])['after_totals_reciepts'].sum().astype(int).reset_index(drop = True)

df = pd.DataFrame()
df['State'] = Admin_totals.state
df['post augment CPS total benefits (annual)'] = total_outlays
df['post augment CPS total infant recipients'] = total_recipients
df['Admin total benefits (annual)'] = Admin_totals['tot_infant_benefits'].astype(int)
df['Admin total recipients'] = Admin_totals['total_infants']
df.to_csv('post_augment_adminCPS_totals_logreg.csv')



'''   Imputation procedure for Children (ages 1- 4)       '''





# important features for children:
# 'a_age', 'fownu6', 'frelu6', 'ch_mc', 'm5gsame', 'hh5to18', 'm5g_mtr4', 
# 'm5g_cbst', 'hfoodsp', 'm5g_mtr3', 'hfdval', 'm5g_st', 'f_mv_fs', 
# 'hfdval_missing_NotIn', 'hfoodno', 'caid', 'hfoodmo', 'pxdisout', 
# 'paw_yn_missing_NotIn', 'ssi_yn_missing_NotIn', 'int_yn_missing_NotIn', 
# 'wemocg', 'hmcaid', 'mcaid', 'uc_yn_missing_NotIn', 'm5g_mtr1', 
# 'fam_earned_income', 'poccu2', 'pxdisear', 'weind', 'clwk', 'ptot_r_missing_NotIn', 
# 'earned_income', 'pxafwhn1', 'age1', 'weclw', 'hhotlun', 'pxdisrem', 'dephi', 

#Regression

CPS_dataset['intercept'] = np.ones(len(CPS_dataset))
CPS_dataset['indicator'] = CPS_dataset.WIC_child
CPS_dataset['child'] = np.where((CPS_dataset.a_age >= 1) & (CPS_dataset.a_age <= 4), 1, 0)
model = sm.Logit(endog = CPS_dataset.indicator, exog = CPS_dataset[['intercept','hfdval','cov_hi','ch_mc' ,'child', 'fwsval' ]].astype(float))
logit_res = model.fit()
probs = logit_res.predict()
targets = CPS_dataset.WIC_child
print 'Accuracy score for logistic regression estimation (Children)', accuracy(targets, probs)
CPS_dataset['probs'] = probs

# CPS total benefits and Administrative total benefits
state_benefit = {}
state_recipients = {}
CPS_dataset['WIC_val'] = 0

for fip in Admin_totals.Fips:
    this_state = (CPS_dataset.gestfips == fip)
    CPS_dataset.loc[this_state & (CPS_dataset.indicator == 1),'WIC_val'] = Admin_totals['Avg_benefit'][Admin_totals['Fips'] == fip].values[0]
    CPS_totalb = (CPS_dataset.WIC_val[CPS_dataset.indicator == 1] * CPS_dataset.marsupwt[CPS_dataset.indicator == 1])[this_state].sum() # The CPS subsidy amount is montly 
    admin_totalb =  Admin_totals['tot_child_benefits'][Admin_totals.Fips == fip].values 
    CPS_totaln = CPS_dataset.marsupwt[this_state & CPS_dataset.indicator==1].sum() 
    admin_totaln =  Admin_totals['total_children'][Admin_totals.Fips == fip].values
    temp = [Admin_totals.state[Admin_totals['Fips'] == fip].values[0], CPS_totalb, admin_totalb[0], CPS_totaln, admin_totaln[0]]
    state_benefit[fip] = temp

pre_augment_benefit = DataFrame(state_benefit).transpose()
pre_augment_benefit.columns = ['State', 'CPS child benefits (annually)','Admin child benefits (annually)',
                               'CPS total child recipients','Admin total child recipients']

pre_augment_benefit['Admin child benefits (annually)'] = pre_augment_benefit['Admin child benefits (annually)'].astype(int)
pre_augment_benefit['CPS child benefits (annually)'] = pre_augment_benefit['CPS child benefits (annually)'].astype(int)
pre_augment_benefit['CPS total child recipients'] = pre_augment_benefit['CPS total child recipients'].astype(int)
pre_augment_benefit.to_csv('admin_cps_totals_before_children.csv')


# caculate difference of UI stats and CPS aggregates on recipients number
# by state
diff = {'Fips':[],'Difference in Population':[],'Mean Benefit':[],'CPS Population':[],'WIC Population':[]}
diff['Fips'] = Admin_totals.Fips
current = (CPS_dataset.indicator==1)
for FIPS in Admin_totals.Fips:
        this_state = (CPS_dataset.gestfips == FIPS)
        current_tots = CPS_dataset.marsupwt[current&this_state].sum()
        valid_num = CPS_dataset.marsupwt[current&this_state].sum() + 0.0000001
        current_mean = ((CPS_dataset.WIC_val * CPS_dataset.marsupwt)[current&this_state].sum())/valid_num
        diff['CPS Population'].append(current_tots)
        diff['WIC Population'].append(float(Admin_totals["total_children"][Admin_totals.Fips == FIPS]))
        diff['Difference in Population'].append(float(Admin_totals["total_children"][Admin_totals.Fips == FIPS]) - current_tots)
        diff['Mean Benefit'].append(current_mean)



d = DataFrame(diff)
d = d[['Fips', 'Mean Benefit', 'Difference in Population', 'CPS Population', 'WIC Population']]
d.to_csv('recipients_diff.csv')


CPS_dataset['impute'] = np.zeros(len(CPS_dataset))
CPS_dataset['WIC_impute'] = np.zeros(len(CPS_dataset))

non_current = (CPS_dataset.indicator == 0)
current = (CPS_dataset.indicator == 1)
random.seed()
over_states = 0
over_amount_total = 0

for FIPS in Admin_totals.Fips:
    
        # print ('we need to impute', d['Difference in Population'][d['Fips'] == FIPS].values[0], 'for state', FIPS)
        
        if d['Difference in Population'][d['Fips'] == FIPS].values[0] < 0:
            continue
        else:
            this_state = (CPS_dataset.gestfips==FIPS)
            not_imputed = (CPS_dataset.impute==0)
            pool_index = CPS_dataset[this_state&not_imputed&non_current].index
            pool = DataFrame({'weight': CPS_dataset.marsupwt[pool_index], 'prob': probs[pool_index]},
                            index=pool_index)
            pool = pool.sort_values(by = 'prob', ascending=False)
            pool['cumsum_weight'] = pool['weight'].cumsum()
            pool['distance'] = abs(pool.cumsum_weight-d['Difference in Population'][d['Fips'] == FIPS].values)
            min_index = pool.sort_values(by='distance')[:1].index
            min_weight = int(pool.loc[min_index].cumsum_weight)
            pool['impute'] = np.where(pool.cumsum_weight<=min_weight+10 , 1, 0)
            CPS_dataset.loc[pool.index[pool['impute']==1], 'impute'] = 1
            CPS_dataset.loc[pool.index[pool['impute']==1], 'WIC_impute'] = Admin_totals['Avg_benefit'][Admin_totals['Fips'] ==FIPS].values[0]
            # print ('Method1: regression gives', 
            #     CPS_dataset.marsupwt[(CPS_dataset.impute==1)&this_state].sum()) 

#Adjustment ratio
results = {}

imputed = (CPS_dataset.impute == 1)
has_val = (CPS_dataset.indicator == 1)
no_val = (CPS_dataset.WIC_val == 0)

for FIPS in Admin_totals.Fips:
    this_state = (CPS_dataset.gestfips == FIPS)
    current_total = (CPS_dataset.WIC_val * CPS_dataset.marsupwt)[this_state].sum() 
    imputed_total = (CPS_dataset.WIC_impute * CPS_dataset.marsupwt)[this_state & imputed].sum()
    on_file = current_total + imputed_total
    admin_total = Admin_totals.tot_child_benefits[Admin_totals.Fips == FIPS]
    adjust_ratio = admin_total / on_file
    this_state_num = [Admin_totals['state'][Admin_totals.Fips == FIPS].values[0], on_file, admin_total.values[0], adjust_ratio.values[0]]
    results[FIPS] = this_state_num
    CPS_dataset.WIC_impute = np.where(has_val & this_state, CPS_dataset.WIC_val * adjust_ratio.values, CPS_dataset.WIC_impute)
    CPS_dataset.WIC_impute = np.where(no_val & this_state, CPS_dataset.WIC_impute * adjust_ratio.values, CPS_dataset.WIC_impute)
CPS_dataset["WIC_participation"] = np.zeros(len(CPS_dataset))
CPS_dataset["WIC_participation"] = np.where(CPS_dataset.impute == 1, 2, 0) #Augmented
CPS_dataset['WIC_participation'] = np.where(has_val, 1, CPS_dataset.WIC_participation)
r = DataFrame(results).transpose()
r.columns = ['State', 'Imputed', 'Admin', 'adjust ratio']
r['Imputed'] = r['Imputed'].astype(int)
r['adjust ratio'] *= 10000
r['adjust ratio'] = r['adjust ratio'].astype(int)
r['adjust ratio'] /= 10000
r.to_csv('amount.csv', index=False)

CPS_dataset.to_csv('WIC_imputation_children_logreg.csv', 
                   columns=['peridnum','WIC_participation', 'WIC_impute'])


## Checking post-adjustment totals to see if they match admin totals
CPS_dataset.WIC_participation = np.where(CPS_dataset.WIC_participation == 2 , 1, CPS_dataset.WIC_participation)
CPS_dataset['after_totals_reciepts'] = (CPS_dataset.WIC_participation * CPS_dataset.marsupwt)
CPS_dataset['after_totals_outlays'] = (CPS_dataset.WIC_impute * CPS_dataset.marsupwt)
total_outlays = CPS_dataset.groupby(['gestfips'])['after_totals_outlays'].sum().astype(int).reset_index(drop = True)
total_recipients = CPS_dataset.groupby(['gestfips'])['after_totals_reciepts'].sum().astype(int).reset_index(drop = True)

df = pd.DataFrame()
df['State'] = Admin_totals.state
df['post augment CPS total benefits (annual)'] = total_outlays
df['post augment CPS total child recipients'] = total_recipients
df['Admin total benefits (annual)'] = Admin_totals['tot_child_benefits'].astype(int)
df['Admin total recipients'] = Admin_totals['total_children']
df.to_csv('post_augment_adminCPS_totals_children_logreg.csv')



'''   Imputation procedure for Women (ages 15 - 44)       '''






# important features for women:
# 'fownu6', 'frelu6', 'a_age', 'a_pfrel', 'rsnnotw', 'age1', 'actc_crd',
#  'a_sex', 'hhdfmx', 'hfdval', 'pyrsn', 'hfdval_missing_NotIn', 'caid', 
#  'hunder15', 'eit_cred', 'hmcaid', 'hfoodno', 'a_famrel', 'paw_val', 
#  'hfoodmo', 'hrpaidcc', 'mcaid', 'hfoodsp', 'perrp', 'paw_typ', 'paw_mon',
#   'hunder18', 'f_mv_fs', 'earned_income', 'frspov', 'fkind', 'paw_yn', 
#   'ptot_r', 'pecohab', 'fwsval', 'h_type', 'povll', 'a_exprrp', 

#Regression
CPS_dataset['intercept'] = np.ones(len(CPS_dataset))
CPS_dataset['indicator'] = CPS_dataset.WIC_woman
CPS_dataset['has_child'] = np.where(CPS_dataset.fownu6 > 0 , 1 , 0 )
CPS_dataset['has_child_relative'] = np.where(CPS_dataset.frelu6 > 0 , 1 , 0 )

CPS_dataset['woman'] = np.where((CPS_dataset.a_age >= 15) & (CPS_dataset.a_age <= 44) & (CPS_dataset.a_sex == 'Female'), 1, 0)
model = sm.Logit(endog = CPS_dataset.indicator, exog = CPS_dataset[['intercept', 'rsnnotw' , 'has_child','hfdval','caid' , 'income_eligibility', 'woman', 'fwsval' ]].astype(float))
logit_res = model.fit()
probs = logit_res.predict()
targets = CPS_dataset.WIC_woman
print 'Accuracy score for logistic regression estimation (Women)',  accuracy(targets, probs)
CPS_dataset['probs'] = probs

# CPS total benefits and Administrative total benefits
state_benefit = {}
state_recipients = {}
CPS_dataset['WIC_val'] = 0

for fip in Admin_totals.Fips:
    this_state = (CPS_dataset.gestfips == fip)
    CPS_dataset.loc[this_state & (CPS_dataset.indicator == 1),'WIC_val'] = Admin_totals['Avg_benefit'][Admin_totals['Fips'] == fip].values[0]
    CPS_totalb = (CPS_dataset.WIC_val[CPS_dataset.indicator == 1] * CPS_dataset.marsupwt[CPS_dataset.indicator == 1])[this_state].sum() # The CPS subsidy amount is montly 
    admin_totalb =  Admin_totals['tot_woman_benefits'][Admin_totals.Fips == fip].values 
    CPS_totaln = CPS_dataset.marsupwt[this_state & CPS_dataset.indicator==1].sum() 
    admin_totaln =  Admin_totals['total_women'][Admin_totals.Fips == fip].values
    temp = [Admin_totals.state[Admin_totals['Fips'] == fip].values[0], CPS_totalb, admin_totalb[0], CPS_totaln, admin_totaln[0]]
    state_benefit[fip] = temp

pre_augment_benefit = DataFrame(state_benefit).transpose()
pre_augment_benefit.columns = ['State', 'CPS women benefits (annually)','Admin women benefits (annually)',
                               'CPS total women recipients','Admin total women recipients']

pre_augment_benefit['Admin women benefits (annually)'] = pre_augment_benefit['Admin women benefits (annually)'].astype(int)
pre_augment_benefit['CPS women benefits (annually)'] = pre_augment_benefit['CPS women benefits (annually)'].astype(int)
pre_augment_benefit['CPS total women recipients'] = pre_augment_benefit['CPS total women recipients'].astype(int)
pre_augment_benefit.to_csv('admin_cps_totals_before_women.csv')


# caculate difference of UI stats and CPS aggregates on recipients number
# by state
diff = {'Fips':[],'Difference in Population':[],'Mean Benefit':[],'CPS Population':[],'WIC Population':[]}
diff['Fips'] = Admin_totals.Fips
current = (CPS_dataset.indicator==1)
for FIPS in Admin_totals.Fips:
        this_state = (CPS_dataset.gestfips == FIPS)
        current_tots = CPS_dataset.marsupwt[current&this_state].sum()
        valid_num = CPS_dataset.marsupwt[current&this_state].sum() + 0.0000001
        current_mean = ((CPS_dataset.WIC_val * CPS_dataset.marsupwt)[current&this_state].sum())/valid_num
        diff['CPS Population'].append(current_tots)
        diff['WIC Population'].append(float(Admin_totals["total_women"][Admin_totals.Fips == FIPS]))
        diff['Difference in Population'].append(float(Admin_totals["total_women"][Admin_totals.Fips == FIPS]) - current_tots)
        diff['Mean Benefit'].append(current_mean)



d = DataFrame(diff)
d = d[['Fips', 'Mean Benefit', 'Difference in Population', 'CPS Population', 'WIC Population']]
d.to_csv('recipients_diff.csv')


CPS_dataset['impute'] = np.zeros(len(CPS_dataset))
CPS_dataset['WIC_impute'] = np.zeros(len(CPS_dataset))

non_current = (CPS_dataset.indicator == 0)
current = (CPS_dataset.indicator == 1)
random.seed()
over_states = 0
over_amount_total = 0

for FIPS in Admin_totals.Fips:
    
        # print ('we need to impute', d['Difference in Population'][d['Fips'] == FIPS].values[0], 'for state', FIPS)
        
        if d['Difference in Population'][d['Fips'] == FIPS].values[0] < 0:
            continue
        else:
            this_state = (CPS_dataset.gestfips==FIPS)
            not_imputed = (CPS_dataset.impute==0)
            pool_index = CPS_dataset[this_state&not_imputed&non_current].index
            pool = DataFrame({'weight': CPS_dataset.marsupwt[pool_index], 'prob': probs[pool_index]},
                            index=pool_index)
            pool = pool.sort_values(by = 'prob', ascending=False)
            pool['cumsum_weight'] = pool['weight'].cumsum()
            pool['distance'] = abs(pool.cumsum_weight-d['Difference in Population'][d['Fips'] == FIPS].values)
            min_index = pool.sort_values(by='distance')[:1].index
            min_weight = int(pool.loc[min_index].cumsum_weight)
            pool['impute'] = np.where(pool.cumsum_weight <= min_weight+10 , 1, 0)
            CPS_dataset.loc[pool.index[pool['impute']==1], 'impute'] = 1
            CPS_dataset.loc[pool.index[pool['impute']==1], 'WIC_impute'] = Admin_totals['Avg_benefit'][Admin_totals['Fips'] ==FIPS].values[0]
            # print ('Method1: regression gives', 
            #     CPS_dataset.marsupwt[(CPS_dataset.impute==1)&this_state].sum()) 

#Adjustment ratio
results = {}

imputed = (CPS_dataset.impute == 1)
has_val = (CPS_dataset.indicator == 1)
no_val = (CPS_dataset.WIC_val == 0)

for FIPS in Admin_totals.Fips:
    this_state = (CPS_dataset.gestfips == FIPS)
    current_total = (CPS_dataset.WIC_val * CPS_dataset.marsupwt)[this_state].sum() 
    imputed_total = (CPS_dataset.WIC_impute * CPS_dataset.marsupwt)[this_state & imputed].sum()
    on_file = current_total + imputed_total
    admin_total = Admin_totals.tot_woman_benefits[Admin_totals.Fips == FIPS]
    adjust_ratio = admin_total / on_file
    this_state_num = [Admin_totals['state'][Admin_totals.Fips == FIPS].values[0], on_file, admin_total.values[0], adjust_ratio.values[0]]
    results[FIPS] = this_state_num
    CPS_dataset.WIC_impute = np.where(has_val & this_state, CPS_dataset.WIC_val * adjust_ratio.values, CPS_dataset.WIC_impute)
    CPS_dataset.WIC_impute = np.where(no_val & this_state, CPS_dataset.WIC_impute * adjust_ratio.values, CPS_dataset.WIC_impute)
CPS_dataset["WIC_participation"] = np.zeros(len(CPS_dataset))
CPS_dataset["WIC_participation"] = np.where(CPS_dataset.impute == 1, 2, 0) #Augmented
CPS_dataset['WIC_participation'] = np.where(has_val, 1, CPS_dataset.WIC_participation)
r = DataFrame(results).transpose()
r.columns = ['State', 'Imputed', 'Admin', 'adjust ratio']
r['Imputed'] = r['Imputed'].astype(int)
r['adjust ratio'] *= 10000
r['adjust ratio'] = r['adjust ratio'].astype(int)
r['adjust ratio'] /= 10000
r.to_csv('amount.csv', index=False)

CPS_dataset.to_csv('WIC_imputation_women_logreg.csv', 
                   columns=['peridnum','WIC_participation', 'WIC_impute'])


## Checking post-adjustment totals to see if they match admin totals
CPS_dataset.WIC_participation = np.where(CPS_dataset.WIC_participation == 2 , 1, CPS_dataset.WIC_participation)
CPS_dataset['after_totals_reciepts'] = (CPS_dataset.WIC_participation * CPS_dataset.marsupwt)
CPS_dataset['after_totals_outlays'] = (CPS_dataset.WIC_impute * CPS_dataset.marsupwt)
total_outlays = CPS_dataset.groupby(['gestfips'])['after_totals_outlays'].sum().astype(int).reset_index(drop = True)
total_recipients = CPS_dataset.groupby(['gestfips'])['after_totals_reciepts'].sum().astype(int).reset_index(drop = True)

df = pd.DataFrame()
df['State'] = Admin_totals.state
df['post augment CPS total benefits (annual)'] = total_outlays
df['post augment CPS total women recipients'] = total_recipients
df['Admin total benefits (annual)'] = Admin_totals['tot_woman_benefits'].astype(int)
df['Admin total recipients'] = Admin_totals['total_women']
df.to_csv('post_augment_adminCPS_totals_women_logreg.csv')



