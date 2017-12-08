'''                                 
About: 
    This script imputes Unemployment Insurance (UI) compensation for regular and worshare programs using random forests.
    We exclude Extended Benefits (EB) since its totals are negligible relative to the regular program totals (<.1 %).
    We impute UI recipients, and their dollar benefit amount to match the aggregates with United States Department of 
    Labor (DOL) statistics for these programs. In this current version, we used 2015 CPS data since its UI question 
    refers to 2014 UI reception, and DOL's "ETA 5195" dataset using all UI program totals (state, UCX, UCFE) 
    for the regular UI, and State UI program totals for workshare. Please refer to the documentation in the same folder 
    for more details on methodology and assumptions. The output this script is a personal level dataset that contains CPS
    individual participation indicator (UI participationc, 0 - not a recipient, 
    1 - current recipient on file, 2 - imputed recipient), and benefit amount.

Input: 
    asec2015_pubuse.csv
    rf_probs.csv from Rf_probs.ipynb.

Output: 
    UI_Imputation_rf.csv 
 
Additional Source links: 
    DOL CY 2014 administrative data at https://workforcesecurity.doleta.gov/unemploy/DataDownloads.asp 
    (ETA 5195, both Regular and Worshare programs )
'''
import pandas as pd
from pandas import DataFrame
import numpy as np
import random
import matplotlib.pyplot as plt

# Administrative level data. 'Admin_totals_all.csv' obtained from create_admin.py
Admin_totals =  pd.read_csv('Admin_totals_all.csv')
Admin_totals['Avg_benefit'] = Admin_totals['tot_UI_outlays'] / Admin_totals['tot_UI_recipients']
Admin_totals['Avg_benefit'] *= 10000
Admin_totals['Avg_benefit'] = Admin_totals['Avg_benefit'].astype(int)
Admin_totals['Avg_benefit'] /= 10000
Admin_totals[['Fips','Avg_benefit']].to_csv('avg.csv')

# Random Forest probabilities of receiving UI compensation. 'rf_probs.csv' obtained from rf_probs.ipynb.
Rf_probs = np.loadtxt('rf_probs.csv')

# Variables we use in CPS:
CPS_dataset = pd.read_csv('../asec2015_pubuse.csv')
columns_to_keep = ['lkweeks', 'lkstrch', 'weuemp', 'wkswork', 'a_explf', 'a_lfsr', 'pruntype', 'a_untype', 'prwkstat',
  'a_age', 'age1', 'ptotval', 'ptot_r', 'pemlr', 'pyrsn', 'filestat', 
  'a_wkslk', 'hrswk', 'agi', 'tax_inc', 'peioocc', 'a_wksch', 'wemind', 'hprop_val','housret', 'prop_tax','fhoussub', 'fownu18', 'fpersons','fspouidx', 'prcitshp', 'gestfips','marsupwt','a_age','wsal_val','semp_val','frse_val',
                  'ss_val','rtm_val','div_val','oi_off','oi_val','uc_yn','uc_val', 'int_yn', 'int_val','pedisdrs', 'pedisear', 'pediseye', 
                    'pedisout', 'pedisphy', 'pedisrem','a_sex','peridnum','h_seq','fh_seq', 'ffpos', 'fsup_wgt',
                        'hlorent', 'hpublic', 'hsup_wgt', 'hfdval', 'fmoop', 'f_mv_fs', 'pppos', 'a_famrel', 'a_ftpt']
CPS_dataset = CPS_dataset[columns_to_keep]
CPS_dataset = CPS_dataset.replace({'None or not in universe' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'Not in universe' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'Not in Universe' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'NIU' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'None' : 0}, regex = True)
# CPS_dataset.to_csv('CPS_UI.csv', index=False)
# CPS_dataset = pd.read_csv('CPS_UI.csv')

#recipient or not of Unemployment Insurance Compensation
CPS_dataset.uc_yn = np.where(CPS_dataset.uc_yn == 'No', 0, CPS_dataset.uc_yn)
CPS_dataset.uc_yn = np.where(CPS_dataset.uc_yn == 'Yes', 1, CPS_dataset.uc_yn)
CPS_dataset.uc_yn = CPS_dataset.uc_yn.astype(int)

#Totals for UI comp
CPS_dataset.uc_val = CPS_dataset.uc_val.astype(int)

#Unemployed status
CPS_dataset.a_explf = np.where(CPS_dataset.a_explf == 'Employed', 0, CPS_dataset.a_explf)
CPS_dataset.a_explf = np.where(CPS_dataset.a_explf == 'Not in experienced labor force', 0, CPS_dataset.a_explf)
CPS_dataset.a_explf = np.where(CPS_dataset.a_explf == 'Unemployed', 1, CPS_dataset.a_explf)
CPS_dataset.a_explf = CPS_dataset.a_explf.astype(int)

#Actively looking for work?
CPS_dataset['lkweeks'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

CPS_dataset.lkweeks = CPS_dataset.lkweeks.replace('01 weeks', 1)
CPS_dataset.lkweeks = CPS_dataset.lkweeks.replace('51 weeks', 51)
CPS_dataset.lkweeks = CPS_dataset.lkweeks.astype(int)
CPS_dataset.lkweeks = np.where((CPS_dataset.lkweeks > 0), 1, 0)

# Unemployed due to own fault?
CPS_dataset.pruntype = np.where((CPS_dataset.pruntype == 'Job loser/on layoff') | (CPS_dataset.pruntype == 'Other job loser'), 1, 0)

# Did you work the full year or are you a nonworker?
CPS_dataset.weuemp = np.where((CPS_dataset.weuemp == 'Full year worker') | (CPS_dataset.weuemp == 'Nonworker'), 0, 1)

# Were they looking for work for a while??
CPS_dataset.lkstrch = np.where((CPS_dataset.lkstrch == 'Yes, 1 stretch'), 1, 0)

#Earned income
p_earned = CPS_dataset.wsal_val.astype(int) + CPS_dataset.semp_val.astype(int) + CPS_dataset.frse_val.astype(int) #individual earned income
CPS_dataset['p_earned'] = p_earned

#Unearned income / without uemployment compensation
p_unearned = CPS_dataset.ss_val.astype(int) + CPS_dataset.rtm_val.astype(int) + CPS_dataset.div_val.astype(int) + CPS_dataset.oi_val.astype(int) + CPS_dataset.int_val.astype(int) #individual unearned income
CPS_dataset['p_unearned'] = p_unearned

#f_mv_fs
CPS_dataset.f_mv_fs = pd.to_numeric(CPS_dataset.f_mv_fs)

#elderly (age of at least 62) 

CPS_dataset.a_age = np.where(CPS_dataset.a_age == "80-84 years of age",
                             random.randrange(80, 84),
                             CPS_dataset.a_age)
CPS_dataset.a_age = np.where(CPS_dataset.a_age == "85+ years of age",
                             random.randrange(85, 95),
                             CPS_dataset.a_age)
CPS_dataset  = CPS_dataset.loc[:,~CPS_dataset.columns.duplicated()]
CPS_dataset.a_age = pd.to_numeric(CPS_dataset.a_age)
CPS_dataset['elderly'] = 0
CPS_dataset.elderly = np.where(CPS_dataset.a_age > 61, 1, CPS_dataset.elderly)

#disabled - must be able to work to receive UI
CPS_dataset['disability'] = np.zeros(len(CPS_dataset))
CPS_dataset.disability = np.where(CPS_dataset.pedisdrs == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisear == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pediseye == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisout == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisphy == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisrem == 'Yes', 1, CPS_dataset.disability)

# Corresponds to random forest probabilities that they received UI compensation
CPS_dataset['RfYes'] = Rf_probs[:, 1]
# These probabilities predicted those who were actually receiving UI in the test set with 88.9% accuracy

CPS_dataset['indicator'] = CPS_dataset.uc_yn

# CPS total benefits and Administrative total benefits
state_benefit = {}
state_recipients = {}

for fip in Admin_totals.Fips:
    this_state = (CPS_dataset.gestfips == fip)
    CPS_totalb = (CPS_dataset.uc_val[CPS_dataset.indicator == 1] * CPS_dataset.marsupwt[CPS_dataset.indicator == 1])[this_state].sum() # The CPS subsidy amount is montly 
    admin_totalb =  Admin_totals['tot_UI_outlays'][Admin_totals.Fips == fip].values # to match montly
    CPS_totaln = CPS_dataset.marsupwt[this_state & CPS_dataset.indicator==1].sum() 
    admin_totaln =  Admin_totals['tot_UI_recipients'][Admin_totals.Fips == fip].values
    temp = [Admin_totals.state[Admin_totals['Fips'] == fip].values[0], CPS_totalb, admin_totalb[0], CPS_totaln, admin_totaln[0]]
    state_benefit[fip] = temp

pre_augment_benefit = DataFrame(state_benefit).transpose()
pre_augment_benefit.columns = ['State', 'CPS total benefits (annually)','Admin total benefits (annually)',
                               'CPS total recipients','Admin total recipients']

pre_augment_benefit['Admin total benefits (annually)'] = pre_augment_benefit['Admin total benefits (annually)'].astype(int)
pre_augment_benefit['CPS total benefits (annually)'] = pre_augment_benefit['CPS total benefits (annually)'].astype(int)
pre_augment_benefit['CPS total recipients'] = pre_augment_benefit['CPS total recipients'].astype(int)
pre_augment_benefit.to_csv('admin_cps_totals_before.csv')


# caculate difference of UI stats and CPS aggregates on recipients number
# by state
diff = {'Fips':[],'Difference in Population':[],'Mean Benefit':[],'CPS Population':[],'UI Population':[]}
diff['Fips'] = Admin_totals.Fips
current = (CPS_dataset.indicator==1)
for FIPS in Admin_totals.Fips:
        this_state = (CPS_dataset.gestfips == FIPS)
        current_tots = CPS_dataset.marsupwt[current&this_state].sum()
        valid_num = CPS_dataset.marsupwt[current&this_state].sum() + 0.0000001
        current_mean = ((CPS_dataset.uc_val * CPS_dataset.marsupwt)[current&this_state].sum())/valid_num
        diff['CPS Population'].append(current_tots)
        diff['UI Population'].append(float(Admin_totals["tot_UI_recipients"][Admin_totals.Fips == FIPS]))
        diff['Difference in Population'].append(float(Admin_totals["tot_UI_recipients"][Admin_totals.Fips == FIPS])- current_tots)
        diff['Mean Benefit'].append(current_mean)



d = DataFrame(diff)
d = d[['Fips', 'Mean Benefit', 'Difference in Population', 'CPS Population', 'UI Population']]
d.to_csv('recipients_diff.csv')

'''Using Random Forest probabilities'''

CPS_dataset['impute'] = np.zeros(len(CPS_dataset))
CPS_dataset['UI_impute'] = np.zeros(len(CPS_dataset))
probs = CPS_dataset['RfYes']

CPS_dataset['impute'] = np.zeros(len(CPS_dataset))
CPS_dataset['UI_impute'] = np.zeros(len(CPS_dataset))

non_current = (CPS_dataset.indicator==0)
current = (CPS_dataset.indicator==1)
random.seed()

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
            CPS_dataset.loc[pool.index[pool['impute']==1], 'UI_impute'] = Admin_totals['Avg_benefit'][Admin_totals['Fips'] ==FIPS].values[0]
            # print ('Method1: regression gives', 
            #     CPS_dataset.marsupwt[(CPS_dataset.impute==1)&this_state].sum()) 



#Adjustment ratio
results = {}

imputed = (CPS_dataset.impute == 1)
has_val = (CPS_dataset.indicator == 1)
no_val = (CPS_dataset.uc_val == 0)

for FIPS in Admin_totals.Fips:
    this_state = (CPS_dataset.gestfips == FIPS)
    current_total = (CPS_dataset.uc_val * CPS_dataset.marsupwt)[this_state].sum() 
    imputed_total = (CPS_dataset.UI_impute * CPS_dataset.marsupwt)[this_state & imputed].sum()
    on_file = current_total + imputed_total
    admin_total = Admin_totals.tot_UI_outlays[Admin_totals.Fips == FIPS]
    adjust_ratio = admin_total / on_file
    this_state_num = [Admin_totals['state'][Admin_totals.Fips == FIPS].values[0], on_file, admin_total.values[0], adjust_ratio.values[0]]
    results[FIPS] = this_state_num
    CPS_dataset.UI_impute = np.where(has_val & this_state, CPS_dataset.uc_val * adjust_ratio.values, CPS_dataset.UI_impute)
    CPS_dataset.UI_impute = np.where(no_val & this_state, CPS_dataset.UI_impute * adjust_ratio.values, CPS_dataset.UI_impute)
CPS_dataset["UI_participation"] = np.zeros(len(CPS_dataset))
CPS_dataset["UI_participation"] = np.where(CPS_dataset.impute == 1, 2, 0) #Augmented
CPS_dataset['UI_participation'] = np.where(has_val, 1, CPS_dataset.UI_participation)

r = DataFrame(results).transpose()
r.columns = ['State', 'Imputed', 'Admin', 'adjust ratio']
r['Imputed'] = r['Imputed'].astype(int)
r['adjust ratio'] *= 10000
r['adjust ratio'] = r['adjust ratio'].astype(int)
r['adjust ratio'] /= 10000
r.to_csv('amount_rf.csv', index=False)


CPS_dataset.to_csv('UI_imputation_rf.csv', 
                      columns=['peridnum','UI_participation', 'UI_impute'])


## Checking post-adjustment totals to see if they match admin totals
CPS_dataset.UI_participation = np.where(CPS_dataset.UI_participation == 2 , 1, CPS_dataset.UI_participation)
CPS_dataset['after_totals_reciepts'] = (CPS_dataset.UI_participation * CPS_dataset.marsupwt)
CPS_dataset['after_totals_outlays'] = (CPS_dataset.UI_impute * CPS_dataset.marsupwt)
total_outlays = CPS_dataset.groupby(['gestfips'])['after_totals_outlays'].sum().astype(int).reset_index(drop = True) 
total_recipients = CPS_dataset.groupby(['gestfips'])['after_totals_reciepts'].sum().astype(int).reset_index(drop = True)

df = pd.DataFrame()
df['State'] = Admin_totals.state
df['post augment CPS total benefits (annual)'] = total_outlays
df['post augment CPS total recipients'] = total_recipients
df['Admin total benefits (annual)'] = Admin_totals['tot_UI_outlays']
df['Admin total recipients'] = Admin_totals['tot_UI_recipients']
df.to_csv('post_augment_adminCPS_totals_rf.csv')


