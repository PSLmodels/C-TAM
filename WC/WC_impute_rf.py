'''                                 
About: 
    This script imputes Worker's Compensation (WC) participation using random forests. 
    Please refer to the documentation in the same folder 
    for more details on methodology and assumptions. The output this script is a personal level dataset that contains CPS
    individual participation indicator (WC participationc, 0 - not a recipient, 
    1 - current recipient on file, 2 - imputed recipient), and benefit amount.

Input: 
    2015 CPS (asec2015_pubuse.csv), number of recipients and their benefits amount by state in 2014 (Administrative.csv),
    rf_probs.csv from Rf_probs.py.

Output: 
    WC_imputation_rf.csv 
 
Additional Source links: 
    SSA 2014 annual statistical report at https://www.ssa.gov/policy/docs/statcomps/supplement/2016/workerscomp.html
    gives the total WC benefits paid out in 2014. Scroll to the bottom under "Program Highlights", to see
    benefit totals. Link at bottom to NASI report.
'''
import pandas as pd
from pandas import DataFrame
import numpy as np
import random
import statsmodels.discrete.discrete_model as sm
import matplotlib.pyplot as plt

# The administrative totals for 2014 come from https://www.ssa.gov/policy/docs/statcomps/supplement/2016/workerscomp.html
claims_per_hundredthousand = np.loadtxt('claims_projected').astype(int)
total_covered = 129600000
total_benefits = 62300000000.
total_recipients = ((total_covered / 100000.) * claims_per_hundredthousand).astype(int)
Avg_benefit = total_benefits / total_recipients
Avg_benefit *= 10000
Avg_benefit = int(Avg_benefit)
Avg_benefit /= 10000.

# Variables we use in CPS:
CPS_dataset = pd.read_csv('../asec2015_pubuse.csv')
columns_to_keep = ['a_whyabs','a_mjind','chsp_val','dis_hp','dis_cs','finc_dis', 'fdisval','dis_yn','wc_yn', 'wc_val','a_pfrel', 'rsnnotw', 'hfoodsp','ch_mc','caid', 
'cov_hi', 'fwsval', 'mcaid','hrwicyn','wicyn','oi_off','paw_yn','paw_typ','paw_val', 
'peioocc', 'a_wksch', 'wemind', 'prcitshp', 'gestfips','marsupwt','a_age','wsal_val','semp_val','frse_val',
                  'ss_val','rtm_val','div_val','oi_off','oi_val','uc_yn','uc_val', 'int_yn', 'int_val','pedisdrs', 'pedisear', 'pediseye', 
                    'pedisout', 'pedisphy', 'pedisrem','a_sex','peridnum']
CPS_dataset = CPS_dataset[columns_to_keep]
CPS_dataset = CPS_dataset.replace({'None or not in universe' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'Not in universe' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'NIU' : 0}, regex = True)
CPS_dataset = CPS_dataset.replace({'None' : 0}, regex = True)
# CPS_dataset.to_csv('CPS_WC.csv', index=False)
# CPS_dataset = pd.read_csv('CPS_WC.csv')
 
CPS_dataset.a_sex = np.where(CPS_dataset.a_sex == 'Male', 1, 0)

CPS_dataset.dis_hp = np.where(CPS_dataset.dis_hp == 'No', 0, CPS_dataset.dis_hp)
CPS_dataset.dis_hp = np.where(CPS_dataset.dis_hp == 'Yes', 1, CPS_dataset.dis_hp)
CPS_dataset.dis_hp = CPS_dataset.dis_hp.astype(int)

CPS_dataset.finc_dis = np.where(CPS_dataset.finc_dis == 'No', 0, CPS_dataset.finc_dis)
CPS_dataset.finc_dis = np.where(CPS_dataset.finc_dis == 'Yes', 1, CPS_dataset.finc_dis)
CPS_dataset.finc_dis = CPS_dataset.finc_dis.astype(int)

CPS_dataset.dis_cs = np.where(CPS_dataset.dis_cs == 'No', 0, CPS_dataset.dis_cs)
CPS_dataset.dis_cs = np.where(CPS_dataset.dis_cs == 'Yes', 1, CPS_dataset.dis_cs)
CPS_dataset.dis_cs = CPS_dataset.dis_cs.astype(int)

#recipient or not of WIC
CPS_dataset.wc_yn = np.where(CPS_dataset.wc_yn == 'No', 0, CPS_dataset.wc_yn)
CPS_dataset.wc_yn = np.where(CPS_dataset.wc_yn == 'Yes', 1, CPS_dataset.wc_yn)
CPS_dataset.wc_yn = CPS_dataset.wc_yn.astype(int)


CPS_dataset.fdisval = CPS_dataset.fdisval.astype(int)
CPS_dataset.wc_val = CPS_dataset.wc_val.astype(int)


CPS_dataset.a_age = np.where(CPS_dataset.a_age == "80-84 years of age",
                             random.randrange(80, 84),
                             CPS_dataset.a_age)
CPS_dataset.a_age = np.where(CPS_dataset.a_age == "85+ years of age",
                             random.randrange(85, 95),
                             CPS_dataset.a_age)
CPS_dataset.a_age = pd.to_numeric(CPS_dataset.a_age)

CPS_dataset.cov_hi = np.where((CPS_dataset.cov_hi == 'No'), 0, CPS_dataset.cov_hi)
CPS_dataset.cov_hi = np.where(CPS_dataset.cov_hi == 'Yes', 1, CPS_dataset.cov_hi)
CPS_dataset.cov_hi = CPS_dataset.cov_hi.astype(int)


CPS_dataset['indicator'] = CPS_dataset.wc_yn

# CPS total benefits and Administrative total benefits

CPS_totalb = (CPS_dataset.wc_val[CPS_dataset.indicator == 1] * CPS_dataset.marsupwt[CPS_dataset.indicator == 1]).sum() # The CPS subsidy amount is montly 
admin_totalb =  total_benefits
CPS_totaln = CPS_dataset.marsupwt[CPS_dataset.indicator==1].sum() 
admin_totaln =  total_recipients
temp = ['USA', CPS_totalb, admin_totalb, CPS_totaln, admin_totaln]
US_benefit = temp

pre_augment_benefit = DataFrame(US_benefit).transpose()
pre_augment_benefit.columns = ['Country', 'CPS benefits (annually)','Admin benefits (annually)',
                               'CPS total recipients','Admin total recipients']

pre_augment_benefit['Admin benefits (annually)'] = pre_augment_benefit['Admin benefits (annually)'].astype(int)
pre_augment_benefit['CPS benefits (annually)'] = pre_augment_benefit['CPS benefits (annually)'].astype(int)
pre_augment_benefit['CPS total recipients'] = pre_augment_benefit['CPS total recipients'].astype(int)
pre_augment_benefit.to_csv('admin_cps_totals_before.csv')

# caculate difference of WC stats and CPS aggregates on recipients number
# by state
diff = {'Fips':[],'Difference in Population':[],'Mean Benefit':[],'CPS Population':[],'WIC Population':[]}
diff['Fips'] = "USA"
current = (CPS_dataset.indicator==1)
current_tots = CPS_dataset.marsupwt[current].sum()
valid_num = CPS_dataset.marsupwt[current].sum() + 0.0000001
current_mean = ((CPS_dataset.wc_val * CPS_dataset.marsupwt)[current].sum())/valid_num
diff['CPS Population'].append(current_tots)
diff['WIC Population'].append(float(total_recipients))
diff['Difference in Population'].append(float(total_recipients)- current_tots)
diff['Mean Benefit'].append(current_mean)



d = DataFrame(diff)
d = d[['Fips', 'Mean Benefit', 'Difference in Population', 'CPS Population', 'WIC Population']]
d.to_csv('recipients_diff.csv')

# Random Forest probabilities of receiving WC compensation. 'rf_probs.csv' obtained from rf_probs.ipynb.
Rf_probs = np.loadtxt('rf_probs.csv')

# RF probabilities predicted with 85% accuracy those receiving WC in the test set, and with 99% accuracy 
# everyone including those who didn't receive WC
CPS_dataset['RfYes'] = Rf_probs[:, 1]

probs = CPS_dataset['RfYes']

CPS_dataset['impute'] = np.zeros(len(CPS_dataset))
CPS_dataset['WC_impute'] = np.zeros(len(CPS_dataset))

non_current = (CPS_dataset.indicator==0)
current = (CPS_dataset.indicator==1)
random.seed()

if d['Difference in Population'][d['Fips'] == 'USA'].values[0] < 0:
    pass
else:
    not_imputed = (CPS_dataset.impute==0)
    pool_index = CPS_dataset[not_imputed&non_current].index
    pool = DataFrame({'weight': CPS_dataset.marsupwt[pool_index], 'prob': probs[pool_index]},
                    index=pool_index)
    pool = pool.sort_values(by = 'prob', ascending=False)
    pool['cumsum_weight'] = pool['weight'].cumsum()
    pool['distance'] = abs(pool.cumsum_weight-d['Difference in Population'][d['Fips'] == 'USA'].values)
    min_index = pool.sort_values(by='distance')[:1].index
    min_weight = int(pool.loc[min_index].cumsum_weight)
    pool['impute'] = np.where(pool.cumsum_weight<=min_weight+10 , 1, 0)
    CPS_dataset.loc[pool.index[pool['impute']==1], 'impute'] = 1
    CPS_dataset.loc[pool.index[pool['impute']==1], 'WC_impute'] = Avg_benefit


#Adjustment ratio
results = {}
imputed = (CPS_dataset.impute == 1)
has_val = (CPS_dataset.indicator == 1)
no_val = (CPS_dataset.wc_val == 0)

current_total = (CPS_dataset.wc_val * CPS_dataset.marsupwt).sum() 
imputed_total = (CPS_dataset.WC_impute * CPS_dataset.marsupwt)[imputed].sum()
on_file = current_total + imputed_total
admin_total = total_benefits
adjust_ratio = admin_total / on_file
this_state_num = ['USA', on_file, admin_total, adjust_ratio]
results['USA'] = this_state_num
CPS_dataset.WC_impute = np.where(has_val, CPS_dataset.wc_val * adjust_ratio, CPS_dataset.WC_impute)
CPS_dataset.WC_impute = np.where(no_val, CPS_dataset.WC_impute * adjust_ratio, CPS_dataset.WC_impute)
CPS_dataset["WC_participation"] = np.zeros(len(CPS_dataset))
CPS_dataset["WC_participation"] = np.where(CPS_dataset.impute == 1, 2, 0) #Augmented
CPS_dataset['WC_participation'] = np.where(has_val, 1, CPS_dataset.WC_participation)
r = DataFrame(results).transpose()
r.columns = ['Country', 'Imputed', 'Admin', 'adjust ratio']
r['Imputed'] = r['Imputed'].astype(int)
r['adjust ratio'] *= 10000
r['adjust ratio'] = r['adjust ratio'].astype(int)
r['adjust ratio'] /= 10000
r.to_csv('amount.csv', index=False)

CPS_dataset.to_csv('WC_Imputation_rf.csv', 
                   columns=['peridnum','WC_participation', 'WC_impute'])


## Checking post-adjustment totals to see if they match admin totals
CPS_dataset.WC_participation = np.where(CPS_dataset.WC_participation == 2 , 1, CPS_dataset.WC_participation)
CPS_dataset['after_totals_reciepts'] = (CPS_dataset.WC_participation * CPS_dataset.marsupwt)
CPS_dataset['after_totals_outlays'] = (CPS_dataset.WC_impute * CPS_dataset.marsupwt)
total_outlays = int(CPS_dataset['after_totals_outlays'].sum())
total_recipients_CPS = int(CPS_dataset['after_totals_reciepts'].sum())

df = pd.DataFrame()
df['Country'] = ['USA']
df['post augment CPS total benefits (annual)'] = [total_outlays]
df['post augment CPS total recipients'] = [total_recipients_CPS]
df['Admin total benefits (annual)'] = [total_benefits]
df['Admin total recipients'] = [total_recipients]
df.to_csv('post_augment_adminCPS_totals_rf.csv')
