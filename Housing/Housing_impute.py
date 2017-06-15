
# This script imputes Supplemental Nutrition Assistance Program (SNAP) recipients dollar benefit amount to match the aggregates with United States Department of Agriculture (USDA) statistics for SNAP. In this current version, we used 2014 CPS data and USDA FY2014 annual reports on SNAP. Please refer to the documentation in the same folder for more details on methodology and assumptions. The output this script is a personal level dataset that contains CPS household level sequence (h_seq), individual participation indicator (snap_participation, 0 - not a recipient, 1 - current recipient on file, 2 - imputed recipient), and benefit amount.
# 
# Input: 2014 CPS (cpsmar2014t.csv), number of recipients and their benefits amount by state in 2014 (Administrative.csv)
# 
# Output: SNAP_Imputation.csv
# 
# Additional Source links: http://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap (zipfile for FY69 through FY16)
'''Check HUD picture for better recipient Totals'''

import pandas as pd
from pandas import DataFrame
import numpy as np
import random
import statsmodels.discrete.discrete_model as sm
import matplotlib.pyplot as plt


# Administrative level data
Admin_totals =  pd.read_csv('Admin_totals.csv')
Admin_totals['Avg_Voucher'] = Admin_totals['voucher_amount'] / Admin_totals['voucher_recipients']
Admin_totals['Avg_Voucher'] *= 10000
Admin_totals['Avg_Voucher'] = Admin_totals['Avg_Voucher'].astype(int)
Admin_totals['Avg_Voucher'] /= 10000
Admin_totals[['state','Avg_Voucher']].to_csv('avg.csv')

# 2014 Income limits for Housing vouchers, w/ 50 and 30 percent median income cutoffs
Income_limits = pd.read_csv('Income_limits.csv')
def income_lim_indicator30(income, family_size, gestfips):
    if income <= Income_limits['Lim30_14p'+ str(int(family_size))][(Income_limits['Fips'] == gestfips)].values:
        return 1
    else:
        return 0

def income_lim_indicator50(income, family_size, gestfips):
    if (income <= Income_limits['lim50_14p'+ str(int(family_size))][(Income_limits['Fips'] == gestfips)].values) & (income > Income_limits['Lim30_14p'+ str(int(family_size))][(Income_limits['Fips'] == gestfips)].values):
        return 1
    else:
        return 0


# Variables we use in CPS:
CPS_dataset = pd.read_csv('cpsmar2014t.csv')

columns_to_keep = ['fhoussub', 'fownu18', 'fpersons','fspouidx', 'prcitshp', 'gestfips','marsupwt','a_age','wsal_val','semp_val','frse_val',
                  'ss_val','rtm_val','div_val','oi_off','oi_val','uc_yn','uc_val', 'int_yn', 'int_val','pedisdrs', 'pedisear', 'pediseye', 
                    'pedisout', 'pedisphy', 'pedisrem','a_sex','peridnum','h_seq','fh_seq', 'ffpos', 'fsup_wgt',
                        'hlorent', 'hpublic', 'hsup_wgt']
CPS_dataset = CPS_dataset[columns_to_keep]
CPS_dataset.to_csv('CPS_Housing.csv', columns= columns_to_keep, index=False)

# CPS_dataset = pd.read_csv('CPS_Housing.csv')


#recipient or not of Housing Voucher Assistance program section 8
CPS_dataset.hlorent = np.where(CPS_dataset.hlorent == 'Not in universe',0,CPS_dataset.hlorent)
CPS_dataset.hlorent = np.where(CPS_dataset.hlorent == 'No', 0, CPS_dataset.hlorent)
CPS_dataset.hlorent = np.where(CPS_dataset.hlorent == 'Yes', 1, CPS_dataset.hlorent)
CPS_dataset.hlorent = CPS_dataset.hlorent.astype(int)

#Totals for household subsidies
CPS_dataset.fhoussub = np.where(CPS_dataset.fhoussub == 'None',0,CPS_dataset.fhoussub)
CPS_dataset.fhoussub = CPS_dataset.fhoussub.astype(int)

#Removing those who receive hlorent, but who have no subsequent fhoussub value to describe it
CPS_dataset.hlorent = np.where((CPS_dataset.hlorent == 1) & (CPS_dataset.fhoussub == 0),
     0, CPS_dataset.hlorent)



#Prepare household level data
family_Houssub = CPS_dataset.groupby(['fh_seq', 'ffpos'])['hlorent'].mean()
family_Houssub_val = CPS_dataset.groupby(['fh_seq', 'ffpos'])['fhoussub'].mean()

family_size = CPS_dataset.groupby(['fh_seq', 'ffpos'])['fpersons'].mean()


#Earned income
wage = pd.to_numeric(np.where(CPS_dataset.wsal_val!= 'None or not in universe', CPS_dataset.wsal_val, 0))
self_employed1 = pd.to_numeric(np.where(CPS_dataset.semp_val!= 'None or not in universe', CPS_dataset.semp_val, 0))
self_employed2 = pd.to_numeric(np.where(CPS_dataset.frse_val!= 'None or not in universe', CPS_dataset.frse_val, 0))
p_earned = wage + self_employed1 + self_employed2 #individual earned income
CPS_dataset['p_earned'] = p_earned


#Unearned income
ss = pd.to_numeric(np.where(CPS_dataset.ss_val != 'None or not in universe', CPS_dataset.ss_val, 0))
pension = pd.to_numeric(np.where(CPS_dataset.rtm_val != 'None or not in universe', CPS_dataset.rtm_val, 0))
dividends = pd.to_numeric(np.where(CPS_dataset.div_val != 'None or not in universe', CPS_dataset.div_val, 0))
disability = pd.to_numeric(np.where(CPS_dataset.oi_off =='State disability payments', CPS_dataset.oi_val, 0))
unemploy = pd.to_numeric(np.where(CPS_dataset.uc_yn =='Yes', CPS_dataset.uc_val, 0))
interest = pd.to_numeric(np.where(CPS_dataset.int_yn =='Yes', CPS_dataset.int_val, 0))
p_unearned = ss + pension + disability + unemploy + interest + dividends #individual unearned income
CPS_dataset['p_unearned'] = p_unearned


# In[8]:

#Net Income
CPS_dataset['family_net_income'] = p_earned + p_unearned
family_net_income = CPS_dataset.groupby(['fh_seq', 'ffpos'])['family_net_income'].sum()

f_House_yn = DataFrame(family_Houssub)
f_House_yn['family_net'] = family_net_income
f_House_yn['family_size'] = family_size



f_House_yn.columns = ['indicator', 'family_net','family_size'] #indicator is whether or not housing received, a dummy variable
f_House_yn['fVouch_val'] = family_Houssub_val

#Citizenship
CPS_dataset['citizen'] = 1
CPS_dataset.citizen = np.where(CPS_dataset.prcitshp != 'Foreign born, not a citizen of', 0, CPS_dataset.citizen)
family_citizenship = CPS_dataset.groupby(['fh_seq', 'ffpos'])['citizen'].sum()
family_citizenship = np.where(family_citizenship > 0 , 1, 0)
f_House_yn['citizenship'] = family_citizenship


#elderly (age of at least 62) (check reg if categorical or binary is better when summing groupby)

CPS_dataset.a_age = np.where(CPS_dataset.a_age == "80-84 years of age",
                             random.randrange(80, 84),
                             CPS_dataset.a_age)
CPS_dataset.a_age = np.where(CPS_dataset.a_age == "85+ years of age",
                             random.randrange(85, 95),
                             CPS_dataset.a_age)
CPS_dataset.a_age = pd.to_numeric(CPS_dataset.a_age)
CPS_dataset['elderly'] = 0
CPS_dataset.elderly = np.where(CPS_dataset.a_age > 61, 1, CPS_dataset.elderly)
f_elderly = CPS_dataset.groupby(['fh_seq', 'ffpos'])['elderly'].sum()
f_House_yn['f_elderly'] = f_elderly

#disabled (check reg if categorical or binary is better after the sum)

CPS_dataset['disability'] = np.zeros(len(CPS_dataset))
CPS_dataset.disability = np.where(CPS_dataset.pedisdrs == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisear == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pediseye == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisout == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisphy == 'Yes', 1, CPS_dataset.disability)
CPS_dataset.disability = np.where(CPS_dataset.pedisrem == 'Yes', 1, CPS_dataset.disability)

f_disability = CPS_dataset.groupby(['fh_seq', 'ffpos'])['disability'].sum()
f_House_yn['f_disability'] = f_disability


# #State residency of family and weights
family_marsupwt = CPS_dataset.groupby(['fh_seq', 'ffpos'])['fsup_wgt'].mean()
house_marsupwt = CPS_dataset.groupby(['fh_seq', 'ffpos'])['hsup_wgt'].mean()
f_House_yn['f_marsupwt'] = family_marsupwt
f_House_yn['h_marsupwt'] = house_marsupwt

family_gestfips = CPS_dataset.groupby(['fh_seq', 'ffpos'])['gestfips'].mean()
f_House_yn['f_gestfips'] = family_gestfips

# Under 30 percent of median income
f_House_yn['under_30_inc'] = f_House_yn.apply(lambda x: income_lim_indicator30(x['family_net'], x['family_size'],
    x['f_gestfips']), axis=1)

# Under 50 percent of median income
f_House_yn['under_50_inc'] = f_House_yn.apply(lambda x: income_lim_indicator50(x['family_net'], x['family_size'],
    x['f_gestfips']), axis=1)
f_House_yn.to_csv('use_df_hlo.csv')
# f_House_yn = pd.read_csv('use_df_hlo.csv')

#Regression
dta = f_House_yn
dta['intercept'] = np.ones(len(dta))
model = sm.Logit(endog = dta.indicator, exog = dta[['family_size','under_30_inc','under_50_inc','f_disability','f_elderly','citizenship']]).fit()
probs = model.predict()
f_House_yn['probs'] = probs

# CPS total benefits and Administrative total benefits
state_benefit = {}
state_recipients = {}

for fip in Admin_totals.Fips:
    this_state = (f_House_yn.f_gestfips == fip)
    CPS_totalb = (f_House_yn.fVouch_val[f_House_yn.indicator == 1] * f_House_yn.f_marsupwt)[this_state].sum() # The CPS subsidy amount is montly 
    admin_totalb =  Admin_totals['voucher_amount'][Admin_totals.Fips == fip].values / 12 # to match montly
    CPS_totaln = f_House_yn.h_marsupwt[this_state & f_House_yn.indicator==1].sum() 
    admin_totaln =  Admin_totals['voucher_recipients'][Admin_totals.Fips == fip].values
    temp = [Admin_totals.state[Admin_totals['Fips'] == fip].values[0], CPS_totalb, admin_totalb[0], CPS_totaln, admin_totaln[0]]
    state_benefit[fip] = temp

pre_augment_benefit = DataFrame(state_benefit).transpose()
pre_augment_benefit.columns = ['State', 'CPS total benefits (annually)','Admin total benefits (annually)',
                               'CPS total family recipients','Admin total family recipients']

pre_augment_benefit['Admin total benefits (annually)'] *= 12
pre_augment_benefit['Admin total benefits (annually)'] = pre_augment_benefit['Admin total benefits (annually)'].astype(int)
pre_augment_benefit['CPS total benefits (annually)'] *= 12
pre_augment_benefit['CPS total benefits (annually)'] = pre_augment_benefit['CPS total benefits (annually)'].astype(int)
pre_augment_benefit['CPS total family recipients'] = pre_augment_benefit['CPS total family recipients'].astype(int)
pre_augment_benefit.to_csv('admin_cps_totals_before.csv')


# caculate difference of Housing stats and CPS aggregates on recipients number
# by state
diff = {'Fips':[],'Difference in Population':[],'Mean Benefit':[],'CPS Population':[],'Housing Population':[]}
diff['Fips'] = Admin_totals.Fips
current = (f_House_yn.indicator==1)
for FIPS in Admin_totals.Fips:
        this_state = (f_House_yn.f_gestfips == FIPS)
        current_tots = f_House_yn.f_marsupwt[current&this_state].sum()
        valid_num = f_House_yn.f_marsupwt[current&this_state].sum() + 0.0000001
        current_mean = ((f_House_yn.fVouch_val * f_House_yn.f_marsupwt)[current&this_state].sum())/valid_num
        diff['CPS Population'].append(current_tots)
        diff['Housing Population'].append(float(Admin_totals["voucher_recipients"][Admin_totals.Fips == FIPS]))
        diff['Difference in Population'].append(float(Admin_totals["voucher_recipients"][Admin_totals.Fips == FIPS])- current_tots)
        diff['Mean Benefit'].append(current_mean)



d = DataFrame(diff)
d = d[['Fips', 'Mean Benefit', 'Difference in Population', 'CPS Population', 'Housing Population']]
d.to_csv('recipients_diff.csv')



f_House_yn['impute'] = np.zeros(len(f_House_yn))
f_House_yn['housing_impute'] = np.zeros(len(f_House_yn))

non_current = (f_House_yn.indicator==0)
current = (f_House_yn.indicator==1)
random.seed()

for FIPS in Admin_totals.Fips:
    
        # print ('we need to impute', d['Difference in Population'][d['Fips'] == FIPS].values[0], 'for state', FIPS)
        
        if d['Difference in Population'][d['Fips'] == FIPS].values[0] < 0:
            continue
        else:
            this_state = (f_House_yn.f_gestfips==FIPS)
            not_imputed = (f_House_yn.impute==0)
            pool_index = f_House_yn[this_state&not_imputed&non_current].index
            pool = DataFrame({'weight': f_House_yn.f_marsupwt[pool_index], 'prob': probs[pool_index]},
                            index=pool_index)
            pool = pool.sort(columns = 'prob', ascending=False)
            pool['cumsum_weight'] = pool['weight'].cumsum()
            pool['distance'] = abs(pool.cumsum_weight-d['Difference in Population'][d['Fips'] == FIPS].values)
            min_index = pool.sort(columns='distance')[:1].index
            min_weight = int(pool.loc[min_index].cumsum_weight)
            pool['impute'] = np.where(pool.cumsum_weight<=min_weight+10 , 1, 0)
            f_House_yn.impute[pool.index[pool['impute']==1]] = 1
            f_House_yn.housing_impute[pool.index[pool['impute']==1]] = Admin_totals['Avg_Voucher'][Admin_totals['Fips'] ==FIPS].values[0] / 12.
        # print ('Method1: regression gives', 
        #         f_House_yn.f_marsupwt[(f_House_yn.impute==1)&this_state].sum()) 

#Adjustment ratio
results = {}

imputed = (f_House_yn.impute == 1)
has_val = (f_House_yn.indicator == 1)
no_val = (f_House_yn.fVouch_val == 0)

for FIPS in Admin_totals.Fips:
    this_state = (f_House_yn.f_gestfips == FIPS)
    
    current_total = (f_House_yn.fVouch_val * f_House_yn.f_marsupwt)[this_state & current].sum() 
    imputed_total = (f_House_yn.housing_impute * f_House_yn.f_marsupwt)[this_state & imputed].sum()
    on_file = current_total + imputed_total
    admin_total = Admin_totals.voucher_amount[Admin_totals.Fips == FIPS] / 12.
    adjust_ratio = admin_total / on_file
    this_state_num = [Admin_totals['state'][Admin_totals.Fips == FIPS].values[0], on_file, admin_total.values[0], adjust_ratio.values[0]]
    results[FIPS] = this_state_num

    f_House_yn.housing_impute = np.where(has_val & this_state, f_House_yn.fVouch_val * adjust_ratio.values, f_House_yn.housing_impute)
    f_House_yn.housing_impute = np.where(no_val & this_state, f_House_yn.housing_impute * adjust_ratio.values, f_House_yn.housing_impute)
f_House_yn["housing_participation"] = np.zeros(len(f_House_yn))
f_House_yn["housing_participation"] = np.where(f_House_yn.impute == 1, 2, 0) #Augmented
f_House_yn["housing_participation"] = np.where(has_val, 1, f_House_yn.housing_participation) #CPS


r = DataFrame(results).transpose()
r.columns = ['State', 'Imputed', 'Admin', 'adjust ratio']
r['Imputed'] = r['Imputed'].astype(int)
r['adjust ratio'] *= 10000
r['adjust ratio'] = r['adjust ratio'].astype(int)
r['adjust ratio'] /= 10000
r.to_csv('amount.csv', index=False)

f_House_yn.to_csv('Housing_Imputation.csv', 
                   columns=['housing_participation', 'housing_impute'])


## Checking post-adjustment totals to see if they match admin totals
f_House_yn.housing_participation = np.where(f_House_yn.housing_participation == 2 , 1, f_House_yn.housing_participation)
f_House_yn['after_totals_reciepts'] = (f_House_yn.housing_participation * f_House_yn.f_marsupwt)
f_House_yn['after_totals_voucher'] = (f_House_yn.housing_impute * f_House_yn.f_marsupwt)
total_voucherprice = f_House_yn.groupby(['f_gestfips'])['after_totals_voucher'].sum().astype(int).reset_index(drop = True) * 12
total_recipients = f_House_yn.groupby(['f_gestfips'])['after_totals_reciepts'].sum().astype(int).reset_index(drop = True)

df = pd.DataFrame()
df['post augment CPS total benefits (annual)'] = total_voucherprice
df['post augment CPS total recipients'] = total_recipients
df['Admin total benefits (annual)'] = Admin_totals['voucher_amount']
df['Admin total recipients'] = Admin_totals['voucher_recipients']
df.to_csv('post_augment_adminCPS_totals.csv')


