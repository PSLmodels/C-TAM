import pandas as pd
from pandas import DataFrame
import numpy as np
import random
import statsmodels.discrete.discrete_model as sm
import matplotlib.pyplot as plt
import seaborn

# Get ACS US person record datasets part a and b
ACS_dataset = pd.read_csv('usparta_allvars.csv')
ACS_dataset2 = pd.read_csv('uspartb_allvars.csv')
ACS_dataset = pd.concat([ACS_dataset, ACS_dataset2])
ACS_dataset = ACS_dataset.fillna(0)
ACS_dataset = ACS_dataset[ACS_dataset.relp == 16] # Keep institutionalized only
ACS_dataset.adjinc /= 1e6

# Get ACS US household record datasets part a and b
ACS_dataset_household = pd.read_csv('usparta_allvars_household.csv')
ACS_dataset2_household = pd.read_csv('uspartb_allvars_household.csv')
ACS_dataset_household = pd.concat([ACS_dataset_household, ACS_dataset2_household])
ACS_dataset_household = ACS_dataset_household[ACS_dataset_household.type == 2] # Keep institutionalized only

# Food stamp variable kept and other demographics that may be helpful, and that are present in household data
ACS_dataset_household = ACS_dataset_household[['serialno', 'division','type', 'region','fs']]
ACS_dataset = pd.merge(ACS_dataset, ACS_dataset_household, on = "serialno")

ACS_dataset['hfoodsp'] = ACS_dataset['fs']

# Earned and unearned_income Distribution
ACS_dataset['earned_income'] = ACS_dataset[['wagp', 'semp']].sum(axis = 1) * ACS_dataset['adjinc']
ACS_dataset['unearned_income'] = ACS_dataset[['retp', 'ssip', 'ssp', 'intp', 'pap', 'oip']].sum(axis = 1) * ACS_dataset['adjinc']


# plotting Educational Attainment Distribution
keylist = {0:'N/A (less than 3 years old)',1:'No schooling completed',2:'Nursery school, preschool',
3:'Kindergarten',4:'Grade 1',5:'Grade 2',6:'Grade 3',7:'Grade 4',8:'Grade 5',9:'Grade 6',10:'Grade 7',11:'Grade 8',12:'Grade 9',13:'Grade 10',
14:'Grade 11',15:'12th grade - no diploma',16:'Regular high school diploma',17:'GED or alternative credential',
18:'Some college, but less than 1 year',19:'1 or more years of college credit, no degree',20:'Associates degree',
21:'Bachelors degree',22:'Masters degree',23:'Professional degree beyond a bachelors degree',24:'Doctorate degree'}
ACS_dataset['Attainment strings'] = ACS_dataset['schl'].map(keylist)
ACS_dataset['schl'] = np.where((ACS_dataset.schl >=1)  & (ACS_dataset.schl<=3) , 31, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl >=4)  & (ACS_dataset.schl<=7) , 32, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl >=8)  & (ACS_dataset.schl<=9) , 33, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl >=10)  & (ACS_dataset.schl<=11) , 34, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl==12) , 35, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl==13) , 36, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl==14) , 37, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl==15) , 38, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl >=16)  & (ACS_dataset.schl<=17) , 39, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl >=18)  & (ACS_dataset.schl<=19) , 40, ACS_dataset.schl)
ACS_dataset.loc[ACS_dataset.schl == 20, "schl"] = np.random.randint(41,43,len(ACS_dataset[ACS_dataset.schl == 20]['schl']))
ACS_dataset['schl'] = np.where((ACS_dataset.schl==21) , 43, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl==22) , 44, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl==23) , 45, ACS_dataset.schl)
ACS_dataset['schl'] = np.where((ACS_dataset.schl==24) , 46, ACS_dataset.schl)

ACS_dataset['a_hga'] = ACS_dataset['schl']
ACS_dataset['a_age'] = ACS_dataset['agep']
# ACS has more data on older individuals (not topcoded, and many in institutions are older)
ACS_dataset['a_sex'] = ACS_dataset['sex']
ACS_dataset['msp'] = np.where((ACS_dataset.msp == 0)  | (ACS_dataset.msp == 6) , 7, ACS_dataset.msp)
ACS_dataset['msp'] = np.where((ACS_dataset.msp == 5) , 6, ACS_dataset.msp)
ACS_dataset['msp'] = np.where((ACS_dataset.msp == 4) , 5, ACS_dataset.msp)
ACS_dataset['msp'] = np.where((ACS_dataset.msp == 3) , 4, ACS_dataset.msp)
ACS_dataset['msp'] = np.where((ACS_dataset.msp == 2) , 3, ACS_dataset.msp)
ACS_dataset['a_maritl'] = ACS_dataset['msp']
# Marital status differences:
# No civilian distinction, but N/A distinction of too young. All institutionalized don't have spouse present

ACS_dataset['ptotval'] = ACS_dataset['pincp'] * ACS_dataset.adjinc
ACS_dataset['pearnval'] = ACS_dataset['pernp'] * ACS_dataset.adjinc
ACS_dataset['pothval'] = ACS_dataset['unearned_income']

ACS_dataset['wsal_val'] = ACS_dataset['wagp'] * ACS_dataset.adjinc
ACS_dataset['semp_val'] = ACS_dataset['semp'] * ACS_dataset.adjinc
# Also counts farm income and self employment

# These questions refer to last 12 months, where
ACS_dataset['ssi_val'] = ACS_dataset['ssip'] * ACS_dataset.adjinc
ACS_dataset['ss_val'] = ACS_dataset['ssp'] * ACS_dataset.adjinc
ACS_dataset['paw_val'] = ACS_dataset['pap'] * ACS_dataset.adjinc
ACS_dataset['rac1p'] = np.where((ACS_dataset.rac1p >= 3) & (ACS_dataset.rac1p <= 5) , 3, ACS_dataset.rac1p)
ACS_dataset['rac1p'] = np.where((ACS_dataset.rac1p== 6) , 4, ACS_dataset.rac1p)
ACS_dataset['rac1p'] = np.where((ACS_dataset.rac1p== 7) , 5, ACS_dataset.rac1p)
ACS_dataset['rac1p'] = np.where((ACS_dataset.rac1p== 8) , 0, ACS_dataset.rac1p)
ACS_dataset.loc[ACS_dataset.rac1p == 9, "rac1p"] = np.random.randint(6,27,len(ACS_dataset[ACS_dataset.rac1p == 9]['rac1p']))

ACS_dataset['prdtrace'] = ACS_dataset['rac1p']

ACS_dataset['gestfips'] = ACS_dataset['st']

ACS_dataset['prison_age'] = np.where((ACS_dataset.a_age < 45) & (ACS_dataset.a_age > 14), 1, 0)
ACS_dataset['prison_male'] = np.where((ACS_dataset.a_sex == 1), 1, 0) 
ACS_dataset['prison_self_care'] = np.where((ACS_dataset.ddrs == 1), -1, 0) 
ACS_dataset['prison_independent'] = np.where((ACS_dataset.dout == 1), -1, 0) 
ACS_dataset['prison_ambulatory'] = np.where((ACS_dataset.dphy == 2), 1, 0) 
ACS_dataset['prisoner'] = ACS_dataset[['prison_age', 'prison_male', 'prison_self_care', 'prison_independent', 'prison_ambulatory']].sum(axis = 1)
ACS_dataset.prisoner = np.where(ACS_dataset['prisoner'] > 0, 1, 0)
ACS_dataset['other_institutionalized'] = np.where(ACS_dataset.prisoner != 1, 1, 0)
ACS_dataset_to_CPS = ACS_dataset[['a_maritl', 'ssi_val' , 'ss_val', 'paw_val', 'hfoodsp',
        'prdtrace', 'gestfips', 'prisoner', 'other_institutionalized', 'semp_val', 'wsal_val',
        'ptotval', 'pothval', 'pearnval', 'a_sex', 'a_age', 'a_hga']]
ACS_dataset_to_CPS.to_csv('ACS_2015_Institutionalized.csv')


