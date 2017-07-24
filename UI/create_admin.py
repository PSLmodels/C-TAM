''' This script creates the administrative data used for imputation'''

''' You can find ar5159.csv at https://workforcesecurity.doleta.gov/unemploy/DataDownloads.asp. Download the regular program and workshare program files'''
import numpy as np
import pandas as pd
import re
#extended benefits were negligible -- recipients and outlays were less than .1% of the regular program totals so we omit them

#Below we consider the workshare benefits 
workshare_ben = pd.read_csv('aw5159.csv')
workshare_ben = workshare_ben.replace(' ', 0)
workshare_ben[['c3','c6', 'c5']] = workshare_ben[['c3','c6', 'c5']].astype(int)
workshare_ben = workshare_ben[(workshare_ben.st != "PR") & (workshare_ben.st != "VI")]
workshare_ben['from_prv_year_aw'] = workshare_ben.c3 
workshare_ben['from_prv_year_aw'] = np.where(workshare_ben.rptdate != '1/31/2014', 0, workshare_ben.from_prv_year_aw)
workshare_ben['from_prv_year_aw'] /= 4.
workshare_ben.rptdate = pd.to_datetime(workshare_ben.rptdate, infer_datetime_format=True)
workshare_ben = workshare_ben.groupby(['st',workshare_ben['rptdate'].map(lambda x: x.year)]).sum()
workshare_ben = workshare_ben.reset_index()
workshare_ben = workshare_ben[workshare_ben.rptdate == 2014]
workshare_ben['tot_UI_outlays_workshare'] = workshare_ben.c5
workshare_ben['tot_UI_recipients_workshare'] = (workshare_ben.c6 + workshare_ben.from_prv_year_aw).astype(int)
workshare_ben = workshare_ben[['st', 'tot_UI_outlays_workshare', 'tot_UI_recipients_workshare']]

#Below we consider the regular program benefits
Admin_totals = pd.read_csv('ar5159.csv')
Admin_totals = Admin_totals.replace(' ', 0)
Admin_totals[['c48', 'c45', 'c51' , 'c55' , 'c54', 'c21', 'c22', 'c27', 'c28', 'c33', 'c34']] = \
	Admin_totals[['c48', 'c45', 'c51' , 'c55' , 'c54', 'c21', 'c22', 'c27', 'c28', 'c33', 'c34']].astype(int)

Admin_totals = Admin_totals[(Admin_totals.st != "PR") & (Admin_totals.st != "VI")]

Admin_totals['from_prv_year_ar'] = Admin_totals.c21 + Admin_totals.c22 + Admin_totals.c27 +\
	Admin_totals.c28 + Admin_totals.c33 + Admin_totals.c34
Admin_totals['from_prv_year_ar'] = np.where(Admin_totals.rptdate != '1/31/2014', 0, Admin_totals.from_prv_year_ar)
Admin_totals['from_prv_year_ar'] /= 4.
Admin_totals['double_counted'] = np.where(Admin_totals.rptdate != '1/31/2014', 0, Admin_totals.c51 + \
	Admin_totals.c54 + Admin_totals.c55)
Admin_totals['from_prv_year_ar'] -= Admin_totals['double_counted'] * .75
Admin_totals['tot_UI_recipients'] = Admin_totals.c51 + Admin_totals.c54 + Admin_totals.c55 + Admin_totals.from_prv_year_ar
Admin_totals.rptdate = pd.to_datetime(Admin_totals.rptdate, infer_datetime_format=True)
Admin_totals = Admin_totals.groupby(['st',Admin_totals['rptdate'].map(lambda x: x.year)]).sum()
Admin_totals = Admin_totals.reset_index()
Admin_totals = Admin_totals[Admin_totals.rptdate == 2014]
Admin_totals['tot_UI_outlays'] = Admin_totals.c48 + Admin_totals.c45
Admin_totals = Admin_totals.merge(workshare_ben, how = 'left', on = 'st')
Admin_totals = Admin_totals.fillna(0)
Admin_totals['tot_UI_recipients'] +=  Admin_totals['tot_UI_recipients_workshare']
Admin_totals['tot_UI_outlays'] += Admin_totals['tot_UI_outlays_workshare']
Admin_totals = Admin_totals[['st', 'tot_UI_outlays', 'tot_UI_recipients']]
Admin_totals.columns = ['state', 'tot_UI_outlays', 'tot_UI_recipients']
states = {
        'AK': 'Alaska','AL': 'Alabama','AR': 'Arkansas','AZ': 'Arizona',
        'CA': 'California','CO': 'Colorado','CT': 'Connecticut','DC': 'District of Columbia',
        'DE': 'Delaware','FL': 'Florida','GA': 'Georgia','HI': 'Hawaii','IA': 'Iowa',
        'ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','KS': 'Kansas','KY': 'Kentucky',
        'LA': 'Louisiana','MA': 'Massachusetts','MD': 'Maryland','ME': 'Maine','MI': 'Michigan',
        'MN': 'Minnesota','MO': 'Missouri','MS': 'Mississippi','MT': 'Montana',
        'NC': 'North Carolina','ND': 'North Dakota','NE': 'Nebraska','NH': 'New Hampshire',
        'NJ': 'New Jersey','NM': 'New Mexico','NV': 'Nevada','NY': 'New York','OH': 'Ohio',
        'OK': 'Oklahoma','OR': 'Oregon','PA': 'Pennsylvania','RI': 'Rhode Island','SC': 'South Carolina'
        ,'SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas','UT': 'Utah','VA': 'Virginia',
        'VT': 'Vermont','WA': 'Washington','WI': 'Wisconsin','WV': 'West Virginia','WY': 'Wyoming'}

fips = [1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,
			25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
			42,44,45,46,47,48,49,50,51,53,54,55,56]

Admin_totals['state'] = Admin_totals['state'].map(states)
Admin_totals = Admin_totals.reset_index(drop = True)
Admin_totals = Admin_totals.sort_values('state')
Admin_totals['Fips'] = fips
Admin_totals[['tot_UI_outlays', 'tot_UI_recipients']] = Admin_totals[['tot_UI_outlays', 'tot_UI_recipients']].astype(int)
print Admin_totals.tot_UI_outlays.sum() / Admin_totals.tot_UI_recipients.sum()
Admin_totals.to_csv('Admin_totals_all.csv')