''' 
This script creates the administrative data used for imputation.

You can download ar5159.csv, aw5159.csv, and ar539.csv at https://workforcesecurity.doleta.gov/unemploy/DataDownloads.asp by following these instructions after having navigated to that page:

*Go to the table titled "ETA 5159: Claims and Payment Activities"
*Download: 
    *the regular program by clicking the "Data" hyperlink in the "Raw Data" column and the "Regular Program" row (should download as ar5159.csv).
    *the workshare program by clicking the "Data" hyperlink in the "Raw Data" column and the "Workshare (STC)" row (should download as ar5159.csv).

*Go to the table titled "ETA 539: Weekly Claims and Extended Benefits Trigger Data" 
*Download: 
    *the regular program by clicking the "Data" hyperlink in the "Raw Data" column and the "Regular Program" row (should download as ar539.csv).

*Move these files to the C-TAM/WC/ directory.
'''

import numpy as np
import pandas as pd
import re
#extended benefits were negligible -- recipients and outlays were less than .1% of the regular program totals so we omit them

# Here we gather administrative data on the previous year UI spillovers into 2014 from Workshare and Regular UI programs
previous_year = pd.read_csv('ar539.csv')
previous_year = previous_year.replace(' ', 0)
previous_year[['c8','c9', 'c10', 'c11']] = previous_year[['c8','c9', 'c10', 'c11']].astype(int)
previous_year = previous_year[(previous_year.st != "PR") & (previous_year.st != "VI")]
previous_year['spillover_UI'] = previous_year[['c8','c9', 'c10', 'c11']].sum(axis = 1)
previous_year.rptdate = pd.to_datetime(previous_year.rptdate, infer_datetime_format=True)
previous_year.c2 = pd.to_datetime(previous_year.c2 , infer_datetime_format=True)
previous_year.spillover_UI = np.where((previous_year.rptdate.dt.year == 2014) & (previous_year.c2.dt.year == 2013), previous_year.spillover_UI, 0)
previous_year = previous_year.reset_index()
previous_year = previous_year[previous_year.spillover_UI > 0]
previous_year = previous_year[['st', 'spillover_UI']]
previous_year = previous_year.reset_index(drop = True)


#Below we consider the workshare benefits 
workshare_ben = pd.read_csv('aw5159.csv')
workshare_ben = workshare_ben.replace(' ', 0)
workshare_ben[['c6', 'c5']] = workshare_ben[['c6', 'c5']].astype(int)
workshare_ben = workshare_ben[(workshare_ben.st != "PR") & (workshare_ben.st != "VI")]
workshare_ben.rptdate = pd.to_datetime(workshare_ben.rptdate, infer_datetime_format=True)
workshare_ben = workshare_ben[workshare_ben.rptdate.dt.year == 2014]
workshare_ben = workshare_ben.groupby(['st']).sum()
workshare_ben = workshare_ben.reset_index()
workshare_ben['tot_UI_outlays_workshare'] = workshare_ben.c5
workshare_ben['tot_UI_recipients_workshare'] = (workshare_ben.c6).astype(int)
workshare_ben = workshare_ben[['st', 'tot_UI_outlays_workshare', 'tot_UI_recipients_workshare']]

#Below we consider the regular program benefits
Admin_totals = pd.read_csv('ar5159.csv')
Admin_totals = Admin_totals.replace(' ', 0)
Admin_totals[['c48', 'c45', 'c51' , 'c55' , 'c54']] = \
	Admin_totals[['c48', 'c45', 'c51' , 'c55' , 'c54']].astype(int)
Admin_totals = Admin_totals[(Admin_totals.st != "PR") & (Admin_totals.st != "VI")]
Admin_totals.rptdate = pd.to_datetime(Admin_totals.rptdate, infer_datetime_format=True)
Admin_totals = Admin_totals[Admin_totals.rptdate.dt.year == 2014]
Admin_totals = Admin_totals.groupby(['st']).sum()
Admin_totals = Admin_totals.reset_index()
Admin_totals['tot_UI_recipients'] = Admin_totals.c51 + Admin_totals.c54 + Admin_totals.c55
Admin_totals['tot_UI_outlays'] = Admin_totals.c48 + Admin_totals.c45
Admin_totals = Admin_totals.merge(workshare_ben, how = 'left', on = 'st')
Admin_totals = Admin_totals.fillna(0)
Admin_totals = Admin_totals.merge(previous_year, how = 'left', on = 'st')
Admin_totals = Admin_totals.fillna(0)
Admin_totals['tot_UI_recipients'] +=  Admin_totals['tot_UI_recipients_workshare']
Admin_totals['tot_UI_recipients'] +=  Admin_totals['spillover_UI']
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
Admin_totals.to_csv('Admin_totals_all.csv')
