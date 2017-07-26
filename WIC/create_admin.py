''' This script creates the administrative data used for imputation'''

''' You can find WICAgencies2014ytd.xls at https://www.fns.usda.gov/pd/wic-program. 
Download the Monthly Data – State Level Participation by Category and Program Costs, 
data corresponding to your year'''

import numpy as np
import pandas as pd
import re

xls = pd.ExcelFile('WICAgencies2014ytd.xls')
tot_infants = xls.parse('Total Infants',header = 1 ,skiprows = 3)
tot_infants = tot_infants.rename(columns = {'State Agency or Indian Tribal Organization' : 'State'})
tot_children = xls.parse('Children Participating', header = 1, skiprows = 3)
tot_children = tot_children.rename(columns = {'State Agency or Indian Tribal Organization' : 'State'})
tot_women = xls.parse('Total Women', header = 1, skiprows = 3)
tot_women = tot_women.rename(columns = {'State Agency or Indian Tribal Organization' : 'State'})


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


tot_infants = tot_infants.replace(states, regex = True)
tot_infants['State_indian'] = tot_infants['State'].str.split(',').str[1]
indian_res = tot_infants.dropna(0)['State_indian']
indian_res[(indian_res == ' Caddo & Delaware (WCD)')] = 'Oklahoma'
indian_res[(indian_res == ' Canoncito & Laguna')] = 'New Mexico'
tot_infants.loc[indian_res.index, "State"] = indian_res
tot_infants.State = tot_infants['State'].str.strip()
tot_infants = tot_infants[tot_infants['State'].isin(states.values())]
tot_infants = tot_infants.groupby(['State'])['Average Participation'].sum().astype(int)

tot_children = tot_children.replace(states, regex = True)
tot_children['State_indian'] = tot_children['State'].str.split(',').str[1]
indian_res = tot_children.dropna(0)['State_indian']
indian_res[(indian_res == ' Caddo & Delaware (WCD)')] = 'Oklahoma'
indian_res[(indian_res == ' Canoncito & Laguna')] = 'New Mexico'
tot_children.loc[indian_res.index, "State"] = indian_res
tot_children.State = tot_children['State'].str.strip()
tot_children = tot_children[tot_children['State'].isin(states.values())]
tot_children = tot_children.groupby(['State'])['Average Participation'].sum().astype(int)

tot_women = tot_women.replace(states, regex = True)
tot_women['State_indian'] = tot_women['State'].str.split(',').str[1]
indian_res = tot_women.dropna(0)['State_indian']
indian_res[(indian_res == ' Caddo & Delaware (WCD)')] = 'Oklahoma'
indian_res[(indian_res == ' Canoncito & Laguna')] = 'New Mexico'
tot_women.loc[indian_res.index, "State"] = indian_res
tot_women.State = tot_women['State'].str.strip()
tot_women = tot_women[tot_women['State'].isin(states.values())]
tot_women = tot_women.groupby(['State'])['Average Participation'].sum().astype(int)

tot_infants.to_csv('Admin_totals_infants.csv')
tot_women.to_csv('Admin_totals_women.csv')
tot_children.to_csv('Admin_totals_children.csv')