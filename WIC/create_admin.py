''' This script creates the administrative data used for imputation'''

''' You can find WICAgencies2014ytd.xls at https://www.fns.usda.gov/pd/wic-program. 
Download the Monthly Data - State Level Participation by Category and Program Costs, 
data corresponding to your year, to include administrative costs, refer use also 
the Nut. Services & Admin Costs excel sheet from the excel file below.
The "Rebates Received" sheet in the data sheet refers to the totals from the WIC Infant Formula Rebate Program '''

import numpy as np
import pandas as pd
import re

xls = pd.ExcelFile('WICAgencies2014ytd.xls')
tot_infants = xls.parse('Total Infants',header = 1 ,skiprows = 3)
tot_infants = tot_infants.rename(columns = {'State Agency or Indian Tribal Organization' : 'state'})
tot_children = xls.parse('Children Participating', header = 1, skiprows = 3)
tot_children = tot_children.rename(columns = {'State Agency or Indian Tribal Organization' : 'state'})
tot_women = xls.parse('Total Women', header = 1, skiprows = 3)
tot_women = tot_women.rename(columns = {'State Agency or Indian Tribal Organization' : 'state'})
benefits_received = xls.parse('Food Costs',header = 1 ,skiprows = 3)
benefits_received = benefits_received.rename(columns = {'State Agency or Indian Tribal Organization' : 'state'})

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
tot_infants['State_indian'] = tot_infants['state'].str.split(',').str[1]
indian_res = tot_infants.dropna(0)['State_indian']
indian_res[(indian_res == ' Caddo & Delaware (WCD)')] = 'Oklahoma'
indian_res[(indian_res == ' Canoncito & Laguna')] = 'New Mexico'
tot_infants.loc[indian_res.index, "state"] = indian_res
tot_infants.state = tot_infants['state'].str.strip()
tot_infants = tot_infants[tot_infants['state'].isin(states.values())]
tot_infants = tot_infants.groupby(['state'])['Average Participation'].sum().astype(int)
print tot_infants.name
tot_infants.name = "total_infants"


tot_children = tot_children.replace(states, regex = True)
tot_children['State_indian'] = tot_children['state'].str.split(',').str[1]
indian_res = tot_children.dropna(0)['State_indian']
indian_res[(indian_res == ' Caddo & Delaware (WCD)')] = 'Oklahoma'
indian_res[(indian_res == ' Canoncito & Laguna')] = 'New Mexico'
tot_children.loc[indian_res.index, "state"] = indian_res
tot_children.state = tot_children['state'].str.strip()
tot_children = tot_children[tot_children['state'].isin(states.values())]
tot_children = tot_children.groupby(['state'])['Average Participation'].sum().astype(int)
tot_children.name = 'total_children'

tot_women = tot_women.replace(states, regex = True)
tot_women['State_indian'] = tot_women['state'].str.split(',').str[1]
indian_res = tot_women.dropna(0)['State_indian']
indian_res[(indian_res == ' Caddo & Delaware (WCD)')] = 'Oklahoma'
indian_res[(indian_res == ' Canoncito & Laguna')] = 'New Mexico'
tot_women.loc[indian_res.index, "state"] = indian_res
tot_women.state = tot_women['state'].str.strip()
tot_women = tot_women[tot_women['state'].isin(states.values())]
tot_women = tot_women.groupby(['state'])['Average Participation'].sum().astype(int)
tot_women.name ='total_women'

benefits_received = benefits_received.replace(states, regex = True)
benefits_received['State_indian'] = benefits_received['state'].str.split(',').str[1]
indian_res = benefits_received.dropna(0)['State_indian']
indian_res[(indian_res == ' Caddo & Delaware (WCD)')] = 'Oklahoma'
indian_res[(indian_res == ' Canoncito & Laguna')] = 'New Mexico'
benefits_received.loc[indian_res.index, "state"] = indian_res
benefits_received.state = benefits_received['state'].str.strip()
benefits_received = benefits_received[benefits_received['state'].isin(states.values())]
benefits_received = benefits_received.groupby(['state'])['Cumulative Cost'].sum().astype(int)
benefits_received.name = 'total_benefits'

final = pd.concat([tot_infants, tot_women, tot_children, benefits_received], axis = 1).reset_index()
final = final.sort_values('state')
final['Fips'] = fips
final['total_recipients'] = final.total_infants + final.total_women + final.total_children
final['Avg_benefit'] = final['total_benefits'] / final['total_recipients']
final['Avg_benefit'] *= 10000
final['Avg_benefit'] = final['Avg_benefit'].astype(int)
final['Avg_benefit'] /= 10000
final['tot_infant_benefits'] = final['total_infants'] * final['Avg_benefit']
final['tot_child_benefits'] = final['total_children'] * final['Avg_benefit']
final['tot_woman_benefits'] = final['total_women'] * final['Avg_benefit']
final.to_csv('Admin_totals_all.csv')
