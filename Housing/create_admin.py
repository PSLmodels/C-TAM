''' This script creates the administrative data used for imputation'''

''' You can find STATE_2014.csv at https://www.huduser.gov/portal/datasets/assthsg.html. Select the 2014 Based on 2010 Census for Year, State for Summary Level, ALL for State, Summary of All HUD Programs for Program, ALL for variable. We would use county level if CPS had county codes for all respondents'''

import numpy as np
import pandas as pd
import re
# Summary of all HUD programs
Admin_totals = pd.read_csv('STATE_2014.csv')
Admin_totals = Admin_totals.groupby(['Code', 'Program']).mean()\
        .reset_index()[Admin_totals.Program == 1].reset_index(drop = True)\
                [['Code', '# Reported', 'Average Family Expenditure per month ($$)']]

Admin_totals = Admin_totals.iloc[:-5]
Admin_totals.Code = Admin_totals.Code.map(lambda x: x.lstrip(\
        '="').rstrip('"'))
Admin_totals['Code'] = Admin_totals.astype(int)
Admin_totals['Average Family Expenditure per year ($$)'] = Admin_totals['Average Family Expenditure per month ($$)'] * Admin_totals['# Reported'] * 12
Admin_totals = Admin_totals[['Code', '# Reported', 'Average Family Expenditure per year ($$)']]
Admin_totals.columns = ['Fips', 'housing_recipients', 'housing_value']
states = ['Alabama' ,'Alaska', 'Arizona' ,'Arkansas', 'California' ,'Colorado',
 'Connecticut' ,'Delaware' ,'District of Columbia' ,'Florida' ,'Georgia',
 'Hawaii' ,'Idaho' ,'Illinois', 'Indiana', 'Iowa' ,'Kansas', 'Kentucky',
 'Louisiana' ,'Maine', 'Maryland' ,'Massachusetts', 'Michigan', 'Minnesota',
 'Mississippi' ,'Missouri', 'Montana' ,'Nebraska', 'Nevada' ,'New Hampshire',
 'New Jersey' ,'New Mexico' ,'New York' ,'North Carolina', 'North Dakota',
 'Ohio' ,'Oklahoma', 'Oregon' ,'Pennsylvania', 'Rhode Island', 'South Carolina',
 'South Dakota', 'Tennessee' ,'Texas' ,'Utah' ,'Vermont', 'Virginia',
 'Washington' ,'West Virginia' ,'Wisconsin' ,'Wyoming']
Admin_totals['state'] = states
Admin_totals.to_csv('Admin_totals_all.csv')

