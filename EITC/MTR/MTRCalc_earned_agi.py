import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from numba import jit
from bokeh.charts import Scatter, output_file, show
from scipy import stats
import tabulate
import seaborn


'''This script calculates the MTRs for EITC using 2015 augmented CPS tax-unit data created
by Jon O'Hare

Outputs a csv that contains implicit MTRs for EITC program'''


filenames = ['cpsrets2015']

df_list = []
for i in xrange(len(filenames)):
    df_list.append(pd.read_csv('2015 CPS Database/' + filenames[i]+'.csv'))

def initCPSvars(cps):
    '''
        This function uses the following variables to create relevant variables 
        to be used in the MTR computation:

        XXOCAH           Number of Children at Home Exemptions
        XXOCAWH          Number of Children Away From Home Exemptions
        JS               Federal Filing Status, 1=Single, 2=Joint, 3=Head of Household
        WT               Sample Weight   Divided by 3.000 for the Pooled CPS Sample.
        JCPS022          Interest Income - Head
        JCPS032          Interest Income - Spouse
        JCPS023          Dividends - Head
        JCPS033          Dividends - Spouse
        JCPS027          Rents, Royalties - Head
        JCPS037          Rents, Royalties - Spouse
        JCPS009          Adjusted Gross Income: Census (Taxpayer)
        JCPS019          Adjusted Gross Income: Census (Spouse)
        ICPS27           Disability Income
        ICPS47           Income From Non-Farm SoleProprietorships
        ICPS48           Income From Farm SoleProperietorships
        JCPS021          Wage and Salary Income - Head
        JCPS025          Business Income or Loss - Head
        JCPS028          Farm Income/Loss - Head
        JCPS031          Wage and Salary Income - Spouse
        JCPS035          Business Income or Loss - Spouse
        JCPS038          Farm Income/Loss - Spouse
         
        OUTPUT:

        AGI             Adjust Gross Income of both taxpayer and Spouse
        earned_income   Sum of all wage, salary and farm earnings
        inv_inc         Sum of Interest, dividend, and rent income 
        children        Number of children in tax tax-unit
        married         Filing status of tax-unit
        weight          Provided CPS sample weights
        '''
    children = cps['XXOCAH'] + cps['XXOCAWH']
    #this variable is a number >0 if married, and 0 if not. Account for this
    cps.ix[cps.JS != 2, 'JS'] = 1 
    married = cps['JS']
    weight = cps['WT']
    #this variable will be used to see if investment income is above threshold
    inv_inc = cps['JCPS22'] + cps['JCPS32'] + cps['JCPS23'] + cps['JCPS33'] + \
        cps['JCPS27'] + cps['JCPS37']
    AGI = cps['JCPS9'] + cps['JCPS19']
    #making sure only disability for non-retirees are counted
    cps.ix[cps.JCPS1 >= 65, 'ICPS27'] = 0
    earned_income = cps['ICPS27'] + cps['ICPS47'] + cps['ICPS48'] + \
        cps['JCPS21'] + cps['JCPS25'] + cps['JCPS28'] + \
        cps['JCPS31'] + cps['JCPS35'] + cps['JCPS38']
    earned_income.ix[earned_income < 0] = 0.
    return AGI, earned_income, inv_inc, children, married, weight

@jit(nopython=True)
def calcEITC(earned_AGI, inv_inc, children, married, eitc_zero_max, eitc_one_max,
    eitc_two_max, eitc_three_max, s_max_none, s_max_one, s_max_two, s_max_three, m_max_none, m_max_one, m_max_two,
    m_max_three, phasein_none, phasein_one, phasein_two, phasein_three, s_phaseout_none, s_phaseout_one,
    s_phaseout_two, s_phaseout_three, m_phaseout_none, m_phaseout_one, m_phaseout_two, m_phaseout_three,
    pctin_none, pctin_one, pctin_two, pctin_three, pctout_none, pctout_one, pctout_two, pctout_three,
    b_s_none, b_s_one, b_s_two, b_s_three, b_m_none, b_m_one, b_m_two, b_m_three, inv_max, AGI = False):
    
    '''This function is the programmed income rules for the EITC benefit'''

    pre_EITC = 0.0
    # The case for singles:
    if married == 1:
        if children == 0:
            # If earned_income and AGI is less than the maximum amount for singles
            # and positive
            if earned_AGI < s_max_none and earned_AGI >= 0:
                if earned_AGI <= phasein_none:
                    if AGI:
                        return 1e10
                    else:
                        EITC = earned_AGI * pctin_none
                elif earned_AGI <= s_phaseout_none:
                    if AGI:
                        return 1e10
                    else:
                        EITC = eitc_zero_max
                else:
                    EITC = pctout_none * earned_AGI + b_s_none
            else:
                return pre_EITC

        elif children == 1:
            # If earned_income is less than the maximum amount for single individuals
            # with one child and positive
            if earned_AGI < s_max_one and earned_AGI >= 0:
                if earned_AGI <= phasein_one:
                    if AGI:
                        return 1e10
                    else:
                        EITC = earned_AGI * pctin_one
                elif earned_AGI <= s_phaseout_one:
                    if AGI:
                        return 1e10
                    else:
                        EITC = eitc_one_max
                else:
                    EITC = pctout_one * earned_AGI + b_s_one
            else:
                return pre_EITC
        elif children == 2:
            # If earned_income is less than the maximum amount for single individuals
            # with two children and positive
            if earned_AGI < s_max_two and earned_AGI >= 0:
                if earned_AGI <= phasein_two:
                    if AGI:
                        return 1e10
                    else:
                        EITC = earned_AGI * pctin_two
                elif earned_AGI <= s_phaseout_two:
                    if AGI:
                        return 1e10
                    else:
                        EITC = eitc_two_max
                else:
                    EITC = pctout_two * earned_AGI + b_s_two
            else:
                return pre_EITC

        else:
            # If earned_income is less than the maximum amount for single individuals
            # with three or more children and positive
            if earned_AGI < s_max_three and earned_AGI >= 0:
                if earned_AGI <= phasein_three:
                    if AGI:
                        return 1e10
                    else:
                        EITC = earned_AGI * pctin_three
                elif earned_AGI <= s_phaseout_three:
                    if AGI:
                        return 1e10
                    else:
                        EITC = eitc_three_max
                else:
                    EITC = pctout_three * earned_AGI + b_s_three
            else:
                return pre_EITC
    # The case for married:
    else:
        if children == 0:
            # If earned_income is positive and less than the maximum amount for
            # married individuals with no children
            if earned_AGI < m_max_none and earned_AGI >= 0:
                if earned_AGI <= phasein_none:
                    if AGI:
                        return 1e10
                    else:
                        EITC = earned_AGI * pctin_none
                elif earned_AGI <= m_phaseout_none:
                    if AGI:
                        return 1e10
                    else:
                        EITC = eitc_zero_max
                else:
                    EITC = pctout_none * earned_AGI + b_m_none
            else:
                return pre_EITC


        elif children == 1:
            # If earned_income is positive and less than the maximum amount for
            # married individuals with one child
            if earned_AGI < m_max_one and earned_AGI >= 0:
                if earned_AGI <= phasein_one:
                    if AGI:
                        return 1e10
                    else:
                        EITC = earned_AGI * pctin_one
                elif earned_AGI <= m_phaseout_one:
                    if AGI:
                        return 1e10
                    else:
                        EITC = eitc_one_max
                else:
                    EITC = pctout_one * earned_AGI + b_m_one
            else:
                return pre_EITC

        elif children == 2:
            # If earned_income is positive and less than the maximum amount for
            # married individuals with two children
            if earned_AGI < m_max_two and earned_AGI >= 0:
                if earned_AGI <= phasein_two:
                    if AGI:
                        return 1e10
                    else:
                        EITC = earned_AGI * pctin_two
                elif earned_AGI <= m_phaseout_two:
                    if AGI:
                        return 1e10
                    else:
                        EITC = eitc_two_max
                else:
                    EITC = pctout_two * earned_AGI + b_m_two
            else:
                return pre_EITC

        else:
            # If earned_income is positive and less than the maximum amount for
            # married individuals with three or more children
            if earned_AGI < m_max_three and earned_AGI >= 0:
                if earned_AGI <= phasein_three:
                    if AGI:
                        return 1e10
                    else:
                        EITC = earned_AGI * pctin_three
                elif earned_AGI <= m_phaseout_three:
                    if AGI:
                        return 1e10
                    else:
                        EITC = eitc_three_max
                else:
                    EITC = pctout_three * earned_AGI + b_m_three
            else:
                return pre_EITC
    return EITC


def eitc_plot(eitc):
    '''This plots eitc against earned income'''
    earned_income = eitc['earned_income']
    eitc1 = eitc['eitc']
    fig, ax  = plt.subplots()
    plt.scatter(earned_income, eitc1, label = None)
    legend = ax.legend(loc = "upper right", shadow = True, title = 'eitc for earned_income')
    plt.xlabel('Earned Income')
    plt.ylabel('EITC Amount')
    plt.xlim(0,58000)
    plt.ylim(0,7000)
    plt.title('Earned Income Tax Credit for Earned Income Amounts')
    plt.show()

def mtr_plot(eitc):
    '''This plots the MTRs against earned income'''
    earned_income = eitc['earned_income']
    mtr = eitc['MTR_computed']
    fig, ax  = plt.subplots()
    plt.scatter(earned_income, mtr, label = None)
    legend = ax.legend(loc = "upper right", shadow = True, title = 'eitc for earned_income')
    plt.xlabel('Earned Income')
    plt.ylabel('MTR')
    plt.xlim(0,58000)
    plt.title('MTR for Earned Income Amounts')
    plt.show()


def lin_Reg(df):
    '''computes the linear regression of EITC amount on earned_income'''
    x = df['earned_income']
    y = df['eitc']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return slope, intercept

# Main:

# Extracting relevant variables from CPS and defining different income rule thresholds:
AGI, earned_income, inv_inc, children, married, weight = initCPSvars(df_list[0])
# Max income for singles to receive EITC, for different amounts of children:
s_max_none = 14820.
s_max_one = 39131.
s_max_two = 44454.
s_max_three = 47747.
# Max income for married individuals to receive EITC, for different amounts of children:
m_max_none = 20330.
m_max_one = 44651.
m_max_two = 49974.
m_max_three = 53267.
# Max EITC amounts for individuals with different amounts of children:
eitc_three_max = 6242.
eitc_two_max = 5548.
eitc_one_max = 3359.
eitc_zero_max = 503.
# Phase-in rates for different numbers of children:
phasein_none = 6580.
phasein_one = 9880.
phasein_two = 13870.
phasein_three = 13870.
# Phase-out rates for different numbers of children and filing status:
s_phaseout_none = 8240.
s_phaseout_one = 18110.
s_phaseout_two = 18110.
s_phaseout_three = 18110.
m_phaseout_none = s_phaseout_none + 5520.
m_phaseout_one = s_phaseout_one + 5520.
m_phaseout_two = s_phaseout_two + 5520.
m_phaseout_three = s_phaseout_three + 5520.
# Phase-in percentage rates:
pctin_none = .0765
pctin_one = .3399
pctin_two = .40
pctin_three = .45
# phase-out percentage rates:
pctout_none =-.0765 
pctout_one =-.1598
pctout_two =-.2106
pctout_three =-.2106
# intercept for phase out line for filing status and children:
b_s_none =-(pctout_none * s_max_none)
b_s_one =-(pctout_one * s_max_one)
b_s_two =-(pctout_two * s_max_two)
b_s_three =-(pctout_three * s_max_three)
b_m_none =-(pctout_none * m_max_none)
b_m_one =-(pctout_one * m_max_one)
b_m_two =-(pctout_two * m_max_two)
b_m_three =-(pctout_three * m_max_three)
# Maximum investment rule:
inv_max = 3400.
eitc = np.zeros(len(earned_income))
eitcAGI = np.zeros(len(earned_income))

# Calculating EITC amounts before income adjustment for earned income:
for k in xrange(len(eitc)):
    eitc[k] = calcEITC(earned_income[k], inv_inc[k], children[k], married[k], eitc_zero_max, eitc_one_max,
    eitc_two_max, eitc_three_max, s_max_none, s_max_one, s_max_two, s_max_three, m_max_none, m_max_one, m_max_two,
    m_max_three, phasein_none, phasein_one, phasein_two, phasein_three, s_phaseout_none, s_phaseout_one,
    s_phaseout_two, s_phaseout_three, m_phaseout_none, m_phaseout_one, m_phaseout_two, m_phaseout_three,
    pctin_none, pctin_one, pctin_two, pctin_three, pctout_none, pctout_one, pctout_two, pctout_three,
    b_s_none, b_s_one, b_s_two, b_s_three, b_m_none, b_m_one, b_m_two, b_m_three, inv_max)

# Calculating EITC amounts before income adjustment for AGI:
for i in xrange(len(eitc)):
    eitcAGI[i] = calcEITC(AGI[i], inv_inc[i], children[i], married[i], eitc_zero_max, eitc_one_max,
    eitc_two_max, eitc_three_max, s_max_none, s_max_one, s_max_two, s_max_three, m_max_none, m_max_one, m_max_two,
    m_max_three, phasein_none, phasein_one, phasein_two, phasein_three, s_phaseout_none, s_phaseout_one,
    s_phaseout_two, s_phaseout_three, m_phaseout_none, m_phaseout_one, m_phaseout_two, m_phaseout_three,
    pctin_none, pctin_one, pctin_two, pctin_three, pctout_none, pctout_one, pctout_two, pctout_three,
    b_s_none, b_s_one, b_s_two, b_s_three, b_m_none, b_m_one, b_m_two, b_m_three, inv_max, True)

df1 = pd.Series(eitc)
df2 = pd.Series(eitcAGI)
eitc_smaller = df1.copy()
# Taking the smaller of the two EITC amounts:
eitc_smaller.ix[eitc_smaller > df2] = df2
EITC = pd.concat([eitc_smaller, earned_income, AGI], axis = 1)
EITC.columns = ['eitc', 'earned_income', 'AGI']
eitc_plot(EITC)
Eitcbefore = EITC.copy()
EITC['earned_income'] += 1
EITC['AGI'] += 1
eitc_new = np.zeros(len(earned_income))
eitc_AGI_new = np.zeros(len(earned_income))

# Calculating EITC amounts after income adjustment for earned income:

for i in xrange(len(eitc_new)):
    eitc_new[i] = calcEITC(EITC['earned_income'][i], inv_inc[i], children[i], married[i], eitc_zero_max, eitc_one_max,
    eitc_two_max, eitc_three_max, s_max_none, s_max_one, s_max_two, s_max_three, m_max_none, m_max_one, m_max_two,
    m_max_three, phasein_none, phasein_one, phasein_two, phasein_three, s_phaseout_none, s_phaseout_one,
    s_phaseout_two, s_phaseout_three, m_phaseout_none, m_phaseout_one, m_phaseout_two, m_phaseout_three,
    pctin_none, pctin_one, pctin_two, pctin_three, pctout_none, pctout_one, pctout_two, pctout_three,
    b_s_none, b_s_one, b_s_two, b_s_three, b_m_none, b_m_one, b_m_two, b_m_three, inv_max)

# Calculating EITC amounts after income adjustment for AGI:

for k in xrange(len(eitc_AGI_new)):
    eitc_AGI_new[k] = calcEITC(EITC['AGI'][k], inv_inc[k], children[k], married[k], eitc_zero_max, eitc_one_max,
    eitc_two_max, eitc_three_max, s_max_none, s_max_one, s_max_two, s_max_three, m_max_none, m_max_one, m_max_two,
    m_max_three, phasein_none, phasein_one, phasein_two, phasein_three, s_phaseout_none, s_phaseout_one,
    s_phaseout_two, s_phaseout_three, m_phaseout_none, m_phaseout_one, m_phaseout_two, m_phaseout_three,
    pctin_none, pctin_one, pctin_two, pctin_three, pctout_none, pctout_one, pctout_two, pctout_three,
    b_s_none, b_s_one, b_s_two, b_s_three, b_m_none, b_m_one, b_m_two, b_m_three, inv_max, True)

df3 = pd.Series(eitc_new)
df4 = pd.Series(eitc_AGI_new)
eitc_smaller_new = df3.copy()
eitc_smaller_new.ix[eitc_smaller_new > df4] = df4
MTR_CPS = (eitc_smaller_new - eitc_smaller).copy()
EITC['earned_income'] -= 1.
EITC['AGI'] -= 1.
EITC = pd.concat([EITC, children, MTR_CPS, weight], axis = 1)
# Creating a column of computed MTRs using income rules
EITC.columns = ['eitc', 'earned_income', 'AGI', 'children', 'MTR_computed', 'weight']
mtr_plot(EITC)
# Defining deciles
income_cutoffs_none = [10, 1100, 2500, 3900, 5300, 6700, 8100, 9800, 11600, 13200, 19600]
income_cutoffs_one = [44, 4930, 8600, 10700, 13000, 15958, 19500, 23372, 27450, 32400, 43200]
income_cutoffs_two = [10, 7233, 11900, 14150, 15830, 18720, 22700, 26700, 31800, 37100, 48300]
income_cutoffs_three = [1, 7620, 12160, 14470, 17060, 21270, 25030, 29900, 34670, 41430, 51500]
income_cutoffs_list = [income_cutoffs_none, income_cutoffs_one, income_cutoffs_two, income_cutoffs_three]

children = [0, 1, 2, 3]
j = 1

#These will contain the program MTR and the regression results dependent on number of children

Program_MTR_none = []
Program_MTR_one = []
Program_MTR_two = []
Program_MTR_three = []
Program_MTR = [Program_MTR_none, Program_MTR_one, Program_MTR_two, Program_MTR_three]


# Taking weighted averages of MTRs over difference income deciles, and number of children
# and performing decile regressions:

for i in xrange(len(income_cutoffs_list[0])-1):
    for c in children:
        if c >= 3:
            df = EITC[(EITC['earned_income'] < income_cutoffs_list[c][j]) & (EITC['earned_income'] >= income_cutoffs_list[c][i])\
                & (EITC['children'] >= c)]
            slope, intercept = lin_Reg(df)
            Program_MTR[c].append([str(income_cutoffs_list[c][i]) + '<= income <' + str(income_cutoffs_list[c][j]), \
                (np.sum(df['weight'] *df['MTR_computed'])/(np.sum(df['weight']))), slope])

        else:
            df = EITC[(EITC['earned_income'] < income_cutoffs_list[c][j]) & (EITC['earned_income'] >= income_cutoffs_list[c][i])\
                & (EITC['children'] == c)]
            slope, intercept = lin_Reg(df)
            Program_MTR[c].append([str(income_cutoffs_list[c][i]) + '<= income <' + str(income_cutoffs_list[c][j]), \
                (np.sum(df['weight'] *df['MTR_computed'])/(np.sum(df['weight']))), slope])

    j+=1

# Outputting EITC MTRs to csv file and
# outputting a table of the average MTRs over deciles for Income Rules, and Regressions
CPS.to_csv('EITC_MTR_Earned_AGI.csv', index=False)
for i in xrange(len(Program_MTR)):
    print tabulate.tabulate(Program_MTR[i], tablefmt = "latex")
