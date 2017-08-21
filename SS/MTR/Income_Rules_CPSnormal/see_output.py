import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import seaborn
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
import sys, os
from subprocess import Popen, PIPE
import pickle
import time
import ast
import csv


# SS_MTR = pd.read_csv('SS_MTR_WithFutureEarningsReg.csv')
SS_MTR = pd.read_csv('SS_MTR_ConstantFuture.csv')

# print SS_MTR[SS_MTR['SS_MTR.1'] > 0.62]
print list(np.unique(np.round(SS_MTR['SS_MTR.1'], 2)))


# print SS_MTR[(SS_MTR['earned_income'] > 116500) & (SS_MTR['earned_income'] < 117001)]
# print SS_MTR[(SS_MTR['SS_MTR.1'] < .56) & (SS_MTR['earned_income'] < 10000)]
plt.scatter(SS_MTR["earned_income"], SS_MTR['SS_MTR.1'])
plt.xlim(-1000, 150000)
plt.xlabel("Earned Income")
plt.ylabel("SS MTR")
plt.show()



quants =  SS_MTR['earned_income'].quantile(np.linspace(0,1,100)).as_matrix()
SS_MTR['decile'] = 0

for i in xrange(len(quants)- 1):
	SS_MTR['decile'][(SS_MTR['earned_income'] > quants[i]) & (SS_MTR['earned_income'] <= quants[i + 1])] = i

grouped = SS_MTR.groupby('decile').mean()

x_vals = np.linspace(1, len(grouped['SS_MTR.1']), len(grouped['SS_MTR.1']))
plt.plot(x_vals/float(len(x_vals)), grouped['SS_MTR.1'])
plt.ylabel("SS MTR")
plt.show()