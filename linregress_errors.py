# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:26:48 2018

@author: llazarus
"""

import pandas as pd
from scipy.stats import linregress

# script: run after learn_network_structure to determine correlation between
# ErrorRate and ValError.  Prints r^2 of the correlation and p-value of 2-sided
# hypothesis test where the null hypothesis is no correlation.

slope,intercept,rval,pval,std_error = linregress(A_df.iloc[1].astype(float),pd.DataFrame(pd.Series(error_dict)).T.iloc[0])

print(" r^2:", rval**2, " p-value:", pval)
