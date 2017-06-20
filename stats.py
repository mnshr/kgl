# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 07:39:26 2017

@author: mnshr
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
VIF = 1 (Not correlated) 1 < VIF < 5 (Moderately correlated) VIF > 5 to 10 (Highly correlated)

VIF is one way to understand whether any two independent variable are highly correlated.
In that case both the variable explain same variance in the model.
So it is generally good to drop any one of them.
Feature selection vs Multicollinearity Checks Vs Mean Centering (Standardization)

Answer by Robert Kubrick in the following link give some interesting information
on the front
https://stats.stackexchange.com/questions/25611/how-to-deal-with-multicollinearity-when-performing-variable-selection
"""
def calculate_vif_(X):
    variables = list(X.columns)
    vif = {variable:variance_inflation_factor(exog=X.values, exog_idx=ix) for ix,variable in enumerate(list(X.columns))}
    return vif


from scipy.stats import kendalltau