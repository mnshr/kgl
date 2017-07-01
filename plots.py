# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 23:18:28 2017

@author: mnshr
"""

#https://www.kaggle.com/arthurtok/feature-ranking-w-randomforest-rfe-linear-models

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Pair plot
g = sns.pairplot(house[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], hue='bedrooms', palette='afmhot',size=1.4)
g.set(xticklabels=[])


# Heat Map
sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)

#https://www.kaggle.com/fhoffa/strata-london-2017-ratings-1st-exploration
# Scatter
ax = df.plot.scatter(x='rating',y='responses')
ax=df[df.starthour>0].plot.scatter(x='starthour',y='responses')

# Horiz Bar
ax = dfmc[(dfmc[('rating', 'count')]>3)].sort_values([('rating', 'mean')]).plot.barh(y=('rating', 'mean'))