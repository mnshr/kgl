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


# Probability Plot
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# Normal probability plot - Data distribution should closely follow the diagonal that represents the normal distribution.

from scipy import stats
res = stats.probplot(df_train['SalePrice'], plot=plt)

# https://www.kaggle.com/arthurtok/sql-and-python-primer-bokeh-and-plotly
#Bokeh
from bokeh.plotting import figure, show
from bokeh.charts import Bar
from bokeh.io import output_notebook

#Plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Strip plot
#   https://www.kaggle.com/piotrplaneta/instacart-data-analysis-on-data-sample-wip
plt.xticks(rotation='vertical')
sns.stripplot(x="department", y="reorder_ratio", data=reorder_ratio_by_department, jitter=0.2)

#https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction
#https://www.kaggle.com/plarmuseau/variance-analysis-ii
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(12, 12))
ax = Axes3D(fig, elev=-150, azim=110)