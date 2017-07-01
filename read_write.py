# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 22:54:57 2017

@author: mnshr
"""

import pandas as pd

data_file ='abc_data.csv'

# Read from a json file
train_df = pd.read_json("../input/train.json")
train_df.head()


# Dump to a json file
json_df = pd.read_csv(data_file)
json_df.to_json('./data.json',orient='index')


#https://www.kaggle.com/fhoffa/strata-london-2017-ratings-1st-exploration
# Read in CSV to dataframe
df = pd.read_csv('../input/Strata London 2017 ratings - results-20170630-170827.csv.csv')
df['responses'] = df['responses'].fillna(0).astype(int)
df['starthour'] = df['starthour'].fillna(np.nan).astype(int)