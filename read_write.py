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