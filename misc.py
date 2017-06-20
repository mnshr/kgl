# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 07:42:12 2017

@author: mnshr
"""


import missingno as msno

# Find and plot missing value columns
missingValueColumns = merged.columns[merged.isnull().any()].tolist()
msno.bar(merged[missingValueColumns],\
            figsize=(20,8),color="#34495e",fontsize=12,labels=True,)