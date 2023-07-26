# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:52:41 2023

@author: sride
"""

import pylab as pl
import numpy as np
import pandas as pd
import statsmodels.api as sm
 


df=pd.read_csv('binary.csv')

df.head()
df.columns=['admit','gre','gpa','prestige']
df.head()


pd.crosstab(df['admit'],df['prestige'],rownames=['admit'])

df.hist()

pl.show()


dummy_ranks=pd.get_dummies(df['prestige'],prefix='prestige')

dummy_ranks.head()



cols_to_keep=['admit','gre','gpa']
data=df[cols_to_keep].join(dummy_ranks.loc[:,'prestige_2':])
data.head()


data['intercept']=1.0
data.head()

train_cols=data.columns[1:]
logit=sm.Logit(data['admit'],data[train_cols])

results=logit.fit()

iroman=results.predict([800,4,0,0,0,1.0])
print(iroman)

results.summary()







