# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 09:29:32 2023

@author: sride
"""

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt




#reading the data from files
data=pd.read_csv('advertising.csv')
data.head()


#to visualise data
fig,axs=plt.subplots(1,3,sharey= True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])


#creating X&y for linear regression
feature_cols=['TV']
X=data[feature_cols]
y=data.Sales


#import linear regression algo
from  sklearn.linear_model  import LinearRegression
lr=LinearRegression()
lr.fit(X,y)
print(lr.intercept_)
print(lr.coef_)



result=6.96773147+0.0555374*50
print(result)

#CREATE  A DATAFRAME WITH MIN AND MAX VALUE OF THE TABLE
X_new = pd.DataFrame({'TV':[data.TV.min(), data.TV.max()]})
X_new.head()



preds = lr.predict(X_new)
preds


data.plot(kind = 'scatter',x='TV',y='Sales')

plt.plot(X_new,preds,c='red',linewidth = 3)
import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV',data = data).fit()
lm.conf_int()

#FINDING THE PROBABILITY VALUES
lm.pvalues

#FINDING THE R=SQUARED VALUES
lm.rsquared



#MULTI LINEAR REGRESSION
feature_cols = ['TV','Radio','Newspaper']
X= data[feature_cols]
y = data.Sales


lr = LinearRegression()
lr.fit(X,y)


print(lr.intercept_)
print(lr.coef_)


X_new=pd.DataFrame({'TV':[50]})
X_new.head()

lm.predict(X_new)


X_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

ram=lm.predict(X_new)
ram

data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(X_new,ram,c='red',linewidth=2)


import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales ~ TV',data=data).fit()
lm.conf_int()



lm.pvalues


lm.rsquared

feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales

lm=LinearRegression()
lm.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
lm.summary()

