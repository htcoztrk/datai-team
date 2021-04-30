# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:07:29 2021

@author: Hp
"""
#multiple_linear_regression
#bir y eksenine birden cok özelliklerin etki etmesidir
#%% y=b0+b1*x1+b2*x2...

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#%%
df=pd.read_csv("multiple_linear_regression_dataset.csv",sep=";")
x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)
#%%
m_l_r=LinearRegression()
m_l_r.fit(x,y)

#%%
m_l_r.intercept_ #b1
m_l_r.coef_#b1,b2
#%%
m_l_r.predict(np.array([[10,23],[3,21]]))