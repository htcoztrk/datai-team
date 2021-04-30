# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:40:00 2021

@author: Hp
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
df=pd.read_csv("polynomial+regression.csv",sep=";")

x=df.araba_max_hiz.values.reshape(-1,1)
y=df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba_max_hiz")
plt.ylabel("araba_fiyat")
plt.show()
#%%
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
#%%
y_head=lr.predict(x)
plt.plot(x,y_head,color="red")
plt.show()
#%%
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=4)

x_poly=poly.fit_transform(x)

#%%
linear_reg2=LinearRegression()
linear_reg2.fit(x_poly,y)


#%%
plt.scatter(x,y)
plt.xlabel("araba_max_hiz")
plt.ylabel("araba_fiyat")

y_head2=linear_reg2.predict(x_poly)

plt.plot(x,y_head2,color="green",label="poly")
plt.legend()
plt.show()















