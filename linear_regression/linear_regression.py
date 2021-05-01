# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:38:49 2021

@author: Hp
"""
#%% İMPORT lib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%#%% İMPORT data
df=pd.read_csv("linear_regression_dataset.csv",sep=";")

#%%
#görselleştirme
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()
#%%
#linear regression için fit line bulacagız.
#line y=b0+b1*x b0=constant(bias) , b1=coeff yani b1=eğim
#maas=b0+b1*deneyim
#amacımız mean square error un en az olacak sekilde line olusturmak gerekir.
# amac min(MSE)
#%% sklearn  line fit
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()
x=df.deneyim.values
y=df.maas.values
#%% shape bize (14,) verir ama sklearn bunu anlamaz o yuzden (14,1) e cevirecegiz
x.shape
y.shape
x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)
#%% linear regression artık fit edildi
linear_reg.fit(x,y)
b0=linear_reg.predict([[0]])
#veya b0=linear_reg.intercept_ ikidi aynı sey
print(b0)

b1=linear_reg.coef_
print(b1)
print(linear_reg.predict([[11]]))

#%%simdi line mizi gorelim
array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
plt.scatter(x,y)
plt.show()

y_head=linear_reg.predict(x)
plt.plot(array,y_head,color="red")


#%%
from sklearn.metrics import r2_score
print("r_square score: ",r2_score(y,y_head))





