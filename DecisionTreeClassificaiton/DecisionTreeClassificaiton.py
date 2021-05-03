# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:06:27 2021

@author: Hp
"""

#%%
import numpy as np
import pandas as pd

#%%
data=pd.read_csv("data.csv")
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

#%%
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)
#%%
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#%%
#iyi huylu ve kötü huylu diye iki tane sınıftan olusan 30 tane farklı futurları olab bir datasettir
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=42)
#%%
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("score",dt.score(x_test,y_test))
#%%
y_pred=dt.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y pred")
plt.ylabel("y true")
plt.show()









