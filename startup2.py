# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:19:54 2020

@author: Varun
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
data=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\NEURAL NETWORKS\\50_Startups.csv")

data.head(3)
data.shape
data.columns
data.isnull().sum()
le=LabelEncoder()
data['State']=le.fit_transform(data.State)
colnames=list(data.columns)
predictors=colnames[0:4]
target=colnames[4]


def prep_model(hidden_dim):
    model = Sequential() # initialize 
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    # To define the dimensions for the output layer
    # activation - sigmoid 
    model.add(Dense(hidden_dim[-1],kernel_initializer="normal",activation="sigmoid"))
    # loss function -> loss parameter
    # algorithm to update the weights - optimizer parameter
    # accuracy - metric to display for 1 epoch
    model.compile(loss="binary_crossentropy",optimizer = "rmsprop",metrics = ["accuracy"])
    return model  

first_model = prep_model([4,50,1])


first_model.fit(np.array(data[predictors]),np.array(data[target]),epochs=10)
pred_train = first_model.predict(np.array(data[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-data[target])**2))
import matplotlib.pyplot as plt
plt.plot(pred_train,data[target],"bo")
np.corrcoef(pred_train,data[target]) # we got high correlation 
