# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:04:51 2020

@author: Varun
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\NEURAL NETWORKS\\forestfires.csv")

le=LabelEncoder()
data['month']=le.fit_transform(data.month)
data['day']=le.fit_transform(data.day)
data.loc[data.size_category=="small","size_category"]=0
data.loc[data.size_category=="large","size_category"]=1

##pllot of no of o &1
data.size_category.value_counts().plot(kind="bar")

train,test=train_test_split(data,test_size=0.3,random_state=0)
trainx= train.drop(["size_category"],axis=1)
trainy= train['size_category']

testx= test.drop(['size_category'],axis=1)
testy=test['size_category']


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




first_model = prep_model([30,50,1])
first_model.fit(np.array(trainx),np.array(trainy),epochs=10)
pred_train = first_model.predict(np.array(trainx))
pred_train = pd.Series([i[0] for i in pred_train])

size_category=["small","large"]
pred_train_class = pd.Series(["small"]*361)
pred_train_class[[i==1 for i in pred_train]] = "large"




from sklearn.metrics import confusion_matrix
train["original_class"] = "small"
train.loc[train.size_category==1,"original_class"] = "large"
train.original_class.value_counts()

# Two way table format 
confusion_matrix(pred_train_class,train.original_class)
np.mean(pred_train_class==pd.Series(train.original_class).reset_index(drop=True))
####72.02% accuracy



###for test data
pred_test = first_model.predict(np.array(testx))

pred_test = pd.Series([i[0] for i in pred_test])
pred_test_class = pd.Series(["small"]*156)

pred_test_class[[i==1 for i in pred_test]] = "large"
test["original_class"] = "small"
test.loc[test.size_category==1,"original_class"] = "large"
test.original_class.value_counts()
temp = pd.Series(test.original_class).reset_index(drop=True)
np.mean(pred_test_class==pd.Series(test.original_class).reset_index(drop=True)) # 97.66
len(pred_test_class==pd.Series(test.original_class).reset_index(drop=True))
confusion_matrix(pred_test_class,temp)
##70% accuracy

from keras.utils import plot_model
plot_model(first_model,to_file="first_model.png")
