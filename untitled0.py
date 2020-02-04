# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:38:48 2020

@author: BelluPr
"""

# importing lib
import numpy as np
import pandas as pd
import matplotlib as plt

# importing data set

Train=pd.read_csv("train.csv")
Test=pd.read_csv("test.csv")

TrainX=Train.iloc[:,1:].values
TrainY=Train.iloc[:,0].values

TestX=Test.iloc[:,:].values

#TrainY=Train[:,1]
#model building

#since it is a classification problem we need to use classification algorithm to get result
#here depenedent variable Y has 4 o/p values or leveles

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

Lable_Y=LabelEncoder()
TrainY=Lable_Y.fit_transform(TrainY)
# now the TrainY is having the the values in numbers





from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(TrainX,TrainY)

TestY=LR.predict(TestX)
np.savetxt("Results.csv",TestY)




