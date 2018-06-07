"""Project - Predict Salary based on Position.
   Rajat Singhal : singhal.rajat97@gmail.com
   B.Tech 4th Semester, Computer Science & Engineering"""

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 5].values

#encode categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#split dataset in training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predict test set
y_pred = map(round , regressor.predict(X_test))
y_pred = map(int , y_pred)
print(list(y_test))
print(list(y_pred))

"""
#Backward ELimination
import statsmodels.formula.api as sm
X=np.append(arr = np.ones((150,1)).astype(int),values = X,axis =1)
X_opt = X[ : , [0,2,3,4]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
#Result: removing 1st coloumn as p>sl(0.5) dosent effect result too much
"""
