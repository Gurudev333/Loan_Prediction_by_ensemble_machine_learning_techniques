# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:58:58 2018

@author: Gurudeo
"""

x_train = train[predictors].values
y_train = train['Loan_Status'].values
model4=sklearn.ensemble.DecisionTreeClassifier()
model4.fit(x_train,y_train)
x_test = test[predictors].values

predicted= model4.predict(x_test)
test['Loan_Status']=predicted
test.to_csv("DecisionTree.csv",columns=['Loan_ID','Loan_Status'])




##randomforest
x_train = train[predictors].values
y_train = train['Loan_Status'].values
model4=sklearn.ensemble.RandomForestClassifier()
model4.fit(x_train,y_train)
x_test = test[predictors].values

predicted= model4.predict(x_test)
test['Loan_Status']=predicted
test.to_csv("RandomForest.csv",columns=['Loan_ID','Loan_Status'])