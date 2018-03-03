# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:49:39 2018

@author: Gurudeo
"""

#XGBoost

##Adaboost
x_train = train[predictors].values
y_train = train['Loan_Status'].values
num_trees=30
model5 = AdaBoostClassifier(n_estimators=num_trees)
model5.fit(x_train,y_train)
x_test = test[predictors].values

predicted= model5.predict(x_test)
test['Loan_Status']=predicted
test.to_csv("AdaBoost.csv",columns=['Loan_ID','Loan_Status'])

##Gradient Boost
x_train = train[predictors].values
y_train = train['Loan_Status'].values
num_trees=30
model5 = GradientBoostingClassifier(n_estimators=num_trees)
model5.fit(x_train,y_train)
x_test = test[predictors].values

predicted= model5.predict(x_test)
test['Loan_Status']=predicted
test.to_csv("GradientBoost.csv",columns=['Loan_ID','Loan_Status'])

