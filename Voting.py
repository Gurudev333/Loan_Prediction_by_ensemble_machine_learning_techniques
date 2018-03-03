# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 22:55:38 2018

@author: Gurudeo
"""

predictors=['Credit_History','TotalIncomelog','TotalIncome','TotalIncomeLoanAmountratio','LoanAmount','Property_Area']
x_train = train[predictors].values
y_train = train['Loan_Status'].values
estimators = []

#
model1 = sklearn.linear_model.LogisticRegression()
estimators.append(('logistic', model1))

model1.fit(x_train, y_train)
x_test = test[predictors].values

predct= model1.predict(x_test)
#encoding interconversion
test['Loan_Status']=predct

#write outputs
test.to_csv("Logistic.csv",columns=['Loan_ID','Loan_Status'])

#Random Forest
#predictors=['Credit_History','TotalIncomelog','Education','Property_Area']



#SVM
model2 = SVC()
estimators.append(('svm', model2))

model2.fit(x_train, y_train)
x_test = test[predictors].values

predct= model2.predict(x_test)
#encoding interconversion
test['Loan_Status']=predct

#write outputs
test.to_csv("SVM.csv",columns=['Loan_ID','Loan_Status'])

#DecisionTreeClassifier
model3 = DecisionTreeClassifier()
estimators.append(('cart', model3))

model3.fit(x_train, y_train)
x_test = test[predictors].values

predct= model3.predict(x_test)
#encoding interconversion
test['Loan_Status']=predct

#write outputs
test.to_csv("DecisionTreeClassifier.csv",columns=['Loan_ID','Loan_Status'])



#ensemble


ensemble = VotingClassifier(estimators)

ensemble.fit(x_train, y_train)
x_test = test[predictors].values

predct= ensemble.predict(x_test)
#encoding interconversion
test['Loan_Status']=predct

#write outputs
test.to_csv("ensemble.csv",columns=['Loan_ID','Loan_Status'])
