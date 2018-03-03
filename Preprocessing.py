# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 23:00:11 2018

@author: Gurudeo
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sklearn.linear_model
import sklearn.ensemble
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier



#step 1:Check missing values
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

#number of missing value in trian and test data
test_Missing = test.isnull().sum()
train_Missing=train.isnull().sum()
#Missing value treatement for train
#Gender ::impute gender which has maximum count
Gender_train_count=train['Gender'].value_counts()
train['Gender'].fillna('Male',inplace=True)
Gender_test_count=test['Gender'].value_counts()
test['Gender'].fillna('Male',inplace=True)
#recheck
temp_test_Missing = test.isnull().sum()
temp_train_Missing=train.isnull().sum()
#Married
Married_train_count=train['Married'].value_counts()
train['Married'].fillna('Yes',inplace=True)
Married_test_count=test['Married'].value_counts()
test['Married'].fillna('Yes',inplace=True)
#recheck
temp_test_Missing = test.isnull().sum()
temp_train_Missing=train.isnull().sum()

#Self_Employed
#Lets find relationship between Education and Self_Education
twowaytable = pd.crosstab(train ["Education"], train ["Self_Employed"], margins=True)
#from twowaytable it is found strong relationship between Education and Self Employment
#thus map relationship between Education and 
d={'Graduate':'No'}
s=train.Education.map(d)
train.Self_Employed=train.Self_Employed.combine_first(s)
#non_Graduate
d={'Not Graduate':'Yes'}
s=train.Education.map(d)
train.Self_Employed=train.Self_Employed.combine_first(s)
#Repeate process for test also
d={'Graduate':'No'}
s=test.Education.map(d)
test.Self_Employed=test.Self_Employed.combine_first(s)
#non_Graduate
d={'Not Graduate':'Yes'}
s=test.Education.map(d)
test.Self_Employed=test.Self_Employed.combine_first(s)
#recheck
temp_test_Missing = test.isnull().sum()
temp_train_Missing=train.isnull().sum()

#Dependant
Dependant_train_count=train['Dependents'].value_counts()
train['Dependents'].fillna('0',inplace=True)
Dependents_test_count=test['Dependents'].value_counts()
test['Dependents'].fillna('0',inplace=True)
#recheck
temp_test_Missing = test.isnull().sum()
temp_train_Missing=train.isnull().sum()

#find Applicant Income and LoanAmount relation using scatter plot
plt.scatter(train['ApplicantIncome'],train['LoanAmount'])
#Find relationship between LoanAmount and Educaation
plt.scatter(train['Education'],train['LoanAmount'])
#Impute 350 for Graduates and 125 for Non Graduate
d={'Graduate': 350}
s=train.Education.map(d)
train.LoanAmount=train.LoanAmount.combine_first(s)
d={'Not Graduate': 125}
s=train.Education.map(d)
train.LoanAmount=train.LoanAmount.combine_first(s)
#test
d={'Graduate': 350}
s=test.Education.map(d)
test.LoanAmount=test.LoanAmount.combine_first(s)
d={'Not Graduate': 125}
s=test.Education.map(d)
test.LoanAmount=test.LoanAmount.combine_first(s)
#recheck
temp_test_Missing = test.isnull().sum()
temp_train_Missing=train.isnull().sum()
#LoanAmountTerm
Loan_Amount_Term_train_count=train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(360.0,inplace=True)
Loan_Amount_Term_test_count=test['Loan_Amount_Term'].value_counts()
test['Loan_Amount_Term'].fillna(360.0,inplace=True)
temp_test_Missing = test.isnull().sum()
temp_train_Missing=train.isnull().sum()
#Credit History
Credit_History_train_count=train['Credit_History'].value_counts()
train['Credit_History'].fillna(1.0,inplace=True)
Credit_History_test_count=test['Credit_History'].value_counts()
test['Credit_History'].fillna(1.0,inplace=True)
temp_test_Missing = test.isnull().sum()
temp_train_Missing=train.isnull().sum()
#Step  2::Categorical into contineous
number = LabelEncoder()
#Gender
train['Gender'] = number.fit_transform(train['Gender'].astype(str))
test['Gender'] = number.fit_transform(test['Gender'].astype(str))
#Married
train['Married'] = number.fit_transform(train['Married'].astype(str))
test['Married'] = number.fit_transform(test['Married'].astype(str))
#Self_Employed
train['Self_Employed'] = number.fit_transform(train['Self_Employed'].astype(str))
test['Self_Employed'] = number.fit_transform(test['Self_Employed'].astype(str))
#Education
train['Education'] = number.fit_transform(train['Education'].astype(str))
test['Education'] = number.fit_transform(test['Education'].astype(str))
#Property_Area
train['Property_Area'] = number.fit_transform(train['Property_Area'].astype(str))
test['Property_Area'] = number.fit_transform(test['Property_Area'].astype(str))
#Dependents
train['Dependents'] = train['Dependents'].str.rstrip('+')
test['Dependents'] = test['Dependents'].str.rstrip('+')
train['Dependents'] = number.fit_transform(train['Dependents'].astype(str))
test['Dependents'] = number.fit_transform(test['Dependents'].astype(str))
##check all observations
train_Missing=train.isnull().sum()
test_Missing=test.isnull().sum()



train.head()
test.head()
#create extra varible
#TotalIncome
train['TotalIncome']=train['ApplicantIncome']+train['CoapplicantIncome']
test['TotalIncome']=test['ApplicantIncome']+test['CoapplicantIncome']
plt.hist(train['TotalIncome'],bins=20)
#check normality of distribution 
train['TotalIncome'].mode()
train['TotalIncome'].mean() 
train['TotalIncome'].median()
train['TotalIncomelog']=np.log(train['TotalIncome'])
test['TotalIncomelog']=np.log(test['TotalIncome'])
#showa stable distribution
train['TotalIncomelog'].mode()
train['TotalIncomelog'].mean() 
train['TotalIncomelog'].median()
#
test['TotalIncomelog'].mode()
test['TotalIncomelog'].mean() 
test['TotalIncomelog'].median()
#Repeate same and recheck values for the LoanAmount
train['LoanAmountlog']=np.log(train['LoanAmount'])
test['LoanAmountlog']=np.log(test['LoanAmount'])

#

##Add some important features
 #TotalIncome/LoanAmount
train['TotalIncomeLoanAmountratio']=train['LoanAmount']/train['TotalIncome']
test['TotalIncomeLoanAmountratio']=test['LoanAmount']/test['TotalIncome']
#ApplicantIncome/LoanAmount
train['ApplicantIncomeLoanAmountratio']=train['LoanAmount']/train['ApplicantIncome']
test['ApplicantIncomeLoanAmountratio']=test['LoanAmount']/test['ApplicantIncome']
##Loan_Status Allinment adjacement
#train
temp=train['Loan_Status']
train=train.drop('Loan_Status',axis=1)
train['Loan_Status']=temp.values
train['Loan_Status'] = number.fit_transform(train['Loan_Status'].astype(str))

#test
test=test.assign(LoanStatus=np.nan)
##finding important features for model
########predictors=['Credit_History','TotalIncomeLoanAmountratio','TotalIncomelog','TotalIncome','Property_Area','ApplicantIncomeLoanAmountratio','LoanAmount']
#predictors and outcomes to numpy array
# Converting predictors and outcome to numpy array
train['TotalIncomelog']=train['TotalIncomelog'].astype(float)
test['TotalIncomelog']=test['TotalIncomelog'].astype(float)
#
train['TotalIncomeLoanAmountratio']=train['TotalIncomeLoanAmountratio'].astype(float)
test['TotalIncomeLoanAmountratio']=test['TotalIncomeLoanAmountratio'].astype(float)
#
train['ApplicantIncomeLoanAmountratio']=train['ApplicantIncomeLoanAmountratio'].astype(float)
test['ApplicantIncomeLoanAmountratio']=test['ApplicantIncomeLoanAmountratio'].astype(float)
predictors=['Credit_History','TotalIncomelog','TotalIncome','TotalIncomeLoanAmountratio','LoanAmount','Property_Area']
