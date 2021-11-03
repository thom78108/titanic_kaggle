# -*- coding: utf-8 -*-
"""
Created on Sat Sep 4 14:49:37 2020

@author: TXF10LQ
"""


#Import data 
import pandas as pd
import numpy as np

titanic_path = r'C:\Users\TXF10LQ\OneDrive - The Home Depot\Documents\ML\titanic\train.csv'
titanic_train = pd.read_csv(titanic_path)
titanic_path = r'C:\Users\TXF10LQ\OneDrive - The Home Depot\Documents\ML\titanic\test.csv'
titanic_test = pd.read_csv(titanic_path)
gender_submission_path = r'C:\Users\TXF10LQ\OneDrive - The Home Depot\Documents\ML\titanic\gender_submission.csv'
gender_submission = pd.read_csv(gender_submission_path)


#titanic_full = pd.concat([titanic_train, titanic_test], ignore_index= False)

####### STEP 1 APPROCHE BEGINNER ###########

#Explore the data

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
titanic_train.hist(bins=50, figsize=(40,30))
plt.show()


#Correlation - we are looking for linear correleation
corr_matrix = titanic_train.corr()
#Fare and Pclass seems to have a high influence on the survival rate - Parch and Age following


#Prepare and clean data

def preprocessing_data(X):
    #names seem hard to use - we can flag mrs-mr and mme
    List=[]
    for i in range(0,len(X["Name"])):
        if 'Miss' in X["Name"][i]:
            List.append('Miss')
        elif 'Mrs' in X["Name"][i]:
            List.append('Mrs')
        else :
            List.append('Mr')
    X["Title"] = List
    #encoding mr, mrs and miss
    X["Title_encoded"], Name_categories = X['Title'].factorize()
    
    #encode age sex
    X['Sex_encoded'] = np.where(X['Sex'] == 'male', 1,0)
    
    #missing age - replace by average
    mean_age = X['Age'].mean()
    X['Age'].fillna(value = mean_age, inplace = True)
    #missing port (2 out of 891) - replace by S since majority
    X['Embarked'].fillna(value = 'S', inplace=True)
    
    
    #Rescale Fare - two methods Min Max or Standarization 
    mean_fare = X["Fare"].mean()
    std_fare = X["Fare"].std()
    X['Fare_standarized'] = (X["Fare"] - mean_fare)/std_fare
    mean_fare_std = X['Fare_standarized'].mean()
    X['Fare_standarized'].fillna(value = mean_fare_std,inplace = True)
    
    #good but not perfect since the distance is equal between the 3 ports (0,1,2)
    Embarked_encoded, Embarked_categories = X['Embarked'].factorize()
    #one hot encoder - create 3 columns with binary value for each port
    from sklearn.preprocessing import OneHotEncoder
    cat_encoder = OneHotEncoder()
    Embarked_encoded_1hot = cat_encoder.fit_transform(Embarked_encoded.reshape(-1,1))
    Embarked_encoded_1hot
    Embarked_encoded_1hot_df = pd.DataFrame(Embarked_encoded_1hot.toarray(),columns=['Port_S','Port_C','Port_Q'])
    #add columns
    
    
    #create a new column - number member family
    X['Members_family'] = X['SibSp'] + X['Parch']
    pd.concat([X,Embarked_encoded_1hot_df],axis = 1)
    
    
    ##corr_matrix = X.corr()
    
    #We can see that Pclass,Fare,Title_Encoded,Sex_encoded are highly influent - Age, SibSp and Parch is less but we will keep them in the model
    #Cabine has a lot of null values -- probably going to remove this column
    #Name is useless now - same as ticket
    #Fare can be removed since we made standardization 
    #It seems like Members_family does make lot of influence - less infl than SibSp or Parch 
    X.drop(columns = ['Name','Cabin','Ticket','Fare','Embarked','Sex','Members_family','Title'],inplace = True)
    
    #Create final dataset for training and remove label
    X.drop(columns = ['PassengerId'],inplace = True)


preprocessing_data(titanic_train)
titanic_label = titanic_train['Survived']
titanic_train.drop(columns = ['Survived'],inplace = True)
preprocessing_data(titanic_test)
titanic_label_test = gender_submission['Survived']


#Train the model


##Linear Regression ... Do not think it is the best in this case since we are more dealing with a classification issue
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(titanic_train, titanic_label)
from sklearn.metrics import mean_squared_error
titanic_prediction = lin_reg.predict(titanic_train)
lin_mse = mean_squared_error(titanic_label,titanic_prediction)

##Logistic Regression
from sklearn.linear_model import LogisticRegression
#training on the data
log_reg = LogisticRegression()
log_reg.fit(titanic_train,titanic_label)
titanic_prediction = log_reg.predict(titanic_train)
log_mse = mean_squared_error(titanic_label,titanic_prediction)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix_train = confusion_matrix(titanic_label, titanic_prediction)
print(confusion_matrix_train)
accuracy_train =  (479+233)/891 ##0.7991

#How to improve score ? Cross validation ?##No it just helps to evaluate your model on smaller samples
from sklearn.model_selection import cross_val_score 
scores = cross_val_score(log_reg,titanic_train,titanic_label,scoring = 'accuracy',cv = 5)
print(scores.mean()) ##0.789

#Lets use test data now
titanic_prediction_test = log_reg.predict(titanic_test)
confusion_matrix_test = confusion_matrix(titanic_label_test, titanic_prediction_test)
print(confusion_matrix_test)
accuracy_test =  (260+144)/418 ##0.9665
##testing is way better than training ... what does this mean ?




##SGD Classifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=100, tol=-np.infty, shuffle = True, penalty = None, random_state=42)
sgd_clf.fit(titanic_train, titanic_label)
from sklearn.model_selection import cross_val_score
scores_sgd = cross_val_score(sgd_clf, titanic_train, titanic_label, cv=5, scoring="accuracy")
scores_sgd.mean() ##0.7812


##### It seems like SGDClassifier perform as well as logistic regression 


##Decision Tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(titanic_train, titanic_label)
scores_tree = cross_val_score(tree_clf, titanic_train, titanic_label, cv=5, scoring="accuracy")
scores_tree.mean() ##0.8103


##Random Forest
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=10, max_depth = 3, random_state=42)
rnd_clf.fit(titanic_train, titanic_label)
scores_rnd = cross_val_score(rnd_clf, titanic_train, titanic_label, cv=5, scoring="accuracy")
scores_rnd.mean() ##0.7923


##SVC 
from sklearn.svm import SVC
from sklearn import datasets
svm_clf = SVC(kernel="linear", C=10)
svm_clf.fit(titanic_train, titanic_label)
scores_tree = cross_val_score(svm_clf, titanic_train, titanic_label, cv=5, scoring="accuracy")
scores_tree.mean() ##0.7878

##use a voting clasifier 
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('svm', svm_clf), ('rf', rnd_clf), ('lg', log_reg)],
    voting='hard')
voting_clf.fit(titanic_train, titanic_label)


###let's compare the models on the test data
from sklearn.metrics import accuracy_score
for model in [svm_clf,rnd_clf,log_reg,voting_clf]:
    titanic_prediction = model.predict(titanic_test)
    print(accuracy_score(titanic_label_test, titanic_prediction))
    



###### NB : APPROCHE USING PIPLINES - GOOD FOR AUTOMATION

from sklearn.base import BaseEstimator, TransformerMixin

### Create a class to select numerical or categorical columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    

### Fill NA with median for numerical values
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

imputer = SimpleImputer(strategy="median")    


### Create the pipeline for numerical values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_attribs = ['Pclass','Age','SibSp','Parch','Fare']

num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

##titanic_num = num_pipeline.fit_transform(titanic_train)


###Pipeline for non numerical values
cat_attribs = ['Name','Sex','Ticket','Cabin','Embarked']
###Create pipeline for non numerical values
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])
    
    
titanic_cat = cat_pipeline.fit_transform(titanic_train)


##Merge the two pipelines
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


titanic_full = full_pipeline.fit_transform(titanic_train)
