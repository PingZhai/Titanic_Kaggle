import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from collections import Counter # Keep track of our term counts
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
data_train = pd.read_csv("D:/Kaggle/Titanic/titanic_train_modified.csv")


#features_select=['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
features_select=['Age']

features_train=data_train[features_select].values.tolist()

labels_train=data_train['Survived'].values.tolist()

X_train, X_test, y_train, y_test = train_test_split( features_train, labels_train, test_size=0.2, random_state=42)

clf = GaussianNB()
#clf = tree.DecisionTreeClassifier(min_samples_split=40)

clf.fit(X_train, y_train)
print 'aa'
y_pred=clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print accuracy


data_test = pd.read_csv("D:/Kaggle/Titanic/titanic_test_modified.csv")

features_test=data_test[features_select].values.tolist()
labels_test=clf.predict(features_test)

test = pd.DataFrame( { 'PassengerId': data_test['PassengerId'].values , 'Survived': labels_test } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )