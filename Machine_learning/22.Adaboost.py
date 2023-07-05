import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

income=pd.read_csv('income.csv')
income
income.isna().sum()

train,test=train_test_split(income,test_size=0.2,random_state=1)
x_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1].values

x_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1].values




ab=AdaBoostClassifier(n_estimators=100)
ab.fit(x_train, y_train)

y_pred=ab.predict(x_test)

print(accuracy_score(y_test, y_pred))
###########################################################################################
