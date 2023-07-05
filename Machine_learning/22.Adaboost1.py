import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier 
income=pd.read_csv('income.csv')
income
income.isna().sum()


train,test=train_test_split(income,test_size=0.2,random_state=1)
x_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1].values

x_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1].values

svc=SVC(probability=True,kernel='rbf')


ab=AdaBoostClassifier(n_estimators=50,base_estimator=svc,)
ab.fit(x_train,y_train)
y_pred=ab.predict(x_test)
