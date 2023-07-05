import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score,r2_score


data=pd.read_csv('iris.csv')
data

le=LabelEncoder()
le.fit_transform(data.iloc[:,-1])
x=data.iloc[:,:-1]
y=data.iloc[:,-1]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


lda=LDA(n_components=2)
x_train=lda.fit_transform(x_train, y_train)
x_test=lda.fit_transform(x_test,y_test)
'''
Qda=QDA()
x_train=Qda.fit(x_train, y_train)
x_test=Qda.fit(x_test,y_test)
'''
lda.explained_variance_ratio_


ls=LogisticRegressionCV()
ls.fit(x_train,y_train)
ypred=ls.predict(x_test)


accuracy_score(y_test, ypred)
