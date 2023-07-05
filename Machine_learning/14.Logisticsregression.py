import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
df=pd.read_csv(r'iris.csv')
df
x=df.where(df['Species']!="setosa")
x=x.dropna()

train,test=train_test_split(x,test_size=0.25,random_state=1)
x_train=train.drop('Species',axis=1)
y_train=train.Species

x_test=test.drop('Species',axis=1)
y_test=test.Species


lr=LogisticRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

accuracy_score(y_test, y_pred)
