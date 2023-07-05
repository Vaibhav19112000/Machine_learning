import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv(r'iris.csv')
df
df.isna().sum()
x=df.drop("Species",axis=1)
x

y=df.Species
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
for i in range(1,10):
    knn=KNeighborsClassifier(n_neighbors=i,metric="minkowski",p=2)
    knn.fit(x_train, y_train)
    y_pred=knn.predict(x_test)
    print(accuracy_score(y_test, y_pred))
