import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt



data= pd.read_csv(r"homeprices.csv")
data

train,test=train_test_split(data,test_size=0.2,random_state=1)
x_train=train.drop('price',axis='columns')
y_train=train.price


lr=LinearRegression()
lr.fit(x_train,y_train)

x_test=test.drop('price',axis='columns')
y_test=test.price


y_train_p=lr.predict(x_train)
r2_score(y_train, y_train_p)

y_test_p=lr.predict(x_test)
r2_score(y_test, y_test_p)


Ri=Ridge(alpha=0.1)
Ri.fit(x_train,y_train)
ypred=Ri.predict(x_test)

r2_score(y_test, ypred)
