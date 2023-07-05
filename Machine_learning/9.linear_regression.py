import pandas as pd
import numpy as np
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

y_pred=lr.predict(x_test)
print(r2_score(y_test, y_pred))
lr.coef_
lr.intercept_


plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.show()
##############################Example 1###############################################################

data2= pd.read_csv(r"canada_per_capita_income.csv")
data2



train,test=train_test_split(data2,test_size=0.2,random_state=1)

x_train=train.drop('per capita income (US$)',axis='columns')

y_train=train['per capita income (US$)']


lr=LinearRegression()
lr.fit(x_train,y_train)


x_test=test.drop('per capita income (US$)',axis='columns')
y_test=test['per capita income (US$)']

y_pred=lr.predict(x_test)

r2_score(y_test, y_pred)



























