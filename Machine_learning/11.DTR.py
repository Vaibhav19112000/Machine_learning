import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

df=pd.read_csv(r"Position_Salaries.csv")
df


x=df.Level.values.reshape(-1,1)
y=df.Salary.values.reshape(-1,1)


dt=DecisionTreeRegressor(random_state = 123434)
dt.fit(x,y)


res=dt.predict([[6.5]])

X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((-1, 1))

plt.scatter(x,y)
plt.plot(X_grid,dt.predict(X_grid))
plt.show()


df2=pd.read_csv(r'Loan_Prediction.csv')
df2
lb=LabelEncoder()
df2.iloc[:,-1]=lb.fit_transform(df2.iloc[:,-1])
train,test=train_test_split(df2,test_size=0.2,random_state=1)

x_train=df2.ApplicantIncome.values.reshape(-1,1)
y_train=df2.Loan_Status.values.reshape(-1,1)

dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)

x_test=df2.ApplicantIncome.values.reshape(-1,1)
y_test=df2.Loan_Status.values.reshape(-1,1)

y_pred=dtr.predict(x_test)

print(r2_score(y_test, y_pred))


































