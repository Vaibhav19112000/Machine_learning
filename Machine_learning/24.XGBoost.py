import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df=pd.read_csv('diabetes.csv')
df.isna().sum()
df.head()


x=df.drop("Outcome",axis=1)
y=df.Outcome


sc=StandardScaler()
x=sc.fit_transform(x)

x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape
X_train=x_train[:400]
X_test=x_train[400:]

y_train.shape

Y_train=y_train[:400]
Y_test=y_train[400:]


xgb=XGBClassifier(eta=0.01,gamma=3)
xgb.fit(X_train,Y_train)
xgb.score(X_train,Y_train)
xgb.score(x_val,y_val)
xgb.score(x_test,y_test)
 