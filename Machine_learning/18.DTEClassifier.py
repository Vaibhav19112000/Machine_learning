import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv('Social_network_ads.csv')
lb=LabelEncoder()
df.iloc[:,1:2]=lb.fit_transform(df.iloc[:,1:2].values)
df=df.drop(["Gender","User ID"],axis=1)

x=df.drop("Purchased",axis=1)
y=df.Purchased

sc=StandardScaler()




x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
dtr=DecisionTreeClassifier(criterion="entropy")
dtr.fit(x_train,y_train)
y_pred=dtr.predict(sc.fit_transform(x_test))


confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)



rc=RandomForestClassifier(n_estimators=10, criterion='entropy')
rc.fit(x,y)
accuracy_score(y_test, rc.predict(x_test))

