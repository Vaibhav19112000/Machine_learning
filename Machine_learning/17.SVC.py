from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
df=pd.read_csv('Social_network_ads.csv')
lb=LabelEncoder()
df.iloc[:,1:2]=lb.fit_transform(df.iloc[:,1:2].values)


x=df.drop('Purchased',axis=1)
y=df.Purchased

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.23,random_state=1)

svc=SVC(kernel='linear',random_state=0)
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
accuracy_score(y_test, y_pred)
 