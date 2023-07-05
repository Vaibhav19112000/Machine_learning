import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

def crossValid (estimator,x,y,cv): 
    cv=cross_val_score(estimator=estimator, X=x,y=y,cv=cv)
    print(cv.mean()*100)
    print(cv.std()*100)
    print("---------------------")

df=pd.read_csv('wine.csv')
df.isna().sum()
train,test=train_test_split(df,test_size=0.2,random_state=1)

x_train=train.drop("Class",axis=1)
y_train=train.Class

x_test=test.drop("Class",axis=1)
y_test=test.Class

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
svc=SVC(kernel='rbf')
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

r2_score(y_test, y_pred)

for i in range(2,10):
    crossValid(svc,x_train,y_train,i)


######################################################################################



df1=pd.read_csv('Cancer.csv')
df1['age'].unique()


oe=OrdinalEncoder(categories=[['20-29','30-39','40-49', '50-59', '60-69', '70-79']])
df1.iloc[:,1:2]=oe.fit_transform(df1.iloc[:,1:2])

le=LabelEncoder()
df1.iloc[:,2:3]=le.fit_transform(df1.iloc[:,2:3])
df1.iloc[:,5:6]=le.fit_transform(df1.iloc[:,5:6])
df1.iloc[:,8:11]=le.fit_transform(df1.iloc[:,8:11])

train,test=train_test_split(df1,test_size=0.2,random_state=1)
x_train=train.drop('Class',axis=1)
y_train=train.Class

svc1=SVR()
svc1.fit(x_train,y_train)
