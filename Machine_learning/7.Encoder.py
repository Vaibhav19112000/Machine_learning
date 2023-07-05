from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


df=pd.read_csv(r"Data.csv")


###############################Seprating data (Independent & Dependent)###############################


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      



##########################################OneHotEncoder##############################################

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder="passthrough")
x=np.array(ct.fit_transform(x))
x

##########################################LabelEncoder################################################

lb=LabelEncoder()
y=lb.fit_transform(y)
y



######################################################################################################

df1=pd.DataFrame({'city':['spain','germany','france','spain','germany','france'],'age':[89,77,56,67,66,77],'gender':['female','male','male','male','female',np.nan],'review':['good','good','bad','good','bad','good'],'education':['UG','PG','PHD','PG','UG','PHD'],'purchase':['yes','yes','no','yes','no','yes']})

#################################replace null value using SimpleImputer###############################

imp=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imp.fit(df1.iloc[:,2:3])
df1.iloc[:,1:2]=imp.transform(df1.iloc[:,2:3])


###################################### OneHotEncoder#################################################



one_hot_encoder=OneHotEncoder(drop='first',sparse=False)
x=df1.iloc[:,0:1].values
one_hot_encoder.fit(x.reshape(-1,1))
x=one_hot_encoder.transform(x.reshape(-1,1))


######################################################################################################

train,test=train_test_split(df1,test_size=0.2,random_state=1)
x_train=train.drop("purchase",axis=1)
y_train=train["purchase"]
x_test=test.drop("purchase",axis=1)
y_test=test["purchase"]


#######################################StandardScaler################################################

sc=StandardScaler()
x_train=sc.fit()
x_train=sc.transform()