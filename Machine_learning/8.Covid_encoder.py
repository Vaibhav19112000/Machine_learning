'''
read data
divide train test 80 20
impute fever
use ordinal encoder mild strong
one hot encoder on gender and city
'''

import pandas as pd 
import seaborn as sns
import numpy as np 
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

data=pd.read_csv(r"\\192.168.1.66\Student_Data\For DBDA\Practical Machine Learning\day2\dataset\covid_toy.csv")
data
data.isna().sum()

train,test=train_test_split(data,test_size=0.2,random_state=1)
#######################split data using Train_test_split########################################
x_train=train.drop("has_covid",axis=1)
y_train=train["has_covid"]

x_test=test.drop("has_covid",axis=1)
y_test=test["has_covid"]
#######################fill null value using SimpleImputer###########################


sc=SimpleImputer(missing_values=np.nan,strategy="mean")

sc.fit(train.iloc[:,2:3])
train.iloc[:,2:3]=sc.transform(train.iloc[:,2:3])

sc.fit(test.iloc[:,2:3])
sc.transform(test.iloc[:,2:3])

##############################ordinal Encoder########################################

ordi=OrdinalEncoder(categories=[['Mild','Strong']])
x=train.iloc[:,3:4]
ordi.fit(x)
ordi.transform(x)

##############################OneHotEncoder##########################################


one_hot_encoder=OneHotEncoder(drop="first",sparse=False)
one_hot_encoder.fit(train.iloc[:,1:2])
train.iloc[:,1:2]=one_hot_encoder.transform(train.iloc[:,1:2])


one_hot_encoder=OneHotEncoder(drop="first",sparse=False)
one_hot_encoder.fit(test.iloc[:,1:2])
test.iloc[:,1:2]=one_hot_encoder.transform(test.iloc[:,1:2])

##############################################################################################















































