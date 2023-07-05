from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df=pd.read_csv(r"hiring.csv")
df

df.isna().sum()


sc=SimpleImputer(missing_values=np.nan,strategy='constant',fill_value='three')
df.iloc[:,0:1]=sc.fit_transform(df.iloc[:,0:1])

sc=SimpleImputer(missing_values=np.nan,strategy='mean')
df.iloc[:,1:2]=sc.fit_transform(df.iloc[:,1:2])

on=OrdinalEncoder(categories=[['two','three','five','seven', 'ten','eleven']])
on.fit(df.iloc[:,0:1])

df.iloc[:,0:1]=on.transform(df.iloc[:,0:1])



x=df.iloc[:,0:1]
y=df['salary($)']



lr=LinearRegression()

lr.fit(x,y)

y_pred=lr.predict(x)

r2_score(y, y_pred)


plt.scatter(x, y)
