from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


df=pd.read_csv(r"Data.csv")

#Seprating data (Independent & Dependent)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


#replace null value using SimpleImputer

imp=SimpleImputer(missing_values=np.nan,strategy='mean')
imp.fit(x[:,1:3])
x[:,1:3]=imp.transform(x[:,1:3])
x



df1=pd.DataFrame({'a':[1,3,4,5,6,None],'b':[3,4,5,4,6,3],'c':[2,3,45,None,None,23]})
x=df1.iloc[:,:].values
imp=SimpleImputer(missing_values=np.NAN,strategy='mean')
imp.fit(x[:,:])
x[:,:]=imp.transform(x[:,:])
x
