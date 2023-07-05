import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('iris.csv')

train,test=train_test_split(df,test_size=0.2,random_state=1)


x_train=train.iloc[:,:-1]
y_train=train.iloc[:,1].values


x_test=test.iloc[:,:-1]
y_test=test.iloc[:,1].values

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


pca=PCA(n_components=2)
x_train=pca.fit_transform(x_train)
x_test=pca.fit_transform(x_test)
pca.explained_variance_ratio_
 



plt.scatter(x_train[:,0],x_train[:,1],marker="*")


sns.scatterplot(x_train[:,0],x_train[:,1],hue=train.Species)



######################################################################################












.









