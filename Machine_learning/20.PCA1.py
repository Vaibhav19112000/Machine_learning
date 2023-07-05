import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data=pd.read_csv('wine.csv')
data


x=data.iloc[:,1:].values
y=data.iloc[:,0].values


sc=StandardScaler()
x=sc.fit_transform(x)
y=sc.fit_transform(y.reshape(-1,1))


pca=PCA(n_components=3)
pca.fit_transform(x)
pca.transform(x)
pca.explained_variance_ratio_
