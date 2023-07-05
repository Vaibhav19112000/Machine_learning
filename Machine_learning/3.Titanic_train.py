import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv(r"titanic_train.csv")

train.head()
train.info()
sns.heatmap(train.isna(),yticklabels=False,cbar=False,cmap="viridis")
sns.set_style("whitegrid")
sns.countplot(x="Survived", data=train)
sns.countplot(x="Survived", data=train,hue="Pclass")
sns.displot(train["Age"].dropna(),kde=False,bins=40)


def func(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if(pd.isna(Age)):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

train["Age"]=train[["Age","Pclass"]].apply(func,axis=1)

train.head(20)
train.info()
