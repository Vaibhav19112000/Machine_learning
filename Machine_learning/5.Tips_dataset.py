'''Use tips dataset from seaborn, and display graph to show male and female smokers and non 
smokers
a. Find daywise min, max, average, and 25 and 75 percentile of total bill
b. On every data how many males and how many females go to hotel
c. On which day number of visitors are maximum and on which day the number of visitors 
are min.'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tip=sns.load_dataset(("tips"))
tip.isna().sum()
tip.describe()
tip['sex'].value_counts(0)
a=tip[tip['size']==tip['size'].max()]
a
b=a['day'].unique()
for i in b:
    print(i)

a=tip[tip['size']==tip['size'].min()]
b=a['day'].unique()
b
for i in b:
    print(i)


sns.heatmap(tip.isna(),yticklabels=False,cmap='viridis')
tip['day'].value_counts()


a=dict(tip['day'].value_counts())
a
*
