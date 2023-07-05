import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r"weather_data_missing.csv")

df=pd.read_csv(r"weather_data_missing.csv",parse_dates=["day"])
df.info()

df.set_index("day",inplace=True)
####### df=pd.read_csv(r"r"weather_data_missing.csv",index="day",parse_dates=["day"])
df
df.loc["2017-01-01"]
df.reset_index("day",inplace=True)
df


df.isna().sum()
df2=df.isna()
sns.heatmap(df.isna(),yticklabels=False,cmap="viridis")
sns.color_palette("rocket",as_cmap=True)
df.fillna(0)
df.fillna({"temperature":0,"windspeed":1,"event":"NO Event"})
copy_df=df.copy()

copy_df.fillna(copy_df["temperature"].median())
copy_df.fillna(copy_df["temperature"].mean())
copy_df.fillna(copy_df["temperature"].mode())
copy_df.fillna(method="ffill")
copy_df.fillna(method="bfill")

new_df=df.loc[:,"windspeed"].ffill()

copy_df.iloc[9,:].drop()


#######################################################################################################################



data=pd.read_csv(r"weather_data_missing_data.csv")
data.info()
data.replace("-99999",np.NaN)
data.replace({"temperature":"-99999","windspeed":"-99999","event":"0"},np.NaN)
data.replace({"-99999":np.nan,"0":"snow"})
data.replace({"temperature":"[A-Za-z]","windspeed":"[A-Za-z]"},"",regex=True)



#########################################################################

data1=pd.DataFrame({"score":["Exceptional","Average","Good","Poor","Average","Exceptional"],"Student":["Rob","Maya","Parthvi","Tom","Julian","Erica"]})
data1
data1.replace(["Poor","Good","Average","Exceptional"],[1,2,3,4],inplace=True)
data1["score"].nunique()
a=data1["score"].value_counts()

def func(pct,allval):
    absolute=int(pct/100.*np.sum(allval))
    return "{:.1f}%\n({:d} cl.)".format(pct,absolute)

plt.pie(a,labels=a.index,autopct=lambda pct:func(pct,a.values),explode=[0,0.2,0,0],shadow=(True))

plt.box(a)
plt.bar(a, a.index)


data1[(data1["score"]>2)]
data1.dropna(thresh=2)
