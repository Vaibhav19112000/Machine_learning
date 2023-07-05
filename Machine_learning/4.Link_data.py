###Maximum null value in current data set
##what is total item price
##what is minimum avarage,25% and 75% of item price
##Display all rows with item name startwith "Chips"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


shop=pd.read_table(r"https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/chipotle.tsv")
sns.heatmap(shop.isna(), yticklabels=False)

shop["item_price"]=shop["item_price"].str.replace("$", "",regex=True)
shop["item_price"]=shop["item_price"].astype(dtype="float64")
shop.info()
shop["item_price"].sum()
shop["item_price"].max()
shop["item_price"].min()
shop["item_price"].mean()
a=shop[shop["item_name"].str.startswith("Chips")]
shop.describe()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

shop.isna().sum()

sns.heatmap(shop.isna(), yticklabels=False)
shop['choice_description']=shop['choice_description'].fillna(shop['choice_description'].mode())
