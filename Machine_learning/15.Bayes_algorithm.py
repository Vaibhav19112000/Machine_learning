from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split 
import pandas as pd

df=pd.read_csv(r"iris.csv")
df.isna().sum()

train,test=train_test_split(df,test_size=0.2,random_state=1)
x_train=train.drop("Species",axis=1)
y_train=train.Species
x_test=test.drop("Species",axis=1)
y_test=test.Species
gr=GaussianNB()
gr.fit(x_train,y_train)
y_pred=gr.predict(x_test)
accuracy_score(y_test, y_pred)
