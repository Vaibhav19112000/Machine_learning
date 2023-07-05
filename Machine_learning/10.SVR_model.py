import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt


df=pd.read_csv(r"Position_Salaries.csv")
df

x=df['Level'].values
y=df['Salary'].values

#sc_x=StandardScaler()
sc_y=StandardScaler()


x=sc_y.fit_transform(x.reshape(-1,1))

y=sc_y.fit_transform(y.reshape(-1,1))


#lr=LinearRegression()
reg=SVR(kernel='rbf')

reg.fit(x,y)


res=reg.predict(sc_y.fit_transform([[7.8]]))
sc_y.inverse_transform(res.reshape(-1,1))


plt.scatter(x,y)
plt.plot(x,reg.predict(x),color='black')
plt.show()
