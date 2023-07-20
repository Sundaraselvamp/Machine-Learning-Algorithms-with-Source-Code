import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp

data=pd.read_csv('D:\sundar\salary_data.csv')

x=data.iloc[:,:-1].values
y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)
x_pred=regression.predict(x_train)
y_pred=regression.predict(x_test)

mtp.scatter(x_train,y_train,color='blue')
mtp.plot(x_train,x_pred,color='black')
mtp.title('salary vs experience(training dataset)')
mtp.xlabel('experience')
mtp.ylabel('salary')
mtp.show()
