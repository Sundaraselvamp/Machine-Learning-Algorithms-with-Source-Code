import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('User_Data.csv')

x=df.iloc[:,[2,3]].values
y=df.iloc[:,[4]].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
cm=accuracy_score(y_test,y_pred)
precise=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
cm_disp=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=[])
cm_disp.plot()
plt.show()

print('accuracy score:',cm)
print('precision:',precise)
print('recall:',recall)
print('f1_score:',f1)

