import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
x=cancer.data[:,:2]
y=cancer.target

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=4)
clr=SVC()
clr.fit(x_train,y_train)
y_pred=clr.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

DecisionBoundaryDisplay.from_estimator(clr,x,response_method='predict',cmap=plt.cm.Spectral,alpha=0.8,x_label=cancer.feature_names[0],
                                       y_label=cancer.feature_names[1])                      
plt.scatter(x[:,0],x[:,1],c=y,s=20,edgecolors='k')
plt.show()

