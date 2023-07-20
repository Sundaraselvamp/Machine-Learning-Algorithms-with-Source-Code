#!/usr/bin/env python
# coding: utf-8

# ## KNN classification Code

# In[1]:


import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')
df_model = df.copy()

#Create KNN Object
knn = KNeighborsClassifier()

#Create x and y variable
x = df_model.drop(columns=['target'])
y = df_model['target']

#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#Training the model
knn.fit(x_train, y_train)

#Predict testing set
y_pred = knn.predict(x_test)

#Check performance using accuracy
print('KNN algorithm accuracy score:-->',accuracy_score(y_test, y_pred))


# ### As you can see there are 5 numerical features that have different units. They are age, trestbps, chol, thalach, and oldpeak.
# 
# 
# ## First Rescaling uses Standard Scaling using Scikit-Learn Library StandardScaler()

# In[7]:



import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('heart.csv')

#Create copy of dataset.
df_model = df.copy()

#Rescaling features age, trestbps, chol, thalach, oldpeak.
scaler = StandardScaler()

features = [['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]

for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])
    
#Create KNN Object
knn = KNeighborsClassifier()

#Create x and y variable
x = df_model.drop(columns=['target'])
y = df_model['target']

#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#Training the model
knn.fit(x_train, y_train)

#Predict testing set
y_pred = knn.predict(x_test)

#Check performance using accuracy
print('KNN with standard scaler algorithm accuracy score:-->',accuracy_score(y_test, y_pred))


# ## Hyper parameter Tunning with Grid search CV

# In[9]:



import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('heart.csv')

#Create copy of dataset.
df_model = df.copy()

#Rescaling features age, trestbps, chol, thalach, oldpeak.
scaler = StandardScaler()

features = [['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]

for feature in features:
    df_model[feature] = scaler.fit_transform(df_model[feature])
    
#Create KNN Object
knn = KNeighborsClassifier()

#Create x and y variable
x = df_model.drop(columns=['target'])
y = df_model['target']

#Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

parameters = {'n_neighbors':[4,5,6,7],
              'leaf_size':[1,3,5],
              'algorithm':['auto', 'kd_tree'],
              'n_jobs':[-1]}

#Fit the model
model = GridSearchCV(knn, param_grid=parameters)
model.fit(x_train,y_train)
# model.cv_results_
print("Best Params", model.best_params_)
print("Best CV Score", model.best_score_)
print(f'Accuracy on Model 1 = {round(accuracy_score(y_test, model.predict(x_test)), 5)}')


# In[7]:


from sklearn.model_selection import RandomizedSearchCV

param_grid=parameters
grid_cv = RandomizedSearchCV(knn, param_grid, scoring="accuracy", n_jobs=-1, cv=3)
grid_cv.fit(x_train, y_train)

print("Best Params", grid_cv.best_params_)
print("Best CV Score", grid_cv.best_score_)
print(f'Accuracy on Model 1 = {round(accuracy_score(y_test, grid_cv.predict(x_test)), 5)}')


# In[ ]:




