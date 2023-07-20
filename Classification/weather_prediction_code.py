#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NaiveBayes project (Weather Prediction)
#Required Modules
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


# In[2]:


#Reading CSV files
df=pd.read_csv('playsheet_dataset.csv')
df


# In[3]:


Numerics=LabelEncoder()


# In[4]:


#Dropping the target variable and make it is as newframe
inputs=df.drop('Play',axis='columns')
target=df['Play']
target


# In[5]:


#Creating the new dataframe
inputs['outlook_n']=Numerics.fit_transform(inputs['Outlook'])
inputs['Temp_n']=Numerics.fit_transform(inputs['Temp'])
inputs['Humidity_n']=Numerics.fit_transform(inputs['Humidity'])
inputs['windy_n']=Numerics.fit_transform(inputs['Windy'])
inputs


# In[ ]:





# In[6]:


#Dropping the string values
inputs_n=inputs.drop(['Outlook','Temp','Humidity','Windy'],axis='columns')
inputs_n


# In[7]:


#Applying the Gaussian naivebayes
Classifier=GaussianNB()
Classifier.fit(inputs_n,target)


# In[8]:


Classifier.score(inputs_n,target)


# In[9]:


Classifier.predict([[0,0,0,1]])


# In[10]:


# New_Outlook_data=input('Rainy Overcast Sunny:-->')
# New_Temp_data=input('Hot Mild Cool')
# New_Humidity_data=input('High Normal')
# New_Windy_data=input('f t')


# In[11]:


New_Outlook_data='Rainy'
New_Temp_data='Hot'
New_Humidity_data='High'
New_Windy_data='f'


# In[16]:



New_Outlook_data_Transform=Numerics.fit_transform([New_Outlook_data])
New_Temp_data_Transform=Numerics.fit_transform([New_Temp_data])
New_Humidity_data_Transform=Numerics.fit_transform([New_Humidity_data])
New_Windy_data_Transform=Numerics.fit_transform([New_Windy_data])


# In[23]:


prediction=Classifier.predict([[New_Outlook_data_Transform[0],New_Temp_data_Transform[0],New_Humidity_data_Transform[0],New_Windy_data_Transform[0]]])


# In[24]:


prediction


# In[ ]:




