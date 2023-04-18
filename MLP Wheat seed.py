#!/usr/bin/env python
# coding: utf-8

# In[3]:


# imports

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier


# In[4]:


# Loading the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
                 sep='\s+', header=None, names=['area', 'perimeter', 'compactness',
                                                'length_of_kernel', 'width_of_kernel', 'asymmetry_coefficient',
                                                'length_of_kernel_groove', 'class'])


# In[5]:


df.head()


# In[6]:


# features - X and labels -y 

X = df.drop('class', axis=1)
y = df['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Create an imputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')


# In[8]:


# Fit the imputer on the training data
imputer.fit(X_train)


# In[9]:


# Impute the missing values in the training and testing data
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)


# In[14]:


# MLP classifier

clf = MLPClassifier(solver='lbfgs', 
                    max_iter=400,
                    alpha=1e-5,
                    hidden_layer_sizes=(64,), 
                    random_state=42)


# Training the MLP classifier
clf.fit(X_train_imputed, y_train)


# In[15]:


# predictions on testing data
y_pred = clf.predict(X_test_imputed)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# ##### Results

# #### Previously three models were trained and got the following result *( the model was built on imputed data)*
# - **Decision tree** accuracy: 83.33%
# - **Random forest** accuracy: 83.33%
# - **Logistic regression** accuracy: 90.50%
# 
# Comparing this result with the present **MLP classifier** gave a better result with an accuracy score of 97.62%
# 

# >End
