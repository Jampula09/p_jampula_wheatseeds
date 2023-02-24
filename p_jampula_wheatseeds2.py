#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# """
# 
# ### Write a randomizer program that selectively randomizes a percentage of the wheat seed features (a cell -- not a column or row).
# 
# """

# In[3]:


# Load the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
                 sep='\s+', header=None, names=['area', 'perimeter', 'compactness',
                                                'length_of_kernel', 'width_of_kernel', 'asymmetry_coefficient',
                                                'length_of_kernel_groove', 'class'])

def randomize_cell(df, row_idx, col_idx, percentage):
    """
    selectively randomizes a percentage of the wheat seed features
    
    """
    val = df.iloc[row_idx, col_idx]
    rand_range = val * percentage / 100
    new_val = np.random.uniform(val - rand_range, val + rand_range)
    df.iloc[row_idx, col_idx] = new_val
    return df


# In[4]:


randomize_cell(df, 10, 3, 20)


# """
# 
# #### Select a learning algorithm or algorithms and use the imputing solutions from scikit learn to compare results from the unaffected original data.
# 
# """

# In[5]:


# Considering missing values 


# Load the dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
                 sep='\s+', header=None, names=['area', 'perimeter', 'compactness',
                                                'length_of_kernel', 'width_of_kernel', 'asymmetry_coefficient',
                                                'length_of_kernel_groove', 'class'])


# In[6]:


# Split into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[7]:


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Create an imputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')


# In[9]:


# Fit the imputer on the training data
imputer.fit(X_train)


# In[10]:


# Impute the missing values in the training and testing data
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)


# In[11]:


# decision tree model
dt = DecisionTreeClassifier(random_state=42)

# Fit the model on the imputed training data
dt.fit(X_train_imputed, y_train)

# Evaluate the model on the imputed testing data
score_dt = dt.score(X_test_imputed, y_test)
print(f"Decision tree accuracy on imputed data: {score_dt:.3f}")


# In[12]:


# random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the imputed training data
rf.fit(X_train_imputed, y_train)

# Evaluate the model on the imputed testing data
score_rf = rf.score(X_test_imputed, y_test)
print(f"Random forest accuracy on imputed data: {score_rf:.3f}")


# In[13]:


# logistic regression model
lr = LogisticRegression(max_iter=1000, random_state=42)

# Fit the model on the imputed training data
lr.fit(X_train_imputed, y_train)

# Evaluate the model on the imputed testing data
score_lr = lr.score(X_test_imputed, y_test)
print(f"Logistic regression accuracy on imputed data: {score_lr:.3f}")


# ##### Logistic regression is giving a better accuracy 
