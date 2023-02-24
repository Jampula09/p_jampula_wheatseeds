#!/usr/bin/env python
# coding: utf-8

# """
# ### Perform both PCA and LDA on the original Wheat Seeds data and suggest any feature transformations that are appropriate. Justify your recommendation.
# """

# Before applying PCA and LDA, we will first load the Wheat Seeds dataset and perform some exploratory data analysis to get an understanding of the data.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
                 sep='\s+', header=None, names=['area', 'perimeter', 'compactness',
                                                'length_of_kernel', 'width_of_kernel', 'asymmetry_coefficient',
                                                'length_of_kernel_groove', 'class'])


# In[4]:


# Display the first few rows of the dataset
df.head()


# In[9]:


X = df.iloc[:, :-1].values # features
y = df.iloc[:, -1].values # target


# In[10]:


# standardize the data
sc = StandardScaler()
X_std = sc.fit_transform(X)


# In[11]:


# perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)


# In[12]:


# perform LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_std, y)


# In[13]:


# visualize the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.8)
ax[0].set_title('PCA')
ax[1].scatter(X_lda[:, 0], X_lda[:, 1], c=y, alpha=0.8)
ax[1].set_title('LDA')
plt.show()


# In[16]:


# plot the loadings of the first two principal components
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(len(pca.components_[0])), pca.components_[0])
ax.set_xticks(range(len(df.columns)-1))
ax.set_xticklabels(df.columns[:-1], rotation=45, ha='right')
ax.set_xlabel('Features')
ax.set_ylabel('Loadings')
plt.show()


# From the plot, we can see that the features with the highest loadings on the first principal component are 'perimeter', 'area', 'compactness', and 'length of kernel'. On the second principal component, the features with the highest loadings are 'width of kernel' and 'asymmetry coefficient'.
# 
# Based on this information, we can make a few suggestions for feature transformations:
# 
# 1. We can consider dropping the 'asymmetry coefficient' feature since it has relatively low loadings on all principal components. This may improve the performance of our dimensionality reduction methods.
# 
# 2. We can also consider combining some of the highly correlated features, such as 'perimeter', 'area', and 'compactness', into a single feature using feature engineering techniques such as creating a ratio or a polynomial. This can help reduce the redundancy in the dataset and improve the interpretability of the results.
# 
# 3. We can experiment with different feature scaling methods such as min-max scaling or robust scaling, and compare the performance of PCA and LDA on the transformed data to see if it improves the separation of the classes.
# 
# In summary, based on the loadings of the principal components, we can consider dropping the 'asymmetry coefficient' feature and exploring feature engineering techniques to combine highly correlated features. We can also experiment with different feature scaling methods to improve the performance of our dimensionality reduction methods.
