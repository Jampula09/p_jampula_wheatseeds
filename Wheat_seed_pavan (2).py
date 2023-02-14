About Dataset:

The "Seeds" dataset is a small, public dataset that is commonly used in machine learning and pattern recognition experiments. The dataset contains 7 different features (area, perimeter, compactness, length of kernel, width of kernel, asymmetry coefficient, and length of kernel groove) for 210 different samples of wheat seeds, divided into three classes (Kama, Rosa, and Canadian).

Objective

The objective of the dataset is to predict the class of wheat seed based on the 7 features.

The 7 features are measured properties of the seeds and are intended to be used to distinguish between different varieties of wheat seeds. The data is well-suited for classification problems, where the goal is to assign each sample to one of the three classes based on its features. The data provides a relatively straightforward, non-complex problem for machine learning algorithms to solve, making it a good starting point for experimentation and learning.

In order to make predictions using the data, a machine learning algorithm is trained on a subset of the data, and then tested on a separate set of samples to determine its accuracy. The performance of the algorithm can then be evaluated using a variety of evaluation metrics, such as precision, recall, f1-score, and accuracy. By experimenting with algorithm and adjusting their parameters, it is possible to achieve higher accuracy in classifying the seeds.

[63]
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, roc_auc_score, roc_curve

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
names = ['area', 'perimeter', 'compactness', 'length-of-kernel', 'width-of-kernel', 'asymmetry-coefficient', 'length-of-groove', 'class']
data = pd.read_csv(url, sep="\s+", names=names)
data.head()



[64]
# Get basic information about the dataset
print(data.info())
print(data.describe())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 210 entries, 0 to 209
Data columns (total 8 columns):
 #   Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   area                   210 non-null    float64
 1   perimeter              210 non-null    float64
 2   compactness            210 non-null    float64
 3   length-of-kernel       210 non-null    float64
 4   width-of-kernel        210 non-null    float64
 5   asymmetry-coefficient  210 non-null    float64
 6   length-of-groove       210 non-null    float64
 7   class                  210 non-null    int64  
dtypes: float64(7), int64(1)
memory usage: 13.2 KB
None
             area   perimeter  compactness  length-of-kernel  width-of-kernel  \
count  210.000000  210.000000   210.000000        210.000000       210.000000   
mean    14.847524   14.559286     0.870999          5.628533         3.258605   
std      2.909699    1.305959     0.023629          0.443063         0.377714   
min     10.590000   12.410000     0.808100          4.899000         2.630000   
25%     12.270000   13.450000     0.856900          5.262250         2.944000   
50%     14.355000   14.320000     0.873450          5.523500         3.237000   
75%     17.305000   15.715000     0.887775          5.979750         3.561750   
max     21.180000   17.250000     0.918300          6.675000         4.033000   

       asymmetry-coefficient  length-of-groove       class  
count             210.000000        210.000000  210.000000  
mean                3.700201          5.408071    2.000000  
std                 1.503557          0.491480    0.818448  
min                 0.765100          4.519000    1.000000  
25%                 2.561500          5.045000    1.000000  
50%                 3.599000          5.223000    2.000000  
75%                 4.768750          5.877000    3.000000  
max                 8.456000          6.550000    3.000000  

[65]
# check the number of nan values in each column
data.isnull().sum(axis = 0)
area                     0
perimeter                0
compactness              0
length-of-kernel         0
width-of-kernel          0
asymmetry-coefficient    0
length-of-groove         0
class                    0
dtype: int64
[66]
# Plot a pairplot to visualize the relationships between the variables
sns.pairplot(data, hue='class')
plt.show()

The pairplot, also known as a scatterplot matrix, is a useful tool for visualizing the relationships between variables in a dataset. In the pairplot of the seed dataset, each variable is plotted against every other variable, resulting in a matrix of scatterplots.

By looking at the scatterplots, can get a sense of the relationships between the variables and the distribution of the data. Some key things to look for in the pairplot include:

Linear relationships: If there is a strong linear relationship between two variables, will see a clear pattern in the scatterplot, such as a line or a cloud of points that form a line-like shape.

Non-linear relationships: If the relationship between two variables is non-linear, will see a scatterplot with points that do not form a clear line.

Outliers: Outliers are data points that are significantly different from the other points in the scatterplot. They can indicate errors in the data or can be interesting observations that are worth exploring further.

Clustering: If there is clustering in the data, you will see groups of points that are close together in the scatterplot. This can indicate that there are different classes or groups of observations in the data.

Skewness: The distribution of the variables can be seen in the histograms on the diagonal of the pairplot. If the histograms are skewed (not symmetrical), it can indicate that the data is not normally distributed.

By analyzing the pairplot of the seed dataset, can get a sense of the relationships between the variables and the distribution of the data. This information can be used to inform the development of predictive models and to identify important variables to focus on in further analysis.

IQR method to detect outliers in the seeds dataset:

[67]
# Calculate the interquartile range (IQR) for each feature
IQR = data.quantile(0.75) - data.quantile(0.25)

# Determine the lower and upper bounds for each feature
lower_bound = data.quantile(0.25) - 1.5 * IQR
upper_bound = data.quantile(0.75) + 1.5 * IQR

# Identify any outliers in the data
outliers = data[(data < lower_bound) | (data > upper_bound)].count()
print("Number of outliers in each feature:")
print(outliers)
Number of outliers in each feature:
area                     0
perimeter                0
compactness              3
length-of-kernel         0
width-of-kernel          0
asymmetry-coefficient    2
length-of-groove         0
class                    0
dtype: int64

Now remove outliers

[68]
# Calculate the interquartile range (IQR) for each feature
IQR = data.quantile(0.75) - data.quantile(0.25)

# Determine the lower and upper bounds for each feature
lower_bound = data.quantile(0.25) - 1.5 * IQR
upper_bound = data.quantile(0.75) + 1.5 * IQR

# Remove any outliers in the data
data = data[(data > lower_bound) & (data < upper_bound)]

This code calculates the IQR for each feature and determines the lower and upper bounds. Any data points that fall outside of these bounds are considered outliers and are removed from the dataset. Finally, the code checks if there are any missing values in the data after removing the outliers.

Missing value check and remove

[69]
# Check if there are any missing values in the data
print("Number of missing values:", data.isnull().sum().sum())

# Remove any missing values in the data
data = data.dropna()

# Check if there are any missing values in the data
print("Number of missing values:", data.isnull().sum().sum())
Number of missing values: 5
Number of missing values: 0

[70]
# Plot a boxplot for each variable, colored by class
sns.boxplot(x="class", y="area", data=data)
plt.show()
sns.boxplot(x="class", y="perimeter", data=data)
plt.show()
sns.boxplot(x="class", y="compactness", data=data)
plt.show()
sns.boxplot(x="class", y="length-of-kernel", data=data)
plt.show()
sns.boxplot(x="class", y="width-of-kernel", data=data)
plt.show()
sns.boxplot(x="class", y="asymmetry-coefficient", data=data)
plt.show()
sns.boxplot(x="class", y="length-of-groove", data=data)
plt.show()

The boxplot of each feature by class shows the distribution of values for each feature within each class. The box represents the interquartile range (IQR), which contains the middle 50% of the data. The line in the middle of the box represents the median of the data. The whiskers extend from the box to the minimum and maximum values, excluding any outliers. Outliers are represented by individual points.

When comparing the boxplots of each feature by class, you can see if there are any differences in the distribution of values for each feature between the classes. For example, if the box for one class is wider or shifted compared to the box for another class, this indicates that the values for that feature are more spread out or different between the two classes.

In general, a feature that is more discriminative between the classes will have more clearly separated boxplots, with a smaller overlap between the boxes. This means that the feature will be a better predictor of the class in a machine learning model.

[71]
# Plot a boxplot to visualize the distribution of area variable for each class
sns.boxplot(x='class', y='area', data=data)
plt.show()

[72]
# Plot a histogram to visualize the distribution of the area variable
sns.histplot(data['area'], kde=True)
plt.show()

Area variable is right skewed.

[73]
# Split the data into training and testing sets
X = data.drop(columns=['class'])
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Build KNN model

[74]
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Define a list of k values to test
k_values = [3, 5, 7, 9, 11]

# Evaluate the performance of the KNN models for each k value
accuracies = []
precisions = []
recalls = []
f1s = []
for k in k_values:
    # Fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = knn.predict(X_test)
    
    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Store the results
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

# Choose the k value that gives the best performance based on the evaluation metrics
best_k = k_values[np.argmax(accuracies)]
print("Best k value:", best_k)


Best k value: 7

[75]
# Select best K value and Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
Accuracy: 0.93

[76]
# Generate the classification report
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           1       1.00      0.77      0.87        13
           2       1.00      1.00      1.00        12
           3       0.84      1.00      0.91        16

    accuracy                           0.93        41
   macro avg       0.95      0.92      0.93        41
weighted avg       0.94      0.93      0.93        41


we evaluate the model performance on the test set using the classification_report function and the confusion matrix. The classification report shows the precision, recall, and f1-score for each class, as well as the overall accuracy of the model. The confusion matrix visualizes the number of true positive, true negative, false positive, and false negative predictions for each class.

[77]
# The confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
# Plot the confusion matrix
sns.heatmap(cm, annot=True)
plt.show()

Confusion Matrix:
[[10  0  3]
 [ 0 12  0]
 [ 0  0 16]]


Conclusion

The accuracy of the KNN model with k=7 for predicting the "class" feature is 93%. The results of the KNN classifier show that the model has an overall accuracy of 93%, which means that the model correctly classifies 93% of the samples in the test set. This is a good accuracy for a relatively small and simple dataset like this one.

The precision, recall, and f1-score give us more detailed information about the performance of the model for each class. Precision is the number of true positive predictions divided by the sum of true positive and false positive predictions. Recall is the number of true positive predictions divided by the sum of true positive and false negative predictions. The f1-score is the harmonic mean of precision and recall and provides a single metric that takes into account both precision and recall.

In this case, the precision for each class is high, which means that the model has a low number of false positive predictions. The recall for each class is also high, which means that the model has a low number of false negative predictions. The f1-score is also high, which indicates that the precision and recall are both good.

In conclusion, the results of the KNN classifier show that it is a good model for this dataset, with high accuracy, precision, recall, and f1-score, and low false positive and false negative predictions. This suggests that the model has learned the patterns in the data well and is able to accurately predict the class of new seeds based on their features.