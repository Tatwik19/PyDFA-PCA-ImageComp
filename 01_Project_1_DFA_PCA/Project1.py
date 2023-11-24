# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:28:18 2023

@author: tatwi
"""

'''
Project 1
PCA Analysis
•	Calculating the discriminant function of data points and analyzing them
•	Calculate pooled covariance and derive statical analysis: confusion matrix.
•	And perform Principal component analysis.

packages used: Pandas, numpy, matplotlib, sklearn

Attribute Information:
The Kecimen and Besni raisin varieties were obtained with CVS. 
A total of 900 raisins were used, including 450 from both varieties, and 7 
morphological features were extracted.

1.) Area: Gives the number of pixels within the boundaries of the raisin. 
2.) Perimeter: It measures the environment by calculating the distance between 
    the boundaries of the raisin and the pixels around it.
3.) MajorAxisLength: Gives the length of the main axis, which is the longest 
    line that can be drawn on the raisin.
4.) MinorAxisLength: Gives the length of the small axis, which is the shortest
    line that can be drawn on the raisin.
5.) Eccentricity: It gives a measure of the eccentricity of the ellipse, which
    has the same moments as raisins. 
6.) ConvexArea: Gives the number of pixels of the smallest convex shell of the
    region formed by the raisin.
7.) Extent: Gives the ratio of the region formed by the raisin to the total
    pixels in the bounding box.
8.) Class: Kecimen and Besni raisin.
    
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('Raisin_Dataset.csv')
print(data.head())

classes = data['Class']
numClasses = len(classes.unique())

print("Number of classes: ", numClasses)

grouped_data = data.groupby('Class')

mean_vectors = {} # new
for classLabel, group in grouped_data:
    mean_vectors[classLabel] = group.iloc[:, :-1].mean().values

classStats = {}
for classLabel in classes.unique():
    
    classData = data[data['Class'] == classLabel]
    classData = classData.drop(columns = ['Class'])
    
    meanVector = np.mean(classData, axis=0)
    covarianceMatrix = np.cov(classData, rowvar=False)
    
    classStats[classLabel] = {
        'meanVector' : meanVector,
        'covarianceMatrix': covarianceMatrix,
        }
    
for classLabel, stats in classStats.items():
    print(f"\nClass {classLabel} Mean Vector:")
    print(*stats['meanVector'])
    print(f"\nClass {classLabel} Covariance Matrix:")
    print(stats['covarianceMatrix'])

pooledCovarianceM = classStats['Kecimen']['covarianceMatrix'] + classStats['Besni']['covarianceMatrix']
pooledCovarianceM = pooledCovarianceM/2

def discriminant_function(x, mean, covariance):
    return -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(covariance)),
                         (x - mean))

discriminant_functions = [] #new
for index, row in data.iterrows():
    x = row[:-1].values
    discriminants = {}
    for classLabel, mean_vector in mean_vectors.items():
        discriminant = discriminant_function(x, mean_vector, pooledCovarianceM)
        discriminants[classLabel] = discriminant
    predictedClass = max(discriminants, key=discriminants.get)
    discriminant_functions.append([row['Class'], predictedClass, discriminants])

result_df = pd.DataFrame(discriminant_functions, 
                         columns=['TrueClass', 'PredictedClass', 'Discriminants'])

result_df.to_csv('DF_results.csv', index=False)

print("Saved to 'DF_results.csv'")
print(pd.read_csv('DF_results.csv').head())

result_df = pd.read_csv('DF_results.csv')
trueLabels = result_df['TrueClass']
PredictedClass = result_df['PredictedClass']
confusionMatrix_pooled = confusion_matrix(trueLabels, PredictedClass)

accPooled = accuracy_score(trueLabels, PredictedClass)

print("Confusion Matrix (Pooled Covariance Matrix):")
print(confusionMatrix_pooled)
print(f"Accuracy of Pooled Covariance Matrix: {accPooled:.2f}")

def calculate_rates(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    true_positive_rate = tp / (tp + fn)
    true_negative_rate = tn / (tn + fp)
    return false_positive_rate, false_negative_rate, true_positive_rate, true_negative_rate

fpr_p, fnr_p, tpr_p, tnr_p = calculate_rates(confusionMatrix_pooled)

print("Rates for Pooled Covariance Matrix:")
print("False Positive Rate:", fpr_p)
print("False Negative Rate:", fnr_p)
print("True Positive Rate:", tpr_p)
print("True Negative Rate:", tnr_p)
print('\n')

indVar = ["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity",
          "Extent", "Perimeter"]
depVar = "ConvexArea"

# Using first 600 datapts as training data
trainData = data.iloc[600:]
model = LinearRegression()
X_train = trainData[indVar]
y_train = trainData[depVar]
model.fit(X_train, y_train)

# Using last 300 datapts as testing data
testData = data.iloc[:300]
X_test = testData[indVar]
y_test = testData[depVar]
y_pred = model.predict(X_test)


pca = PCA()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_pca = pca.fit_transform(X_train_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()
num_components = len(explained_variance_ratio)
components = range(1, num_components + 1)

print("Explained variance Ratio:", explained_variance_ratio)
plt.figure()
plt.title('Pareto Chart')
plt.xlabel('Principal Component')
plt.ylabel('PoV')
plt.bar(components, explained_variance_ratio, align='center',
        label='Individual Variance')
plt.legend()

plt.plot(components, cumulative_variance_ratio, marker='o', color='r',
         label='Cumulative Variance Ratio')
plt.ylabel('Explained Variance Ratio')
plt.legend()
plt.show()


pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2): {r2}")