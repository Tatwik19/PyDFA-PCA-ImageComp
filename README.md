# PyDFA-PCA-ImageComp
 **Applying Machine Learning concepts with Python**

## Project 1: DFA PCA Analytics
Explore advanced statistical methods for dimensionality reduction and analytics using Discriminant Function Analysis (DFA) and Principal Component Analysis (PCA).  
[01_PyDFA-PCA-Analytics.ipynb](https://github.com/Tatwik19/PyDFA-PCA-ImageComp/blob/main/Project_1_DFA_PCA/01_PyDFA-PCA-Analytics.ipynb)  

### Discriminant Function Analysis (DFA)
DFA is a statistical method used to classify unknown individuals into a certain group. DFA is a supervised classification technique that uses a mathematical function to distinguish between predefined groups of samples. DFA is used to determine which variables contribute most to group separation.  

### Principal Component Analysis (PCA)
PCA is an unsupervised learning method in that it does not use the output information. It is a statistical procedure that reduces the dimensionality of large datasets while preserving crucial information. PCA is a popular technique for analyzing large datasets containing a high number of dimensions/features per observation    

The project uses [Raisin Classification](https://github.com/Tatwik19/PyDFA-PCA-ImageComp/blob/main/Project_1_DFA_PCA/Raisin_Dataset.csv) data set.

## Project 2: Image Compression using Kmeans, Agglomerative
Dive into the world of image compression with K-means and Agglomerative clustering algorithms. Efficiently reduce image size while preserving essential visual information.  
[01_ImageComp-Kmeans-Agglomerative.ipynb](https://github.com/Tatwik19/PyDFA-PCA-ImageComp/blob/main/Project_2_Imgcompression/01_ImageComp-Kmeans-Agglomerative.ipynb)

### Clustering in Unsupervised Learning
Unsupervised learning involves learning from data without predefined labels. Clustering finds similarities between objects, grouping them into clusters based on characteristics.

### Types of Clustering
**Centroid-based clustering (e.g., K-means)**  
Clusters are represented by a central vector, which may not necessarily be a member of the data set. When the number of clusters is fixed at K, the algorithm finds the K cluster centers and assigns the objects to the nearest cluster center, such that the distances from the cluster are minimized.  

**Connectivity-based (Hierarchical) clustering (e.g., Agglomerative)**  
Cluster analysis based on the core idea of objects being more related to nearby objects than to objects farther away. The algorithm connect "objects" to form "clusters" based on their distance.  

This project utilized [Tatwik.jpg](https://github.com/Tatwik19/PyDFA-PCA-ImageComp/blob/main/Project_2_Imgcompression/Tatwik.jpg)

***
Packages used: pandas, numpy, matplotlib, sklearn, PIL, and os