# PyDFA-PCA-ImageComp
 **Applying Machine Learning concepts with Python**

## Project 1: DFA PCA Analytics
Explore advanced statistical methods for dimensionality reduction and analytics using Discriminant Function Analysis (DFA) and Principal Component Analysis (PCA).

### Discriminant function analysis (DFA)
DFA is a statistical method used to classify unknown individuals into a certain group. DFA is a supervised classification technique that uses a mathematical function to distinguish between predefined groups of samples. DFA is used to determine which variables contribute most to group separation.  

### Principal component analysis (PCA)
PCA is an unsupervised learning method in that it does not use the output information; the criterion to be maximized is the variance.
This statistical technique helps enormous data to be condensed into a more manageable data for easier visualization and interpretation.  

The project uses [Raisin Classification](https://github.com/Tatwik19/PyDFA-PCA-ImageComp/blob/main/Project_1_DFA_PCA/Raisin_Dataset.csv) data set (more information is avalable in [01_PyDFA-PCA-Analytics.ipynb](https://github.com/Tatwik19/PyDFA-PCA-ImageComp/blob/main/Project_1_DFA_PCA/01_PyDFA-PCA-Analytics.ipynb))

## Project 2: Image Compression using Kmeans, Agglomerative
Dive into the world of image compression with K-means and Agglomerative clustering algorithms. Efficiently reduce image size while preserving essential visual information. Unlock the power of clustering for image data.  

### Clustering in Unsupervised Learning
Unsupervised learning involves learning from data without predefined labels. Clustering finds similarities between objects, grouping them into clusters based on characteristics.

### Types of Clustering
- **Centroid-based clustering (e.g., K-means)**  
  - Represents clusters by central vectors.
  - Assigns objects to the nearest cluster center to minimize distances.

- **Connectivity-based (Hierarchical) clustering (e.g., Agglomerative)**  
  - Objects are more related to nearby objects than those farther away.
  - Connects objects to form clusters based on distance (average, single, and complete).

This project utilized [Tatwik.jpg](https://github.com/Tatwik19/PyDFA-PCA-ImageComp/blob/main/Project_2_Imgcompression/Tatwik.jpg)

***
Packages used: pandas, numpy, matplotlib, sklearn, PIL, and os