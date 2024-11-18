#EXPERIMENT-8 
# Implementing Dimensionality Reduction using Principal 
Component Analysis (PCA). 
Step-1: Loading the dataset 
# Importing necessary libraries 
import numpy as np 
import pandas as pd 
from sklearn.datasets import make_classification 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
# Generating synthetic dataset 
X, y = make_classification( 
n_samples=200,    # Number of samples 
n_features=10,    # Total number of features 
n_informative=5,  # Number of informative features 
n_redundant=2,    # Number of redundant features 
n_classes=3,      
# Number of classes 
random_state=42 
) 
# Converting to DataFrame for easier manipulation 
df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in 
range(X.shape[1])]) 
df['target'] = y 
Step-2: Standardizing the data 
Ensures each feature has a mean 0 and a standard deviation of 1 
# Standardizing 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
Step-3: Co-variance matrix 
# Calculating matrix 
cov_matrix = np.cov(X_scaled.T) 
print("Covariance Matrix:\n", cov_matrix) 
Step-4: Finding Eigenvalues and Eigenvectors 
# Calculating eigenvalues and eigenvectors 
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix) 
print("Eigenvalues:\n", eigenvalues) 
print("Eigenvectors:\n", eigenvectors) 
Step-5: Choosing number of components 
# Calculating explained variance 
explained_variance = eigenvalues / np.sum(eigenvalues) 
print("Explained Variance:", explained_variance) 
# Determining number of components to retain 95% of variance 
cumulative_variance = np.cumsum(explained_variance) 
components = np.argmax(cumulative_variance >= 0.95) + 1 
print("Number of components to retain 95% variance:", components) 
Step-6: Transforming the data 
# Sort eigenvalues and eigenvectors, and select top components 
sorted_indices = np.argsort(eigenvalues)[::-1] 
sorted_eigenvectors = eigenvectors[:, sorted_indices[:components]] 
# Transform data 
X_pca = X_scaled.dot(sorted_eigenvectors) 
print("Transformed Data (PCA):\n", X_pca) 
Step-7: Visualizing the results 
# Plot transformed data 
plt.figure(figsize=(8, 6)) 
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k') 
plt.xlabel('Principal Component 1') 
plt.ylabel('Principal Component 2') 
plt.title('PCA of Synthetic Dataset') 
plt.colorbar(label='Target Class') 
plt.show()
