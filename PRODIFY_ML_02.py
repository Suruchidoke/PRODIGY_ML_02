import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

# Generate synthetic data for 200 customers
n_customers = 200
customer_id = range(1, n_customers+1)
annual_income = np.random.randint(15000, 150000, n_customers)  # income between 15k and 150k
spending_score = np.random.randint(1, 101, n_customers)       # spending score 1-100

# Create DataFrame
data = pd.DataFrame({
    'CustomerID': customer_id,
    'Annual Income': annual_income,
    'Spending Score': spending_score
})

print(data.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Basic info and summary
print("\nDataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Scatter plot: Annual Income vs Spending Score
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Annual Income', y='Spending Score', data=data, s=60)
plt.title('Customer Distribution')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

# Correlation heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(data[['Annual Income', 'Spending Score']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

from sklearn.cluster import KMeans

# Select features for clustering
X = data[['Annual Income', 'Spending Score']]

# Elbow Method to find optimal clusters
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Apply KMeans with k=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

print("\nFirst 10 Customers with Cluster Labels:")
print(data.head(10))

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual Income', y='Spending Score', hue='Cluster', palette='viridis', data=data, s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
