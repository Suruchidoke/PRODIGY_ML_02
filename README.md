🧠 Task 02 – Customer Segmentation using K-Means Clustering
Intern Name: Suruchi
Company: Prodigy InfoTech – Machine Learning Internship

📌 Problem Statement
Create a K-Means clustering algorithm to group retail store customers based on their:

Annual Income

Spending Score

This helps businesses identify and target different customer segments more efficiently.

📊 Dataset
Dataset used for this project:
🔗 Customer Segmentation Tutorial (Kaggle)

🛠️ Tools & Technologies
Python 🐍

Libraries:
Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

📈 Methodology
Data Preparation: Used synthetic data for 200 customers with attributes – income & spending score.

Visualization:

Scatter plot for customer distribution

Correlation heatmap

Elbow Method:

To find the optimal number of clusters (k)

Clustering with K-Means:

Applied K-Means with k=5 clusters

Visualization:

Final cluster visualization with centroids

🔍 Output Snippets
📊 Elbow Method Graph

🟢 Clustered Scatter Plot with Centroids

💻 Code Example
python
Copy
Edit
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(X)
🧠 Key Insights
Successfully segmented customers into 5 meaningful clusters.

Helps retail business analyze spending behavior.

Useful for targeted marketing and strategic decision-making.

