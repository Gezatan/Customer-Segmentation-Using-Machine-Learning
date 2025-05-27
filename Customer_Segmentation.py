import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
# Update the path if needed
# If your columns are 'CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'
df = pd.read_csv('/Users/gezatannimalraj/Desktop/AI:ML Proj/Mall_Customers.csv')
print(df.columns)

# Select features for clustering (X: Spending Score, Y: Annual Income)
X = df[['Spending Score (1-100)', 'Annual Income (k$)']]

# Elbow Method to find optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Choose optimal clusters (e.g., 5 based on elbow plot)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Optional: PCA for visualization (not strictly needed here since 2D)
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# Scatter plot of clusters (X: Spending Score (1-100), Y: Annual Income)
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink', 'gray']
for i in range(optimal_clusters):
    plt.scatter(
        df[df['Cluster'] == i]['Spending Score (1-100)'],
        df[df['Cluster'] == i]['Annual Income (k$)'],
        s=60,
        c=colors[i],
        label=f'Cluster {i}'
    )
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Annual Income (k$)')
plt.title('Customer Segments')
plt.legend()
plt.show()

# Cluster summary table
summary = df.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Gender': lambda x: x.value_counts().index[0],
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean'
}).rename(columns={'CustomerID': 'Count', 'Gender': 'Most Common Gender', 'Age': 'Avg Age',
                   'Annual Income (k$)': 'Avg Income', 'Spending Score (1-100)': 'Avg Spending Score'})

print("\nCluster Summary Table:")
print(summary)

# Insights from each cluster
print("\nCluster Insights:")
for i, row in summary.iterrows():
    print(f"\nCluster {i}:")
    print(f"- Number of Customers: {row['Count']}")
    print(f"- Most Common Gender: {row['Most Common Gender']}")
    print(f"- Average Age: {row['Avg Age']:.1f}")
    print(f"- Average Annual Income: {row['Avg Income']:.2f}")
    print(f"- Average Spending Score: {row['Avg Spending Score']:.2f}")
    if row['Avg Income'] > summary['Avg Income'].mean() and row['Avg Spending Score'] > summary['Avg Spending Score'].mean():
        print("  > Likely VIP customers: high income and high spending.")
    elif row['Avg Income'] < summary['Avg Income'].mean() and row['Avg Spending Score'] > summary['Avg Spending Score'].mean():
        print("  > Bargain hunters: lower income but high spending.")
    elif row['Avg Income'] > summary['Avg Income'].mean() and row['Avg Spending Score'] < summary['Avg Spending Score'].mean():
        print("  > Potential savers: high income but low spending.")
    else:
        print("  > Low income and low spending: price-sensitive or less engaged customers.")
