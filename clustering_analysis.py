# clustering_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def prepare_data(df):
    """Aggregate and normalize the data for clustering."""
    country_subscription_counts = df.groupby(['Country', 'Subscription Type']).size().unstack(fill_value=0)
    scaler = StandardScaler()
    country_subscription_scaled = scaler.fit_transform(country_subscription_counts)
    country_subscription_scaled_df = pd.DataFrame(country_subscription_scaled, index=country_subscription_counts.index, columns=country_subscription_counts.columns)
    return country_subscription_counts, country_subscription_scaled_df

def determine_optimal_clusters(data, max_clusters=10, n_init=10):
    """Determine the optimal number of clusters using the Elbow method."""
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=n_init)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_elbow_method(wcss):
    """Plot the Elbow method graph."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def perform_clustering(data, n_clusters, n_init=10):
    """Perform K-means clustering and return the cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init)
    clusters = kmeans.fit_predict(data)
    return clusters

def add_cluster_labels(df, clusters):
    """Add the cluster labels to the original DataFrame."""
    df['Cluster'] = clusters

def plot_heatmap(data):
    """Plot a heatmap of the subscription types by country."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.drop('Cluster', axis=1), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Subscription Types by Country')
    plt.show()

def plot_clusters(data):
    """Visualize the cluster assignment."""
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='basic', y='premium', hue='Cluster', data=data, palette='viridis')
    plt.title('Clusters of Countries Based on Subscription Type Preferences')
    plt.xlabel('Basic Subscriptions')
    plt.ylabel('Premium Subscriptions')
    plt.show()
