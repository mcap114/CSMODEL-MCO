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

def prepare_for_apriori(country_subscription_counts):
    """Convert the counts DataFrame to a one-hot encoded DataFrame for Apriori."""
    country_subscription_one_hot = country_subscription_counts.map(lambda x: 1 if x > 0 else 0)
    return country_subscription_one_hot

def apply_threshold(x, threshold):
    if pd.isnull(x) or x < threshold:
        return 0
    else:
        return 1

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
    # Calculate the total subscriptions for size differentiation
    df['Total Subscriptions'] = df.iloc[:, :-1].sum(axis=1)

def plot_barplot(data):
    """Plot a stacked bar plot of the subscription types by country."""
    data = data.drop('Cluster', axis=1).reset_index()
    
    data.set_index('Country', inplace=True)
    
    ax = data.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')

    plt.title('Subscription Types by Country')
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Subscription Type', loc='upper right')
    plt.show()


def plot_clusters(data):
    """Visualize the cluster assignment with size differentiation."""
    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        data=data, 
        x='basic', 
        y='premium', 
        hue='Cluster', 
        size='Total Subscriptions', 
        sizes=(20, 200), 
        palette='viridis'
    )
    plt.title('Clusters of Countries Based on Subscription Type Preferences')
    plt.xlabel('Basic Subscriptions')
    plt.ylabel('Premium Subscriptions')
    plt.legend(title='Cluster', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.show()
    
    
    
def display_cluster_info(data):
    """Display information about the clusters obtained."""
    cluster_info = data.groupby('Cluster').mean()
    cluster_sizes = data['Cluster'].value_counts().sort_index()
    
    # Collect the countries in each cluster
    countries_in_clusters = data.groupby('Cluster').apply(lambda x: ', '.join(x.index)).sort_index()

    print("Cluster Information:")
    print("====================")
    for cluster in cluster_sizes.index:
        print(f"\nCluster {cluster}:")
        print(f"Number of countries: {cluster_sizes[cluster]}")
        print("Countries: ", countries_in_clusters[cluster])
        print("Average subscription counts:")
        print(cluster_info.loc[cluster])
        print("--------------------------")
