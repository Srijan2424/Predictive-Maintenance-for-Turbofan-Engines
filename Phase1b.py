import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

class B:
    def __init__(self,data,b1):
        self.dataset=data
        self.dataset1=b1

    def cluster_b(self):

        # === CONFIG ===
        USE_KMEANS = True  # 🔄 Set to False to use Agglomerative Clustering
        N_CLUSTERS = 5     # Targeting Stages 0–4

        df_clean=self.dataset

        # ===  Normalize ===
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        # ===  PCA (for visualization only) ===
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])

        # ===  Clustering ===
        if USE_KMEANS:
            model = KMeans(n_clusters=N_CLUSTERS, random_state=42)
            cluster_labels = model.fit_predict(X_scaled)
            clustering_method = "KMeans"
        else:
            model = AgglomerativeClustering(n_clusters=N_CLUSTERS)
            cluster_labels = model.fit_predict(X_scaled)
            clustering_method = "Agglomerative"

        df_pca['cluster'] = cluster_labels

        # ===  Map Clusters to Degradation Stages ===
        cluster_means = pd.DataFrame(X_scaled, columns=df_clean.columns)
        cluster_means['cluster'] = df_pca['cluster']
        severity = cluster_means.groupby('cluster').mean().mean(axis=1)
        order = severity.sort_values().index.tolist()
        cluster_to_stage = {cluster: i for i, cluster in enumerate(order)}
        df_pca['degradation_stage'] = df_pca['cluster'].map(cluster_to_stage)

        # 🔢 Print number of samples in each cluster
        print("\n📊 Cluster Size Distribution:")
        print(df_pca['cluster'].value_counts().sort_index())

        print("\n📊 Mapped Degradation Stage Distribution:")
        print(df_pca['degradation_stage'].value_counts().sort_index())

        # ===  Merge and Save ===
        df_out = self.dataset1.reset_index(drop=True).copy()
        df_out['cluster'] = df_pca['cluster']
        df_out['degradation_stage'] = df_pca['degradation_stage']
        filename = f"GroupA_Clustered_{clustering_method}.csv"
        df_out.to_csv(filename, index=False)
        print(f"💾 Saved: {filename}")

        # === Visualize ===
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='degradation_stage', palette='viridis', s=15)
        plt.title(f"Group A (FD001 + FD003): Degradation Stages (0–4) — {clustering_method}")
        plt.legend(title='Stage')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
