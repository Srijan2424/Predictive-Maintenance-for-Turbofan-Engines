import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans  # or AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

class A:
    def __init__(self,data):
        self.dataset=data
    def cluster_a(self):
        # ===  Drop Redundant Columns ===
        # Sensors with constant readings across datasets
        redundant_cols = [
            'unit', 'time_in_cycles', 'op_set_1', 'op_set_2', 'op_set_3',
            'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
            'sensor_16', 'sensor_18', 'sensor_19'
        ]

        df_features = self.dataset.drop(columns=redundant_cols + ['dataset_id', 'fault_modes', 'ops_vary'])

        # ===  Normalize Features ===
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features)

        # === Dimensionality Reduction (PCA) ===
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])

        # ===  Clustering ===
        # Choose model: KMeans or Agglomerative
        USE_KMEANS = True
        n_clusters = 5  # Corresponding to Stage 0 → Stage 4

        if USE_KMEANS:
            model = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            from sklearn.cluster import AgglomerativeClustering
            model = AgglomerativeClustering(n_clusters=n_clusters)

        df_pca['cluster'] = model.fit_predict(X_pca)

        # ===  Merge back with original data ===
        df_full = self.dataset.copy().reset_index(drop=True)
        df_full['cluster'] = df_pca['cluster']

        # ===  Manual Mapping (Optional) ===
        # You can remap the cluster numbers to meaningful stages
        # (Example: Map based on sensor severity or PCA distance)
        # Example mapping: cluster with highest mean sensor values → Stage 4

        # Automatically compute rough ordering by average sensor magnitude
        cluster_severity = df_full.groupby('cluster')[df_features.columns].mean().mean(axis=1)
        cluster_order = cluster_severity.sort_values().index.tolist()
        cluster_to_stage = {cluster: i for i, cluster in enumerate(cluster_order)}

        df_full['degradation_stage'] = df_full['cluster'].map(cluster_to_stage)

        print("✅ Clustering completed and stages assigned.")

        # ===  Save Output ===
        df_full.to_csv("CMAPSS_Clustered_All.csv", index=False)
        print("📁 Output saved to: CMAPSS_Clustered_All.csv")

        # === Visualize Clusters ===
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue=df_full['degradation_stage'], palette='viridis', s=10)
        plt.title("Clusters (Mapped to Stages 0–4)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
