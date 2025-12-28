import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_pipeline(df, n_components=20, n_clusters=5, random_state=42):
    X = df.drop(columns=["target", "ID"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_pca)
    return {
        "n_components": int(n_components),
        "n_clusters": int(n_clusters),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "silhouette": float(silhouette_score(X_pca, labels)),
        "inertia": float(km.inertia_),
    }


def main():
    df = pd.read_csv("Analytica.csv")
    print("shape:", df.shape)
    print("columns:", df.columns.tolist()[:30], "...")
    print("target_value_counts_incl_nan:")
    print(df["target"].value_counts(dropna=False))
    print("target_missing_rate:")
    print(float(df["target"].isna().mean()))

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("columns_with_missing_values:")
    print(missing)
    print("missing_percent:")
    print((missing / len(df) * 100).round(2))

    X_df = df.drop(columns=["target", "ID"])
    X_scaled = StandardScaler().fit_transform(X_df)

    pca_full = PCA().fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    best_n_90 = int(np.argmax(cum >= 0.90) + 1)
    best_n_95 = int(np.argmax(cum >= 0.95) + 1)
    print("components_for_90pct:", best_n_90)
    print("components_for_95pct:", best_n_95)

    plt.figure(figsize=(7, 4))
    plt.plot(cum)
    plt.axhline(0.90, linestyle="--")
    plt.axhline(0.95, linestyle="--")
    plt.axvline(best_n_90, linestyle="--")
    plt.axvline(best_n_95, linestyle="--")
    plt.xlabel("n_components")
    plt.ylabel("cumulative_explained_variance")
    plt.tight_layout()
    plt.show()

    best_n = best_n_95
    pca = PCA(n_components=best_n, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print("using_best_n:", int(best_n))
    print("explained_variance_sum:", float(pca.explained_variance_ratio_.sum()))
    print("X_pca_shape:", X_pca.shape)

    n_clusters = 5
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster = km.fit_predict(X_pca).astype(int)
    print("silhouette_score:", float(silhouette_score(X_pca, cluster)))
    print("unique_labels:", np.unique(cluster))
    print("counts:", np.bincount(cluster))

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster, s=10, alpha=0.8, cmap="tab10")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans clusters on first two PCA components")
    plt.tight_layout()
    plt.show()

    df["cluster"] = cluster
    feature_cols = df.columns.drop(["ID", "target", "cluster"], errors="ignore")
    cluster_sizes = df["cluster"].value_counts().sort_index()
    cluster_means = df.groupby("cluster")[feature_cols].mean(numeric_only=True)
    global_mean = df[feature_cols].mean(numeric_only=True)
    delta = cluster_means.subtract(global_mean, axis=1)
    top_pos = delta.apply(lambda s: s.nlargest(5).index.tolist(), axis=1)
    top_neg = delta.apply(lambda s: s.nsmallest(5).index.tolist(), axis=1)
    summary = pd.DataFrame({"cluster_size": cluster_sizes, "top_pos(5)": top_pos, "top_neg(5)": top_neg})
    print("cluster_feature_summary:")
    print(summary.to_string())

    metrics = run_pipeline(df, n_components=best_n, n_clusters=n_clusters, random_state=42)
    print("pipeline_metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
