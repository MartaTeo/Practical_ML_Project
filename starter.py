import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def missing_report(df):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return missing, (missing / len(df) * 100).round(2)


def scale_features(df, drop_cols):
    X_df = df.drop(columns=drop_cols, errors="ignore")
    if X_df.isna().any().any():
        raise ValueError("Found missing values in features (non-target columns).")
    X_scaled = StandardScaler().fit_transform(X_df)
    return X_df, X_scaled


def choose_n_components_by_variance(X_scaled, threshold=0.95):
    pca_full = PCA().fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n = int(np.argmax(cum >= threshold) + 1)
    return n, cum


def fit_pca(X_scaled, n_components, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca


def fit_kmeans(X_pca, n_clusters, random_state=42):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_pca).astype(int)
    return km, labels


def cluster_feature_summary(df, feature_cols, cluster_col="cluster", top_n=5):
    cluster_sizes = df[cluster_col].value_counts().sort_index()
    cluster_means = df.groupby(cluster_col)[feature_cols].mean(numeric_only=True)
    global_mean = df[feature_cols].mean(numeric_only=True)
    delta = cluster_means.subtract(global_mean, axis=1)
    top_pos = delta.apply(lambda s: s.nlargest(top_n).index.tolist(), axis=1)
    top_neg = delta.apply(lambda s: s.nsmallest(top_n).index.tolist(), axis=1)
    return pd.DataFrame({"cluster_size": cluster_sizes, "top_pos": top_pos, "top_neg": top_neg})


def infer_targets_from_clusters(df, cluster_col="cluster", target_col="target", threshold=0.5):
    known = df[target_col].notna()
    mu = df.loc[known].groupby(cluster_col)[target_col].mean()
    global_mu = df.loc[known, target_col].mean()
    prob = df[cluster_col].map(mu).fillna(global_mu)
    pred = (prob > threshold).astype(int)
    return prob, pred, mu, float(global_mu)


def make_submit(df, unlabeled_mask, id_col="ID", pred_col="target_pred", out_path="submit.csv"):
    sub = df.loc[unlabeled_mask, [id_col]].copy()
    sub["target"] = df.loc[unlabeled_mask, pred_col].astype(int).values
    sub.to_csv(out_path, index=False)
    return sub


def cv_random_forest_auc_acc(X, y, n_splits=5, random_state=42, n_estimators=500, min_samples_leaf=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, accs = [], []
    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        p = rf.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_te, p))
        accs.append(accuracy_score(y_te, pred))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)), float(np.std(accs))


def fit_predict_random_forest(X_train, y_train, X_test, random_state=42, n_estimators=500, min_samples_leaf=5):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return rf, proba, pred


def rf_feature_importances(rf, feature_names, top_n=25):
    s = pd.Series(rf.feature_importances_, index=list(feature_names)).sort_values(ascending=False)
    return s.head(top_n)


def disagreement_summary_unlabeled(df, cluster_col="cluster"):
    cols = [cluster_col, "target_prob_cluster", "target_pred_cluster", "target_prob_rf", "target_pred_rf"]
    tmp = df.loc[df["target"].isna(), cols].copy()
    tmp["disagree"] = (tmp["target_pred_cluster"] != tmp["target_pred_rf"]).astype(int)
    out = tmp.groupby(cluster_col).agg(
        n=("disagree", "size"),
        disagree_rate=("disagree", "mean"),
        cluster_prob=("target_prob_cluster", "mean"),
        rf_prob_mean=("target_prob_rf", "mean"),
        rf_prob_median=("target_prob_rf", "median"),
        pct_cluster_pred1=("target_pred_cluster", "mean"),
        pct_rf_pred1=("target_pred_rf", "mean"),
    ).sort_values("disagree_rate", ascending=False)
    return out


def rf_trait_diff_by_cluster_unlabeled(df, traits, cluster_col="cluster"):
    unl = df["target"].isna()
    rows = []
    for c in sorted(df.loc[unl, cluster_col].dropna().unique()):
        sub = df.loc[unl & (df[cluster_col] == c)].copy()
        if sub.get("target_pred_rf") is None or sub["target_pred_rf"].nunique() < 2:
            continue
        m = sub.groupby("target_pred_rf")[traits].mean(numeric_only=True)
        diff = (m.loc[1] - m.loc[0]).rename(lambda x: f"diff_{x}")
        rows.append(pd.concat([pd.Series({"cluster": c, "n": len(sub), "rf1_rate": sub["target_pred_rf"].mean()}), diff]))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("cluster").sort_values("n", ascending=False)


def build_parser():
    p = argparse.ArgumentParser(prog="analytica_cli.py")
    p.add_argument("--data", type=str, default="Analytica.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--basic", action="store_true")
    p.add_argument("--missing", action="store_true")
    p.add_argument("--cluster", action="store_true")
    p.add_argument("--n-clusters", type=int, default=5)
    p.add_argument("--pca-threshold", type=float, default=0.95)
    p.add_argument("--pca-components", type=int, default=None)
    p.add_argument("--cluster-summary", action="store_true")
    p.add_argument("--cluster-means", action="store_true")
    p.add_argument("--export-cluster-submit", type=str, default=None)
    p.add_argument("--rf", action="store_true")
    p.add_argument("--cv", action="store_true")
    p.add_argument("--export-rf-submit", type=str, default=None)
    p.add_argument("--feature-importance", type=int, nargs="?", const=25, default=None)
    p.add_argument("--disagreement", action="store_true")
    p.add_argument("--trait-diffs", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--sweep-pca", action="store_true",
                   help="Sweep PCA n_components and report silhouette/inertia/explained_variance.")
    p.add_argument("--comp-min", type=int, default=20)
    p.add_argument("--comp-max", type=int, default=55)
    p.add_argument("--comp-step", type=int, default=1)

    p.add_argument("--sweep-k", action="store_true",
                   help="Sweep KMeans k for a fixed PCA n_components (elbow/inertia + silhouette).")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=15)

    p.add_argument("--prefer", type=str, default="silhouette", choices=["silhouette", "inertia"],
                   help="How to auto-pick best setting when sweeping.")

    p.add_argument("--export-sweep", type=str, default=None,
                   help="Optional CSV path to save sweep results (like notebook results_df).")

    p.add_argument("--auto-set", action="store_true",
                   help="If sweeping, automatically set best n_components or k for the downstream run.")

    return p



def sweep_pca_components(df, n_clusters, comp_min=5, comp_max=60, comp_step=1, seed=42):
    X_df, X_scaled = scale_features(df, drop_cols=["target", "ID"])

    rows = []
    for n_components in range(int(comp_min), int(comp_max) + 1, int(comp_step)):
        pca, X_pca = fit_pca(X_scaled, n_components=n_components, random_state=seed)
        km, labels = fit_kmeans(X_pca, n_clusters=n_clusters, random_state=seed)

        rows.append({
            "n_components": int(n_components),
            "n_clusters": int(n_clusters),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "silhouette": float(silhouette_score(X_pca, labels)),
            "inertia": float(km.inertia_),
        })

    return pd.DataFrame(rows).sort_values("silhouette", ascending=False)


def sweep_kmeans_k(df, n_components, k_min=2, k_max=15, seed=42):
    X_df, X_scaled = scale_features(df, drop_cols=["target", "ID"])
    pca, X_pca = fit_pca(X_scaled, n_components=int(n_components), random_state=seed)

    rows = []
    for k in range(int(k_min), int(k_max) + 1):
        km, labels = fit_kmeans(X_pca, n_clusters=k, random_state=seed)
        rows.append({
            "n_components": int(n_components),
            "n_clusters": int(k),
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "silhouette": float(silhouette_score(X_pca, labels)),
            "inertia": float(km.inertia_),
        })

    return pd.DataFrame(rows).sort_values("n_clusters")


def pick_best_row(results_df, prefer="silhouette"):
    if results_df.empty:
        raise ValueError("Empty sweep results.")
    if prefer == "silhouette":
        return results_df.sort_values("silhouette", ascending=False).iloc[0]
    if prefer == "inertia":
        return results_df.sort_values("inertia", ascending=True).iloc[0]
    raise ValueError(f"Unknown prefer={prefer}")



def main():
    args = build_parser().parse_args()

    need_cluster = (
        args.cluster
        or args.cluster_summary
        or args.cluster_means
        or args.export_cluster_submit is not None
        or args.disagreement
        or args.trait_diffs
    )
    need_rf = (
        args.rf
        or args.cv
        or args.export_rf_submit is not None
        or args.feature_importance is not None
        or args.disagreement
        or args.trait_diffs
    )

    df = pd.read_csv(args.data)

        # --- NEW: sweep PCA components (notebook-like results_df) ---
    if args.sweep_pca:
        results = sweep_pca_components(
            df,
            n_clusters=args.n_clusters,
            comp_min=args.comp_min,
            comp_max=args.comp_max,
            comp_step=args.comp_step,
            seed=args.seed,
        )
        print("pca_sweep_top10_by_silhouette:")
        print(results.head(10).to_string(index=False))

        if args.export_sweep:
            results.to_csv(args.export_sweep, index=False)
            print("wrote:", args.export_sweep, "| shape:", results.shape)

        if args.plots:
            # inertia + silhouette vs n_components (two separate simple plots)
            plt.figure(figsize=(7, 4))
            plt.plot(results.sort_values("n_components")["n_components"],
                     results.sort_values("n_components")["silhouette"])
            plt.xlabel("n_components")
            plt.ylabel("silhouette")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(7, 4))
            plt.plot(results.sort_values("n_components")["n_components"],
                     results.sort_values("n_components")["inertia"])
            plt.xlabel("n_components")
            plt.ylabel("inertia")
            plt.tight_layout()
            plt.show()

        if args.auto_set:
            best = pick_best_row(results, prefer=args.prefer)
            args.pca_components = int(best["n_components"])
            print("auto_set pca_components =", args.pca_components)

    if args.sweep_k:
        ncomp = args.pca_components
        if ncomp is None:
            X_df, X_scaled = scale_features(df, drop_cols=["target", "ID"])
            ncomp, _ = choose_n_components_by_variance(X_scaled, threshold=args.pca_threshold)

        results = sweep_kmeans_k(
            df,
            n_components=int(ncomp),
            k_min=args.k_min,
            k_max=args.k_max,
            seed=args.seed,
        )
        print("k_sweep (for n_components=%d):" % int(ncomp))
        print(results.to_string(index=False))

        if args.export_sweep:
            results.to_csv(args.export_sweep, index=False)
            print("wrote:", args.export_sweep, "| shape:", results.shape)

        if args.plots:
            plt.figure(figsize=(7, 4))
            plt.plot(results["n_clusters"], results["inertia"])
            plt.xlabel("k (n_clusters)")
            plt.ylabel("inertia (elbow)")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(7, 4))
            plt.plot(results["n_clusters"], results["silhouette"])
            plt.xlabel("k (n_clusters)")
            plt.ylabel("silhouette")
            plt.tight_layout()
            plt.show()

        if args.auto_set:
            best_k = int(results.sort_values("silhouette", ascending=False).iloc[0]["n_clusters"])
            args.n_clusters = best_k
            print("auto_set n_clusters =", args.n_clusters)


    if args.basic:
        print("shape:", df.shape)
        print("columns:", df.columns.tolist()[:30], "...")
        print("target_value_counts_incl_nan:")
        print(df["target"].value_counts(dropna=False))
        print("target_missing_rate:")
        print(float(df["target"].isna().mean()))

    if args.missing:
        miss_counts, miss_pct = missing_report(df)
        print("columns_with_missing_values:")
        print(miss_counts)
        print("missing_percent:")
        print(miss_pct)

    X_df = None
    X_pca = None

    if need_cluster:
        X_df, X_scaled = scale_features(df, drop_cols=["target", "ID"])
        if args.pca_components is None:
            best_n, cum = choose_n_components_by_variance(X_scaled, threshold=args.pca_threshold)
        else:
            best_n = int(args.pca_components)
            pca_full = PCA().fit(X_scaled)
            cum = np.cumsum(pca_full.explained_variance_ratio_)

        if args.cluster and not args.basic:
            print("pca_n_components:", int(best_n))

        if args.plots:
            plt.figure(figsize=(7, 4))
            plt.plot(cum)
            plt.axhline(args.pca_threshold, linestyle="--")
            plt.axvline(best_n, linestyle="--")
            plt.xlabel("n_components")
            plt.ylabel("cumulative_explained_variance")
            plt.tight_layout()
            plt.show()

        pca, X_pca = fit_pca(X_scaled, n_components=best_n, random_state=args.seed)
        km, labels = fit_kmeans(X_pca, n_clusters=args.n_clusters, random_state=args.seed)
        df["cluster"] = labels

        if args.cluster:
            print("explained_variance_sum:", float(pca.explained_variance_ratio_.sum()))
            print("kmeans_silhouette:", float(silhouette_score(X_pca, labels)))
            print("cluster_counts:")
            print(df["cluster"].value_counts().sort_index())

        if args.plots:
            plt.figure(figsize=(8, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=10, alpha=0.8, cmap="tab10")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("KMeans clusters on first two PCA components")
            plt.tight_layout()
            plt.show()

        prob_c, pred_c, mu_c, _ = infer_targets_from_clusters(df, cluster_col="cluster", target_col="target", threshold=0.5)
        df["target_prob_cluster"] = prob_c
        df["target_pred_cluster"] = pred_c

        if args.cluster_means:
            print("cluster_target_means_labeled_only:")
            print(mu_c.sort_index().to_string())

        if args.cluster_summary:
            summ = cluster_feature_summary(df.join(X_df), feature_cols=X_df.columns.tolist(), cluster_col="cluster", top_n=5)
            print("cluster_feature_summary:")
            print(summ.to_string())

        if args.export_cluster_submit is not None:
            unlabeled = df["target"].isna()
            sub = make_submit(df, unlabeled, id_col="ID", pred_col="target_pred_cluster", out_path=args.export_cluster_submit)
            print("wrote:", args.export_cluster_submit, "| shape:", sub.shape)

    if need_rf:
        known = df["target"].notna()
        df_lab = df.loc[known].copy()
        y = df_lab["target"].astype(int)

        drop_cols = ["target", "ID", "cluster", "target_prob_cluster", "target_pred_cluster", "target_prob_rf", "target_pred_rf"]
        X_train = df_lab.drop(columns=drop_cols, errors="ignore")

        if args.cv:
            auc_m, auc_s, acc_m, acc_s = cv_random_forest_auc_acc(
                X_train, y, n_splits=5, random_state=args.seed, n_estimators=500, min_samples_leaf=5
            )
            print("rf_cv_auc_mean:", auc_m)
            print("rf_cv_auc_std:", auc_s)
            print("rf_cv_acc_mean:", acc_m)
            print("rf_cv_acc_std:", acc_s)

        unlabeled = df["target"].isna()
        df_unlab = df.loc[unlabeled].copy()
        X_test = df_unlab.drop(columns=drop_cols, errors="ignore")

        rf, proba_rf, pred_rf = fit_predict_random_forest(
            X_train=X_train, y_train=y, X_test=X_test, random_state=args.seed, n_estimators=500, min_samples_leaf=5
        )
        df.loc[unlabeled, "target_prob_rf"] = proba_rf
        df.loc[unlabeled, "target_pred_rf"] = pred_rf

        if args.export_rf_submit is not None:
            sub = make_submit(df, unlabeled, id_col="ID", pred_col="target_pred_rf", out_path=args.export_rf_submit)
            print("wrote:", args.export_rf_submit, "| shape:", sub.shape)

        if args.feature_importance is not None:
            top = int(args.feature_importance)
            print("rf_feature_importances_top" + str(top) + ":")
            print(rf_feature_importances(rf, X_train.columns, top_n=top).to_string())

    if args.disagreement:
        dis = disagreement_summary_unlabeled(df, cluster_col="cluster")
        print("disagreement_summary_unlabeled:")
        print(dis.to_string())

    if args.trait_diffs:
        traits = ["OPN", "CSN", "EST", "EXT", "AGR", "OPN_time", "CSN_time", "EST_time", "EXT_time", "AGR_time"]
        traits = [t for t in traits if t in df.columns]
        diff = rf_trait_diff_by_cluster_unlabeled(df, traits=traits, cluster_col="cluster")
        print("rf_trait_diff_by_cluster_unlabeled:")
        print(diff.to_string())


if __name__ == "__main__":
    main()
