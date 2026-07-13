"""
clustering.py
=============
Phase 2 of Zoe's research: do consistent profile TYPES actually emerge?

We take every child's numeric feature_vector (from profile_engine) and run
K-means clustering. If the children fall into clean, separated groups, that
supports the central claim: ADHD profiles have structure -- they are not random.

We measure cluster quality with the "silhouette score":
  +1.0 = perfectly separated groups
   0.0 = groups overlap completely
It's a standard, judge-friendly number.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_profiles(profiles, n_clusters=4):
    """
    profiles: list of profile dicts (each has a 'feature_vector')
    Returns: dict with cluster labels, silhouette score, and the scaler/model.
    """
    # 1. Stack all feature vectors into a matrix (one row per child).
    X = np.array([p["feature_vector"] for p in profiles])

    # 2. Standardize: put every feature on the same scale so no single
    #    number (like response time in the hundreds) dominates the distance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Run K-means to find groups.
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    # 4. Score how clean the clusters are.
    score = silhouette_score(X_scaled, labels)

    return {
        "labels": labels,
        "silhouette": round(float(score), 3),
        "n_clusters": n_clusters,
        "X_scaled": X_scaled,
        "model": model,
    }


def find_best_k(profiles, k_range=range(2, 8)):
    """Try several cluster counts and report the silhouette score for each.
    Helps answer 'how many distinct profile types are there?'"""
    X = np.array([p["feature_vector"] for p in profiles])
    X_scaled = StandardScaler().fit_transform(X)
    results = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        results.append({"k": k, "silhouette": round(float(score), 3)})
    return results


if __name__ == "__main__":
    from simulator import generate_cohort
    from profile_engine import build_profile

    cohort = generate_cohort(120)
    profiles = [build_profile(c) for c in cohort]

    print("How many profile types fit best?")
    for r in find_best_k(profiles):
        bar = "#" * int(r["silhouette"] * 40)
        print(f"  k={r['k']}: silhouette={r['silhouette']:.3f}  {bar}")
