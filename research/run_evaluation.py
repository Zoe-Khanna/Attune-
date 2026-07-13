"""
run_evaluation.py
=================
One script that demonstrates the whole engine and prints the numbers you'd
put in a research write-up: how well the profiler recovers hidden profiles,
and whether profile clusters emerge.

Run:  python run_evaluation.py
"""
from simulator import generate_cohort
from profile_engine import build_profile
from clustering import find_best_k


def expected_sensory(hidden_type):
    # The audio test can only cleanly detect noise-sensitivity; other
    # sensory types look "flat" on audio, so we score them as low_sensitivity.
    return "noise_sensitive" if hidden_type == "noise_sensitive" else "low_sensitivity"


def main():
    cohort = generate_cohort(n_children=300, seed=7)
    profiles = [build_profile(c) for c in cohort]

    print("=" * 52)
    print("PROFILE RECOVERY  (engine vs. hidden truth, n=300)")
    print("=" * 52)
    dims = ["attention", "switching", "motivation"]
    hits = {d: 0 for d in dims}
    sens = 0
    for child, prof in zip(cohort, profiles):
        for d in dims:
            if prof[d]["type"] == child["hidden"][d]:
                hits[d] += 1
        if prof["sensory"]["type"] == expected_sensory(child["hidden"]["sensory"]):
            sens += 1
    for d in dims:
        print(f"  {d:12s}: {hits[d]/len(cohort)*100:5.0f}% recovered")
    print(f"  {'sensory':12s}: {sens/len(cohort)*100:5.0f}% recovered")

    print("\n" + "=" * 52)
    print("PROFILE CLUSTERING  (do distinct types emerge?)")
    print("=" * 52)
    for r in find_best_k(profiles):
        bar = "#" * int(r["silhouette"] * 40)
        print(f"  k={r['k']}: silhouette={r['silhouette']:.3f}  {bar}")
    print("\n(higher silhouette = cleaner, more separated profile groups)")


if __name__ == "__main__":
    main()
