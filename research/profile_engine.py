"""
profile_engine.py
=================
The scientific heart of Attune.

Takes ONLY a child's behavioral data (the numbers in each session) and infers
their cognitive profile across the four dimensions. It never looks at the
'hidden' true profile from the simulator -- that's the whole point. In the
real product, this same code would run on data from real games.

Each function returns:
  - a human-readable "type" label
  - the key numbers that led to that label
  - a numeric feature vector (used later for clustering)
"""

import numpy as np


# ---------------------------------------------------------------------------
# DIMENSION 1: attention window
# Logic: find the longest task duration the child could handle before their
# performance clearly degraded. We look for where error rate / response time
# start climbing steeply.
# ---------------------------------------------------------------------------

def infer_attention(attention_rows):
    durations   = np.array([r["duration"]   for r in attention_rows])
    resp_times  = np.array([r["resp_time"]  for r in attention_rows])
    errors      = np.array([r["error_rate"] for r in attention_rows])

    # Use the cleanest baseline: the fastest response time recorded (their best).
    baseline_rt = resp_times.min()
    # "degradation point" = first duration where response time is >15% above best.
    # A tighter threshold catches short-window children who degrade early.
    degraded = durations[resp_times > baseline_rt * 1.15]
    if len(degraded) == 0:
        optimal = durations[-1]        # never degraded -> long window
    else:
        first_bad = degraded[0]
        # optimal window is the last "good" duration before things fell apart
        good = durations[durations < first_bad]
        optimal = good[-1] if len(good) else durations[0]

    if optimal <= 8:
        window_type = "short"
    elif optimal <= 15:
        window_type = "medium"
    else:
        window_type = "hyperfocus"

    return {
        "type": window_type,
        "optimal_minutes": int(optimal),
        "features": [float(optimal), float(errors[-1])],
    }


# ---------------------------------------------------------------------------
# DIMENSION 2: task-switching tolerance
# Logic: the average performance drop right after a switch IS the switching
# cost. Small drop = high tolerance; big drop = low tolerance.
# ---------------------------------------------------------------------------

def infer_switching(switching_rows):
    drops = np.array([r["perf_drop"] for r in switching_rows])
    avg_drop = float(np.mean(drops))

    if avg_drop < 0.20:
        tol_type = "high_tolerance"
    elif avg_drop < 0.40:
        tol_type = "medium_tolerance"
    else:
        tol_type = "low_tolerance"

    recovery = float(np.mean([r["recovery_s"] for r in switching_rows]))
    return {
        "type": tol_type,
        "avg_switch_cost": round(avg_drop, 3),
        "avg_recovery_s": round(recovery, 1),
        "features": [avg_drop, recovery],
    }


# ---------------------------------------------------------------------------
# DIMENSION 3: motivation response
# Logic: whichever reward condition produced the highest engagement is the
# child's dominant motivation type.
# ---------------------------------------------------------------------------

def infer_motivation(motivation_rows):
    best = max(motivation_rows, key=lambda r: r["engagement"])
    # build a feature vector = engagement in each condition, in fixed order
    order = ["progress", "mastery", "social", "challenge", "consistency"]
    eng_by_cond = {r["condition"]: r["engagement"] for r in motivation_rows}
    features = [eng_by_cond.get(c, 0.0) for c in order]
    return {
        "type": best["condition"],
        "top_engagement": round(best["engagement"], 3),
        "features": features,
    }


# ---------------------------------------------------------------------------
# DIMENSION 4: sensory sensitivity
# Logic: compare error rate across audio conditions. If one condition is much
# better than the others, the child is sensitive; if all are similar, they're
# low-sensitivity.
# ---------------------------------------------------------------------------

def infer_sensory(sensory_rows):
    errs = {r["sensory_condition"]: r["error_rate"] for r in sensory_rows}
    best_cond = min(errs, key=errs.get)     # lowest error = best environment
    spread = max(errs.values()) - min(errs.values())

    # The audio test can cleanly separate noise-sensitive children (who do
    # much better in silence) from everyone else. Non-audio sensitivities show
    # up as a flat audio profile, so we group them under "other_or_low".
    if spread < 0.06:
        sens_type = "low_sensitivity"
    elif best_cond in ("silence", "low_music"):
        sens_type = "noise_sensitive"
    else:
        sens_type = "mixed_sensitivity"

    return {
        "type": sens_type,
        "best_environment": best_cond,
        "sensitivity_spread": round(spread, 3),
        "features": [spread],
    }


# ---------------------------------------------------------------------------
# Put all four together into one profile for a child.
# ---------------------------------------------------------------------------

def build_profile(child):
    """Given one child dict from the simulator, return their inferred profile."""
    attention  = infer_attention(child["attention"])
    switching  = infer_switching(child["switching"])
    motivation = infer_motivation(child["motivation"])
    sensory    = infer_sensory(child["sensory"])

    # a single flat feature vector across all dimensions (for clustering)
    feature_vector = (attention["features"] + switching["features"]
                      + motivation["features"] + sensory["features"])

    return {
        "id": child["id"],
        "attention": attention,
        "switching": switching,
        "motivation": motivation,
        "sensory": sensory,
        "feature_vector": feature_vector,
    }


if __name__ == "__main__":
    from simulator import generate_cohort
    cohort = generate_cohort(5)
    for child in cohort:
        prof = build_profile(child)
        print(f"\n{child['id']}")
        print("  TRUE :", child["hidden"])
        print("  GUESS:", {
            "attention": prof["attention"]["type"],
            "switching": prof["switching"]["type"],
            "motivation": prof["motivation"]["type"],
            "sensory": prof["sensory"]["type"],
        })
