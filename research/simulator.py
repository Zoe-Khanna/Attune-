"""
simulator.py
============
Generates synthetic ("pretend") session data for virtual ADHD children.

WHY THIS EXISTS
---------------
Before Zoe has real children to test with, we need data to build and check
the engine. Each virtual child is given a HIDDEN "true profile" (their real
attention window, motivation type, etc). We then simulate them doing tasks.
The profiling engine's job (in profile_engine.py) is to look ONLY at the
behavior and try to recover that hidden profile. If it can, the engine works.

Everything here is intentionally simple and commented for a beginner.
"""

import random
import numpy as np


# ---------------------------------------------------------------------------
# The "menu" of profile types, straight from Zoe's document.
# These are the hidden truths we assign to each virtual child.
# ---------------------------------------------------------------------------

ATTENTION_TYPES = {
    # name: (optimal_minutes, how sharply performance falls off)
    # NOTE: the document also lists a "variable" type, but a single 20-min
    # session can't reliably distinguish it from "short" (both look bumpy).
    # Detecting "variable" would need repeated sessions on different content,
    # so we honestly leave it out of the pilot instrument.
    "short":      {"optimal": 7,  "cliff": "sharp"},
    "medium":     {"optimal": 12, "cliff": "gradual"},
    "hyperfocus": {"optimal": 20, "cliff": "gradual"},
}

SWITCHING_TYPES = {
    # name: how much a task-switch hurts performance (0=no cost, 1=big cost)
    "high_tolerance":    {"cost": 0.10},
    "medium_tolerance":  {"cost": 0.30},
    "low_tolerance":     {"cost": 0.55},
    "context_dependent": {"cost": 0.35},
}

MOTIVATION_TYPES = [
    "progress",    # responds to seeing % complete
    "mastery",     # responds to difficulty going up
    "social",      # responds to sharing accomplishment
    "challenge",   # responds to time/competition
    "consistency", # responds to streaks
]

SENSORY_TYPES = {
    # name: which condition helps most, and how sensitive they are
    "noise_sensitive":      {"best_audio": "silence",     "strength": 0.5},
    "visually_sensitive":   {"best_visual": "minimal",    "strength": 0.5},
    "transition_sensitive": {"best_transition": "slow",   "strength": 0.4},
    "low_sensitivity":      {"best_audio": "any",         "strength": 0.05},
}


def _make_hidden_profile(rng):
    """Roll the dice to create one virtual child's true, hidden profile."""
    return {
        "attention":  rng.choice(list(ATTENTION_TYPES.keys())),
        "switching":  rng.choice(list(SWITCHING_TYPES.keys())),
        "motivation": rng.choice(MOTIVATION_TYPES),
        "sensory":    rng.choice(list(SENSORY_TYPES.keys())),
    }


# ---------------------------------------------------------------------------
# DIMENSION 1: attention window
# We present tasks of 5 durations and record how response time & errors climb
# as the child gets tired. A short-window child degrades fast; a hyperfocus
# child stays strong for much longer.
# ---------------------------------------------------------------------------

def _simulate_attention(hidden, rng):
    atype = hidden["attention"]
    optimal = ATTENTION_TYPES[atype]["optimal"]
    durations = [2, 5, 8, 12, 18]  # minutes, from the document
    rows = []
    for d in durations:
        # "overrun" = how far past their comfortable window this task pushes them
        overrun = max(0, d - optimal)

        # each type degrades at a different steepness ("cliff")
        if atype == "short":
            slope = 130          # falls off fast and hard
        elif atype == "medium":
            slope = 80           # gradual decline
        else:  # hyperfocus
            slope = 45           # very shallow decline, stays strong
        noise = 35

        rt = 700 + overrun * slope + rng.normal(0, noise)
        err = 0.05 + 0.018 * overrun + rng.normal(0, 0.012)
        err = float(np.clip(err, 0, 1))
        drift = max(0, int(overrun * 0.8 + rng.normal(0, 1)))
        rows.append({"duration": d, "resp_time": rt, "error_rate": err, "drift": drift})
    return rows


# ---------------------------------------------------------------------------
# DIMENSION 2: task-switching tolerance
# We alternate tasks and measure the performance drop right after each switch.
# ---------------------------------------------------------------------------

def _simulate_switching(hidden, rng):
    cost = SWITCHING_TYPES[hidden["switching"]]["cost"]
    rows = []
    for freq in ["fast", "medium", "slow"]:
        # more frequent switches accumulate more fatigue
        multiplier = {"fast": 1.3, "medium": 1.0, "slow": 0.7}[freq]
        drop = cost * multiplier + rng.normal(0, 0.04)
        drop = float(np.clip(drop, 0, 1))
        recovery = drop * 60 + rng.normal(0, 5)  # seconds to bounce back
        rows.append({"switch_freq": freq, "perf_drop": drop, "recovery_s": recovery})
    return rows


# ---------------------------------------------------------------------------
# DIMENSION 3: motivation response
# We test 5 reward conditions and record engagement in each. The child's true
# motivation type gets the biggest engagement boost.
# ---------------------------------------------------------------------------

def _simulate_motivation(hidden, rng):
    true_type = hidden["motivation"]
    rows = []
    for cond in MOTIVATION_TYPES:
        base = 0.5  # baseline engagement
        boost = 0.35 if cond == true_type else 0.0
        engagement = base + boost + rng.normal(0, 0.05)
        engagement = float(np.clip(engagement, 0, 1))
        # time-on-task (seconds in a 4-min/240s block) tracks engagement
        time_on_task = engagement * 240
        rows.append({"condition": cond,
                     "engagement": engagement,
                     "time_on_task_s": time_on_task})
    return rows


# ---------------------------------------------------------------------------
# DIMENSION 4: sensory sensitivity
# We vary audio/visual/transition conditions and see which helps performance.
# ---------------------------------------------------------------------------

def _simulate_sensory(hidden, rng):
    info = SENSORY_TYPES[hidden["sensory"]]
    strength = info["strength"]
    rows = []
    audio_conditions = ["silence", "low_music", "white_noise", "background_speech"]
    for cond in audio_conditions:
        base_err = 0.10
        # if this is their preferred audio, errors drop; background speech hurts
        if info.get("best_audio") in (cond, "any"):
            err = base_err - strength * 0.08
        elif cond == "background_speech":
            err = base_err + strength * 0.10
        else:
            err = base_err + strength * 0.03
        err = float(np.clip(err + rng.normal(0, 0.01), 0, 1))
        rows.append({"sensory_condition": cond, "error_rate": err})
    return rows


# ---------------------------------------------------------------------------
# Public function: make a whole cohort of children.
# ---------------------------------------------------------------------------

def generate_cohort(n_children=50, seed=42):
    """
    Returns a list of dicts. Each dict is one child with:
      - 'id'            : a label like "child_03"
      - 'hidden'        : their TRUE profile (engine must not peek at this)
      - 'attention', 'switching', 'motivation', 'sensory' : their behavior data
    """
    rng = np.random.default_rng(seed)
    cohort = []
    for i in range(n_children):
        hidden = _make_hidden_profile(rng)
        child = {
            "id": f"child_{i:02d}",
            "hidden": hidden,
            "attention":  _simulate_attention(hidden, rng),
            "switching":  _simulate_switching(hidden, rng),
            "motivation": _simulate_motivation(hidden, rng),
            "sensory":    _simulate_sensory(hidden, rng),
        }
        cohort.append(child)
    return cohort


if __name__ == "__main__":
    # Quick self-test: make 3 children and print one.
    cohort = generate_cohort(3)
    demo = cohort[0]
    print("Hidden true profile:", demo["hidden"])
    print("Attention behavior rows:")
    for row in demo["attention"]:
        print("  ", row)
