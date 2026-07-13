"""
intervention.py
===============
The product logic: given a child's profile, decide HOW to run their session.

This is the "matching" that Attune's whole thesis rests on -- turning measured
profile into concrete, specific choices (chunk size, reward style, interface).
Also generates the plain-language parent dashboard text from the document.
"""


def match_session_settings(profile):
    """Turn an inferred profile into concrete session settings."""
    attn = profile["attention"]
    switch = profile["switching"]
    motiv = profile["motivation"]["type"]
    sens = profile["sensory"]

    # --- chunk size: never longer than the measured attention window ---
    window = attn["optimal_minutes"]
    chunk_minutes = max(4, min(window - 1, window))  # keep a small safety margin

    # --- break warning timing ---
    warn_seconds = 90 if attn["type"] == "short" else 120

    # --- transition style: low-tolerance kids need a slow, ritual switch ---
    if switch["type"] == "low_tolerance":
        transition = "slow_ritual"      # explicit pause + announcement
    elif switch["type"] == "medium_tolerance":
        transition = "predictable"
    else:
        transition = "flexible"

    # --- reward style straight from motivation type ---
    reward_style = {
        "progress":    "show running completion percentage",
        "mastery":     "show difficulty level increasing",
        "social":      "prompt to share what they learned",
        "challenge":   "offer a personal-best timer",
        "consistency": "show day streak",
    }.get(motiv, "show running completion percentage")

    # --- interface from sensory profile ---
    if sens["type"] == "noise_sensitive":
        audio = "silence"
    else:
        audio = "low instrumental (optional)"
    visual = "minimal" if sens["type"] != "low_sensitivity" else "standard"

    return {
        "chunk_minutes": chunk_minutes,
        "break_warning_seconds": warn_seconds,
        "transition_style": transition,
        "reward_style": reward_style,
        "audio": audio,
        "visual_complexity": visual,
    }


def parent_dashboard_text(profile, settings, child_name="Your child"):
    """Generate the plain-language weekly insight the parent sees."""
    window = profile["attention"]["optimal_minutes"]
    motiv = profile["motivation"]["type"]

    reward_phrase = {
        "progress":    f"Try saying: 'You've done 3 of the 5 problems \u2014 more than halfway.'",
        "mastery":     f"Try saying: 'That question was harder than last time, and you got it.'",
        "social":      f"Try asking: 'Can you teach me the thing you just learned?'",
        "challenge":   f"Try saying: 'Your best was 4:23 \u2014 want to try to beat it?'",
        "consistency": f"Try saying: 'That's {2} days in a row now \u2014 nice streak.'",
    }.get(motiv, "")

    transition_line = ""
    if settings["transition_style"] == "slow_ritual":
        transition_line = (
            f"{child_name} needs a moment to settle after switching subjects. "
            "When moving from maths to reading, try: 'Okay, we're done with maths. "
            "Let's take a breath. Ready for reading?' \u2014 the ritual helps."
        )

    lines = [
        f"{child_name}'s focus is strongest in blocks of about "
        f"{settings['chunk_minutes']} minutes.",
        f"Their strongest motivator is {motiv}. {reward_phrase}",
    ]
    if settings["audio"] == "silence":
        lines.append(f"{child_name} works best in a quiet space \u2014 low background noise helps.")
    if transition_line:
        lines.append(transition_line)
    return "\n".join(lines)


def school_report(profile, settings, child_name="Your child"):
    """One-page, educator-language accommodation summary."""
    return {
        "optimal_session_length": f"under {settings['chunk_minutes'] + 1} minutes",
        "recommended_blocks": f"{settings['chunk_minutes']}-minute tasks with short breaks",
        "motivation_approach": settings["reward_style"],
        "transition_support": (
            "2-3 minute buffer between subjects"
            if settings["transition_style"] == "slow_ritual"
            else "standard transitions are fine"
        ),
        "sensory": f"audio: {settings['audio']}; interface: {settings['visual_complexity']}",
    }


if __name__ == "__main__":
    from simulator import generate_cohort
    from profile_engine import build_profile

    child = generate_cohort(1)[0]
    profile = build_profile(child)
    settings = match_session_settings(profile)

    print("PROFILE (inferred):")
    for dim in ["attention", "switching", "motivation", "sensory"]:
        print(f"  {dim:11s}: {profile[dim]['type']}")
    print("\nMATCHED SESSION SETTINGS:")
    for k, v in settings.items():
        print(f"  {k:22s}: {v}")
    print("\nPARENT DASHBOARD:")
    print(parent_dashboard_text(profile, settings, "Aanya"))
