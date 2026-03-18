"""
Greyhound Race Predictor - Top 4 per race.

Adapts the predictive-models project methodology:
  1. ELO-style Speed Rating  (cf. AFL/NRL/EPL elo_applier)
  2. EWMA Form Score          (cf. AFL create_exp_weighted_avgs)
  3. Box Advantage Factor      (cf. NRL/Super Rugby HGA factor)
  4. Consistency Rating        (cf. NRL rolling ave_margin window)
  5. Feature Differentials     (cf. AFL diff_df / EPL feature diffs)
  6. Composite probability     (cf. NRL elo_prob + trueskill_mu blend)

All scores normalised within each race field (differential vs field average),
then combined into a single win-probability estimate per runner.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

AEST = timezone(timedelta(hours=11))

# ──────────────────────────────────────────────
# 1. SPEED RATING  (ELO-style)
#    Instead of Elo updates over time, we compute a "speed Elo" per runner
#    relative to the field.  A dog's best_time is converted to an Elo-like
#    rating:  rating = BASE - (time * scale).  Lower time → higher rating.
#    Then prob = 1 / (1 + 10^((opp_rating - rating) / 400))  averaged across
#    all opponents in the field, exactly like the project's Elo formula.
# ──────────────────────────────────────────────
ELO_BASE = 1500
ELO_SCALE = 50   # rating points per second of best_time


def speed_elo_rating(best_time_sec, distance_m):
    """Convert a best time to an Elo-style rating, normalised per 100m."""
    if best_time_sec is None or np.isnan(best_time_sec):
        return None
    pace = best_time_sec / (distance_m / 100)  # sec per 100m
    return ELO_BASE - (pace * ELO_SCALE)


def elo_win_prob(rating_a, rating_b):
    """Standard Elo probability: P(A beats B)."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def compute_speed_score(group):
    """For each runner, average Elo win-prob vs every other runner in the race."""
    ratings = group["speed_rating"].values
    n = len(ratings)
    scores = []
    for i in range(n):
        if np.isnan(ratings[i]):
            scores.append(np.nan)
            continue
        probs = []
        for j in range(n):
            if i != j and not np.isnan(ratings[j]):
                probs.append(elo_win_prob(ratings[i], ratings[j]))
        scores.append(np.mean(probs) if probs else 0.5)
    group["speed_score"] = scores
    return group


# ──────────────────────────────────────────────
# 2. FORM SCORE  (EWMA-inspired)
#    last_4_starts string e.g. "1533" → positions [1,5,3,3].
#    We apply exponential weighting (most recent = highest weight),
#    mirroring the project's ewm(span=N).mean().shift(1) pattern.
#    Score = weighted average of (1 - (pos-1)/max_pos), so 1st → 1.0.
# ──────────────────────────────────────────────
FORM_SPAN = 4  # equivalent to ewm span in project
MAX_FIELD = 10  # typical max runners


def parse_form(form_str):
    """Parse last_4_starts string into list of finishing positions.
    Handles digits and letters: F=fall(10), digits as-is."""
    positions = []
    if not isinstance(form_str, str):
        return positions
    for ch in form_str:
        if ch.isdigit():
            positions.append(int(ch))
        elif ch.upper() == 'F':
            positions.append(MAX_FIELD)
        elif ch.upper() == 'T':
            positions.append(MAX_FIELD)  # Trial / did not place
    return positions


def ewma_form_score(positions):
    """EWMA-weighted form score.  Most recent start has highest weight.
    Mirrors: row.ewm(span=FORM_SPAN).mean()  from AFL feature creation."""
    if not positions:
        return 0.5  # neutral
    # Convert positions to scores: 1st → 1.0, 10th → 0.0
    scores = [max(0, 1 - (p - 1) / (MAX_FIELD - 1)) for p in positions]
    # Apply exponential weights (newest last in the string = index -1)
    alpha = 2 / (FORM_SPAN + 1)
    weights = [(1 - alpha) ** i for i in range(len(scores))]
    weights.reverse()  # most recent gets highest weight
    return np.average(scores, weights=weights)


# ──────────────────────────────────────────────
# 3. BOX ADVANTAGE  (Home Ground Advantage analogue)
#    In the project, HGA adds flat rating points (NRL: +20, Super Rugby: +90).
#    For greyhounds, inside boxes (1-2) have a statistical edge in shorter
#    races; middle boxes are best over longer distances.
#    We model this as a small additive bonus (like HGA Elo points).
# ──────────────────────────────────────────────

# Box advantage lookup: (distance_category, box) → bonus score [0, 1]
# Based on well-known greyhound racing statistics
BOX_ADVANTAGE = {
    "short": {1: 0.80, 2: 0.70, 3: 0.55, 4: 0.50, 5: 0.50, 6: 0.45, 7: 0.55, 8: 0.65, 9: 0.30, 10: 0.30},
    "middle": {1: 0.75, 2: 0.65, 3: 0.55, 4: 0.50, 5: 0.50, 6: 0.50, 7: 0.55, 8: 0.60, 9: 0.30, 10: 0.30},
    "long":   {1: 0.65, 2: 0.60, 3: 0.55, 4: 0.55, 5: 0.55, 6: 0.55, 7: 0.55, 8: 0.55, 9: 0.30, 10: 0.30},
}


def distance_category(distance_m):
    if distance_m <= 350:
        return "short"
    elif distance_m <= 450:
        return "middle"
    else:
        return "long"


def box_advantage_score(box, dist_m):
    cat = distance_category(dist_m)
    return BOX_ADVANTAGE[cat].get(box, 0.40)


# ──────────────────────────────────────────────
# 4. CONSISTENCY RATING  (Rolling window approach)
#    Mirrors NRL's ave_margin rolling window.  We look at the variance
#    in recent finishing positions.  Low variance = consistent = bonus.
# ──────────────────────────────────────────────

def consistency_score(positions):
    """Lower variance in recent finishes → higher consistency score."""
    if len(positions) < 2:
        return 0.5
    std = np.std(positions)
    # Normalise: std of 0 → score 1.0, std of 4+ → score ~0.2
    return max(0.1, 1 - std / 5)


# ──────────────────────────────────────────────
# 5. COMPOSITE SCORE  (Feature blend)
#    Like the project combines elo_prob, trueskill_mu, ave_margin
#    as features into an ML model, we combine our scores with optimised
#    weights into a single composite rating per runner.
# ──────────────────────────────────────────────

WEIGHTS = {
    "speed":       0.35,  # Best time (ELO-style)
    "form":        0.30,  # Recent form (EWMA)
    "box":         0.15,  # Box draw advantage (HGA)
    "consistency": 0.20,  # Consistency (rolling window)
}


def normalise_within_race(series):
    """Min-max normalise within a race field (feature differential approach)."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


# ──────────────────────────────────────────────
# MAIN PREDICTION PIPELINE
# ──────────────────────────────────────────────

def predict(csv_path):
    df = pd.read_csv(csv_path)

    # Parse distance to metres
    df["distance_m"] = df["distance"].str.replace("m", "").astype(float)

    # Parse best_time to seconds
    def parse_time(t):
        if pd.isna(t) or t == "NBT":
            return np.nan
        try:
            return float(t)
        except ValueError:
            return np.nan

    df["best_time_sec"] = df["best_time"].apply(parse_time)

    # --- Feature 1: Speed Rating (ELO-style) ---
    df["speed_rating"] = df.apply(
        lambda r: speed_elo_rating(r["best_time_sec"], r["distance_m"]), axis=1
    )
    # Fill missing speed ratings with race-field median (conservative estimate)
    race_key = ["venue", "race_number"]
    df["speed_rating"] = df.groupby(race_key)["speed_rating"].transform(
        lambda s: s.fillna(s.median())
    )
    # Compute speed score per race (Elo win-prob vs each opponent)
    speed_scores = []
    for _, group in df.groupby(race_key):
        ratings = group["speed_rating"].values
        n = len(ratings)
        for i in range(n):
            if np.isnan(ratings[i]):
                speed_scores.append(np.nan)
                continue
            probs = []
            for j in range(n):
                if i != j and not np.isnan(ratings[j]):
                    probs.append(elo_win_prob(ratings[i], ratings[j]))
            speed_scores.append(np.mean(probs) if probs else 0.5)
    df["speed_score"] = speed_scores

    # --- Feature 2: Form Score (EWMA) ---
    df["form_positions"] = df["last_4_starts"].apply(parse_form)
    df["form_score"] = df["form_positions"].apply(ewma_form_score)

    # --- Feature 3: Box Advantage ---
    df["box_score"] = df.apply(
        lambda r: box_advantage_score(r["box"], r["distance_m"]), axis=1
    )

    # --- Feature 4: Consistency ---
    df["consist_score"] = df["form_positions"].apply(consistency_score)

    # --- Normalise each feature within race (differential approach) ---
    for col in ["speed_score", "form_score", "box_score", "consist_score"]:
        df[col + "_norm"] = df.groupby(race_key)[col].transform(normalise_within_race)

    # --- Composite Score ---
    df["composite"] = (
        WEIGHTS["speed"]       * df["speed_score_norm"]
        + WEIGHTS["form"]      * df["form_score_norm"]
        + WEIGHTS["box"]       * df["box_score_norm"]
        + WEIGHTS["consistency"] * df["consist_score_norm"]
    )

    # --- Convert to implied probability (like project's 1/odds) ---
    df["win_prob"] = df.groupby(race_key)["composite"].transform(
        lambda s: s / s.sum()
    )
    df["implied_odds"] = 1 / df["win_prob"]

    return df


def get_top4_per_race(df):
    """Return top 4 ranked runners per race."""
    race_key = ["venue", "race_number"]
    top4 = (
        df.sort_values(["venue", "race_number", "composite"], ascending=[True, True, False])
        .groupby(race_key, group_keys=False)
        .head(4)
        .assign(predicted_rank=lambda d: d.groupby(race_key).cumcount() + 1)
    )
    return top4


def print_predictions(top4, all_df):
    """Pretty-print predictions grouped by venue and race."""
    races = all_df.groupby(["venue", "state", "race_number", "race_name", "race_time", "distance", "grade"])
    race_info = races.first().reset_index()

    current_venue = None
    for _, info in race_info.sort_values(["venue", "race_number"]).iterrows():
        venue = info["venue"]
        if venue != current_venue:
            print(f"\n{'='*70}")
            print(f"  {venue} ({info['state']})")
            print(f"{'='*70}")
            current_venue = venue

        race_num = info["race_number"]
        race_time = info["race_time"]
        # Parse time for display
        try:
            t = datetime.fromisoformat(race_time)
            time_str = t.strftime("%I:%M %p")
        except Exception:
            time_str = race_time

        print(f"\n  R{race_num} | {time_str} | {info['distance']} {info['grade']}")
        print(f"  {info['race_name']}")
        print(f"  {'─'*60}")
        print(f"  {'Rank':<6}{'Box':<5}{'Dog':<24}{'Score':<8}{'Prob':<8}{'Odds':<8}")
        print(f"  {'─'*60}")

        race_top4 = top4[(top4["venue"] == venue) & (top4["race_number"] == race_num)]
        for _, runner in race_top4.iterrows():
            print(
                f"  {runner['predicted_rank']:<6}"
                f"{runner['box']:<5}"
                f"{runner['dog_name']:<24}"
                f"{runner['composite']:.3f}   "
                f"{runner['win_prob']:.1%}   "
                f"${runner['implied_odds']:.2f}"
            )

    # Summary counts
    n_races = all_df.groupby(["venue", "race_number"]).ngroups
    n_venues = all_df["venue"].nunique()
    print(f"\n{'='*70}")
    print(f"  TOTAL: {n_venues} venues | {n_races} races | Top 4 predicted per race")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "greyhound/data/form_guide_2026-03-18.csv"
    df = predict(csv_path)
    top4 = get_top4_per_race(df)
    print_predictions(top4, df)

    # Save predictions
    out_path = csv_path.replace(".csv", "_predictions.csv")
    keep_cols = [
        "venue", "state", "race_number", "race_name", "race_time",
        "distance", "grade", "box", "dog_name", "trainer", "best_time",
        "last_4_starts", "speed_score", "form_score", "box_score",
        "consist_score", "composite", "win_prob", "implied_odds",
    ]
    top4[keep_cols + ["predicted_rank"]].to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")
