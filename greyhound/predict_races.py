"""
Enhanced Greyhound Race Predictor - Top 4 per race.

Uses detailed form data (past 6 races, box history, sectionals, class).
Adapts the predictive-models project methodology:
  1. Speed Rating (ELO-style)     - past race times vs winning times
  2. EWMA Form Score              - recent placings, exponentially weighted
  3. Box Bias                     - dog's actual box win/place % from profile
  4. Class Rating                 - grade changes up/down, distance suitability
  5. Sectional / Early Speed      - 1st sectional times (pace factor)
  6. Consistency / Margins        - rolling margin analysis
  7. Track Fitness                - best time at today's track, days since last run
  8. Composite Probability        - weighted blend → normalised win probability
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timezone, timedelta

AEST = timezone(timedelta(hours=11))
TODAY = datetime(2026, 3, 18, tzinfo=AEST)


# ──────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────

def safe_float(val, default=np.nan):
    if pd.isna(val) or val == "" or val == "NBT":
        return default
    try:
        return float(str(val).replace("$", "").replace("kg", "").strip())
    except (ValueError, TypeError):
        return default


def parse_placing(val):
    """Parse '3rd/8' → (3, 8)."""
    if pd.isna(val) or not isinstance(val, str):
        return (np.nan, np.nan)
    m = val.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
    parts = m.split("/")
    try:
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (np.nan, np.nan)


def elo_win_prob(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def normalise(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


# ──────────────────────────────────────────
# 1. SPEED RATING (ELO-style)
#    Uses ACTUAL past race times (not just best_time).
#    For each of the last 6 races, compute time_behind_winner.
#    EWMA-weight these (most recent = highest weight).
#    Also factor in the winning time to normalise across tracks/conditions.
# ──────────────────────────────────────────
ELO_BASE = 1500
ELO_SCALE = 60  # points per second

def compute_speed_rating(row, distance_m):
    """Build a speed rating from past race times, weighted by recency."""
    times = []
    win_times = []
    weights = []
    alpha = 2 / (6 + 1)  # EWMA span=6

    for i in range(1, 7):
        rt = safe_float(row.get(f"pr{i}_time"))
        wt = safe_float(row.get(f"pr{i}_win_time"))
        dist = safe_float(row.get(f"pr{i}_dist"))
        if np.isnan(rt) or np.isnan(wt) or rt <= 0 or wt <= 0:
            continue
        # Normalise time to per-100m pace if distance differs from today's
        if not np.isnan(dist) and dist > 0:
            pace = rt / (dist / 100)
            win_pace = wt / (dist / 100)
        else:
            pace = rt / (distance_m / 100) if distance_m > 0 else rt
            win_pace = wt / (distance_m / 100) if distance_m > 0 else wt

        times.append(pace)
        win_times.append(win_pace)
        weights.append((1 - alpha) ** (i - 1))

    if not times:
        # Fallback to best_time
        bt = safe_float(row.get("best_time"))
        if not np.isnan(bt) and distance_m > 0:
            pace = bt / (distance_m / 100)
            return ELO_BASE - pace * ELO_SCALE
        return np.nan

    # Weighted average pace relative to winners
    weights = np.array(weights) / sum(weights)
    avg_pace = np.average(times, weights=weights)
    avg_win_pace = np.average(win_times, weights=weights)

    # Rating: faster pace = higher rating
    # Also reward being close to winner (time_behind small)
    time_behind = avg_pace - avg_win_pace
    return ELO_BASE - avg_pace * ELO_SCALE - time_behind * ELO_SCALE * 2


def compute_field_speed_score(ratings):
    """Elo pairwise win-prob vs each other runner."""
    n = len(ratings)
    scores = []
    for i in range(n):
        if np.isnan(ratings[i]):
            scores.append(np.nan)
            continue
        probs = [elo_win_prob(ratings[i], ratings[j])
                 for j in range(n) if i != j and not np.isnan(ratings[j])]
        scores.append(np.mean(probs) if probs else 0.5)
    return scores


# ──────────────────────────────────────────
# 2. FORM SCORE (EWMA of recent placings)
#    Uses actual placing position from past 6 races, not just last_4_starts string.
# ──────────────────────────────────────────

def compute_form_score(row):
    """EWMA of normalised placings across last 6 races."""
    positions = []
    field_sizes = []
    alpha = 2 / (6 + 1)

    for i in range(1, 7):
        placing_str = row.get(f"pr{i}_placing", "")
        pos, field = parse_placing(placing_str)
        if not np.isnan(pos) and not np.isnan(field) and field > 0:
            positions.append(pos)
            field_sizes.append(field)

    if not positions:
        # Fallback to last_4_starts string
        l4 = str(row.get("last_4_starts", ""))
        for ch in l4:
            if ch.isdigit():
                positions.append(int(ch))
                field_sizes.append(8)  # assume 8-dog field
            elif ch.upper() == "F":
                positions.append(8)
                field_sizes.append(8)
        if not positions:
            return 0.5

    # Normalised score: 1st in 8-dog field = 1.0, last = 0.0
    scores = [max(0, 1 - (p - 1) / max(1, f - 1)) for p, f in zip(positions, field_sizes)]
    weights = [(1 - alpha) ** i for i in range(len(scores))]
    return np.average(scores, weights=weights)


# ──────────────────────────────────────────
# 3. BOX BIAS (dog's actual box history)
#    Uses the dog's real win% and place% from their drawn box.
# ──────────────────────────────────────────

def compute_box_bias(row):
    """Use the dog's actual box draw stats for today's box."""
    box_starts = safe_float(row.get("box_starts"), 0)
    box_win_pct = safe_float(row.get("box_win_pct"), 0) / 100
    box_place_pct = safe_float(row.get("box_place_pct"), 0) / 100

    if box_starts >= 3:
        # Confident: use actual data, blend of win and place
        return 0.6 * box_win_pct + 0.4 * box_place_pct
    elif box_starts >= 1:
        # Some data but low sample - blend with generic box advantage
        actual = 0.6 * box_win_pct + 0.4 * box_place_pct
        generic = generic_box_advantage(row.get("box", 1), safe_float(row.get("distance", "0").replace("m", ""), 350))
        return 0.5 * actual + 0.5 * generic
    else:
        # No data for this box - use generic
        return generic_box_advantage(row.get("box", 1), safe_float(row.get("distance", "0").replace("m", ""), 350))


def generic_box_advantage(box, dist_m):
    """Statistical box advantage by distance category."""
    if dist_m <= 350:
        adv = {1: 0.18, 2: 0.15, 3: 0.12, 4: 0.11, 5: 0.10, 6: 0.09, 7: 0.10, 8: 0.13, 9: 0.05, 10: 0.05}
    elif dist_m <= 450:
        adv = {1: 0.16, 2: 0.13, 3: 0.12, 4: 0.11, 5: 0.11, 6: 0.11, 7: 0.12, 8: 0.12, 9: 0.05, 10: 0.05}
    else:
        adv = {1: 0.14, 2: 0.12, 3: 0.12, 4: 0.12, 5: 0.12, 6: 0.12, 7: 0.12, 8: 0.12, 9: 0.05, 10: 0.05}
    return adv.get(int(box), 0.08)


# ──────────────────────────────────────────
# 4. CLASS RATING
#    Grade changes (stepping up vs down), distance suitability.
# ──────────────────────────────────────────

GRADE_MAP = {
    "M": 1, "maiden": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
    "FFA": 7, "Free For All": 7, "Mixed": 5, "Other": 5,
    "M5": 6, "M2/M3": 4, "GM": 2,
}


def grade_to_num(grade_str):
    if pd.isna(grade_str) or not isinstance(grade_str, str):
        return 5  # neutral
    g = grade_str.strip()
    if g in GRADE_MAP:
        return GRADE_MAP[g]
    # Try to extract number
    for key in sorted(GRADE_MAP.keys(), key=len, reverse=True):
        if key.lower() in g.lower():
            return GRADE_MAP[key]
    try:
        return int(g)
    except ValueError:
        return 5


def compute_class_rating(row, today_grade, today_dist_m):
    """Score based on grade history and distance suitability."""
    today_g = grade_to_num(today_grade)
    scores = []
    dist_scores = []

    for i in range(1, 7):
        g = row.get(f"pr{i}_grade", "")
        d = safe_float(row.get(f"pr{i}_dist"))
        if pd.isna(g) or g == "":
            continue
        past_g = grade_to_num(g)
        # Dropping in class = positive, rising = harder
        grade_diff = past_g - today_g
        # Score: if dog was running in higher grades and now drops, that's good
        scores.append(min(1.0, max(0.0, 0.5 + grade_diff * 0.15)))

        # Distance suitability
        if not np.isnan(d) and d > 0 and today_dist_m > 0:
            dist_diff = abs(d - today_dist_m) / today_dist_m
            dist_scores.append(max(0.0, 1.0 - dist_diff * 2))

    class_score = np.mean(scores) if scores else 0.5
    dist_score = np.mean(dist_scores) if dist_scores else 0.5
    return 0.6 * class_score + 0.4 * dist_score


# ──────────────────────────────────────────
# 5. EARLY SPEED / SECTIONAL
#    1st sectional times from past races + speedmap from race page.
# ──────────────────────────────────────────

def compute_early_speed(row):
    """Average 1st sectional from past races (lower = faster = better)."""
    sectionals = []
    alpha = 2 / (4 + 1)  # span=4

    # Try speedmap sectional from race page first
    sm = safe_float(row.get("speedmap_sectional"))
    if not np.isnan(sm) and sm > 0:
        sectionals.append(sm)

    for i in range(1, 7):
        sec = safe_float(row.get(f"pr{i}_sec1"))
        if not np.isnan(sec) and sec > 0:
            sectionals.append(sec)

    if not sectionals:
        return np.nan

    # EWMA weighted
    weights = [(1 - alpha) ** i for i in range(len(sectionals))]
    return np.average(sectionals, weights=weights)


# ──────────────────────────────────────────
# 6. CONSISTENCY / MARGINS
#    How close the dog runs to the winner in past races.
# ──────────────────────────────────────────

def compute_consistency(row):
    """Average margin behind winner (lower = more competitive)."""
    margins = []
    for i in range(1, 7):
        mgn = safe_float(row.get(f"pr{i}_margin"))
        placing = str(row.get(f"pr{i}_placing", ""))
        if np.isnan(mgn):
            continue
        # If dog won, margin is behind 2nd place (positive = good)
        if "1st" in placing:
            margins.append(-mgn)  # negative margin = dog was the winner
        else:
            margins.append(mgn)

    if not margins:
        return np.nan

    # Lower average margin behind winner = better
    return np.mean(margins)


# ──────────────────────────────────────────
# 7. TRACK FITNESS
#    Best time at today's track + days since last race.
# ──────────────────────────────────────────

def compute_track_fitness(row, today_dist_m):
    """Combine track best time advantage + recent race fitness."""
    score = 0.5

    # Track best time advantage
    tbt = safe_float(row.get("track_best_time"))
    bt = safe_float(row.get("best_time"))
    if not np.isnan(tbt) and today_dist_m > 0:
        track_pace = tbt / (today_dist_m / 100)
        score = max(0, 1 - track_pace * 0.15)  # lower pace = better
    elif not np.isnan(bt) and today_dist_m > 0:
        track_pace = bt / (today_dist_m / 100)
        score = max(0, 1 - track_pace * 0.15)

    # Days since last run (fitness)
    pr1_date = row.get("pr1_date", "")
    if isinstance(pr1_date, str) and pr1_date:
        try:
            last_run = datetime.strptime(pr1_date, "%Y-%m-%d").replace(tzinfo=AEST)
            days_since = (TODAY - last_run).days
            # Ideal: 7-14 days. Penalise >28 days or <4 days
            if 7 <= days_since <= 14:
                fitness_bonus = 0.1
            elif 4 <= days_since <= 21:
                fitness_bonus = 0.05
            elif days_since > 35:
                fitness_bonus = -0.1
            else:
                fitness_bonus = 0
            score += fitness_bonus
        except ValueError:
            pass

    return max(0, min(1, score))


# ──────────────────────────────────────────
# 8. COMPOSITE + PREDICTION
# ──────────────────────────────────────────

WEIGHTS = {
    "speed":       0.25,   # Past race times (ELO-style)
    "form":        0.22,   # Recent placings (EWMA)
    "box_bias":    0.12,   # Dog's actual box draw history
    "class":       0.10,   # Grade changes, distance suitability
    "early_speed": 0.10,   # 1st sectional (pace advantage)
    "consistency": 0.11,   # Average margins behind winner
    "track_fit":   0.10,   # Track best time + fitness
}


def predict(csv_path):
    df = pd.read_csv(csv_path)
    race_key = ["venue", "race_number"]

    # Parse distance
    df["distance_m"] = df["distance"].str.replace("m", "").apply(lambda x: safe_float(x, 350))

    # ── Compute features per runner ──
    speed_ratings = []
    form_scores = []
    box_biases = []
    class_ratings = []
    early_speeds = []
    consistencies = []
    track_fits = []

    for _, row in df.iterrows():
        dist_m = row["distance_m"]
        speed_ratings.append(compute_speed_rating(row, dist_m))
        form_scores.append(compute_form_score(row))
        box_biases.append(compute_box_bias(row))
        class_ratings.append(compute_class_rating(row, row.get("grade", ""), dist_m))
        early_speeds.append(compute_early_speed(row))
        consistencies.append(compute_consistency(row))
        track_fits.append(compute_track_fitness(row, dist_m))

    df["speed_rating"] = speed_ratings
    df["form_score"] = form_scores
    df["box_bias"] = box_biases
    df["class_rating"] = class_ratings
    df["early_speed_raw"] = early_speeds
    df["consistency_raw"] = consistencies
    df["track_fitness"] = track_fits

    # Fill NaN speed ratings with field median
    df["speed_rating"] = df.groupby(race_key)["speed_rating"].transform(
        lambda s: s.fillna(s.median())
    )

    # Compute pairwise speed score per race
    all_speed_scores = []
    for _, group in df.groupby(race_key):
        scores = compute_field_speed_score(group["speed_rating"].values)
        all_speed_scores.extend(scores)
    df["speed_score"] = all_speed_scores

    # Early speed: lower sectional = better, so invert for normalisation
    def invert_normalise(s):
        filled = s.fillna(s.median())
        if filled.isna().all():
            return pd.Series(0.5, index=s.index)
        return 1 - normalise(filled)

    df["early_speed"] = df.groupby(race_key)["early_speed_raw"].transform(invert_normalise)

    # Consistency: lower margin = better, so invert
    df["consistency"] = df.groupby(race_key)["consistency_raw"].transform(invert_normalise)

    # Normalise all features within each race
    for col in ["speed_score", "form_score", "box_bias", "class_rating", "early_speed", "consistency", "track_fitness"]:
        df[col + "_norm"] = df.groupby(race_key)[col].transform(normalise)

    # Composite score
    df["composite"] = sum(
        WEIGHTS[key] * df[col + "_norm"]
        for key, col in [
            ("speed", "speed_score"), ("form", "form_score"),
            ("box_bias", "box_bias"), ("class", "class_rating"),
            ("early_speed", "early_speed"), ("consistency", "consistency"),
            ("track_fit", "track_fitness"),
        ]
    )

    # Normalised probability
    df["win_prob"] = df.groupby(race_key)["composite"].transform(lambda s: s / s.sum())
    df["implied_odds"] = 1 / df["win_prob"]

    return df


def get_top4(df):
    race_key = ["venue", "race_number"]
    top4 = (
        df.sort_values(["venue", "race_number", "composite"], ascending=[True, True, False])
        .groupby(race_key, group_keys=False)
        .head(4)
        .assign(predicted_rank=lambda d: d.groupby(race_key).cumcount() + 1)
    )
    return top4


def print_predictions(top4, all_df):
    races = all_df.groupby(["venue", "state", "race_number", "race_name", "race_time", "distance", "grade"])
    race_info = races.first().reset_index()

    current_venue = None
    for _, info in race_info.sort_values(["venue", "race_number"]).iterrows():
        venue = info["venue"]
        if venue != current_venue:
            print(f"\n{'='*80}")
            print(f"  {venue} ({info['state']})")
            print(f"{'='*80}")
            current_venue = venue

        race_num = info["race_number"]
        try:
            t = datetime.fromisoformat(info["race_time"])
            time_str = t.strftime("%I:%M %p")
        except Exception:
            time_str = info["race_time"]

        print(f"\n  R{race_num} | {time_str} | {info['distance']} {info['grade']}")
        print(f"  {info['race_name']}")
        print(f"  {'─'*74}")
        print(f"  {'#':<4}{'Box':<5}{'Dog':<22}{'Speed':<7}{'Form':<7}{'BoxB':<7}{'Class':<7}{'Pace':<7}{'Con':<6}{'TOTAL':<8}{'Prob':<7}{'Odds'}")
        print(f"  {'─'*74}")

        race_top4 = top4[(top4["venue"] == venue) & (top4["race_number"] == race_num)]
        for _, r in race_top4.iterrows():
            print(
                f"  {r['predicted_rank']:<4}"
                f"{r['box']:<5}"
                f"{str(r['dog_name'])[:20]:<22}"
                f"{r['speed_score_norm']:.2f}  "
                f"{r['form_score_norm']:.2f}  "
                f"{r['box_bias_norm']:.2f}  "
                f"{r['class_rating_norm']:.2f}  "
                f"{r['early_speed_norm']:.2f}  "
                f"{r['consistency_norm']:.2f} "
                f"{r['composite']:.3f}  "
                f"{r['win_prob']:.0%}   "
                f"${r['implied_odds']:.1f}"
            )

    n_races = all_df.groupby(race_key).ngroups if (race_key := ["venue", "race_number"]) else 0
    n_venues = all_df["venue"].nunique()
    print(f"\n{'='*80}")
    print(f"  TOTAL: {n_venues} venues | {n_races} races | Top 4 predicted per race")
    print(f"  Weights: Speed={WEIGHTS['speed']}, Form={WEIGHTS['form']}, BoxBias={WEIGHTS['box_bias']}, "
          f"Class={WEIGHTS['class']}, Pace={WEIGHTS['early_speed']}, Consistency={WEIGHTS['consistency']}, "
          f"TrackFit={WEIGHTS['track_fit']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "greyhound/data/detailed_form_2026-03-18.csv"
    df = predict(csv_path)
    top4 = get_top4(df)
    print_predictions(top4, df)

    # Save
    out_path = csv_path.replace(".csv", "_predictions.csv")
    keep = [
        "venue", "state", "race_number", "race_name", "race_time", "distance",
        "grade", "box", "dog_name", "trainer", "best_time", "last_4_starts",
        "trait", "runner_grade", "track_dist_record", "speedmap_sectional",
        "box_starts", "box_wins", "box_win_pct", "box_place_pct",
        "speed_score", "form_score", "box_bias", "class_rating",
        "early_speed", "consistency", "track_fitness", "composite",
        "win_prob", "implied_odds", "predicted_rank",
    ]
    existing = [c for c in keep if c in top4.columns]
    top4[existing].to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")
