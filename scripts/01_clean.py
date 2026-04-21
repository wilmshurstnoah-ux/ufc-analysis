"""
01_clean.py — UFC Data Cleaning Script — BEE2041 Data Science in Economics
=======================================
Takes the raw UFC dataset from data/raw/ufcdataset.csv and produces
a cleaned, analysis-ready CSV at data/cleaned/ufc_cleaned.csv.

Run from the project root:
    python scripts/01_clean.py

Author: [YOUR STUDENT NUMBER]
"""

import pandas as pd
import numpy as np
import os

# ── 0. FILE PATHS ──────────────────────────────────────────────────────────
RAW_PATH = os.path.join("data", "raw", "ufcdataset.csv")
CLEAN_PATH = os.path.join("data", "cleaned", "ufc_cleaned.csv")

# Make sure the output folder exists
os.makedirs(os.path.join("data", "cleaned"), exist_ok=True)

# ── 1. LOAD RAW DATA ──────────────────────────────────────────────────────
print("Loading raw data...")
df = pd.read_csv(RAW_PATH)
rows_before = len(df)
cols_before = df.shape[1]
print(f"  Loaded {rows_before} rows, {cols_before} columns")

# ── 2. REMOVE DRAWS AND NO CONTESTS ───────────────────────────────────────
# We only want fights with a clear winner (red or blue corner).
# Draws (16 fights) and no contests (24 fights) can't be used for
# predicting win/loss, so we drop them.
df = df[df["winner"].isin(["red", "blue"])].copy()
print(f"  Removed draws/no contests: {rows_before - len(df)} rows dropped")

# ── 3. PARSE DATES ────────────────────────────────────────────────────────
# The Date column is in MM/DD/YYYY format. Convert to proper datetime
# so we can calculate fighter age at the time of each fight.
df["date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
df = df.drop(columns=["Date"])
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ── 4. CREATE THE WIN INDICATOR ───────────────────────────────────────────
# For our analysis, we look at things from the RED corner's perspective.
# r_win = 1 if red corner won, 0 if blue corner won.
df["r_win"] = (df["winner"] == "red").astype(int)

# ── 5. SIMPLIFY WIN METHOD ────────────────────────────────────────────────
# The winby column already has clean categories: KO/TKO, SUB, DEC.
# We rename for clarity and drop the 36 fights where winby is missing.
df = df.dropna(subset=["winby"])
df["win_method"] = df["winby"].replace({
    "KO/TKO": "KO/TKO",
    "SUB": "Submission",
    "DEC": "Decision"
})
print(f"  Win methods: {df['win_method'].value_counts().to_dict()}")

# ── 6. CREATE WEIGHT CLASS FROM WEIGHT ────────────────────────────────────
# The dataset has fighter weights in kg but no weight class column.
# We map each weight to the standard UFC weight class name.
WEIGHT_CLASS_MAP = {
    52: "Strawweight",
    56: "Flyweight",
    61: "Bantamweight",
    65: "Featherweight",
    70: "Lightweight",
    77: "Welterweight",
    84: "Middleweight",
    93: "Light Heavyweight",
    120: "Heavyweight"   # We'll group all 100kg+ as Heavyweight
}

def assign_weight_class(weight_kg):
    """Map a fighter's weight in kg to their UFC weight class."""
    if pd.isna(weight_kg):
        return np.nan
    # Find the closest standard weight class
    standard_weights = [52, 56, 61, 65, 70, 77, 84, 93, 120]
    closest = min(standard_weights, key=lambda w: abs(w - weight_kg))
    # For heavyweights (100kg+), always map to Heavyweight
    if weight_kg >= 100:
        closest = 120
    return WEIGHT_CLASS_MAP[closest]

df["weight_class"] = df["R_Weight"].apply(assign_weight_class)
print(f"  Weight classes: {df['weight_class'].value_counts().to_dict()}")

# ── 7. CREATE PHYSICAL ADVANTAGE VARIABLES ────────────────────────────────
# Height advantage: how much taller (in cm) the red corner is vs blue.
# Positive = red corner is taller. Negative = blue corner is taller.
df["height_diff"] = df["R_Height"] - df["B_Height"]

# Note: this dataset does NOT include reach data, so we use height
# difference as a proxy for physical size advantage. We mention this
# limitation in the blog post.

# Age difference: how much older the red corner fighter is.
df["age_diff"] = df["R_Age"] - df["B_Age"]

# ── 8. AGGREGATE ROUND-BY-ROUND STATS INTO TOTALS ─────────────────────────
# The raw data has stats broken down by round (Round1-5). We sum across
# all rounds to get total fight stats. This makes the data much simpler
# and easier to work with.
print("  Aggregating round-by-round stats into totals...")

def sum_across_rounds(df, prefix, stat_name):
    """Sum a stat across rounds 1-5 for a given corner (R_ or B_)."""
    round_cols = []
    for r in range(1, 6):
        col = f"{prefix}_Round{r}_{stat_name}"
        if col in df.columns:
            round_cols.append(col)
    if round_cols:
        return df[round_cols].sum(axis=1, min_count=1)  # NaN if all rounds are NaN
    return np.nan

# Key stats we want totals for (these are the most important for analysis)
stats_to_aggregate = {
    "Strikes_Significant Strikes_Landed": "sig_strikes_landed",
    "Strikes_Significant Strikes_Attempts": "sig_strikes_attempted",
    "Strikes_Total Strikes_Landed": "total_strikes_landed",
    "Strikes_Total Strikes_Attempts": "total_strikes_attempted",
    "Strikes_Knock Down_Landed": "knockdowns",
    "Grappling_Takedowns_Landed": "takedowns_landed",
    "Grappling_Takedowns_Attempts": "takedowns_attempted",
    "Grappling_Submissions_Attempts": "sub_attempts",
    "Strikes_Head Significant Strikes_Landed": "head_sig_strikes",
    "Strikes_Body Significant Strikes_Landed": "body_sig_strikes",
    "Strikes_Distance Strikes_Landed": "distance_strikes",
    "Strikes_Clinch Significant Strikes_Landed": "clinch_sig_strikes",
    "Strikes_Ground Significant Strikes_Landed": "ground_sig_strikes",
    "TIP_Control Time": "control_time",
}

# Create totals for both red (R) and blue (B) corners
for raw_stat, clean_name in stats_to_aggregate.items():
    df[f"r_{clean_name}"] = sum_across_rounds(df, "R_", raw_stat)
    df[f"b_{clean_name}"] = sum_across_rounds(df, "B_", raw_stat)

# ── 9. CREATE ACCURACY PERCENTAGES ────────────────────────────────────────
# Striking accuracy = strikes landed / strikes attempted (as a %)
df["r_sig_strike_accuracy"] = (
    df["r_sig_strikes_landed"] / df["r_sig_strikes_attempted"].replace(0, np.nan) * 100
)
df["b_sig_strike_accuracy"] = (
    df["b_sig_strikes_landed"] / df["b_sig_strikes_attempted"].replace(0, np.nan) * 100
)

# Takedown accuracy
df["r_takedown_accuracy"] = (
    df["r_takedowns_landed"] / df["r_takedowns_attempted"].replace(0, np.nan) * 100
)
df["b_takedown_accuracy"] = (
    df["b_takedowns_landed"] / df["b_takedowns_attempted"].replace(0, np.nan) * 100
)

# ── 10. CREATE DIFFERENTIAL STATS (RED MINUS BLUE) ────────────────────────
# These differentials are useful for the regression model. They capture
# how much better/worse one fighter was than the other in each stat.
df["sig_strike_diff"] = df["r_sig_strikes_landed"] - df["b_sig_strikes_landed"]
df["takedown_diff"] = df["r_takedowns_landed"] - df["b_takedowns_landed"]
df["knockdown_diff"] = df["r_knockdowns"] - df["b_knockdowns"]
df["control_time_diff"] = df["r_control_time"] - df["b_control_time"]
df["strike_accuracy_diff"] = df["r_sig_strike_accuracy"] - df["b_sig_strike_accuracy"]

# ── 11. SELECT AND RENAME FINAL COLUMNS ───────────────────────────────────
# Drop all the original round-by-round columns (894 columns → ~50 clean ones)
# Keep only what we actually need for analysis.

keep_cols = [
    # Fight identifiers
    "Fight_ID", "Event_ID", "date",
    # Fighter info
    "R_Name", "B_Name", "R_Age", "B_Age",
    "R_Height", "B_Height", "R_Weight", "B_Weight",
    "weight_class",
    # Experience
    "RPrev", "BPrev", "BStreak",
    # Outcome
    "winner", "r_win", "win_method",
    "Last_round", "Max_round",
    # Physical advantages
    "height_diff", "age_diff",
    # Red corner fight stats
    "r_sig_strikes_landed", "r_sig_strikes_attempted", "r_sig_strike_accuracy",
    "r_total_strikes_landed", "r_total_strikes_attempted",
    "r_knockdowns", "r_takedowns_landed", "r_takedowns_attempted",
    "r_takedown_accuracy", "r_sub_attempts",
    "r_head_sig_strikes", "r_body_sig_strikes",
    "r_distance_strikes", "r_clinch_sig_strikes", "r_ground_sig_strikes",
    "r_control_time",
    # Blue corner fight stats
    "b_sig_strikes_landed", "b_sig_strikes_attempted", "b_sig_strike_accuracy",
    "b_total_strikes_landed", "b_total_strikes_attempted",
    "b_knockdowns", "b_takedowns_landed", "b_takedowns_attempted",
    "b_takedown_accuracy", "b_sub_attempts",
    "b_head_sig_strikes", "b_body_sig_strikes",
    "b_distance_strikes", "b_clinch_sig_strikes", "b_ground_sig_strikes",
    "b_control_time",
    # Differentials (for regression)
    "sig_strike_diff", "takedown_diff", "knockdown_diff",
    "control_time_diff", "strike_accuracy_diff",
]

# Only keep columns that actually exist
keep_cols = [c for c in keep_cols if c in df.columns]
df_clean = df[keep_cols].copy()

# ── 12. CLEAN UP COLUMN NAMES ─────────────────────────────────────────────
# Make all column names lowercase with underscores (no spaces, no caps)
df_clean.columns = (
    df_clean.columns
    .str.lower()
    .str.replace(" ", "_")
)

# ── 13. HANDLE REMAINING MISSING VALUES ───────────────────────────────────
# For physical stats (height, weight, age): drop rows where these are missing,
# since they are central to our analysis.
before_drop = len(df_clean)
df_clean = df_clean.dropna(subset=["r_height", "b_height", "r_weight", "b_weight"])
print(f"  Dropped {before_drop - len(df_clean)} rows with missing physical stats")

# For fight stats: fill remaining NaN with 0 (if a fighter landed 0 strikes
# in a round that wasn't recorded, 0 is a reasonable assumption).
stat_cols = [c for c in df_clean.columns if c.startswith(("r_", "b_")) and c not in
             ["r_name", "b_name", "r_age", "b_age", "r_height", "b_height",
              "r_weight", "b_weight", "r_win"]]
df_clean[stat_cols] = df_clean[stat_cols].fillna(0)

# Fill differential columns
diff_cols = [c for c in df_clean.columns if c.endswith("_diff")]
df_clean[diff_cols] = df_clean[diff_cols].fillna(0)

# ── 14. SAVE CLEANED DATA ─────────────────────────────────────────────────
df_clean.to_csv(CLEAN_PATH, index=False)

# ── 15. PRINT SUMMARY ─────────────────────────────────────────────────────
rows_after = len(df_clean)
cols_after = df_clean.shape[1]

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)
print(f"  Rows:    {rows_before} → {rows_after}  (dropped {rows_before - rows_after})")
print(f"  Columns: {cols_before} → {cols_after}  (from {cols_before} round-level columns)")
print(f"  Date range: {str(df_clean['date'].min())[:10]} to {str(df_clean['date'].max())[:10]}")
print(f"  Saved to: {CLEAN_PATH}")
print(f"\n  Weight classes:")
for wc, count in df_clean["weight_class"].value_counts().sort_values(ascending=False).items():
    print(f"    {wc}: {count} fights")
print(f"\n  Win methods:")
for wm, count in df_clean["win_method"].value_counts().items():
    print(f"    {wm}: {count} fights")
print(f"\n  Red corner win rate: {df_clean['r_win'].mean():.1%}")
print(f"\n  Sample of cleaned data:")
print(df_clean[["r_name", "b_name", "weight_class", "win_method", "r_win",
                "height_diff", "r_sig_strikes_landed"]].head(5).to_string())
print("=" * 60)
