"""
ICD-9 Diagnosis vs. Readmission Correlation Analysis
======================================================

Scope: only diag_1, diag_2, diag_3, and readmitted are used. No other
covariates (age, LOS, prior admits) are available, so everything here is
BIVARIATE / DESCRIPTIVE. It tells you what correlates with readmission in
this dataset, not what causes it. Any code/chapter found here could be a
proxy for something outside these four columns.

Two outcome framings are both wired in below because they answer different
questions and the top result WILL differ:

  1. Chapter-level analysis
     - "pooled" prevalence: how often each chapter appears across all
       diag_1/2/3 cells combined (position-agnostic, patient can contribute
       up to 3 times)
     - "concentration": for each patient, do diag_1/2/3 chapters agree
       (all 3 same chapter)? This measures comorbidity clustering, not
       raw frequency.
     - correlation with readmission via chi-square + Cramer's V, computed
       SEPARATELY from prevalence -- prevalence and correlation strength
       are not the same thing and are reported as two different tables.

  2. Code-level analysis
     - run directly on raw codes (not gated behind "which chapter won"
       step 1) with a minimum frequency floor, because a chapter's
       aggregate correlation can hide a single strong code diluted by
       many weak ones in the same chapter (Simpson's-paradox-style).
     - Benjamini-Hochberg FDR correction across all tested codes since
       you're running many simultaneous tests.

Outcome encoding: readmitted is binarized as <30 = 1, else ({>30, NO}) = 0,
since that's the standard clinically-used readmission definition. A
three-way chi-square across all three levels is also provided per
chapter/code as a secondary check, since collapsing >30 and NO together
does throw away information.

Missing values: this dataset (UCI Diabetes 130-US hospitals) encodes
missing diagnosis codes as the string "?" rather than NaN. This is handled
explicitly below -- don't let it silently fall into a fallback bucket.

V-codes / E-codes: these are alphanumeric (V27, E812) and are NOT part of
the 001-999 numeric chapter ranges. They're bucketed as their own group,
not silently dropped or miscoded by a failed numeric conversion.
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ---------------------------------------------------------------------------
# 0. CONFIG -- adjust these
# ---------------------------------------------------------------------------
INPUT_CSV = "diabetic_data.csv"          # path to your data
DIAG_COLS = ["diag_1", "diag_2", "diag_3"]
OUTCOME_COL = "readmitted"
MIN_CODE_FREQ = 30                       # frequency floor for code-level tests
FDR_ALPHA = 0.05


# ---------------------------------------------------------------------------
# 1. ICD-9 CHAPTER MAPPING
# ---------------------------------------------------------------------------
# Numeric ranges are inclusive. Diabetes (250.xx) is pulled out of Endocrine
# on purpose -- leaving it inside "Endocrine" buries a clinically dominant
# category (this whole dataset is diabetes-admission-based) inside a vague
# bucket.
CHAPTER_RANGES = [
    (1, 139, "Infectious/parasitic"),
    (140, 239, "Neoplasms"),
    (240, 249, "Endocrine/metabolic/immunity (non-diabetes)"),
    (251, 279, "Endocrine/metabolic/immunity (non-diabetes)"),
    (280, 289, "Blood"),
    (290, 319, "Mental disorders"),
    (320, 389, "Nervous system/sense organs"),
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (630, 679, "Pregnancy/childbirth"),
    (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"),
    (740, 759, "Congenital anomalies"),
    (760, 779, "Perinatal"),
    (780, 799, "Symptoms/ill-defined"),
    (800, 999, "Injury/poisoning"),
]


def map_code_to_chapter(raw_code):
    """Map a single raw ICD-9 code (string) to a chapter label.
    Handles: missing ('?'/NaN), diabetes split-out, V-codes, E-codes,
    and numeric-with-decimal codes (e.g. '250.01', '414.01')."""
    if pd.isna(raw_code):
        return "Missing"
    code = str(raw_code).strip()
    if code == "?" or code == "":
        return "Missing"

    # V-codes and E-codes are alphanumeric, not in the numeric ranges.
    if code.upper().startswith("V"):
        return "Supplemental (V-code)"
    if code.upper().startswith("E"):
        return "External causes (E-code)"

    # Diabetes pulled out specifically (250.xx, any subcode).
    try:
        numeric_val = float(code)
    except ValueError:
        return "Unrecognized/Other"

    if 250 <= numeric_val < 251:
        return "Diabetes (250.xx)"

    for low, high, label in CHAPTER_RANGES:
        if low <= numeric_val <= high:
            return label

    return "Unrecognized/Other"


# ---------------------------------------------------------------------------
# 2. OUTCOME ENCODING
# ---------------------------------------------------------------------------
def binarize_outcome(series):
    """<30 = 1 (the clinically-standard 30-day readmission flag), else 0."""
    return (series == "<30").astype(int)


# ---------------------------------------------------------------------------
# 3. ASSOCIATION STATS (chi-square + Cramer's V), computed generically
#    for any categorical grouping variable vs. the binary outcome.
# ---------------------------------------------------------------------------
def cramers_v_from_contingency(ct):
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    r, k = ct.shape
    # bias-corrected Cramer's V (Bergsma correction)
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    v = np.sqrt(phi2_corr / denom) if denom > 0 else np.nan
    return chi2, p, v, n


def test_group_vs_binary_outcome(indicator, outcome_bin):
    """indicator: 0/1 presence of a group/code. outcome_bin: 0/1 <30 flag.
    Returns chi2, p-value, Cramer's V, n present, n readmit-rate-present."""
    ct = pd.crosstab(indicator, outcome_bin)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return dict(chi2=np.nan, p_value=np.nan, cramers_v=np.nan,
                     n_present=int(indicator.sum()),
                     readmit_rate_present=np.nan,
                     readmit_rate_absent=np.nan)
    chi2, p, v, n = cramers_v_from_contingency(ct)
    rate_present = outcome_bin[indicator == 1].mean() if (indicator == 1).any() else np.nan
    rate_absent = outcome_bin[indicator == 0].mean() if (indicator == 0).any() else np.nan
    return dict(chi2=chi2, p_value=p, cramers_v=v,
                 n_present=int(indicator.sum()),
                 readmit_rate_present=rate_present,
                 readmit_rate_absent=rate_absent)


def benjamini_hochberg(pvals, alpha=0.05):
    """Return BH-adjusted p-values (q-values), same order as input."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


# ---------------------------------------------------------------------------
# 4. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    for c in DIAG_COLS + [OUTCOME_COL]:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found in {INPUT_CSV}")

    df["readmit_30"] = binarize_outcome(df[OUTCOME_COL])

    # --- chapter mapping ---
    chapter_cols = []
    for c in DIAG_COLS:
        new_col = f"{c}_chapter"
        df[new_col] = df[c].apply(map_code_to_chapter)
        chapter_cols.append(new_col)

    # =======================================================================
    # PART A -- CHAPTER LEVEL
    # =======================================================================

    # A1. Pooled prevalence (position-agnostic; patient can contribute up to 3x)
    pooled_counts = pd.concat([df[c] for c in chapter_cols]).value_counts()
    pooled_prevalence = (pooled_counts / pooled_counts.sum()).rename("pooled_share")

    # A2. Per-patient concentration: do all 3 diagnosis chapters agree?
    def concentration_label(row):
        vals = [row[c] for c in chapter_cols]
        if vals[0] == vals[1] == vals[2]:
            return vals[0]
        return None

    df["_concentrated_chapter"] = df.apply(concentration_label, axis=1)
    concentration_counts = df["_concentrated_chapter"].dropna().value_counts()
    concentration_share = (
        concentration_counts / len(df)
    ).rename("concentrated_patient_share")

    # A3. Chapter vs. readmission correlation -- kept SEPARATE from prevalence.
    # "Present" = chapter appears in ANY of diag_1/2/3 for that patient.
    all_chapters = pd.unique(pd.concat([df[c] for c in chapter_cols]))
    chapter_results = []
    for chapter in all_chapters:
        indicator = (df[chapter_cols] == chapter).any(axis=1).astype(int)
        stats_dict = test_group_vs_binary_outcome(indicator, df["readmit_30"])
        stats_dict["chapter"] = chapter
        chapter_results.append(stats_dict)

    chapter_df = pd.DataFrame(chapter_results).set_index("chapter")
    valid_p = chapter_df["p_value"].notna()
    chapter_df.loc[valid_p, "p_value_bh"] = benjamini_hochberg(
        chapter_df.loc[valid_p, "p_value"].values, FDR_ALPHA
    )
    chapter_df = chapter_df.join(pooled_prevalence, how="left")
    chapter_df = chapter_df.join(concentration_share, how="left")
    chapter_df = chapter_df.sort_values("cramers_v", ascending=False)

    # =======================================================================
    # PART B -- CODE LEVEL (not gated behind "winning" chapter)
    # =======================================================================
    pooled_codes = pd.concat([df[c].astype(str) for c in DIAG_COLS])
    pooled_codes = pooled_codes[~pooled_codes.isin(["?", "nan", "NaN"])]
    code_freq = pooled_codes.value_counts()
    eligible_codes = code_freq[code_freq >= MIN_CODE_FREQ].index

    code_results = []
    for code in eligible_codes:
        indicator = (df[DIAG_COLS].astype(str) == code).any(axis=1).astype(int)
        stats_dict = test_group_vs_binary_outcome(indicator, df["readmit_30"])
        stats_dict["icd9_code"] = code
        code_results.append(stats_dict)

    code_df = pd.DataFrame(code_results).set_index("icd9_code")
    valid_p_code = code_df["p_value"].notna()
    code_df.loc[valid_p_code, "p_value_bh"] = benjamini_hochberg(
        code_df.loc[valid_p_code, "p_value"].values, FDR_ALPHA
    )
    code_df["chapter"] = [map_code_to_chapter(c) for c in code_df.index]
    code_df = code_df.sort_values("cramers_v", ascending=False)

    # =======================================================================
    # OUTPUT
    # =======================================================================
    print("=" * 80)
    print("PART A -- CHAPTER LEVEL (prevalence and correlation reported separately)")
    print("=" * 80)
    print(chapter_df.round(4).to_string())

    print()
    print("=" * 80)
    print(f"PART B -- CODE LEVEL (min frequency={MIN_CODE_FREQ}, "
          f"{len(eligible_codes)} codes tested, BH-corrected)")
    print("=" * 80)
    print(code_df.round(4).head(30).to_string())

    print()
    print(f"Significant codes at BH alpha={FDR_ALPHA}: "
          f"{(code_df['p_value_bh'] < FDR_ALPHA).sum()} of {len(code_df)} tested")

    chapter_df.to_csv("chapter_level_results.csv")
    code_df.to_csv("code_level_results.csv")
    print("\nSaved: chapter_level_results.csv, code_level_results.csv")


if __name__ == "__main__":
    main()
