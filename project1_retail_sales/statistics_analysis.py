from scipy import stats
import numpy as np
import pandas as pd


# -------------------------------------------------
# 1️⃣ WEEKEND vs WEEKDAY COMPARISON
# -------------------------------------------------
def t_test_weekend_sales(df):

    weekend = df[df['Weekday'] >= 5]['Sales']
    weekday = df[df['Weekday'] < 5]['Sales']

    n1, n2 = len(weekend), len(weekday)

    if n1 < 3 or n2 < 3:
        return {"error": "Not enough data for statistical testing."}

    # ---------------------------
    # Normality Test (Shapiro)
    # ---------------------------
    shapiro_weekend = stats.shapiro(weekend)
    shapiro_weekday = stats.shapiro(weekday)

    normal = (
        shapiro_weekend.pvalue > 0.05 and
        shapiro_weekday.pvalue > 0.05
    )

    # ---------------------------
    # Variance Test (Levene)
    # ---------------------------
    levene_stat, levene_p = stats.levene(weekend, weekday)
    equal_var = levene_p > 0.05

    # ---------------------------
    # Choose Appropriate Test
    # ---------------------------
    if normal:
        test_stat, p_value = stats.ttest_ind(
            weekend, weekday, equal_var=equal_var
        )
        test_used = "Independent T-Test"
    else:
        test_stat, p_value = stats.mannwhitneyu(
            weekend, weekday, alternative="two-sided"
        )
        test_used = "Mann-Whitney U Test"

    # ---------------------------
    # Effect Size (Cohen's d)
    # ---------------------------
    mean_diff = weekend.mean() - weekday.mean()
    pooled_std = np.sqrt(
        ((weekend.std() ** 2) + (weekday.std() ** 2)) / 2
    )
    cohen_d = mean_diff / pooled_std if pooled_std != 0 else 0

    # Effect size interpretation
    if abs(cohen_d) < 0.2:
        effect_strength = "Small"
    elif abs(cohen_d) < 0.5:
        effect_strength = "Medium"
    else:
        effect_strength = "Large"

    # ---------------------------
    # Confidence Interval (95%)
    # ---------------------------
    se = np.sqrt((weekend.var() / n1) + (weekday.var() / n2))
    margin = 1.96 * se
    ci_lower = mean_diff - margin
    ci_upper = mean_diff + margin

    interpretation = (
        "Statistically Significant Difference"
        if p_value < 0.05
        else "No Statistically Significant Difference"
    )

    return {
        "test_used": test_used,
        "sample_sizes": {"weekend": n1, "weekday": n2},
        "normality_assumption_met": normal,
        "equal_variance_assumed": equal_var,
        "test_statistic": test_stat,
        "p_value": p_value,
        "effect_size_cohens_d": cohen_d,
        "effect_strength": effect_strength,
        "confidence_interval": (ci_lower, ci_upper),
        "interpretation": interpretation
    }


# -------------------------------------------------
# 2️⃣ ANOVA (Multi-group comparison)
# -------------------------------------------------
def category_anova(df):

    groups = [
        group["Sales"].values
        for _, group in df.groupby("Category")
    ]

    if len(groups) < 2:
        return {"error": "Not enough groups for ANOVA."}

    f_stat, p_value = stats.f_oneway(*groups)

    interpretation = (
        "At least one category differs significantly"
        if p_value < 0.05
        else "No significant difference between categories"
    )

    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "interpretation": interpretation
    }


# -------------------------------------------------
# 3️⃣ CORRELATION TEST
# -------------------------------------------------
def correlation_test(df, col1, col2):

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "Invalid column names provided."}

    pearson_corr, pearson_p = stats.pearsonr(df[col1], df[col2])
    spearman_corr, spearman_p = stats.spearmanr(df[col1], df[col2])

    # Strength interpretation
    def interpret_strength(corr):
        abs_corr = abs(corr)
        if abs_corr < 0.3:
            return "Weak"
        elif abs_corr < 0.7:
            return "Moderate"
        else:
            return "Strong"

    return {
        "pearson_correlation": pearson_corr,
        "pearson_p_value": pearson_p,
        "pearson_strength": interpret_strength(pearson_corr),
        "spearman_correlation": spearman_corr,
        "spearman_p_value": spearman_p,
        "spearman_strength": interpret_strength(spearman_corr)
    }
