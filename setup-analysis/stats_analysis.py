import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency, pointbiserialr
from statsmodels.stats.proportion import proportions_ztest  # Correct import here
from itertools import combinations


def correlation_correct_rt(data):
    data["correct_numerical"] = data["correct"].astype(int)
    return data[["correct_numerical", "rt"]].corr()


def anova_rt_pattern(data):
    model = ols("rt ~ C(pattern)", data=data).fit()
    return sm.stats.anova_lm(model, typ=2)


def chi_square_pattern_correct(data):
    contingency_table = pd.crosstab(data["pattern"], data["correct"])
    return chi2_contingency(contingency_table)


def pairwise_proportion_tests(data):
    unique_patterns = data["pattern"].unique()
    results = []
    total_tests = len(list(combinations(unique_patterns, 2)))
    alpha = 0.05 / total_tests
    for combo in combinations(unique_patterns, 2):
        pattern1_data = data[data["pattern"] == combo[0]]
        pattern2_data = data[data["pattern"] == combo[1]]
        count = [pattern1_data["correct"].sum(), pattern2_data["correct"].sum()]
        nobs = [len(pattern1_data), len(pattern2_data)]
        stat, pval = proportions_ztest(count, nobs)
        significant = "Yes" if pval < alpha else "No"
        results.append(
            {
                "Pattern 1": combo[0],
                "Pattern 2": combo[1],
                "Z-Statistic": stat,
                "P-Value": pval,
                "Significant": significant,
            }
        )
    return pd.DataFrame(results)


def point_biserial_correlation(data):
    pattern_means = data.groupby("pattern")["correct"].mean().rename("pattern_mean")
    data = data.join(pattern_means, on="pattern")
    return pointbiserialr(data["correct"].astype(int), data["pattern_mean"])


# Usage example (Assuming data is loaded)
# data = load_data('path_to_your_dataset.csv')
# print(correlation_correct_rt(data))
# print(anova_rt_pattern(data))
# chi2_stat, p, dof, expected = chi_square_pattern_correct(data)
# print(pairwise_proportion_tests(data))
# print(point_biserial_correlation(data))
