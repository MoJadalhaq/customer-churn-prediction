"""
Recommendation engine driven by per-customer SHAP values.

For each customer we look at the features that push their probability
*toward churn* (positive SHAP values) and map them to actionable
retention strategies.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Keyword → recommendation mapping
# Keys are lowercase substrings that might appear in feature names.
# The first matching keyword wins, so order more-specific keys first.
# ---------------------------------------------------------------------------
_RECOMMENDATION_MAP: list[tuple[str, str]] = [
    ("tenure",      "Offer a loyalty reward or long-term contract discount to increase retention value."),
    ("contract",    "Propose an upgrade to an annual or two-year contract with a sign-up incentive."),
    ("monthlychar", "Provide a customised bundle discount to reduce the perceived monthly cost."),
    ("totalchar",   "Review billing history and offer a one-time credit or discount adjustment."),
    ("charges",     "Offer a price-match or temporary discount on the current plan."),
    ("techsupport", "Assign a dedicated technical support specialist for the next 90 days."),
    ("tech",        "Provide enhanced technical support or a free onboarding session."),
    ("support",     "Assign a dedicated customer success representative."),
    ("internetser", "Upgrade the internet service tier or offer a speed boost at the current price."),
    ("internet",    "Promote higher-speed internet plans with a limited-time promotional rate."),
    ("fiber",       "Offer a fiber-optic migration package with installation fee waived."),
    ("streaming",   "Bundle streaming services at a discounted rate or offer a free trial upgrade."),
    ("onlinesec",   "Highlight the value of Online Security and offer it at a reduced add-on price."),
    ("onlinebackup","Offer an expanded cloud backup plan or free storage upgrade."),
    ("deviceprot",  "Provide a device protection plan at a discounted rate."),
    ("payment",     "Enable autopay with a 5 % discount to improve payment convenience."),
    ("paperless",   "Enrol the customer in paperless billing for monthly statement credits."),
    ("multiple",    "Offer a multi-line or multi-service bundle discount."),
    ("senior",      "Provide a senior-specific plan with simplified features and pricing."),
    ("partner",     "Offer a family or partner bundle deal."),
    ("dependents",  "Introduce a family plan with dependents discount."),
    ("phone",       "Bundle phone service with the existing plan for a combined discount."),
    ("dsl",         "Suggest upgrading from DSL to a faster service tier."),
    ("gender",      None),   # not actionable — skip
    ("customerid",  None),   # identifier — skip
]

_DEFAULT_RECOMMENDATIONS = [
    "Offer a personalised retention discount based on the customer's usage history.",
    "Schedule a proactive customer success check-in call within the next 7 days.",
    "Send a targeted win-back campaign highlighting new features or plan improvements.",
]


def get_recommendations(
    shap_values: "np.ndarray",
    feature_names: list[str],
    top_n: int = 3,
) -> list[str]:
    """
    Return up to *top_n* actionable retention recommendations for one customer.

    Parameters
    ----------
    shap_values   : 1-D array of SHAP values for the churn class (one per feature)
    feature_names : list of feature names matching shap_values
    top_n         : number of recommendations to return

    Returns
    -------
    List of recommendation strings (at least one, even for low-risk customers).
    """
    shap_values = np.asarray(shap_values, dtype=float)

    # Rank features by how much they push toward churn (positive SHAP)
    ranked_indices = np.argsort(shap_values)[::-1]

    recommendations: list[str] = []
    seen: set[str] = set()

    for idx in ranked_indices:
        if shap_values[idx] <= 0:
            break  # remaining features push *away* from churn — stop early
        feature_lower = feature_names[idx].lower().replace(" ", "").replace("_", "")
        rec = _lookup_recommendation(feature_lower)
        if rec is not None and rec not in seen:
            recommendations.append(rec)
            seen.add(rec)
        if len(recommendations) >= top_n:
            break

    # Pad with defaults if we didn't find enough actionable items
    for default in _DEFAULT_RECOMMENDATIONS:
        if len(recommendations) >= top_n:
            break
        if default not in seen:
            recommendations.append(default)
            seen.add(default)

    return recommendations


def _lookup_recommendation(feature_lower: str) -> str | None:
    """Return the first matching recommendation or None if the feature is non-actionable."""
    for keyword, suggestion in _RECOMMENDATION_MAP:
        if keyword in feature_lower:
            return suggestion  # None means intentionally skip
    # No keyword matched — build a generic but specific message
    return f"Investigate and optimise the '{feature_lower}' metric through targeted outreach."
