import json
from pathlib import Path

_RULES = json.loads(
    (Path(__file__).parent.parent / "data" / "ss_rules.json").read_text()
)


def get_fra(birth_year: int) -> float:
    """Full Retirement Age as a decimal (e.g. 66.5 = 66 yrs 6 months)."""
    for entry in _RULES["fra_table"]:
        if birth_year <= entry["birth_year_max"]:
            return entry["fra"]
    return 67.0


def benefit_at_age(pia_monthly: float, claiming_age: float, birth_year: int) -> float:
    """
    Monthly Social Security benefit given the PIA (benefit at FRA),
    the chosen claiming age, and birth year.
    Claiming before FRA permanently reduces the benefit.
    Claiming after FRA earns delayed credits up to age 70.
    """
    fra = get_fra(birth_year)
    claiming_age = max(float(_RULES["earliest_claiming_age"]),
                       min(float(_RULES["max_credit_age"]), claiming_age))

    if claiming_age >= fra:
        months_late = (claiming_age - fra) * 12
        factor = 1.0 + months_late * _RULES["delayed_credit_monthly"]
    else:
        months_early = (fra - claiming_age) * 12
        first_36 = min(months_early, 36)
        extra   = max(0.0, months_early - 36)
        reduction = (first_36 * _RULES["early_reduction_first_36_monthly"]
                     + extra  * _RULES["early_reduction_after_36_monthly"])
        factor = 1.0 - reduction

    return pia_monthly * factor


def breakeven_age(pia_monthly: float, age_a: float, age_b: float,
                  birth_year: int) -> float:
    """
    Age at which cumulative lifetime SS from claiming at age_b exceeds
    cumulative SS from claiming at age_a (age_a < age_b).
    Returns inf if b never catches up (shouldn't happen in valid inputs).
    """
    ben_a = benefit_at_age(pia_monthly, age_a, birth_year)
    ben_b = benefit_at_age(pia_monthly, age_b, birth_year)
    if ben_b <= ben_a:
        return float("inf")
    # Solve: ben_a*(X - age_a) = ben_b*(X - age_b)
    return (ben_b * age_b - ben_a * age_a) / (ben_b - ben_a)


def analyze_claiming_scenarios(pia_monthly: float, birth_year: int) -> dict:
    """
    Returns benefits and breakeven ages for the three canonical
    claiming strategies: 62 (early), FRA, and 70 (maximum).
    """
    if pia_monthly <= 0:
        return {"pia_monthly": 0, "fra": 67.0, "fra_label": "67",
                "claim_62": {"monthly": 0, "annual": 0, "pct_vs_fra": 0.0},
                "claim_fra": {"monthly": 0, "annual": 0, "pct_vs_fra": 0.0},
                "claim_70": {"monthly": 0, "annual": 0, "pct_vs_fra": 0.0},
                "breakeven_62_vs_fra": 0.0, "breakeven_fra_vs_70": 0.0, "breakeven_62_vs_70": 0.0}

    fra = get_fra(birth_year)

    ben_62  = benefit_at_age(pia_monthly, 62.0, birth_year)
    ben_fra = pia_monthly
    ben_70  = benefit_at_age(pia_monthly, 70.0, birth_year)

    m62  = round(ben_62)
    mfra = round(ben_fra)
    m70  = round(ben_70)

    return {
        "fra": round(fra, 4),
        "fra_label": _fra_label(fra),
        "pia_monthly": mfra,
        "claim_62": {
            "monthly": m62,
            "annual":  m62 * 12,
            "pct_vs_fra": round((ben_62 / pia_monthly - 1) * 100, 1),
        },
        "claim_fra": {
            "monthly": mfra,
            "annual":  mfra * 12,
            "pct_vs_fra": 0.0,
        },
        "claim_70": {
            "monthly": m70,
            "annual":  m70 * 12,
            "pct_vs_fra": round((ben_70 / pia_monthly - 1) * 100, 1),
        },
        "breakeven_62_vs_fra": round(breakeven_age(pia_monthly, 62.0, fra,  birth_year), 1),
        "breakeven_fra_vs_70": round(breakeven_age(pia_monthly, fra,  70.0, birth_year), 1),
        "breakeven_62_vs_70":  round(breakeven_age(pia_monthly, 62.0, 70.0, birth_year), 1),
    }


def _fra_label(fra: float) -> str:
    """Human-readable FRA like '67' or '66 and 6 months'."""
    years = int(fra)
    months = round((fra - years) * 12)
    if months == 0:
        return str(years)
    return f"{years} and {months} months"
