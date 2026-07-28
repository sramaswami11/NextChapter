import json
from pathlib import Path

from core.digital_twin import HouseholdTwin

_DATA = json.loads(
    (Path(__file__).parent.parent / "data" / "tax_brackets_2025.json").read_text()
)
_RMD_START = _DATA["rmd_start_age"]
_RMD_DIVISORS: dict[int, float] = {int(k): v for k, v in _DATA["rmd_divisors"].items()}


def _std_deduction(filing_status: str) -> float:
    return _DATA["standard_deduction"][filing_status]


def _brackets(filing_status: str) -> list[dict]:
    return _DATA["brackets"][filing_status]


def tax_owed(gross_income: float, filing_status: str) -> float:
    """Federal income tax on gross_income after standard deduction."""
    taxable = max(0.0, gross_income - _std_deduction(filing_status))
    total = 0.0
    for b in _brackets(filing_status):
        if taxable <= b["from"]:
            break
        cap = b["to"] if b["to"] is not None else float("inf")
        total += (min(taxable, cap) - b["from"]) * b["rate"]
    return total


def marginal_rate(gross_income: float, filing_status: str) -> float:
    """Marginal federal rate on the last dollar of gross_income."""
    taxable = max(0.0, gross_income - _std_deduction(filing_status))
    for b in _brackets(filing_status):
        cap = b["to"] if b["to"] is not None else float("inf")
        if taxable <= cap:
            return b["rate"]
    return _brackets(filing_status)[-1]["rate"]


def bracket_headroom(gross_income: float, filing_status: str, target_rate: float = 0.22) -> float:
    """
    Dollars of additional gross income that fit before crossing above target_rate bracket.
    Returns 0 if already above that bracket.
    The standard deduction is already applied; additional income maps 1-to-1 to taxable income.
    """
    taxable = max(0.0, gross_income - _std_deduction(filing_status))
    for b in _brackets(filing_status):
        if b["rate"] == target_rate:
            ceiling = b["to"] if b["to"] is not None else float("inf")
            return max(0.0, ceiling - taxable)
    return 0.0


def _highest_rate_below(rate: float, filing_status: str) -> float | None:
    """Highest bracket rate strictly below the given rate, or None if rate is the lowest."""
    result = None
    for b in _brackets(filing_status):
        if b["rate"] < rate:
            result = b["rate"]
    return result


def optimize_roth_conversion(twin: HouseholdTwin) -> dict:
    """
    Compute the optimal annual Roth conversion during the gap window
    (retirement age → RMD start age 73). Returns a dict consumed by
    the dashboard and the LLM prompt.

    Core rule: only recommend conversions when projected RMD marginal rate
    exceeds the current spending marginal rate. Fill brackets up to (but not
    including) the projected RMD rate so every converted dollar is taxed less
    than it would be as a forced RMD.
    """
    fs = twin.tax_profile.filing_status
    a = twin.assumptions
    mu = a.stock_pct * a.stock_return + (1 - a.stock_pct) * a.bond_return

    retirement_age = twin.person.retirement_age
    current_age = twin.person.age
    gap_years = max(0, _RMD_START - retirement_age)
    years_to_retirement = max(0, retirement_age - current_age)
    annual_spending = twin.spending.annual

    traditional_now = twin.accounts.traditional_balance
    roth_now = twin.accounts.roth_balance

    if gap_years == 0 or traditional_now == 0:
        trad_at_73 = traditional_now * (1 + mu) ** max(0, _RMD_START - current_age)
        first_rmd = trad_at_73 / _RMD_DIVISORS.get(73, 26.5)
        return {
            "gap_years": gap_years,
            "annual_conversion": 0,
            "conversion_tax_rate": 0.0,
            "rmd_no_conversion": round(first_rmd),
            "rmd_with_conversion": round(first_rmd),
            "annual_tax_cost": 0,
            "annual_rmd_tax_savings": 0,
            "lifetime_tax_savings": 0,
            "current_bracket": marginal_rate(annual_spending, fs),
            "no_opportunity": True,
            "no_opportunity_reason": "no_gap" if gap_years == 0 else "all_roth",
        }

    # --- Grow to retirement ---
    trad_at_ret = traditional_now * (1 + mu) ** years_to_retirement
    roth_at_ret = roth_now * (1 + mu) ** years_to_retirement

    # --- First pass: simulate gap years with NO conversion to get projected RMD ---
    trad_no_conv = trad_at_ret
    for _ in range(gap_years):
        trad_no_conv = max(0.0, trad_no_conv * (1 + mu) - annual_spending)

    divisor_73 = _RMD_DIVISORS.get(73, 26.5)
    rmd_no_conv = trad_no_conv / divisor_73

    # --- Determine optimal conversion target ---
    # Only convert if future RMD bracket > current spending bracket
    spending_rate = marginal_rate(annual_spending, fs)
    rmd_rate = marginal_rate(rmd_no_conv, fs)

    if rmd_rate <= spending_rate:
        # Future RMDs taxed at same or lower rate — converting now costs more, not less
        return {
            "gap_years": gap_years,
            "annual_conversion": 0,
            "conversion_tax_rate": spending_rate,
            "rmd_no_conversion": round(rmd_no_conv),
            "rmd_with_conversion": round(rmd_no_conv),
            "annual_tax_cost": 0,
            "annual_rmd_tax_savings": 0,
            "lifetime_tax_savings": 0,
            "current_bracket": spending_rate,
            "no_opportunity": True,
            "no_opportunity_reason": "low_rmd",
        }

    # Fill brackets up to (but not including) the projected RMD rate.
    # Every converted dollar is then taxed below what an RMD would cost.
    target_rate = _highest_rate_below(rmd_rate, fs)
    if target_rate is None:
        annual_conversion = 0.0
    else:
        headroom = bracket_headroom(annual_spending, fs, target_rate=target_rate)
        annual_conversion = min(headroom, trad_at_ret / gap_years)
        annual_conversion = max(0.0, annual_conversion)

    # --- Second pass: simulate gap years WITH conversion ---
    trad_conv = trad_at_ret
    roth_conv = roth_at_ret
    for _ in range(gap_years):
        trad_conv = max(0.0, trad_conv * (1 + mu) - annual_spending - annual_conversion)
        roth_conv = roth_conv * (1 + mu) + annual_conversion

    rmd_with_conv = trad_conv / divisor_73

    # --- Tax math ---
    annual_tax_cost = (
        tax_owed(annual_spending + annual_conversion, fs)
        - tax_owed(annual_spending, fs)
    )
    annual_rmd_tax_savings = tax_owed(rmd_no_conv, fs) - tax_owed(rmd_with_conv, fs)
    years_of_rmds = max(90 - _RMD_START, 0)
    lifetime_tax_savings = (annual_rmd_tax_savings * years_of_rmds
                            - annual_tax_cost * gap_years)

    # Rate differential is necessary but not sufficient — the actual dollar savings
    # must also exceed the upfront conversion tax cost.
    if lifetime_tax_savings <= 0:
        return {
            "gap_years": gap_years,
            "annual_conversion": 0,
            "conversion_tax_rate": spending_rate,
            "rmd_no_conversion": round(rmd_no_conv),
            "rmd_with_conversion": round(rmd_no_conv),
            "annual_tax_cost": 0,
            "annual_rmd_tax_savings": 0,
            "lifetime_tax_savings": 0,
            "current_bracket": spending_rate,
            "no_opportunity": True,
            "no_opportunity_reason": "negative_return",
            "rmd_bracket": rmd_rate,
        }

    return {
        "gap_years": gap_years,
        "annual_conversion": round(annual_conversion),
        "conversion_tax_rate": marginal_rate(annual_spending + annual_conversion, fs),
        "rmd_no_conversion": round(rmd_no_conv),
        "rmd_with_conversion": round(rmd_with_conv),
        "annual_tax_cost": round(annual_tax_cost),
        "annual_rmd_tax_savings": round(annual_rmd_tax_savings),
        "lifetime_tax_savings": round(lifetime_tax_savings),
        "current_bracket": spending_rate,
        "no_opportunity": False,
        "no_opportunity_reason": None,
    }


def current_year_roth_advisor(current_income: float, filing_status: str) -> dict:
    """
    Given the user's current-year gross income, compute Roth conversion
    headroom for the current tax year — both within the current bracket
    and optionally one bracket higher.
    """
    current_rate = marginal_rate(current_income, filing_status)
    headroom_raw = bracket_headroom(current_income, filing_status, target_rate=current_rate)
    # Top bracket has no ceiling — headroom is infinite; treat as 0 (no room to fill)
    headroom_current = 0.0 if headroom_raw == float("inf") else headroom_raw
    tax_to_fill_current = (
        tax_owed(current_income + headroom_current, filing_status)
        - tax_owed(current_income, filing_status)
    )

    next_rate = None
    for b in _brackets(filing_status):
        if b["rate"] > current_rate:
            next_rate = b["rate"]
            break

    result: dict = {
        "current_income": current_income,
        "current_bracket": current_rate,
        "headroom_current": round(headroom_current),
        "tax_to_fill_current": round(tax_to_fill_current),
        "next_bracket": next_rate,
        "headroom_to_next_ceiling": 0,
        "incremental_for_next": 0,
        "tax_to_fill_next": 0,
    }

    if next_rate is not None:
        headroom_to_next = bracket_headroom(current_income, filing_status, target_rate=next_rate)
        incremental = max(0, headroom_to_next - headroom_current)
        tax_to_fill_next = (
            tax_owed(current_income + headroom_to_next, filing_status)
            - tax_owed(current_income, filing_status)
        )
        result.update({
            "headroom_to_next_ceiling": round(headroom_to_next),
            "incremental_for_next": round(incremental),
            "tax_to_fill_next": round(tax_to_fill_next),
        })

    return result
