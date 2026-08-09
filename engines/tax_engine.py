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


def rmd_elimination_calculator(twin: HouseholdTwin) -> dict:
    """
    Compute the annual Roth conversion needed to fully drain the traditional IRA
    to $0 by age 73, eliminating all forced RMDs. Uses an annuity-drain formula.
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

    if gap_years == 0 or traditional_now == 0:
        return {
            "possible": False,
            "annual_conversion": 0,
            "conversion_tax_rate": 0.0,
            "annual_tax_cost": 0,
            "gap_years": gap_years,
            "lifetime_rmd_taxes_saved": 0,
            "lifetime_conversion_tax_cost": 0,
            "lifetime_net_savings": 0,
            "rmd_no_conversion": 0,
            "reason": "no_gap" if gap_years == 0 else "all_roth",
        }

    trad_at_ret = traditional_now * (1 + mu) ** years_to_retirement

    # First RMD at 73 without any conversion (for context and lifetime savings calc)
    trad_no_conv = trad_at_ret
    for _ in range(gap_years):
        trad_no_conv = max(0.0, trad_no_conv * (1 + mu) - annual_spending)
    divisor_73 = _RMD_DIVISORS.get(73, 26.5)
    rmd_no_conv = trad_no_conv / divisor_73

    # Annuity drain: find total annual withdrawal W such that B_n = 0
    # B_n = B_0 * g^n - W * (g^n - 1)/(g - 1) = 0  =>  W = B_0 * g^n * (g-1) / (g^n - 1)
    g = 1 + mu
    gn = g ** gap_years
    total_withdrawal = trad_at_ret * gn * (g - 1) / (gn - 1)
    annual_conversion_elim = total_withdrawal - annual_spending

    years_of_rmds = max(90 - _RMD_START, 0)
    lifetime_rmd_taxes_saved = tax_owed(rmd_no_conv, fs) * years_of_rmds

    if annual_conversion_elim <= 0:
        return {
            "possible": True,
            "annual_conversion": 0,
            "conversion_tax_rate": marginal_rate(annual_spending, fs),
            "annual_tax_cost": 0,
            "gap_years": gap_years,
            "lifetime_rmd_taxes_saved": round(lifetime_rmd_taxes_saved),
            "lifetime_conversion_tax_cost": 0,
            "lifetime_net_savings": round(lifetime_rmd_taxes_saved),
            "rmd_no_conversion": round(rmd_no_conv),
            "reason": "spending_sufficient",
        }

    annual_tax_cost = (
        tax_owed(annual_spending + annual_conversion_elim, fs)
        - tax_owed(annual_spending, fs)
    )
    lifetime_conversion_tax_cost = annual_tax_cost * gap_years
    lifetime_net_savings = lifetime_rmd_taxes_saved - lifetime_conversion_tax_cost

    return {
        "possible": True,
        "annual_conversion": round(annual_conversion_elim),
        "conversion_tax_rate": marginal_rate(annual_spending + annual_conversion_elim, fs),
        "annual_tax_cost": round(annual_tax_cost),
        "gap_years": gap_years,
        "lifetime_rmd_taxes_saved": round(lifetime_rmd_taxes_saved),
        "lifetime_conversion_tax_cost": round(lifetime_conversion_tax_cost),
        "lifetime_net_savings": round(lifetime_net_savings),
        "rmd_no_conversion": round(rmd_no_conv),
        "reason": None,
    }


def roth_conversion_window_optimizer(
    twin: HouseholdTwin,
    ss_claiming_age: float,
    ss_monthly_at_claiming: float,
    life_expectancy: int = 90,
    current_taxable_income: float | None = None,
) -> dict:
    """
    Project taxable income across life stages and identify the optimal Roth conversion
    window: which phase has the lowest bracket, when to start, when to stop, and how
    much to convert. Answers 'when is the best time?' without LLM reasoning.

    ss_claiming_age: age SS benefits begin (62, FRA, or 70); pass 999 if no SS.
    ss_monthly_at_claiming: actual monthly benefit at that age (not PIA).
    """
    fs = twin.tax_profile.filing_status
    a = twin.assumptions
    mu = a.stock_pct * a.stock_return + (1 - a.stock_pct) * a.bond_return

    current_age = twin.person.age
    retirement_age = twin.person.retirement_age
    annual_spending = twin.spending.annual
    ss_annual = ss_monthly_at_claiming * 12
    already_retired = current_age >= retirement_age

    years_to_ret = max(0, retirement_age - current_age)
    trad_at_ret = twin.accounts.traditional_balance * (1 + mu) ** years_to_ret

    phases: list[dict] = []

    # ── Phase 1: Working years ─────────────────────────────────────────────────
    if current_age < retirement_age:
        pre_bracket = marginal_rate(current_taxable_income, fs) if current_taxable_income else None
        phases.append({
            "name": "Working Years",
            "start_age": current_age,
            "end_age": retirement_age - 1,
            "years": retirement_age - current_age,
            "base_taxable": round(current_taxable_income) if current_taxable_income else None,
            "bracket": pre_bracket,
            "headroom": 0,
            "recommended_annual_conversion": 0,
            "conversion_friendly": False,
            "note": "Earned income active — bracket varies with salary.",
        })

    # ── Phase 2: Gap window (retired, no SS, no RMD) ──────────────────────────
    gap_start = retirement_age
    gap_end = (
        min(int(ss_claiming_age) - 1, _RMD_START - 1) if ss_annual > 0
        else _RMD_START - 1
    )

    if gap_start <= gap_end and retirement_age < _RMD_START:
        gap_years = gap_end - gap_start + 1
        # Use annual_spending as income proxy (all from portfolio); consistent with optimize_roth_conversion
        gap_taxable = annual_spending
        gap_bracket = marginal_rate(gap_taxable, fs)
        gap_headroom = bracket_headroom(gap_taxable, fs, target_rate=gap_bracket)
        if gap_headroom == float("inf"):
            gap_headroom = 0.0
        gap_conv = min(gap_headroom, trad_at_ret / gap_years) if gap_years > 0 else 0.0
        gap_conv = max(0.0, gap_conv)
        phase_name = "Early Retirement (Pre-SS)" if ss_annual > 0 else "Retirement (Pre-RMD)"
        phases.append({
            "name": phase_name,
            "start_age": gap_start,
            "end_age": gap_end,
            "years": gap_years,
            "base_taxable": round(gap_taxable),
            "bracket": gap_bracket,
            "headroom": round(gap_headroom),
            "recommended_annual_conversion": round(gap_conv),
            "conversion_friendly": gap_headroom > 0 and gap_conv > 0,
            "note": "No SS, no RMDs — typically the lowest-bracket window.",
        })

    # ── Phase 3: After SS starts (pre-RMD) ────────────────────────────────────
    if ss_annual > 0:
        ss_phase_start = max(retirement_age, int(ss_claiming_age))
        ss_phase_end = _RMD_START - 1  # 72
        if ss_phase_start <= ss_phase_end:
            ss_years = ss_phase_end - ss_phase_start + 1
            # Spending not covered by SS comes from portfolio; plus 85% of SS is taxable
            spending_net = max(0.0, annual_spending - ss_annual)
            ss_taxable = spending_net + ss_annual * 0.85
            ss_bracket = marginal_rate(ss_taxable, fs)
            ss_headroom = bracket_headroom(ss_taxable, fs, target_rate=ss_bracket)
            if ss_headroom == float("inf"):
                ss_headroom = 0.0
            ss_conv = min(ss_headroom, trad_at_ret / ss_years) if ss_years > 0 else 0.0
            ss_conv = max(0.0, ss_conv)
            phases.append({
                "name": f"After SS Starts (age {int(ss_claiming_age)})",
                "start_age": ss_phase_start,
                "end_age": ss_phase_end,
                "years": ss_years,
                "base_taxable": round(ss_taxable),
                "bracket": ss_bracket,
                "headroom": round(ss_headroom),
                "recommended_annual_conversion": round(ss_conv),
                "conversion_friendly": ss_headroom > 0 and ss_conv > 0,
                "note": f"SS ${ss_annual:,.0f}/yr raises taxable income; bracket typically higher.",
            })

    # ── Phase 4: RMD years (73+) ──────────────────────────────────────────────
    if life_expectancy >= _RMD_START:
        # Estimate trad balance at 73: grow to retirement, then simulate drawdown
        trad_running = trad_at_ret
        for yr in range(max(0, _RMD_START - retirement_age)):
            age_in_yr = retirement_age + yr
            ss_in_yr = ss_annual if age_in_yr >= ss_claiming_age else 0.0
            withdrawal = max(0.0, annual_spending - ss_in_yr)
            trad_running = max(0.0, trad_running * (1 + mu) - withdrawal)
        first_rmd = trad_running / _RMD_DIVISORS.get(73, 26.5)
        rmd_taxable = first_rmd + ss_annual * 0.85
        rmd_bracket = marginal_rate(rmd_taxable, fs)
        phases.append({
            "name": "RMD Years (73+)",
            "start_age": _RMD_START,
            "end_age": life_expectancy,
            "years": life_expectancy - _RMD_START + 1,
            "base_taxable": round(rmd_taxable),
            "bracket": rmd_bracket,
            "headroom": 0,
            "recommended_annual_conversion": 0,
            "conversion_friendly": False,
            "note": f"Forced RMDs ~${round(first_rmd):,}/yr. Goal is to shrink this via earlier conversions.",
        })

    # ── Find optimal window ────────────────────────────────────────────────────
    # Converting only helps if it's taxed STRICTLY BELOW the projected RMD rate.
    # Use exact name match to avoid matching "Retirement (Pre-RMD)" phase.
    _RMD_PHASE_NAME = "RMD Years (73+)"
    rmd_phase = next((p for p in phases if p["name"] == _RMD_PHASE_NAME), None)
    rmd_bracket_val = rmd_phase["bracket"] if rmd_phase else None

    candidates = [
        p for p in phases
        if p["name"] not in ("Working Years", _RMD_PHASE_NAME)
        and p.get("bracket") is not None
        and p.get("conversion_friendly")
        and p.get("recommended_annual_conversion", 0) > 0
        and (rmd_bracket_val is None or p["bracket"] < rmd_bracket_val)
    ]

    if not candidates:
        ret_phases = [p for p in phases if p.get("bracket") is not None]
        pre_rmd_phases = [
            p for p in ret_phases
            if p["name"] not in ("Working Years", _RMD_PHASE_NAME)
        ]
        if rmd_bracket_val is not None and pre_rmd_phases and all(
            p["bracket"] >= rmd_bracket_val for p in pre_rmd_phases
        ):
            lowest_pre = min(pre_rmd_phases, key=lambda p: p["bracket"])
            same_rate = lowest_pre["bracket"] == rmd_bracket_val
            note = (
                f"Your projected RMDs at 73 will be taxed at {int(rmd_bracket_val * 100)}% — "
                f"{'the same rate as' if same_rate else 'lower than'} your "
                f"{int(lowest_pre['bracket'] * 100)}% retirement bracket. "
                f"Converting now {'offers no tax advantage' if same_rate else 'costs more than letting RMDs happen'}. "
                f"No conversion recommended."
            )
        elif ret_phases:
            best = min(ret_phases, key=lambda p: p["bracket"])
            note = (
                f"In your lowest-bracket phase ({best['name']}, "
                f"{int(best['bracket'] * 100)}%), there is no headroom for tax-efficient "
                f"conversion. Roth conversion is not recommended."
            )
        else:
            note = "No retirement phases found — retiring at or after RMD age."
        return {
            "phases": phases,
            "optimal_phase_name": None,
            "optimal_start_age": None,
            "optimal_end_age": None,
            "optimal_annual_conversion": 0,
            "optimal_bracket": None,
            "total_converted": 0,
            "pct_converted": 0.0,
            "current_recommendation": "never",
            "current_recommendation_note": note,
            "already_in_retirement": already_retired,
        }

    optimal = min(candidates, key=lambda p: p["bracket"])
    in_window_now = (
        already_retired
        and optimal["start_age"] <= current_age <= optimal["end_age"]
    )

    if in_window_now:
        rec = "start_now"
        years_left = optimal["end_age"] - current_age
        rec_note = (
            f"You're in the optimal window now ({optimal['name']}). "
            f"Convert ${optimal['recommended_annual_conversion']:,}/yr at the "
            f"{int(optimal['bracket'] * 100)}% bracket. "
            f"Window closes at age {optimal['end_age']} — {years_left} year{'s' if years_left != 1 else ''} remaining."
        )
    elif not already_retired:
        rec = "wait_for_retirement"
        rec_note = (
            f"Start converting at retirement (age {retirement_age}). "
            f"Your bracket drops to {int(optimal['bracket'] * 100)}% when earned income stops. "
            f"Convert ${optimal['recommended_annual_conversion']:,}/yr through age {optimal['end_age']}."
        )
    elif current_age < optimal["start_age"]:
        rec = "wait_for_window"
        rec_note = (
            f"Your best window opens at age {optimal['start_age']} ({optimal['name']}). "
            f"Convert ${optimal['recommended_annual_conversion']:,}/yr at {int(optimal['bracket'] * 100)}% bracket then."
        )
    else:
        rec = "window_passed"
        rec_note = (
            f"The optimal window ({optimal['name']}) has passed. "
            f"Check the Roth Conversion section above for current-year opportunities."
        )

    total_conv = optimal["recommended_annual_conversion"] * optimal["years"]
    pct_conv = round(total_conv / trad_at_ret * 100, 1) if trad_at_ret > 0 else 0.0

    return {
        "phases": phases,
        "optimal_phase_name": optimal["name"],
        "optimal_start_age": optimal["start_age"],
        "optimal_end_age": optimal["end_age"],
        "optimal_annual_conversion": optimal["recommended_annual_conversion"],
        "optimal_bracket": optimal["bracket"],
        "total_converted": round(total_conv),
        "pct_converted": pct_conv,
        "current_recommendation": rec,
        "current_recommendation_note": rec_note,
        "already_in_retirement": already_retired,
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
