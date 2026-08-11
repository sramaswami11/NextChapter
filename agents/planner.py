import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationState:
    retirement_age: Optional[int] = None
    age: Optional[int] = None
    savings: Optional[float] = None
    annual_spending: Optional[float] = None
    traditional_pct: Optional[float] = None   # 0.0–1.0
    filing_status: Optional[str] = None       # "single" or "married"
    spouse_working: Optional[bool] = None
    spouse_age: Optional[int] = None
    spouse_income: Optional[float] = None
    spouse_retirement_age: Optional[int] = None
    spouse_ss_benefit: Optional[float] = None  # monthly at FRA; 0 = no SS
    spouse_life_expectancy: Optional[int] = None
    spouse_questions_answered: bool = False
    ss_monthly_benefit: Optional[float] = None  # monthly PIA at FRA; 0 = no SS
    current_taxable_income: Optional[float] = None  # gross income this year; None = skipped
    current_income_answered: bool = False
    life_expectancy: Optional[int] = None
    life_expectancy_answered: bool = False
    analysis_complete: bool = False
    last_mc_results: Optional[dict] = None
    last_tax_results: Optional[dict] = None
    last_ss_results: Optional[dict] = None
    last_cy_roth_results: Optional[dict] = None
    last_elim_results: Optional[dict] = None
    last_window_results: Optional[dict] = None

    def is_ready(self) -> bool:
        _spouse_ok = self.spouse_questions_answered or self.filing_status != "married"
        return (
            self.retirement_age is not None
            and self.age is not None
            and self.savings is not None
            and self.annual_spending is not None
            and self.traditional_pct is not None
            and self.filing_status is not None
            and _spouse_ok
            and self.ss_monthly_benefit is not None
            and self.current_income_answered
            and self.life_expectancy_answered
        )

    def next_question(self) -> str:
        if self.retirement_age is None:
            return "At what age would you like to retire?"
        if self.age is None:
            return "What is your current age?"
        if self.savings is None:
            return "What is your total retirement savings today? (401k + IRA + investments, e.g. $1.2M or $800k)"
        if self.annual_spending is None:
            return "How much do you expect to spend per year in retirement? (today's dollars, e.g. $80k)"
        if self.traditional_pct is None:
            return (
                f"Of your ${self.savings:,.0f} in savings, what percentage is in traditional "
                f"(pre-tax) accounts like a 401k or traditional IRA? "
                f"(e.g. '80%', 'all of it', 'about half')"
            )
        if self.filing_status is None:
            return "Are you filing taxes as single or married?"
        if not self.spouse_questions_answered and self.filing_status == "married":
            if self.spouse_age is None:
                return "How old is your spouse?"
            if self.spouse_working is None:
                return "Is your spouse currently working? (yes or no)"
            if self.spouse_working and self.spouse_income is None:
                return "What is your spouse's annual income from work? (e.g. $85k)"
            if self.spouse_working and self.spouse_retirement_age is None:
                return "At what age does your spouse plan to retire?"
            if self.spouse_ss_benefit is None:
                return (
                    "What is your spouse's estimated monthly Social Security benefit at their "
                    "full retirement age? (Check ssa.gov — type 0 if they have no SS benefits.)"
                )
            if self.spouse_ss_benefit > 0 and self.spouse_life_expectancy is None:
                return (
                    "What age do you expect your spouse to live to? "
                    "(e.g. 85, 90 — or type 'average' for 84)"
                )
        if self.ss_monthly_benefit is None:
            return (
                "What is your estimated monthly Social Security benefit at full retirement age? "
                "(Check ssa.gov — it's on your statement. Type 0 if you don't have SS benefits.)"
            )
        if not self.current_income_answered:
            return (
                "What is your expected total taxable income this year from all sources — "
                "wages, interest, dividends, and any other income? "
                "(e.g. '$95k', '$200k') Type 'skip' if you'd prefer to skip this."
            )
        if not self.life_expectancy_answered:
            return (
                "What age do you expect to live to? This helps us recommend the right "
                "Social Security claiming age. (e.g. 85, 90 — or type 'average' for 84)"
            )
        return ""

    def results_context(self) -> str:
        """Build a context string summarising the completed analysis for follow-up LLM calls."""
        mc = self.last_mc_results or {}
        tax = self.last_tax_results or {}
        ss = self.last_ss_results or {}
        fs_label = "married" if self.filing_status == "married" else "single"

        if self.filing_status == "married" and self.spouse_age is not None:
            ss_str = ""
            if self.spouse_ss_benefit and self.spouse_ss_benefit > 0:
                le_str = f", life expectancy {self.spouse_life_expectancy}" if self.spouse_life_expectancy else ""
                ss_str = f", SS ${self.spouse_ss_benefit:,.0f}/mo at FRA{le_str}"
            elif self.spouse_ss_benefit == 0.0:
                ss_str = ", no SS benefits"
            if self.spouse_working and self.spouse_income:
                spouse_note = (
                    f" Spouse (age {self.spouse_age}): working (${self.spouse_income:,.0f}/yr, "
                    f"retiring at {self.spouse_retirement_age}){ss_str}."
                )
            else:
                spouse_note = f" Spouse (age {self.spouse_age}): not currently working{ss_str}."
        else:
            spouse_note = ""

        profile = (
            f"Client: age {self.age}, retiring at {self.retirement_age}. "
            f"Saved ${self.savings:,.0f} ({self.traditional_pct:.0%} in traditional pre-tax accounts). "
            f"Spending ${self.annual_spending:,.0f}/yr in retirement. Filing {fs_label}.{spouse_note}"
        )

        mc_part = (
            f"Monte Carlo ({mc['years_in_retirement']}-yr horizon to age 90): "
            f"{mc['success_rate']}% success rate across 10,000 simulations. "
            f"Portfolio at retirement: ${mc['portfolio_at_retirement']:,.0f}. "
            f"Median remaining at 90: ${mc['median_portfolio']:,.0f}."
        ) if mc else ""

        if tax and not tax.get("no_opportunity"):
            tax_part = (
                f"Roth conversion: {tax['gap_years']}-year window (ages {self.retirement_age}–73). "
                f"Recommended ${tax['annual_conversion']:,.0f}/yr conversion at "
                f"{int(tax['conversion_tax_rate'] * 100)}% marginal rate. "
                f"First RMD at 73 drops from ${tax['rmd_no_conversion']:,.0f} "
                f"to ${tax['rmd_with_conversion']:,.0f}/yr. "
                f"Estimated lifetime tax savings: ${tax['lifetime_tax_savings']:,.0f}."
            )
        elif tax.get("no_opportunity_reason") == "low_rmd":
            tax_part = (
                f"Roth conversion: not recommended — projected RMDs at 73 "
                f"(${tax['rmd_no_conversion']:,.0f}/yr) stay in the "
                f"{int(tax['current_bracket'] * 100)}% bracket, same as retirement spending. "
                f"No tax benefit from converting now."
            )
        else:
            tax_part = "Roth conversion: no gap window (retiring at or after RMD age 73)."

        if ss and ss.get("pia_monthly", 0) > 0:
            le = self.life_expectancy or 84
            be_fra_70 = ss["breakeven_fra_vs_70"]
            be_62_fra = ss["breakeven_62_vs_fra"]
            if le <= be_62_fra:
                rec = f"Recommended: claim at 62 (life expectancy {le} is before breakeven at {be_62_fra})."
            elif le <= be_fra_70:
                rec = f"Recommended: claim at FRA (life expectancy {le} is before FRA-vs-70 breakeven at {be_fra_70})."
            else:
                rec = f"Recommended: delay to 70 (life expectancy {le} exceeds breakeven at {be_fra_70})."
            ss_part = (
                f"Social Security: PIA ${ss['pia_monthly']:,}/mo at FRA {ss['fra_label']}. "
                f"Claiming at 62: ${ss['claim_62']['monthly']:,}/mo ({ss['claim_62']['pct_vs_fra']}% vs FRA). "
                f"Claiming at 70: ${ss['claim_70']['monthly']:,}/mo (+{ss['claim_70']['pct_vs_fra']}% vs FRA). "
                f"Breakeven FRA vs 70: age {be_fra_70}. {rec}"
            )
        elif ss:
            ss_part = "Social Security: client has no SS benefits."
        else:
            ss_part = ""

        cy = self.last_cy_roth_results or {}
        if cy:
            rate_pct = int(cy["current_bracket"] * 100)
            cy_part = (
                f"Current year Roth opportunity: income ${cy['current_income']:,.0f}, "
                f"bracket {rate_pct}%, "
                f"room to fill bracket ${cy['headroom_current']:,.0f} "
                f"(tax cost ${cy['tax_to_fill_current']:,.0f})."
            )
        else:
            cy_part = ""

        elim = self.last_elim_results or {}
        if elim and elim.get("possible") and elim.get("annual_conversion", 0) > 0:
            elim_part = (
                f"Full RMD elimination option: convert ${elim['annual_conversion']:,.0f}/yr "
                f"at {int(elim['conversion_tax_rate'] * 100)}% marginal rate "
                f"to drain traditional IRA to $0 by age 73. "
                f"Lifetime net savings vs no conversion: ${elim['lifetime_net_savings']:,.0f}."
            )
        else:
            elim_part = ""

        window = self.last_window_results or {}
        if window:
            rec = window.get("current_recommendation", "")
            note = window.get("current_recommendation_note", "")
            if rec not in ("", "never") and window.get("optimal_bracket") is not None:
                window_part = (
                    f"Roth conversion timing: {note} "
                    f"Optimal phase: '{window.get('optimal_phase_name', '')}' at "
                    f"{int(window['optimal_bracket'] * 100)}% bracket — "
                    f"convert ${window['optimal_annual_conversion']:,}/yr "
                    f"(ages {window['optimal_start_age']}–{window['optimal_end_age']})."
                )
            else:
                window_part = f"Roth conversion timing: {note}" if note else ""
        else:
            window_part = ""

        return "  ".join(filter(None, [profile, mc_part, tax_part, elim_part, window_part, ss_part, cy_part]))


def _parse_percentage(text: str) -> Optional[float]:
    t = text.lower().strip()
    if any(w in t for w in ("all of it", "all of", "everything", "100")):
        return 1.0
    if any(w in t for w in ("none", "nothing", "zero", "no traditional")):
        return 0.0
    if "half" in t or "50" in t:
        return 0.5
    if "most" in t or "mostly" in t:
        return 0.80
    m = re.search(r"(\d+\.?\d*)\s*%", t)
    if m:
        val = float(m.group(1)) / 100.0
        return max(0.0, min(1.0, val))
    m = re.search(r"(\d+\.?\d*)", t)
    if m:
        val = float(m.group(1))
        if 0 < val <= 100:
            return val / 100.0
    return None


def _parse_filing_status(text: str) -> Optional[str]:
    t = text.lower()
    # Check negations before "married" so "not married" / "unmarried" don't match the married branch
    if "not married" in t or "unmarried" in t or "single" in t:
        return "single"
    if "married" in t or "joint" in t or "mfj" in t or "spouse" in t:
        return "married"
    return None


def _parse_number(text: str) -> Optional[float]:
    text = text.lower().replace(",", "").replace("$", "").strip()
    m = re.search(r"(\d+\.?\d*)\s*(k|m)?", text)
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2)
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    return val


def extract_retirement_age(text: str) -> Optional[int]:
    """Pull age from phrases like 'retire at 63', 'retire at age 65', 'retire when I'm 67'."""
    m = re.search(
        r"retire\s+(?:at\s+age|at|around|by|when\s+i(?:'?m|\s+am))\s+(\d+)",
        text.lower(),
    )
    if m:
        return int(m.group(1))
    return None


def process_message(state: ConversationState, text: str) -> None:
    """Fill the next missing field from the user's message."""
    # If the user restates a retirement age at any point, respect it.
    restated_age = extract_retirement_age(text)
    if restated_age and state.retirement_age is not None:
        state.retirement_age = restated_age
        return

    if state.retirement_age is None:
        age = extract_retirement_age(text)
        if age:
            state.retirement_age = age
            return
        num = _parse_number(text)
        if num and 45 <= num <= 80:
            state.retirement_age = int(num)
        return

    if state.age is None:
        num = _parse_number(text)
        if num and 20 <= num <= 80:
            state.age = int(num)
        return

    if state.savings is None:
        num = _parse_number(text)
        if num and num >= 1_000:
            state.savings = num
        return

    if state.annual_spending is None:
        num = _parse_number(text)
        if num and num >= 1_000:
            state.annual_spending = num
        return

    if state.traditional_pct is None:
        pct = _parse_percentage(text)
        if pct is not None:
            state.traditional_pct = pct
        return

    if state.filing_status is None:
        fs = _parse_filing_status(text)
        if fs is not None:
            state.filing_status = fs
            if fs == "single":
                state.spouse_questions_answered = True
        return

    if not state.spouse_questions_answered:
        t = text.lower().strip()
        if state.spouse_age is None:
            num = _parse_number(text)
            if num is not None and 20 <= num <= 85:
                state.spouse_age = int(num)
            return
        if state.spouse_working is None:
            if any(w in t for w in ("yes", "yeah", "yep", "working", "employed", "still work", "she work", "he work", "works")):
                state.spouse_working = True
            elif any(w in t for w in ("no", "nope", "not working", "retired", "not employed", "doesn't", "does not", "not currently")):
                state.spouse_working = False
            return
        if state.spouse_working and state.spouse_income is None:
            num = _parse_number(text)
            if num is not None and num >= 1_000:
                state.spouse_income = num
            return
        if state.spouse_working and state.spouse_retirement_age is None:
            num = _parse_number(text)
            if num is not None and 40 <= num <= 85:
                state.spouse_retirement_age = int(num)
            return
        if state.spouse_ss_benefit is None:
            if any(w in t for w in ("skip", "none", "no ss", "no social", "don't have", "do not have")):
                state.spouse_ss_benefit = 0.0
            else:
                num = _parse_number(text)
                if num is not None:
                    state.spouse_ss_benefit = num / 12.0 if num > 5_000 else num
            if state.spouse_ss_benefit is not None and state.spouse_ss_benefit == 0.0:
                state.spouse_questions_answered = True  # no LE needed without SS
            return
        if state.spouse_ss_benefit > 0 and state.spouse_life_expectancy is None:
            if any(w in t for w in ("average", "don't know", "not sure", "typical", "normal")):
                state.spouse_life_expectancy = 84
                state.spouse_questions_answered = True
            else:
                num = _parse_number(text)
                if num is not None and 65 <= num <= 110:
                    state.spouse_life_expectancy = int(num)
                    state.spouse_questions_answered = True
            return

    if state.ss_monthly_benefit is None:
        t = text.lower().strip()
        if any(w in t for w in ("skip", "none", "no ss", "no social", "don't have", "do not have")):
            state.ss_monthly_benefit = 0.0
        else:
            num = _parse_number(text)
            if num is not None:
                # Input could be monthly (e.g. 2400) or annual (e.g. 28800)
                # Heuristic: if > 5000 assume annual and convert
                state.ss_monthly_benefit = num / 12.0 if num > 5_000 else num
        return

    if not state.current_income_answered:
        t = text.lower().strip()
        if any(w in t for w in ("skip", "pass", "no", "not sure", "don't know", "none")):
            state.current_income_answered = True
        else:
            num = _parse_number(text)
            if num is not None and num >= 0:
                state.current_taxable_income = num
                state.current_income_answered = True
        return

    if not state.life_expectancy_answered:
        t = text.lower().strip()
        if any(w in t for w in ("average", "don't know", "not sure", "typical", "normal")):
            state.life_expectancy = 84
            state.life_expectancy_answered = True
        else:
            num = _parse_number(text)
            if num is not None and 65 <= num <= 110:
                state.life_expectancy = int(num)
                state.life_expectancy_answered = True
        return
