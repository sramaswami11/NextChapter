"""
Tests for api/app.py — dashboard HTML generation.
"""
import pytest
from api.app import _build_dashboard
from core.digital_twin import (
    HouseholdTwin, Person, Accounts, Spending, Assumptions, TaxProfile, SocialSecurity,
)
from engines.ss_engine import analyze_claiming_scenarios, recommended_strategy

_RESULTS = {
    "success_rate": 100.0,
    "median_portfolio": 1_000_000,
    "portfolio_at_retirement": 1_000_000,
    "years_in_retirement": 30,
}
_MC = {"success_rate": 100.0}
_TAX = {
    "no_opportunity": True,
    "annual_conversion": 0,
    "no_opportunity_reason": "all_roth",
}


def _twin(primary_pia: float) -> HouseholdTwin:
    return HouseholdTwin(
        person=Person(age=60, retirement_age=65),
        accounts=Accounts(total_savings=1_000_000, traditional_pct=0.5),
        spending=Spending(annual=50_000),
        assumptions=Assumptions(),
        tax_profile=TaxProfile(filing_status="married"),
        ss=SocialSecurity(monthly_pia=primary_pia, birth_year=1964),
    )


# ---------------------------------------------------------------------------
# Spousal benefit flag
# ---------------------------------------------------------------------------

class TestSpousalBenefitFlag:
    def test_flag_shown_when_spouse_pia_below_half(self):
        primary_pia = 3_000.0
        spouse_pia = 1_000.0  # < 50% of 3000 = 1500
        twin = _twin(primary_pia)
        ss_data = analyze_claiming_scenarios(primary_pia, 1964)
        s_data = analyze_claiming_scenarios(spouse_pia, 1964)
        s_rec = recommended_strategy(s_data, 84)
        html = _build_dashboard(
            _RESULTS, twin, _TAX, ss_data, _MC, _MC, _MC,
            spouse_ss_data=s_data, spouse_ss_rec=s_rec,
        )
        assert "Spousal Benefit May Apply" in html
        assert "$1,500/mo" in html  # 50% of $3,000

    def test_flag_absent_when_spouse_pia_equal_to_half(self):
        primary_pia = 3_000.0
        spouse_pia = 1_500.0  # exactly 50% — no flag
        twin = _twin(primary_pia)
        ss_data = analyze_claiming_scenarios(primary_pia, 1964)
        s_data = analyze_claiming_scenarios(spouse_pia, 1964)
        s_rec = recommended_strategy(s_data, 84)
        html = _build_dashboard(
            _RESULTS, twin, _TAX, ss_data, _MC, _MC, _MC,
            spouse_ss_data=s_data, spouse_ss_rec=s_rec,
        )
        assert "Spousal Benefit May Apply" not in html

    def test_flag_absent_when_spouse_pia_above_half(self):
        primary_pia = 3_000.0
        spouse_pia = 1_800.0  # > 50% — no flag
        twin = _twin(primary_pia)
        ss_data = analyze_claiming_scenarios(primary_pia, 1964)
        s_data = analyze_claiming_scenarios(spouse_pia, 1964)
        s_rec = recommended_strategy(s_data, 84)
        html = _build_dashboard(
            _RESULTS, twin, _TAX, ss_data, _MC, _MC, _MC,
            spouse_ss_data=s_data, spouse_ss_rec=s_rec,
        )
        assert "Spousal Benefit May Apply" not in html

    def test_flag_absent_when_no_spouse_ss(self):
        primary_pia = 3_000.0
        twin = _twin(primary_pia)
        ss_data = analyze_claiming_scenarios(primary_pia, 1964)
        html = _build_dashboard(
            _RESULTS, twin, _TAX, ss_data, _MC, _MC, _MC,
        )
        assert "Spousal Benefit May Apply" not in html
