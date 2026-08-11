"""
Tests for engines/tax_engine.py

All expected values derived from 2025 IRS brackets in data/tax_brackets_2025.json.
Standard deduction: single $15,000 / married $30,000.
"""
import pytest
from core.digital_twin import (
    Accounts, Assumptions, HouseholdTwin, Person, Spending, TaxProfile,
)
from engines.tax_engine import (
    bracket_headroom,
    current_year_roth_advisor,
    marginal_rate,
    optimize_roth_conversion,
    rmd_elimination_calculator,
    roth_conversion_window_optimizer,
    tax_owed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _twin(
    age=55,
    retirement_age=63,
    savings=800_000,
    traditional_pct=0.80,
    spending=60_000,
    filing_status="single",
):
    return HouseholdTwin(
        person=Person(age=age, retirement_age=retirement_age),
        accounts=Accounts(total_savings=savings, traditional_pct=traditional_pct),
        spending=Spending(annual=spending),
        assumptions=Assumptions(),
        tax_profile=TaxProfile(filing_status=filing_status),
    )


# ---------------------------------------------------------------------------
# tax_owed — single filer
# ---------------------------------------------------------------------------

class TestTaxOwedSingle:
    def test_below_standard_deduction_is_zero(self):
        assert tax_owed(14_000, "single") == 0.0

    def test_exactly_standard_deduction_is_zero(self):
        assert tax_owed(15_000, "single") == 0.0

    def test_10pct_bracket(self):
        # Gross $25,000 → taxable $10,000 → 10% * $10,000 = $1,000
        assert tax_owed(25_000, "single") == pytest.approx(1_000.0)

    def test_12pct_bracket(self):
        # Gross $35,000 → taxable $20,000
        # 10% on $11,925 = $1,192.50
        # 12% on ($20,000 - $11,925) = $8,075 → $969
        # Total: $2,161.50
        assert tax_owed(35_000, "single") == pytest.approx(2_161.50)

    def test_22pct_bracket(self):
        # Gross $80,000 → taxable $65,000
        # 10% on $11,925 = $1,192.50
        # 12% on ($48,475 - $11,925) = $36,550 → $4,386
        # 22% on ($65,000 - $48,475) = $16,525 → $3,635.50
        # Total: $9,214
        assert tax_owed(80_000, "single") == pytest.approx(9_214.0)

    def test_24pct_bracket(self):
        # Gross $120,000 → taxable $105,000
        # 10% on $11,925 = $1,192.50
        # 12% on $36,550 = $4,386
        # 22% on $54,875 = $12,072.50
        # 24% on ($105,000 - $103,350) = $1,650 → $396
        # Total: $18,047
        assert tax_owed(120_000, "single") == pytest.approx(18_047.0)


# ---------------------------------------------------------------------------
# tax_owed — married filer (standard deduction $30,000)
# ---------------------------------------------------------------------------

class TestTaxOwedMarried:
    def test_below_standard_deduction_is_zero(self):
        assert tax_owed(29_000, "married") == 0.0

    def test_10pct_bracket(self):
        # Gross $40,000 → taxable $10,000 → 10% * $10,000 = $1,000
        assert tax_owed(40_000, "married") == pytest.approx(1_000.0)


# ---------------------------------------------------------------------------
# marginal_rate
# ---------------------------------------------------------------------------

class TestMarginalRate:
    @pytest.mark.parametrize("gross,expected", [
        (15_000, 0.10),   # taxable $0 — first bracket
        (20_000, 0.10),   # taxable $5,000
        (30_000, 0.12),   # taxable $15,000
        (65_000, 0.22),   # taxable $50,000 → above 12% ceiling ($48,475) → 22%
        (70_000, 0.22),   # taxable $55,000 → 22%
        (120_000, 0.24),  # taxable $105,000 → 24%
        (215_000, 0.32),  # taxable $200,000 → 32%
        (650_000, 0.37),  # taxable $635,000 → 37%
    ])
    def test_single_brackets(self, gross, expected):
        assert marginal_rate(gross, "single") == pytest.approx(expected)

    def test_married_10pct(self):
        assert marginal_rate(40_000, "married") == pytest.approx(0.10)  # taxable $10,000

    def test_married_22pct(self):
        assert marginal_rate(130_000, "married") == pytest.approx(0.22)  # taxable $100,000


# ---------------------------------------------------------------------------
# bracket_headroom
# ---------------------------------------------------------------------------

class TestBracketHeadroom:
    def test_room_in_22pct_bracket_single(self):
        # Single, $60,000 gross → taxable $45,000; 22% bracket ceiling $103,350
        # headroom = $103,350 - $45,000 = $58,350
        assert bracket_headroom(60_000, "single", target_rate=0.22) == pytest.approx(58_350.0)

    def test_no_room_above_ceiling(self):
        # Already above 22% bracket ceiling
        assert bracket_headroom(120_000, "single", target_rate=0.22) == pytest.approx(0.0)

    def test_full_room_from_bottom(self):
        # $16,000 gross → taxable $1,000; 10% bracket ceiling $11,925
        # headroom = $11,925 - $1,000 = $10,925
        assert bracket_headroom(16_000, "single", target_rate=0.10) == pytest.approx(10_925.0)


# ---------------------------------------------------------------------------
# optimize_roth_conversion — edge cases
# ---------------------------------------------------------------------------

class TestOptimizeRothConversion:
    def test_no_gap_when_retiring_at_73(self):
        twin = _twin(age=60, retirement_age=73)
        result = optimize_roth_conversion(twin)
        assert result["no_opportunity"] is True
        assert result["no_opportunity_reason"] == "no_gap"
        assert result["annual_conversion"] == 0

    def test_no_gap_when_retiring_after_73(self):
        twin = _twin(age=65, retirement_age=75)
        result = optimize_roth_conversion(twin)
        assert result["no_opportunity"] is True
        assert result["no_opportunity_reason"] == "no_gap"

    def test_all_roth_no_opportunity(self):
        twin = _twin(savings=500_000, traditional_pct=0.0)
        result = optimize_roth_conversion(twin)
        assert result["no_opportunity"] is True
        assert result["no_opportunity_reason"] == "all_roth"

    def test_small_balance_low_rmd_no_opportunity(self):
        # $100k traditional, spending $80k/yr — RMDs will be tiny, stay in low bracket
        twin = _twin(savings=100_000, traditional_pct=1.0, spending=80_000)
        result = optimize_roth_conversion(twin)
        assert result["no_opportunity"] is True

    def test_large_balance_recommends_conversion(self):
        # Already at retirement age (no pre-retirement growth), very large traditional
        # balance, modest spending → RMDs land in 35% bracket vs 12% spending bracket
        twin = _twin(
            age=63, retirement_age=63,
            savings=5_000_000, traditional_pct=1.0,
            spending=40_000, filing_status="single",
        )
        result = optimize_roth_conversion(twin)
        assert result["no_opportunity"] is False
        assert result["annual_conversion"] > 0
        assert result["lifetime_tax_savings"] > 0
        assert result["gap_years"] == 10

    def test_gap_years_calculation(self):
        twin = _twin(age=55, retirement_age=63)
        result = optimize_roth_conversion(twin)
        assert result["gap_years"] == 10  # 73 - 63

    def test_rmd_drops_with_conversion(self):
        twin = _twin(
            age=55, retirement_age=63,
            savings=2_000_000, traditional_pct=0.90,
            spending=60_000,
        )
        result = optimize_roth_conversion(twin)
        if not result["no_opportunity"]:
            assert result["rmd_with_conversion"] < result["rmd_no_conversion"]

    def test_lifetime_savings_positive_when_recommended(self):
        twin = _twin(
            age=50, retirement_age=60,
            savings=3_000_000, traditional_pct=1.0,
            spending=70_000, filing_status="single",
        )
        result = optimize_roth_conversion(twin)
        if not result["no_opportunity"]:
            assert result["lifetime_tax_savings"] > 0


# ---------------------------------------------------------------------------
# rmd_elimination_calculator — edge cases
# ---------------------------------------------------------------------------

class TestRmdEliminationCalculator:
    def test_no_gap_returns_not_possible(self):
        twin = _twin(age=68, retirement_age=73)
        result = rmd_elimination_calculator(twin)
        assert result["possible"] is False
        assert result["reason"] == "no_gap"

    def test_all_roth_returns_not_possible(self):
        twin = _twin(savings=500_000, traditional_pct=0.0)
        result = rmd_elimination_calculator(twin)
        assert result["possible"] is False
        assert result["reason"] == "all_roth"

    def test_possible_with_gap_and_traditional(self):
        twin = _twin(
            age=55, retirement_age=63,
            savings=1_000_000, traditional_pct=0.80,
            spending=60_000,
        )
        result = rmd_elimination_calculator(twin)
        assert result["possible"] is True
        assert result["gap_years"] == 10

    def test_annual_conversion_is_positive(self):
        twin = _twin(
            age=55, retirement_age=63,
            savings=1_000_000, traditional_pct=0.80,
            spending=60_000,
        )
        result = rmd_elimination_calculator(twin)
        if result["possible"]:
            assert result["annual_conversion"] >= 0

    def test_spending_sufficient_reason(self):
        # Very small traditional balance + large spending → spending alone drains it
        twin = _twin(
            age=70, retirement_age=72,
            savings=50_000, traditional_pct=1.0,
            spending=80_000,
        )
        result = rmd_elimination_calculator(twin)
        if result["possible"]:
            assert result["reason"] in ("spending_sufficient", None)


# ---------------------------------------------------------------------------
# current_year_roth_advisor
# ---------------------------------------------------------------------------

class TestCurrentYearRothAdvisor:
    def test_returns_correct_bracket_single_22pct(self):
        # $80,000 gross single → taxable $65,000 → 22% bracket
        result = current_year_roth_advisor(80_000, "single")
        assert result["current_bracket"] == pytest.approx(0.22)

    def test_headroom_is_nonnegative(self):
        result = current_year_roth_advisor(80_000, "single")
        assert result["headroom_current"] >= 0

    def test_next_bracket_is_higher(self):
        result = current_year_roth_advisor(80_000, "single")
        assert result["next_bracket"] > result["current_bracket"]

    def test_top_bracket_has_no_headroom(self):
        # $650,000 gross → 37% top bracket, no room left
        result = current_year_roth_advisor(650_000, "single")
        assert result["headroom_current"] == 0

    def test_married_filing_uses_married_brackets(self):
        # $100,000 married → taxable $70,000 → 12% bracket (married 12% goes to $96,950)
        result = current_year_roth_advisor(100_000, "married")
        assert result["current_bracket"] == pytest.approx(0.12)

    def test_headroom_to_next_ceiling_gte_current_headroom(self):
        result = current_year_roth_advisor(80_000, "single")
        assert result["headroom_to_next_ceiling"] >= result["headroom_current"]


# ---------------------------------------------------------------------------
# roth_conversion_window_optimizer — spouse phase
# ---------------------------------------------------------------------------

class TestWindowOptimizerSpousePhase:
    def _base_twin(self, filing_status="married"):
        return _twin(
            age=55, retirement_age=63,
            savings=1_200_000, traditional_pct=0.80,
            spending=70_000, filing_status=filing_status,
        )

    def test_no_spouse_phase_when_spouse_not_working(self):
        twin = self._base_twin()
        result = roth_conversion_window_optimizer(
            twin, ss_claiming_age=999, ss_monthly_at_claiming=0,
            spouse_working=False,
        )
        phase_names = [p["name"] for p in result["phases"]]
        assert "Spouse Working / You Retired" not in phase_names

    def test_no_spouse_phase_when_single(self):
        twin = self._base_twin(filing_status="single")
        result = roth_conversion_window_optimizer(
            twin, ss_claiming_age=999, ss_monthly_at_claiming=0,
            spouse_working=False,
        )
        phase_names = [p["name"] for p in result["phases"]]
        assert "Spouse Working / You Retired" not in phase_names

    def test_spouse_phase_inserted_when_spouse_works_longer(self):
        twin = self._base_twin()
        result = roth_conversion_window_optimizer(
            twin, ss_claiming_age=999, ss_monthly_at_claiming=0,
            spouse_working=True, spouse_income=90_000, spouse_retirement_age=67,
        )
        phase_names = [p["name"] for p in result["phases"]]
        assert "Spouse Working / You Retired" in phase_names

    def test_spouse_phase_ages_are_correct(self):
        twin = self._base_twin()
        result = roth_conversion_window_optimizer(
            twin, ss_claiming_age=999, ss_monthly_at_claiming=0,
            spouse_working=True, spouse_income=90_000, spouse_retirement_age=67,
        )
        spouse_phase = next(p for p in result["phases"] if p["name"] == "Spouse Working / You Retired")
        assert spouse_phase["start_age"] == 63  # user's retirement age
        assert spouse_phase["end_age"] == 66    # spouse_retirement_age - 1

    def test_gap_window_delayed_when_spouse_works_longer(self):
        twin = self._base_twin()
        result = roth_conversion_window_optimizer(
            twin, ss_claiming_age=999, ss_monthly_at_claiming=0,
            spouse_working=True, spouse_income=90_000, spouse_retirement_age=67,
        )
        gap_phase = next(
            (p for p in result["phases"] if "Pre-RMD" in p["name"] or "Retirement" in p["name"]),
            None,
        )
        if gap_phase:
            assert gap_phase["start_age"] >= 67

    def test_no_spouse_phase_when_spouse_retires_same_time(self):
        twin = self._base_twin()
        result = roth_conversion_window_optimizer(
            twin, ss_claiming_age=999, ss_monthly_at_claiming=0,
            spouse_working=True, spouse_income=90_000, spouse_retirement_age=63,
        )
        phase_names = [p["name"] for p in result["phases"]]
        assert "Spouse Working / You Retired" not in phase_names

    def test_recommendation_valid_with_spouse_phase(self):
        twin = self._base_twin()
        result = roth_conversion_window_optimizer(
            twin, ss_claiming_age=999, ss_monthly_at_claiming=0,
            spouse_working=True, spouse_income=90_000, spouse_retirement_age=67,
        )
        assert result["current_recommendation"] in (
            "start_now", "wait_for_retirement", "wait_for_window", "window_passed", "never"
        )
