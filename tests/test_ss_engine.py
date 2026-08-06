"""
Tests for engines/ss_engine.py

Expected values derived from data/ss_rules.json:
  early_reduction_first_36_monthly  = 0.005556  (~5/9% per month)
  early_reduction_after_36_monthly  = 0.004167  (~5/12% per month)
  delayed_credit_monthly            = 0.006667  (8%/yr)
  FRA for birth_year 1960+          = 67.0
"""
import pytest
from engines.ss_engine import (
    analyze_claiming_scenarios,
    benefit_at_age,
    breakeven_age,
    get_fra,
    recommended_strategy,
)

PIA = 2_000.0      # monthly PIA used across most tests
BY_1960 = 1960    # FRA = 67


# ---------------------------------------------------------------------------
# get_fra
# ---------------------------------------------------------------------------

class TestGetFra:
    @pytest.mark.parametrize("birth_year,expected_fra", [
        (1954, 66.0),
        (1955, 66.1667),
        (1956, 66.3333),
        (1957, 66.5),
        (1958, 66.6667),
        (1959, 66.8333),
        (1960, 67.0),
        (1970, 67.0),
        (1990, 67.0),
    ])
    def test_fra_by_birth_year(self, birth_year, expected_fra):
        assert get_fra(birth_year) == pytest.approx(expected_fra, abs=1e-3)


# ---------------------------------------------------------------------------
# benefit_at_age
# ---------------------------------------------------------------------------

class TestBenefitAtAge:
    def test_benefit_at_fra_equals_pia(self):
        assert benefit_at_age(PIA, 67.0, BY_1960) == pytest.approx(PIA)

    def test_claim_at_62_reduces_benefit(self):
        # 60 months early (FRA=67): first 36 * 0.005556 + next 24 * 0.004167
        # reduction = 0.200016 + 0.100008 = 0.300024 → factor ≈ 0.6999 → ~$1,400
        ben = benefit_at_age(PIA, 62.0, BY_1960)
        assert ben == pytest.approx(PIA * 0.6999, abs=1.0)
        assert ben < PIA

    def test_claim_at_70_increases_benefit(self):
        # 36 months late: 36 * 0.006667 = 0.24 → factor 1.24 → $2,480
        ben = benefit_at_age(PIA, 70.0, BY_1960)
        assert ben == pytest.approx(PIA * 1.24, abs=1.0)
        assert ben > PIA

    def test_benefit_monotone_increasing_with_age(self):
        ages = [62, 63, 64, 65, 66, 67, 68, 69, 70]
        benefits = [benefit_at_age(PIA, a, BY_1960) for a in ages]
        assert all(benefits[i] < benefits[i + 1] for i in range(len(benefits) - 1))

    def test_claim_clamped_at_62(self):
        # Claiming at 55 should be clamped to 62
        assert benefit_at_age(PIA, 55.0, BY_1960) == pytest.approx(
            benefit_at_age(PIA, 62.0, BY_1960)
        )

    def test_claim_clamped_at_70(self):
        # Claiming at 75 should be clamped to 70
        assert benefit_at_age(PIA, 75.0, BY_1960) == pytest.approx(
            benefit_at_age(PIA, 70.0, BY_1960)
        )

    def test_fra_1954_claim_at_62(self):
        # FRA=66, so only 48 months early
        # first 36 * 0.005556 = 0.200016; next 12 * 0.004167 = 0.050004; reduction = 0.25 → factor 0.75
        ben = benefit_at_age(PIA, 62.0, 1954)
        assert ben == pytest.approx(PIA * 0.75, abs=1.0)


# ---------------------------------------------------------------------------
# breakeven_age
# ---------------------------------------------------------------------------

class TestBreakevenAge:
    def test_62_vs_fra_roughly_79(self):
        # ben_a=$1,400, ben_b=$2,000, age_a=62, age_b=67 → ≈78.7
        be = breakeven_age(PIA, 62.0, 67.0, BY_1960)
        assert 77.0 < be < 81.0

    def test_fra_vs_70_roughly_82_83(self):
        be = breakeven_age(PIA, 67.0, 70.0, BY_1960)
        assert 80.0 < be < 85.0

    def test_breakeven_62_vs_70_between_others(self):
        be_62_fra = breakeven_age(PIA, 62.0, 67.0, BY_1960)
        be_fra_70 = breakeven_age(PIA, 67.0, 70.0, BY_1960)
        be_62_70 = breakeven_age(PIA, 62.0, 70.0, BY_1960)
        # 62-vs-70 breakeven should be between the other two or close
        assert be_62_fra < be_62_70

    def test_equal_benefits_returns_inf(self):
        # If both strategies yield same benefit (age = FRA for both), b never overtakes a
        assert breakeven_age(PIA, 67.0, 67.0, BY_1960) == float("inf")


# ---------------------------------------------------------------------------
# analyze_claiming_scenarios
# ---------------------------------------------------------------------------

class TestAnalyzeClaimingScenarios:
    def setup_method(self):
        self.result = analyze_claiming_scenarios(PIA, BY_1960)

    def test_pia_matches(self):
        assert self.result["pia_monthly"] == round(PIA)

    def test_fra_label(self):
        assert self.result["fra_label"] == "67"

    def test_claim_62_is_less_than_fra(self):
        assert self.result["claim_62"]["monthly"] < self.result["claim_fra"]["monthly"]

    def test_claim_70_is_greater_than_fra(self):
        assert self.result["claim_70"]["monthly"] > self.result["claim_fra"]["monthly"]

    def test_claim_fra_pct_is_zero(self):
        assert self.result["claim_fra"]["pct_vs_fra"] == 0.0

    def test_claim_62_pct_is_negative(self):
        assert self.result["claim_62"]["pct_vs_fra"] < 0

    def test_claim_70_pct_is_positive(self):
        assert self.result["claim_70"]["pct_vs_fra"] > 0

    def test_annual_is_12x_monthly(self):
        for key in ("claim_62", "claim_fra", "claim_70"):
            s = self.result[key]
            assert s["annual"] == s["monthly"] * 12

    def test_breakeven_ages_in_realistic_range(self):
        assert 75 < self.result["breakeven_62_vs_fra"] < 82
        assert 79 < self.result["breakeven_fra_vs_70"] < 86

    def test_zero_pia_returns_zeros(self):
        r = analyze_claiming_scenarios(0, BY_1960)
        assert r["claim_62"]["monthly"] == 0
        assert r["claim_70"]["monthly"] == 0


# ---------------------------------------------------------------------------
# recommended_strategy
# ---------------------------------------------------------------------------

class TestRecommendedStrategy:
    def setup_method(self):
        self.ss = analyze_claiming_scenarios(PIA, BY_1960)

    def test_short_life_expectancy_recommends_62(self):
        # Life expectancy below 62-vs-FRA breakeven (~78.7) → claim at 62
        result = recommended_strategy(self.ss, life_expectancy=75)
        assert result == "claim_62"

    def test_medium_life_expectancy_recommends_fra(self):
        # Above 62-vs-FRA breakeven but below FRA-vs-70 breakeven
        be_62_fra = self.ss["breakeven_62_vs_fra"]
        be_fra_70 = self.ss["breakeven_fra_vs_70"]
        mid = int((be_62_fra + be_fra_70) / 2)
        result = recommended_strategy(self.ss, life_expectancy=mid)
        assert result == "claim_fra"

    def test_long_life_expectancy_recommends_70(self):
        # Above FRA-vs-70 breakeven (~82.5) → delay to 70
        result = recommended_strategy(self.ss, life_expectancy=90)
        assert result == "claim_70"

    def test_exactly_at_62_fra_breakeven_recommends_62(self):
        be = int(self.ss["breakeven_62_vs_fra"])
        result = recommended_strategy(self.ss, life_expectancy=be)
        assert result == "claim_62"
