"""
Tests for agents/planner.py

Covers the parsers, ConversationState field ordering, is_ready(), and the
full process_message() flow from blank state to ready.
"""
import pytest
from agents.planner import (
    ConversationState,
    _parse_filing_status,
    _parse_number,
    _parse_percentage,
    extract_retirement_age,
    process_message,
)


# ---------------------------------------------------------------------------
# _parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    @pytest.mark.parametrize("text,expected", [
        ("$1.2M",    1_200_000),
        ("1.2m",     1_200_000),
        ("$800k",    800_000),
        ("800K",     800_000),
        ("$80,000",  80_000),
        ("80000",    80_000),
        ("$60k",     60_000),
        ("2400",     2_400),
        ("1200000",  1_200_000),
    ])
    def test_parses_amounts(self, text, expected):
        assert _parse_number(text) == pytest.approx(expected)

    def test_returns_none_for_no_digits(self):
        assert _parse_number("skip") is None

    def test_returns_none_for_empty(self):
        assert _parse_number("") is None


# ---------------------------------------------------------------------------
# _parse_percentage
# ---------------------------------------------------------------------------

class TestParsePercentage:
    @pytest.mark.parametrize("text,expected", [
        ("80%",         0.80),
        ("75%",         0.75),
        ("100%",        1.0),
        ("0%",          0.0),
        ("all of it",   1.0),
        ("all of",      1.0),
        ("everything",  1.0),
        ("none",        0.0),
        ("nothing",     0.0),
        ("zero",        0.0),
        ("half",        0.5),
        ("most of it",  0.80),
        ("mostly",      0.80),
        ("75",          0.75),
        ("50",          0.50),
    ])
    def test_common_phrases(self, text, expected):
        assert _parse_percentage(text) == pytest.approx(expected)

    def test_returns_none_for_nonsense(self):
        assert _parse_percentage("hello there") is None

    def test_clamps_above_100(self):
        result = _parse_percentage("120%")
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _parse_filing_status
# ---------------------------------------------------------------------------

class TestParseFilingStatus:
    @pytest.mark.parametrize("text,expected", [
        ("single",                    "single"),
        ("I'm single",                "single"),
        ("not married",               "single"),
        ("unmarried",                 "single"),
        ("married",                   "married"),
        ("filing jointly",            "married"),
        ("joint",                     "married"),
        ("MFJ",                       "married"),
        ("I have a spouse",           "married"),
        ("married filing jointly",    "married"),
    ])
    def test_recognized_phrases(self, text, expected):
        assert _parse_filing_status(text) == expected

    def test_returns_none_for_unknown(self):
        assert _parse_filing_status("complicated") is None


# ---------------------------------------------------------------------------
# extract_retirement_age
# ---------------------------------------------------------------------------

class TestExtractRetirementAge:
    @pytest.mark.parametrize("text,expected", [
        ("Can I retire at 63?",          63),
        ("retire at age 65",             65),
        ("I want to retire at 60",       60),
        ("retire around 62",             62),
        ("retire by 58",                 58),
        ("retire when I'm 67",           67),
        ("retire when im 68",            68),
        ("retire when I am 70",          70),
    ])
    def test_standard_phrases(self, text, expected):
        assert extract_retirement_age(text) == expected

    def test_returns_none_when_no_age(self):
        assert extract_retirement_age("I want to retire someday") is None

    def test_case_insensitive(self):
        assert extract_retirement_age("RETIRE AT 65") == 65


# ---------------------------------------------------------------------------
# ConversationState.is_ready
# ---------------------------------------------------------------------------

class TestIsReady:
    def test_empty_state_not_ready(self):
        assert ConversationState().is_ready() is False

    def test_partial_state_not_ready(self):
        s = ConversationState(retirement_age=63, age=55, savings=800_000)
        assert s.is_ready() is False

    def test_all_fields_but_income_not_answered_not_ready(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="single", ss_monthly_benefit=2_000.0,
        )
        assert s.is_ready() is False

    def test_fully_populated_is_ready(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="single", ss_monthly_benefit=2_000.0,
            current_income_answered=True, life_expectancy_answered=True,
        )
        assert s.is_ready() is True

    def test_zero_traditional_pct_is_ready(self):
        # traditional_pct=0.0 must be treated as valid (not falsy)
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.0,
            filing_status="single", ss_monthly_benefit=2_000.0,
            current_income_answered=True, life_expectancy_answered=True,
        )
        assert s.is_ready() is True

    def test_zero_ss_benefit_is_ready(self):
        # ss_monthly_benefit=0.0 means no SS — still a valid answer
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="single", ss_monthly_benefit=0.0,
            current_income_answered=True, life_expectancy_answered=True,
        )
        assert s.is_ready() is True

    def test_married_not_ready_without_spouse_answers(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married", ss_monthly_benefit=2_000.0,
            current_income_answered=True, life_expectancy_answered=True,
        )
        assert s.is_ready() is False

    def test_married_ready_with_spouse_questions_answered(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married", spouse_questions_answered=True,
            ss_monthly_benefit=2_000.0,
            current_income_answered=True, life_expectancy_answered=True,
        )
        assert s.is_ready() is True


# ---------------------------------------------------------------------------
# ConversationState.next_question ordering
# ---------------------------------------------------------------------------

class TestNextQuestionOrder:
    def test_first_question_is_retirement_age(self):
        q = ConversationState().next_question()
        assert "retire" in q.lower() or "retirement" in q.lower()

    def test_second_question_is_current_age(self):
        s = ConversationState(retirement_age=63)
        assert "current age" in s.next_question().lower()

    def test_third_question_is_savings(self):
        s = ConversationState(retirement_age=63, age=55)
        assert "savings" in s.next_question().lower()

    def test_fourth_question_is_spending(self):
        s = ConversationState(retirement_age=63, age=55, savings=800_000)
        assert "spend" in s.next_question().lower()

    def test_fifth_question_is_traditional_pct(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000, annual_spending=60_000
        )
        assert "traditional" in s.next_question().lower()

    def test_sixth_question_is_filing_status(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
        )
        assert "single" in s.next_question().lower() or "married" in s.next_question().lower()

    def test_after_married_asks_spouse_age(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married",
        )
        q = s.next_question().lower()
        assert "spouse" in q and ("old" in q or "age" in q)

    def test_after_spouse_age_asks_working(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married", spouse_age=60,
        )
        q = s.next_question().lower()
        assert "spouse" in q and ("working" in q or "currently" in q)

    def test_after_spouse_working_yes_asks_income(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married", spouse_age=60, spouse_working=True,
        )
        q = s.next_question().lower()
        assert "spouse" in q and "income" in q

    def test_after_spouse_income_asks_retirement_age(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married", spouse_age=60, spouse_working=True, spouse_income=90_000,
        )
        q = s.next_question().lower()
        assert "retire" in q or "age" in q

    def test_after_spouse_retire_age_asks_ss_benefit(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married", spouse_age=60, spouse_working=True,
            spouse_income=90_000, spouse_retirement_age=67,
        )
        q = s.next_question().lower()
        assert "spouse" in q and "social security" in q

    def test_after_spouse_not_working_asks_ss_benefit(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married", spouse_age=60, spouse_working=False,
        )
        q = s.next_question().lower()
        assert "spouse" in q and "social security" in q

    def test_after_spouse_ss_benefit_asks_life_expectancy(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married", spouse_age=60, spouse_working=False,
            spouse_ss_benefit=1_500.0,
        )
        q = s.next_question().lower()
        assert "spouse" in q and ("live" in q or "age" in q or "life" in q)

    def test_no_le_question_when_spouse_has_no_ss(self):
        # When spouse SS = 0, spouse_questions_answered flips True without asking LE
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="married", spouse_age=60, spouse_working=False,
            spouse_ss_benefit=0.0, spouse_questions_answered=True,
        )
        # Next question should be primary SS, not spouse LE
        assert "social security" in s.next_question().lower() or "estimated" in s.next_question().lower()

    def test_seventh_question_is_ss_benefit(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="single",
        )
        assert "social security" in s.next_question().lower()

    def test_eighth_question_is_current_income(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="single", ss_monthly_benefit=2_000.0,
        )
        assert "income" in s.next_question().lower()

    def test_ninth_question_is_life_expectancy(self):
        s = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
            filing_status="single", ss_monthly_benefit=2_000.0,
            current_income_answered=True,
        )
        assert "age" in s.next_question().lower() or "live" in s.next_question().lower()


# ---------------------------------------------------------------------------
# process_message — full conversation flow
# ---------------------------------------------------------------------------

class TestProcessMessage:
    def _run_flow(self, messages):
        state = ConversationState()
        for msg in messages:
            process_message(state, msg)
        return state

    def test_full_9_question_flow(self):
        state = self._run_flow([
            "Can I retire at 63?",
            "55",
            "$800k",
            "$60k",
            "80%",
            "single",
            "2200",
            "$120k",
            "87",
        ])
        assert state.retirement_age == 63
        assert state.age == 55
        assert state.savings == pytest.approx(800_000)
        assert state.annual_spending == pytest.approx(60_000)
        assert state.traditional_pct == pytest.approx(0.80)
        assert state.filing_status == "single"
        assert state.ss_monthly_benefit == pytest.approx(2_200)
        assert state.current_taxable_income == pytest.approx(120_000)
        assert state.life_expectancy == 87
        assert state.is_ready()

    def test_ss_skip_sets_zero(self):
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%", "single", "none",
        ])
        assert state.ss_monthly_benefit == 0.0

    def test_ss_annual_amount_converted_to_monthly(self):
        # $28,800 annual SS → $2,400/mo (>5000 so divided by 12)
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%", "single", "28800",
        ])
        assert state.ss_monthly_benefit == pytest.approx(2_400.0)

    def test_income_skip_marks_answered(self):
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%", "single", "2200", "skip",
        ])
        assert state.current_income_answered is True
        assert state.current_taxable_income is None

    def test_life_expectancy_average_sets_84(self):
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%",
            "single", "2200", "skip", "average",
        ])
        assert state.life_expectancy == 84
        assert state.life_expectancy_answered is True

    def test_retirement_age_restatement_mid_conversation(self):
        # After retirement age is set, a restatement should update it
        state = ConversationState(
            retirement_age=63, age=55, savings=800_000,
            annual_spending=60_000, traditional_pct=0.80,
        )
        process_message(state, "Actually, I want to retire at 65")
        assert state.retirement_age == 65

    def test_savings_with_m_suffix(self):
        state = self._run_flow(["retire at 63", "55", "1.5M"])
        assert state.savings == pytest.approx(1_500_000)

    def test_married_filing_status(self):
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%", "married",
        ])
        assert state.filing_status == "married"

    def test_single_skips_spouse_questions(self):
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%", "single",
        ])
        assert state.spouse_questions_answered is True
        assert state.spouse_working is None

    def test_married_spouse_not_working(self):
        # spouse_age → not working → spouse SS benefit → spouse LE → questions complete
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%", "married", "60", "no", "1500", "85",
        ])
        assert state.spouse_age == 60
        assert state.spouse_working is False
        assert state.spouse_ss_benefit == pytest.approx(1_500)
        assert state.spouse_life_expectancy == 85
        assert state.spouse_questions_answered is True
        assert state.spouse_income is None

    def test_married_spouse_not_working_no_ss(self):
        # When SS = 0, no LE question — done after SS answer
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%", "married", "60", "no", "0",
        ])
        assert state.spouse_ss_benefit == pytest.approx(0.0)
        assert state.spouse_life_expectancy is None
        assert state.spouse_questions_answered is True

    def test_married_spouse_working_full_flow(self):
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%",
            "married",
            "60",    # spouse age
            "yes",
            "$95k",
            "67",
            "1800",  # spouse SS benefit
            "85",    # spouse life expectancy
            "2200",
            "$120k",
            "87",
        ])
        assert state.filing_status == "married"
        assert state.spouse_age == 60
        assert state.spouse_working is True
        assert state.spouse_income == pytest.approx(95_000)
        assert state.spouse_retirement_age == 67
        assert state.spouse_ss_benefit == pytest.approx(1_800)
        assert state.spouse_life_expectancy == 85
        assert state.spouse_questions_answered is True
        assert state.ss_monthly_benefit == pytest.approx(2_200)
        assert state.life_expectancy == 87
        assert state.is_ready()

    def test_spouse_ss_annual_converted_to_monthly(self):
        # $21,600/yr → $1,800/mo; then LE is required before questions complete
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%", "married", "60", "no", "21600", "83",
        ])
        assert state.spouse_ss_benefit == pytest.approx(1_800.0)
        assert state.spouse_life_expectancy == 83
        assert state.spouse_questions_answered is True

    def test_spouse_le_average_sets_84(self):
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "80%", "married", "60", "no", "1500", "average",
        ])
        assert state.spouse_life_expectancy == 84
        assert state.spouse_questions_answered is True

    def test_all_roth_traditional_pct_zero(self):
        state = self._run_flow([
            "retire at 63", "55", "$800k", "$60k", "none",
        ])
        assert state.traditional_pct == pytest.approx(0.0)
