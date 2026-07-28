# NextChapter — Architecture & Design Document

**Version:** Step 3 Complete  
**Last updated:** 2026-07-26  
**Status:** Monte Carlo + Roth Conversion Optimizer + Social Security Agent + Follow-up Conversation live

---

## 1. What Is NextChapter?

NextChapter is a local-first retirement planning agent. A user types a natural language question — *"Can I retire at 63?"* — and gets back a Monte Carlo simulation, a Roth conversion analysis, a plain-English explanation, and a live dashboard, all running on their own machine with no paid subscriptions.

The guiding constraint that shapes every architectural decision is:

> **$0 recurring API cost.** All math is deterministic Python. All financial rules are encoded locally. LLM is optional and used only for language, never calculation.

---

## 2. Architecture Principles

These four rules were decided up front and override any convenience tradeoff:

| Principle | What it means in practice |
|---|---|
| **No paid financial APIs** | Market data, tax rules, SS rules — all stored as local JSON/Python. Never call a data provider. |
| **LLM = explainer only** | The LLM never does math or makes decisions. It receives already-computed results and wraps them in plain English. |
| **Deterministic core** | Every calculation must be reproducible and auditable. `numpy.random.default_rng(42)` — same inputs always produce the same output. |
| **Digital Twin model** | All user data lives in a single structured household model (`HouseholdTwin`). Every analysis copies that model, changes assumptions, and compares. No isolated one-off calculators. |

---

## 3. High-Level System Architecture

```
Browser (dark-theme two-pane UI)
        │  HTTP POST /chat  (form data)
        │  ← Server-Sent Events (SSE) stream
        ▼
┌─────────────────────────────────────────────┐
│  FastAPI  (api/app.py)                      │
│  • Session management (cookie-based)        │
│  • Dual-mode routing:                       │
│      Q&A mode  → Planner                    │
│      Follow-up → LLM directly               │
│  • SSE event streaming                      │
│  • Dashboard HTML builder                   │
└──────────┬──────────────────────────────────┘
           │
     ┌─────▼──────────────────┐
     │  Planner Agent          │
     │  (agents/planner.py)    │
     │  6-field Q&A state      │
     │  Natural language parser│
     │  results_context() for  │
     │  follow-up LLM calls    │
     └─────┬───────────────────┘
           │  HouseholdTwin
     ┌─────▼───────────────────┐     ┌──────────────────────────┐
     │  Monte Carlo Engine      │     │  Tax Engine               │
     │  (engines/monte_carlo.py)│     │  (engines/tax_engine.py) │
     │  10,000 simulations      │     │  Roth conversion optimizer│
     │  numpy — pure Python     │     │  2025 IRS brackets (JSON) │
     └─────┬───────────────────┘     └────────────┬─────────────┘
           │  results dict                         │  tax dict
           └───────────────────┬───────────────────┘
                               ▼
                 ┌─────────────────────────────────────┐
                 │  LLM Client  (llm/client.py)        │
                 │  Priority chain:                    │
                 │    1. Ollama (local — qwen3:8b)     │
                 │    2. Anthropic API (cloud fallback)│
                 │    3. Static string (always works)  │
                 └─────────────────────────────────────┘
```

The browser never talks directly to Ollama or to any calculation engine. Everything is mediated by FastAPI.

---

## 4. The Retirement Digital Twin

The central concept of the product is the **Household Twin** — a single structured snapshot of a person's complete financial life. It lives in `core/digital_twin.py`.

```python
@dataclass
class HouseholdTwin:
    person:      Person       # age, retirement_age
    accounts:    Accounts     # total_savings + traditional_pct; computed traditional/roth balances
    spending:    Spending     # annual spending in today's dollars
    assumptions: Assumptions  # market return, inflation, allocation
    tax_profile: TaxProfile   # filing_status (added Step 2)
```

**Why this matters architecturally:** Every future feature (SS Agent, Medicare Agent) adds fields to this model rather than creating its own separate data structures. A "what-if" scenario is literally `copy.deepcopy(twin)` followed by changing one field. This prevents the common failure mode of calculators that don't talk to each other.

---

## 5. Component Reference

### 5.1 `core/digital_twin.py`

**Purpose:** The shared data model. Every engine and agent reads from this.

**Classes:**

| Class | Fields | Notes |
|---|---|---|
| `Person` | `age`, `retirement_age` | Both integers. Will gain spouse in Step 5. |
| `Accounts` | `total_savings`, `traditional_pct=1.0` | `traditional_pct` is the fraction in pre-tax accounts. Computed properties `traditional_balance` and `roth_balance` derive the split. Defaults to 100% traditional if user doesn't specify. |
| `Spending` | `annual` | Today's dollars. Inflation adjustment happens inside the engine, not here. |
| `Assumptions` | `stock_return=0.09`, `bond_return=0.04`, `inflation=0.025`, `stock_pct=0.60` | All defaults. Step 5 will let users override in what-if scenarios. |
| `TaxProfile` | `filing_status="single"` | Added in Step 2. `"single"` or `"married"`. Defaults to single. |
| `SocialSecurity` | `monthly_pia=0.0`, `birth_year=1960` | Added in Step 3. PIA = Primary Insurance Amount (benefit at FRA). `birth_year` drives FRA lookup from `ss_rules.json`. Defaults to 1960 (FRA = 67). |
| `HouseholdTwin` | Composes all above | Root object passed between all layers. |

**Design decision:** Plain Python `@dataclass` with no ORM, no serialization magic. Computed properties (`traditional_balance`, `roth_balance`) are `@property` methods rather than stored fields — they derive from `total_savings × traditional_pct` so they stay in sync automatically. When persistence is needed (Step 5+), the twin serializes trivially to JSON.

---

### 5.2 `data/tax_brackets_2025.json`

**Purpose:** Single source of truth for all IRS numbers used by the tax engine. Encoded once, easy to update annually.

**Contents:**

```
standard_deduction   single: $15,000  /  married: $30,000
brackets             7 brackets for single and married (10% → 37%)
rmd_divisors         IRS Uniform Lifetime Table, ages 72–90
rmd_start_age        73
```

**2025 Federal Tax Brackets (Single):**

| Rate | Taxable Income Range |
|---|---|
| 10% | $0 – $11,925 |
| 12% | $11,925 – $48,475 |
| 22% | $48,475 – $103,350 |
| 24% | $103,350 – $197,300 |
| 32% | $197,300 – $250,525 |
| 35% | $250,525 – $626,350 |
| 37% | $626,350+ |

Married Filing Jointly brackets are the same rates at approximately double the thresholds.

**RMD Divisors:** From IRS Publication 590-B Uniform Lifetime Table (2022 revision). Age 73 divisor = 26.5. Used to estimate the first Required Minimum Distribution.

**Design decision — why JSON not Python constants:** Tax rules change annually. Keeping them in a data file means an annual update is a one-line edit with no code change. The Python tax engine loads the file at module import time and caches it in memory.

---

### 5.3 `engines/tax_engine.py`

**Purpose:** Federal tax math and the Roth conversion optimizer. Takes a `HouseholdTwin`, returns an actionable recommendation.

**Public functions:**

| Function | Purpose |
|---|---|
| `tax_owed(gross_income, filing_status)` | Total federal income tax after standard deduction |
| `marginal_rate(gross_income, filing_status)` | Rate on the last dollar of income |
| `bracket_headroom(gross_income, filing_status, target_rate)` | Additional income that fits before exceeding `target_rate` bracket |
| `optimize_roth_conversion(twin)` | Full Roth conversion analysis — see algorithm below |

**`optimize_roth_conversion()` algorithm:**

```
1. gap_years = max(0, 73 - retirement_age)
   If gap_years == 0 or traditional_balance == 0 → return early (no opportunity)

2. Grow traditional balance to retirement age using blended return mu.

3. FIRST PASS — simulate gap years with NO conversion:
   For each year: trad = trad × (1 + mu) − annual_spending
   Compute rmd_no_conversion = trad_at_73 / IRS_divisor[73]

4. Determine optimal conversion target:
   - spending_rate = marginal_rate(annual_spending)      ← current bracket
   - rmd_rate = marginal_rate(rmd_no_conversion)         ← projected RMD bracket
   - If rmd_rate ≤ spending_rate: STOP — converting now costs more than it saves
   - Else: target_rate = highest bracket rate BELOW rmd_rate
     (e.g. if RMDs will hit 24%, target = 22%)

5. annual_conversion = bracket_headroom(annual_spending, target_rate)
   Capped at traditional_balance / gap_years to avoid over-converting.

6. SECOND PASS — simulate gap years WITH conversion:
   For each year: trad = trad × (1 + mu) − spending − conversion
                  roth = roth × (1 + mu) + conversion
   Compute rmd_with_conversion = trad_at_73 / IRS_divisor[73]

7. Compute:
   annual_tax_cost  = tax_owed(spending + conversion) − tax_owed(spending)
   annual_savings   = tax_owed(rmd_no_conv) − tax_owed(rmd_with_conv)
   lifetime_savings = annual_savings × 17 − annual_tax_cost × gap_years
   (17 = years of RMDs from age 73 to 90)
```

**The key insight:** Only recommend Roth conversions when the projected RMD bracket is *higher* than the current spending bracket. Converting at a higher rate than future RMDs would cost more tax, not less. This check prevents the optimizer from recommending conversions in cases where traditional balances are modest and RMDs will stay in the same or lower bracket.

**Returns dict with:** `gap_years`, `annual_conversion`, `conversion_tax_rate`, `rmd_no_conversion`, `rmd_with_conversion`, `annual_tax_cost`, `annual_rmd_tax_savings`, `lifetime_tax_savings`, `current_bracket`, `no_opportunity`, `no_opportunity_reason`.

---

### 5.4 `data/ss_rules.json`

**Purpose:** Social Security rules encoded locally — no SSA API needed.

**Contents:**

```
fra_by_birth_year   FRA for birth years 1954–1960+ (ranges from 66y0m to 67y0m)
early_reduction     5/9% per month for first 36 months early, 5/12% beyond
delayed_credit      8% per year for each year past FRA up to age 70
claiming_ages       62–70
```

**Design decision:** Birth year determines FRA exactly per SSA rules. Rather than a formula, the table stores the actual FRA for each birth year range, matching the SSA's published schedule.

---

### 5.5 `engines/ss_engine.py`

**Purpose:** Social Security math — benefit calculation at any claiming age, breakeven analysis, and three-scenario comparison.

**Public functions:**

| Function | Purpose |
|---|---|
| `get_fra(birth_year)` | Returns `(years, months)` FRA tuple for a given birth year |
| `benefit_at_age(pia, birth_year, claim_age)` | Monthly benefit after early/delayed adjustment |
| `breakeven_age(pia, birth_year, age_a, age_b)` | Age at which later-claiming strategy overtakes earlier one in cumulative lifetime benefits |
| `analyze_claiming_scenarios(pia, birth_year)` | Returns dict with claim-at-62, claim-at-FRA, claim-at-70 scenarios |
| `_fra_label(birth_year)` | Human-readable FRA string e.g. `"67"` |

**`benefit_at_age()` algorithm:**

```
fra_months = fra_years × 12 + fra_partial_months
claim_months = claim_age × 12
delta = claim_months − fra_months

If delta < 0 (early):
  first_36 = max(delta, −36)
  beyond   = min(0, delta + 36)
  reduction = first_36 × (5/9 × 0.01) + beyond × (5/12 × 0.01)
  monthly_benefit = pia × (1 + reduction)

If delta > 0 (delayed):
  credit = delta/12 × 0.08
  monthly_benefit = pia × (1 + credit)
```

**SS dashboard section:** Three side-by-side cards (yellow = early, blue = FRA, green = delayed) showing monthly benefit, annual income, % vs FRA, and per-strategy Monte Carlo success rate. Breakeven ages shown in the assumptions box below.

**Monte Carlo integration:** Each scenario runs `run_monte_carlo()` with SS income offsetting withdrawals once the claiming age is reached. The offset is inflation-adjusted over time so SS purchasing power remains constant in real terms.

---

### 5.6 `engines/monte_carlo.py`

**Purpose:** The core financial math. Takes a `HouseholdTwin`, returns a results dict.

**Algorithm:**

```
1. Compute blended expected return (mu) and volatility (sigma)
   from stock/bond split in Assumptions.

2. Grow savings deterministically from today to retirement age
   using mu (simplified — pre-retirement volatility not yet simulated).

3. Simulate 10,000 retirement paths:
   For each year in retirement:
     portfolio = portfolio × (1 + annual_return) − inflation-adjusted spending

4. Count paths where portfolio > 0 at plan_to_age (default 90).
   That percentage is the success rate.
```

**Key parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `n_sims` | 10,000 | Sufficient for stable percentile estimates; runs in < 100ms with numpy |
| `plan_to_age` | 90 | Conservative planning horizon covering most life expectancy scenarios |
| RNG seed | 42 | Fixed for reproducibility — same inputs always yield same output |
| Stock volatility | 17% | Historical S&P 500 annual standard deviation |
| Bond volatility | 6% | Historical intermediate-term bond standard deviation |

**Returns:** `success_rate`, `median_portfolio`, `p10_portfolio`, `years_in_retirement`, `portfolio_at_retirement`.

**Known simplification:** Pre-retirement growth is deterministic (no volatility before retirement age). Step 4 charts will make this visible; Step 5 will simulate the full path.

---

### 5.7 `agents/planner.py`

**Purpose:** Manages the conversation — tracks what we know, asks for what we don't, parses natural language, and provides context for follow-up questions.

**`ConversationState`** has seven data-collection fields plus three state-tracking fields:

```
Data fields (collected via Q&A):
  retirement_age → age → savings → annual_spending → traditional_pct → filing_status → ss_monthly_benefit

State fields:
  analysis_complete: bool        — True after first full analysis run
  last_mc_results:  dict | None  — stored Monte Carlo output
  last_tax_results: dict | None  — stored Roth optimizer output
```

**`is_ready()`** checks all seven data fields are non-None. Note: `ss_monthly_benefit` can be `0.0` (no SS), so uses `is not None` rather than bare truthiness. Note: `traditional_pct` can be `0.0` (all Roth), so uses `is not None` rather than bare truthiness.

**`next_question()`** implements strict priority order. The sequence is deliberate:
- Retirement age first — it's the intent the user already stated
- Current age second — needed to compute years to retirement
- Savings and spending — simulation inputs
- Traditional % — needed for tax analysis only, asked after core Monte Carlo inputs
- Filing status — final tax input

**`results_context()`** builds a structured text summary of the completed analysis for use in follow-up LLM calls. Includes the full user profile, Monte Carlo results, and Roth conversion recommendation in a format the LLM can reference directly when answering questions like "Is my success rate good enough?" or "Should I convert now or wait?"

**`process_message()`** parsing logic:

1. **Always checks for retirement age restatement first** — if the user says "Can I retire at 58?" mid-conversation, `extract_retirement_age()` fires and updates `state.retirement_age` regardless of which question was pending. This prevents the number from being misinterpreted as the answer to a different field.
2. Then fills the next missing field in order.

**Parsers:**

| Parser | Handles |
|---|---|
| `extract_retirement_age()` | "retire at 63", "retire at age 65", "retire around 60", "retire when I'm 67", "retire when im 68", "retire by 62" |
| `_parse_number()` | `$1.2M`, `800k`, `$80,000`, `1200000` — strips symbols, expands k/M |
| `_parse_percentage()` | "80%", "all of it", "none", "half", "most of it", bare numbers 0–100 |
| `_parse_filing_status()` | "single", "married", "joint", "mfj", "spouse" |
| `_parse_ss_benefit()` | Dollar amounts monthly or annual (annual ÷12 if >5000); "0", "none", "skip" → 0.0 |

**Bug fixed (Step 2):** The original `extract_retirement_age` regex had `\s+` inside only the `when i'm` branch, causing "retire **at** 63" (the most common phrasing) to never match. Fixed by restructuring to `retire\s+(?:at\s+age|at|around|by|when\s+i(?:'?m|\s+am))\s+(\d+)`.

---

### 5.8 `llm/client.py`

**Purpose:** Wrap LLM access behind a single `explain(system, user) → str` function. The rest of the codebase never imports a model provider directly.

**Priority chain:**

```
1. Ollama (local)    → try first; $0 cost; needs Ollama service running
2. Anthropic API     → cloud fallback; needs ANTHROPIC_API_KEY env var
3. Static string     → always works; no dependencies
```

Each layer returns `None` on failure so `explain()` cascades cleanly.

**`_ollama()` — key details:**

- URL: `http://localhost:11434/api/chat`
- Model: reads `OLLAMA_MODEL` env var, defaults to `qwen3:8b`
- `"think": False` — **critical.** Qwen3:8b defaults to chain-of-thought thinking mode, routing all generated tokens to a `thinking` field and leaving `content` empty. Without this flag every call returns an empty string. Requires Ollama >= 0.7 (installed: 0.32.1).
- `num_predict: 400` — enough for 2–3 sentences
- `timeout: 60s` — model cold-start on first call takes 15–20s

**Bug fixed (Step 2):** Original implementation silently swallowed all exceptions and returned the static fallback string, making it impossible to diagnose failures. Now uses `logger.error()` and returns `None` so `explain()` can cascade.

---

### 5.9 `api/app.py`

**Purpose:** The FastAPI application — HTTP routing, session management, dual-mode conversation handling, SSE event production, and dashboard HTML generation.

**Routes:**

| Method | Path | Purpose |
|---|---|---|
| `GET /` | Serves `index.html` | Sets session cookie on first visit |
| `POST /chat` | Main interaction | Returns SSE stream |
| `POST /reset` | Clear session | Replaces `ConversationState` with fresh instance |

**Dual-mode conversation routing (added Step 2):**

Every `/chat` request now checks `state.analysis_complete` at the top of the handler before any other logic:

```
if state.analysis_complete:
    → Follow-up mode: call explain() with results_context() + question
    → Yield one SSE "chat" event with the answer
    → Return (no Q&A, no re-analysis, no dashboard update)
else:
    → Q&A mode: gather missing fields, then run full analysis
```

This is the simplest possible routing strategy and keeps the two paths completely independent. The follow-up path never touches the engines; the Q&A path never uses `results_context()`.

**Analysis flow (Q&A mode):**

```
1. Run SS claiming analysis (3 scenarios: claim 62 / FRA / 70)
2. Run Monte Carlo × 3 (one per SS scenario, SS income offsets withdrawals)
3. Run Roth conversion optimizer (uses best-success-rate SS scenario)
4. Build combined context string for LLM
5. Call explain() with Monte Carlo + Roth + SS context
6. Set state.analysis_complete = True, store results
7. Yield SSE "chat" (LLM summary) + SSE "dashboard" (full HTML)
```

**SSE event protocol:**

```
event: chat       → append a message bubble to the chat pane
event: status     → update the running/thinking status badge
event: dashboard  → replace the entire dashboard pane with new HTML
```

**`_build_dashboard()` — Step 2 additions:**

Now takes `tax: dict` as a third argument. Renders two sections:

*Section 1 — Monte Carlo (unchanged from Step 1):* Success rate KPI (color-coded green/yellow/red), portfolio at retirement, median at 90, annual spending, traditional savings breakdown.

*Section 2 — Roth Conversion Strategy (new):*
- If `no_opportunity`: shows a single informational note explaining why (no gap window, all Roth, or RMDs stay in same bracket as spending)
- If beneficial: shows conversion window (years + age range), annual conversion amount, tax rate, side-by-side RMD comparison (before vs after), and lifetime tax savings in green

**Success rate color thresholds:**

```
≥ 85%   → green  (#22c55e)   Strong plan
≥ 70%   → yellow (#f59e0b)   Marginal
< 70%   → red    (#ef4444)   Needs intervention
```

---

### 5.10 `web/templates/index.html` + `web/static/style.css`

**Layout:** Two-pane side-by-side. Chat on the left (42% width), dashboard on the right (flex: 1).

**SSE client:** Uses Fetch API with `ReadableStream` rather than `EventSource` because `EventSource` only supports GET and the chat endpoint is POST.

**CSS classes added Step 2:** `.section-header`, `.roth-note` (no-opportunity info card).

**CSS classes added Step 3:** `.kpi-grid-3` (3-column grid for SS cards), `.ss-early` / `.ss-fra` / `.ss-late` (yellow/blue/green accent borders for the three claiming strategy cards).

---

## 6. Request Lifecycle — Full Walkthrough

### 6A — First Analysis (Q&A Mode)

A user types `"Can I retire at 63?"`. Complete path:

```
1. Browser POSTs to /chat with message="Can I retire at 63?"

2. api/app.py checks state.analysis_complete → False → Q&A mode.

3. "retire" detected → extract_retirement_age() parses "63" → state.retirement_age = 63
   SSE "chat": "I can help you plan a retirement at 63!"
   SSE "chat": next_question() → "What is your current age?"

4. User: "55" → state.age = 55
   SSE "chat": "What is your total retirement savings today?"

5. User: "$800k" → state.savings = 800,000
   SSE "chat": "How much do you expect to spend per year in retirement?"

6. User: "$60k" → state.annual_spending = 60,000
   SSE "chat": "Of your $800,000 in savings, what percentage is in traditional accounts?"

7. User: "80%" → state.traditional_pct = 0.80
   SSE "chat": "Are you filing taxes as single or married?"

8. User: "single" → state.filing_status = "single"
   SSE "chat": "What is your estimated monthly Social Security benefit at FRA?"

9. User: "2200" → state.ss_monthly_benefit = 2200.0 → state.is_ready() = True

10. HouseholdTwin constructed with all fields + TaxProfile + SocialSecurity.

11. analyze_claiming_scenarios(pia=2200, birth_year=1960):
    - Claim 62: $1,540/mo, Claim FRA: $2,200/mo, Claim 70: $2,728/mo
    SSE "chat": "✓ Social Security claiming analysis"

12. run_monte_carlo() × 3 (one per SS scenario, SS offsets withdrawals from claiming age):
    SSE "chat": "✓ Monte Carlo (10,000 simulations)"
    SSE "chat": "✓ Portfolio projection to age 90"

13. optimize_roth_conversion(twin):
    - gap_years = 10 (retirement 63 → RMDs at 73)
    SSE "chat": "✓ Roth conversion optimizer"

14. explain() called with combined context → Qwen3 returns 2-3 sentence summary.

15. state.analysis_complete = True; results stored.

16. SSE "chat": LLM summary
    SSE "dashboard": full _build_dashboard() HTML (Monte Carlo + Roth + SS sections)

15. Browser injects HTML → requestAnimationFrame triggers progress bar animation.
```

### 6B — Follow-up Question (Conversation Mode)

After the analysis, user types `"Should I start converting now or wait till I retire?"`:

```
1. Browser POSTs to /chat.

2. api/app.py checks state.analysis_complete → True → Follow-up mode.

3. explain() called with:
   system: "You are a retirement planning advisor in an active conversation..."
   user:   state.results_context() + "\n\nFollow-up question: " + msg

4. results_context() includes:
   - Full user profile (age, retirement age, savings, spending, filing status)
   - Monte Carlo result (success rate, portfolio at retirement, median at 90)
   - Roth conversion recommendation (or "not recommended" with reason)

5. LLM returns a direct, grounded answer referencing the actual numbers.

6. SSE "chat": the answer. Dashboard unchanged. No re-analysis.
```

Total time for follow-up: ~1–2 seconds (just the LLM call, no engines).

---

## 7. Key Architectural Decisions

### Why local-first / Ollama instead of always-on cloud LLM

**The alternative:** Call Claude or GPT-4 for every explanation.

**Why we didn't:** Every retirement session involves multiple LLM calls (analysis summary + follow-up questions). At typical API rates, an active user costs $1–5/month in LLM fees — recurring, scaling with usage. The local-first approach makes the cost curve flat. The LLM is not the product; the math is.

**The tradeoff accepted:** Qwen3:8b quality is lower than Claude Sonnet/Opus. For factual summaries of structured numeric data, this is acceptable.

### Why the Roth optimizer only recommends when beneficial

The naive approach fills to the 22% bracket unconditionally. We discovered this produces negative lifetime savings for users with modest traditional balances — they'd be paying 22% now to avoid 12% later.

The correct algorithm first projects the RMD at age 73 under the no-conversion scenario and checks whether its marginal rate exceeds the current spending rate. Only if `rmd_rate > spending_rate` does conversion make sense. The target bracket ceiling is set to just below the projected RMD rate so every converted dollar is taxed less than it would be as a forced RMD.

This means many typical users (moderate traditional balances, reasonable spending) correctly see "no conversion recommended" — which is honest financial advice, not a gap in the system.

### Why follow-up questions route to the LLM directly, not to the engines

**The alternative:** Parse follow-up questions for scenario changes (e.g., extract a new retirement age, re-run Monte Carlo with updated inputs).

**Why we didn't (yet):** NLP parsing of open-ended questions is fragile. "What if I retire at 65?" and "Is retiring at 65 a good idea?" look identical to a parser but require different handling. The LLM handles both correctly when given the stored results as context, and can gracefully suggest "New session" for a full re-run when the user wants new numbers. Structured scenario switching (Step 5) will add this properly.

### Why SSE instead of WebSockets

The communication pattern is strictly one-directional during a response — server pushes events, client listens. SSE is HTTP, works through proxies without configuration, and is natively supported by the streaming Fetch API. WebSockets add stateful connection management for no benefit in this use case.

### Why server-side HTML for the dashboard

Keeps the JavaScript thin and stateless — the browser just does `innerHTML = data`. The alternative (JSON + client-side rendering) would require a client-side template engine or virtual DOM. The tradeoff is that dashboard layout changes require a server deploy.

This was also forced by a Starlette 1.3.1 + Jinja2 cache bug that caused stale renders. Direct f-string HTML sidesteps the issue entirely.

### Why fixed numpy seed

`numpy.random.default_rng(42)` makes results fully reproducible — same inputs always produce same output. This makes the tool auditable and testable. A result that changes on every run would be impossible to verify.

---

## 8. Dependency Manifest

```
fastapi>=0.111          Web framework + dependency injection
uvicorn[standard]>=0.30 ASGI server (includes watchfiles for --reload)
jinja2>=3.1             Template engine (installed; currently unused due to cache bug)
python-multipart>=0.0.9 Required by FastAPI to parse Form(...) data
numpy>=1.26             Monte Carlo matrix operations
anthropic>=0.28         Cloud LLM fallback (optional at runtime; lazy import)
pdfplumber>=0.11        PDF parsing for Step 6 document upload (not yet used)
```

**No external data libraries:** No yfinance, Alpha Vantage, or similar. All market data is encoded as constants in `Assumptions`. Tax rules are in `data/tax_brackets_2025.json`.

---

## 9. File Structure

```
NextChapter/
├── api/
│   ├── __init__.py
│   └── app.py              FastAPI routes, dual-mode chat handler, SSE, dashboard HTML
│
├── agents/
│   ├── __init__.py
│   └── planner.py          ConversationState (6 fields + analysis state),
│                           sequential Q&A, parsers, results_context()
│
├── core/
│   ├── __init__.py
│   └── digital_twin.py     HouseholdTwin, Person, Accounts (with traditional_pct),
│                           Spending, Assumptions, TaxProfile
│
├── data/
│   ├── tax_brackets_2025.json   2025 IRS brackets, standard deductions, RMD divisors
│   └── ss_rules.json            FRA by birth year, early reduction, delayed credit rates
│
├── engines/
│   ├── __init__.py
│   ├── monte_carlo.py      10,000-path retirement simulation (numpy), SS income offset
│   ├── tax_engine.py       tax_owed(), marginal_rate(), bracket_headroom(),
│   │                       optimize_roth_conversion()
│   └── ss_engine.py        get_fra(), benefit_at_age(), breakeven_age(),
│                           analyze_claiming_scenarios()
│
├── llm/
│   ├── __init__.py
│   └── client.py           explain(): Ollama → Anthropic → static fallback
│
├── web/
│   ├── static/
│   │   └── style.css       Dark theme, KPI grid, progress bar, .section-header, .roth-note
│   └── templates/
│       └── index.html      Two-pane layout, SSE JS client
│
├── requirements.txt
└── ARCHITECTURE.md         This document
```

---

## 10. Development Guide

**Run the server:**

```powershell
cd C:\Siva\Projects\Python\NextChapter
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload --port 8000
```

Then open `http://localhost:8000`.

**Important:** Always use `.venv\Scripts\python.exe -m uvicorn`, not the `uvicorn` script directly. The script entry point has a known silent failure on Windows with this venv configuration.

**Environment variables (all optional):**

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:8b` | Which Ollama model to use |
| `ANTHROPIC_API_KEY` | (none) | Enables cloud LLM fallback; not required |

**Ollama setup:**

```powershell
# One-time: pull the model (5.2 GB)
ollama pull qwen3:8b

# Verify service is running:
# http://localhost:11434/api/tags should return 200
```

**Typical conversation flow (Step 2):**

1. "Can I retire at 63?"
2. Current age → "55"
3. Total savings → "$800k"
4. Annual spending → "$60k"
5. % in traditional accounts → "80%"
6. Filing status → "single"
7. → Analysis runs (Monte Carlo + Roth optimizer + LLM summary + dashboard)
8. Follow-up: "Should I start converting now?" → direct LLM answer, no re-analysis

---

## 11. Roadmap

| Step | Feature | Status | Key files |
|---|---|---|---|
| **1** | Core skeleton — Monte Carlo, SSE streaming, two-pane UI, Ollama LLM | **Complete** | `api/app.py`, `engines/monte_carlo.py`, `agents/planner.py`, `llm/client.py` |
| **2** | Tax Agent — Roth conversion optimizer, follow-up conversation, regex fixes | **Complete** | `engines/tax_engine.py`, `data/tax_brackets_2025.json`, extended `digital_twin.py`, `planner.py`, `app.py` |
| **3** | SS Agent — Social Security claiming scenarios. FRA by birth year, early penalty (5/9%+5/12%/mo), delayed credit (+8%/yr to 70), breakeven analysis, 3× Monte Carlo. | **Complete** | `engines/ss_engine.py`, `data/ss_rules.json`, extended `digital_twin.py`, `planner.py`, `app.py` |
| **4** | Charts — Monte Carlo fan chart (p10/p50/p90 bands over time), SS comparison bar. No npm build step — inline SVG or CDN Chart.js. | Pending | Dashboard additions in `app.py`, `web/static/charts.js` |
| **5** | Full Digital Twin + what-if scenarios. Side-by-side comparison (e.g., retire at 63 vs 65). Spouse support, spending categories, user-adjustable assumptions. Session persistence. | Pending | `core/digital_twin.py` expansion, `api/scenarios.py` |
| **6** | Document upload. SS statement PDF, 1040, brokerage statement → agent extracts and pre-fills the twin. | Pending | `agents/document_agent.py`, `api/upload.py` (uses `pdfplumber` already in requirements) |
