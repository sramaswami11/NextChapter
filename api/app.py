import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agents.planner import ConversationState, extract_retirement_age, process_message
from core.digital_twin import (
    Accounts, Assumptions, HouseholdTwin, Person, Spending, TaxProfile, SocialSecurity
)
from engines.monte_carlo import run_monte_carlo
from engines.tax_engine import optimize_roth_conversion, current_year_roth_advisor
from engines.ss_engine import analyze_claiming_scenarios, benefit_at_age
from llm.client import explain

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")

_INDEX_HTML = Path("web/templates/index.html")

# In-memory sessions keyed by cookie (Step 1: single-user demo)
sessions: dict[str, ConversationState] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session_id = request.cookies.get("nc_session") or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = ConversationState()
    response = HTMLResponse(content=_INDEX_HTML.read_text(encoding="utf-8"))
    response.set_cookie("nc_session", session_id, httponly=True)
    return response


@app.post("/reset")
async def reset(request: Request):
    session_id = request.cookies.get("nc_session")
    if session_id and session_id in sessions:
        sessions[session_id] = ConversationState()
    return HTMLResponse("")


@app.post("/chat")
async def chat(request: Request, message: str = Form(...)):
    session_id = request.cookies.get("nc_session") or str(uuid.uuid4())
    state = sessions.setdefault(session_id, ConversationState())
    msg = message.strip()

    async def generate():
        # ── Follow-up conversation (analysis already complete) ──────────────────
        if state.analysis_complete:
            yield _sse("status", "thinking")
            answer = explain(
                system=(
                    "You are a retirement planning advisor in an active conversation. "
                    "The user's analysis is already complete — the numbers are below. "
                    "Answer the follow-up question directly and specifically using those numbers. "
                    "Be conversational, 2-3 sentences max. No bullet points. "
                    "If the question asks about a different scenario (e.g. retiring later, "
                    "spending less), give a qualitative answer and suggest they click "
                    "'New session' to run a fresh analysis with the new numbers."
                ),
                user=f"{state.results_context()}\n\nFollow-up question: {msg}",
            )
            yield _sse("chat", answer)
            return

        is_retirement_q = any(w in msg.lower() for w in ["retire", "retirement", "stop working"])

        # First message — greet and try to extract retirement age
        if is_retirement_q and state.retirement_age is None and state.age is None:
            extracted = extract_retirement_age(msg)
            if extracted:
                state.retirement_age = extracted
                yield _sse("chat", f"I can help you plan a retirement at {extracted}!")
            else:
                yield _sse("chat", "I can help with that! Let me gather a few details.")
            await asyncio.sleep(0.15)
            yield _sse("chat", state.next_question())
            return

        # Parse user's answer into next missing field
        process_message(state, msg)

        if not state.is_ready():
            q = state.next_question()
            if q:
                yield _sse("chat", q)
            else:
                yield _sse("chat", "I didn't catch that — could you give me a number?")
            return

        # All fields collected — run analysis
        yield _sse("chat", "Got everything I need. Running your analysis now...")
        yield _sse("status", "running")
        await asyncio.sleep(0.1)

        birth_year = 2026 - state.age
        pia_monthly = state.ss_monthly_benefit or 0.0

        twin = HouseholdTwin(
            person=Person(age=state.age, retirement_age=state.retirement_age),
            accounts=Accounts(
                total_savings=state.savings,
                traditional_pct=state.traditional_pct,
            ),
            spending=Spending(annual=state.annual_spending),
            assumptions=Assumptions(),
            tax_profile=TaxProfile(filing_status=state.filing_status),
            ss=SocialSecurity(monthly_pia=pia_monthly, birth_year=birth_year),
        )

        # ── SS claiming scenarios ───────────────────────────────────────────────
        ss_data = None
        if pia_monthly > 0:
            ss_data = analyze_claiming_scenarios(pia_monthly, birth_year)
            yield _sse("chat", "✓ Social Security claiming analysis")
            await asyncio.sleep(0.2)

        # ── Monte Carlo — three SS scenarios ───────────────────────────────────
        mc_no_ss   = run_monte_carlo(twin)  # baseline: no SS income
        yield _sse("chat", "✓ Monte Carlo (10,000 simulations)")
        await asyncio.sleep(0.2)

        if pia_monthly > 0 and ss_data:
            ben_62  = benefit_at_age(pia_monthly, 62.0, birth_year)
            ben_fra = pia_monthly
            ben_70  = benefit_at_age(pia_monthly, 70.0, birth_year)
            fra     = ss_data["fra"]

            mc_62  = run_monte_carlo(twin, ss_monthly=ben_62,  ss_start_age=62.0)
            mc_fra = run_monte_carlo(twin, ss_monthly=ben_fra, ss_start_age=fra)
            mc_70  = run_monte_carlo(twin, ss_monthly=ben_70,  ss_start_age=70.0)
        else:
            mc_62 = mc_fra = mc_70 = mc_no_ss

        # Use the FRA scenario as the "main" result for overall summary
        results = mc_fra if pia_monthly > 0 else mc_no_ss

        yield _sse("chat", "✓ Portfolio projection to age 90")
        await asyncio.sleep(0.2)

        tax = optimize_roth_conversion(twin)
        yield _sse("chat", "✓ Roth conversion optimizer")
        await asyncio.sleep(0.2)

        cy_roth = None
        if state.current_taxable_income is not None:
            cy_roth = current_year_roth_advisor(state.current_taxable_income, state.filing_status)
            yield _sse("chat", "✓ Current year Roth opportunity")
            await asyncio.sleep(0.2)

        # ── Build LLM context ──────────────────────────────────────────────────
        if not tax.get("no_opportunity") and tax["annual_conversion"] > 0:
            tax_context = (
                f"Tax analysis: {tax['gap_years']}-year Roth conversion window "
                f"(ages {state.retirement_age} to 73). "
                f"Suggested annual conversion: ${tax['annual_conversion']:,.0f} "
                f"(marginal rate {tax['conversion_tax_rate']:.0%}). "
                f"First RMD at 73 drops from ${tax['rmd_no_conversion']:,.0f} to "
                f"${tax['rmd_with_conversion']:,.0f}/yr. "
                f"Estimated lifetime tax savings: ${tax['lifetime_tax_savings']:,.0f}."
            )
        elif tax.get("no_opportunity_reason") == "low_rmd":
            tax_context = (
                f"Tax analysis: {tax['gap_years']}-year gap window exists but Roth conversion "
                f"is NOT recommended — projected RMDs at 73 (${tax['rmd_no_conversion']:,.0f}/yr) "
                f"will be taxed at the same {int(tax['current_bracket']*100)}% bracket as retirement "
                f"spending. Converting now at the same rate offers no tax savings."
            )
        elif tax.get("no_opportunity_reason") == "negative_return":
            cur_pct = int(tax['current_bracket'] * 100)
            rmd_pct = int(tax.get('rmd_bracket', tax['current_bracket']) * 100)
            tax_context = (
                f"Tax analysis: {tax['gap_years']}-year gap window exists but Roth conversion "
                f"is NOT recommended — the {rmd_pct - cur_pct}% rate differential ({cur_pct}% now "
                f"vs {rmd_pct}% at RMD time) is too small to offset the upfront conversion tax cost."
            )
        else:
            tax_context = "No Roth conversion gap window (retiring at or after RMD age 73, or all savings in Roth)."

        if pia_monthly > 0 and ss_data:
            ss_context = (
                f"Social Security: PIA ${pia_monthly:,.0f}/mo at FRA {ss_data['fra_label']}. "
                f"Claiming at 62: ${ss_data['claim_62']['monthly']:,}/mo "
                f"({ss_data['claim_62']['pct_vs_fra']}% vs FRA). "
                f"Claiming at 70: ${ss_data['claim_70']['monthly']:,}/mo "
                f"(+{ss_data['claim_70']['pct_vs_fra']}% vs FRA). "
                f"Success rate with SS: claim 62={mc_62['success_rate']}%, "
                f"FRA={mc_fra['success_rate']}%, age 70={mc_70['success_rate']}%."
            )
        else:
            ss_context = "Social Security: none."

        try:
            summary = explain(
                system=(
                    "You are a friendly, direct retirement planning advisor. "
                    "Give a 2-3 sentence plain-English summary covering the Monte Carlo result, "
                    "the most important Roth conversion insight, and the Social Security strategy "
                    "if applicable. Be encouraging but honest. No bullet points."
                ),
                user=(
                    f"Age {state.age}, wants to retire at {state.retirement_age}. "
                    f"Saved ${state.savings:,.0f} ({state.traditional_pct:.0%} traditional), "
                    f"plans to spend ${state.annual_spending:,.0f}/yr. "
                    f"Filing status: {state.filing_status}. "
                    f"Monte Carlo: {results['success_rate']}% success rate over "
                    f"{results['years_in_retirement']} years. "
                    f"Median portfolio at 90: ${results['median_portfolio']:,.0f}. "
                    f"{tax_context} {ss_context}"
                ),
            )
        except Exception:
            summary = (
                f"Your plan shows a {results['success_rate']}% success rate — "
                "see the dashboard for the full breakdown."
            )

        state.analysis_complete = True
        state.last_mc_results = results
        state.last_tax_results = tax
        state.last_ss_results = ss_data
        state.last_cy_roth_results = cy_roth

        yield _sse("chat", summary)
        yield _sse("dashboard", _build_dashboard(results, twin, tax, ss_data, mc_62, mc_fra, mc_70, cy_roth))

    response = StreamingResponse(generate(), media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.set_cookie("nc_session", session_id, httponly=True)
    return response


# ── SSE helpers ────────────────────────────────────────────────────────────────

def _sse(event: str, html: str) -> str:
    data = " ".join(html.split())  # collapse whitespace so data fits on one line
    return f"event: {event}\ndata: {data}\n\n"


def _build_dashboard(
    results: dict,
    twin: HouseholdTwin,
    tax: dict,
    ss_data: dict | None,
    mc_62: dict,
    mc_fra: dict,
    mc_70: dict,
    cy_roth: dict | None = None,
) -> str:
    rate = results["success_rate"]
    bar = int(rate)
    median = results["median_portfolio"]
    port_at_ret = results["portfolio_at_retirement"]
    yrs = results["years_in_retirement"]
    fs_label = "Married" if twin.tax_profile.filing_status == "married" else "Single"

    mc_color = "#22c55e" if rate >= 85 else "#f59e0b" if rate >= 70 else "#ef4444"

    # --- Current year Roth advisor section ---
    if cy_roth:
        cy_rate_pct = int(cy_roth["current_bracket"] * 100)
        cy_income = cy_roth["current_income"]
        cy_headroom = cy_roth["headroom_current"]
        cy_tax = cy_roth["tax_to_fill_current"]
        cy_next = cy_roth["next_bracket"]

        if cy_headroom > 0:
            cy_room_sub = f"convert up to ${cy_headroom:,.0f} and stay in {cy_rate_pct}% bracket"
        else:
            cy_room_sub = "at bracket ceiling — conversions taxed at next rate"

        long_term_no = tax.get("no_opportunity") or tax["annual_conversion"] == 0
        cy_header = "Current Year Tax Bracket (For Reference)" if long_term_no else "Current Year Roth Opportunity"

        cy_section = f"""
<div class="section-header">{cy_header}</div>
<div class="kpi-grid">
  <div class="kpi kpi-primary">
    <div class="kpi-label">Your Tax Bracket</div>
    <div class="kpi-value">{cy_rate_pct}%</div>
    <div class="kpi-sub">on ${cy_income:,.0f} taxable income this year &nbsp;|&nbsp; Filing: {fs_label}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Room in {cy_rate_pct}% Bracket</div>
    <div class="kpi-value">${cy_headroom:,.0f}</div>
    <div class="kpi-sub">{cy_room_sub}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Tax Cost to Fill Bracket</div>
    <div class="kpi-value">${cy_tax:,.0f}</div>
    <div class="kpi-sub">additional federal tax on conversion</div>
  </div>
</div>"""

        if cy_next is not None and cy_headroom > 0:
            cy_next_pct = int(cy_next * 100)
            cy_section += f"""
<div class="assumptions-box">
  <div class="assumptions-title">Stretch option: extend into {cy_next_pct}% bracket</div>
  <div class="assumptions-row"><span>Additional conversion into {cy_next_pct}% bracket</span><span>${cy_roth['incremental_for_next']:,.0f}</span></div>
  <div class="assumptions-row"><span>Total conversion (fill both brackets)</span><span>${cy_roth['headroom_to_next_ceiling']:,.0f}</span></div>
  <div class="assumptions-row"><span>Total additional tax</span><span>${cy_roth['tax_to_fill_next']:,.0f}</span></div>
</div>"""
    else:
        cy_section = ""

    # --- Roth conversion section ---
    if tax.get("no_opportunity") or tax["annual_conversion"] == 0:
        reason = tax.get("no_opportunity_reason")
        if reason == "all_roth":
            msg = "All savings already in Roth — no conversion needed."
        elif reason == "no_gap":
            msg = "No conversion window — you retire at or after RMD age 73."
        elif reason == "low_rmd":
            bracket_pct = int(tax["current_bracket"] * 100)
            msg = (
                f"Your projected RMDs at 73 will stay in the {bracket_pct}% bracket — "
                f"the same as your retirement spending. Converting now at the same rate "
                f"costs more (pay sooner) without saving tax. No conversion recommended."
            )
        elif reason == "negative_return":
            cur_pct = int(tax["current_bracket"] * 100)
            rmd_pct = int(tax.get("rmd_bracket", tax["current_bracket"]) * 100)
            msg = (
                f"Although your projected RMDs at 73 will be taxed at {rmd_pct}% vs your "
                f"current {cur_pct}% bracket, the {rmd_pct - cur_pct}% rate differential "
                f"is too small to offset the upfront conversion tax cost over the "
                f"{tax['gap_years']}-year window. No conversion recommended."
            )
        else:
            msg = "No Roth conversion opportunity identified."
        roth_section = f'<div class="roth-note">{msg}</div>'
    else:
        gap = tax["gap_years"]
        ret_age = twin.person.retirement_age
        conv = tax["annual_conversion"]
        rate_pct = int(tax["conversion_tax_rate"] * 100)
        rmd_before = tax["rmd_no_conversion"]
        rmd_after  = tax["rmd_with_conversion"]
        savings    = tax["lifetime_tax_savings"]
        cur_bracket = int(tax["current_bracket"] * 100)
        savings_color = "#22c55e" if savings > 0 else "#f59e0b"

        roth_section = f"""
<div class="section-header">Roth Conversion Strategy</div>
<div class="kpi-grid">
  <div class="kpi kpi-primary">
    <div class="kpi-label">Conversion Window</div>
    <div class="kpi-value">{gap} years</div>
    <div class="kpi-sub">Ages {ret_age} &rarr; 73 &nbsp;|&nbsp; Current bracket: {cur_bracket}% &nbsp;|&nbsp; Filing: {fs_label}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Convert Per Year</div>
    <div class="kpi-value">${conv:,.0f}</div>
    <div class="kpi-sub">fills to {rate_pct}% bracket ceiling</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Tax Cost per Year</div>
    <div class="kpi-value">${tax['annual_tax_cost']:,.0f}</div>
    <div class="kpi-sub">additional tax on conversion</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">First RMD at 73 &mdash; Without Strategy</div>
    <div class="kpi-value" style="color:#ef4444">${rmd_before:,.0f}<span style="font-size:14px;font-weight:400">/yr</span></div>
    <div class="kpi-sub">forced taxable withdrawal</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">First RMD at 73 &mdash; With Strategy</div>
    <div class="kpi-value" style="color:#22c55e">${rmd_after:,.0f}<span style="font-size:14px;font-weight:400">/yr</span></div>
    <div class="kpi-sub">after converting during gap</div>
  </div>
  <div class="kpi kpi-primary">
    <div class="kpi-label">Estimated Lifetime Tax Savings</div>
    <div class="kpi-value" style="color:{savings_color}">${savings:,.0f}</div>
    <div class="kpi-sub">RMD tax reduction &minus; conversion tax cost (ages 73&ndash;90)</div>
  </div>
</div>"""

    # --- Social Security section ---
    if ss_data and twin.ss.monthly_pia > 0:
        fra_label = ss_data["fra_label"]
        c62  = ss_data["claim_62"]
        cfra = ss_data["claim_fra"]
        c70  = ss_data["claim_70"]
        be_62_fra  = ss_data["breakeven_62_vs_fra"]
        be_fra_70  = ss_data["breakeven_fra_vs_70"]
        be_62_70   = ss_data["breakeven_62_vs_70"]

        def _sr_color(r):
            return "#22c55e" if r >= 85 else "#f59e0b" if r >= 70 else "#ef4444"

        ss_section = f"""
<div class="section-header">Social Security Claiming Strategy</div>
<div class="kpi-grid-3">
  <div class="kpi ss-early">
    <div class="kpi-label">Claim at 62 (Early)</div>
    <div class="kpi-value">${c62['monthly']:,}<span style="font-size:14px;font-weight:400">/mo</span></div>
    <div class="kpi-sub">${c62['annual']:,}/yr &nbsp;&bull;&nbsp; {c62['pct_vs_fra']}% vs FRA</div>
    <div class="kpi-sub" style="margin-top:8px">Success rate: <strong style="color:{_sr_color(mc_62['success_rate'])}">{mc_62['success_rate']}%</strong></div>
  </div>
  <div class="kpi ss-fra kpi-primary">
    <div class="kpi-label">Claim at FRA ({fra_label})</div>
    <div class="kpi-value">${cfra['monthly']:,}<span style="font-size:14px;font-weight:400">/mo</span></div>
    <div class="kpi-sub">${cfra['annual']:,}/yr &nbsp;&bull;&nbsp; your PIA</div>
    <div class="kpi-sub" style="margin-top:8px">Success rate: <strong style="color:{_sr_color(mc_fra['success_rate'])}">{mc_fra['success_rate']}%</strong></div>
  </div>
  <div class="kpi ss-late">
    <div class="kpi-label">Claim at 70 (Maximum)</div>
    <div class="kpi-value">${c70['monthly']:,}<span style="font-size:14px;font-weight:400">/mo</span></div>
    <div class="kpi-sub">${c70['annual']:,}/yr &nbsp;&bull;&nbsp; +{c70['pct_vs_fra']}% vs FRA</div>
    <div class="kpi-sub" style="margin-top:8px">Success rate: <strong style="color:{_sr_color(mc_70['success_rate'])}">{mc_70['success_rate']}%</strong></div>
  </div>
</div>
<div class="assumptions-box">
  <div class="assumptions-title">Breakeven Ages (when delayed claiming pays off)</div>
  <div class="assumptions-row"><span>Claim 62 vs FRA &mdash; break even at age</span><span>{be_62_fra}</span></div>
  <div class="assumptions-row"><span>Claim FRA vs 70 &mdash; break even at age</span><span>{be_fra_70}</span></div>
  <div class="assumptions-row"><span>Claim 62 vs 70 &mdash; break even at age</span><span>{be_62_70}</span></div>
</div>"""
    else:
        ss_section = '<div class="roth-note">Social Security: not included in this analysis.</div>'

    return f"""
<div class="dash-header">Retirement at {twin.person.retirement_age}</div>
<div class="kpi-grid">
  <div class="kpi kpi-primary">
    <div class="kpi-label">Monte Carlo Success Rate</div>
    <div class="kpi-value" style="color:{mc_color}">{rate}%</div>
    <div class="progress-bar">
      <div class="progress-fill" style="--fill:{bar}%; background:{mc_color}"></div>
    </div>
    <div class="kpi-sub">10,000 simulations &nbsp;|&nbsp; {yrs}-yr horizon</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Portfolio at Retirement</div>
    <div class="kpi-value">${port_at_ret:,.0f}</div>
    <div class="kpi-sub">at age {twin.person.retirement_age}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Median Portfolio at 90</div>
    <div class="kpi-value">${median:,.0f}</div>
    <div class="kpi-sub">surviving simulations</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Annual Spending</div>
    <div class="kpi-value">${twin.spending.annual:,.0f}</div>
    <div class="kpi-sub">today's dollars</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Traditional Savings</div>
    <div class="kpi-value">${twin.accounts.traditional_balance:,.0f}</div>
    <div class="kpi-sub">{twin.accounts.traditional_pct:.0%} of portfolio (pre-tax)</div>
  </div>
</div>
<div class="assumptions-box">
  <div class="assumptions-title">Assumptions</div>
  <div class="assumptions-row"><span>Portfolio mix</span><span>60% stocks / 40% bonds</span></div>
  <div class="assumptions-row"><span>Stock return</span><span>9.0% nominal</span></div>
  <div class="assumptions-row"><span>Bond return</span><span>4.0% nominal</span></div>
  <div class="assumptions-row"><span>Inflation</span><span>2.5%</span></div>
</div>
{roth_section}
{cy_section}
{ss_section}
"""
