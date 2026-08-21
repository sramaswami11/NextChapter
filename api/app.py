import asyncio
import json
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
from engines.tax_engine import (
    optimize_roth_conversion, current_year_roth_advisor, rmd_elimination_calculator,
    roth_conversion_window_optimizer, ltcg_harvest_advisor,
)
from engines.ss_engine import analyze_claiming_scenarios, benefit_at_age, recommended_strategy
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
        le = state.life_expectancy or 84

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

        # ── Spouse SS ──────────────────────────────────────────────────────────
        spouse_ss_data = None
        spouse_ss_claiming_age = 999.0
        spouse_ss_monthly_at_claiming = 0.0
        _spouse_rec = None
        if (state.filing_status == "married"
                and state.spouse_age is not None
                and state.spouse_ss_benefit is not None
                and state.spouse_ss_benefit > 0):
            spouse_birth_year = 2026 - state.spouse_age
            spouse_ss_data = analyze_claiming_scenarios(state.spouse_ss_benefit, spouse_birth_year)
            _spouse_le = state.spouse_life_expectancy or le
            _spouse_rec = recommended_strategy(spouse_ss_data, _spouse_le)
            if _spouse_rec == "claim_62":
                spouse_ss_claiming_age = 62.0
                spouse_ss_monthly_at_claiming = benefit_at_age(state.spouse_ss_benefit, 62.0, spouse_birth_year)
            elif _spouse_rec == "claim_fra":
                spouse_ss_claiming_age = float(spouse_ss_data["fra"])
                spouse_ss_monthly_at_claiming = state.spouse_ss_benefit
            else:
                spouse_ss_claiming_age = 70.0
                spouse_ss_monthly_at_claiming = benefit_at_age(state.spouse_ss_benefit, 70.0, spouse_birth_year)

        # ── Monte Carlo — three SS scenarios ───────────────────────────────────
        mc_no_ss = run_monte_carlo(
            twin,
            spouse_ss_monthly=spouse_ss_monthly_at_claiming,
            spouse_ss_start_age=spouse_ss_claiming_age,
        )
        yield _sse("chat", "✓ Monte Carlo (10,000 simulations)")
        await asyncio.sleep(0.2)

        if pia_monthly > 0 and ss_data:
            ben_62  = benefit_at_age(pia_monthly, 62.0, birth_year)
            ben_fra = pia_monthly
            ben_70  = benefit_at_age(pia_monthly, 70.0, birth_year)
            fra     = ss_data["fra"]

            mc_62  = run_monte_carlo(twin, ss_monthly=ben_62,  ss_start_age=62.0,
                                     spouse_ss_monthly=spouse_ss_monthly_at_claiming,
                                     spouse_ss_start_age=spouse_ss_claiming_age)
            mc_fra = run_monte_carlo(twin, ss_monthly=ben_fra, ss_start_age=fra,
                                     spouse_ss_monthly=spouse_ss_monthly_at_claiming,
                                     spouse_ss_start_age=spouse_ss_claiming_age)
            mc_70  = run_monte_carlo(twin, ss_monthly=ben_70,  ss_start_age=70.0,
                                     spouse_ss_monthly=spouse_ss_monthly_at_claiming,
                                     spouse_ss_start_age=spouse_ss_claiming_age)
        else:
            ben_62 = ben_fra = ben_70 = 0.0
            mc_62 = mc_fra = mc_70 = mc_no_ss

        # Use the FRA scenario as the "main" result for overall summary
        results = mc_fra if pia_monthly > 0 else mc_no_ss

        yield _sse("chat", "✓ Portfolio projection to age 90")
        await asyncio.sleep(0.2)

        tax = optimize_roth_conversion(twin)
        yield _sse("chat", "✓ Roth conversion optimizer")
        await asyncio.sleep(0.2)

        elim = rmd_elimination_calculator(twin)
        yield _sse("chat", "✓ RMD elimination analysis")
        await asyncio.sleep(0.2)

        # ── Roth Conversion Window Optimizer ──────────────────────────────────
        if pia_monthly > 0 and ss_data:
            _win_rec = recommended_strategy(ss_data, le)
            if _win_rec == "claim_62":
                _win_ss_age, _win_ss_monthly = 62.0, ben_62
            elif _win_rec == "claim_fra":
                _win_ss_age, _win_ss_monthly = float(ss_data["fra"]), ben_fra
            else:
                _win_ss_age, _win_ss_monthly = 70.0, ben_70
        else:
            _win_ss_age, _win_ss_monthly = 999.0, 0.0
        window = roth_conversion_window_optimizer(
            twin,
            ss_claiming_age=_win_ss_age,
            ss_monthly_at_claiming=_win_ss_monthly,
            life_expectancy=le,
            current_taxable_income=state.current_taxable_income,
            spouse_working=state.spouse_working or False,
            spouse_income=state.spouse_income,
            spouse_retirement_age=state.spouse_retirement_age,
            spouse_ss_monthly=spouse_ss_monthly_at_claiming,
            spouse_ss_start_age=spouse_ss_claiming_age,
        )
        yield _sse("chat", "✓ Roth conversion timeline")
        await asyncio.sleep(0.2)

        # ── LTCG Harvest Advisor ───────────────────────────────────────────────
        ltcg = None
        if state.unrealized_ltcg is not None and state.unrealized_ltcg > 0:
            ltcg = ltcg_harvest_advisor(
                twin,
                unrealized_ltcg=state.unrealized_ltcg,
                estate_intent=state.estate_intent,
                ss_annual=_win_ss_monthly * 12,
                ss_claiming_age=_win_ss_age,
                life_expectancy=le,
                current_age=state.age,
                spouse_age=state.spouse_age,
                spouse_ss_annual=spouse_ss_monthly_at_claiming * 12,
                spouse_ss_start_age=spouse_ss_claiming_age,
            )
            yield _sse("chat", "✓ Capital gains harvesting analysis")
            await asyncio.sleep(0.2)

        cy_roth = None
        if state.current_taxable_income is not None:
            cy_roth = current_year_roth_advisor(state.current_taxable_income, state.filing_status)
            yield _sse("chat", "✓ Current year Roth opportunity")
            await asyncio.sleep(0.2)

        # ── Build LLM context ──────────────────────────────────────────────────
        if not tax.get("no_opportunity") and tax["annual_conversion"] > 0:
            elim_note = ""
            if elim.get("possible") and elim.get("annual_conversion", 0) > 0:
                elim_note = (
                    f" Full RMD elimination alternative: convert ${elim['annual_conversion']:,.0f}/yr "
                    f"to drain traditional IRA to $0 by 73 "
                    f"(net lifetime savings ${elim['lifetime_net_savings']:,.0f})."
                )
            tax_context = (
                f"Tax analysis: {tax['gap_years']}-year Roth conversion window "
                f"(ages {state.retirement_age} to 73). "
                f"Suggested annual conversion: ${tax['annual_conversion']:,.0f} "
                f"(marginal rate {tax['conversion_tax_rate']:.0%}). "
                f"First RMD at 73 drops from ${tax['rmd_no_conversion']:,.0f} to "
                f"${tax['rmd_with_conversion']:,.0f}/yr. "
                f"Estimated lifetime tax savings: ${tax['lifetime_tax_savings']:,.0f}.{elim_note}"
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
            ss_rec = recommended_strategy(ss_data, le)
            rec_labels = {
                "claim_62": "claim at 62",
                "claim_fra": f"claim at FRA ({ss_data['fra_label']})",
                "claim_70": "delay to 70",
            }
            ss_context = (
                f"Social Security: PIA ${pia_monthly:,.0f}/mo at FRA {ss_data['fra_label']}. "
                f"Claiming at 62: ${ss_data['claim_62']['monthly']:,}/mo "
                f"({ss_data['claim_62']['pct_vs_fra']}% vs FRA). "
                f"Claiming at 70: ${ss_data['claim_70']['monthly']:,}/mo "
                f"(+{ss_data['claim_70']['pct_vs_fra']}% vs FRA). "
                f"Breakeven FRA vs 70: age {ss_data['breakeven_fra_vs_70']}. "
                f"Client's estimated life expectancy: {le}. "
                f"RECOMMENDED strategy based on longevity: {rec_labels[ss_rec]}. "
                f"Success rates: claim 62={mc_62['success_rate']}%, "
                f"FRA={mc_fra['success_rate']}%, age 70={mc_70['success_rate']}%."
            )
            if spouse_ss_data and spouse_ss_data.get("pia_monthly", 0) > 0:
                _srec_label = rec_labels.get(_spouse_rec, "see analysis")
                _sle_str = f" (life expectancy {state.spouse_life_expectancy})" if state.spouse_life_expectancy else ""
                ss_context += (
                    f" SPOUSE SS: PIA ${state.spouse_ss_benefit:,.0f}/mo at FRA "
                    f"{spouse_ss_data['fra_label']}{_sle_str}. Recommended: {_srec_label} "
                    f"(${spouse_ss_monthly_at_claiming:,.0f}/mo)."
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
        state.last_elim_results = elim
        state.last_window_results = window
        state.last_ltcg_results = ltcg

        yield _sse("chat", summary)
        yield _sse("dashboard", _build_dashboard(results, twin, tax, ss_data, mc_62, mc_fra, mc_70, cy_roth, state.life_expectancy, elim, window, spouse_ss_data, _spouse_rec, state.spouse_life_expectancy, ltcg))

        # ── Chart data (separate event so JS can init Chart.js after canvas is in DOM)
        chart_payload: dict = {
            "fan": {
                "ages": results["ages"],
                "p10":  results["p10_by_year"],
                "p50":  results["p50_by_year"],
                "p90":  results["p90_by_year"],
            },
            "ss": None,
        }
        if pia_monthly > 0 and ss_data:
            rec_idx = ["claim_62", "claim_fra", "claim_70"].index(ss_rec)
            chart_payload["ss"] = {
                "labels":  ["Claim at 62", f"FRA ({ss_data['fra_label']})", "Claim at 70"],
                "monthly": [ss_data["claim_62"]["monthly"], ss_data["claim_fra"]["monthly"], ss_data["claim_70"]["monthly"]],
                "success": [mc_62["success_rate"], mc_fra["success_rate"], mc_70["success_rate"]],
                "rec":     rec_idx,
            }
        yield _sse("chartdata", json.dumps(chart_payload, separators=(',', ':')))

    response = StreamingResponse(generate(), media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.set_cookie("nc_session", session_id, httponly=True)
    return response


# ── SSE helpers ────────────────────────────────────────────────────────────────

def _sse(event: str, html: str) -> str:
    data = " ".join(html.split())  # collapse whitespace so data fits on one line
    return f"event: {event}\ndata: {data}\n\n"


def _window_section(window: dict) -> str:
    if not window:
        return ""

    rec = window.get("current_recommendation", "")
    rec_note = window.get("current_recommendation_note", "")
    optimal_start = window.get("optimal_start_age")
    optimal_end = window.get("optimal_end_age")
    optimal_conv = window.get("optimal_annual_conversion", 0)
    optimal_bracket = window.get("optimal_bracket")
    phases = window.get("phases", [])

    _rec_colors = {
        "start_now": "#22c55e",
        "wait_for_retirement": "#f59e0b",
        "wait_for_window": "#f59e0b",
        "window_passed": "#94a3b8",
        "never": "#94a3b8",
    }
    _rec_labels = {
        "start_now": "Start Converting Now",
        "wait_for_retirement": "Wait Until Retirement",
        "wait_for_window": "Wait for Optimal Window",
        "window_passed": "Optimal Window Has Passed",
        "never": "Conversion Not Recommended",
    }
    rec_color = _rec_colors.get(rec, "#94a3b8")
    rec_label = _rec_labels.get(rec, "See Analysis")

    total_conv = window.get("total_converted", 0)
    pct_conv = window.get("pct_converted", 0.0)

    if optimal_start is not None and optimal_end is not None and optimal_conv > 0 and optimal_bracket is not None:
        top = f"""<div class="kpi-grid">
  <div class="kpi kpi-primary">
    <div class="kpi-label">When to Convert</div>
    <div class="kpi-value" style="font-size:20px;color:{rec_color}">{rec_label}</div>
    <div class="kpi-sub">{rec_note}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Optimal Window</div>
    <div class="kpi-value">Ages {optimal_start}&ndash;{optimal_end}</div>
    <div class="kpi-sub">{optimal_end - optimal_start + 1} years &nbsp;&bull;&nbsp; {window.get("optimal_phase_name","")}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Convert Per Year (in window)</div>
    <div class="kpi-value">${optimal_conv:,.0f}</div>
    <div class="kpi-sub">at {int(optimal_bracket * 100)}% bracket</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Total Pre-Tax Converted to Roth</div>
    <div class="kpi-value">${total_conv:,.0f}</div>
    <div class="kpi-sub">{pct_conv:.1f}% of traditional balance at retirement &nbsp;&bull;&nbsp; before RMD age 73</div>
  </div>
</div>"""
    else:
        top = f'<div class="roth-note" style="color:{rec_color};margin-top:12px">{rec_note}</div>'

    _col = "display:grid;grid-template-columns:2fr 0.8fr 1.1fr 0.7fr 1.2fr;gap:6px;align-items:center"
    header = (
        f'<div style="{_col};padding:4px 0 8px;border-bottom:1px solid #1c2035;'
        f'font-size:11px;font-weight:600;color:#5a6080;text-transform:uppercase;letter-spacing:.5px">'
        f'<span>Phase</span><span>Ages</span><span>Est. Income</span><span>Bracket</span><span>Verdict</span></div>'
    )

    rows = ""
    opt_name = window.get("optimal_phase_name")
    for p in phases:
        name = p["name"]
        ages = f"{p['start_age']}&ndash;{p['end_age']}"
        inc = f"${p['base_taxable']:,}" if p.get("base_taxable") is not None else "Variable"
        brk = f"{int(p['bracket'] * 100)}%" if p.get("bracket") is not None else "&mdash;"
        is_opt = name == opt_name and p.get("conversion_friendly") and p.get("recommended_annual_conversion", 0) > 0
        is_ok = p.get("conversion_friendly") and not is_opt and p.get("recommended_annual_conversion", 0) > 0
        if is_opt:
            badge = '<span style="background:#22c55e;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">&#9733; Best</span>'
        elif is_ok:
            badge = '<span style="background:#f59e0b;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">Possible</span>'
        elif name == "Working Years":
            badge = '<span style="color:#5a6080;font-size:12px">Variable</span>'
        else:
            badge = '<span style="color:#5a6080;font-size:12px">Not optimal</span>'

        note_html = f'<div style="font-size:11px;color:#5a6080;padding:0 0 4px;grid-column:1/-1">{p.get("note","")}</div>'
        rows += (
            f'<div style="{_col};padding:6px 0 2px;border-bottom:1px solid #1c2035;font-size:13px;color:#c8cdd8">'
            f'<span>{name}</span>'
            f'<span style="color:#5a6080">{ages}</span>'
            f'<span style="color:#5a6080">{inc}</span>'
            f'<span style="color:#5a6080">{brk}</span>'
            f'<span>{badge}</span></div>'
            f'<div style="{_col};padding:0 0 4px;border-bottom:1px solid #1c2035">{note_html}</div>'
        )

    table = f'<div class="assumptions-box"><div class="assumptions-title">Tax Bracket by Life Stage</div>{header}{rows}</div>'

    return f"""<details class="accordion">
<summary class="accordion-header">Roth Conversion Timeline &mdash; When to Convert</summary>
{top}
{table}
</details>"""


def _ltcg_section(ltcg: dict) -> str:
    rec = ltcg.get("recommendation", "")
    rec_note = ltcg.get("recommendation_note", "")
    best = ltcg.get("best_phase")
    annual_harvest = ltcg.get("annual_harvest", 0)
    hidden_cost = ltcg.get("hidden_roth_cost", 0.0)
    ordinary_rate = ltcg.get("ordinary_rate_in_gap", 0.0)
    phases = ltcg.get("phases", [])
    unrealized = ltcg.get("unrealized_ltcg", 0)
    ltcg_ceiling = ltcg.get("ltcg_taxable_ceiling", 0)
    has_opp = ltcg.get("has_harvest_opportunity", False)
    estate_intent = ltcg.get("estate_intent")

    _rec_colors = {
        "harvest_first": "#22c55e",
        "heirs_convert": "#f59e0b",
        "no_room": "#94a3b8",
    }
    _rec_labels = {
        "harvest_first": "Harvest Gains First",
        "heirs_convert": "Skip Harvest — Convert IRA",
        "no_room": "No Room for 0% Harvesting",
    }
    rec_color = _rec_colors.get(rec, "#94a3b8")
    rec_label = _rec_labels.get(rec, "See Analysis")

    intent_label = {"spend": "Spend in retirement", "heirs": "Leave to heirs"}.get(estate_intent or "", "Not specified")

    if has_opp and best:
        hidden_pct = int(hidden_cost * 100)
        ordinary_pct = int(ordinary_rate * 100)
        top = f"""<div class="kpi-grid">
  <div class="kpi kpi-primary">
    <div class="kpi-label">Recommendation</div>
    <div class="kpi-value" style="font-size:20px;color:{rec_color}">{rec_label}</div>
    <div class="kpi-sub">{rec_note}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Unrealized Long-Term Gains</div>
    <div class="kpi-value">${unrealized:,.0f}</div>
    <div class="kpi-sub">in taxable brokerage &nbsp;&bull;&nbsp; {intent_label}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Annual Tax-Free Harvest</div>
    <div class="kpi-value" style="color:#22c55e">${annual_harvest:,.0f}</div>
    <div class="kpi-sub">during {best['name']} &nbsp;&bull;&nbsp; {best['ltcg_room_taxable']:,} room in 0% bracket</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Hidden Cost of Roth Conversion Here</div>
    <div class="kpi-value" style="color:#ef4444">{hidden_pct}%</div>
    <div class="kpi-sub">{ordinary_pct}% ordinary income tax + 15% LTCG displaced from 0% bucket</div>
  </div>
</div>"""
    else:
        top = f'<div class="roth-note" style="color:{rec_color};margin-top:12px">{rec_note}</div>'

    _col = "display:grid;grid-template-columns:2.2fr 0.9fr 1.1fr 1.2fr 1.4fr;gap:6px;align-items:center"
    header = (
        f'<div style="{_col};padding:4px 0 8px;border-bottom:1px solid #1c2035;'
        f'font-size:11px;font-weight:600;color:#5a6080;text-transform:uppercase;letter-spacing:.5px">'
        f'<span>Phase</span><span>Ages</span><span>Ordinary Income</span>'
        f'<span>0% LTCG Room</span><span>Constraint</span></div>'
    )

    rows = ""
    best_name = best["name"] if best else None
    for p in phases:
        name = p["name"]
        ages = f"{p['start_age']}&ndash;{p['end_age']}"
        inc = f"${p['ordinary_income']:,}"
        room = f"${p['ltcg_room_taxable']:,}" if p["ltcg_room_taxable"] > 0 else "&mdash;"
        constraint = p.get("constraint", "")
        is_best = name == best_name and p["ltcg_room_taxable"] > 0
        if is_best:
            badge = f'<span style="background:#22c55e;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">&#9733; Best window</span>'
            room_color = "#22c55e"
        elif p["aca_applies"]:
            badge = f'<span style="color:#f59e0b;font-size:12px">ACA cliff applies</span>'
            room_color = "#f59e0b"
        else:
            badge = f'<span style="color:#5a6080;font-size:12px">See note</span>'
            room_color = "#5a6080"

        rows += (
            f'<div style="{_col};padding:6px 0 2px;border-bottom:1px solid #1c2035;font-size:13px;color:#c8cdd8">'
            f'<span>{name}</span>'
            f'<span style="color:#5a6080">{ages}</span>'
            f'<span style="color:#5a6080">{inc}</span>'
            f'<span style="color:{room_color};font-weight:600">{room}</span>'
            f'<span>{badge}</span></div>'
            f'<div style="{_col};padding:0 0 4px;border-bottom:1px solid #1c2035">'
            f'<div style="font-size:11px;color:#5a6080;grid-column:1/-1">{constraint}</div></div>'
        )

    table = (
        f'<div class="assumptions-box">'
        f'<div class="assumptions-title">0% LTCG Room by Life Phase '
        f'(2026 threshold: ${ltcg_ceiling:,} taxable income)</div>'
        f'{header}{rows}</div>'
    )

    return f"""<details class="accordion">
<summary class="accordion-header">Capital Gains Harvesting &mdash; The Other 0% Window</summary>
{top}
{table}
</details>"""


def _build_dashboard(
    results: dict,
    twin: HouseholdTwin,
    tax: dict,
    ss_data: dict | None,
    mc_62: dict,
    mc_fra: dict,
    mc_70: dict,
    cy_roth: dict | None = None,
    life_expectancy: int | None = None,
    elim: dict | None = None,
    window: dict | None = None,
    spouse_ss_data: dict | None = None,
    spouse_ss_rec: str | None = None,
    spouse_le: int | None = None,
    ltcg: dict | None = None,
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

        cy_section = f"""<details class="accordion">
<summary class="accordion-header">{cy_header}</summary>
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
        cy_section += "</details>"
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
        roth_section = f'<details class="accordion"><summary class="accordion-header">Roth Conversion</summary><div class="roth-note">{msg}</div></details>'
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

        roth_section = f"""<details class="accordion">
<summary class="accordion-header">Roth Conversion Strategy</summary>
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

        if elim and elim.get("possible") and elim.get("annual_conversion", 0) > 0:
            elim_conv = elim["annual_conversion"]
            elim_rate_pct = int(elim["conversion_tax_rate"] * 100)
            elim_tax = elim["annual_tax_cost"]
            elim_net = elim["lifetime_net_savings"]
            roth_section += f"""
<div class="assumptions-box">
  <div class="assumptions-title">Strategy Comparison: Partial vs. Full RMD Elimination</div>
  <div class="assumptions-row"><span></span><span>Partial (recommended)</span><span>Full Elimination</span></div>
  <div class="assumptions-row"><span>Convert per year</span><span>${conv:,.0f}</span><span>${elim_conv:,.0f}</span></div>
  <div class="assumptions-row"><span>Marginal rate</span><span>{rate_pct}%</span><span>{elim_rate_pct}%</span></div>
  <div class="assumptions-row"><span>Annual tax cost</span><span>${tax['annual_tax_cost']:,.0f}</span><span>${elim_tax:,.0f}</span></div>
  <div class="assumptions-row"><span>Est. lifetime savings</span><span>${savings:,.0f}</span><span>${elim_net:,.0f}</span></div>
</div>"""
        roth_section += "</details>"

    # --- Social Security section ---
    if ss_data and twin.ss.monthly_pia > 0:
        fra_label = ss_data["fra_label"]
        c62  = ss_data["claim_62"]
        cfra = ss_data["claim_fra"]
        c70  = ss_data["claim_70"]
        be_62_fra  = ss_data["breakeven_62_vs_fra"]
        be_fra_70  = ss_data["breakeven_fra_vs_70"]
        be_62_70   = ss_data["breakeven_62_vs_70"]
        le = life_expectancy or 84
        ss_rec = recommended_strategy(ss_data, le)

        def _sr_color(r):
            return "#22c55e" if r >= 85 else "#f59e0b" if r >= 70 else "#ef4444"

        def _rec_badge(key):
            return ' &nbsp;<span style="font-size:11px;background:#22c55e;color:#fff;padding:2px 6px;border-radius:4px">&#9733; Recommended</span>' if ss_rec == key else ""

        ss_section = f"""<details class="accordion">
<summary class="accordion-header">Social Security Claiming Strategy</summary>
<div class="kpi-grid-3">
  <div class="kpi ss-early{"  kpi-primary" if ss_rec == "claim_62" else ""}">
    <div class="kpi-label">Claim at 62 (Early){_rec_badge("claim_62")}</div>
    <div class="kpi-value">${c62['monthly']:,}<span style="font-size:14px;font-weight:400">/mo</span></div>
    <div class="kpi-sub">${c62['annual']:,}/yr &nbsp;&bull;&nbsp; {c62['pct_vs_fra']}% vs FRA</div>
    <div class="kpi-sub" style="margin-top:8px">Success rate: <strong style="color:{_sr_color(mc_62['success_rate'])}">{mc_62['success_rate']}%</strong></div>
  </div>
  <div class="kpi ss-fra{"  kpi-primary" if ss_rec == "claim_fra" else ""}">
    <div class="kpi-label">Claim at FRA ({fra_label}){_rec_badge("claim_fra")}</div>
    <div class="kpi-value">${cfra['monthly']:,}<span style="font-size:14px;font-weight:400">/mo</span></div>
    <div class="kpi-sub">${cfra['annual']:,}/yr &nbsp;&bull;&nbsp; your PIA</div>
    <div class="kpi-sub" style="margin-top:8px">Success rate: <strong style="color:{_sr_color(mc_fra['success_rate'])}">{mc_fra['success_rate']}%</strong></div>
  </div>
  <div class="kpi ss-late{"  kpi-primary" if ss_rec == "claim_70" else ""}">
    <div class="kpi-label">Claim at 70 (Maximum){_rec_badge("claim_70")}</div>
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
  <div class="assumptions-row"><span>Your estimated life expectancy</span><span>age {le}</span></div>
</div>"""

        if spouse_ss_data and spouse_ss_data.get("pia_monthly", 0) > 0:
            s62  = spouse_ss_data["claim_62"]
            sfra = spouse_ss_data["claim_fra"]
            s70  = spouse_ss_data["claim_70"]
            s_fra_label  = spouse_ss_data["fra_label"]
            s_be_fra_70  = spouse_ss_data["breakeven_fra_vs_70"]
            def _s_badge(key):
                return ' &nbsp;<span style="font-size:11px;background:#22c55e;color:#fff;padding:2px 6px;border-radius:4px">&#9733; Rec</span>' if spouse_ss_rec == key else ""
            ss_section += f"""
<div class="assumptions-box" style="margin-top:12px">
  <div class="assumptions-title">Spouse&#8217;s Social Security (FRA: {s_fra_label})</div>
  <div class="assumptions-row"><span>Claim at 62{_s_badge("claim_62")}</span><span>${s62['monthly']:,}/mo &nbsp;&bull;&nbsp; ${s62['annual']:,}/yr</span></div>
  <div class="assumptions-row"><span>Claim at FRA ({s_fra_label}){_s_badge("claim_fra")}</span><span>${sfra['monthly']:,}/mo &nbsp;&bull;&nbsp; ${sfra['annual']:,}/yr</span></div>
  <div class="assumptions-row"><span>Claim at 70{_s_badge("claim_70")}</span><span>${s70['monthly']:,}/mo &nbsp;&bull;&nbsp; ${s70['annual']:,}/yr</span></div>
  <div class="assumptions-row"><span>Breakeven FRA vs 70</span><span>age {s_be_fra_70}</span></div>
  <div class="assumptions-row"><span>Spouse&#8217;s estimated life expectancy</span><span>age {spouse_le or 84}</span></div>
</div>"""

        ss_section += "</details>"
    else:
        ss_section = '<details class="accordion"><summary class="accordion-header">Social Security</summary><div class="roth-note">Social Security: not included in this analysis.</div></details>'

    win_section = _window_section(window) if window else ""
    ltcg_section = _ltcg_section(ltcg) if ltcg else ""

    has_ss_chart = ss_data is not None and twin.ss.monthly_pia > 0
    ss_canvas = '<div class="chart-box"><div class="chart-title">Social Security — Monthly Benefit by Claiming Age</div><canvas id="ss-bar-chart"></canvas></div>' if has_ss_chart else ''

    return f"""
<div class="dash-header">Retirement at {twin.person.retirement_age}</div>
<div class="chart-row{"" if has_ss_chart else " single"}">
  <div class="chart-box">
    <div class="chart-title">Portfolio Projection &mdash; 10,000 Simulations</div>
    <canvas id="mc-fan-chart"></canvas>
  </div>
  {ss_canvas}
</div>
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
{win_section}
{ltcg_section}
{cy_section}
{ss_section}
"""
