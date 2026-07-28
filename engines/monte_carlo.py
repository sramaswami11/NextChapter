import numpy as np

from core.digital_twin import HouseholdTwin


def run_monte_carlo(
    twin: HouseholdTwin,
    n_sims: int = 10_000,
    plan_to_age: int = 90,
    ss_monthly: float = 0.0,
    ss_start_age: float = 0.0,
) -> dict:
    years_to_retirement = max(0, twin.person.retirement_age - twin.person.age)
    years_in_retirement = plan_to_age - twin.person.retirement_age

    if years_in_retirement <= 0:
        return {"success_rate": 0.0, "median_portfolio": 0.0, "p10_portfolio": 0.0, "years_in_retirement": 0}

    a = twin.assumptions
    mu = a.stock_pct * a.stock_return + (1 - a.stock_pct) * a.bond_return
    sigma = a.stock_pct * 0.17 + (1 - a.stock_pct) * 0.06

    portfolio_at_retirement = twin.accounts.total_savings * (1 + mu) ** years_to_retirement

    rng = np.random.default_rng(42)
    annual_returns = rng.normal(mu, sigma, (n_sims, years_in_retirement))

    portfolios = np.full(n_sims, portfolio_at_retirement)
    annual_spending = twin.spending.annual
    ss_annual = ss_monthly * 12.0

    for yr in range(years_in_retirement):
        current_age = twin.person.retirement_age + yr
        real_spending = annual_spending * (1 + a.inflation) ** yr

        # SS income reduces portfolio withdrawals once claiming age is reached
        if ss_annual > 0 and ss_start_age > 0 and current_age >= ss_start_age:
            real_ss = ss_annual * (1 + a.inflation) ** yr
            net_withdrawal = max(0.0, real_spending - real_ss)
        else:
            net_withdrawal = real_spending

        portfolios = portfolios * (1 + annual_returns[:, yr]) - net_withdrawal

    success_mask = portfolios > 0
    success_rate = float(np.mean(success_mask)) * 100
    surviving = portfolios[success_mask]
    median_portfolio = float(np.median(surviving)) if len(surviving) else 0.0
    p10_portfolio = float(np.percentile(portfolios, 10))

    return {
        "success_rate": round(success_rate, 1),
        "median_portfolio": round(median_portfolio),
        "p10_portfolio": round(p10_portfolio),
        "years_in_retirement": years_in_retirement,
        "portfolio_at_retirement": round(portfolio_at_retirement),
    }
