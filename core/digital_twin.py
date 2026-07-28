from dataclasses import dataclass, field


@dataclass
class Person:
    age: int
    retirement_age: int = 65


@dataclass
class Accounts:
    total_savings: float
    traditional_pct: float = 1.0  # fraction in pre-tax accounts (401k, trad IRA)

    @property
    def traditional_balance(self) -> float:
        return self.total_savings * self.traditional_pct

    @property
    def roth_balance(self) -> float:
        return self.total_savings * (1.0 - self.traditional_pct)


@dataclass
class Spending:
    annual: float


@dataclass
class Assumptions:
    stock_return: float = 0.09
    bond_return: float = 0.04
    inflation: float = 0.025
    stock_pct: float = 0.60


@dataclass
class TaxProfile:
    filing_status: str = "single"  # "single" or "married"


@dataclass
class SocialSecurity:
    monthly_pia: float = 0.0   # monthly benefit at FRA (0 = no SS / skip)
    birth_year: int = 1960     # determines FRA; approximated from age if not asked


@dataclass
class HouseholdTwin:
    person: Person
    accounts: Accounts
    spending: Spending
    assumptions: Assumptions = field(default_factory=Assumptions)
    tax_profile: TaxProfile = field(default_factory=TaxProfile)
    ss: SocialSecurity = field(default_factory=SocialSecurity)
