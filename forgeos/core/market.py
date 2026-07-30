"""Capacity Market — subscription quota priced as expiring inventory.

The insight this module exists for: **unused quota at reset is a total loss.**
A weekly allowance is not a budget you should minimise, it is inventory with an
expiry date. Treating it like cash leads to the worst possible outcome — paying an
API for work a subscription would have covered, while the subscription expires
unspent.

So each resource gets a shadow price that moves in BOTH directions:

    positive  demand exceeds remaining allowance -> scarce, conserve it
    near zero supply roughly matches demand      -> use normally
    negative  allowance will likely expire unused -> SPEND IT FIRST

A negative price is the part conventional cost-minimising routers cannot express,
and it is where most of the value is: it turns "don't spend" into "spend this one
before midnight, and save the metered provider for tomorrow."

Every figure carries whether it was measured or estimated. Providers differ wildly
in what telemetry they expose, and a confident number derived from a guess is how
an optimiser quietly makes bad decisions on bad data.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ..contracts import now

# How strongly an about-to-expire surplus discounts a resource. Higher means more
# aggressive use-it-or-lose-it behaviour.
SURPLUS_WEIGHT = 0.8

# How much a fully-exhausted resource costs relative to its alternative. Must be
# > 1: a subscription you are about to run out of is worth MORE than the API you
# would fall back to, because running out costs you the fallback price on every
# later task too. At <= 1 the scarcity term can never outrank the alternative and
# the router keeps draining the resource it should be protecting.
SCARCITY_WEIGHT = 2.0

# Urgency ramps up inside this window before reset. Outside it, a surplus is not
# yet urgent because demand may still materialise.
URGENCY_WINDOW_SECONDS = 6 * 3600

# Below this confidence, telemetry is too weak to justify aggressive pricing and
# the market falls back to neutral. Acting decisively on a guess is worse than
# acting normally.
MIN_CONFIDENCE_TO_PRICE = 0.4


class TelemetrySource(str, Enum):
    """Where a reading came from. Ordered roughly by trustworthiness."""

    OFFICIAL_API = "official_api"
    OFFICIAL_CLI = "official_cli"
    RESPONSE_HEADER = "official_response_header"
    PROVIDER_DASHBOARD = "provider_reported_dashboard"
    LOCAL_USAGE_LOG = "local_usage_log"
    USER_ENTERED = "user_entered"
    ESTIMATED = "statistically_estimated"

    @property
    def confidence(self) -> float:
        return {
            TelemetrySource.OFFICIAL_API: 0.99,
            TelemetrySource.OFFICIAL_CLI: 0.95,
            TelemetrySource.RESPONSE_HEADER: 0.9,
            TelemetrySource.PROVIDER_DASHBOARD: 0.8,
            TelemetrySource.LOCAL_USAGE_LOG: 0.6,
            TelemetrySource.USER_ENTERED: 0.5,
            TelemetrySource.ESTIMATED: 0.3,
        }[self]

    @property
    def is_measured(self) -> bool:
        """Whether this reading may be presented as a fact rather than an estimate."""
        return self in (
            TelemetrySource.OFFICIAL_API,
            TelemetrySource.OFFICIAL_CLI,
            TelemetrySource.RESPONSE_HEADER,
        )


class Entitlement(BaseModel):
    """One resource's allowance, with the provenance of every claim about it."""

    resource_id: str
    remaining: float = Field(ge=0.0, le=1.0, description="fraction of the window left")
    resets_at: float | None = None
    source: TelemetrySource = TelemetrySource.ESTIMATED
    measured_at: float = Field(default_factory=now)
    error_bound: float = 0.0
    # Marginal cash cost per unit of work. Zero for a flat subscription — which is
    # exactly why its quota needs a shadow price: cash cost alone says "free", and
    # "free" tells a router nothing about scarcity.
    cash_cost: float = 0.0
    unit: str = "provider-window-fraction"

    @property
    def confidence(self) -> float:
        return self.source.confidence

    @property
    def is_measured(self) -> bool:
        return self.source.is_measured

    def render(self) -> str:
        pct = self.remaining * 100
        if self.is_measured:
            return f"{pct:.0f}% measured"
        bound = f" ±{self.error_bound * 100:.0f}%" if self.error_bound else ""
        return f"{pct:.0f}% estimated{bound}"

    def seconds_until_reset(self, at: float | None = None) -> float | None:
        if self.resets_at is None:
            return None
        return max(0.0, self.resets_at - (at or now()))

    def urgency(self, at: float | None = None) -> float:
        """0 far from reset, approaching 1 as it nears. Drives surplus discounting."""
        secs = self.seconds_until_reset(at)
        if secs is None:
            return 0.0
        if secs >= URGENCY_WINDOW_SECONDS:
            return 0.0
        return 1.0 - (secs / URGENCY_WINDOW_SECONDS)


class Forecast(BaseModel):
    """Expected demand on a resource before its reset."""

    resource_id: str
    expected_demand: float = Field(ge=0.0, description="fraction of the window")
    confidence: float = 0.5
    basis: str = ""


class ShadowPrice(BaseModel):
    """What a unit of this resource is really worth right now."""

    resource_id: str
    price: float
    scarcity: float
    surplus: float
    urgency: float
    confidence: float
    advice: str

    @property
    def scarce(self) -> bool:
        return self.price > 0.05

    @property
    def expiring_unused(self) -> bool:
        return self.price < -0.05

    def render(self) -> str:
        return f"{self.resource_id}: {self.price:+.2f} — {self.advice}"


class CapacityMarket:
    def __init__(self, *, surplus_weight: float = SURPLUS_WEIGHT,
                 scarcity_weight: float = SCARCITY_WEIGHT):
        self.surplus_weight = surplus_weight
        self.scarcity_weight = scarcity_weight
        self._entitlements: dict[str, Entitlement] = {}
        self._forecasts: dict[str, Forecast] = {}

    # ------------------------------------------------------------ inventory

    def record(self, ent: Entitlement) -> None:
        self._entitlements[ent.resource_id] = ent

    def forecast(self, fc: Forecast) -> None:
        self._forecasts[fc.resource_id] = fc

    def entitlement(self, resource_id: str) -> Entitlement | None:
        return self._entitlements.get(resource_id)

    def resources(self) -> list[str]:
        return sorted(self._entitlements)

    # --------------------------------------------------------------- pricing

    def alternative_cost(self, exclude: str | None = None) -> float:
        """What the cheapest PAID fallback costs.

        Deliberately ignores other zero-cost resources. Taking a plain minimum
        across everything meant one free provider drove the alternative cost to
        zero, and since the scarcity term is scaled by it, that silently disabled
        pricing for every resource in the market.

        Conserving is only worth what you would otherwise have to pay, so the
        relevant comparison is the cheapest resource that actually charges.
        """
        paid = [e.cash_cost for rid, e in self._entitlements.items()
                if rid != exclude and e.cash_cost > 0]
        return min(paid) if paid else 0.0

    def shadow_price(self, resource_id: str, at: float | None = None) -> ShadowPrice:
        """Price this resource's remaining allowance.

            price = alt * scarcity_weight * scarcity     (conserve — relative)
                  - surplus_weight * surplus * urgency   (spend — absolute)

        The two terms are scaled DIFFERENTLY on purpose, and getting this wrong was
        the original bug:

        - **Conserving** is only worth what you would otherwise pay, so scarcity
          scales by the cheapest paid alternative.
        - **Spending an expiring allowance** is worth doing regardless of what the
          alternatives cost, because the allowance evaporates either way. Scaling
          this term by `alt` meant a market containing any free provider priced
          every surplus at exactly zero, hiding the loss completely.
        """
        ent = self._entitlements.get(resource_id)
        if ent is None:
            return ShadowPrice(resource_id=resource_id, price=0.0, scarcity=0.0,
                               surplus=0.0, urgency=0.0, confidence=0.0,
                               advice="unknown resource — priced neutral")

        fc = self._forecasts.get(resource_id)
        demand = fc.expected_demand if fc else ent.remaining  # no forecast: assume balanced
        q = max(ent.remaining, 1e-6)
        d = max(demand, 0.0)

        scarcity = max(0.0, d - ent.remaining) / max(d, q, 1e-6)
        surplus = max(0.0, ent.remaining - d) / q
        urgency = ent.urgency(at)

        confidence = min(ent.confidence, fc.confidence if fc else ent.confidence)

        # Weak telemetry must not drive confident action. Neutral is the honest
        # position when we do not really know how much is left.
        if confidence < MIN_CONFIDENCE_TO_PRICE:
            return ShadowPrice(
                resource_id=resource_id, price=0.0, scarcity=scarcity, surplus=surplus,
                urgency=urgency, confidence=confidence,
                advice=f"telemetry confidence {confidence:.2f} too low to price — using normally",
            )

        alt = self.alternative_cost(exclude=resource_id)
        price = (alt * self.scarcity_weight * scarcity) - (
            self.surplus_weight * surplus * urgency
        )

        if price > 0.05:
            advice = "scarce — reserve for high-value work"
        elif price < -0.05:
            advice = "expiring unused — spend this FIRST"
        else:
            advice = "balanced — use normally"

        return ShadowPrice(resource_id=resource_id, price=round(price, 4),
                           scarcity=round(scarcity, 4), surplus=round(surplus, 4),
                           urgency=round(urgency, 4), confidence=round(confidence, 3),
                           advice=advice)

    def prices(self, at: float | None = None) -> list[ShadowPrice]:
        return [self.shadow_price(r, at) for r in self.resources()]

    # ---------------------------------------------------------- allocation

    def effective_cost(self, resource_id: str, at: float | None = None) -> float:
        """Cash cost plus the opportunity cost of the quota consumed.

        A free-but-scarce subscription can be MORE expensive than a cheap API call,
        and an expiring subscription can be cheaper than free. Cash price alone
        cannot express either.
        """
        ent = self._entitlements.get(resource_id)
        if ent is None:
            return 0.0
        return ent.cash_cost + self.shadow_price(resource_id, at).price

    def rank(self, candidates: list[str], at: float | None = None) -> list[tuple[str, float]]:
        """Candidates ordered by true effective cost, cheapest first."""
        priced = [(c, self.effective_cost(c, at)) for c in candidates
                  if c in self._entitlements]
        priced.sort(key=lambda t: t[1])
        return priced

    def prefer(self, candidates: list[str], at: float | None = None) -> str | None:
        ranked = self.rank(candidates, at)
        return ranked[0][0] if ranked else None

    def use_it_or_lose_it(self, at: float | None = None) -> list[ShadowPrice]:
        """Resources whose allowance is about to expire unspent, most urgent first."""
        expiring = [p for p in self.prices(at) if p.expiring_unused]
        expiring.sort(key=lambda p: p.price)
        return expiring

    def report(self, at: float | None = None) -> str:
        lines = ["resource                 remaining        price   advice",
                 "-" * 78]
        for rid in self.resources():
            ent = self._entitlements[rid]
            sp = self.shadow_price(rid, at)
            lines.append(f"{rid:<24} {ent.render():<16} {sp.price:+7.3f}   {sp.advice}")
        return "\n".join(lines)


__all__ = [
    "SCARCITY_WEIGHT",
    "CapacityMarket",
    "Entitlement",
    "Forecast",
    "MIN_CONFIDENCE_TO_PRICE",
    "SURPLUS_WEIGHT",
    "ShadowPrice",
    "TelemetrySource",
    "URGENCY_WINDOW_SECONDS",
]
