"""Regional scenario dispatch for the three retained voice domains."""

from tau2.multilingual.regional import (
    PublishedRegionalBundle,
    ScenarioBundle,
    canonical_scenario,
    runtime_rules_digest,
    scenario_environment_kwargs,
)

RegionalBundle = PublishedRegionalBundle
Scenario = PublishedRegionalBundle | ScenarioBundle


def bundle_model(domain: str):
    if domain not in {"airline", "retail", "telecom"}:
        raise ValueError(f"Unsupported voice domain: {domain}")
    return PublishedRegionalBundle


__all__ = [
    "RegionalBundle",
    "Scenario",
    "bundle_model",
    "canonical_scenario",
    "runtime_rules_digest",
    "scenario_environment_kwargs",
]
