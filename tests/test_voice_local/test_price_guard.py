import pytest

from tau2.voice.price_guard import (
    missing_price_quote,
    pending_price_consent,
    price_followup,
)


def agent(text):
    return {"speaker": "service_agent", "content": text}


@pytest.mark.parametrize(
    "limit", ["up to $650", "at most 20 dollars", "a maximum of $25"]
)
def test_capped_request_needs_an_agent_price_quote(limit):
    assert missing_price_quote("Upgrade for " + limit, [agent("Done.")])


@pytest.mark.parametrize(
    "quote",
    [
        "The total cancellation fee is $10.",
        "The upgrade will cost $600.",
        "The total for both passengers is $600.50.",
        "The total cost for the business class upgrade and two extra bags would be $598.00.",
        "The total cost is $598.00, within your $650 budget.",
        "Your budget is $650, but the total upgrade cost is $598.00.",
        "The fee is $10, which will be deducted from the gift card balance.",
        "The fee is 10 dollars.",
        "The changes are free.",
        "There is no charge.",
    ],
)
def test_stated_price_or_free_change_is_evidence_without_inventing_a_stop(quote):
    history = [agent(quote), agent("I have updated it. Anything else?")]
    assert not missing_price_quote("Upgrade for up to $650.", history)
    assert price_followup("Upgrade for up to $650.", history) is None


@pytest.mark.parametrize(
    "statement",
    [
        "Your gift card balance is $60.",
        "Your total balance is $60.",
        "Your budget is $650.",
        "The upgrade cost is unknown. Your gift card balance is $60.",
        "Your reservation is ABC123.",
        "You mentioned the total cost is $650.",
    ],
)
def test_balances_limits_and_unknown_costs_do_not_establish_a_price(statement):
    assert missing_price_quote("Upgrade for up to $650.", [agent(statement)])


def test_customer_quote_is_not_agent_evidence():
    history = [{"speaker": "customer", "content": "The total cost is $10."}]
    assert missing_price_quote("Upgrade for up to $650.", history)


def test_quoted_offer_is_pending_but_later_completion_is_not():
    offer = agent("The total cost is $598. Would you like me to proceed?")
    scenario = "Upgrade for up to $650."
    assert pending_price_consent(scenario, [offer])
    assert not pending_price_consent(
        scenario, [offer, agent("I have upgraded the booking for $598.")]
    )
    assert not pending_price_consent("What is my balance?", [offer])
    assert not pending_price_consent(scenario, [])


def test_ordinary_balance_question_and_read_consent_are_not_price_followups():
    assert price_followup("Check my $60 balance.", [agent("May I proceed?")]) is None
    assert (
        price_followup("Upgrade for up to $650.", [agent("May I check your account?")])
        is None
    )
    assert (
        price_followup("Upgrade for up to $650.", [agent("What is your ID?")]) is None
    )


@pytest.mark.parametrize(
    "heard",
    [
        "You want business class and two bags. May I proceed with these changes?",
        "I have successfully upgraded the booking. Is there anything else I can help you with?",
    ],
)
def test_only_caller_visible_facts_trigger_a_cost_question(heard):
    response = price_followup("Upgrade for up to $650.", [agent(heard)])
    assert response and "total cost" in response and "?" in response
    assert "650" not in response
    assert "###STOP###" not in response
