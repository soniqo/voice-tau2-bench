"""Conservative price-evidence checks using only caller-visible English text."""

import re

MONEY = r"(?:\$\s*\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s*(?:dollars|USD))"
SPENDING_LIMIT = re.compile(
    r"\b(?:up to|at most|no more than|not more than|under|"
    r"limit(?: is| of)?|budget(?: is| of)?|maximum(?: of)?)\s*" + MONEY,
    re.I,
)
PRICE = re.compile(
    r"\b(?:costs?|fees?|prices?|charges?|total)\b(?:\s+\w+){0,32}\s*:?\s*" + MONEY,
    re.I,
)
FREE = re.compile(
    r"\b(?:at no (?:additional )?(?:cost|charge)|free of charge|"
    r"no (?:additional )?(?:fee|charge)|"
    r"(?:upgrade|changes?|cancellation) (?:is|are|was|were|will be) free)\b",
    re.I,
)
NOT_A_QUOTE = re.compile(
    r"\b(?:balance|budget|limit|maximum|unknown|unquoted)\b|"
    r"\byou (?:said|mentioned|are willing)\b",
    re.I,
)
REPEATED_CUSTOMER_STATEMENT = re.compile(
    r"\byou (?:said|mentioned|are willing)\b[^,.!?;]*$", re.I
)
CONSENT = re.compile(
    r"\b(?:may|can|shall|should) (?:I|we) (?:proceed|go ahead)|"
    r"\bwould you like (?:me|us) to (?:proceed|upgrade|cancel|make .*changes)",
    re.I,
)
COMPLETED = re.compile(
    r"\b(?:have|has|successfully|already)\b.*\b"
    r"(?:upgraded|updated|changed|cancelled|canceled|added|completed|charged|processed)\b",
    re.I | re.S,
)


def missing_price_quote(scenario, conversation):
    if not SPENDING_LIMIT.search(scenario):
        return False
    for row in conversation:
        if row["speaker"] != "service_agent":
            continue
        for clause in re.split(r"[!?;\n]|\.(?!\d)", row.get("content") or ""):
            for pattern in (PRICE, FREE):
                for match in pattern.finditer(clause):
                    # A budget or balance elsewhere in the sentence does not
                    # invalidate an actual quoted price.
                    if NOT_A_QUOTE.search(match.group()):
                        continue
                    if REPEATED_CUSTOMER_STATEMENT.search(clause[: match.start()]):
                        continue
                    return False
    return True


def pending_price_consent(scenario, conversation):
    """A quoted price and request for approval are still an offer, not completion."""
    if not SPENDING_LIMIT.search(scenario):
        return False
    latest = next(
        (
            row.get("content") or ""
            for row in reversed(conversation)
            if row["speaker"] == "service_agent"
        ),
        "",
    )
    return bool(CONSENT.search(latest))


def price_followup(scenario, conversation, *, rejected_completion=False):
    """Ask for evidence; never infer prices or confirm a completed task."""
    if not missing_price_quote(scenario, conversation):
        return None
    latest = next(
        (
            row.get("content") or ""
            for row in reversed(conversation)
            if row["speaker"] == "service_agent"
        ),
        "",
    )
    if rejected_completion or COMPLETED.search(latest):
        return "What is the total cost of the changes? I need to check it against my spending limit."
    if CONSENT.search(latest):
        return "What is the total cost before I approve the changes?"
    return None
