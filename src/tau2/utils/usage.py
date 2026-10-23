"""Cost and token accounting without loading an inference provider."""

from loguru import logger

from tau2.data_model.message import AssistantMessage, Message, ToolMessage, UserMessage


def get_cost(messages: list[Message]) -> tuple[float | None, float | None]:
    """
    Get the (agent_cost, user_cost) of the interaction.

    Each side is computed independently: a side is None if any of its
    messages has no cost. This way an uncosted agent message (e.g. an
    audio-native provider without usage reporting) doesn't discard the
    user side's cost, and vice versa.
    """
    agent_cost: float | None = 0.0
    user_cost: float | None = 0.0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AssistantMessage):
            if message.cost is None:
                logger.warning(f"Agent message has no cost: {message.content}")
                agent_cost = None
            elif agent_cost is not None:
                agent_cost += message.cost
        elif isinstance(message, UserMessage):
            if message.cost is None:
                logger.warning(f"User message has no cost: {message.content}")
                user_cost = None
            elif user_cost is not None:
                user_cost += message.cost
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
