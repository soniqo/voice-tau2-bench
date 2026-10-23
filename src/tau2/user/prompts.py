"""Caller prompt assembly shared by text and voice simulators."""

from typing import Literal, Optional

from tau2.data_model.persona import PersonaConfig
from tau2.user.guidelines import finalize_system_prompt

SYSTEM_PROMPT = """
{global_user_sim_guidelines_with_persona}

<scenario>
{instructions}
</scenario>
""".strip()


def build_user_system_prompt(
    instructions: Optional[str],
    *,
    persona_config: PersonaConfig,
    mode: Literal["text", "voice"],
    guidelines: str,
) -> str:
    """Assemble caller instructions identically for runtime and offline review."""
    persona_guidelines = persona_config.to_guidelines_text(mode=mode) or ""
    if persona_guidelines:
        persona_guidelines = f"\n\n{persona_guidelines}\n"
    prompt = SYSTEM_PROMPT.format(
        global_user_sim_guidelines_with_persona=guidelines.replace(
            "<PERSONA_GUIDELINES>", persona_guidelines
        ),
        instructions=instructions,
    )
    return finalize_system_prompt(persona_config, prompt)
