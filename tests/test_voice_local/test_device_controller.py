import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.telecom.environment import get_environment
from tau2.voice.caller_policy import QualifiedCallerBackend
from tau2.voice.device_controller import CustomerDeviceController, Observations


class NoModel:
    def generate(self, *args, **kwargs):
        raise AssertionError("Device facts must not depend on generated model prose")


def setup(
    device=None, scenario="Restore my internet and follow the agent's instructions."
):
    env = get_environment()
    current = env.user_tools.db.device
    env.user_tools.db.device = type(current).model_validate(
        {**current.model_dump(), "airplane_mode": False, **(device or {})}
    )
    return env, QualifiedCallerBackend(NoModel(), scenario, device_grounding=True)


def generate(caller, env, history):
    return caller.generate(
        "model",
        history,
        tools=env.get_user_tools(),
        call_name="user_streaming_response",
    )


def execute(env, history, response):
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    tool = next(t for t in env.get_user_tools() if t.name == call.name)
    result = tool(**call.arguments)
    history.extend(
        [
            response,
            ToolMessage(role="tool", requestor="user", id=call.id, content=result),
        ]
    )
    return call.name, result


def complete(caller, env, history):
    calls = []
    for _ in range(10):
        response = generate(caller, env, history)
        if not response.is_tool_call():
            return response, calls
        calls.append(execute(env, history, response)[0])
    raise AssertionError("Device execution did not reach a grounded spoken answer")


@pytest.mark.parametrize(
    "enabled,connected", [(False, False), (True, False), (True, True)]
)
def test_vpn_answer_reports_the_actual_observation_for_opposite_states(
    enabled, connected
):
    env, caller = setup({"vpn_enabled_setting": enabled, "vpn_connected": connected})
    history = [UserMessage(role="user", content="Are you currently using a VPN?")]
    response, calls = complete(caller, env, history)
    assert calls == ["check_vpn_status"]
    expected = (
        "My VPN is connected."
        if connected
        else "My VPN is enabled in settings, but it is not connected."
        if enabled
        else "My VPN is disabled."
    )
    assert response.content == expected
    assert (
        response.raw_data["device_grounding"]["observations"][0]["text"]
        == history[-1].content
    )
    evidence = response.raw_data["device_grounding"]["observations"][0]
    assert evidence["tool_call_id"] == history[-1].id
    assert evidence["tool"] == "check_vpn_status"


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("desired", [False, True])
@pytest.mark.parametrize(
    "field,phrase,read,write",
    [
        (
            "airplane_mode",
            "airplane mode",
            "check_network_status",
            "toggle_airplane_mode",
        ),
        ("roaming_enabled", "data roaming", "check_network_status", "toggle_roaming"),
        ("data_enabled", "mobile data", "check_network_status", "toggle_data"),
        (
            "data_saver_mode",
            "Data Saver mode",
            "check_data_restriction_status",
            "toggle_data_saver_mode",
        ),
        ("wifi_enabled", "Wi-Fi", "check_wifi_status", "toggle_wifi"),
        (
            "wifi_calling_enabled",
            "Wi-Fi Calling",
            "check_wifi_calling_status",
            "toggle_wifi_calling",
        ),
    ],
)
def test_setting_change_observes_before_acting_and_never_reverses_a_correct_state(
    initial, desired, field, phrase, read, write
):
    env, caller = setup({field: initial})
    history = [
        UserMessage(
            role="user", content=f"Please turn {phrase} {'on' if desired else 'off'}."
        )
    ]
    response, calls = complete(caller, env, history)
    assert calls == [read, *([write] if initial != desired else [])]
    assert getattr(env.user_tools.db.device, field) == desired
    assert ("ON" if desired else "OFF") in response.content or (
        "Yes" if desired else "No"
    ) in response.content
    # Asking again reports the established result, without another toggle.
    history.extend(
        [
            response,
            UserMessage(
                role="user",
                content=f"Please turn {phrase} {'on' if desired else 'off'}.",
            ),
        ]
    )
    repeated, second_calls = complete(caller, env, history)
    assert second_calls == []
    assert getattr(env.user_tools.db.device, field) == desired
    assert repeated.raw_data["device_grounding"]["observations"]


def test_customer_claim_cannot_supply_a_device_observation():
    env, caller = setup({"airplane_mode": True})
    history = [
        AssistantMessage(role="assistant", content="I have turned airplane mode off."),
        UserMessage(role="user", content="Is airplane mode off now?"),
    ]
    response, calls = complete(caller, env, history)
    assert calls == ["check_network_status"]
    assert response.content == "Airplane Mode: ON"


def test_service_agent_private_tool_result_is_not_a_customer_observation():
    env, caller = setup({"vpn_enabled_setting": False, "vpn_connected": False})
    history = [
        ToolMessage(
            role="tool",
            requestor="assistant",
            id="agent-private",
            content="VPN is ON and connected.",
        ),
        UserMessage(role="user", content="Are you using a VPN?"),
    ]
    assert all(
        row.get("tool_call_id") != "agent-private"
        for row in caller.conversation(history)
    )
    response, calls = complete(caller, env, history)
    assert calls == ["check_vpn_status"]
    assert response.content == "My VPN is disabled."


def test_agent_first_person_statement_cannot_answer_a_customer_device_question():
    env, caller = setup({"vpn_enabled_setting": True, "vpn_connected": True})
    history = [
        UserMessage(
            role="user",
            content="I am not using a VPN. I am asking if you have a VPN active on your phone.",
        )
    ]
    response, calls = complete(caller, env, history)
    assert calls == ["check_vpn_status"]
    assert response.content == "My VPN is connected."


def test_agent_promising_its_own_check_does_not_cause_an_unsolicited_customer_action():
    env, caller = setup()
    response = generate(
        caller,
        env,
        [
            UserMessage(
                role="user",
                content="I'll check your data roaming status now. One moment.",
            )
        ],
    )
    assert not response.is_tool_call()
    assert response.raw_data["device_grounding"]["action"] == "wait_for_agent"


def test_write_invalidates_previous_vpn_observations():
    env, caller = setup({"vpn_connected": True, "vpn_enabled_setting": True})
    history = [UserMessage(role="user", content="Are you using a VPN?")]
    first, _ = complete(caller, env, history)
    assert first.content == "My VPN is connected."
    history.append(first)
    call = ToolCall(id="external-write", name="toggle_airplane_mode", arguments={})
    history.extend(
        [
            AssistantMessage(role="assistant", tool_calls=[call]),
            ToolMessage(
                role="tool",
                requestor="user",
                id=call.id,
                content=env.user_tools.toggle_airplane_mode(),
            ),
            UserMessage(role="user", content="Are you using a VPN?"),
        ]
    )
    second, calls = complete(caller, env, history)
    assert calls == ["check_vpn_status"]
    assert "not connected" in second.content or "disabled" in second.content


def test_rejected_tool_result_cannot_become_a_fact():
    env, caller = setup()
    history = [
        AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(id="failed", name="check_vpn_status", arguments={})],
        ),
        ToolMessage(
            role="tool",
            requestor="user",
            id="failed",
            content="VPN is turned OFF.",
            error=True,
            execution_status="rejected",
        ),
        UserMessage(role="user", content="Are you using a VPN?"),
    ]
    facts = Observations(caller.conversation(history), env.get_user_tools())
    assert "vpn" not in facts.facts
    response = generate(caller, env, history)
    assert response.tool_calls[0].name == "check_vpn_status"


@pytest.mark.parametrize(
    "instruction",
    [
        "Please don't turn airplane mode off yet.",
        "Do you know how to disable airplane mode?",
        "Please turn airplane mode on and off.",
        "Please change your APN settings.",
    ],
)
def test_ambiguous_or_negated_actions_require_clarification(instruction):
    env, caller = setup({"airplane_mode": True})
    response = generate(caller, env, [UserMessage(role="user", content=instruction)])
    assert not response.is_tool_call()
    assert response.raw_data["device_grounding"]["action"] == "clarify"
    assert env.user_tools.db.device.airplane_mode is True


def test_compound_airplane_cycle_executes_requested_order_once_then_tests_speed():
    env, caller = setup()
    history = [
        UserMessage(
            role="user",
            content="Please turn airplane mode on, then turn it off, then run a speed test.",
        )
    ]
    response, calls = complete(caller, env, history)
    assert calls == [
        "check_network_status",
        "toggle_airplane_mode",
        "toggle_airplane_mode",
        "run_speed_test",
    ]
    assert env.user_tools.db.device.airplane_mode is False
    assert "ON" in response.content and "OFF" in response.content
    assert "Speed" in response.content


def test_write_and_read_in_one_request_does_not_change_the_setting_only_being_checked():
    env, caller = setup({"airplane_mode": True, "vpn_connected": True})
    history = [
        UserMessage(
            role="user",
            content="Please turn off airplane mode and check whether a VPN is active.",
        )
    ]
    _, calls = complete(caller, env, history)
    assert calls == ["check_network_status", "toggle_airplane_mode", "check_vpn_status"]
    assert "disconnect_vpn" not in calls


def test_coordinated_device_checks_report_both_actual_observations():
    env, caller = setup()
    history = [
        UserMessage(
            role="user", content="Please check your mobile data and network mode."
        )
    ]
    reply, calls = complete(caller, env, history)
    assert calls == ["check_network_status", "check_network_mode_preference"]
    assert (
        "Mobile Data Enabled" in reply.content
        and "Network Mode Preference" in reply.content
    )


@pytest.mark.parametrize(
    "statement",
    [
        "Are your photos safe?",
        "Your photos might be safe.",
        "Your photos are not safe.",
    ],
)
def test_uncertain_or_negative_photo_statement_does_not_authorize_reboot(statement):
    env, caller = setup(
        scenario="I am worried rebooting might delete my photos. Ask if my photos will be safe before rebooting."
    )
    history = [
        UserMessage(role="user", content=statement + " Please reboot your phone.")
    ]
    response = generate(caller, env, history)
    assert "photos" in response.content and not response.is_tool_call()


def test_fresh_test_request_does_not_reuse_an_earlier_speed_result():
    env, caller = setup()
    history = [UserMessage(role="user", content="Please run a speed test.")]
    response, _ = complete(caller, env, history)
    history.extend(
        [response, UserMessage(role="user", content="Please run a speed test again.")]
    )
    _, calls = complete(caller, env, history)
    assert calls == ["run_speed_test"]


@pytest.mark.parametrize(
    "tool,instruction",
    [
        ("run_speed_test", "Please run a speed test."),
        ("can_send_mms", "Can you send an MMS message?"),
    ],
)
def test_vague_followup_repeats_the_actual_previous_functional_check(tool, instruction):
    env, caller = setup()
    history = [UserMessage(role="user", content=instruction)]
    report, calls = complete(caller, env, history)
    assert calls == [tool]
    history.extend([report, UserMessage(role="user", content="Does it work now?")])
    _, calls = complete(caller, env, history)
    assert calls == [tool]


def test_vague_followup_without_known_test_requires_clarification():
    env, caller = setup()
    history = [UserMessage(role="user", content="Please check your network status.")]
    report, _ = complete(caller, env, history)
    history.extend([report, UserMessage(role="user", content="Does it work now?")])
    reply = generate(caller, env, history)
    assert not reply.is_tool_call()
    assert reply.raw_data["device_grounding"]["action"] == "clarify"


def test_exact_network_mode_is_taken_only_from_the_heard_request():
    env, caller = setup({"network_mode_preference": "2g_only"})
    history = [
        UserMessage(
            role="user", content="Please set the network mode to 4g_5g_preferred."
        )
    ]
    _, calls = complete(caller, env, history)
    assert calls == ["check_network_mode_preference", "set_network_mode_preference"]
    assert env.user_tools.db.device.network_mode_preference.value == "4g_5g_preferred"


def test_cellular_technology_question_reads_connection_not_network_preference():
    env, caller = setup()
    history = [UserMessage(role="user", content="Are you connected to 5G?")]
    response, calls = complete(caller, env, history)
    assert calls == ["check_network_status"]
    assert "Cellular Network Type" in response.content


def test_permissions_are_granted_only_for_the_app_and_permissions_actually_named():
    env, caller = setup()
    history = [
        UserMessage(
            role="user",
            content="Please grant storage and SMS permissions to your phone's messaging app.",
        )
    ]
    response, calls = complete(caller, env, history)
    assert calls == ["grant_app_permission", "grant_app_permission"]
    arguments = [
        message.tool_calls[0].arguments
        for message in history
        if isinstance(message, AssistantMessage) and message.is_tool_call()
    ]
    assert arguments == [
        {"app_name": "messaging", "permission": "storage"},
        {"app_name": "messaging", "permission": "sms"},
    ]
    assert "Success" in response.content


def test_missing_app_or_permission_needs_clarification():
    env, caller = setup()
    for request in [
        "Please grant the app permissions.",
        "Please grant permissions to the messaging app.",
    ]:
        reply = generate(caller, env, [UserMessage(role="user", content=request)])
        assert not reply.is_tool_call()
        assert reply.raw_data["device_grounding"]["action"] == "clarify"


def test_apn_reset_reports_future_reboot_requirement_without_adding_a_reboot():
    env, caller = setup()
    reply, calls = complete(
        caller,
        env,
        [UserMessage(role="user", content="Please reset your APN settings.")],
    )
    assert calls == ["reset_apn_settings"]
    assert "will reset at reboot" in reply.content


def test_apn_name_argument_comes_from_the_heard_value():
    env, caller = setup()
    history = [
        UserMessage(role="user", content="Please set your APN name to internet.")
    ]
    reply, calls = complete(caller, env, history)
    assert calls == ["set_apn_settings"]
    assert history[1].tool_calls[0].arguments == {
        "apn_settings": {"apn_name": "internet"}
    }
    assert "internet" in reply.content


def test_photo_concern_is_resolved_by_agent_reassurance_before_reboot():
    env, caller = setup(
        scenario="I am worried rebooting might delete my photos. Ask if my photos will be safe before rebooting."
    )
    history = [UserMessage(role="user", content="Please reboot your phone.")]
    question = generate(caller, env, history)
    assert "photos" in question.content and not question.is_tool_call()
    history.extend(
        [
            question,
            UserMessage(
                role="user",
                content="Your photos will be safe. Please reboot your phone.",
            ),
        ]
    )
    _, calls = complete(caller, env, history)
    assert calls == ["reboot_device"]


def test_unknown_account_usage_stays_with_conversation_model():
    controller = CustomerDeviceController("My phone number is 555-123-2002.")
    for question in [
        "Have you reached your plan's data limit?",
        "Have you received a payment request?",
    ]:
        assert (
            controller.respond([{"speaker": "service_agent", "content": question}], [])
            is None
        )


def test_describing_the_reported_problem_does_not_request_a_device_setting_read():
    env, caller = setup()
    controller = caller.device_controller
    statement = "I'm here. I've located your account. I see that your line is active and roaming is enabled. Could you please describe what happens when you try to use your data?"
    assert (
        controller.respond(
            [{"speaker": "service_agent", "content": statement}], env.get_user_tools()
        )
        is None
    )


def test_role_flip_preserves_tool_failure_and_execution_status():
    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    original = ToolMessage(
        role="tool",
        requestor="user",
        id="rejected",
        content="VPN is OFF",
        error=True,
        execution_status="rejected",
    )
    caller = object.__new__(VoiceStreamingUserSimulator)
    flipped = caller._flip_roles_for_llm([original])[0]
    assert flipped.error and flipped.execution_status == "rejected"
    assert flipped.requestor == "user" and flipped.id == original.id


def test_role_flip_preserves_incomplete_agent_speech_boundary():
    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    caller = object.__new__(VoiceStreamingUserSimulator)
    partial = AssistantMessage(
        role="assistant",
        content="Which phone ",
        is_final_chunk=False,
        utterance_ids=["agent-turn"],
        chunk_id=3,
    )
    flipped = caller._flip_roles_for_llm([partial])[0]
    assert flipped.is_final_chunk is False
    assert flipped.utterance_ids == ["agent-turn"]
    assert flipped.chunk_id == 3


def test_partial_phone_question_waits_without_inventing_a_device_request():
    env, caller = setup(scenario="My phone number is 555-123-2002.")
    history = [
        UserMessage(
            role="user",
            content="Could you please tell me which phone ",
            is_final_chunk=False,
        )
    ]
    reply = generate(caller, env, history)
    assert not reply.is_tool_call() and not (reply.content or "").strip()
    assert reply.raw_data["device_grounding"]["action"] == "wait_for_agent_turn"


@pytest.mark.parametrize(
    "question",
    [
        "Which phone number is experiencing the data issues?",
        "Which of your phone numbers is experiencing this issue?",
        "Which of your lines is experiencing the data issues?",
    ],
)
def test_known_phone_question_does_not_turn_into_a_device_status_check(question):
    env, caller = setup(
        scenario="Known info:\nYou are John Smith with phone number 555-123-2002.\nTask instructions:\nFollow the agent instructions."
    )
    reply = generate(caller, env, [UserMessage(role="user", content=question)])
    assert (
        not reply.is_tool_call() and reply.content == "My phone number is 555-123-2002."
    )
    assert reply.raw_data["device_grounding"]["action"] == "known_phone"


def test_merged_speech_retains_the_last_chunk_boundary_through_role_flipping():
    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    chunks = [
        AssistantMessage(
            role="assistant",
            content="Which phone ",
            is_final_chunk=False,
            utterance_ids=["turn"],
            chunk_id=0,
        ),
        AssistantMessage(
            role="assistant",
            content="number",
            is_final_chunk=False,
            utterance_ids=["turn"],
            chunk_id=1,
        ),
    ]
    caller = object.__new__(VoiceStreamingUserSimulator)
    merged = AssistantMessage.merge_chunks(chunks)
    flipped = caller._flip_roles_for_llm([merged])[0]
    assert flipped.content == "Which phone number"
    assert flipped.is_final_chunk is False and flipped.chunk_id is None
    chunks.append(
        AssistantMessage(
            role="assistant",
            content="? ",
            is_final_chunk=True,
            utterance_ids=["turn"],
            chunk_id=2,
        )
    )
    final = caller._flip_roles_for_llm([AssistantMessage.merge_chunks(chunks)])[0]
    assert final.is_final_chunk is True and final.chunk_id is None


def test_complete_question_during_speech_can_still_trigger_a_device_check():
    env, caller = setup()
    reply = generate(
        caller,
        env,
        [
            UserMessage(
                role="user",
                content="Is airplane mode on? While you check,",
                is_final_chunk=False,
            )
        ],
    )
    assert reply.tool_calls[0].name == "check_network_status"


@pytest.mark.parametrize(
    "silence,expected", [(0, ""), (14.95, ""), (15, "Are you still there?")]
)
def test_incomplete_turn_waits_but_retains_the_silence_check_in(silence, expected):
    env, caller = setup()
    reply = caller.generate(
        "model",
        [
            UserMessage(
                role="user",
                content="Could you please tell me which phone ",
                is_final_chunk=False,
            )
        ],
        tools=env.get_user_tools(),
        call_name="user_streaming_response",
        agent_silence_seconds=silence,
        caller_idle_seconds=15,
    )
    assert reply.content == expected and not reply.is_tool_call()


def test_waiting_after_an_answer_does_not_repeat_the_phone_number():
    env, caller = setup(scenario="My phone number is 555-123-9876.")
    history = [
        UserMessage(role="user", content="What is your phone number?"),
        AssistantMessage(role="assistant", content="My phone number is 555-123-9876."),
    ]
    reply = caller.generate(
        "model",
        history,
        tools=env.get_user_tools(),
        call_name="user_streaming_response",
        agent_silence_seconds=15,
    )
    assert reply.content == "Are you still there?" and not reply.is_tool_call()


def test_phone_answer_uses_only_known_info_and_admits_missing_information():
    env, caller = setup(
        scenario="Known info:\nYou don't know your phone number.\nTask instructions:\nAsk the agent to call 555-000-9999 after fixing your internet."
    )
    reply = generate(
        caller,
        env,
        [UserMessage(role="user", content="Which phone number is affected?")],
    )
    assert reply.content == "I don't know that phone number."
    assert not reply.is_tool_call()


def test_listener_waits_for_a_question_without_querying_a_model():
    from types import SimpleNamespace

    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    caller = object.__new__(VoiceStreamingUserSimulator)
    caller.wait_for_complete_agent_turns = True
    state = SimpleNamespace(
        is_talking=False,
        input_ongoing_speech_duration=lambda: 1,
        input_from_environment=lambda: False,
        input_turn_taking_buffer=[
            AssistantMessage(
                role="assistant",
                content="Hello, thank you for calling. Could you",
                is_final_chunk=False,
            )
        ],
    )
    action = caller._compute_turn_taking_action(state)
    assert action.action == "wait"
