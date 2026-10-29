import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.telecom.environment import get_environment
from tau2.voice.caller_policy import QualifiedCallerBackend
from tau2.voice.device_controller import (
    CustomerDeviceController,
    Observations,
    account_question_only,
    agent_action_only,
)


def test_v5_line_selection_answers_known_number_without_running_a_speed_test():
    env, _ = setup()
    controller = CustomerDeviceController(
        "My phone number is 555-123-2002.", routing_focus=True
    )
    reply = controller.respond(
        [
            {
                "speaker": "service_agent",
                "content": "I've found your account. Could you please tell me which line's data is not working?",
            }
        ],
        env.get_user_tools(),
    )
    assert reply.content == "My phone number is 555-123-2002."
    assert not reply.is_tool_call()


def test_v5_passive_network_mode_question_reads_actual_network_status():
    env, _ = setup()
    controller = CustomerDeviceController("Restore my internet.", routing_focus=True)
    reply = controller.respond(
        [
            {
                "speaker": "service_agent",
                "content": "Please check your phone's network settings to see if it is set to a 2G or 3G connection.",
            }
        ],
        env.get_user_tools(),
    )
    assert [call.name for call in reply.tool_calls] == ["check_network_status"]


@pytest.mark.parametrize(
    "statement",
    [
        "I can check your account. Could you please enable mobile data?",
        "Please turn on data roaming and then check your data usage.",
        "Please enable mobile data if you have exceeded your plan limit.",
    ],
)
def test_v5_action_ownership_keeps_real_customer_instructions(statement):
    assert not account_question_only(statement)
    assert not agent_action_only(statement, routing_focus=True)


@pytest.mark.parametrize("focused", [False, True])
def test_closing_help_question_still_runs_required_acceptance_speed_test(focused):
    env, _ = setup()
    controller = CustomerDeviceController(
        "Only consider the issue resolved only when the speed test returns excellent internet speed.",
        routing_focus=focused,
    )
    response = controller.respond(
        [
            {
                "speaker": "service_agent",
                "content": "Your connection is restored. Is there anything else I can help you with?",
            }
        ],
        env.get_user_tools(),
    )
    assert [call.name for call in response.tool_calls] == ["run_speed_test"]


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


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize(
    "instruction,desired",
    [
        (
            "Please turn off Airplane Mode on your device. This will restore your cellular and data connections. Would you like me to check anything else?",
            False,
        ),
        ("Please turn Airplane Mode off on your phone.", False),
        ("Please switch off flight mode on the device.", False),
        ("Please set Airplane Mode to off on your phone.", False),
        ("Please disable Airplane Mode on your device.", False),
        ("Please turn on Airplane Mode on your device.", True),
        ("Please turn Airplane Mode on on your phone.", True),
        ("Please enable Airplane Mode on your phone.", True),
        ("Please activate flight mode on your phone.", True),
    ],
)
def test_setting_direction_excludes_the_on_in_a_device_location(
    initial, instruction, desired
):
    env, caller = setup({"airplane_mode": initial})
    response, calls = complete(
        caller, env, [UserMessage(role="user", content=instruction)]
    )
    assert calls == [
        "check_network_status",
        *(["toggle_airplane_mode"] if initial != desired else []),
    ]
    assert env.user_tools.db.device.airplane_mode is desired
    assert ("ON" if desired else "OFF") in response.content


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("data_enabled", [False, True])
def test_explanation_does_not_extend_the_switch_object_list(initial, data_enabled):
    env, caller = setup({"airplane_mode": initial, "data_enabled": data_enabled})
    _, calls = complete(
        caller,
        env,
        [
            UserMessage(
                role="user",
                content="Please turn off Airplane Mode on your device to restore mobile data.",
            )
        ],
    )
    assert calls == [
        "check_network_status",
        *(["toggle_airplane_mode"] if initial else []),
    ]
    assert env.user_tools.db.device.airplane_mode is False
    assert env.user_tools.db.device.data_enabled is data_enabled


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize(
    "instruction",
    [
        "Please turn off Wi-Fi and VPN on your device.",
        "Please turn Wi-Fi and VPN off on your device.",
        "Please disable Wi-Fi and VPN on your device.",
    ],
)
def test_coordinated_switch_objects_share_only_the_requested_direction(
    initial, instruction
):
    env, caller = setup(
        {
            "wifi_enabled": initial,
            "vpn_enabled_setting": initial,
            "vpn_connected": initial,
        }
    )
    _, calls = complete(caller, env, [UserMessage(role="user", content=instruction)])
    assert calls == [
        "check_wifi_status",
        *(["toggle_wifi"] if initial else []),
        "check_vpn_status",
        *(["disconnect_vpn"] if initial else []),
    ]
    assert env.user_tools.db.device.wifi_enabled is False
    assert env.user_tools.db.device.vpn_connected is False


@pytest.mark.parametrize("initial", [False, True])
def test_coordinated_mobile_data_and_roaming_are_both_requested(initial):
    env, caller = setup({"data_enabled": initial, "roaming_enabled": initial})
    _, calls = complete(
        caller,
        env,
        [
            UserMessage(
                role="user",
                content="Please enable mobile data and roaming on your phone.",
            )
        ],
    )
    assert calls[0] == "check_network_status"
    assert calls.count("toggle_data") == (0 if initial else 1)
    assert calls.count("toggle_roaming") == (0 if initial else 1)
    assert set(calls) <= {"check_network_status", "toggle_data", "toggle_roaming"}
    assert env.user_tools.db.device.data_enabled is True
    assert env.user_tools.db.device.roaming_enabled is True


def test_customer_claim_cannot_supply_a_device_observation():
    env, caller = setup({"airplane_mode": True})
    history = [
        AssistantMessage(role="assistant", content="I have turned airplane mode off."),
        UserMessage(role="user", content="Is airplane mode off now?"),
    ]
    response, calls = complete(caller, env, history)
    assert calls == ["check_network_status"]
    assert response.content == "Airplane Mode: ON"


@pytest.mark.parametrize("airplane_mode", [False, True])
@pytest.mark.parametrize("data_enabled", [False, True])
@pytest.mark.parametrize(
    "question",
    [
        'Since your data is enabled and you haven\'t reached your limit, I can check your network status. Could you please tell me if you are seeing any specific icons like "No Signal" or "Data Disabled" in your status bar?',
        "Please check your status bar for No Signal or Data Disabled.",
        "What icons do you see in your status bar, for example Data Disabled or Airplane Mode?",
    ],
)
def test_status_bar_request_reads_the_view_instead_of_example_settings(
    airplane_mode, data_enabled, question
):
    env, caller = setup({"airplane_mode": airplane_mode, "data_enabled": data_enabled})
    env.user_tools.simulate_network_search()
    history = [UserMessage(role="user", content="Is mobile data enabled?")]
    earlier, _ = complete(caller, env, history)
    history.extend([earlier, UserMessage(role="user", content=question)])
    before = env.user_tools.db.device.model_dump()
    reply, calls = complete(caller, env, history)
    assert calls == ["check_status_bar"]
    assert reply.content == CustomerDeviceController.status_speech(
        env.user_tools.check_status_bar()
    )
    assert (
        reply.raw_data["device_grounding"]["observations"][0]["text"]
        == env.user_tools.check_status_bar()
    )
    assert env.user_tools.db.device.model_dump() == before


@pytest.mark.parametrize(
    "question,expected",
    [
        (
            "Please check your mobile data and status bar.",
            ["check_network_status", "check_status_bar"],
        ),
        (
            "Please check your status bar and mobile data.",
            ["check_status_bar", "check_network_status"],
        ),
    ],
)
def test_explicit_coordinated_setting_and_status_reads_preserve_both(
    question, expected
):
    env, caller = setup()
    reply, calls = complete(caller, env, [UserMessage(role="user", content=question)])
    assert calls == expected
    assert (
        CustomerDeviceController.status_speech(env.user_tools.check_status_bar())
        in reply.content
    )
    assert "Mobile Data Enabled" in reply.content


@pytest.mark.parametrize(
    "raw,spoken",
    [
        (
            "Status Bar: ✈️ Airplane Mode | 🔋 80%",
            "My status bar shows Airplane Mode, battery 80 percent.",
        ),
        (
            "Status Bar: 📶⁴ Excellent | 5G | 📱 Data Enabled | 🔋 80%",
            "My status bar shows 4 cellular signal bars with Excellent strength, 5G, Data Enabled, battery 80 percent.",
        ),
        (
            "Status Bar: 📶² Poor | 2G | 📱 Data Disabled | 🔋 21%",
            "My status bar shows 2 cellular signal bars with Poor strength, 2G, Data Disabled, battery 21 percent.",
        ),
        (
            "Status Bar: ❌ No Service | 🔋 99%",
            "My status bar shows No Service, battery 99 percent.",
        ),
        ("The device check failed.", "The device check failed."),
    ],
)
def test_status_speech_preserves_visible_labels_and_values_without_emoji(raw, spoken):
    assert CustomerDeviceController.status_speech(raw) == spoken


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
        "Please turn airplane mode on or off.",
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
    assert "speed test" in response.content.lower()


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


@pytest.mark.parametrize(
    "question,second_tool,second_fact",
    [
        (
            "Please check your mobile data and network mode.",
            "check_network_mode_preference",
            "Network Mode Preference",
        ),
        (
            "Please check your mobile data and speed test.",
            "run_speed_test",
            "speed test",
        ),
    ],
)
def test_coordinated_device_checks_report_both_actual_observations(
    question, second_tool, second_fact
):
    env, caller = setup()
    history = [UserMessage(role="user", content=question)]
    reply, calls = complete(caller, env, history)
    assert calls == ["check_network_status", second_tool]
    assert "Mobile Data Enabled" in reply.content and second_fact in reply.content


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


@pytest.mark.parametrize("airplane_mode", [False, True])
@pytest.mark.parametrize(
    "question",
    [
        "Are you connected to 5G?",
        "I see. Since your data is enabled and your plan limit has not been reached, could you please check if your phone is currently using a 2G or 3G network?",
        "Please check your network settings to see if your phone is connected to a 2G or 3G network.",
    ],
)
def test_cellular_technology_question_reads_connection_not_network_preference(
    question, airplane_mode
):
    env, caller = setup(
        {"airplane_mode": airplane_mode, "network_mode_preference": "2g_only"}
    )
    env.user_tools.simulate_network_search()
    history = [UserMessage(role="user", content=question)]
    response, calls = complete(caller, env, history)
    assert calls == ["check_network_status"]
    assert (
        "Cellular Network Type: " + ("none" if airplane_mode else "2G")
        in response.content
    )
    assert env.user_tools.db.device.network_mode_preference.value == "2g_only"
    assert env.user_tools.db.device.airplane_mode is airplane_mode


def test_unspecified_technology_change_is_not_treated_as_a_connection_read():
    env, caller = setup()
    response = generate(
        caller,
        env,
        [UserMessage(role="user", content="Please disable 5G on your phone.")],
    )
    assert not response.is_tool_call() and "clarify" in response.content.lower()


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


@pytest.mark.parametrize(
    "statement",
    [
        "I'm here. I've located your account. I see that your line is active and roaming is enabled. Could you please describe what happens when you try to use your data?",
        "I see that your line is roaming enabled. Could you please tell me more about the issue you are experiencing with your mobile data?",
    ],
)
def test_describing_the_reported_problem_does_not_request_a_device_setting_read(
    statement,
):
    env, caller = setup()
    controller = caller.device_controller
    assert (
        controller.respond(
            [{"speaker": "service_agent", "content": statement}], env.get_user_tools()
        )
        is None
    )


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Is your mobile data working now?", "run_speed_test"),
        ("Could you run a speed test on your mobile data?", "run_speed_test"),
        ("Can you send MMS over mobile data now?", "can_send_mms"),
        ("Could you try your data again?", "run_speed_test"),
        ("Is your internet connection working now?", "run_speed_test"),
        ("Could you please check if your data speed is still slow?", "run_speed_test"),
        ("Is your mobile data still slow?", "run_speed_test"),
        (
            "Please try to connect to your mobile data now. Is it working correctly?",
            "run_speed_test",
        ),
        ("Could you please try to connect to the internet?", "run_speed_test"),
        (
            "Please let me know if your connection is restored now that Airplane Mode is off.",
            "run_speed_test",
        ),
        (
            "I am still here. Please let me know if your connection is working now that Airplane Mode is off.",
            "run_speed_test",
        ),
    ],
)
def test_functional_questions_test_service_rather_than_reading_enabled_setting(
    question, expected
):
    env, caller = setup({"airplane_mode": True, "data_enabled": True})
    history = [UserMessage(role="user", content=question)]
    reply = generate(caller, env, history)
    assert reply.tool_calls[0].name == expected
    execute(env, history, reply)
    report = generate(caller, env, history)
    assert "No Connection" in report.content or "cannot" in report.content.lower()
    assert env.user_tools.db.device.airplane_mode is True


def test_vpn_question_with_explanatory_connection_speed_keeps_the_requested_topic():
    env, caller = setup({"vpn_enabled_setting": False, "vpn_connected": False})
    reply, calls = complete(
        caller,
        env,
        [
            UserMessage(
                role="user",
                content="I am not using a VPN. I am asking if you have a VPN active on your phone, as that can sometimes affect your connection speed.",
            )
        ],
    )
    assert calls == ["check_vpn_status"]
    assert reply.content == "My VPN is disabled."


def test_turn_off_airplane_then_report_if_connection_restored_tests_the_service():
    env, caller = setup({"airplane_mode": True})
    _, calls = complete(
        caller,
        env,
        [
            UserMessage(
                role="user",
                content="Please turn off Airplane Mode. This will allow your phone to connect to the network. Could you please let me know if that restores your connection?",
            )
        ],
    )
    assert calls == ["check_network_status", "toggle_airplane_mode", "run_speed_test"]
    assert env.user_tools.db.device.airplane_mode is False


def test_enabling_data_for_mms_still_changes_the_requested_setting():
    env, caller = setup({"data_enabled": False})
    reply, calls = complete(
        caller,
        env,
        [
            UserMessage(
                role="user", content="Please enable mobile data so I can send MMS."
            )
        ],
    )
    assert calls == ["check_network_status", "toggle_data"]
    assert env.user_tools.db.device.data_enabled is True


@pytest.mark.parametrize(
    "question",
    [
        "Could you please tell me where you are using your mobile data?",
        "I will run a speed test on the account, could you please tell me where you are using your mobile data?",
    ],
)
def test_location_question_with_data_topic_stays_with_conversation(question):
    env, caller = setup(scenario="I am using my mobile data in France.")
    assert (
        caller.device_controller.respond(
            [
                {
                    "speaker": "service_agent",
                    "content": question,
                }
            ],
            env.get_user_tools(),
        )
        is None
    )


def test_negated_functional_test_never_runs():
    env, caller = setup()
    reply = generate(
        caller,
        env,
        [UserMessage(role="user", content="Please don't run a speed test.")],
    )
    assert not reply.is_tool_call()
    assert reply.raw_data["device_grounding"]["action"] == "clarify"


def test_new_explicit_speed_test_request_does_not_reuse_an_old_result():
    env, caller = setup({"airplane_mode": True})
    history = [UserMessage(role="user", content="Please run a speed test.")]
    report, calls = complete(caller, env, history)
    assert calls == ["run_speed_test"]
    history.extend(
        [report, UserMessage(role="user", content="Please run a speed test.")]
    )
    reply = generate(caller, env, history)
    assert reply.tool_calls and reply.tool_calls[0].name == "run_speed_test"


@pytest.mark.parametrize(
    "offer",
    [
        "Since the speed test shows no connection, I can check if your data is being restricted. Would you like me to check that?",
        "The speed test failed. Can I check that for you?",
    ],
)
def test_agent_offer_after_a_device_test_defers_to_conversation(offer):
    env, caller = setup({"airplane_mode": True})
    history = [UserMessage(role="user", content="Please run a speed test.")]
    report, calls = complete(caller, env, history)
    assert calls == ["run_speed_test"]
    history.extend([report, UserMessage(role="user", content=offer)])
    assert (
        caller.device_controller.respond(
            caller.conversation(history), env.get_user_tools()
        )
        is None
    )


def test_agent_promise_after_a_test_defers_without_a_customer_recheck():
    env, caller = setup({"airplane_mode": True})
    history = [UserMessage(role="user", content="Please run a speed test.")]
    report, _ = complete(caller, env, history)
    history.extend([report, UserMessage(role="user", content="I will check that now.")])
    assert (
        caller.device_controller.respond(
            caller.conversation(history), env.get_user_tools()
        )
        is None
    )


def test_agent_callback_offer_is_not_a_request_for_the_phone_number():
    env, caller = setup(scenario="My phone number is 555-123-2002.")
    assert (
        caller.device_controller.respond(
            [
                {
                    "speaker": "service_agent",
                    "content": "Would you like me to call your phone number?",
                }
            ],
            env.get_user_tools(),
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
        "Which line are you using for your data in France?",
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


def test_listener_keeps_waiting_during_an_incomplete_turn_audio_gap():
    from types import SimpleNamespace

    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    caller = object.__new__(VoiceStreamingUserSimulator)
    caller.wait_for_complete_agent_turns = True
    caller.wait_to_respond_threshold_self = 300
    state = SimpleNamespace(
        is_talking=False,
        input_ongoing_speech_duration=lambda: 0,
        input_from_environment=lambda: False,
        time_since_last_other_talk=20,
        input_turn_taking_buffer=[
            AssistantMessage(
                role="assistant",
                content="Could you please tell me your ",
                is_final_chunk=False,
            ),
            AssistantMessage(role="assistant", content=None, contains_speech=False),
        ],
    )
    assert caller._compute_turn_taking_action(state).action == "wait"


EXCELLENT_SPEED_GOAL = "You will consider the issue resolved only when speed test returns excellent internet speed and nothing else."


@pytest.mark.parametrize("airplane_mode", [False, True])
def test_closing_verifies_explicit_speed_goal_from_the_actual_tool(airplane_mode):
    env, caller = setup({"airplane_mode": airplane_mode}, scenario=EXCELLENT_SPEED_GOAL)
    env.user_tools.simulate_network_search()
    history = [
        UserMessage(
            role="user", content="Is there anything else I can help you with today?"
        )
    ]
    reply, calls = complete(caller, env, history)
    assert calls == ["run_speed_test"]
    assert ("Excellent internet speed" in reply.content) is (not airplane_mode)
    assert caller.device_controller.goal_verified(
        caller.conversation(history), env.get_user_tools()
    ) is (not airplane_mode)
    assert (
        reply.raw_data["device_grounding"]["trigger"]
        == "explicit_caller_acceptance_condition"
    )


def test_closing_does_not_introduce_an_unspecified_acceptance_goal():
    env, caller = setup(scenario="Fix my phone's signal strength.")
    assert (
        caller.device_controller.respond(
            [
                {
                    "speaker": "service_agent",
                    "content": "Is there anything else I can help with?",
                }
            ],
            env.get_user_tools(),
        )
        is None
    )


def test_verified_speed_is_invalidated_by_a_write_and_does_not_repair_the_device():
    env, caller = setup(scenario=EXCELLENT_SPEED_GOAL)
    history = [UserMessage(role="user", content="Please run a speed test.")]
    report, calls = complete(caller, env, history)
    assert calls == ["run_speed_test"] and "Excellent internet speed" in report.content
    assert caller.device_controller.goal_verified(
        caller.conversation(history), env.get_user_tools()
    )
    history.extend(
        [report, UserMessage(role="user", content="Please disable mobile data.")]
    )
    report, _ = complete(caller, env, history)
    assert not caller.device_controller.goal_verified(
        caller.conversation(history), env.get_user_tools()
    )
    history.extend([report, UserMessage(role="user", content="Have a wonderful day!")])
    report, calls = complete(caller, env, history)
    assert calls == ["run_speed_test"] and "No Connection" in report.content
    assert env.user_tools.db.device.data_enabled is False


@pytest.mark.parametrize(
    "quality,rate",
    [
        ("Very Poor", "0.50"),
        ("Poor", "2.00"),
        ("Fair", "15.00"),
        ("Good", "80.00"),
        ("Excellent", "275.00"),
    ],
)
def test_speed_speech_preserves_the_observed_quality_and_rate(quality, rate):
    source = f"Speed Test Result: {rate} Mbps ({quality}). Connection advice."
    assert (
        CustomerDeviceController.speed_speech(source)
        == f"The speed test reports {quality} internet speed, at {rate} megabits per second."
    )
