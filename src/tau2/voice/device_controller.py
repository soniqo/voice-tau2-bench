"""Customer device execution from heard requests and actual tool observations.

This module never reads a device database, task answers, or service-agent tools.
It returns calls for the ordinary orchestrator, and speaks their results without
asking a language model to reconstruct device facts.
"""

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field

from tau2.data_model.message import AssistantMessage, ToolCall


def normalized(text):
    return " ".join((text or "").replace("’", "'").split())


@dataclass(frozen=True)
class Setting:
    name: str
    topic: str
    read: str
    on: str
    off: str
    observation: str


SETTINGS = (
    Setting(
        "airplane",
        r"\b(?:airplane|flight) mode\b",
        "check_network_status",
        "toggle_airplane_mode",
        "toggle_airplane_mode",
        r"Airplane Mode\s*(?::|is now)\s*(ON|OFF)",
    ),
    Setting(
        "roaming",
        r"\b(?:data )?roaming\b",
        "check_network_status",
        "toggle_roaming",
        "toggle_roaming",
        r"Data Roaming\s*(?:Enabled:|is now)\s*(Yes|No|ON|OFF)",
    ),
    Setting(
        "data_saver",
        r"\b(?:data saver|data[- ]sav(?:ing|er))\b",
        "check_data_restriction_status",
        "toggle_data_saver_mode",
        "toggle_data_saver_mode",
        r"Data Saver\s*(?:Mode)?\s*(?:is now|mode is|is|:)\s*(ON|OFF)",
    ),
    Setting(
        "wifi_calling",
        r"\bwi[- ]?fi calling\b",
        "check_wifi_calling_status",
        "toggle_wifi_calling",
        "toggle_wifi_calling",
        r"Wi-Fi Calling\s*(?:is currently turned|is now|:)\s*(ON|OFF)",
    ),
    Setting(
        "wifi",
        r"\bwi[- ]?fi\b(?! calling)",
        "check_wifi_status",
        "toggle_wifi",
        "toggle_wifi",
        r"Wi-Fi\s*(?:Radio:|is turned|is now|is)\s*(ON|OFF)",
    ),
    Setting(
        "data",
        r"\b(?:mobile data(?!\s+(?:roaming|saver))|cellular data|data connection|data(?!\s+(?:roaming|saver|limit|usage|plan)))\b",
        "check_network_status",
        "toggle_data",
        "toggle_data",
        r"Mobile Data\s*(?:Enabled:|is now)\s*(Yes|No|ON|OFF)",
    ),
    Setting(
        "vpn",
        r"\b(?:vpn|virtual private network)\b",
        "check_vpn_status",
        "connect_vpn",
        "disconnect_vpn",
        r"VPN\s*(?:is turned|is|connected)\s*(ON|OFF|successfully)",
    ),
)

READS = {
    "speed": (
        r"\b(?:speed test|download speed|internet speed|connection speed)\b",
        "run_speed_test",
    ),
    "status": (
        r"\b(?:status bar|signal bars?|icons? on .{0,15}screen)\b",
        "check_status_bar",
    ),
    "mode": (
        r"\b(?:network mode|network preference|preferred network|(?:switch|set|select|change).{0,20}(?:[2345]g|lte))\b",
        "check_network_mode_preference",
    ),
    "network": (
        r"\b(?:network status|cellular (?:connection|signal|network)|signal(?: strength)?|network connection|mobile network)\b",
        "check_network_status",
    ),
    "sim": (r"\bsim(?: card)?\b", "check_sim_status"),
    "apn": (r"\b(?:apn|access point|mmsc)\b", "check_apn_settings"),
    "mms": (r"\b(?:mms|picture messag\w*|multimedia messag\w*)\b", "can_send_mms"),
    "apps": (r"\b(?:installed apps?|apps? installed)\b", "check_installed_apps"),
}

DEVICE_HINT = re.compile(
    r"\b(?:phone|device|setting|vpn|apn|mmsc|airplane|roaming|wi[- ]?fi|"
    r"mobile data|data saver|signal|sim card|reboot|restart|mms|permissions?)\b",
    re.I,
)
REQUEST = re.compile(
    r"\b(?:(?:could|can|would|will) you|please|let's|asking (?:if|whether)|you (?:need to|should|must))\b|^(?:is|are|what|which|does|do|have|did|can|check|"
    r"turn|switch|enable|disable|connect|disconnect|run|try|reboot|restart|reset|set|reseat|grant|allow)\b",
    re.I,
)
ACTION = re.compile(
    r"\b(?:turn|switch|enable|disable|activate|deactivate|connect|disconnect|"
    r"toggle|set|change|reset|reboot|restart|reseat|reinsert|remove|grant|allow)\b",
    re.I,
)
CONFIRMATION = re.compile(
    r"\b(?:have|did) you\b|\b(?:is|are) .{0,60}(?:on|off|enabled|disabled)\b", re.I
)
NEGATED_ACTION = re.compile(
    r"\b(?:don't|do not|never|not yet|wait before|avoid)\b", re.I
)


@dataclass(frozen=True)
class Step:
    topic: str
    read: str
    write: str | None = None
    target: bool | str | None = None
    arguments: dict = field(default_factory=dict)
    fresh: bool = False


@dataclass(frozen=True)
class Observation:
    value: bool | str
    text: str
    source_id: str
    tool: str | None

    def evidence(self):
        return {
            "value": self.value,
            "text": self.text,
            "tool_call_id": self.source_id,
            "tool": self.tool,
        }


class Observations:
    """Reconstruct facts from results; every write invalidates prior device facts."""

    def __init__(self, conversation, tools):
        self.facts = {}
        self.results = {}
        self.pending = {}
        calls = {}
        declared = {tool.name: tool for tool in tools}
        for row in conversation:
            if row["speaker"] == "customer":
                for call in row.get("tool_calls", []):
                    calls[call["id"]] = call
                    self.pending[call["id"]] = call
                    tool = declared.get(call["name"])
                    if tool and getattr(tool._func, "__mutates_state__", True):
                        self.facts.clear()
            if row["speaker"] != "customer_tool":
                continue
            source_id = row.get("tool_call_id")
            call = calls.get(source_id)
            self.pending.pop(source_id, None)
            self.results[source_id] = row
            if (
                row.get("error")
                or row.get("execution_status", "executed") != "executed"
            ):
                continue
            text = row.get("content") or ""
            tool_name = call["name"] if call else None
            for setting in SETTINGS:
                match = re.search(setting.observation, text, re.I)
                if match:
                    value = match.group(1).lower() in {"on", "yes", "successfully"}
                    self.facts[setting.name] = Observation(
                        value,
                        text.splitlines()[0]
                        if setting.name == "vpn"
                        else match.group(0),
                        source_id,
                        tool_name,
                    )
            # Disconnected and enabled-but-disconnected are distinct reports,
            # both proving there is no active VPN connection.
            if re.search(
                r"VPN (?:disconnected successfully|is turned ON in settings, but currently not connected)",
                text,
                re.I,
            ):
                self.facts["vpn"] = Observation(
                    False, text.splitlines()[0], source_id, tool_name
                )
            if tool_name:
                key = self.key(tool_name, call["arguments"])
                self.facts[key] = Observation(text, text, source_id, tool_name)
            mode = re.search(
                r"(?:Network Mode Preference:|Preferred Network Mode set to:)\s*(\w+)",
                text,
                re.I,
            )
            if mode:
                self.facts["mode"] = Observation(
                    mode.group(1), mode.group(0), source_id, tool_name
                )

    @staticmethod
    def key(tool, arguments):
        return tool + ":" + json.dumps(arguments, sort_keys=True)


def clauses(text):
    # Keep separate requested steps and distinguish the agent's own work from
    # instructions to the customer. A carried topic resolves "then turn it off".
    return re.split(
        r"(?<=[.!?;])\s+|\s*,?\s+and then\s+|\s*,?\s+then\s+|"
        r"\s+(?:and|but)\s+(?=(?:please\s+)?(?:check|turn|switch|enable|disable|"
        r"run|test|reboot|restart|reset|set|grant|disconnect|connect|do not|don't)\b)",
        normalized(text),
        flags=re.I,
    )


def requested_steps(text):
    steps = []
    ambiguous = False
    previous = None
    for clause in clauses(text):
        if not REQUEST.search(clause):
            continue
        if re.search(
            r"\b(?:would you like me|shall i|can i|i'll|i will|let me)\b", clause, re.I
        ) and not re.search(r"\b(?:could|can|would|will) you\b", clause, re.I):
            continue
        if re.search(
            r"\b(?:phone number|account number|data limit|data plan|plan usage|data usage|billing)\b",
            clause,
            re.I,
        ) and not re.search(r"\b(?:vpn|airplane|data saver|wi[- ]?fi)\b", clause, re.I):
            continue
        if NEGATED_ACTION.search(clause) and ACTION.search(clause):
            ambiguous = True
            continue
        if re.search(r"\b(?:know how|explain how|understand how)\b", clause, re.I):
            ambiguous = True
            continue
        if re.search(r"\b(?:app permissions?|permissions?|app status)\b", clause, re.I):
            app = re.search(
                r"\b([\w-]+) app\b|\bapp (?:called|named) ['\"]?([\w-]+)", clause, re.I
            )
            if not app or (app.group(1) or app.group(2)).lower() in {
                "the",
                "your",
                "my",
                "an",
                "this",
            }:
                ambiguous = True
                continue
            name = (app.group(1) or app.group(2)).lower()
            if re.search(r"\b(?:grant|allow|give|enable)\b", clause, re.I):
                permission_names = r"(?:sms|storage|phone|network)"
                scope = re.search(
                    rf"\b({permission_names}(?:\s*(?:,|and)\s*{permission_names})*)\s+permissions?\b",
                    clause,
                    re.I,
                ) or re.search(
                    rf"\bpermissions?\s+(?:for|to)\s+({permission_names}(?:\s*(?:,|and)\s*{permission_names})*)\b",
                    clause,
                    re.I,
                )
                permissions = (
                    re.findall(permission_names, scope.group(1), re.I) if scope else []
                )
                if not permissions:
                    ambiguous = True
                    continue
                for permission in dict.fromkeys(value.lower() for value in permissions):
                    steps.append(
                        Step(
                            "permission:" + name + ":" + permission,
                            "",
                            "grant_app_permission",
                            arguments={"app_name": name, "permission": permission},
                        )
                    )
            elif ACTION.search(clause):
                ambiguous = True
            else:
                read = (
                    "check_app_permissions"
                    if "permission" in clause.lower()
                    else "check_app_status"
                )
                steps.append(Step("app:" + name, read, arguments={"app_name": name}))
            continue
        fresh = bool(
            re.search(r"\b(?:again|recheck|another|new test|once more)\b", clause, re.I)
        )
        found = [
            (match.start(), setting)
            for setting in SETTINGS
            if (match := re.search(setting.topic, clause, re.I))
        ]
        if (
            not found
            and previous
            and re.search(r"\b(?:it|that setting)\b", clause, re.I)
        ):
            found = [(0, previous)]
        if found:
            for _, setting in sorted(found, key=lambda item: item[0]):
                previous = setting
                action = ACTION.search(clause) and not CONFIRMATION.search(clause)
                target = None
                if action:
                    on = bool(
                        re.search(
                            r"\b(?:enable|activate|connect)\b|\b(?:turn|switch|set)\b.{0,45}\bon\b",
                            clause,
                            re.I,
                        )
                    )
                    off = bool(
                        re.search(
                            r"\b(?:disable|deactivate|disconnect)\b|\b(?:turn|switch|set)\b.{0,45}\boff\b",
                            clause,
                            re.I,
                        )
                    )
                    if on == off:
                        ambiguous = True
                        continue
                    target = on
                steps.append(
                    Step(
                        setting.name,
                        setting.read,
                        (setting.on if target else setting.off)
                        if target is not None
                        else None,
                        target,
                        fresh=fresh,
                    )
                )
            # A coordinated read can combine a setting with another observation
            # ("check mobile data and network mode"). Keep both requested reads.
            if not ACTION.search(clause):
                for topic, (pattern, read) in READS.items():
                    match = re.search(pattern, clause, re.I)
                    if match and re.search(
                        r"(?:\band|,)\s+(?:(?:your|the)\s+)?$",
                        clause[: match.start()],
                        re.I,
                    ):
                        steps.append(
                            Step(topic, read, fresh=fresh or topic in {"speed", "mms"})
                        )
            continue
        if re.search(r"\b(?:reboot|restart)\b", clause, re.I):
            steps.append(Step("reboot", "", "reboot_device"))
            continue
        matched = False
        for topic, (pattern, read) in READS.items():
            if not re.search(pattern, clause, re.I):
                continue
            matched = True
            write = None
            arguments = {}
            if topic == "sim" and re.search(
                r"\b(?:reseat|reinsert|remove)\b", clause, re.I
            ):
                write = "reseat_sim_card"
            elif topic == "apn" and re.search(r"\breset\b", clause, re.I):
                write = "reset_apn_settings"
            elif topic == "apn" and re.search(r"\bset\b", clause, re.I):
                name = re.search(
                    r"\bapn(?: name)?\s+to\s+['\"]?([\w.-]+)", clause, re.I
                )
                if not name or re.search(r"\bmmsc\b", clause, re.I):
                    ambiguous = True
                    continue
                write = "set_apn_settings"
                arguments = {
                    "apn_settings": {"apn_name": name.group(1).lower().rstrip(".")}
                }
            elif topic == "mode" and ACTION.search(clause):
                mode = re.search(
                    r"\b(?:4g_5g_preferred|4g_only|3g_only|2g_only)\b", clause, re.I
                )
                if mode:
                    selected = mode.group(0).lower()
                elif re.search(
                    r"\b(?:4g[/ -]5g|4g and 5g|5g preferred)\b", clause, re.I
                ):
                    selected = "4g_5g_preferred"
                else:
                    only = re.search(r"\b([234])g\s+only\b", clause, re.I)
                    selected = only.group(1) + "g_only" if only else None
                if selected is None:
                    ambiguous = True
                    continue
                steps.append(
                    Step(
                        topic,
                        read,
                        "set_network_mode_preference",
                        selected,
                        {"mode": selected},
                    )
                )
                continue
            elif ACTION.search(clause):
                # Parameterized changes need exact heard values, not guessed
                # defaults or values read from private task state.
                ambiguous = True
                continue
            steps.append(
                Step(
                    topic,
                    read,
                    write,
                    arguments=arguments,
                    fresh=fresh or topic in {"speed", "mms"},
                )
            )
        if not matched and DEVICE_HINT.search(clause):
            ambiguous = True
        elif not matched and re.search(r"\b(?:[2345]g|lte)\b", clause, re.I):
            steps.append(Step("network", "check_network_status"))
    return steps, ambiguous


class CustomerDeviceController:
    def __init__(self, scenario):
        self.scenario = scenario

    def respond(self, conversation, tools, system_text=""):
        heard = [
            (i, row["content"])
            for i, row in enumerate(conversation)
            if row["speaker"] == "service_agent" and (row["content"] or "").strip()
        ]
        if not heard:
            return None
        position, statement = heard[-1]
        steps, ambiguous = requested_steps(statement)
        if (
            not steps
            and not ambiguous
            and re.search(
                r"\b(?:does|is)\s+(?:it|that)\s+(?:work|working|better|fixed)\b|"
                r"\b(?:try|test|check)\s+(?:it|that|again)\b|\bhow fast\b",
                statement,
                re.I,
            )
            and not re.search(
                r"\b(?:payment|account|bill|order|booking)\b", statement, re.I
            )
        ):
            previous_test = next(
                (
                    call["name"]
                    for row in reversed(conversation)
                    if row["speaker"] == "customer"
                    for call in reversed(row.get("tool_calls", []))
                    if call["name"] in {"run_speed_test", "can_send_mms"}
                ),
                None,
            )
            if previous_test:
                steps = [
                    Step(
                        "mms" if previous_test == "can_send_mms" else "speed",
                        previous_test,
                        fresh=True,
                    )
                ]
            elif any(row["speaker"] == "customer_tool" for row in conversation):
                ambiguous = True
        if not steps and not ambiguous:
            if DEVICE_HINT.search(statement) and re.search(
                r"\b(?:i'll|i will|let me)\b", statement, re.I
            ):
                return self.speak(
                    "Okay, I'll wait for your update.",
                    {"policy": "device-observations-v1", "heard_request": statement},
                    "wait_for_agent",
                )
            return None
        request_id = hashlib.sha256(
            f"{position}:{normalized(statement)}".encode()
        ).hexdigest()[:12]
        context = {
            "policy": "device-observations-v1",
            "heard_request": statement,
            "request_id": request_id,
        }
        declared = {tool.name: tool for tool in tools}
        observations = Observations(conversation, tools)
        if ambiguous:
            return self.speak(
                "Could you clarify the exact device check or setting you want me to use?",
                context,
                "clarify",
            )
        if any(step.topic == "reboot" for step in steps) and self.needs_reassurance(
            conversation, system_text
        ):
            return self.speak(
                "Will rebooting delete any of my photos?",
                context,
                "persona_clarification",
            )
        reports = []
        for index, step in enumerate(steps):
            prefix = f"dv-{request_id}-{index}-"
            done = [
                (call_id, row)
                for call_id, row in observations.results.items()
                if call_id
                and call_id.startswith(prefix + ("act-" if step.write else "read-"))
            ]
            if done:
                call_id, result = done[-1]
                if (
                    result.get("error")
                    or result.get("execution_status", "executed") != "executed"
                ):
                    return self.speak(
                        result.get("content") or "The device check failed.",
                        {**context, "tool_call_id": call_id},
                        "tool_error",
                    )
                reports.append(
                    Observation(
                        result["content"],
                        result["content"],
                        call_id,
                        step.write or step.read,
                    )
                )
                continue
            if any(call_id.startswith(prefix) for call_id in observations.pending):
                return self.speak(
                    "I'm waiting for the result of that device check.",
                    context,
                    "pending",
                )
            fact = observations.facts.get(step.topic) or observations.facts.get(
                Observations.key(step.read, step.arguments)
            )
            if not step.write:
                if fact is not None and not step.fresh:
                    reports.append(fact)
                    continue
                return self.call(
                    step.read, step.arguments, declared, prefix + "read-", context
                )
            if step.target is not None:
                if fact is None:
                    return self.call(
                        step.read, {}, declared, prefix + "observe-", context
                    )
                if fact.value == step.target:
                    reports.append(fact)
                    continue
            return self.call(
                step.write, step.arguments, declared, prefix + "act-", context
            )
        # Results may contain a status bar in addition to the requested setting.
        # Only report the requested setting for Boolean changes; raw checks keep
        # the actual human-readable tool output.
        text = []
        for step, report in zip(steps, reports, strict=True):
            setting = next((item for item in SETTINGS if item.name == step.topic), None)
            match = (
                re.search(setting.observation, report.text, re.I) if setting else None
            )
            text.append(
                self.vpn_speech(report.text)
                if step.topic == "vpn"
                else match.group(0)
                if match
                else report.text
            )
        return self.speak(
            " ".join(text),
            {**context, "observations": [item.evidence() for item in reports]},
            "report",
        )

    @staticmethod
    def vpn_speech(text):
        """Keep settings and connection distinct, using words clear over speech."""
        line = text.splitlines()[0]
        if re.search(r"VPN is turned OFF", line, re.I):
            return "My VPN is disabled."
        if re.search(
            r"VPN is turned ON in settings, but currently not connected", line, re.I
        ):
            return "My VPN is enabled in settings, but it is not connected."
        if re.search(
            r"VPN disconnected successfully|No active VPN connection", line, re.I
        ):
            return "My VPN is disconnected."
        if re.search(
            r"VPN is ON and connected|VPN connected successfully|VPN already connected",
            line,
            re.I,
        ):
            return "My VPN is connected."
        return line

    def needs_reassurance(self, conversation, system_text):
        persona = self.scenario + "\n" + system_text
        concern = re.search(
            r"\b(?:worried|worries|worry|anxious|afraid|reassurance|uneasy)\b",
            persona,
            re.I,
        ) and re.search(r"\bphotos\b", persona, re.I)
        reassured = any(
            re.search(r"\b(?:photos|pictures|files|data)\b", clause, re.I)
            and re.search(
                r"\b(?:safe|won't|will not|does not|not delete|not erase|not lost)\b",
                clause,
                re.I,
            )
            and not re.search(
                r"\b(?:not safe|might|may|could|unsure)\b|\?", clause, re.I
            )
            for row in conversation
            if row["speaker"] == "service_agent"
            for clause in clauses(row.get("content") or "")
        )
        return bool(concern and not reassured)

    @staticmethod
    def speak(text, context, action):
        return AssistantMessage(
            role="assistant",
            content=text,
            cost=0,
            generation_time_seconds=0,
            raw_data={"device_grounding": {**context, "action": action}},
        )

    def call(self, name, arguments, declared, prefix, context):
        if name not in declared:
            return self.speak(
                "I don't have a way to perform that device check. Could you give me another instruction?",
                context,
                "unavailable_tool",
            )
        return AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=prefix + uuid.uuid4().hex[:12], name=name, arguments=arguments
                )
            ],
            cost=0,
            generation_time_seconds=0,
            raw_data={
                "device_grounding": {
                    **context,
                    "action": "tool",
                    "tool": name,
                    "arguments": arguments,
                }
            },
        )
