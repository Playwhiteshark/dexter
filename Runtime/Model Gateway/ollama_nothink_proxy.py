# Experimental advisor proxy v9: tool-name healer + inference/context diagnostics
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import copy
import hashlib
import json
import socket
import threading
import time

UPSTREAM = "http://127.0.0.1:11434"
PORT = 11435
UPSTREAM_TIMEOUT_SECONDS = 3600

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "ollama_advisor_config.json"
TRACE_FILE = BASE_DIR / "ollama_advisor_trace.jsonl"
ADVISOR_DEBUG_FILE = BASE_DIR / "ollama_advisor_debug.jsonl"
TRACE_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()

DEFAULT_CONFIG = {
    # Stages can be combined with "+": e.g. "preflight+shadow_final".
    #
    # off          = current no-think proxy behavior
    # preflight    = fresh planner creates a compact task contract before execution
    # checkpoint   = fresh progress auditor runs every N observed tool results
    # final        = BLOCKING final-answer gate; candidate is hidden until approved
    # shadow_final = final checker runs after answer is released, logging only
    #
    # Useful experiments:
    # shadow_final
    # preflight+shadow_final
    # checkpoint+shadow_final
    # preflight+checkpoint+shadow_final
    # final
    # preflight+final
    "advisor_mode": "off",

    "advisor_model": "gemma4:12b",
    "advisor_temperature": 0.2,
    "advisor_num_ctx": 8192,
    "advisor_keep_alive": "30m",
    "advisor_timeout_seconds": 600,

    "planner_num_predict": 320,
    "checkpoint_num_predict": 320,
    "checker_num_predict": 384,

    "checkpoint_every_tool_results": 4,
    "max_final_review_rounds": 3,

    # If the advisor infrastructure itself fails:
    # "pass"  = release the main-model response
    # "block" = replace it with a verification-error message
    "advisor_error_policy": "pass",

    # Maximum deterministic transcript text handed to an advisor call.
    # The proxy keeps the beginning and end when a trajectory is larger.
    "advisor_trajectory_max_chars": 28000,

    "trace_enabled": True,

    # Deterministic separator-only repair for model-emitted tool names.
    # Example: filesystem-list_directory -> filesystem-list-directory.
    # A repair is made only when exactly one registered tool matches after
    # treating '-' and '_' as equivalent at the same character positions.
    "tool_name_healer": True,

    # Qwen3.5 inference mode:
    #   "thinking_general" = Qwen's recommended thinking profile for general/agentic tasks
    #   "thinking_coding"  = Qwen's recommended thinking profile for precise coding tasks
    #   "nonthinking"      = Qwen's recommended direct/instruct profile
    #
    # Qwen3.5 thinks by default; use thinking_general for the first Jarvis benchmark.
    # Gemma 4 inference mode:
    #   "thinking"    = enable Gemma 4's reasoning mode
    #   "nonthinking" = direct mode
    #
    # Sampling stays at Google's recommended 1.0 / 0.95 / 64 in both modes.
    "gemma4_mode": "nonthinking",

    "qwen35_mode": "thinking_general",

    # MagenticBrain adapter.
    #
    # Current MagenticLite requires every orchestration turn to either:
    #   - issue a tool call, or
    #   - return a final answer.
    #
    # AnythingLLM ends an agent turn on bare assistant text, so a progress-only
    # response such as "Next I should inspect schema.py" can prematurely end the
    # task. "magenticlite" mode adds a compact completion contract and retries
    # one bare-text response internally before AnythingLLM sees it.
    "magentic_harness_mode": "magenticlite",
    "magentic_bare_text_retries": 1,

    # TEMPORARY diagnostics only. When true, writes the exact user request,
    # real trajectory, candidate answer, and advisor result to a separate file.
    # This may contain file contents/tool output, so turn it off after debugging.
    "trace_advisor_packets": False
}

PLAN_CACHE = {}
CHECKPOINT_STATE = {}
TURN_REGISTRY = {}
MAX_CACHED_TURNS = 64

HOP_BY_HOP = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
}


# Main-model inference profiles.
#
# These exist so model comparisons are not contaminated by forcing one
# model family's recommended sampling settings onto another.
#
# Unknown models are passed through unchanged.
MAIN_MODEL_PROFILES = (
    {
        "name": "magenticbrain-recommended",
        "matches": ("magenticbrain",),
        "think": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.0,
        },
    },
)

GEMMA4_PROFILES = {
    "thinking": {
        "name": "gemma4-recommended-thinking",
        "think": True,
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        },
    },
    "nonthinking": {
        "name": "gemma4-recommended-nonthinking",
        "think": False,
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        },
    },
}

QWEN35_PROFILES = {
    "thinking_general": {
        "name": "qwen3.5-recommended-thinking-general",
        "think": True,
        "options": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
        },
    },
    "thinking_coding": {
        "name": "qwen3.5-recommended-thinking-coding",
        "think": True,
        "options": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        },
    },
    "nonthinking": {
        "name": "qwen3.5-recommended-nonthinking",
        "think": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
        },
    },
}



def apply_main_model_profile(request_data, config):
    """Apply a model-family-specific inference profile in-place.

    Gemma 4 and Qwen3.5 are switchable through config so thinking/non-thinking
    A/B tests do not require editing this Python file. Unknown models are passed through.
    """
    if not isinstance(request_data, dict):
        return "passthrough"

    model = str(request_data.get("model") or "").lower()

    if "gemma4:" in model:
        requested_mode = str(config.get("gemma4_mode", "nonthinking")).strip().lower()
        profile = GEMMA4_PROFILES.get(requested_mode, GEMMA4_PROFILES["nonthinking"])

        request_data["think"] = bool(profile["think"])

        options = request_data.get("options")
        if not isinstance(options, dict):
            options = {}
            request_data["options"] = options

        for key, value in profile["options"].items():
            options[key] = value

        return profile["name"]

    if "qwen3.5:" in model:
        requested_mode = str(config.get("qwen35_mode", "thinking_general")).strip().lower()
        profile = QWEN35_PROFILES.get(requested_mode, QWEN35_PROFILES["thinking_general"])

        request_data["think"] = bool(profile["think"])

        options = request_data.get("options")
        if not isinstance(options, dict):
            options = {}
            request_data["options"] = options

        for key, value in profile["options"].items():
            options[key] = value

        return profile["name"]

    for profile in MAIN_MODEL_PROFILES:
        if not any(token in model for token in profile["matches"]):
            continue

        request_data["think"] = bool(profile["think"])

        options = request_data.get("options")
        if not isinstance(options, dict):
            options = {}
            request_data["options"] = options

        for key, value in profile["options"].items():
            options[key] = value

        return profile["name"]

    return "passthrough"


MAGENTIC_HARNESS_MARKER = "[MAGENTICLITE HARNESS CONTRACT]"

MAGENTIC_HARNESS_GUIDANCE = """[MAGENTICLITE HARNESS CONTRACT]
For agent/tool tasks:
- If more work is needed, call the next declared tool. Do not send a progress-only response with no tool call.
- When the task is complete or no further tool calls are useful, put the final user-facing answer inside <answer>...</answer>.
- After every tool result, either call another tool or return <answer>...</answer>. Never stop on bare planning/progress text.
- When done, stop immediately; do not keep exploring or repeat completed steps.

Task-state rules:
- Treat the current agent invocation as the current task. Do not assume that similar work from another task or conversation was already performed.
- Tool calls/results present in the CURRENT trajectory are the authoritative record of environment work already performed for this task.
- Never claim that you already searched, inspected, read, found, moved, opened, modified, created, ran, or verified something unless the current trajectory contains tool evidence supporting that claim.
- Familiarity, memory, or a belief that you answered a similar request before is not evidence that this task is complete.
"""


def is_magenticbrain_request(request_data):
    if not isinstance(request_data, dict):
        return False
    model = str(request_data.get("model") or "").lower()
    return "magenticbrain" in model


def magentic_harness_enabled(request_data, config):
    return (
        is_magenticbrain_request(request_data)
        and is_agent_request(request_data)
        and str(config.get("magentic_harness_mode", "magenticlite")).lower()
        == "magenticlite"
    )


def magentic_task_evidence_text(messages):
    """Return objective current-task evidence derived only from the request."""
    if not isinstance(messages, list):
        messages = []
    evidence_count = tool_result_count(messages)
    if evidence_count == 0:
        return (
            "\n[MAGENTIC CURRENT-TASK EVIDENCE]\n"
            "Tool results present in this task: 0.\n"
            "Therefore no environment/tool work has yet been proven complete "
            "for this task. If the request requires inspecting, searching, "
            "reading, changing, running, opening, moving, finding, or verifying "
            "the environment, obtain relevant tool evidence before claiming "
            "completion.\n"
        )
    return (
        "\n[MAGENTIC CURRENT-TASK EVIDENCE]\n"
        f"Tool results present in this task: {evidence_count}.\n"
        "Only those current-task tool results may be treated as evidence of "
        "environment work already completed.\n"
    )


def inject_magentic_harness_guidance(request_data, config):
    """Add the MagenticLite-style contract plus objective task evidence."""
    if not magentic_harness_enabled(request_data, config):
        return request_data, False

    working = copy.deepcopy(request_data)
    messages = working.get("messages")
    if not isinstance(messages, list):
        messages = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        if MAGENTIC_HARNESS_MARKER in content_to_text(message.get("content")):
            working["messages"] = messages
            return working, False

    evidence_text = magentic_task_evidence_text(messages)
    guidance = MAGENTIC_HARNESS_GUIDANCE + evidence_text

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            current = content_to_text(message.get("content"))
            message["content"] = (
                (current.rstrip() + "\n\n") if current.strip() else ""
            ) + guidance
            working["messages"] = messages
            return working, True

    messages.insert(
        0,
        {
            "role": "system",
            "content": guidance,
        },
    )
    working["messages"] = messages
    return working, True


def extract_magentic_answer(content):
    """Return text inside a complete <answer>...</answer> block, else None."""
    content = content_to_text(content)
    start_tag = "<answer>"
    end_tag = "</answer>"
    start = content.find(start_tag)
    if start < 0:
        return None
    end = content.rfind(end_tag)
    if end < start:
        return None
    return content[start + len(start_tag):end].strip()


def final_object_from_parsed(parsed):
    if not isinstance(parsed, dict):
        return None
    objects = parsed.get("objects")
    if not isinstance(objects, list):
        return None
    for obj in reversed(objects):
        if isinstance(obj, dict) and obj.get("done") is True:
            return obj
    for obj in reversed(objects):
        if isinstance(obj, dict):
            return obj
    return None


def synthetic_answer_response(request_data, answer, source_obj=None):
    """Create a clean Ollama response containing only the final answer."""
    model = request_data.get("model", "")
    stream = request_data.get("stream", True)

    payload = {
        "model": model,
        "created_at": utc_now(),
        "message": {
            "role": "assistant",
            "content": answer,
        },
        "done": True,
        "done_reason": "stop",
    }

    if isinstance(source_obj, dict):
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        ):
            if key in source_obj:
                payload[key] = source_obj[key]

    if stream is False:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def make_magentic_bare_text_retry_request(current_request, candidate):
    """Append the hidden MagenticLite-style one-shot correction turn."""
    retry_request = copy.deepcopy(current_request)
    messages = list(retry_request.get("messages") or [])

    messages.append(
        {
            "role": "assistant",
            "content": candidate,
        }
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "<tool_response>Your response did not contain a tool call or "
                "a final <answer> block. Use only tool calls/results in this "
                "CURRENT task as evidence of environment work already done. "
                "Do not claim prior completion from memory or familiarity. "
                "If you are genuinely done, wrap the final user-facing answer "
                "in <answer>...</answer>. Otherwise call the next declared "
                "tool now. Do not send bare planning or progress text."
                "</tool_response>"
            ),
        }
    )
    retry_request["messages"] = messages
    return retry_request


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_config():
    config = dict(DEFAULT_CONFIG)
    try:
        if CONFIG_FILE.exists():
            loaded = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
    except Exception as error:
        trace_event({
            "event": "config_error",
            "error": repr(error),
            "config_file": str(CONFIG_FILE),
        }, force=True)
    return config


def parse_stages(config):
    raw = str(config.get("advisor_mode", "off")).strip().lower()
    if not raw or raw == "off":
        return set()

    stages = {part.strip() for part in raw.replace(",", "+").split("+") if part.strip()}

    if "all" in stages:
        stages.remove("all")
        stages.update({"preflight", "checkpoint", "final"})

    valid = {"preflight", "checkpoint", "final", "shadow_final"}
    stages &= valid

    # A blocking final gate already performs the final audit.
    if "final" in stages:
        stages.discard("shadow_final")

    return stages


def trace_event(event, force=False):
    try:
        config = load_config() if not force else DEFAULT_CONFIG
        if not force and not config.get("trace_enabled", True):
            return
        payload = {"ts": utc_now(), **event}
        line = json.dumps(payload, ensure_ascii=False)
        with TRACE_LOCK:
            with TRACE_FILE.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        pass


def trace_advisor_packet(event):
    """Write exact advisor inputs only when explicitly enabled.

    WARNING: This file can contain prompts, file contents, and tool results.
    """
    try:
        config = load_config()
        if not config.get("trace_advisor_packets", False):
            return
        payload = {"ts": utc_now(), **event}
        line = json.dumps(payload, ensure_ascii=False)
        with TRACE_LOCK:
            with ADVISOR_DEBUG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        pass


def cleanup_state():
    with STATE_LOCK:
        if len(PLAN_CACHE) > MAX_CACHED_TURNS:
            ordered = sorted(
                PLAN_CACHE.items(),
                key=lambda item: item[1].get("last_used", 0)
            )
            for key, _ in ordered[:len(PLAN_CACHE) - MAX_CACHED_TURNS]:
                PLAN_CACHE.pop(key, None)

        if len(CHECKPOINT_STATE) > MAX_CACHED_TURNS:
            ordered = sorted(
                CHECKPOINT_STATE.items(),
                key=lambda item: item[1].get("last_used", 0)
            )
            for key, _ in ordered[:len(CHECKPOINT_STATE) - MAX_CACHED_TURNS]:
                CHECKPOINT_STATE.pop(key, None)

        if len(TURN_REGISTRY) > MAX_CACHED_TURNS:
            ordered = sorted(
                TURN_REGISTRY.items(),
                key=lambda item: item[1].get("last_used", 0)
            )
            for key, _ in ordered[:len(TURN_REGISTRY) - MAX_CACHED_TURNS]:
                TURN_REGISTRY.pop(key, None)


def content_to_text(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def last_user_index(messages):
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return -1


def turn_fingerprint(messages):
    """Stable fingerprint for the same logical user prompt/history.

    This is NOT itself the run ID: repeated benchmark conversations can have
    byte-identical history and therefore the same fingerprint.
    """
    index = last_user_index(messages)
    if index < 0:
        return None
    prefix = messages[:index + 1]
    raw = json.dumps(prefix, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def current_turn_key(messages):
    """Return a per-execution run ID that survives tool-loop continuations.

    A repeated fresh conversation with identical text must NOT reuse the
    preflight/checkpoint caches from the previous benchmark run.

    AnythingLLM's continuation requests contain assistant/tool messages after
    the current user message. Once a run has shown such progress, a later
    no-progress request with the same fingerprint is treated as a new run.
    """
    fingerprint = turn_fingerprint(messages)
    if fingerprint is None:
        return None

    index = last_user_index(messages)
    post_user = messages[index + 1:] if index >= 0 else []
    has_progress = bool(post_user)
    now = time.time()

    with STATE_LOCK:
        state = TURN_REGISTRY.get(fingerprint)

        new_run = state is None

        if state is not None and not has_progress:
            # If the previous identical prompt already entered a tool loop,
            # this no-progress request is a fresh benchmark/conversation.
            if state.get("saw_progress", False):
                new_run = True
            # Also separate identical no-tool runs after a modest idle gap.
            elif now - state.get("last_used", 0) > 15:
                new_run = True

        if new_run:
            state = {
                "turn_key": uuid4().hex[:24],
                "saw_progress": bool(has_progress),
                "created": now,
                "last_used": now,
            }
            TURN_REGISTRY[fingerprint] = state
        else:
            state["last_used"] = now
            if has_progress:
                state["saw_progress"] = True

        return state["turn_key"]


def current_user_request(messages):
    index = last_user_index(messages)
    if index < 0:
        return ""
    return content_to_text(messages[index].get("content"))


def extract_tool_names(request_data):
    names = []
    for tool in request_data.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
        elif tool.get("name"):
            names.append(str(tool["name"]))
    return names



def request_diagnostics(request_data):
    """Return safe request metadata only; never log prompt/tool-schema contents."""
    if not isinstance(request_data, dict):
        return {}

    options = request_data.get("options")
    if not isinstance(options, dict):
        options = {}

    messages = request_data.get("messages")
    if not isinstance(messages, list):
        messages = []

    role_counts = {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool": 0,
        "other": 0,
    }
    for message in messages:
        if not isinstance(message, dict):
            role_counts["other"] += 1
            continue
        role = str(message.get("role", "other"))
        if role in role_counts:
            role_counts[role] += 1
        else:
            role_counts["other"] += 1

    return {
        "model": request_data.get("model"),
        "stream": request_data.get("stream"),
        "think": request_data.get("think"),
        "keep_alive": request_data.get("keep_alive"),
        "temperature": options.get("temperature"),
        "top_p": options.get("top_p"),
        "top_k": options.get("top_k"),
        "min_p": options.get("min_p"),
        "presence_penalty": options.get("presence_penalty"),
        "repeat_penalty": options.get("repeat_penalty"),
        "num_ctx": options.get("num_ctx"),
        "num_predict": options.get("num_predict"),
        "seed": options.get("seed"),
        "message_count": len(messages),
        "system_message_count": role_counts["system"],
        "user_message_count": role_counts["user"],
        "assistant_message_count": role_counts["assistant"],
        "tool_message_count": role_counts["tool"],
        "other_message_count": role_counts["other"],
        "tool_schema_count": len(extract_tool_names(request_data)),
        "post_user_tool_result_count": tool_result_count(messages) if messages else 0,
    }


def parse_ollama_json_object(raw):
    """Parse one Ollama JSON/NDJSON line without retaining/logging its content."""
    if not raw:
        return None

    payload = raw[:-1] if raw.endswith(b"\n") else raw
    if not payload.strip():
        return None

    try:
        obj = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    return obj if isinstance(obj, dict) else None


def duration_ns_to_ms(value):
    try:
        return round(float(value) / 1_000_000.0, 3)
    except (TypeError, ValueError):
        return None


def eval_tokens_per_second(eval_count, eval_duration_ns):
    try:
        duration_seconds = float(eval_duration_ns) / 1_000_000_000.0
        if duration_seconds <= 0:
            return None
        return round(float(eval_count) / duration_seconds, 3)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def trace_main_inference_metrics(
    final_obj,
    forwarded_request,
    runtime_meta,
    request_id,
    response_tool_names=None,
    response_has_content=False,
    phase="stream",
):
    """Log final Ollama counters for one main-model inference.

    Only metadata/counters are written. Prompt text, tool results, and response
    text are deliberately excluded.
    """
    if not isinstance(final_obj, dict):
        return

    request_info = request_diagnostics(forwarded_request or {})
    prompt_eval_count = final_obj.get("prompt_eval_count")
    num_ctx = request_info.get("num_ctx")

    context_usage_percent = None
    try:
        if prompt_eval_count is not None and num_ctx:
            context_usage_percent = round(
                (float(prompt_eval_count) / float(num_ctx)) * 100.0,
                2,
            )
    except (TypeError, ValueError, ZeroDivisionError):
        context_usage_percent = None

    tool_names = []
    for name in response_tool_names or []:
        if name and name not in tool_names:
            tool_names.append(name)

    if tool_names:
        response_kind = "tool_call"
    elif response_has_content:
        response_kind = "text"
    else:
        response_kind = "empty"

    trace_event({
        "event": "main_inference_metrics",
        "request_id": request_id,
        "turn_key": runtime_meta.get("turn_key") if runtime_meta else None,
        "turn_fingerprint": runtime_meta.get("turn_fingerprint") if runtime_meta else None,
        "phase": phase,
        "model": request_info.get("model"),
        "response_kind": response_kind,
        "response_tool_call_names": tool_names,
        "done_reason": final_obj.get("done_reason"),
        "prompt_eval_count": prompt_eval_count,
        "eval_count": final_obj.get("eval_count"),
        "num_ctx": num_ctx,
        "context_usage_percent": context_usage_percent,
        "message_count": request_info.get("message_count"),
        "tool_schema_count": request_info.get("tool_schema_count"),
        "post_user_tool_result_count": request_info.get("post_user_tool_result_count"),
        "temperature": request_info.get("temperature"),
        "top_p": request_info.get("top_p"),
        "top_k": request_info.get("top_k"),
        "min_p": request_info.get("min_p"),
        "presence_penalty": request_info.get("presence_penalty"),
        "repeat_penalty": request_info.get("repeat_penalty"),
        "sampling_profile": runtime_meta.get("sampling_profile") if runtime_meta else None,
        "think": request_info.get("think"),
        "total_duration_ms": duration_ns_to_ms(final_obj.get("total_duration")),
        "load_duration_ms": duration_ns_to_ms(final_obj.get("load_duration")),
        "prompt_eval_duration_ms": duration_ns_to_ms(final_obj.get("prompt_eval_duration")),
        "eval_duration_ms": duration_ns_to_ms(final_obj.get("eval_duration")),
        "eval_tokens_per_second": eval_tokens_per_second(
            final_obj.get("eval_count"),
            final_obj.get("eval_duration"),
        ),
    })


def trace_metrics_from_buffered_body(
    body,
    forwarded_request,
    runtime_meta,
    request_id,
    phase="buffered",
):
    """Find the final metrics object in a buffered Ollama JSON/NDJSON body."""
    parsed = parse_ollama_response(body)
    objects = parsed.get("objects") or []

    response_tool_names = []
    response_has_content = False
    final_obj = None

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        message = obj.get("message")
        if isinstance(message, dict):
            if message.get("content"):
                response_has_content = True
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    function = call.get("function")
                    if isinstance(function, dict) and function.get("name"):
                        response_tool_names.append(str(function["name"]))

        if obj.get("done") is True:
            final_obj = obj

    if final_obj is not None:
        trace_main_inference_metrics(
            final_obj,
            forwarded_request,
            runtime_meta,
            request_id,
            response_tool_names=response_tool_names,
            response_has_content=response_has_content,
            phase=phase,
        )


def normalize_tool_name_separators(name):
    """Canonicalize only '-' vs '_' separators; do not fuzzy-match semantics.

    Same length/positions are preserved. For example:
      filesystem-list_directory -> filesystem-list-directory
      filesystem_read_text_file -> filesystem-read-text-file

    But unrelated names such as filesystem-search-directory remain unrelated.
    """
    return "".join("-" if char in {"-", "_"} else char for char in str(name))


def unique_separator_tool_match(emitted_name, registered_names):
    """Return the one safe registered match, or None if exact/none/ambiguous."""
    emitted_name = str(emitted_name)
    if emitted_name in registered_names:
        return emitted_name

    normalized = normalize_tool_name_separators(emitted_name)
    matches = [
        name for name in registered_names
        if normalize_tool_name_separators(name) == normalized
    ]

    if len(matches) == 1:
        return matches[0]
    return None


def heal_tool_calls_in_object(obj, registered_names, request_id=None, turn_key=None):
    """Repair separator-only tool-name mistakes in one Ollama response object.

    Returns (changed, repairs). This intentionally does not repair argument names,
    path strings, semantic tool inventions, typos, case differences, or ambiguous
    names. The registered tool schema remains the sole source of valid names.
    """
    if not isinstance(obj, dict) or not registered_names:
        return False, []

    message = obj.get("message")
    if not isinstance(message, dict):
        return False, []

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False, []

    changed = False
    repairs = []

    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue

        original = function.get("name")
        if not isinstance(original, str) or not original:
            continue
        if original in registered_names:
            continue

        repaired = unique_separator_tool_match(original, registered_names)
        if repaired is None or repaired == original:
            continue

        function["name"] = repaired
        changed = True
        repair = {
            "tool_call_index": index,
            "original_name": original,
            "repaired_name": repaired,
        }
        repairs.append(repair)
        trace_event({
            "event": "tool_name_healed",
            "request_id": request_id,
            "turn_key": turn_key,
            **repair,
        })

    return changed, repairs


def heal_ollama_json_bytes(raw, registered_names, request_id=None, turn_key=None):
    """Heal one JSON/NDJSON line while preserving untouched bytes verbatim."""
    if not raw or not registered_names:
        return raw

    # readline() normally gives one NDJSON object, including its newline.
    newline = b"\n" if raw.endswith(b"\n") else b""
    payload = raw[:-1] if newline else raw
    if not payload.strip():
        return raw

    try:
        obj = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return raw

    changed, _ = heal_tool_calls_in_object(
        obj,
        registered_names,
        request_id=request_id,
        turn_key=turn_key,
    )
    if not changed:
        return raw

    return json.dumps(
        obj,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + newline


def heal_ollama_body(body, registered_names, request_id=None, turn_key=None):
    """Heal a buffered Ollama response containing JSON or NDJSON."""
    if not body or not registered_names:
        return body

    # Preserve NDJSON framing. splitlines(keepends=True) also works for a single
    # non-streaming JSON object, in which case there is one item.
    parts = body.splitlines(keepends=True)
    if not parts:
        return body

    return b"".join(
        heal_ollama_json_bytes(
            part,
            registered_names,
            request_id=request_id,
            turn_key=turn_key,
        )
        for part in parts
    )


def is_agent_request(request_data):
    return (
        isinstance(request_data, dict)
        and isinstance(request_data.get("messages"), list)
        and bool(request_data.get("tools"))
    )


def tool_result_count(messages):
    user_index = last_user_index(messages)
    if user_index < 0:
        return 0
    count = 0
    for message in messages[user_index + 1:]:
        role = message.get("role")
        if role == "tool":
            count += 1
            continue

        # Tolerate wrappers that encode tool results in a nonstandard role.
        if message.get("tool_call_id") and role not in {"assistant", "system"}:
            count += 1
    return count


def shorten(text, limit):
    text = content_to_text(text)
    if len(text) <= limit:
        return text
    if limit < 200:
        return text[:limit]
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n...[TRUNCATED BY PROXY]...\n" + text[-tail:]


def trajectory_text(messages, max_chars):
    """Mechanically serialize the real post-user trajectory.

    This does not ask the main model to summarize its own work. Tool calls and
    tool results come directly from the request transcript.
    """
    user_index = last_user_index(messages)
    if user_index < 0:
        relevant = messages
    else:
        relevant = messages[user_index + 1:]

    parts = []

    for message in relevant:
        role = str(message.get("role", "unknown")).upper()
        tool_calls = message.get("tool_calls")

        if tool_calls:
            parts.append(
                f"{role} TOOL_CALLS:\n"
                + json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
            )

        content = content_to_text(message.get("content"))
        if content:
            # Keep tool results richer than ordinary assistant narration.
            per_message_limit = 8000 if message.get("role") == "tool" else 2000
            parts.append(f"{role}:\n{shorten(content, per_message_limit)}")

    combined = "\n\n".join(parts)

    if len(combined) <= max_chars:
        return combined

    # Keep both early discovery and recent evidence rather than only the tail.
    half = max_chars // 2
    return (
        combined[:half]
        + "\n\n...[MIDDLE OF TRAJECTORY TRUNCATED BY PROXY]...\n\n"
        + combined[-half:]
    )


def make_advisor_request(messages, config, num_predict):
    payload = {
        "model": config.get("advisor_model", "gemma4:12b"),
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": config.get("advisor_keep_alive", "30m"),
        "options": {
            "temperature": float(config.get("advisor_temperature", 0.2)),
            "num_ctx": int(config.get("advisor_num_ctx", 8192)),
            "num_predict": int(num_predict),
        },
    }

    body = json.dumps(payload).encode("utf-8")
    request = Request(
        UPSTREAM + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    timeout = float(config.get("advisor_timeout_seconds", 600))
    start = time.perf_counter()

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except Exception as error:
        return {
            "ok": False,
            "error": repr(error),
            "duration_seconds": time.perf_counter() - start,
        }

    try:
        parsed = json.loads(raw.decode("utf-8"))
        text = content_to_text((parsed.get("message") or {}).get("content")).strip()
        return {
            "ok": True,
            "text": text,
            "duration_seconds": time.perf_counter() - start,
        }
    except Exception as error:
        return {
            "ok": False,
            "error": f"Could not parse advisor response: {error!r}",
            "raw": raw.decode("utf-8", errors="replace"),
            "duration_seconds": time.perf_counter() - start,
        }


def planner_call(user_request, tool_names, config):
    system = """You are a requirements-extraction pass for a tool-using local assistant.

Do not solve the task. Do not propose an execution strategy. Do not mention tool names.
Do not add any requirement, constraint, path-discovery rule, search rule, or ordering rule
that the user did not explicitly state. User-supplied paths, names, and scopes are
authoritative and must not be treated as guesses that need rediscovery.

Extract only a compact checklist of the user's explicit requirements.

Return plain text with exactly these headings:
MUST DO:
MUST NOT:
DONE WHEN:

Rules:
- Preserve every explicit user requirement and constraint.
- Split independently testable requirements into separate bullets.
- Do not infer extra process requirements.
- Do not advise how to use tools.
- Do not restate optional ideas as requirements.
- Keep it concise."""
    user = f"USER REQUEST:\n{user_request}"
    return make_advisor_request(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        config,
        min(config.get("planner_num_predict", 320), 200),
    )


def checkpoint_call(user_request, trajectory, plan, config):
    system = """You are an independent progress auditor for a tool-using agent.

The executor is mid-task. Use the real tool trajectory as evidence.
Do not redo the task and do not write a final answer.

Return one of:
ON_TRACK
or
COURSE_CORRECTION:
- <specific missing, failed, repeated, unsupported, or out-of-scope item>
- <minimal next step>

Rules:
- Tool results are authoritative.
- Assistant narration is not proof an action happened.
- Compare progress to the user's explicit requirements.
- Do not complain about optional work.
- Do not invent requirements.
- Prefer a minimal correction, not a new elaborate plan."""
    user = (
        f"USER REQUEST:\n{user_request}\n\n"
        f"PRE-FLIGHT PLAN (may be absent and is not authoritative):\n{plan or '[none]'}\n\n"
        f"REAL TRAJECTORY:\n{trajectory or '[no tool evidence yet]'}"
    )
    return make_advisor_request(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        config,
        config.get("checkpoint_num_predict", 320),
    )


def final_checker_call(user_request, trajectory, candidate, plan, config):
    system = """You are an independent completion auditor for a tool-using agent.

Your job is not to judge whether the attempt seems generally reasonable.
Your job is to verify every explicit user requirement against the actual
trajectory before approving completion.

The ORIGINAL USER REQUEST is authoritative.
The PRE-FLIGHT PLAN is only a hint and may be wrong or absent.
The REAL TOOL TRAJECTORY is evidence.
The CANDIDATE FINAL ANSWER is a set of claims to audit, not evidence that work happened.

You MUST produce a concise structured audit in this exact order:

REQUIREMENTS:
R1: <one explicit requirement or constraint>
R2: <...>

REQUIREMENT AUDIT:
R1: PASS | FAIL | UNKNOWN — <specific tool evidence or specific missing evidence>
R2: PASS | FAIL | UNKNOWN — <specific tool evidence or specific missing evidence>

CANDIDATE CLAIM AUDIT:
C1: SUPPORTED | UNSUPPORTED — <important completion/factual claim and evidence>
C2: ...

FINAL VERDICT:
OK

OR:

FINAL VERDICT:
ISSUES:
- <specific unresolved requirement or unsupported claim>
- <another issue if necessary>

Hard rules:
- First extract ALL explicit requirements and constraints from the original request.
- Do not merge separately testable requirements just to make the checklist shorter.
- If the user says to list every discovered subdirectory, each discovered subdirectory
  must have evidence of an actual successful directory-list call. Reading a file inside
  a directory is not evidence that the directory itself was listed.
- If the user says to inspect a file or documentation, listing or mentioning it is not
  evidence that its contents were inspected.
- A failed tool call is not successful work.
- Repeated failed or invalid calls must not be treated as progress.
- Tool results are authoritative.
- Assistant narration is never proof an action happened.
- A candidate statement such as "completed", "full review", "all", or a checked-off
  checklist must be supported by the trajectory.
- Flag material factual claims that contradict or overstate the observed file contents.
- Distinguish historical/documented procedures from current requirements when the file
  itself makes that distinction.
- Do not invent optional requirements or complain that the agent failed to do work the
  user did not request.
- Do not fail a run merely because its process was inefficient if all requirements were
  ultimately completed, unless the inefficiency itself contradicts an explicit constraint
  or produces an unsupported candidate claim.
- You may return OK only when EVERY explicit requirement is PASS and every material
  candidate claim is supported.
- If any requirement is FAIL or UNKNOWN, FINAL VERDICT must be ISSUES.
- Keep the audit concise. Evidence citations should name the relevant tool action/result,
  not reproduce the whole transcript."""

    user = (
        f"ORIGINAL USER REQUEST:\n{user_request}\n\n"
        f"PRE-FLIGHT PLAN (non-authoritative):\n{plan or '[none]'}\n\n"
        f"REAL TOOL TRAJECTORY:\n{trajectory or '[no tool evidence]'}\n\n"
        f"CANDIDATE FINAL ANSWER:\n{candidate}"
    )
    return make_advisor_request(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        config,
        config.get("checker_num_predict", 384),
    )


def get_or_create_plan(turn_key, user_request, tool_names, config):
    now = time.time()
    with STATE_LOCK:
        cached = PLAN_CACHE.get(turn_key)
        if cached:
            cached["last_used"] = now
            return cached["plan"], False, cached.get("advisor_meta")

    result = planner_call(user_request, tool_names, config)

    if result.get("ok"):
        plan = result.get("text", "").strip()
        with STATE_LOCK:
            PLAN_CACHE[turn_key] = {
                "plan": plan,
                "created": now,
                "last_used": now,
                "advisor_meta": result,
            }
        cleanup_state()
        return plan, True, result

    trace_event({
        "event": "preflight_error",
        "turn_key": turn_key,
        "error": result.get("error"),
        "duration_seconds": result.get("duration_seconds"),
    })
    return None, True, result


def maybe_checkpoint(turn_key, user_request, messages, plan, config):
    every = max(1, int(config.get("checkpoint_every_tool_results", 4)))
    count = tool_result_count(messages)
    if count < every:
        return None

    bucket = count // every
    now = time.time()

    with STATE_LOCK:
        state = CHECKPOINT_STATE.get(turn_key, {"bucket": 0, "last_used": now})
        if bucket <= state.get("bucket", 0):
            state["last_used"] = now
            CHECKPOINT_STATE[turn_key] = state
            return None

        # Reserve this bucket before the potentially slow advisor call so two
        # concurrent requests do not duplicate the same checkpoint.
        CHECKPOINT_STATE[turn_key] = {"bucket": bucket, "last_used": now}

    trajectory = trajectory_text(
        messages,
        int(config.get("advisor_trajectory_max_chars", 28000)),
    )
    result = checkpoint_call(user_request, trajectory, plan, config)

    trace_event({
        "event": "checkpoint",
        "turn_key": turn_key,
        "tool_result_count": count,
        "bucket": bucket,
        "ok": result.get("ok"),
        "verdict": result.get("text"),
        "error": result.get("error"),
        "duration_seconds": result.get("duration_seconds"),
    })

    if result.get("ok"):
        return result.get("text", "").strip()

    return None


def inject_runtime_guidance(request_data, stages, config):
    """Return a modified Ollama request plus runtime metadata.

    AnythingLLM itself never sees these injected system messages; they exist
    only in the request forwarded from this proxy to Ollama.
    """
    working = copy.deepcopy(request_data)

    metadata = {
        "turn_key": None,
        "turn_fingerprint": None,
        "user_request": "",
        "plan": None,
        "checkpoint": None,
        # Preserve the exact AnythingLLM transcript before this proxy injects
        # planning/checkpoint guidance. Advisors must audit real evidence, not
        # their own runtime hints.
        "evidence_messages": copy.deepcopy(working.get("messages") or []),
    }

    if not is_agent_request(working):
        return working, metadata

    messages = list(working.get("messages") or [])
    turn_key = current_turn_key(messages)
    fingerprint = turn_fingerprint(messages)
    user_request = current_user_request(messages)

    metadata["turn_key"] = turn_key
    metadata["turn_fingerprint"] = fingerprint
    metadata["user_request"] = user_request

    plan = None

    if "preflight" in stages and turn_key:
        plan, created, result = get_or_create_plan(
            turn_key,
            user_request,
            extract_tool_names(working),
            config,
        )
        metadata["plan"] = plan

        if created:
            trace_event({
                "event": "preflight",
                "turn_key": turn_key,
                "ok": result.get("ok") if result else None,
                "plan": plan,
                "error": result.get("error") if result else None,
                "duration_seconds": result.get("duration_seconds") if result else None,
            })

        if plan:
            contract = (
                "[RUNTIME REQUIREMENTS CHECKLIST]\n"
                "Use this as a checklist, but adapt it when real tool results reveal new facts. "
                "The original user request remains authoritative.\n\n"
                + plan
            )

            # IMPORTANT: keep normal chat/tool role ordering intact.
            # Do NOT append a fresh system message after the latest tool result.
            # Merge the task contract into the earliest existing system message
            # so that, after tool use, the actual TOOL result remains the most
            # recent conversational message before the next inference.
            merged = False
            for message in messages:
                if message.get("role") == "system":
                    existing = content_to_text(message.get("content"))
                    message["content"] = existing + "\n\n" + contract
                    merged = True
                    break

            if not merged:
                messages.insert(0, {
                    "role": "system",
                    "content": contract,
                })

    if "checkpoint" in stages and turn_key:
        checkpoint = maybe_checkpoint(
            turn_key,
            user_request,
            working.get("messages") or [],
            plan,
            config,
        )
        metadata["checkpoint"] = checkpoint

        if checkpoint and checkpoint.strip() != "ON_TRACK":
            messages.append({
                "role": "system",
                "content": (
                    "[RUNTIME MID-TASK AUDIT]\n"
                    "This is independent progress feedback based on the real tool trajectory. "
                    "Address it minimally if valid; do not blindly expand scope.\n\n"
                    + checkpoint
                ),
            })

    working["messages"] = messages
    return working, metadata


def parse_ollama_response(body_bytes):
    """Parse either a normal JSON response or Ollama's NDJSON stream."""
    text = body_bytes.decode("utf-8", errors="replace")
    stripped = text.strip()

    if not stripped:
        return {
            "content": "",
            "tool_calls": [],
            "done": False,
            "objects": [],
            "parse_error": "empty response",
        }

    objects = []

    # Non-streaming response.
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            objects = [obj]
    except json.JSONDecodeError:
        pass

    # Streaming NDJSON response.
    if not objects:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    objects.append(obj)
            except json.JSONDecodeError:
                return {
                    "content": "",
                    "tool_calls": [],
                    "done": False,
                    "objects": objects,
                    "parse_error": f"invalid NDJSON line: {line[:200]}",
                }

    content_parts = []
    tool_calls = []
    done = False

    for obj in objects:
        message = obj.get("message")
        if isinstance(message, dict):
            if message.get("content"):
                content_parts.append(content_to_text(message.get("content")))
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                tool_calls.extend(calls)
        if obj.get("done") is True:
            done = True

    return {
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "done": done,
        "objects": objects,
        "parse_error": None,
    }


def verdict_kind(result):
    if not result.get("ok"):
        return "error"

    text = result.get("text", "").strip()

    # Legacy compact checker format.
    if text == "OK":
        return "ok"
    if text.startswith("ISSUES:"):
        return "issues"

    # Structured checker format. Parse only the explicit FINAL VERDICT block,
    # not incidental occurrences of words such as "PASS", "FAIL", or "OK".
    marker = "FINAL VERDICT:"
    pos = text.rfind(marker)
    if pos >= 0:
        final = text[pos + len(marker):].strip()
        if final == "OK":
            return "ok"
        if final.startswith("ISSUES:"):
            return "issues"

    return "malformed"


def build_headers_for_upstream(original_headers):
    return {
        key: value
        for key, value in original_headers.items()
        if key.lower() not in HOP_BY_HOP
    }


def open_upstream(path, method, headers, body):
    request = Request(
        UPSTREAM + path,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        upstream = urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS)
    except HTTPError as error:
        upstream = error

    return upstream


def read_all(upstream):
    chunks = []
    while True:
        if hasattr(upstream, "read1"):
            chunk = upstream.read1(8192)
        else:
            chunk = upstream.readline()
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def synthetic_block_response(request_data, message):
    model = request_data.get("model", "")
    stream = request_data.get("stream", True)
    payload = {
        "model": model,
        "created_at": utc_now(),
        "message": {
            "role": "assistant",
            "content": message,
        },
        "done": True,
        "done_reason": "stop",
    }

    if stream is False:
        return json.dumps(payload).encode("utf-8")

    return (json.dumps(payload) + "\n").encode("utf-8")


def send_buffered_response(handler, status, upstream_headers, body):
    handler.send_response(status)
    content_type = upstream_headers.get("Content-Type", "application/json")
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()

    try:
        handler.wfile.write(body)
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass

    handler.close_connection = True


def run_shadow_final(user_request, messages, candidate, plan, turn_key, config):
    trajectory = trajectory_text(
        messages,
        int(config.get("advisor_trajectory_max_chars", 28000)),
    )

    trace_event({
        "event": "shadow_final_input_summary",
        "turn_key": turn_key,
        "evidence_message_count": len(messages),
        "trajectory_chars": len(trajectory),
        "candidate_chars": len(candidate),
        "has_preflight_plan": bool(plan),
    })

    trace_advisor_packet({
        "event": "shadow_final_packet",
        "turn_key": turn_key,
        "user_request": user_request,
        "preflight_plan": plan,
        "trajectory": trajectory,
        "candidate": candidate,
    })

    result = final_checker_call(user_request, trajectory, candidate, plan, config)

    trace_advisor_packet({
        "event": "shadow_final_result",
        "turn_key": turn_key,
        "verdict_kind": verdict_kind(result),
        "verdict": result.get("text"),
        "error": result.get("error"),
        "duration_seconds": result.get("duration_seconds"),
    })

    trace_event({
        "event": "shadow_final",
        "turn_key": turn_key,
        "verdict_kind": verdict_kind(result),
        "verdict": result.get("text"),
        "error": result.get("error"),
        "duration_seconds": result.get("duration_seconds"),
    })


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/proxy-health":
            config = load_config()
            self.send_json(
                200,
                {
                    "ok": True,
                    "upstream": UPSTREAM,
                    "config_file": str(CONFIG_FILE),
                    "trace_file": str(TRACE_FILE),
                    "advisor_debug_file": str(ADVISOR_DEBUG_FILE),
                    "trace_advisor_packets": config.get("trace_advisor_packets", False),
                    "advisor_mode": config.get("advisor_mode"),
                    "advisor_stages": sorted(parse_stages(config)),
                    "tool_name_healer": bool(config.get("tool_name_healer", True)),
                    "main_model_profiles": (
                        [profile["name"] for profile in MAIN_MODEL_PROFILES]
                        + [profile["name"] for profile in GEMMA4_PROFILES.values()]
                        + [profile["name"] for profile in QWEN35_PROFILES.values()]
                    ),
                    "gemma4_mode": config.get("gemma4_mode", "nonthinking"),
                    "qwen35_mode": config.get("qwen35_mode", "thinking_general"),
                    "magentic_harness_mode": config.get(
                        "magentic_harness_mode", "magenticlite"
                    ),
                    "magentic_bare_text_retries": int(
                        config.get("magentic_bare_text_retries", 1)
                    ),
                },
            )
            return
        self.proxy()

    def do_POST(self):
        self.proxy()

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def proxy(self):
        request_id = uuid4().hex
        config = load_config()
        stages = parse_stages(config)

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length else None

        path_only = self.path.split("?", 1)[0]
        is_generation_request = path_only in {"/api/chat", "/api/generate"}

        parsed_request = None
        sampling_profile = "passthrough"
        if raw_body and is_generation_request:
            try:
                parsed_request = json.loads(raw_body)
                sampling_profile = apply_main_model_profile(parsed_request, config)
            except (json.JSONDecodeError, UnicodeDecodeError):
                parsed_request = None
                sampling_profile = "passthrough"

        if parsed_request is not None and path_only == "/api/chat":
            forwarded_request, runtime_meta = inject_runtime_guidance(
                parsed_request,
                stages,
                config,
            )
            body = json.dumps(forwarded_request).encode("utf-8")
        elif parsed_request is not None:
            forwarded_request = parsed_request
            runtime_meta = {
                "turn_key": None,
                "user_request": "",
                "plan": None,
                "checkpoint": None,
                "evidence_messages": copy.deepcopy(parsed_request.get("messages") or []),
            }
            body = json.dumps(parsed_request).encode("utf-8")
        else:
            forwarded_request = None
            runtime_meta = {
                "turn_key": None,
                "user_request": "",
                "plan": None,
                "checkpoint": None,
                "evidence_messages": [],
            }
            body = raw_body

        magentic_guidance_injected = False
        if (
            forwarded_request is not None
            and path_only == "/api/chat"
            and magentic_harness_enabled(forwarded_request, config)
        ):
            forwarded_request, magentic_guidance_injected = (
                inject_magentic_harness_guidance(forwarded_request, config)
            )
            body = json.dumps(forwarded_request).encode("utf-8")

        runtime_meta["sampling_profile"] = sampling_profile
        runtime_meta["magentic_harness"] = bool(
            forwarded_request
            and magentic_harness_enabled(forwarded_request, config)
        )

        trace_event({
            "event": "request",
            "request_id": request_id,
            "path": self.path,
            "agent_request": bool(forwarded_request and is_agent_request(forwarded_request)),
            "advisor_stages": sorted(stages),
            "turn_key": runtime_meta.get("turn_key"),
            "turn_fingerprint": runtime_meta.get("turn_fingerprint"),
            "magentic_harness": runtime_meta.get("magentic_harness", False),
            "magentic_guidance_injected": magentic_guidance_injected,
            "magentic_task_tool_evidence_count": (
                tool_result_count(forwarded_request.get("messages") or [])
                if forwarded_request
                and runtime_meta.get("magentic_harness", False)
                else None
            ),
        })

        if forwarded_request is not None and is_generation_request:
            trace_event({
                "event": "main_request_settings",
                "request_id": request_id,
                "path": self.path,
                "turn_key": runtime_meta.get("turn_key"),
                "turn_fingerprint": runtime_meta.get("turn_fingerprint"),
                "sampling_profile": runtime_meta.get("sampling_profile"),
                **request_diagnostics(forwarded_request),
                "serialized_request_bytes": len(body) if body is not None else 0,
            })

        headers = build_headers_for_upstream(self.headers)

        try:
            upstream = open_upstream(self.path, self.command, headers, body)
        except (URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as error:
            reason = getattr(error, "reason", error)
            trace_event({
                "event": "upstream_error",
                "request_id": request_id,
                "error": str(reason),
            })
            self.send_json(
                502,
                {
                    "error": "Ollama upstream is unavailable",
                    "upstream": UPSTREAM,
                    "detail": str(reason),
                },
            )
            return

        is_agent_chat = (
            path_only == "/api/chat"
            and forwarded_request is not None
            and is_agent_request(forwarded_request)
        )

        if (
            is_agent_chat
            and forwarded_request is not None
            and magentic_harness_enabled(forwarded_request, config)
        ):
            self.handle_magentic_agent(
                upstream,
                forwarded_request,
                runtime_meta,
                headers,
                config,
                request_id,
            )
            return

        if is_agent_chat and "final" in stages:
            self.handle_blocking_final(
                upstream,
                forwarded_request,
                runtime_meta,
                headers,
                config,
                request_id,
            )
            return

        self.handle_streaming(
            upstream,
            forwarded_request,
            runtime_meta,
            config,
            request_id,
            enable_shadow=bool(is_agent_chat and "shadow_final" in stages),
        )

    def handle_magentic_agent(
        self,
        upstream,
        forwarded_request,
        runtime_meta,
        headers,
        config,
        request_id,
    ):
        """MagenticLite-style turn adapter for AnythingLLM.

        AnythingLLM treats any assistant text without a tool call as the end of
        an agent run. Current MagenticLite instead requires a tool call or an
        explicit final-answer marker and retries one bare-text turn internally.
        """
        status = getattr(upstream, "status", upstream.code)
        upstream_headers = upstream.headers

        registered_tool_names = (
            extract_tool_names(forwarded_request)
            if config.get("tool_name_healer", True)
            else []
        )

        current_request = forwarded_request
        current_body = read_all(upstream)
        upstream.close()
        current_body = heal_ollama_body(
            current_body,
            registered_tool_names,
            request_id=request_id,
            turn_key=runtime_meta.get("turn_key"),
        )
        trace_metrics_from_buffered_body(
            current_body,
            current_request,
            runtime_meta,
            request_id,
            phase="magentic_initial",
        )

        max_retries = max(
            0,
            int(config.get("magentic_bare_text_retries", 1)),
        )

        for attempt in range(max_retries + 1):
            parsed = parse_ollama_response(current_body)

            if parsed.get("parse_error"):
                trace_event({
                    "event": "magentic_parse_error",
                    "request_id": request_id,
                    "turn_key": runtime_meta.get("turn_key"),
                    "attempt": attempt,
                    "error": parsed.get("parse_error"),
                })
                send_buffered_response(
                    self, status, upstream_headers, current_body
                )
                return

            tool_calls = parsed.get("tool_calls") or []
            if tool_calls:
                trace_event({
                    "event": "magentic_tool_call_passthrough",
                    "request_id": request_id,
                    "turn_key": runtime_meta.get("turn_key"),
                    "attempt": attempt,
                    "tool_call_count": len(tool_calls),
                })
                send_buffered_response(
                    self, status, upstream_headers, current_body
                )
                return

            candidate = content_to_text(parsed.get("content", "")).strip()

            answer = extract_magentic_answer(candidate)
            if answer is not None:
                trace_event({
                    "event": "magentic_explicit_answer",
                    "request_id": request_id,
                    "turn_key": runtime_meta.get("turn_key"),
                    "attempt": attempt,
                    "answer_chars": len(answer),
                })
                cleaned = synthetic_answer_response(
                    forwarded_request,
                    answer,
                    source_obj=final_object_from_parsed(parsed),
                )
                send_buffered_response(
                    self, 200, upstream_headers, cleaned
                )
                return

            if not candidate:
                trace_event({
                    "event": "magentic_empty_response",
                    "request_id": request_id,
                    "turn_key": runtime_meta.get("turn_key"),
                    "attempt": attempt,
                })
                send_buffered_response(
                    self, status, upstream_headers, current_body
                )
                return

            if attempt >= max_retries:
                trace_event({
                    "event": "magentic_bare_text_released",
                    "request_id": request_id,
                    "turn_key": runtime_meta.get("turn_key"),
                    "attempt": attempt,
                    "content_chars": len(candidate),
                })
                send_buffered_response(
                    self, status, upstream_headers, current_body
                )
                return

            trace_event({
                "event": "magentic_bare_text_retry",
                "request_id": request_id,
                "turn_key": runtime_meta.get("turn_key"),
                "attempt": attempt + 1,
                "content_chars": len(candidate),
            })

            retry_request = make_magentic_bare_text_retry_request(
                current_request,
                candidate,
            )
            retry_body = json.dumps(retry_request).encode("utf-8")

            try:
                retry_upstream = open_upstream(
                    self.path,
                    self.command,
                    headers,
                    retry_body,
                )
                status = getattr(
                    retry_upstream, "status", retry_upstream.code
                )
                upstream_headers = retry_upstream.headers
                current_body = read_all(retry_upstream)
                retry_upstream.close()

                current_body = heal_ollama_body(
                    current_body,
                    extract_tool_names(retry_request)
                    if config.get("tool_name_healer", True)
                    else [],
                    request_id=request_id,
                    turn_key=runtime_meta.get("turn_key"),
                )
                trace_metrics_from_buffered_body(
                    current_body,
                    retry_request,
                    runtime_meta,
                    request_id,
                    phase=f"magentic_bare_retry_{attempt + 1}",
                )
                current_request = retry_request
            except Exception as error:
                trace_event({
                    "event": "magentic_retry_error",
                    "request_id": request_id,
                    "turn_key": runtime_meta.get("turn_key"),
                    "attempt": attempt + 1,
                    "error": repr(error),
                })
                send_buffered_response(
                    self, 200, upstream_headers, current_body
                )
                return

    def handle_streaming(
        self,
        upstream,
        forwarded_request,
        runtime_meta,
        config,
        request_id,
        enable_shadow,
    ):
        status = getattr(upstream, "status", upstream.code)
        self.send_response(status)

        for key, value in upstream.headers.items():
            if key.lower() not in HOP_BY_HOP:
                self.send_header(key, value)

        self.send_header("Connection", "close")
        self.end_headers()

        captured = bytearray()
        registered_tool_names = (
            extract_tool_names(forwarded_request)
            if forwarded_request and config.get("tool_name_healer", True)
            else []
        )

        response_tool_names = []
        response_has_content = False
        final_metrics_obj = None

        try:
            while True:
                # Ollama chat streaming is NDJSON. Reading a full line lets us
                # rewrite only the JSON object that contains message.tool_calls
                # without buffering the user's ordinary streamed text response.
                chunk = upstream.readline()

                if not chunk:
                    break

                outgoing = heal_ollama_json_bytes(
                    chunk,
                    registered_tool_names,
                    request_id=request_id,
                    turn_key=runtime_meta.get("turn_key"),
                )

                response_obj = parse_ollama_json_object(outgoing)
                if response_obj is not None:
                    message = response_obj.get("message")
                    if isinstance(message, dict):
                        if message.get("content"):
                            response_has_content = True

                        calls = message.get("tool_calls")
                        if isinstance(calls, list):
                            for call in calls:
                                if not isinstance(call, dict):
                                    continue
                                function = call.get("function")
                                if isinstance(function, dict) and function.get("name"):
                                    response_tool_names.append(str(function["name"]))

                    if response_obj.get("done") is True:
                        final_metrics_obj = response_obj

                if enable_shadow:
                    captured.extend(outgoing)

                self.wfile.write(outgoing)
                self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError):
            trace_event({
                "event": "client_disconnected",
                "request_id": request_id,
            })
        finally:
            upstream.close()
            self.close_connection = True

        if final_metrics_obj is not None:
            trace_main_inference_metrics(
                final_metrics_obj,
                forwarded_request,
                runtime_meta,
                request_id,
                response_tool_names=response_tool_names,
                response_has_content=response_has_content,
                phase="stream",
            )

        if enable_shadow and captured:
            parsed = parse_ollama_response(bytes(captured))
            if not parsed.get("parse_error") and not parsed.get("tool_calls") and parsed.get("content"):
                thread = threading.Thread(
                    target=run_shadow_final,
                    args=(
                        runtime_meta.get("user_request", ""),
                        runtime_meta.get("evidence_messages") or [],
                        parsed.get("content", ""),
                        runtime_meta.get("plan"),
                        runtime_meta.get("turn_key"),
                        config,
                    ),
                    daemon=True,
                )
                thread.start()

    def handle_blocking_final(
        self,
        upstream,
        forwarded_request,
        runtime_meta,
        headers,
        config,
        request_id,
    ):
        status = getattr(upstream, "status", upstream.code)
        upstream_headers = upstream.headers
        body = read_all(upstream)
        upstream.close()

        registered_tool_names = (
            extract_tool_names(forwarded_request)
            if config.get("tool_name_healer", True)
            else []
        )
        body = heal_ollama_body(
            body,
            registered_tool_names,
            request_id=request_id,
            turn_key=runtime_meta.get("turn_key"),
        )
        trace_metrics_from_buffered_body(
            body,
            forwarded_request,
            runtime_meta,
            request_id,
            phase="blocking_initial",
        )

        current_request = forwarded_request
        current_body = body

        max_rounds = max(1, int(config.get("max_final_review_rounds", 3)))

        for round_number in range(1, max_rounds + 1):
            parsed = parse_ollama_response(current_body)

            if parsed.get("parse_error"):
                trace_event({
                    "event": "final_parse_error",
                    "request_id": request_id,
                    "turn_key": runtime_meta.get("turn_key"),
                    "round": round_number,
                    "error": parsed.get("parse_error"),
                })
                send_buffered_response(self, status, upstream_headers, current_body)
                return

            # Tool call: this is not a final answer. Hand it back to AnythingLLM.
            if parsed.get("tool_calls"):
                trace_event({
                    "event": "final_gate_tool_call_passthrough",
                    "request_id": request_id,
                    "turn_key": runtime_meta.get("turn_key"),
                    "round": round_number,
                    "tool_call_count": len(parsed.get("tool_calls") or []),
                })
                send_buffered_response(self, status, upstream_headers, current_body)
                return

            candidate = parsed.get("content", "")
            if not candidate:
                send_buffered_response(self, status, upstream_headers, current_body)
                return

            trajectory = trajectory_text(
                runtime_meta.get("evidence_messages") or [],
                int(config.get("advisor_trajectory_max_chars", 28000)),
            )
            result = final_checker_call(
                runtime_meta.get("user_request", ""),
                trajectory,
                candidate,
                runtime_meta.get("plan"),
                config,
            )
            kind = verdict_kind(result)

            trace_event({
                "event": "final_gate",
                "request_id": request_id,
                "turn_key": runtime_meta.get("turn_key"),
                "round": round_number,
                "verdict_kind": kind,
                "verdict": result.get("text"),
                "error": result.get("error"),
                "duration_seconds": result.get("duration_seconds"),
            })

            if kind == "ok":
                send_buffered_response(self, status, upstream_headers, current_body)
                return

            if kind in {"error", "malformed"}:
                if str(config.get("advisor_error_policy", "pass")).lower() == "pass":
                    send_buffered_response(self, status, upstream_headers, current_body)
                    return

                blocked = synthetic_block_response(
                    forwarded_request,
                    "The runtime verifier failed, so this answer was not released. "
                    "Check ollama_advisor_trace.jsonl for the verifier error.",
                )
                send_buffered_response(self, 200, upstream_headers, blocked)
                return

            # ISSUES: ask the main model to continue *inside the proxy*. If it
            # emits a tool call, AnythingLLM receives and executes that call.
            feedback = result.get("text", "").strip()
            correction_request = copy.deepcopy(current_request)
            correction_messages = list(correction_request.get("messages") or [])
            correction_messages.append({
                "role": "assistant",
                "content": candidate,
            })
            correction_messages.append({
                "role": "system",
                "content": (
                    "[INTERNAL COMPLETION AUDIT]\n"
                    "The candidate answer is not approved. Address only the concrete issues below. "
                    "Use tools if more evidence/action is required. Do not claim completion until resolved.\n\n"
                    + feedback
                ),
            })
            correction_request["messages"] = correction_messages
            correction_request["think"] = current_request.get("think", False)
            correction_body = json.dumps(correction_request).encode("utf-8")

            try:
                correction_upstream = open_upstream(
                    self.path,
                    self.command,
                    headers,
                    correction_body,
                )
                status = getattr(correction_upstream, "status", correction_upstream.code)
                upstream_headers = correction_upstream.headers
                current_body = read_all(correction_upstream)
                correction_upstream.close()
                current_body = heal_ollama_body(
                    current_body,
                    extract_tool_names(correction_request) if config.get("tool_name_healer", True) else [],
                    request_id=request_id,
                    turn_key=runtime_meta.get("turn_key"),
                )
                trace_metrics_from_buffered_body(
                    current_body,
                    correction_request,
                    runtime_meta,
                    request_id,
                    phase=f"blocking_correction_{round_number}",
                )
                current_request = correction_request
            except Exception as error:
                trace_event({
                    "event": "final_correction_error",
                    "request_id": request_id,
                    "turn_key": runtime_meta.get("turn_key"),
                    "round": round_number,
                    "error": repr(error),
                })
                if str(config.get("advisor_error_policy", "pass")).lower() == "pass":
                    send_buffered_response(self, 200, upstream_headers, body)
                else:
                    blocked = synthetic_block_response(
                        forwarded_request,
                        "The runtime verifier requested a correction, but the correction pass failed. "
                        "Check ollama_advisor_trace.jsonl.",
                    )
                    send_buffered_response(self, 200, upstream_headers, blocked)
                return

        # Too many unapproved final answers.
        last_result = result.get("text", "") if "result" in locals() else "Unknown verifier issue."
        blocked = synthetic_block_response(
            forwarded_request,
            "The agent could not produce a verifier-approved answer within the configured review limit.\n\n"
            f"Last verifier result:\n{last_result}",
        )
        send_buffered_response(self, 200, upstream_headers, blocked)

    def log_message(self, format, *args):
        # Safe under pythonw.exe: do not write to a missing stderr console.
        return


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    trace_event({
        "event": "proxy_starting",
        "listen": f"127.0.0.1:{PORT}",
        "upstream": UPSTREAM,
        "config_file": str(CONFIG_FILE),
    }, force=True)

    try:
        server = ProxyServer(("127.0.0.1", PORT), ProxyHandler)
    except OSError as error:
        trace_event({
            "event": "proxy_start_error",
            "listen": f"127.0.0.1:{PORT}",
            "error": repr(error),
        }, force=True)
        raise

    trace_event({
        "event": "proxy_started",
        "listen": f"127.0.0.1:{PORT}",
        "upstream": UPSTREAM,
        "advisor_mode": load_config().get("advisor_mode"),
    }, force=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        trace_event({
            "event": "proxy_runtime_error",
            "error": repr(error),
        }, force=True)
        raise
    finally:
        server.server_close()
        trace_event({"event": "proxy_stopped"}, force=True)
