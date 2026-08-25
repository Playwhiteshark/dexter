from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT / "goose_ollama_capture"

MODEL = "gemma4-jarvis:12b"
NATIVE_URL = "http://127.0.0.1:11434/api/chat"
OPENAI_URL = "http://127.0.0.1:11434/v1/chat/completions"
SEED = 424242
QUESTION = "Why does Windows use so much RAM even when I'm not doing much?"

MINIMAL_DEXTER = """You are Dexter, Ahmed's local computer assistant.

Speak naturally, familiarly, and with a dry, understated personality.

Default to a short reply: usually 1-3 sentences. Expand only when Ahmed asks for detail or the question genuinely cannot be answered well that briefly.

Use ordinary conversational prose by default. Acknowledge completed actions naturally rather than like status telemetry. Be direct when Ahmed is mistaken. Avoid generic assistant pleasantries."""


def latest_capture() -> Path:
    candidates = sorted(CAPTURE_DIR.glob("*_request.json"))
    if not candidates:
        raise RuntimeError(
            f"No *_request.json files found in {CAPTURE_DIR}. "
            "Run the Goose capture first."
        )
    return candidates[-1]


def metrics(text: str) -> dict:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return {
        "words": len(re.findall(r"\S+", text)),
        "heading": any(re.match(r"^#{1,6}\s+", x) for x in lines),
        "numbered": any(re.match(r"^\d+[.)]\s+", x) for x in lines),
        "bullets": any(re.match(r"^[-*+]\s+", x) for x in lines),
        "bold_label": any(re.match(r"^\*\*[^*]+\*\*:?", x) for x in lines),
    }


def post_json(url: str, payload: dict) -> tuple[dict, float]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc
    return json.loads(raw), time.perf_counter() - started


def native_payload(system: str, tools: list, think: bool = False) -> dict:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": QUESTION},
        ],
        "tools": tools,
        "stream": False,
        "think": think,
        "options": {
            "num_ctx": 16384,
            "temperature": 1.0,
            "seed": SEED,
        },
    }


def v1_payload(system: str, tools: list, reasoning_none: bool) -> dict:
    p = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": QUESTION},
        ],
        "tools": tools,
        "temperature": 1.0,
        "seed": SEED,
        "stream": False,
        "options": {"num_ctx": 16384},
    }
    if reasoning_none:
        p["reasoning_effort"] = "none"
    return p


def extract_native(out: dict) -> tuple[str, object]:
    msg = out.get("message") or {}
    return msg.get("content") or "", msg.get("thinking")


def extract_v1(out: dict) -> tuple[str, object]:
    choices = out.get("choices") or []
    msg = (choices[0].get("message") if choices else {}) or {}
    thinking = (
        msg.get("reasoning")
        or msg.get("thinking")
        or msg.get("reasoning_content")
    )
    return msg.get("content") or "", thinking


def main():
    cap_path = latest_capture()
    captured = json.loads(cap_path.read_text(encoding="utf-8"))
    full_system = captured["messages"][0]["content"]
    tools = captured.get("tools") or []

    marker = "# Additional Instructions:"
    if marker not in full_system:
        raise RuntimeError("Could not split Goose base from Additional Instructions.")
    goose_base, additional = full_system.split(marker, 1)
    goose_base = goose_base.rstrip()
    additional = additional.strip()

    print("=" * 86)
    print("DEXTER PROMPT / ENDPOINT / THINKING MATRIX")
    print("=" * 86)
    print(f"Capture          : {cap_path.name}")
    print(f"Tools            : {len(tools)}")
    print(f"Full system chars: {len(full_system)}")
    print(f"Goose base chars : {len(goose_base)}")
    print(f"Additional chars : {len(additional)}")
    print(f"Question         : {QUESTION}")
    print(f"Seed             : {SEED}")
    print()
    print("This is a diagnostic. It executes NO tools.")

    conditions = [
        (
            "A_NATIVE_MINIMAL_THINK_FALSE",
            NATIVE_URL,
            native_payload(MINIMAL_DEXTER, tools, think=False),
            extract_native,
        ),
        (
            "B_V1_MINIMAL_GOOSE_LIKE",
            OPENAI_URL,
            v1_payload(MINIMAL_DEXTER, tools, reasoning_none=False),
            extract_v1,
        ),
        (
            "C_V1_MINIMAL_REASONING_NONE",
            OPENAI_URL,
            v1_payload(MINIMAL_DEXTER, tools, reasoning_none=True),
            extract_v1,
        ),
        (
            "D_NATIVE_FULL_CAPTURE_THINK_FALSE",
            NATIVE_URL,
            native_payload(full_system, tools, think=False),
            extract_native,
        ),
        (
            "E_V1_FULL_CAPTURE_GOOSE_LIKE",
            OPENAI_URL,
            v1_payload(full_system, tools, reasoning_none=False),
            extract_v1,
        ),
        (
            "F_V1_FULL_CAPTURE_REASONING_NONE",
            OPENAI_URL,
            v1_payload(full_system, tools, reasoning_none=True),
            extract_v1,
        ),
        (
            "G_V1_GOOSE_BASE_PLUS_MINIMAL",
            OPENAI_URL,
            v1_payload(goose_base + "\n\n# Additional Instructions:\n\n" + MINIMAL_DEXTER, tools, reasoning_none=False),
            extract_v1,
        ),
        (
            "H_V1_ADDITIONAL_ONLY",
            OPENAI_URL,
            v1_payload(additional, tools, reasoning_none=False),
            extract_v1,
        ),
    ]

    results = []

    for name, url, payload, extractor in conditions:
        print("\n" + "#" * 86)
        print(name)
        print("#" * 86)
        out, elapsed = post_json(url, payload)
        text, thinking = extractor(out)
        m = metrics(text)

        print("\nASSISTANT:")
        print(text or "<no visible content>")
        print("\nMETRICS:")
        print(
            f"words={m['words']} | heading={m['heading']} | "
            f"numbered={m['numbered']} | bullets={m['bullets']} | "
            f"bold_label={m['bold_label']} | elapsed={elapsed:.2f}s | "
            f"thinking_field_present={bool(thinking)}"
        )

        usage = out.get("usage")
        if usage:
            print("USAGE:", json.dumps(usage, ensure_ascii=False))

        results.append(
            {
                "condition": name,
                "url": url,
                "request": payload,
                "response": out,
                "metrics": m,
                "thinking_field_present": bool(thinking),
                "elapsed_seconds": elapsed,
            }
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = ROOT / f"prompt_endpoint_matrix_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "capture_file": str(cap_path),
                "full_system_chars": len(full_system),
                "goose_base_chars": len(goose_base),
                "additional_chars": len(additional),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 86)
    print(f"Saved: {out_path}")
    print("=" * 86)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
