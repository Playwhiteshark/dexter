import json
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4-jarvis:12b"
SEED = 424242

PROMPTS = [
    "What's fifteen percent of 200?",
    "Why does Windows use so much RAM even when I'm not doing much?",
    "you alive?",
    "Explain why SSDs slow down when they're nearly full.",
]

MINIMAL_DEXTER = """You are Dexter, Ahmed's local computer assistant.

Speak naturally, familiarly, and with a dry, understated personality.

Default to a short reply: usually 1-3 sentences. Expand only when Ahmed asks for detail or the question genuinely cannot be answered well that briefly.

Use ordinary conversational prose by default. Acknowledge completed actions naturally rather than like status telemetry. Be direct when Ahmed is mistaken. Avoid generic assistant pleasantries."""

HARD_SHAPE = """You are Dexter, Ahmed's local computer assistant.

Speak naturally and familiarly, with a dry, understated personality.

For this diagnostic only, every answer must be one or two sentences, no more than 50 words total. Use plain prose only. Do not use headings, lists, bullets, numbered sections, markdown section labels, summaries, or closing invitations. Answer the question directly and stop."""

CONDITIONS = [
    ("A_BASELINE_NO_SYSTEM", None),
    ("B_MINIMAL_DEXTER", MINIMAL_DEXTER),
    ("C_HARD_SHAPE_DIAGNOSTIC", HARD_SHAPE),
]


def call_ollama(system_prompt, user_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {"seed": SEED},
    }

    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_CHAT_URL}. "
            f"Make sure Ollama is running. Original error: {e}"
        ) from e

    elapsed = time.perf_counter() - started
    return payload, json.loads(raw), elapsed


def analyze_text(text):
    lines = text.splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    words = re.findall(r"\S+", text)

    return {
        "word_count": len(words),
        "line_count": len(lines),
        "nonempty_line_count": len(nonempty),
        "has_markdown_heading": any(re.match(r"^#{1,6}\s+", line) for line in nonempty),
        "has_numbered_list": any(re.match(r"^\d+[.)]\s+", line) for line in nonempty),
        "has_bullet_list": any(re.match(r"^[-*+]\s+", line) for line in nonempty),
        "has_horizontal_rule": any(re.match(r"^(-{3,}|\*{3,}|_{3,})$", line) for line in nonempty),
        "has_bold_leading_label": any(re.match(r"^\*\*[^*]+\*\*:?", line) for line in nonempty),
    }


def main():
    print("=" * 78)
    print("DEXTER PERSONALITY ISOLATION PROBE")
    print("=" * 78)
    print(f"Endpoint : {OLLAMA_CHAT_URL}")
    print(f"Model    : {MODEL}")
    print(f"Seed     : {SEED}")
    print("Tools    : NONE")
    print("Thinking : false")
    print()
    print("This talks DIRECTLY to Ollama on port 11434. Goose, Dexter Host, Gateway,")
    print("AGENTS.md, and tool schemas are not part of these requests.")

    all_results = []

    for condition_name, system_prompt in CONDITIONS:
        print("\n" + "#" * 78)
        print(condition_name)
        print("#" * 78)
        print("\nSYSTEM PROMPT:")
        print(system_prompt if system_prompt else "<none>")

        for i, prompt in enumerate(PROMPTS, 1):
            print("\n" + "-" * 78)
            print(f"[{condition_name} / {i}] USER: {prompt}")

            payload, result, elapsed = call_ollama(system_prompt, prompt)
            text = (result.get("message") or {}).get("content", "")
            metrics = analyze_text(text)

            print("\nASSISTANT:")
            print(text)
            print("\nMETRICS:")
            print(
                f"words={metrics['word_count']} | "
                f"nonempty_lines={metrics['nonempty_line_count']} | "
                f"heading={metrics['has_markdown_heading']} | "
                f"numbered={metrics['has_numbered_list']} | "
                f"bullets={metrics['has_bullet_list']} | "
                f"bold_label={metrics['has_bold_leading_label']} | "
                f"elapsed={elapsed:.2f}s | "
                f"prompt_tokens={result.get('prompt_eval_count')} | "
                f"output_tokens={result.get('eval_count')}"
            )

            all_results.append({
                "condition": condition_name,
                "system_prompt": system_prompt,
                "user_prompt": prompt,
                "request_payload": payload,
                "response": result,
                "analysis": metrics,
                "wall_time_seconds": elapsed,
            })

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(f"personality_probe_results_{stamp}.json")
    out_path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "endpoint": OLLAMA_CHAT_URL,
                "seed": SEED,
                "results": all_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print(f"Saved full results to: {out_path.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
