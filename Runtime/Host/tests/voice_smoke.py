r"""Manual smoke test for Runtime.Voice.stt."""

from __future__ import annotations

import sys
import time
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    current = start.resolve()

    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate

    raise RuntimeError("Could not find Jarvis repository root.")


ROOT = find_repo_root(Path(__file__).parent)

# Make Jarvis/ importable, allowing Runtime.Voice.stt to resolve.
sys.path.insert(0, str(ROOT))

from Runtime.Voice.stt import PromptResult, VoiceConfig, VoiceInput, VoiceState


def on_partial(session_id: int, text: str) -> None:
    print(
        f"\r[S{session_id:03d}] PARTIAL: {text:<100}",
        end="",
        flush=True,
    )


def on_prompt(result: PromptResult) -> None:
    print()
    print(f"[S{result.session_id:03d}] PROMPT: {result.text}")
    print(
        f"[S{result.session_id:03d}] "
        f"AUDIO: {result.captured_audio_seconds:.2f}s | "
        f"TAIL: {result.release_to_prompt_ms:.0f}ms"
    )
    print()


def on_state(state: VoiceState) -> None:
    print(f"\n[STATE] {state.value}")


def on_error(message: str) -> None:
    print(f"\n[ERROR] {message}")


def main() -> None:
    debug_dir = ROOT / "Runtime" / "Voice" / "ptt_debug"

    print(f"Repo root: {ROOT}")
    print(f"Debug WAVs: {debug_dir}")

    voice = VoiceInput(
        config=VoiceConfig(
            debug_wav_dir=debug_dir,
        ),
        on_partial=on_partial,
        on_prompt=on_prompt,
        on_state=on_state,
        on_error=on_error,
    )

    try:
        voice.start()

        print()
        print("VoiceInput running.")
        print("Hold Ctrl+/ to speak; release to finalize.")
        print("Ctrl+C exits.")
        print()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        voice.stop()


if __name__ == "__main__":
    main()