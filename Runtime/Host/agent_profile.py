"""Declarative Dexter agent/profile configuration.

Profiles describe identity-level choices such as model, Voice preset, tools preset,
and appearance. Machine/runtime settings (ports, hardware tuning, microphone,
filesystem roots, etc.) deliberately remain outside the profile boundary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROFILE_SCHEMA_VERSION = 1
DEFAULT_PROFILE_ID = "dexter"
PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HEX_ACCENT_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class ProfileError(ValueError):
    """Raised when an agent profile is missing, malformed, or unsafe."""


@dataclass(frozen=True, slots=True)
class ModelProfile:
    provider: str
    name: str
    reasoning_effort: str


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    backend: str
    preset: str
    speaker: str
    emotion: str


@dataclass(frozen=True, slots=True)
class ToolsProfile:
    preset: str


@dataclass(frozen=True, slots=True)
class AppearanceProfile:
    accent: str | None


@dataclass(frozen=True, slots=True)
class AgentProfile:
    schema_version: int
    profile_id: str
    display_name: str
    instructions_file: Path | None
    model: ModelProfile
    voice: VoiceProfile
    tools: ToolsProfile
    appearance: AppearanceProfile


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileError(f"{label} must be a JSON object.")
    return value


def _exact_keys(value: dict[str, Any], *, required: set[str], label: str) -> None:
    missing = required - value.keys()
    extra = value.keys() - required
    if missing:
        raise ProfileError(f"{label} is missing: {', '.join(sorted(missing))}.")
    if extra:
        raise ProfileError(f"{label} has unknown fields: {', '.join(sorted(extra))}.")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{label} must be a non-empty string.")
    return value.strip()


def _resolve_instructions(profile_dir: Path, value: Any) -> Path | None:
    if value is None:
        return None
    relative = Path(_string(value, "instructions_file"))
    if relative.is_absolute():
        raise ProfileError("instructions_file must be relative to its profile directory.")

    resolved = (profile_dir / relative).resolve()
    profile_root = profile_dir.resolve()
    try:
        resolved.relative_to(profile_root)
    except ValueError as exc:
        raise ProfileError("instructions_file must stay inside its profile directory.") from exc

    if not resolved.is_file():
        raise ProfileError(f"instructions_file does not exist: {resolved}")
    return resolved


def load_profile(
    profile_id: str = DEFAULT_PROFILE_ID,
    *,
    profiles_dir: Path = PROFILES_DIR,
) -> AgentProfile:
    """Load and validate one immutable profile by id."""

    selected = _string(profile_id, "profile_id")
    if _PROFILE_ID_RE.fullmatch(selected) is None:
        raise ProfileError(
            "profile_id may contain only lowercase letters, numbers, '-' and '_'."
        )

    profile_dir = Path(profiles_dir) / selected
    profile_path = profile_dir / "profile.json"
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"Profile not found: {selected}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(
            f"Invalid JSON in {profile_path.name}: line {exc.lineno}, column {exc.colno}."
        ) from exc

    root = _object(raw, "profile")
    _exact_keys(
        root,
        required={
            "schema_version",
            "id",
            "display_name",
            "instructions_file",
            "model",
            "voice",
            "tools",
            "appearance",
        },
        label="profile",
    )

    schema_version = root["schema_version"]
    if schema_version != PROFILE_SCHEMA_VERSION:
        raise ProfileError(
            f"Unsupported profile schema_version {schema_version!r}; "
            f"expected {PROFILE_SCHEMA_VERSION}."
        )

    file_profile_id = _string(root["id"], "id")
    if file_profile_id != selected:
        raise ProfileError(
            f"Profile id mismatch: directory is {selected!r}, file says {file_profile_id!r}."
        )

    model = _object(root["model"], "model")
    _exact_keys(
        model,
        required={"provider", "name", "reasoning_effort"},
        label="model",
    )

    voice = _object(root["voice"], "voice")
    _exact_keys(
        voice,
        required={"backend", "preset", "speaker", "emotion"},
        label="voice",
    )

    tools = _object(root["tools"], "tools")
    _exact_keys(tools, required={"preset"}, label="tools")

    appearance = _object(root["appearance"], "appearance")
    _exact_keys(appearance, required={"accent"}, label="appearance")
    accent = appearance["accent"]
    if accent is not None and (
        not isinstance(accent, str) or _HEX_ACCENT_RE.fullmatch(accent) is None
    ):
        raise ProfileError("appearance.accent must be null or a '#RRGGBB' color.")

    return AgentProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=selected,
        display_name=_string(root["display_name"], "display_name"),
        instructions_file=_resolve_instructions(profile_dir, root["instructions_file"]),
        model=ModelProfile(
            provider=_string(model["provider"], "model.provider"),
            name=_string(model["name"], "model.name"),
            reasoning_effort=_string(
                model["reasoning_effort"],
                "model.reasoning_effort",
            ),
        ),
        voice=VoiceProfile(
            backend=_string(voice["backend"], "voice.backend"),
            preset=_string(voice["preset"], "voice.preset"),
            speaker=_string(voice["speaker"], "voice.speaker"),
            emotion=_string(voice["emotion"], "voice.emotion"),
        ),
        tools=ToolsProfile(
            preset=_string(tools["preset"], "tools.preset"),
        ),
        appearance=AppearanceProfile(accent=accent),
    )
