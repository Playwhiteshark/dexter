"""Load versioned Dexter profiles from ``Runtime/Profiles/<id>/``.

The loader owns the disk representation. Callers work only with normalized,
immutable ``AgentProfile`` objects and do not need to know which profile fields
came from JSON versus sibling resources such as Markdown instruction files.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from Runtime.Profiles.profile import (
    AgentProfile,
    AppearanceConfig,
    ModelConfig,
    ToolsConfig,
    VoiceConfig,
)

SCHEMA_VERSION = 1
PROFILES_ROOT = Path(__file__).resolve().parent
PROFILE_ID_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?"
)


class ProfileLoadError(RuntimeError):
    """A profile exists but could not be read or decoded."""


class ProfileValidationError(ValueError):
    """A profile or one of its resources violates the profile contract."""


def _validate_profile_id(profile_id: str) -> str:
    if not isinstance(profile_id, str):
        raise TypeError("profile_id must be a string")
    if PROFILE_ID_PATTERN.fullmatch(profile_id) is None:
        raise ValueError(f"Invalid profile ID: {profile_id!r}")
    return profile_id


def _resolve_profile_dir(profile_id: str) -> Path:
    """Return the existing direct child directory for ``profile_id``."""
    profile_id = _validate_profile_id(profile_id)
    root = PROFILES_ROOT.resolve()
    profile_dir = (root / profile_id).resolve()

    # The identifier grammar already excludes separators/traversal. Keep this
    # containment check as a second boundary so a profile-directory symlink
    # cannot silently redirect profile loading outside Runtime/Profiles.
    if profile_dir.parent != root:
        raise ProfileValidationError(
            f"Profile directory escapes Profiles root: {profile_id!r}"
        )
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile not found: {profile_id!r}")
    return profile_dir


def _resolve_profile_resource(profile_dir: Path, resource: str) -> Path:
    """Resolve one manifest resource without allowing profile-root escape."""
    if not isinstance(resource, str):
        raise ProfileValidationError("Profile resource path must be a string")
    if not resource.strip():
        raise ProfileValidationError("Profile resource path cannot be empty")

    relative = Path(resource)
    if relative.is_absolute():
        raise ProfileValidationError(
            f"Profile resource must be relative: {resource!r}"
        )

    root = profile_dir.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ProfileValidationError(
            f"Profile resource escapes profile directory: {resource!r}"
        )
    if not target.is_file():
        raise FileNotFoundError(f"Profile resource not found: {resource!r}")
    return target


def _read_text(path: Path, *, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileLoadError(f"Could not read {description}: {path}") from exc


def _read_manifest(path: Path, *, profile_id: str) -> Mapping[str, Any]:
    raw = _read_text(path, description=f"profile {profile_id!r} manifest")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProfileLoadError(
            f"Invalid JSON in profile {profile_id!r} at "
            f"line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(value, dict):
        raise ProfileValidationError(
            f"Profile {profile_id!r} manifest must contain a JSON object"
        )
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    location: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ProfileValidationError(
            f"Unknown field(s) in {location}: {names}"
        )


def _require_object(
    value: Mapping[str, Any],
    key: str,
    *,
    location: str,
) -> Mapping[str, Any]:
    if key not in value:
        raise ProfileValidationError(f"Missing required field {location}.{key}")
    child = value[key]
    if not isinstance(child, dict):
        raise ProfileValidationError(f"{location}.{key} must be an object")
    return child


def _require_string(
    value: Mapping[str, Any],
    key: str,
    *,
    location: str,
) -> str:
    if key not in value:
        raise ProfileValidationError(f"Missing required field {location}.{key}")
    child = value[key]
    if not isinstance(child, str) or not child.strip():
        raise ProfileValidationError(
            f"{location}.{key} must be a non-empty string"
        )
    return child.strip()


def _optional_string(
    value: Mapping[str, Any],
    key: str,
    *,
    location: str,
) -> str | None:
    child = value.get(key)
    if child is None:
        return None
    if not isinstance(child, str) or not child.strip():
        raise ProfileValidationError(
            f"{location}.{key} must be null or a non-empty string"
        )
    return child.strip()


def _parse_profile(
    profile_id: str,
    profile_dir: Path,
    data: Mapping[str, Any],
) -> AgentProfile:
    _reject_unknown_fields(
        data,
        allowed={
            "schema_version",
            "id",
            "name",
            "model",
            "instructions",
            "voice",
            "tools",
            "appearance",
        },
        location="profile",
    )

    schema_version = data.get("schema_version")
    if type(schema_version) is not int:
        raise ProfileValidationError("profile.schema_version must be an integer")
    if schema_version != SCHEMA_VERSION:
        raise ProfileValidationError(
            f"Unsupported profile schema_version {schema_version!r}; "
            f"expected {SCHEMA_VERSION}"
        )

    manifest_id = _require_string(data, "id", location="profile")
    _validate_profile_id(manifest_id)
    if manifest_id != profile_id:
        raise ProfileValidationError(
            f"Profile ID mismatch: requested {profile_id!r}, "
            f"manifest declares {manifest_id!r}"
        )

    name = _require_string(data, "name", location="profile")

    model_data = _require_object(data, "model", location="profile")
    _reject_unknown_fields(
        model_data,
        allowed={"provider", "model"},
        location="profile.model",
    )
    model = ModelConfig(
        provider=_require_string(model_data, "provider", location="profile.model"),
        model=_require_string(model_data, "model", location="profile.model"),
    )

    voice_data = _require_object(data, "voice", location="profile")
    _reject_unknown_fields(
        voice_data,
        allowed={"backend", "preset", "speaker", "emotion"},
        location="profile.voice",
    )
    voice = VoiceConfig(
        backend=_require_string(voice_data, "backend", location="profile.voice"),
        preset=_require_string(voice_data, "preset", location="profile.voice"),
        speaker=_optional_string(voice_data, "speaker", location="profile.voice"),
        emotion=_optional_string(voice_data, "emotion", location="profile.voice"),
    )

    tools_data = _require_object(data, "tools", location="profile")
    _reject_unknown_fields(
        tools_data,
        allowed={"preset"},
        location="profile.tools",
    )
    tools = ToolsConfig(
        preset=_require_string(tools_data, "preset", location="profile.tools")
    )

    appearance_data = _require_object(data, "appearance", location="profile")
    _reject_unknown_fields(
        appearance_data,
        allowed={"accent"},
        location="profile.appearance",
    )
    appearance = AppearanceConfig(
        accent=_require_string(
            appearance_data,
            "accent",
            location="profile.appearance",
        )
    )

    instructions_resource = _require_string(
        data,
        "instructions",
        location="profile",
    )
    instructions_path = _resolve_profile_resource(
        profile_dir,
        instructions_resource,
    )
    instructions = _read_text(
        instructions_path,
        description=f"profile {profile_id!r} instructions",
    )

    return AgentProfile(
        id=manifest_id,
        name=name,
        model=model,
        instructions=instructions,
        voice=voice,
        tools=tools,
        appearance=appearance,
    )


def load_profile(profile_id: str) -> AgentProfile:
    """Load and validate one profile by stable profile ID."""
    profile_dir = _resolve_profile_dir(profile_id)
    manifest_path = _resolve_profile_resource(profile_dir, "profile.json")
    data = _read_manifest(manifest_path, profile_id=profile_id)
    return _parse_profile(profile_id, profile_dir, data)


def list_profile_ids() -> tuple[str, ...]:
    """List syntactically valid profile directories containing a manifest."""
    try:
        entries = tuple(PROFILES_ROOT.iterdir())
    except OSError as exc:
        raise ProfileLoadError(f"Could not list profile root: {PROFILES_ROOT}") from exc

    profile_ids: list[str] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        profile_id = entry.name
        if PROFILE_ID_PATTERN.fullmatch(profile_id) is None:
            continue
        try:
            profile_dir = _resolve_profile_dir(profile_id)
            _resolve_profile_resource(profile_dir, "profile.json")
        except (FileNotFoundError, ProfileValidationError):
            continue
        profile_ids.append(profile_id)
    return tuple(sorted(profile_ids))
