"""Immutable runtime profile domain model.

Profiles describe assistant identity/capability choices. Machine-specific runtime
settings (ports, GPU tuning, microphone selection, filesystem roots, etc.) stay
outside this model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    """Profile-facing Voice selection.

    ``backend`` selects the implementation adapter while ``preset`` names a
    backend-owned configuration. ``speaker`` and ``emotion`` are optional
    because not every TTS backend exposes those concepts directly; a backend
    such as a reference-voice/cloning engine can keep its backend-specific
    details behind the preset instead of forcing them into the shared profile
    schema.
    """

    backend: str
    preset: str
    speaker: str | None = None
    emotion: str | None = None


@dataclass(frozen=True, slots=True)
class ToolsConfig:
    preset: str


@dataclass(frozen=True, slots=True)
class AppearanceConfig:
    accent: str


@dataclass(frozen=True, slots=True)
class AgentProfile:
    id: str
    name: str
    model: ModelConfig
    instructions: str
    voice: VoiceConfig
    tools: ToolsConfig
    appearance: AppearanceConfig
