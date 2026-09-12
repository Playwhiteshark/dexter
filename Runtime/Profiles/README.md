# Dexter Profiles

Profiles are versioned assistant-identity manifests. A profile is stored as a
directory under `Runtime/Profiles/`:

```text
Runtime/Profiles/
  dexter/
    profile.json
    instructions.md
```

`load_profile("dexter")` is the public storage boundary. Callers receive an
immutable `AgentProfile`; they do not need to know which values came directly
from the JSON manifest and which came from sibling resources.

## Schema v1

```json
{
  "schema_version": 1,
  "id": "dexter",
  "name": "Dexter",
  "model": {
    "provider": "ollama",
    "model": "gemma4-jarvis:12b"
  },
  "instructions": "instructions.md",
  "voice": {
    "backend": "neutts2e",
    "preset": "production",
    "speaker": "paul",
    "emotion": "neutral"
  },
  "tools": { "preset": "default" },
  "appearance": { "accent": "default" }
}
```

The manifest's `instructions` value is a relative resource path. The loader
reads that file and exposes its complete text as `AgentProfile.instructions`.
All referenced resources are resolved inside their owning profile directory;
absolute paths and `..` escapes are rejected.

## Voice boundary

The shared profile schema deliberately does not model backend-specific runtime
knobs. `backend` selects a Voice adapter and `preset` selects a configuration
owned by that adapter. `speaker` and `emotion` are optional conveniences for
engines such as NeuTTS-2E that expose those concepts directly.

This keeps profiles compatible with a different future TTS backend (including a
reference-voice/cloning backend) without adding its model paths, reference audio,
device placement, or tuning parameters to the identity schema. Those details
belong behind the backend/preset implementation.

## What does not belong in a profile

Machine/runtime configuration stays outside profiles, including Ollama host and
port, GPU/NPU/CPU tuning, batch sizes, microphone selection, PTT binding,
filesystem permission roots, cache paths, and similar deployment details.

Adding another profile requires only another profile directory. Changing the
meaning of the profile schema requires a deliberate schema-version change and
corresponding loader/domain update.
