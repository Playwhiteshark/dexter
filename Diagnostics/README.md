# Dexter diagnostics

This directory contains current operational diagnostics for Dexter.

Keep this directory small. Short-lived experiments, completed A/B investigations,
and superseded probes belong in `Development/Archive`, while reusable benchmark
suites belong in `Harnesses`.

## Current tools

- `audit_jarvis_config.ps1` — inspect the machine/runtime configuration used by Dexter.
- `audit_runtime_performance.ps1` — capture a repeatable full-runtime performance snapshot.
- `tts_production_worker_tree_memory.py` — measure the complete current production Voice worker process tree.
- `tts_production_worker_identical_repeat_memory.py` — reproduce the residual identical-request Voice memory growth tracked separately.

Generated reports and benchmark outputs are deliberately ignored by Git. Important
results should be summarized in the relevant README, handoff, or GitHub issue rather
than committed as raw machine-specific output.

Historical Voice optimization experiments are under:

`Development/Archive/TTS/`

The authoritative current Voice architecture and benchmark conclusions live in:

`Runtime/Voice/README.md`