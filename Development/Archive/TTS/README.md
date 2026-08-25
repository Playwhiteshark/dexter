# TTS optimization archive

Historical experiments from the August 2026 NeuTTS optimization work.

These files are retained for provenance: they show the controlled experiments that
led to Dexter's production Voice configuration. They are not part of the current
diagnostic or benchmark surface and are not expected to remain maintained as the
runtime evolves.

Some archived scripts contain paths or assumptions from their original location and
may therefore require adjustment if deliberately rerun.

Key conclusions and the current production architecture are maintained in:

- `Runtime/Voice/README.md`
- `Development/HANDOFF_TTS_OPTIMIZATION_2026-08-26.md`
- GitHub Voice issues and commit history

Notable superseded artifacts include:

- component-level backbone, codec, Transformers and Perth A/B tests;
- early memory probes;
- the original single-PID production-worker memory test, which was later shown to
  measure only the small Windows Python launcher rather than the complete worker tree;
- temporary seed/decimal experiments;
- earlier repeated-request investigations superseded by the current production-backend
  reproducer.

Raw generated benchmark output is not kept in Git.