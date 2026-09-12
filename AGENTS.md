# Agent instructions — Dexter

Dexter is a standalone project with its own architecture and lifecycle. This repository is the source of truth for Dexter; do not move its implementation into Local Windows Tools or Personal Operations merely because Projects Index can catalogue it.

Before changing Dexter, read:

1. `README.md` for project ownership and storage/cleanup rules;
2. the README/runtime documentation for the subsystem being changed;
3. `Runtime/Host/RUNTIME.md` for Host process/dependency/state ownership when Host or voice behavior is involved;
4. `Runtime/Voice/README.md` before changing the production voice stack;
5. existing diagnostics/development notes before repeating performance or memory experiments.

`main` is the canonical long-lived branch. Use temporary feature/fix branches and return durable current truth to `main`.

## Required behavior

- Keep durable architecture, runtime, recovery and measurement conclusions in this repository rather than only in chat.
- Never commit passwords, API tokens, cookies, private keys, `.env` secrets, local trusted-root policy, browser/session credentials, model-auth data, or machine-specific private configuration.
- Do not weaken Windows security globally to make Dexter work. Preserve the existing approach of avoiding unnecessary blocked dependency paths rather than disabling Smart App Control / Windows Application Control.
- Preserve the root `uv` / Python 3.13 dependency contract unless a deliberate dependency migration changes it.
- Treat the root `.venv` as reproducible environment state, not source.
- Treat `%LOCALAPPDATA%\Dexter\state\computer_memory.json` as persistent user-owned application data, not disposable cache.
- Treat `%LOCALAPPDATA%\Dexter\Voice\neutts2e\cache` conservatively: the current runtime documentation describes model/auth/cache material there, so it is not generic automatic cleanup even though it is under a cache path.
- `Runtime/Tools/dexter-tools/local_config.py` is machine-local policy and must remain uncommitted; `local_config.example.py` is the portable example.
- Use GitHub Issues for unfinished product work. Update current-truth docs when behavior changes instead of leaving a closed issue or chat transcript as the only explanation.

## External writes and storage ownership

Every new persistent/generated write must be classified and documented as one of:

- logs;
- cache;
- runtime/machine state;
- configuration;
- retained/generated artifacts;
- user-owned data.

`.local-windows-tools.json` is an optional machine-readable declaration consumed by Projects Index. It may contain only portable intrinsic Dexter facts: repository-relative paths or portable environment-variable paths, cleanup classification, and other supported typed metadata. Do not put absolute checkout paths or personal machine-management preferences in it.

Use the same semantics as the project README:

- `none` means Dexter explicitly creates none in that category;
- `declared` means the relevant ownership paths are fully documented for that category;
- `unknown` means disclosure is incomplete.

Do not change `unknown` to `none` just to clear a warning. Only genuinely disposable logs/cache may be marked `cleanup: safe`; user data, configuration, state, credentials and valuable artifacts must remain non-automatic cleanup.

Dexter currently leaves generated development artifacts `unknown` because several producers still write to scattered/patterned ignored paths. Normalize or enumerate those outputs before claiming that category is complete.

## Before committing

- run the tests/checks documented by the subsystem you changed;
- verify secrets, local config, generated diagnostics, model files and caches are not staged;
- update relevant docs and `.local-windows-tools.json` together if storage/write behavior changed;
- keep user-visible/runtime launch documentation consistent with the actual Host entry point;
- record unresolved follow-up in a GitHub Issue.
