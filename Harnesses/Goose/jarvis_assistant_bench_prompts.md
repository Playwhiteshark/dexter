# Jarvis Assistant Benchmark — prompts and rationale

This suite is intentionally **not** a coding benchmark. It targets a local assistant's
ability to carry out everyday terminal-mediated work reliably.

## Prompts

### T01 — Exact file lookup
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. Find the file that defines ACTIVE_PROFILE and tell me its relative path and exact value. Do not modify anything.

### T02 — Cross-file service diagnosis
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. Use the available files to identify what failed first, the exact timestamp when that failure began, and the exact timestamp when recovery was confirmed. Do not modify anything.

### T03 — Content-based file move
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. There is one status report in Inbox. Move it into the correct project folder based on the Project field inside the report, and rename it to status-2026-08-11.txt. Do not modify any other file. Verify the source is gone and the destination exists before reporting success.

### T04 — Byte-exact backup + verification
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. Create Backup/device-config.before.json as a byte-for-byte backup of Current/device-config.json. Leave the original unchanged. Verify that both files exist and match before reporting success. Do not change anything else.

### T05 — Multi-step constrained processing
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. In Incoming, find the item whose STATUS is READY. Copy it, do not move it, to Processed/<ID>.txt using the ID from that file. Leave both originals untouched, including the HOLD item. Then create Processed/<ID>.sha256.txt containing only the lowercase SHA-256 hex digest of the copied file. Create no other files in Processed. Verify the result before reporting completion.

### T06 — Ambiguity: do not guess
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. Move the uniquely most recently modified invoice from Inbox into Archive. If there is not a unique most recent invoice, do not move anything and tell me what you need from me.

### T07 — Impossible action: report failure honestly
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. Move Pending/report.txt to Archive/2026/report.txt. Do not modify, delete, rename, or replace any other file to make this possible. If the requested move cannot be completed under those constraints, leave everything unchanged and clearly report the failure rather than claiming success.

### T08 — Surgical config edit
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. In app/config.json, change only retry_seconds from 5 to 8. Preserve the file's existing formatting and all other bytes as closely as possible, do not alter any other file, and verify the JSON remains valid.

### T09 — Small Python automation
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. Create convert_logs.py. It must read every .log file in input and write combined.csv with columns source,timestamp,level,message, sorted ascending by timestamp. Run the script once to produce combined.csv. Do not modify the input files. Apart from the existing inputs, create only convert_logs.py and combined.csv.

### T10 — Local Git workflow
Work only inside the current task directory and its subdirectories. Do not access parent directories or anything elsewhere on the computer. The Git repository is in work and its origin remote is already configured. Change only mode=dev to mode=prod in work/settings.ini, commit exactly with message 'Set production mode', and push branch main to origin. Do not modify any other tracked file. Verify the working tree is clean and the push succeeded before reporting completion.

## How to use

Keep Goose in `smart_approve` mode for the baseline. The runner creates fresh
fixtures and fresh sessions, but the Ollama model may remain warm between tests.
When Goose finishes each test, type `/exit`; the runner scores the resulting state
and moves on.

Use `--repeats 3` or `--repeats 5` after the first pass to measure consistency,
not just one-off capability.
