# Bounded semantic decisions — design note

Status: **candidate architecture / not yet implemented**

This note records a possible reusable boundary for places where Dexter has already gathered a bounded set of real, safe candidates but deterministic code alone cannot reliably decide which candidate best matches the user's intent or the current UI/system state.

The initial model/provider being considered is TypeSafe Jev, but the architecture must not depend on Jev specifically. The product-facing concept is a **bounded semantic decision provider**. Jev would be one implementation behind that boundary and should be replaceable without rewriting resource resolution, watchers, browser/UI automation, or tool routing.

## Why this exists

Dexter currently has a deliberate split:

- deterministic code owns permissions, discovery, exact identities, execution, verification, and postconditions;
- the main Dexter/Goose model handles open-ended language, reasoning, planning, and values that must be generated rather than selected;
- saved-resource and app resolution already prefer exact names/aliases and refuse ambiguity rather than guessing.

That remains the default.

There is, however, a recurring middle class of problem:

1. trusted/deterministic code can discover the **actual possibilities**;
2. several possibilities may all be plausible from natural speech or current context;
3. choosing among them needs semantic judgment, but does not require a full generative-agent turn.

Examples:

- "open that Dexter work" when the saved `Local AI` / `Jarvis` VS Code project and a saved Dexter desktop workspace are both plausible;
- a browser/Windows UI contains several real controls and a small task controller needs to choose the next one from the current goal and recent action history;
- a watched installer exposes progress text, buttons, process state, and a progress bar and Dexter needs to classify it as `still_running`, `finished`, `failed`, `needs_input`, or `uncertain`;
- a large tool surface has already been narrowed to a bounded set of relevant capabilities and a cheap semantic chooser may improve selection.

The goal is **not** to create a second general Dexter agent or to replace deterministic resolvers with probabilistic guesses.

## Core rule

Use the cheapest/strongest source that can determine the answer:

```text
hard deterministic evidence
        -> use code directly

bounded real candidates + semantic ambiguity
        -> optional bounded semantic decision provider

open-ended reasoning / generation / novel value construction
        -> Dexter main model
```

A semantic decision provider may only choose among caller-supplied options (plus explicit outcomes such as `none`, `ambiguous`, `uncertain`, `done`, or `escalate`). It does not invent paths, URLs, executable names, window handles, selectors, resource IDs, or permission-bearing targets.

## Shared provider boundary

Do not build a separate Jev wrapper for every feature. Prefer one small reusable interface in a neutral module/package. Exact API is intentionally deferred until a prototype is benchmarked, but the conceptual operations are:

```text
choose(candidates, evidence, question)
rank(candidates, evidence, question)
classify(states, evidence, question)
choose_next_action(actions, goal, history, observation)
```

Each call should return:

- selected candidate/state/action ID(s), never an invented target;
- calibrated/confidence-like provider output when available;
- enough metadata to apply caller-owned thresholds and ambiguity policy;
- provider identity/version for diagnostics and benchmarks;
- an explicit `uncertain`/`none` path rather than forcing a choice.

Callers remain responsible for:

- generating and validating candidates;
- defining allowed states/actions;
- permissions and safety policy;
- execution and postcondition verification;
- deciding whether confidence is sufficient;
- asking the user for clarification or escalating to Dexter.

## Resource/project resolution

This is the first candidate integration because it extends the resolver work already planned in #17 and #28 without replacing it.

### Keep deterministic retrieval first

Existing name/alias logic remains valuable and should still handle:

- canonical-name exact matches;
- explicit aliases;
- normalized STT spacing such as `FriendBox` vs `friend box`;
- strong unique partial/token matches;
- deterministic ambiguity when records tie.

Future unified retrieval may additionally use lexical/semantic search and current-context evidence to produce a small typed candidate set.

The decision provider is only an optional **final reranker/resolver** when deterministic retrieval cannot establish one answer safely.

Example:

```text
utterance: "open that Dexter work"

candidates:
R0  vscode_project  canonical=Local AI  aliases=[Jarvis, Dexter project]
R1  workspace       canonical=Dexter desktop
R2  folder          canonical=Dexter docs

context:
- VS Code recently focused: Local AI
- Dexter desktop workspace restored yesterday

provider result:
R0 0.46
R1 0.43
R2 0.06
uncertain 0.05
```

The caller may decide that R0/R1 are too close and ask:

> Which did you mean: the Dexter/Local AI VS Code project, or the saved Dexter desktop workspace?

This is preferable to either silent guessing or forcing the main LLM to inspect raw records.

### Alias model

Do not treat every successful fuzzy phrase as a permanent alias. Keep at least three concepts distinct:

- **explicit aliases** — user-provided names such as `Jarvis`;
- **confirmed/learned aliases** — phrases promoted only after explicit correction or sufficiently strong repeated confirmation under a conservative policy;
- **transient semantic matches** — phrases that can resolve for the current request without mutating persistent memory.

The stable resource ID/canonical target remains authoritative even when human terminology changes over time.

## Browser and native UI task control

Do not use a semantic model merely to click a control whose identity is already known. If code has already resolved `Save As`, click it deterministically.

The stronger use case is a bounded reactive controller for simple/medium UI tasks:

```text
goal + recent action history + current structured UI
        -> choose next available action
        -> execute deterministically
        -> observe again
        -> repeat / done / stuck / escalate
```

Candidate actions must be grounded in the live DOM, accessibility tree, Windows UI Automation tree, or another deterministic app adapter. Examples:

```text
A0 click element E17 "Destination"
A1 click element E22 "Save"
A2 type into E19 "File name"
A3 wait
A4 done
A5 stuck/escalate
```

The provider chooses an action ID. It never creates a CSS selector, HWND, coordinate, URL, or executable path.

This may be useful for routine interaction where the next step depends on interpreting the current page/dialog but does not justify a full Dexter reasoning turn. Consequential actions remain subject to normal policy regardless of which controller selected them.

If a required parameter is genuinely constructive/open-ended (for example a new filename or message body) the controller should preserve a literal user span when safe or ask/escalate to the main model rather than expecting the bounded provider to generate it.

## Condition Watcher

A semantic decision provider can act as a cheap **semantic sensor** inside the generic watcher planned in #18. It is not the watcher itself.

The watcher still owns:

- target identity and lifecycle;
- polling/event subscriptions and backoff;
- process/window/file/job/UI evidence gathering;
- cancellation and persistence;
- follow-up action policy.

Use deterministic evidence when it is conclusive. Only invoke semantic classification when the evidence is structured but not reducible to a safe hard rule.

Example evidence:

```text
window: NVIDIA Installer
process_running: true
progressbar: 73%
text: "Installing Graphics Driver"
button: Cancel (enabled)
```

Allowed states:

```text
still_running | finished | failed | needs_input | uncertain
```

Later evidence might contain `Installation complete` and an enabled `Close` button, allowing a cheap classifier to mark `finished` without waking the main LLM or generating a one-off watcher script/template.

Raw image interpretation remains a separate fallback unless the selected provider explicitly supports vision. Structured DOM/UIA/process/log evidence should be preferred first.

## Tool/domain routing

Issue #28 currently says not to create a second agent/router model. Preserve that default.

A bounded decision provider should only be introduced here if benchmark evidence shows a material improvement over the planned deterministic/domain-level tool surface. Plausible experiments include:

- choosing a domain from a small set after obvious deterministic routing fails;
- choosing among an already narrowed set of candidate actions/tools;
- shadow-mode comparison against the main Goose/Gemma choice.

Do not expose hundreds of micro-tools solely because a chooser can technically select among them. Domain-level capabilities, compact schemas, and deterministic internal dispatch remain desirable for context size, maintainability, permissions, and non-AI callers.

## What should stay deterministic

Do not add a semantic model where normal code already expresses the contract more reliably. Examples:

- filesystem/path allowlisting and validation;
- exact saved aliases/names when unique;
- app discovery and exact launch identities;
- known process/window state;
- permission/confirmation policy;
- close-success prerequisites such as `closed=false` / remaining windows;
- PDF/document transformations after inputs/operation are resolved;
- macro execution after a recipe is created;
- thresholds/hysteresis for resource-pressure policy;
- postcondition verification.

## Escalation model

A reusable caller policy should support three outcomes:

```text
resolved confidently -> continue
close/unsafe/uncertain -> ask the user or wait for more evidence
not representable as bounded choices -> escalate to Dexter main model
```

The provider should not be forced to choose simply to avoid an escalation.

## Provider modularity

The first experiment may use Jev, but consumers should depend on a project-owned interface such as `SemanticDecisionProvider`, not on Jev SDK objects or Jev-specific schemas.

Desired properties:

- provider configured/injected at one boundary;
- no Jev imports in resource, watcher, browser, or UI-target modules;
- deterministic/mock provider for tests;
- clean `disabled` provider so correctness never depends on cloud availability;
- timeout/failure degrades to caller-owned ambiguity/escalation, not an unsafe guess;
- provider-specific telemetry stays behind the boundary;
- future local/specialized decision models can replace or coexist with Jev.

Do not create a generic `do_anything` MCP tool around this layer. It is primarily an **internal decision primitive** consumed by concrete capabilities. A later bounded task-controller capability may be exposed to Dexter/Goose if real workflows justify it.

## Initial evaluation order

1. **Resource/search reranking (#17 / #28)** — easiest to shadow safely and directly tests natural names, aliases, context, and ambiguity.
2. **Semantic UI targeting (#19)** — choose among grounded DOM/UIA candidates where deterministic text matching is insufficient.
3. **Condition Watcher (#18)** — classify structured state where hard evidence is inconclusive.
4. **Bounded browser/UI task controller** — only after the decision primitive is reliable on the simpler cases.
5. **Tool/domain routing (#28)** — only if measurements show it is actually better than deterministic compact routing.

For each stage measure at least:

- accuracy against expected choice/state;
- abstention/ambiguity quality;
- unsafe/wrong confident choices;
- latency;
- provider availability/failure behavior;
- main-model turns avoided;
- end-to-end task success, not merely classifier agreement.

Run early integrations in shadow mode where practical before granting them action authority.

## Relationship to current issues

- #17 — unified personal/project memory and saved-resource retrieval;
- #18 — generic Condition Watcher;
- #19 — semantic UI target recognition and button actions;
- #20 — deterministic macros may consume resolved UI targets but should not use semantic decisions between ordinary saved steps;
- #28 — tool-surface scaling/domain routing and unified local lookup;
- #1 — workspace recognition remains deterministic evidence-first; semantic ranking should not replace exact workspace identity logic.

A dedicated implementation/experiment issue should own the provider abstraction and first shadow-mode benchmark so these feature issues do not each grow a separate Jev integration.