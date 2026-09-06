---
name: abslayer-experiment
description: Submit and manage durable ABSlayer workspace, authorized-screen, reviewed-vector, and development-verification jobs from Pool CLI or Codex CLI.
---

Invoke the shared bridge through the current CLI's command tool, from the ABSlayer
project root. The CLI owns model selection; Swift owns job state. The bridge
builds the isolated controller on first use. It requires Swift 6.3+ and Ruby.

Inspect capabilities before selecting an operation:

```sh
ruby Scripts/abslayer-tool.rb '{"method":"capabilities"}'
```

`workspace_preflight` is always present. A configured experiment host may also
advertise `laguna_authorized_screen`, `laguna_reviewed_vector`, and
`laguna_independent_verify`. The host configuration fixes every worker, input,
runtime setting, and output path; JSON requests cannot inject commands or choose
unbound files.

Submit a listed operation with a stable key for that intended run. Reuse the key
for reconnects. A deliberate rerun requires a new key and, for an operation that
publishes artifacts, a newly configured output path.

```sh
ruby Scripts/abslayer-tool.rb <<'JSON'
{"method":"submit","operation":"workspace_preflight","idempotencyKey":"preflight-attempt-1","timeoutSeconds":60}
JSON
```

For Laguna work, run `laguna_authorized_screen` first. It screens only manifest-
validated, explicitly authorized controls, saves exact rendered-template and
response hashes, and leaves refusal candidates unjudged. Inspect the bound
private result and semantically review any candidates. Never substitute the
dataset's excluded `contrast` prompts or use a length-limited answer as either
side of a vector.

Run `laguna_reviewed_vector` only when its configured schema-2 selection binds
the screen, dataset, manifest, model, and server hashes. Every pair must contain
a completed authorized false refusal and a completed substantive authorized
answer, have reviewed same-task/category/request-type equivalence, use distinct
responses, and stay within both rendered-byte and rendered-token length balance
gates. The worker renders both sides with Laguna's live chat template, verifies
the exact screen-bound template hashes, and requires their tokenizations to share
the final assistant-generation boundary token before publishing a controller-bound
reversible GGUF vector. Generation success is not behavioral acceptance.

`laguna_independent_verify` is a private development comparison for a configured
candidate. Prefer a schema-2 fixture that declares exact counts and disjoint,
exhaustive authorized, benign, and protected-boundary cohort lists and identifies
itself as development/not-held-out. Read per-cohort refusal, visibility, stop, and
marker counts alongside the separately reported authorized flips, benign marker
retention, and protected-boundary refusal retention. These are triage signals;
semantic correctness and a separately governed held-out evaluation remain
required. The worker always leaves acceptance unevaluated.

Pass subsequent JSON requests to the same bridge via stdin or one quoted argument:

- `{"method":"status","jobID":"JOB_ID"}`: inspect the returned job ID. Omit `jobID` to list the latest 20 jobs.
- `{"method":"cancel","jobID":"JOB_ID"}`: request cancellation and check the eventual terminal state.
- `{"method":"evidence","jobID":"JOB_ID","stream":"stderr","offset":0,"limit":4096}`: read bound output after the worker stops. Use `stdout` for the manifest and `nextOffset` only when more detail is needed.
- `{"method":"recover"}`: reconcile a supervisor interruption; interrupted jobs are never retried automatically.

Prefer compact status to raw logs; inspect stderr on failure. Poll no faster than
once per second. Submitted jobs belong to a separate local process and can be
inspected from either CLI using the same project and job ID.

`completed` means the configured worker and declared artifact checks completed.
`acceptance` stays `not_evaluated`: execution alone supplies no model-quality,
correctness, refusal, or retention verdict. Do not invent scores, treat truncated
answers as complete, or treat historic artifacts as fresh evidence.

After a supervisor crash, an active direct worker retains its execution lease.
Recovery waits for the lease to clear, then marks the old job `interrupted`.
Report a persistently held lease rather than starting competing work or assuming
a PID is safe to kill. See `Documentation/Harness.md` for implementation limits.
