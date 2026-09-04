---
name: abslayer-experiment
description: Submit ABSlayer workspace preflight jobs and inspect, cancel, or recover their durable execution. Use for ABSlayer preflight and job-management requests in Pool CLI or Codex CLI.
---

Invoke the shared bridge through the current CLI's command tool, from the ABSlayer
project root. The CLI owns model selection; Swift owns job state. The bridge
builds the isolated controller on first use. It requires Swift 6.3+ and Ruby.

Inspect capabilities before selecting an operation:

```sh
ruby Scripts/abslayer-tool.rb '{"method":"capabilities"}'
```

This version implements `workspace_preflight`: SwiftPM manifest evaluation with
no model dependency resolution, model loading, or dataset reads. Training, model
evaluation, and export are not supported by this bridge yet.

Submit a requested preflight with a stable key for that intended run. Reuse the
key for retries; choose a new key for a deliberate rerun or changed inputs.

```sh
ruby Scripts/abslayer-tool.rb <<'JSON'
{"method":"submit","operation":"workspace_preflight","idempotencyKey":"preflight-attempt-1","timeoutSeconds":60}
JSON
```

Pass subsequent JSON requests to the same bridge via stdin or one quoted argument:

- `{"method":"status","jobID":"JOB_ID"}`: inspect the returned job ID. Omit `jobID` to list the latest 20 jobs.
- `{"method":"cancel","jobID":"JOB_ID"}`: request cancellation and check the eventual terminal state.
- `{"method":"evidence","jobID":"JOB_ID","stream":"stderr","offset":0,"limit":4096}`: read bound output after the worker stops. Use `stdout` for the manifest and `nextOffset` only when more detail is needed.
- `{"method":"recover"}`: reconcile a supervisor interruption; interrupted jobs are never retried automatically.

Prefer compact status to raw logs; inspect stderr on failure. Poll no faster than
once per second. Submitted jobs belong to a separate local process and can be
inspected from either CLI using the same project and job ID.

`completed` means workspace preflight completed. `acceptance` stays
`not_evaluated`: this supplies no model-quality, correctness, refusal, or retention
result. Do not invent scores or treat historic artifacts as fresh evidence.

After a supervisor crash, an active direct worker retains its execution lease.
Recovery waits for the lease to clear, then marks the old job `interrupted`.
Report a persistently held lease rather than starting competing work or assuming
a PID is safe to kill. See `Documentation/Harness.md` for implementation limits.
