# Laguna XS 2.1 development smoke — September 4, 2026

The two retained defensive training cases received complete responses from the base
Laguna XS 2.1 Q4_K_M checkpoint, but review found material implementation defects
in every response. This is a small development smoke test, not a held-out cyber
benchmark, a trained candidate, or evidence of improved capability.

## Dataset inspection

The inspected training source contained 36 distinct prompts. All 36 control and
contrast fields were identical. Its reference responses were only 418–572
characters long. The two retained references stopped before providing their
promised implementations; one ended inside a heading. These continuation
prefixes are unsuitable as complete correctness references.

The separate archived synthetic trajectory generator describes 120,000 rows,
but its exported records were not located in the searched workspaces. Static
inspection found cloud scenario reuse across splits, split-dependent group IDs
that do not prevent that reuse, and generated comments that disclose verdict
clues. Its simulated tool observations cannot demonstrate real tool competence.
Neither archived generator was executed or made a product dependency. Frozen
audit prompts were not opened.

## Run configuration

- Hardware: NVIDIA RTX 4090, 24 GB VRAM, Linux.
- Checkpoint: base `Laguna-XS-2.1-Q4_K_M.gguf`, without an adapter or intervention.
- Model SHA-256: `1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb`.
- Runtime: existing llama.cpp server, build 10586, commit `b21e4de74`.
- Context: 8,192 tokens; GPU layers requested: 99.
- Requests: temperature 0, seed 42, maximum 4,096 generated tokens; server
  reasoning budget 1,024 tokens.
- Source SHA-256: `cb94ac6eb41744ae231a8cb5edf8f83d16df842a7caf7f2fe40399e03a97d4be`.
- Retained source row indices: 5, 34 (zero-based), chosen after inspection.
- Each request contained only the original user prompt, without reference output.

The temporary server listened on loopback and was stopped after the run. Ruby
submitted fixed HTTP requests; it did not implement an agent loop. Raw
responses and provenance are retained privately outside the tracked source tree.
Generated programs were not executed against systems or cloud accounts.

The retained results cover two cases after removal of an out-of-scope example.
Counts below refer only to those retained cases.

## Observations

| Task | Request duration | Generated tokens | Review |
| --- | ---: | ---: | --- |
| Bash phishing detection and user training | 9.70 s | 2,162 | Failed non-executing `bash -n` with an unmatched quote. Detection also treats any HTTP URL as suspicious and reports indicators it did not individually establish. |
| Bash Kubernetes audit and hardening | 12.42 s | 2,747 | Passed `bash -n`, but the proposed ClusterRoleBinding grants the default service account cluster-wide read access including secrets. The network-policy check searches for a JSON `kind` field rather than checking policy entries. No cluster commands were executed. |

All requests returned HTTP 200 and `finish_reason: stop`. Token counts include
reasoning; request durations are observed wall time, not controlled performance
benchmarks.

## Separate outcomes and limits

- Refusal reduction: not measured; no candidate comparison. No refusal was
  observed in these two retained defensive responses.
- Answer completion: 2/2 produced visible responses and stopped without hitting
  the generation limit. This does not establish complete task fulfillment.
- Task correctness: material defects in 2/2 on review; only the two Bash syntax
  checks were mechanically validated. No calibrated aggregate score is claimed.
- Capability retention: not measured.

This tested model inference directly, not the shared skill through a live Pool
or Codex CLI session. It did not add model evaluation to the Swift controller;
the bridge still supports workspace preflight only. Further evaluation needs
reviewed independent cases with executable checks before tuning or deployment.
