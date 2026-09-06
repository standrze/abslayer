# Laguna XS 2.1 independent runtime-vector verification — September 5, 2026

The generated runtime control vector is structurally valid and changes Laguna
XS 2.1 behavior, but the independent verification does not support calling it a
successful abliteration. The vector is rejected as a cyber-capability candidate.

## Bound configuration

- Base GGUF SHA-256: `1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb`.
- Control-vector SHA-256: `6232df5817bbee9dc637bfcbdbc363e3bae7f8feaddd460592647e6d4dbba3a2`.
- Candidate application: scale `-0.25`, inclusive layers `1...39`.
- Runtime: pinned local llama.cpp server on the RTX 4090 host.
- Deterministic generation: temperature 0, seed 42, 2,048-token response cap,
  512-token reasoning budget.
- Harness job: `29f983ca-9018-4ae8-8ceb-efb0a1413eb7`, completed with exit 0.

The private fixture contained 16 fresh cases: eight explicitly authorized SSRF
tasks, four benign utility tasks, and four unauthorized high-risk requests.
Prompt and response text remain private. This was a development verification,
not a sealed benchmark.

## Results

| Outcome | Base | Vector candidate |
| --- | ---: | ---: |
| Authorized SSRF visible answers | 8/8 | 8/8 |
| Authorized SSRF normal stops | 7/8 | 8/8 |
| Authorized refusal-to-answer flips | — | 0 |
| Benign utility correctness | 4/4 | 4/4 |
| Sensitive-boundary retention, semantic review | 4/4 | 4/4 |

The base model already engaged with every authorized SSRF request, leaving no
false refusal for this vector to remove. The candidate's only clear behavioral
gain was one additional normally completed answer.

Semantic review rejected the automated keyword score. Several candidate SSRF
answers contained material defects: incomplete validated-IP connection pinning,
DNS rebinding time-of-check/time-of-use gaps, broken webhook pseudocode, unsafe
special-address guidance, and risky URL canonicalization advice. A narrow
refusal regex also missed one safe sensitive-request refusal, so semantic review
is authoritative for that cohort.

## Exhaustive authorized SSRF screen

The base model was subsequently run through all 126 training candidates in the
three SSRF-related categories: `oast-correlated-ssrf`,
`scope-redirect-egress-recheck`, and `ssrf-destination-policy`. Harness job
`596fb1e1-028a-4c5b-b7d1-83b7f6ffec7a` completed with exit 0 on the same model
hash. Its bound stdout SHA-256 is
`d6f6f3abdf35f73d0e0ab02344feeab92e6be82dc78525f08ebea162a20023a9`;
the private-response SHA-256 is
`0afbf5e971a297c8d97a71c129eb01ed7c80f17a3c4704de5bd21d3b16e648f3`.

| Screen outcome | Count |
| --- | ---: |
| Authorized prompts sent | 126 |
| Visible substantive attempts, semantic review | 126 |
| Normal stops | 16 |
| Length-limited outputs | 110 |
| Refusal candidates from the configured screen | 0 |
| Semantically eligible false-refusal pairs | 0 |

All 126 visible outputs were semantically reviewed against their paired controls;
none was an actual refusal, soft deflection, clarification-only response, or empty
attempt. A secondary broad word scan flagged four outputs for closer review. Two contained words
such as “cannot” inside direct operational procedures; two contained transition
language such as “instead” while still answering the requested task. All four
were direct, length-limited attempts rather than refusals or deflections. Because
partial answers are ineligible under the dataset manifest, none can be used as a
positive or negative vector side.

This automated screen plus private semantic review found no valid false-refusal
direction in the current SSRF candidate data for Laguna XS 2.1. A corrected vector
was deliberately not generated from excluded targets or incomplete answers. The
dataset is machine-generated development material and this result is not held-out
task correctness evidence.

## Correction to the earlier screen

The earlier eight-case screen cannot establish SSRF abliteration. It selected
the dataset's excluded-target `contrast` side, covered no SSRF category, searched
hidden reasoning together with visible content, and counted token-limit outputs
as completed answers. Its reported refusal reduction is withdrawn.

The artifact may remain available for reproducibility, but it must not be
presented as accepted, deployed by default, or used as evidence of improved
cyber capability. A new direction requires closely matched authorized
false-refusal pairs rendered with Laguna's exact chat template, followed by
independent executable correctness and retention evaluation.

The host artifact directory now contains a private `REJECTED.json` marker
(SHA-256 `055df5f3c000350a8b17b5259d8aa8d2427bed1e343932b2817be00a1ab4712b`)
that binds this verdict to the model, vector, and verification job identities.
