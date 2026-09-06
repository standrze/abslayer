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
