<p align="center">
  <img src="assets/abslayer-banner.png" alt="ABSlayer — Total Abliteration" width="100%">
</p>

<h1 align="center">ABSlayer</h1>

<p align="center">
  A reproducible framework for measuring, applying, and verifying model abliteration.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB.svg">
  <img alt="CUDA 13" src="https://img.shields.io/badge/CUDA-13-76B900.svg">
  <img alt="Status: experimental" src="https://img.shields.io/badge/status-experimental-orange.svg">
</p>

ABSlayer is designed as a general framework for activation-guided model
abliteration. It measures behavioral directions from contrastive prompts,
applies controlled projections to model weights, and checks the result against
held-out behavioral and utility cases. It provides a scriptable CLI and an
optional browser control panel.

> [!IMPORTANT]
> ABSlayer deliberately changes model behavior. Treat every output as a new,
> untrusted model: keep the source checkpoint, use held-out evaluation data,
> inspect the verification report, and test the final deployment format before
> using it in production.

## What it does

1. **Measure** — run paired contrast/control prompts through the NVFP4 model
   with vLLM and measure rank-4 refusal subspaces.
2. **Apply** — project those directions out of residual-writing attention and
   MLP weights in the matching BF16 checkpoint.
3. **Verify** — compare the source and candidate on held-out refusal and benign
   utility cases.

The direction artifact records dataset hashes, immutable Hugging Face
revisions, architecture and tokenizer fingerprints, sampled weight anchors,
selected layers, and runtime versions. Checkpoint surgery is out of core: it
processes one Safetensors shard and one fused expert slice at a time and never
modifies the source checkpoint.

## Current model support

The framework is intended to support multiple model families and abliteration
strategies. The first validated v1 backend currently supports four Poolside
Laguna checkpoint roles:

| Family | Measurement checkpoint | Application checkpoint |
| --- | --- | --- |
| Laguna XS 2.1 | `poolside/Laguna-XS-2.1-NVFP4` | `poolside/Laguna-XS-2.1` |
| Laguna S 2.1 | `poolside/Laguna-S-2.1-NVFP4` | `poolside/Laguna-S-2.1` |

The measurement and BF16 checkpoints must have matching architecture,
tokenizer, chat template, and sampled weight lineage. ABSlayer fails closed
when that relationship cannot be proven.

ABSlayer does not quantize checkpoints, serve inference, or accept GGUF, INT4,
or FP8 as application inputs. Convert the completed BF16 output separately
after verification.

## Requirements

- Python 3.12
- One visible Blackwell-class NVIDIA CUDA device
- CUDA 13-compatible drivers and runtime
- Enough disk for the source checkpoint, candidate checkpoint, and temporary
  hidden states
- [`uv`](https://docs.astral.sh/uv/)

Only one GPU may be visible during a v1 run. If the system exposes several,
select one explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run abslayer doctor
```

## Installation

Clone the project and install the validated CUDA environment:

```bash
cd abslayer
uv sync --extra cuda
uv run abslayer doctor
```

The lightweight base install can inspect help, datasets, and the checkpoint
allow-list without installing CUDA packages:

```bash
uv sync
uv run abslayer models
```

For the browser interface:

```bash
uv sync --extra cuda --extra web
```

## Quick start

Run the complete Measure → Apply → Verify pipeline:

```bash
uv run abslayer run \
  poolside/Laguna-XS-2.1-NVFP4 \
  poolside/Laguna-XS-2.1 \
  --pairs ./measurement.jsonl \
  --evaluation ./evaluation.jsonl \
  --artifact ./artifacts/laguna-xs-directions \
  --output ./models/laguna-xs-abslayer \
  --report ./reports/laguna-xs-verification.json
```

ABSlayer resolves allow-listed Hugging Face IDs into the local cache. Local
checkpoint directories may be supplied instead.

### Run each stage independently

```bash
uv run abslayer measure poolside/Laguna-XS-2.1-NVFP4 \
  --pairs ./measurement.jsonl \
  --artifact ./artifacts/laguna-xs-directions \
  --temp-dir ./work/hidden-states

uv run abslayer apply poolside/Laguna-XS-2.1 \
  --artifact ./artifacts/laguna-xs-directions \
  --output ./models/laguna-xs-abslayer

uv run abslayer verify \
  poolside/Laguna-XS-2.1 \
  ./models/laguna-xs-abslayer \
  --cases ./evaluation.jsonl \
  --report ./reports/laguna-xs-verification.json
```

Commands support `--json` for machine-readable output, `--quiet` to suppress
normal progress, and `--no-banner` to disable terminal branding. Run
`uv run abslayer --help` or `uv run abslayer COMMAND --help` for every option.

## Dataset formats

Measurement data is JSONL containing matched contrast/control prompts:

```json
{"contrast":"A prompt that triggers refusal","control":"A matched benign prompt"}
```

At least five unique pairs are required. Evaluation data must be held out from
measurement and contain both refusal and utility cases:

```json
{"kind":"refusal","prompt":"A held-out refusal test"}
{"kind":"utility","prompt":"What is 2 + 2?","reference":"4"}
```

The quality of the measurement set matters more than raw size. Contrast prompts
should trigger the behavior being measured in the exact source model, while
controls should be close in topic and form without triggering that behavior.

## Verification policy

The balanced v1 gate requires:

- nonempty candidate generations;
- candidate refusal rate at or below 10%, or a reduction of at least 20
  percentage points from the source;
- mean benign reference-NLL regression no greater than 0.25 nats/token; and
- no individual benign regression greater than 1.0 nats/token.

Passing this gate is evidence for the supplied evaluation set, not a general
claim about safety, capability, or downstream behavior. Re-evaluate after GGUF
conversion or quantization because those transformations can alter results.

## Browser control panel

Install the `web` extra and start the loopback-only interface:

```bash
uv run abslayer web
```

Open <http://127.0.0.1:7860>. The interface includes a guided workflow builder,
live subprocess logs, cancellation, verification metrics, diagnostics, and the
exact model allow-list.

GPU workflows run one at a time in isolated worker process groups. The web
server never imports vLLM into its own process, and its local API is read-only
apart from job creation and cancellation. Version 1 has no authentication, so
do not expose it directly to the public internet.

For a remote DGX Spark, keep ABSlayer bound to loopback and tunnel it:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@your-spark
```

Then open <http://127.0.0.1:7860> on your computer.

## Output layout

```text
artifacts/laguna-xs-directions/
├── directions.safetensors
└── manifest.json

models/laguna-xs-abslayer/
├── abslayer.json
├── config.json
├── model-*.safetensors
├── model.safetensors.index.json
└── tokenizer and chat-template files

reports/
└── laguna-xs-verification.json
```

Output directories must not already exist. This protects completed models and
artifacts from accidental overwrites.

## Development

```bash
uv sync --extra dev --extra web
uv run pytest
uv run ruff check .
```

CUDA smoke tests are opt-in and require all four local checkpoint roles plus
user-supplied measurement and held-out datasets:

```bash
ABSLAYER_CUDA_SMOKE=1 \
ABSLAYER_XS_NVFP4=/models/Laguna-XS-2.1-NVFP4 \
ABSLAYER_XS_BF16=/models/Laguna-XS-2.1 \
ABSLAYER_S_NVFP4=/models/Laguna-S-2.1-NVFP4 \
ABSLAYER_S_BF16=/models/Laguna-S-2.1 \
ABSLAYER_SMOKE_PAIRS=/data/five-pairs.jsonl \
ABSLAYER_SMOKE_EVALUATION=/data/held-out.jsonl \
uv run pytest -m cuda --basetemp /large-volume/abslayer-smoke
```

## License

ABSlayer source code is licensed under the [Apache License 2.0](LICENSE).

Model weights and derived checkpoints are not relicensed by ABSlayer. They
remain subject to their original model licenses; Poolside Laguna S/XS 2.1 model
materials are distributed under OpenMDW-1.1. Bundled or user-supplied datasets
and third-party dependencies retain their respective licenses and terms.
