# Development

## Retained implementation

The Swift package keeps its original `Sources`, native tests, dependency pins and
patches. Native executable workers are declared in `Package.swift`. They are
existing research tools, not the completed consolidated product interface.

The package requires Swift 6.3 or later. Its declared macOS minimum is 15.
The production `abslayer` JSON backend currently requires CUDA and supported
full-BF16 Gemma 4 metadata; other research entrypoints have different support.
Choosing a Swift implementation does not imply that every worker runs on every
platform or supports every model architecture.

## Existing build paths

The original build entrypoints are retained at their original locations:

- `prepare-dependencies.sh` resolves and patches the pinned native dependencies.
- `build-metal.sh` builds the Metal shader library on macOS.
- `build-cuda-4090.sh` configures the existing Linux/CUDA build.
- `test-metal.sh` invokes the existing Metal test workflow.

These inherited shell files and their helper paths are read by existing build
contract tests. Their Ruby replacement must preserve dependency pins, patch
validation, resource settings, and platform behavior. Controller tests do not
establish that the full native engine builds on each supported platform.

New supporting scripts use Ruby. The current Swift build/runtime requires none
of the archived Python harness or dataset tools.

## Lightweight checks

The new controller has an isolated build/test path that does not resolve the
model graph or rewrite its dependency pins:

```sh
ruby Scripts/build-harness.rb build
ruby Scripts/build-harness.rb test
ruby Tests/HarnessIntegrationTests.rb
ruby Tests/LagunaWorkerTests.rb
```

The helper stages a manifest using `ABSLAYER_HARNESS_ONLY=1` under
`.build-harness/`, with links to the real controller sources and tests. Do not use
that reduced graph to resolve dependencies in the product root. The bridge
auto-builds this controller and always supports `workspace_preflight`. A host may
also register the Laguna authorized-screen, reviewed-vector, and independent
development-verification workers; see
[Harness](Harness.md) for its JSON contract, guarantees, and current limitations.

Original migration archives and their verification scripts are local recovery
tools and are not distributed with the product. They are not prerequisites for
building or running the controller.

`swift package dump-package` validates the package manifest without running a
model. Full builds require dependency resolution and the platform setup above.

Model-bearing tests and experiments must use the existing resource guards and
avoid concurrent model processes. A filesystem cleanup is not a reason to run
training or infer performance from historical results.
