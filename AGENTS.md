# ABSlayer product instructions

These instructions record the user's decisions for the consolidated product.

- The product is named `abslayer` / ABSlayer.
- Use Swift for product implementation. Do not split the controller into Go.
- Use Ruby when supporting scripts are needed. Do not introduce Python code,
  Python runtime dependencies, or Python-based tests into the active product.
- Existing shell build helpers are retained for compatibility. Port them to Ruby
  with their behavior and build-contract checks preserved when that migration is
  undertaken; do not silently wrap archived Python tooling.
- The required agent clients are Pool CLI from Poolside and Codex CLI. They own
  the conversation loop, model selection, skill loading, and agent tools. Pool
  must support the user's option to use local inference to reduce paid token
  usage. Do not build an ABSlayer-owned agent/model loop for this scope.
- Use shared project skills in `.agents/skills/` and the clients' existing command
  tools to reach a structured ABSlayer bridge. MCP is not the chosen integration.
  Keep model/provider selection outside experiment state and acceptance policy;
  record the reasoning model used as provenance when available.
- A custom SwiftUI app is not a decided requirement. Do not add one or a
  new user-facing CLI merely because the implementation language is Swift.
- Reuse existing native workers and useful external CLIs. A future portable tool
  interface should operate one durable Swift experiment controller.
- Keep experiment state, resource management, acceptance policy, and artifact
  identity in deterministic code. Optional models may propose experiments or
  supply calibrated judgments; a trained controller model is not required.
- Report refusal reduction, answer completion, task correctness, and capability
  retention separately. Mark planned features and historical results explicitly.
- Do not treat archived run artifacts as fresh held-out evidence. Do not expose
  frozen audit prompts during development.
- Preserve the Swift package's source/test ancestry and pinned dependency patches
  unless the task explicitly requires changing them.
- Historical implementations and data live outside this directory. Local
  workspaces may include an untracked `Documentation/Migration.md` with recovery
  paths. These records and archives are not distributed and must not become
  dependencies of the product.
- `intro-video/` is an independent active media workspace, excluded from the
  product cleanup. Do not move or alter it as part of product work while that
  task is active. These product migration decisions do not replace its task scope.

Current scope and implementation status are in `README.md` and `Documentation/`.
