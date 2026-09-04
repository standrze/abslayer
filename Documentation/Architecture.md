# Product architecture

## Product decision

Build one Swift ABSlayer engine that existing agent clients can operate. Support
Pool CLI from Poolside and Codex CLI as the conversation and research interfaces.
The user can choose Pool with local inference to reduce paid token usage. Ruby handles
occasional supporting scripts. A dedicated UI and a tuned controller model are
deferred decisions, not prerequisites.

Use shared project skills in `.agents/skills/`. Both CLIs own skill loading,
conversation management, model selection, and their existing function/tool loop.
Skills use the client's command tool to call a structured bridge into the same
deterministic Swift experiment controller. MCP and an ABSlayer-owned model loop
are outside this scope. See [Harness structure](Harness.md) for the proposed
components and operation surface.

The existing engine is retained. A shared skill, structured command bridge, and
durable Swift controller now execute workspace-preflight jobs. The integrated
model-experiment lifecycle and model worker adapters below still need implementation.

## Responsibilities

| Component | Responsibility | Current state |
| --- | --- | --- |
| Native engine | Model inspection, activation capture, interventions, adapter training, evaluation, export | Existing Swift/MLX code retained |
| Experiment controller | Candidate lifecycle, budgets, scheduling, checkpoints, resource locks, acceptance policy | First durable job lifecycle implemented; multi-candidate/model policy remains planned |
| Durable worker service | Run jobs independently of the client connection; provide job IDs, progress, cancellation and recovery | Implemented for workspace preflight; model workers and exact resume remain planned |
| Shared skills and command bridge | Supply the same workflows to Pool CLI and Codex CLI; submit typed operations through their existing command tools | One shared skill and Ruby bridge implemented; live model-driven client sessions not yet tested |
| Research assistant | Interpret development evidence and propose structured experiments | Initially supplied by the agent client |
| Evaluation | Executable task checks, calibrated judgments, independent retention measurements, final held-out assessment | Existing pieces require consolidation |
| Artifact package | Exact model identity, intervention, runtime settings, results and provenance | Existing formats require a shared contract |

## Integration approach

1. Keep the existing Swift engine and its patched dependencies operational.
2. Port the main harness's deterministic lifecycle and evidence checks into a
   Swift controller. Incorporate the experimental harness's useful backend
   contracts without preserving two competing state machines.
3. Give every engine operation a shared request/result identity. Record the exact
   model, tokenizer/template, data, intervention, generation limits, termination
   reason, output hashes, metric definitions, and software version.
4. Implement persistent job ownership before allowing clients to launch long
   experiments. Returning a job ID and reconnecting to it must not depend on a
   chat staying open. Resumability and process survival are separate properties.
5. Expose small, structured operations to agent clients. Existing CLIs may remain
   internal workers. Skills explain their workflow; the bridge submits structured
   requests and Swift handlers validate them. The initial integration uses each
   CLI's existing command tool; it does not require injecting custom functions.
   The client integration should not own scientific policy or state.
6. Require complete-task evaluation and a final held-out assessment for accepted
   results. Development screens must remain visibly distinct.

The current production JSON backend and older evaluation/training executables
have different interfaces and support limits. Consolidation must explicitly adapt
them; this cleanup does not establish a universal backend or expand model support.

## Models in the product

The target model is the model being modified and eventually used for cyber tasks.
The chosen CLI's model supplies research reasoning; this may be a local model
through Pool. It is separate from the target model. A judge is a
separately calibrated evaluation role. The archived experimental controller's
action-copying training corpus is not evidence of research ability or cyber skill.

A small tuned research model can be evaluated later on real experiment histories.
Its value should be measured through validated gains per resource budget and
fewer wasted experiments, not merely valid JSON or state-machine imitation.

Provider choice does not change the controller, workers, artifact identities, or
acceptance rules. Keep summaries bounded and large outputs in artifacts so
switching clients can reconstruct the current run without replaying full chats.
Account for local reasoning-model resource use alongside experiment workers.
Changing to paid inference remains an explicit user choice.

## Acceptance and evidence

Keep willingness, substantive attempt, complete output, executable task success,
and preserved capability as separate outcomes. Short refusal screens and low
first-token KL do not establish useful cyber performance. General and safe-cyber
retention must be evaluated independently under declared metrics.

Original experiments remain historical development evidence. No model has been
newly accepted, benchmarked, or certified by the project cleanup.
