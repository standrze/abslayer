import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

/// Same-token prompt-end causal tracing from a LoRA donor into its untouched
/// BF16 base model.
///
/// The adapter is loaded only long enough to capture donor residuals and
/// next-token distributions, then unloaded and numerically verified. Every
/// intervention and generated response therefore runs in the untouched model.
public enum PromptEndCrossConditionPatchingEngine {
    public static let maximumPromptTokens = 512
    public static let unloadTolerance = 1e-5

    public static func run(
        modelDirectory: String,
        adapterDirectory: String,
        adapterScaleOverride: Float? = nil,
        document: MatchedResponsePatchDocument,
        configuration: PromptEndPatchConfiguration,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> PromptEndPatchStudy {
        try document.validate()
        let inspection = try ModelFolderValidator.validateFullBF16(
            path: modelDirectory)
        if let count = inspection.decoderLayerCount {
            try configuration.validate(
                caseCount: document.cases.count,
                decoderLayerCount: count)
        }
        let end = min(
            document.cases.count,
            configuration.caseOffset + configuration.maximumCases)
        let selected = Array(
            document.cases[configuration.caseOffset ..< end])
        guard !selected.isEmpty else {
            throw PromptEndPatchError.caseOffsetOutsideInput(
                configuration.caseOffset, caseCount: document.cases.count)
        }

        let modelURL = URL(fileURLWithPath: inspection.path)
            .standardizedFileURL
        let adapterURL = URL(fileURLWithPath: adapterDirectory)
            .standardizedFileURL
        let started = ContinuousClock.now
        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(
                directory: modelURL,
                extraEOSTokens: ["<end_of_turn>"]))
        progress?(
            "fresh BF16 model loaded in "
                + started.duration(to: .now).formatted(
                    .units(
                        allowed: [.seconds, .milliseconds],
                        width: .abbreviated)))

        let base = try await capture(
            container: container, cases: selected,
            layers: configuration.layersZeroBased)
        progress?("captured untouched prompt-end states")

        let adapter = try LoRAAdapterLoader.load(
            directory: adapterURL.path,
            scaleOverride: adapterScaleOverride)
        try await container.perform { context in
            try adapter.load(into: context.model)
        }
        let donor = try await capture(
            container: container, cases: selected,
            layers: configuration.layersZeroBased)
        progress?("captured LoRA-donor prompt-end states")

        await container.perform { context in
            adapter.unload(from: context.model)
        }
        let afterUnload = try await capture(
            container: container, cases: selected,
            layers: configuration.layersZeroBased)
        let unload = unloadValidation(before: base, after: afterUnload)
        guard unload.passed else {
            throw PromptEndPatchError.adapterUnloadValidationFailed(
                unload.maximumStateAbsoluteDifference,
                unload.maximumLogProbabilityAbsoluteDifference,
                tolerance: unload.tolerance)
        }
        progress?("adapter unloaded and numerical restoration verified")

        var results = [PromptEndPatchResult]()
        results.reserveCapacity(
            selected.count * configuration.layersZeroBased.count)
        let total = selected.count * configuration.layersZeroBased.count
        var completed = 0
        for caseIndex in selected.indices {
            for layer in configuration.layersZeroBased {
                let baseline = base[caseIndex]
                let source = donor[caseIndex]
                guard let baseState = baseline.states[layer],
                      let donorState = source.states[layer]
                else { throw PromptEndPatchError.incompatibleStates }
                let delta = try PromptEndPatchMath.adapterDelta(
                    base: baseState, donor: donorState)
                let reverseState = try PromptEndPatchMath.adding(
                    delta, to: baseState, scale: -1)

                let matched = try await condition(
                    container: container, item: selected[caseIndex],
                    promptTokens: baseline.promptTokens,
                    layer: layer, replacement: donorState,
                    maximumTokens: configuration.maximumGenerationTokens)
                let reversed = try await condition(
                    container: container, item: selected[caseIndex],
                    promptTokens: baseline.promptTokens,
                    layer: layer, replacement: reverseState,
                    maximumTokens: configuration.maximumGenerationTokens)

                var randomName: String?
                var random: ConditionCapture?
                if let randomIndex = MatchedResponsePatchMath
                    .randomControlIndex(
                        caseIndex: caseIndex,
                        caseCount: selected.count,
                        seed: configuration.randomControlSeed),
                   let unrelatedBase = base[randomIndex].states[layer],
                   let unrelatedDonor = donor[randomIndex].states[layer]
                {
                    randomName = selected[randomIndex].name
                    let replacement = try PromptEndPatchMath
                        .normMatchedRandomReplacement(
                            targetBase: baseState,
                            matchedDelta: delta,
                            unrelatedBase: unrelatedBase,
                            unrelatedDonor: unrelatedDonor)
                    random = try await condition(
                        container: container, item: selected[caseIndex],
                        promptTokens: baseline.promptTokens,
                        layer: layer, replacement: replacement,
                        maximumTokens:
                            configuration.maximumGenerationTokens)
                }

                results.append(PromptEndPatchResult(
                    caseName: selected[caseIndex].name,
                    category: selected[caseIndex].category,
                    layerZeroBased: layer,
                    promptTokenCount: baseline.promptTokens.count,
                    matchedDeltaL2Norm: PromptEndPatchMath.l2Norm(delta),
                    randomControlCaseName: randomName,
                    matchedDonor: try PromptEndPatchMath.effect(
                        donorLogProbabilities: source.logProbabilities,
                        baseLogProbabilities: baseline.logProbabilities,
                        conditionLogProbabilities:
                            matched.logProbabilities),
                    signReversed: try PromptEndPatchMath.effect(
                        donorLogProbabilities: source.logProbabilities,
                        baseLogProbabilities: baseline.logProbabilities,
                        conditionLogProbabilities:
                            reversed.logProbabilities),
                    normMatchedRandom: try random.map {
                        try PromptEndPatchMath.effect(
                            donorLogProbabilities:
                                source.logProbabilities,
                            baseLogProbabilities:
                                baseline.logProbabilities,
                            conditionLogProbabilities:
                                $0.logProbabilities)
                    },
                    matchedDonorResponse: matched.response,
                    signReversedResponse: reversed.response,
                    normMatchedRandomResponse: random?.response))
                completed += 1
                progress?(
                    "prompt-end matched/reverse/random \(completed)/\(total) "
                        + "[\(selected[caseIndex].name) L0=\(layer)]")
            }
        }

        await clearIntervention(container)
        return PromptEndPatchStudy(
            modelPath: modelURL.path,
            adapterPath: adapterURL.path,
            adapterScale:
                adapterScaleOverride
                ?? adapter.configuration.loraParameters.scale,
            inputSplit: document.split,
            inputModelCondition: document.modelCondition,
            configuration: configuration,
            cases: selected,
            unloadValidation: unload,
            results: results)
    }

    private struct StateCapture: Sendable {
        let promptTokens: [Int]
        let states: [Int: [Float]]
        let logProbabilities: [Float]
    }

    private struct ConditionCapture: Sendable {
        let logProbabilities: [Float]
        let response: String
    }

    private static func capture(
        container: ModelContainer, cases: [MatchedResponsePatchCase],
        layers: [Int]
    ) async throws -> [StateCapture] {
        try await container.perform { context in
            guard let backbone = gemma4Backbone(context.model) else {
                throw PromptEndPatchError.unsupportedModel
            }
            guard layers.allSatisfy(backbone.layers.indices.contains) else {
                throw PromptEndPatchError.layerOutsideDecoder(
                    layers: layers,
                    decoderLayerCount: backbone.layers.count)
            }
            backbone.residualIntervention = nil
            var result = [StateCapture]()
            result.reserveCapacity(cases.count)
            for item in cases {
                let tokens = try promptTokens(
                    item.prompt, name: item.name, context: context)
                let input = MLXArray(tokens).expandedDimensions(axis: 0)
                let hidden = backbone.layerHiddenStates(input)
                var states = [Int: [Float]]()
                for layer in layers {
                    let state = hidden[layer][0, -1].asType(.float32)
                    eval(state)
                    states[layer] = state.asArray(Float.self)
                }
                let logs = MLXNN.logSoftmax(
                    context.model(input, cache: nil)[0, -1]
                        .asType(.float32),
                    axis: -1)
                eval(logs)
                result.append(StateCapture(
                    promptTokens: tokens,
                    states: states,
                    logProbabilities: logs.asArray(Float.self)))
            }
            return result
        }
    }

    private static func condition(
        container: ModelContainer, item: MatchedResponsePatchCase,
        promptTokens: [Int], layer: Int, replacement: [Float],
        maximumTokens: Int
    ) async throws -> ConditionCapture {
        try await installIntervention(
            container, layer: layer, promptTokenCount: promptTokens.count,
            replacement: replacement)
        do {
            let logs = try await container.perform { context in
                let input = MLXArray(promptTokens)
                    .expandedDimensions(axis: 0)
                let values = MLXNN.logSoftmax(
                    context.model(input, cache: nil)[0, -1]
                        .asType(.float32),
                    axis: -1)
                eval(values)
                return values.asArray(Float.self)
            }
            let session = ChatSession(
                container,
                generateParameters: GenerateParameters(
                    maxTokens: maximumTokens, temperature: 0))
            let response = try await session.respond(to: item.prompt)
            await clearIntervention(container)
            return ConditionCapture(
                logProbabilities: logs, response: response)
        } catch {
            await clearIntervention(container)
            throw error
        }
    }

    private static func installIntervention(
        _ container: ModelContainer, layer: Int,
        promptTokenCount: Int, replacement: [Float]
    ) async throws {
        try await container.perform { context in
            guard let backbone = gemma4Backbone(context.model) else {
                throw PromptEndPatchError.unsupportedModel
            }
            backbone.residualIntervention = {
                currentLayer, current in
                guard currentLayer == layer,
                      current.dim(-2) == promptTokenCount,
                      current.dim(-1) == replacement.count
                else { return current }
                let replacementArray = MLXArray(replacement)
                    .asType(current.dtype)
                let selected = current[0, promptTokenCount - 1]
                let delta = (replacementArray - selected)
                    .reshaped(1, 1, replacement.count)
                var mask = Array(
                    repeating: Float.zero, count: promptTokenCount)
                mask[promptTokenCount - 1] = 1
                let selector = MLXArray(mask)
                    .reshaped(1, promptTokenCount, 1)
                    .asType(current.dtype)
                return current + selector * delta
            }
        }
    }

    private static func clearIntervention(_ container: ModelContainer) async {
        await container.perform { context in
            gemma4Backbone(context.model)?.residualIntervention = nil
        }
    }

    private static func promptTokens(
        _ prompt: String, name: String, context: ModelContext
    ) throws -> [Int] {
        let tokens = try context.tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": prompt]
        ])
        guard !tokens.isEmpty, tokens.count <= maximumPromptTokens else {
            throw PromptEndPatchError.promptTooLong(
                name: name, tokenCount: tokens.count)
        }
        return tokens
    }

    private static func unloadValidation(
        before: [StateCapture], after: [StateCapture]
    ) -> PromptEndPatchUnloadValidation {
        guard before.count == after.count else {
            return PromptEndPatchUnloadValidation(
                maximumStateAbsoluteDifference: .infinity,
                maximumLogProbabilityAbsoluteDifference: .infinity,
                tolerance: unloadTolerance)
        }
        var stateMaximum = 0.0
        var logMaximum = 0.0
        for index in before.indices {
            guard before[index].promptTokens == after[index].promptTokens,
                  before[index].states.keys == after[index].states.keys
            else {
                return PromptEndPatchUnloadValidation(
                    maximumStateAbsoluteDifference: .infinity,
                    maximumLogProbabilityAbsoluteDifference: .infinity,
                    tolerance: unloadTolerance)
            }
            for layer in before[index].states.keys {
                stateMaximum = max(
                    stateMaximum,
                    PromptEndPatchMath.maximumAbsoluteDifference(
                        before[index].states[layer] ?? [],
                        after[index].states[layer] ?? []))
            }
            logMaximum = max(
                logMaximum,
                PromptEndPatchMath.maximumAbsoluteDifference(
                    before[index].logProbabilities,
                    after[index].logProbabilities))
        }
        return PromptEndPatchUnloadValidation(
            maximumStateAbsoluteDifference: stateMaximum,
            maximumLogProbabilityAbsoluteDifference: logMaximum,
            tolerance: unloadTolerance)
    }

    private static func gemma4Backbone(
        _ model: LanguageModel
    ) -> Gemma4TextModelInner? {
        if let model = model as? Gemma4TextModel { return model.model }
        if let model = model as? Gemma4Model {
            return model.languageModel.model
        }
        return nil
    }
}
