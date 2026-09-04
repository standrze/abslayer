import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

/// Keeps one quantized (or BF16) model resident while optimizer trials swap
/// tiny LoRA/QLoRA adapters or exact residual interventions in and out.
public final class ResidentTrialRuntime: Sendable {
    private let container: ModelContainer
    private let modelName: String

    public init(modelDirectory: String) async throws {
        let url = URL(fileURLWithPath: modelDirectory).standardizedFileURL
        modelName = url.path
        let started = ContinuousClock.now
        try MLXResourceGuard.apply()
        container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(
                directory: url, extraEOSTokens: ["<end_of_turn>"]))
        let duration = started.duration(to: .now)
        print("resident trial model loaded in \(duration.formatted(.units(allowed: [.seconds, .milliseconds], width: .abbreviated)))")
    }

    public func makeAdapter(
        directions: [[Float]], subspaces: [[[Float]]]? = nil,
        configuration: AbliterationConfiguration, fullNormalizationRank: Int = 3
    ) async throws -> LoRAContainer {
        try await container.perform { context in
            try AbliterationAdapterFactory.make(
                model: context.model, directions: directions, subspaces: subspaces,
                configuration: configuration,
                fullNormalizationRank: fullNormalizationRank)
        }
    }

    public func load(_ adapter: LoRAContainer) async throws {
        try await container.perform { context in try adapter.load(into: context.model) }
    }

    public func unload(_ adapter: LoRAContainer) async {
        await container.perform { context in adapter.unload(from: context.model) }
    }

    /// Installs an exact residual-stream transform in Gemma 4's real forward path.
    /// Unlike the lightweight LoRA trials, this runs after the selected decoder
    /// layer for both prompt prefill and cached generation.
    public func installResidualIntervention(
        _ intervention: @escaping @Sendable (_ layer: Int, _ state: MLXArray) -> MLXArray
    ) async throws {
        try await container.perform { context in
            guard let backbone = Self.gemma4Backbone(context.model) else {
                throw ResidualInterventionError.unsupportedModel
            }
            backbone.residualIntervention = intervention
        }
    }

    public func install(_ intervention: ExactResidualIntervention) async throws {
        try await installResidualIntervention(intervention.makeTransform())
    }

    public func install(_ interventions: [ExactResidualIntervention]) async throws {
        try await installResidualIntervention(
            ExactResidualIntervention.combinedTransform(interventions))
    }

    public func clearResidualIntervention() async {
        await container.perform { context in
            Self.gemma4Backbone(context.model)?.residualIntervention = nil
        }
    }

    /// Reuses the already-resident model for activation collection, baseline
    /// generation, KL fingerprints, and every exact causal intervention trial.
    public func activationCollections(
        pairs: [PromptPair],
        positions: Set<ActivationTokenPosition> = [.lastUser, .postInstruction],
        progress: (@Sendable (_ completed: Int, _ total: Int) -> Void)? = nil
    ) async throws -> [ActivationTokenPosition: ActivationCollection] {
        try await collectActivationCollections(
            container: container, modelName: modelName, pairs: pairs,
            positions: positions, progress: progress)
    }

    public func responses(
        pairs: [PromptPair], maximumCases: Int, maximumTokens: Int = 100,
        systemPrompt: String? = nil
    ) async throws -> [PromptResult] {
        let selected = Self.evenlySpaced(pairs, maximum: maximumCases)
        let parameters = GenerateParameters(maxTokens: maximumTokens, temperature: 0)
        var results = [PromptResult]()
        results.reserveCapacity(selected.count)
        for (index, pair) in selected.enumerated() {
            let contrast = try await streamedResponse(
                prompt: pair.contrast, parameters: parameters,
                systemPrompt: systemPrompt, shouldStop: { _ in false })
            let control = try await streamedResponse(
                prompt: pair.control, parameters: parameters,
                systemPrompt: systemPrompt, shouldStop: { _ in false })
            results.append(PromptResult(
                name: pair.name, contrastResponse: contrast, controlResponse: control,
                category: pair.category, systemPrompt: systemPrompt,
                contrastPrompt: pair.contrast, controlPrompt: pair.control))
            print("evaluated resident trial \(index + 1)/\(selected.count)")
        }
        return results
    }

    private func streamedResponse(
        prompt: String, parameters: GenerateParameters,
        systemPrompt: String? = nil,
        shouldStop: (String) -> Bool
    ) async throws -> String {
        let session = ChatSession(container, generateParameters: parameters)
        var output = ""
        if let systemPrompt {
            for try await chunk in session.streamResponse(to: [
                .system(systemPrompt), .user(prompt),
            ]) {
                output += chunk
                if shouldStop(output) { break }
            }
        } else {
            for try await chunk in session.streamResponse(to: prompt) {
                output += chunk
                if shouldStop(output) { break }
            }
        }
        return output
    }

    public func response(
        to prompt: String, maximumTokens: Int = 256,
        systemPrompt: String? = nil
    ) async throws -> String {
        try await streamedResponse(
            prompt: prompt,
            parameters: GenerateParameters(maxTokens: maximumTokens, temperature: 0),
            systemPrompt: systemPrompt,
            shouldStop: { _ in false })
    }

    public func fingerprint(
        pairs: [PromptPair], maximumCases: Int
    ) async throws -> LogitFingerprint {
        let selected = Self.evenlySpaced(pairs, maximum: maximumCases)
        var rows = [[Float]]()
        rows.reserveCapacity(selected.count)
        for (index, pair) in selected.enumerated() {
            let row = try await container.perform { context in
                let tokens = try context.tokenizer.applyChatTemplate(messages: [
                    ["role": "user", "content": pair.control]
                ])
                guard tokens.count <= 512 else {
                    throw ProbeError.promptTooLong(name: pair.name, tokenCount: tokens.count)
                }
                let input = MLXArray(tokens).expandedDimensions(axis: 0)
                let logits = context.model(input, cache: nil)[0, -1].asType(.float32)
                let probabilities = MLXNN.logSoftmax(logits, axis: -1)
                eval(probabilities)
                return probabilities.asArray(Float.self)
            }
            rows.append(row)
            print("fingerprinted resident trial \(index + 1)/\(selected.count)")
        }
        return LogitFingerprint(
            promptNames: selected.map(\.name), vocabularySize: rows.first?.count ?? 0,
            logProbabilities: rows)
    }

    /// Control-prompt-prefix utility sketch captured with the currently
    /// installed exact intervention. Candidate support is fixed to the
    /// untouched baseline's top-K tokens at each prompt position. This legacy
    /// metric does not cover assistant continuation tokens.
    public func controlPromptPrefixFingerprint(
        pairs: [PromptPair], maximumCases: Int, topK: Int = 64,
        reference: SequenceLogitFingerprint? = nil
    ) async throws -> SequenceLogitFingerprint {
        let selected = Self.evenlySpaced(pairs, maximum: maximumCases)
        guard topK > 0, reference == nil || reference?.cases.count == selected.count else {
            throw FingerprintError.incompatible
        }
        let requestedTopK = reference?.topK ?? topK
        var cases = [SequenceCaseFingerprint]()
        var vocabularySize: Int?
        for (index, pair) in selected.enumerated() {
            let referenceCase = reference?.cases[index]
            guard referenceCase == nil || referenceCase?.name == pair.name else {
                throw FingerprintError.incompatible
            }
            let captured = try await container.perform { context in
                let tokens = try context.tokenizer.applyChatTemplate(messages: [
                    ["role": "user", "content": pair.control]
                ])
                guard tokens.count <= 512 else {
                    throw ProbeError.promptTooLong(name: pair.name, tokenCount: tokens.count)
                }
                guard tokens.count >= 2,
                      referenceCase == nil || referenceCase?.tokenIDs == tokens
                else { throw FingerprintError.incompatible }

                let positionCount = tokens.count - 1
                let input = MLXArray(tokens).expandedDimensions(axis: 0)
                let logits = context.model(input, cache: nil)[0, 0 ..< positionCount]
                    .asType(.float32)
                let logProbabilities = MLXNN.logSoftmax(logits, axis: -1)
                let vocabulary = logProbabilities.dim(-1)
                guard requestedTopK < vocabulary else { throw FingerprintError.incompatible }

                let supportIndices: MLXArray
                if let referenceCase {
                    guard referenceCase.positions.count == positionCount,
                          referenceCase.positions.allSatisfy({
                              $0.supportTokenIDs.count == requestedTopK
                          })
                    else { throw FingerprintError.incompatible }
                    supportIndices = MLXArray(
                        referenceCase.positions.flatMap(\.supportTokenIDs),
                        [positionCount, requestedTopK])
                } else {
                    supportIndices = argPartition(
                        logProbabilities, kth: -requestedTopK, axis: -1
                    )[.ellipsis, (-requestedTopK)...]
                }
                let supportLogs = takeAlong(
                    logProbabilities, supportIndices, axis: -1)
                let retainedMass = exp(supportLogs).sum(axis: -1)
                let tailLogs = log(clip(1 - retainedMass, min: Float(1e-30)))
                let targetIDs = Array(tokens.dropFirst())
                let targetLogs = takeAlong(
                    logProbabilities,
                    MLXArray(targetIDs).expandedDimensions(axis: -1),
                    axis: -1).squeezed(axis: -1)
                eval(supportIndices, supportLogs, tailLogs, targetLogs)

                let supportIDs = supportIndices.asArray(Int.self)
                let supportValues = supportLogs.asArray(Float.self)
                let tailValues = tailLogs.asArray(Float.self)
                let targetValues = targetLogs.asArray(Float.self)
                let positions = (0 ..< positionCount).map { position in
                    let range = position * requestedTopK ..< (position + 1) * requestedTopK
                    return SequencePositionFingerprint(
                        targetTokenID: targetIDs[position],
                        supportTokenIDs: Array(supportIDs[range]),
                        supportLogProbabilities: Array(supportValues[range]),
                        tailLogProbability: tailValues[position],
                        targetLogProbability: targetValues[position])
                }
                return (vocabulary, SequenceCaseFingerprint(
                    name: pair.name, tokenIDs: tokens, positions: positions))
            }
            if let vocabularySize, vocabularySize != captured.0 {
                throw FingerprintError.incompatible
            }
            vocabularySize = captured.0
            cases.append(captured.1)
            print("sequence-fingerprinted resident trial \(index + 1)/\(selected.count)")
        }
        guard let vocabularySize else { throw FingerprintError.incompatible }
        return SequenceLogitFingerprint(
            vocabularySize: vocabularySize, topK: requestedTopK, cases: cases)
    }

    /// Backward-compatible spelling for `controlPromptPrefixFingerprint`.
    /// Existing artifacts and call sites use "sequence" for this prompt-only
    /// top-K-plus-tail sketch, so removing it would be needlessly disruptive.
    @available(*, deprecated, renamed: "controlPromptPrefixFingerprint")
    public func sequenceFingerprint(
        pairs: [PromptPair], maximumCases: Int, topK: Int = 64,
        reference: SequenceLogitFingerprint? = nil
    ) async throws -> SequenceLogitFingerprint {
        try await controlPromptPrefixFingerprint(
            pairs: pairs, maximumCases: maximumCases, topK: topK,
            reference: reference)
    }

    /// Measures exact full-vocabulary KL over fixed benign assistant answers.
    ///
    /// Each selected pair must provide `controlReferenceResponse`. The method
    /// runs the untouched model and the currently installed exact residual
    /// intervention through independent KV caches in lockstep. Prompt prefill
    /// and one-token decode calls therefore follow the intervention's deployed
    /// generation schedule exactly, while teacher forcing holds the answer
    /// tokens fixed for a like-for-like distribution comparison.
    ///
    /// Returns `nil` when no selected pair contains a non-empty reference
    /// response, preserving compatibility with older prompt corpora.
    public func teacherForcedControlContinuationMetrics(
        pairs: [PromptPair], maximumCases: Int,
        maximumReferenceTokens: Int = 128
    ) async throws -> TeacherForcedContinuationMetricSummary? {
        guard maximumCases > 0, maximumReferenceTokens > 0 else {
            throw FingerprintError.incompatible
        }
        let selected = Self.evenlySpaced(pairs, maximum: maximumCases).filter {
            $0.controlReferenceResponse?.trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty == false
        }
        guard !selected.isEmpty else { return nil }

        var samples = [TeacherForcedContinuationCaseSamples]()
        samples.reserveCapacity(selected.count)
        for (index, pair) in selected.enumerated() {
            let sample = try await container.perform {
                (context: ModelContext) async throws
                    -> TeacherForcedContinuationCaseSamples in
                guard let referenceResponse = pair.controlReferenceResponse else {
                    throw FingerprintError.incompatible
                }
                let promptTokens = try context.tokenizer.applyChatTemplate(
                    messages: [["role": "user", "content": pair.control]])
                guard promptTokens.count <= 512 else {
                    throw ProbeError.promptTooLong(
                        name: pair.name, tokenCount: promptTokens.count)
                }
                // Derive content tokens from the exact generation prompt that
                // will be prefetched at runtime. This avoids assuming that a
                // completed-chat rendering uses a byte/token-identical
                // assistant header. The completed template is only a checked
                // fallback for tokenizers whose decode/encode round trip is
                // not prefix-stable.
                let renderedPrompt = context.tokenizer.decode(
                    tokenIds: promptTokens, skipSpecialTokens: false)
                let promptPlusReference = context.tokenizer.encode(
                    text: renderedPrompt + referenceResponse,
                    addSpecialTokens: false)
                var templatedConversation: [Int]?
                if !promptPlusReference.starts(with: promptTokens) {
                    templatedConversation = try context.tokenizer.applyChatTemplate(
                        messages: [
                            ["role": "user", "content": pair.control],
                            ["role": "assistant", "content": referenceResponse],
                        ],
                        tools: nil,
                        additionalContext: ["add_generation_prompt": false])
                }
                let derivedContinuation: [Int]
                do {
                    derivedContinuation = try
                        TeacherForcedContinuationTokenDerivation.continuation(
                            promptTokens: promptTokens,
                            promptPlusReferenceTokens: promptPlusReference,
                            templatedConversationTokens: templatedConversation)
                } catch {
                    throw ProbeError.referenceContinuationTokenizationMismatch(
                        name: pair.name)
                }
                let continuation = Array(
                    derivedContinuation.prefix(maximumReferenceTokens))
                guard let backbone = Self.gemma4Backbone(context.model) else {
                    throw ResidualInterventionError.unsupportedModel
                }

                // Keep the candidate closure (and its schedule state) installed,
                // but switch it off only for baseline calls. Separate caches
                // prevent either path from contaminating the other's history.
                let candidateTransform = backbone.residualIntervention
                defer { backbone.residualIntervention = candidateTransform }
                let baselineCache = try context.model.newCache(parameters: nil)
                let candidateCache = try context.model.newCache(parameters: nil)
                var tokenKL = [Double]()
                var baselineTargets = [Double]()
                var candidateTargets = [Double]()
                tokenKL.reserveCapacity(continuation.count)
                baselineTargets.reserveCapacity(continuation.count)
                candidateTargets.reserveCapacity(continuation.count)

                for position in continuation.indices {
                    let inputTokens = position == 0
                        ? promptTokens
                        : [continuation[position - 1]]
                    let input = MLXArray(inputTokens).expandedDimensions(axis: 0)

                    backbone.residualIntervention = nil
                    let baselineLogs = MLXNN.logSoftmax(
                        context.model(input, cache: baselineCache)[0, -1]
                            .asType(.float32),
                        axis: -1)
                    backbone.residualIntervention = candidateTransform
                    let candidateLogs = MLXNN.logSoftmax(
                        context.model(input, cache: candidateCache)[0, -1]
                            .asType(.float32),
                        axis: -1)

                    let divergence = (
                        exp(baselineLogs) * (baselineLogs - candidateLogs)
                    ).sum()
                    let target = continuation[position]
                    let baselineTarget = baselineLogs[target]
                    let candidateTarget = candidateLogs[target]
                    eval(divergence, baselineTarget, candidateTarget)
                    tokenKL.append(max(
                        0, Double(divergence.item(Float.self))))
                    baselineTargets.append(Double(
                        baselineTarget.item(Float.self)))
                    candidateTargets.append(Double(
                        candidateTarget.item(Float.self)))
                }
                return TeacherForcedContinuationCaseSamples(
                    name: pair.name,
                    tokenKLDivergences: tokenKL,
                    baselineTargetLogProbabilities: baselineTargets,
                    candidateTargetLogProbabilities: candidateTargets)
            }
            samples.append(sample)
            print(
                "teacher-forced continuation \(index + 1)/\(selected.count)")
        }
        return try TeacherForcedContinuationMetricEngine.summarize(samples)
    }

    private static func evenlySpaced(
        _ pairs: [PromptPair], maximum: Int
    ) -> [PromptPair] {
        guard maximum > 0, pairs.count > maximum else { return pairs }
        return (0 ..< maximum).map { pairs[$0 * pairs.count / maximum] }
    }

    private static func gemma4Backbone(_ model: LanguageModel) -> Gemma4TextModelInner? {
        if let model = model as? Gemma4TextModel { return model.model }
        if let model = model as? Gemma4Model { return model.languageModel.model }
        return nil
    }
}

public enum ResidualInterventionError: LocalizedError {
    case unsupportedModel

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel:
            "Exact residual interventions currently require a Gemma 4 text backbone."
        }
    }
}
