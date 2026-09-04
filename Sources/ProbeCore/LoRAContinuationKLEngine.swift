import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Tokenizers

/// Provenance-bearing result for exact teacher-forced benign-continuation KL.
public struct LoRAContinuationKLReport: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let modelDirectory: String
    public let adapterDirectory: String
    public let adapterScale: Double
    public let requestedMaximumCases: Int
    public let maximumReferenceTokens: Int
    public let summary: TeacherForcedContinuationMetricSummary

    public init(
        schemaVersion: Int = 1,
        modelDirectory: String,
        adapterDirectory: String,
        adapterScale: Double,
        requestedMaximumCases: Int,
        maximumReferenceTokens: Int,
        summary: TeacherForcedContinuationMetricSummary
    ) {
        self.schemaVersion = schemaVersion
        self.modelDirectory = modelDirectory
        self.adapterDirectory = adapterDirectory
        self.adapterScale = adapterScale
        self.requestedMaximumCases = requestedMaximumCases
        self.maximumReferenceTokens = maximumReferenceTokens
        self.summary = summary
    }
}

public enum LoRAContinuationKLError: LocalizedError, Equatable {
    case invalidLimits
    case missingControlReferences
    case emptyReference(String)

    public var errorDescription: String? {
        switch self {
        case .invalidLimits:
            "MAX_CASES and MAX_REFERENCE_TOKENS must be positive integers."
        case .missingControlReferences:
            "No selected prompt has a non-empty controlReferenceResponse."
        case .emptyReference(let name):
            "The reference continuation for '\(name)' produced no tokens."
        }
    }
}

/// Measures exact `KL(base || adapter)` over every vocabulary entry at each
/// teacher-forced token in fixed benign control responses.
public enum LoRAContinuationKLEngine {
    public static func run(
        modelDirectory: String,
        adapterDirectory: String,
        adapterScaleOverride: Float? = nil,
        pairs: [PromptPair],
        maximumCases: Int,
        maximumReferenceTokens: Int
    ) async throws -> LoRAContinuationKLReport {
        guard maximumCases > 0, maximumReferenceTokens > 0 else {
            throw LoRAContinuationKLError.invalidLimits
        }
        let selected = evenlySpaced(pairs, maximum: maximumCases).filter {
            $0.controlReferenceResponse?.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty == false
        }
        guard !selected.isEmpty else {
            throw LoRAContinuationKLError.missingControlReferences
        }

        let modelURL = URL(fileURLWithPath: modelDirectory).standardizedFileURL
        let adapterURL = URL(fileURLWithPath: adapterDirectory).standardizedFileURL
        let adapter = try LoRAAdapterLoader.load(
            directory: adapterURL.path, scaleOverride: adapterScaleOverride)
        let effectiveScale = Double(adapter.configuration.loraParameters.scale)

        let started = ContinuousClock.now
        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(
                directory: modelURL, extraEOSTokens: ["<end_of_turn>"]))
        print(
            "fresh untouched model loaded in "
                + started.duration(to: .now).formatted(
                    .units(
                        allowed: [.seconds, .milliseconds],
                        width: .abbreviated)))

        var samples = [TeacherForcedContinuationCaseSamples]()
        samples.reserveCapacity(selected.count)
        for (index, pair) in selected.enumerated() {
            let sample = try await container.perform { context in
                guard let referenceResponse = pair.controlReferenceResponse else {
                    throw LoRAContinuationKLError.emptyReference(pair.name)
                }
                let promptTokens = try context.tokenizer.applyChatTemplate(
                    messages: [["role": "user", "content": pair.control]])
                guard promptTokens.count <= 512 else {
                    throw ProbeError.promptTooLong(
                        name: pair.name, tokenCount: promptTokens.count)
                }

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
                let derived: [Int]
                do {
                    derived = try TeacherForcedContinuationTokenDerivation.continuation(
                        promptTokens: promptTokens,
                        promptPlusReferenceTokens: promptPlusReference,
                        templatedConversationTokens: templatedConversation)
                } catch {
                    throw ProbeError.referenceContinuationTokenizationMismatch(
                        name: pair.name)
                }
                let continuation = Array(
                    derived.prefix(maximumReferenceTokens))
                guard !continuation.isEmpty else {
                    throw LoRAContinuationKLError.emptyReference(pair.name)
                }

                // The freshly loaded model is untouched here. Fully evaluate
                // and retain the base distributions before installing LoRA so
                // MLX laziness cannot accidentally bind them to adapter weights.
                let baselineCache = try context.model.newCache(parameters: nil)
                var baselineLogs = [MLXArray]()
                var baselineTargets = [Double]()
                baselineLogs.reserveCapacity(continuation.count)
                baselineTargets.reserveCapacity(continuation.count)
                for position in continuation.indices {
                    let inputTokens = position == 0
                        ? promptTokens : [continuation[position - 1]]
                    let input = MLXArray(inputTokens).expandedDimensions(axis: 0)
                    let logs = MLXNN.logSoftmax(
                        context.model(input, cache: baselineCache)[0, -1]
                            .asType(.float32),
                        axis: -1)
                    let targetLog = logs[continuation[position]]
                    eval(logs, targetLog)
                    baselineLogs.append(logs)
                    baselineTargets.append(Double(targetLog.item(Float.self)))
                }

                // A zero-scale control must be the untouched model. Installing
                // a LoRA wrapper at scale zero is not numerically identical to
                // leaving the base Linear in place: mlx-swift-lm's LoRALinear
                // casts its input to the weight dtype and still evaluates
                // `base + 0 * update`. That produced a measurable false KL
                // floor in the identity control. Skip module replacement
                // entirely so scale zero remains a meaningful engine check.
                let installsAdapter = shouldInstallAdapter(
                    effectiveScale: effectiveScale)
                if installsAdapter {
                    try adapter.load(into: context.model)
                }
                defer {
                    if installsAdapter {
                        adapter.unload(from: context.model)
                    }
                }

                // Never reuse the base cache: it contains keys and values
                // produced by the untouched weights. The candidate gets a
                // separately allocated cache and the identical token schedule.
                let candidateCache = try context.model.newCache(parameters: nil)
                var tokenKL = [Double]()
                var candidateTargets = [Double]()
                tokenKL.reserveCapacity(continuation.count)
                candidateTargets.reserveCapacity(continuation.count)
                for position in continuation.indices {
                    let inputTokens = position == 0
                        ? promptTokens : [continuation[position - 1]]
                    let input = MLXArray(inputTokens).expandedDimensions(axis: 0)
                    let candidateLogs = MLXNN.logSoftmax(
                        context.model(input, cache: candidateCache)[0, -1]
                            .asType(.float32),
                        axis: -1)
                    let baseLogs = baselineLogs[position]
                    let divergence = (
                        exp(baseLogs) * (baseLogs - candidateLogs)
                    ).sum()
                    let candidateTarget = candidateLogs[continuation[position]]
                    eval(divergence, candidateTarget)
                    tokenKL.append(max(
                        0, Double(divergence.item(Float.self))))
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
            print("teacher-forced continuation \(index + 1)/\(selected.count)")
        }

        return LoRAContinuationKLReport(
            modelDirectory: modelURL.path,
            adapterDirectory: adapterURL.path,
            adapterScale: effectiveScale,
            requestedMaximumCases: maximumCases,
            maximumReferenceTokens: maximumReferenceTokens,
            summary: try TeacherForcedContinuationMetricEngine.summarize(samples))
    }

    static func evenlySpaced(
        _ pairs: [PromptPair], maximum: Int
    ) -> [PromptPair] {
        guard maximum > 0, pairs.count > maximum else { return pairs }
        return (0 ..< maximum).map { pairs[$0 * pairs.count / maximum] }
    }

    static func shouldInstallAdapter(effectiveScale: Double) -> Bool {
        effectiveScale != 0
    }
}
