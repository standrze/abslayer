import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public struct GemmaParityRecord: Equatable, Sendable {
    public let datasetName: String
    public let id: String
    public let instruction: String
    public let response: String

    public init(datasetName: String, id: String, instruction: String, response: String) {
        self.datasetName = datasetName
        self.id = id
        self.instruction = instruction
        self.response = response
    }
}

public enum GemmaParityRecordLoader {
    /// Selects the exact normalized record the prefix trainer consumes. This
    /// deliberately requires the versioned harness envelope: a diagnostic
    /// result without a stable record identity is not reproducible evidence.
    public static func load(_ data: Data, recordID: String? = nil) throws -> GemmaParityRecord {
        guard let dataset = try? JSONDecoder().decode(HarnessTrainingDataset.self, from: data)
        else {
            throw GemmaParityDiagnosticError.versionedDatasetRequired
        }
        let normalized = try HarnessTrainingDatasetLoader.load(data)
        precondition(dataset.records.count == normalized.count)

        let selectedIndex: Int
        if let recordID {
            let requested = recordID.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !requested.isEmpty else {
                throw GemmaParityDiagnosticError.emptyRecordID
            }
            guard
                let index = dataset.records.firstIndex(where: {
                    $0.id.trimmingCharacters(in: .whitespacesAndNewlines) == requested
                })
            else {
                throw GemmaParityDiagnosticError.recordNotFound(requested)
            }
            selectedIndex = index
        } else {
            selectedIndex = 0
        }

        let source = dataset.records[selectedIndex]
        let completion = normalized[selectedIndex]
        return GemmaParityRecord(
            datasetName: dataset.datasetName.trimmingCharacters(in: .whitespacesAndNewlines),
            id: source.id.trimmingCharacters(in: .whitespacesAndNewlines),
            instruction: completion.prompt,
            response: completion.target)
    }
}

/// The token partition used verbatim by `abslayer-prefix-train`: the deployed
/// user-only generation prompt is the masked prefix, while every token added by
/// the completed assistant turn—including end-of-turn framing—is supervised.
public struct GemmaTrainingTokenPartition: Equatable, Sendable {
    public let completeTokens: [Int]
    public let deployedPromptTokens: [Int]
    public let trainingPrefixTokens: [Int]
    public let targetTokens: [Int]
    public let trailingTemplateTokens: [Int]
    public let assistantStart: Int
    public let assistantEnd: Int

    public static func derive(
        completeTokens: [Int], deployedPromptTokens: [Int]
    ) throws -> Self {
        guard !completeTokens.isEmpty, !deployedPromptTokens.isEmpty else {
            throw GemmaParityDiagnosticError.emptyTemplateTokens
        }
        let assistant = try SupervisedCompletionBoundary.resolve(
            promptTokens: deployedPromptTokens, completedTokens: completeTokens)
        return Self(
            completeTokens: completeTokens,
            deployedPromptTokens: deployedPromptTokens,
            trainingPrefixTokens: deployedPromptTokens,
            targetTokens: Array(completeTokens[assistant]),
            trailingTemplateTokens: [],
            assistantStart: assistant.lowerBound,
            assistantEnd: assistant.upperBound)
    }
}

public struct GemmaPromptTokenDivergence: Codable, Equatable, Sendable {
    public let tokenIndex: Int
    public let trainingTokenID: Int?
    public let inferenceTokenID: Int?
    public let trainingToken: String?
    public let inferenceToken: String?
}

public struct GemmaPromptParityReport: Codable, Equatable, Sendable {
    public let exact: Bool
    public let trainingPrefixTokenCount: Int
    public let inferencePrefixTokenCount: Int
    public let trainingPrefixSHA256: String
    public let inferencePrefixSHA256: String
    public let assistantStart: Int
    public let assistantEnd: Int
    public let targetTokenCount: Int
    public let trailingTemplateTokenCount: Int
    public let firstDivergence: GemmaPromptTokenDivergence?

    public static func make(
        partition: GemmaTrainingTokenPartition,
        inferencePrefixTokens: [Int],
        tokenText: (Int) -> String
    ) -> Self {
        let training = partition.trainingPrefixTokens
        let index = firstDifference(training, inferencePrefixTokens)
        let divergence = index.map { index in
            let trainingID = training.indices.contains(index) ? training[index] : nil
            let inferenceID =
                inferencePrefixTokens.indices.contains(index)
                ? inferencePrefixTokens[index] : nil
            return GemmaPromptTokenDivergence(
                tokenIndex: index,
                trainingTokenID: trainingID,
                inferenceTokenID: inferenceID,
                trainingToken: trainingID.map(tokenText),
                inferenceToken: inferenceID.map(tokenText))
        }
        return Self(
            exact: divergence == nil,
            trainingPrefixTokenCount: training.count,
            inferencePrefixTokenCount: inferencePrefixTokens.count,
            trainingPrefixSHA256: GemmaParityHashes.tokenIDs(training),
            inferencePrefixSHA256: GemmaParityHashes.tokenIDs(inferencePrefixTokens),
            assistantStart: partition.assistantStart,
            assistantEnd: partition.assistantEnd,
            targetTokenCount: partition.targetTokens.count,
            trailingTemplateTokenCount: partition.trailingTemplateTokens.count,
            firstDivergence: divergence)
    }
}

public struct GemmaTeacherForcedMismatch: Codable, Equatable, Sendable {
    public let targetOffset: Int
    public let completeTokenIndex: Int
    public let expectedTokenID: Int
    public let predictedTokenID: Int
    public let expectedToken: String
    public let predictedToken: String
}

public struct GemmaTeacherForcedReport: Codable, Equatable, Sendable {
    public let targetTokenCount: Int
    public let greedyMatchCount: Int
    public let greedyAccuracy: Double
    public let exact: Bool
    public let targetTokenSHA256: String
    public let predictionTokenSHA256: String
    public let allLogitsFinite: Bool
    public let firstNonFiniteLogitTargetOffset: Int?
    public let firstMismatch: GemmaTeacherForcedMismatch?

    public static func make(
        expected: [Int], predicted: [Int], assistantStart: Int,
        allLogitsFinite: Bool = true,
        firstNonFiniteLogitTargetOffset: Int? = nil,
        tokenText: (Int) -> String
    ) throws -> Self {
        guard expected.count == predicted.count else {
            throw GemmaParityDiagnosticError.predictionCountMismatch(
                label: "full-sequence", expected: expected.count, actual: predicted.count)
        }
        guard !expected.isEmpty else {
            throw GemmaParityDiagnosticError.emptyAssistantSpan
        }
        let mismatchIndex = expected.indices.first(where: {
            expected[$0] != predicted[$0]
        })
        let mismatch = mismatchIndex.map { index in
            GemmaTeacherForcedMismatch(
                targetOffset: index,
                completeTokenIndex: assistantStart + index,
                expectedTokenID: expected[index],
                predictedTokenID: predicted[index],
                expectedToken: tokenText(expected[index]),
                predictedToken: tokenText(predicted[index]))
        }
        let matches = zip(expected, predicted).count(where: { $0 == $1 })
        return Self(
            targetTokenCount: expected.count,
            greedyMatchCount: matches,
            greedyAccuracy: Double(matches) / Double(expected.count),
            exact: mismatch == nil,
            targetTokenSHA256: GemmaParityHashes.tokenIDs(expected),
            predictionTokenSHA256: GemmaParityHashes.tokenIDs(predicted),
            allLogitsFinite: allLogitsFinite,
            firstNonFiniteLogitTargetOffset: firstNonFiniteLogitTargetOffset,
            firstMismatch: mismatch)
    }
}

public enum GemmaPrefillMode: String, Codable, Equatable, Sendable {
    case balanced
    case unchunked
}

public struct GemmaCachePredictionDivergence: Codable, Equatable, Sendable {
    public let targetOffset: Int
    public let completeTokenIndex: Int
    public let expectedTokenID: Int
    public let fullSequencePredictionTokenID: Int
    public let cachedPredictionTokenID: Int
    public let expectedToken: String
    public let fullSequencePredictionToken: String
    public let cachedPredictionToken: String
}

public struct GemmaCachePredictionReport: Codable, Equatable, Sendable {
    public let prefillMode: GemmaPrefillMode
    public let repetition: Int
    public let targetTokenCount: Int
    public let fullSequenceMatchCount: Int
    public let predictionsMatchFullSequence: Bool
    public let greedyTargetMatchCount: Int
    public let greedyTargetAccuracy: Double
    public let cachedPredictionTokenSHA256: String
    public let allLogitsFinite: Bool
    public let firstNonFiniteLogitTargetOffset: Int?
    public let allCacheStatesFinite: Bool
    public let firstNonFiniteCacheTargetOffset: Int?
    public let firstDivergence: GemmaCachePredictionDivergence?

    public static func make(
        mode: GemmaPrefillMode,
        repetition: Int = 1,
        expected: [Int],
        fullSequencePredictions: [Int],
        cachedPredictions: [Int],
        assistantStart: Int,
        allLogitsFinite: Bool = true,
        firstNonFiniteLogitTargetOffset: Int? = nil,
        allCacheStatesFinite: Bool = true,
        firstNonFiniteCacheTargetOffset: Int? = nil,
        tokenText: (Int) -> String
    ) throws -> Self {
        guard expected.count == fullSequencePredictions.count else {
            throw GemmaParityDiagnosticError.predictionCountMismatch(
                label: "full-sequence", expected: expected.count,
                actual: fullSequencePredictions.count)
        }
        guard expected.count == cachedPredictions.count else {
            throw GemmaParityDiagnosticError.predictionCountMismatch(
                label: mode.rawValue, expected: expected.count,
                actual: cachedPredictions.count)
        }
        guard !expected.isEmpty else {
            throw GemmaParityDiagnosticError.emptyAssistantSpan
        }

        let divergenceIndex = fullSequencePredictions.indices.first(where: {
            fullSequencePredictions[$0] != cachedPredictions[$0]
        })
        let divergence = divergenceIndex.map { index in
            GemmaCachePredictionDivergence(
                targetOffset: index,
                completeTokenIndex: assistantStart + index,
                expectedTokenID: expected[index],
                fullSequencePredictionTokenID: fullSequencePredictions[index],
                cachedPredictionTokenID: cachedPredictions[index],
                expectedToken: tokenText(expected[index]),
                fullSequencePredictionToken: tokenText(fullSequencePredictions[index]),
                cachedPredictionToken: tokenText(cachedPredictions[index]))
        }
        let fullMatches = zip(fullSequencePredictions, cachedPredictions)
            .count(where: { $0 == $1 })
        let targetMatches = zip(expected, cachedPredictions).count(where: { $0 == $1 })
        return Self(
            prefillMode: mode,
            repetition: repetition,
            targetTokenCount: expected.count,
            fullSequenceMatchCount: fullMatches,
            predictionsMatchFullSequence: divergence == nil,
            greedyTargetMatchCount: targetMatches,
            greedyTargetAccuracy: Double(targetMatches) / Double(expected.count),
            cachedPredictionTokenSHA256: GemmaParityHashes.tokenIDs(cachedPredictions),
            allLogitsFinite: allLogitsFinite,
            firstNonFiniteLogitTargetOffset: firstNonFiniteLogitTargetOffset,
            allCacheStatesFinite: allCacheStatesFinite,
            firstNonFiniteCacheTargetOffset: firstNonFiniteCacheTargetOffset,
            firstDivergence: divergence)
    }
}

public enum GemmaPredictionContext: String, Codable, Equatable, Sendable {
    case fullSequence = "full_sequence"
    case balancedCache = "balanced_cache"
    case unchunkedCache = "unchunked_cache"
}

public struct GemmaRepeatedPredictionDivergence: Codable, Equatable, Sendable {
    public let targetOffset: Int
    public let completeTokenIndex: Int
    public let referenceRepetition: Int
    public let differingRepetition: Int
    public let referenceTokenID: Int
    public let differingTokenID: Int
    public let referenceToken: String
    public let differingToken: String
}

public struct GemmaPredictionRepeatabilityReport: Codable, Equatable, Sendable {
    public let context: GemmaPredictionContext
    public let repetitionCount: Int
    public let predictionsIdentical: Bool
    public let predictionTokenSHA256s: [String]
    public let allLogitsFiniteByRepetition: [Bool]
    public let allCacheStatesFiniteByRepetition: [Bool]?
    public let firstDivergence: GemmaRepeatedPredictionDivergence?

    public static func make(
        context: GemmaPredictionContext,
        predictions: [[Int]],
        assistantStart: Int,
        allLogitsFiniteByRepetition: [Bool],
        allCacheStatesFiniteByRepetition: [Bool]? = nil,
        tokenText: (Int) -> String
    ) throws -> Self {
        guard predictions.count >= 2,
            allLogitsFiniteByRepetition.count == predictions.count,
            allCacheStatesFiniteByRepetition.map({ $0.count == predictions.count }) != false,
            let expectedCount = predictions.first?.count,
            expectedCount > 0,
            predictions.allSatisfy({ $0.count == expectedCount })
        else {
            throw GemmaParityDiagnosticError.invalidRepetitionEvidence(context.rawValue)
        }

        var first: (offset: Int, repetition: Int)?
        for offset in 0..<expectedCount {
            if let repetition = (1..<predictions.count).first(where: {
                predictions[$0][offset] != predictions[0][offset]
            }) {
                first = (offset, repetition)
                break
            }
        }
        let divergence = first.map { value in
            let reference = predictions[0][value.offset]
            let differing = predictions[value.repetition][value.offset]
            return GemmaRepeatedPredictionDivergence(
                targetOffset: value.offset,
                completeTokenIndex: assistantStart + value.offset,
                referenceRepetition: 1,
                differingRepetition: value.repetition + 1,
                referenceTokenID: reference,
                differingTokenID: differing,
                referenceToken: tokenText(reference),
                differingToken: tokenText(differing))
        }
        return Self(
            context: context,
            repetitionCount: predictions.count,
            predictionsIdentical: divergence == nil,
            predictionTokenSHA256s: predictions.map(GemmaParityHashes.tokenIDs),
            allLogitsFiniteByRepetition: allLogitsFiniteByRepetition,
            allCacheStatesFiniteByRepetition: allCacheStatesFiniteByRepetition,
            firstDivergence: divergence)
    }
}

public enum GemmaParityDiagnosis: String, Codable, Equatable, Sendable {
    case promptPrefixMismatch = "prompt_prefix_mismatch"
    case nonFiniteNumerics = "non_finite_numerics"
    case fullSequenceNondeterminism = "full_sequence_nondeterminism"
    case cachedPredictionNondeterminism = "cached_prediction_nondeterminism"
    case incrementalCacheDivergence = "incremental_cache_divergence"
    case balancedPrefillDivergence = "balanced_prefill_divergence"
    case teacherForcedGreedyMismatch = "teacher_forced_greedy_mismatch"
    case noDetectedDivergence = "no_detected_divergence"
}

public struct GemmaParityDiagnosticReport: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let modelDirectory: String
    public let loadedModelType: String
    public let adapterDirectory: String?
    public let adapterScale: Float?
    public let mlxExecutionBackend: String
    public let mlxUseCUDAGraphs: String
    public let mlxCUDAGraphsEnabled: Bool?
    public let repetitions: Int
    public let datasetPath: String
    public let datasetSHA256: String
    public let datasetName: String
    public let recordID: String
    public let instructionSHA256: String
    public let responseSHA256: String
    public let promptParity: GemmaPromptParityReport
    public let teacherForced: GemmaTeacherForcedReport
    public let cacheComparisons: [GemmaCachePredictionReport]
    public let predictionRepeatability: [GemmaPredictionRepeatabilityReport]
    public let diagnosis: GemmaParityDiagnosis
}

public enum GemmaParityDiagnosticEngine {
    public static let repetitions = 3

    public static func run(
        modelDirectory: String,
        datasetPath: String,
        recordID: String? = nil,
        adapterDirectory: String? = nil,
        adapterScaleOverride: Float? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) async throws -> GemmaParityDiagnosticReport {
        let model = try ModelFolderValidator.validateFullBF16(path: modelDirectory)
        let datasetURL = URL(fileURLWithPath: datasetPath).standardizedFileURL
            .resolvingSymlinksInPath()
        let datasetData = try Data(contentsOf: datasetURL)
        let record = try GemmaParityRecordLoader.load(datasetData, recordID: recordID)
        let normalizedAdapterDirectory = adapterDirectory.map {
            URL(fileURLWithPath: $0).standardizedFileURL.resolvingSymlinksInPath().path
        }

        try MLXResourceGuard.apply(environment: environment)
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(
                directory: URL(fileURLWithPath: model.path),
                extraEOSTokens: ["<end_of_turn>"]))

        var effectiveAdapterScale: Float?
        if let normalizedAdapterDirectory {
            let adapter = try LoRAAdapterLoader.load(
                directory: normalizedAdapterDirectory,
                scaleOverride: adapterScaleOverride)
            effectiveAdapterScale = adapter.configuration.loraParameters.scale
            try await container.perform { context in
                try adapter.load(into: context.model)
            }
        }

        let values = DiagnosticInputs(
            record: record,
            modelDirectory: model.path,
            adapterDirectory: normalizedAdapterDirectory,
            adapterScale: effectiveAdapterScale,
            datasetPath: datasetURL.path,
            datasetSHA256: GemmaParityHashes.data(datasetData))
        return try await container.perform(values: values) { context, values in
            guard context.model is Gemma4Model || context.model is Gemma4TextModel else {
                throw GemmaParityDiagnosticError.incompatibleModel(
                    String(reflecting: type(of: context.model)))
            }
            context.model.train(false)

            let prepared = try await context.processor.prepare(
                input: UserInput(prompt: values.record.instruction))
            eval(prepared.text.tokens)
            let inferencePrefix = prepared.text.tokens.asArray(Int.self)

            let trainingPrefix = try context.tokenizer.applyChatTemplate(messages: [
                ["role": "user", "content": values.record.instruction]
            ])
            let complete = try context.tokenizer.applyChatTemplate(messages: [
                ["role": "user", "content": values.record.instruction],
                ["role": "assistant", "content": values.record.response],
            ], tools: nil, additionalContext: ["add_generation_prompt": false])
            let partition = try GemmaTrainingTokenPartition.derive(
                completeTokens: complete, deployedPromptTokens: trainingPrefix)
            let tokenText: (Int) -> String = { tokenID in
                context.tokenizer.decode(tokenIds: [tokenID], skipSpecialTokens: false)
            }
            let promptParity = GemmaPromptParityReport.make(
                partition: partition,
                inferencePrefixTokens: inferencePrefix,
                tokenText: tokenText)

            var fullRuns = [PredictionRun]()
            var cacheRuns = [GemmaPrefillMode: [CachedPredictionRun]]()
            for _ in 0..<repetitions {
                fullRuns.append(
                    try fullSequencePredictions(
                        model: context.model, partition: partition))
                for mode in [GemmaPrefillMode.balanced, .unchunked] {
                    cacheRuns[mode, default: []].append(
                        try cachedPredictions(
                            model: context.model,
                            prefixTokens: inferencePrefix,
                            targetTokens: partition.targetTokens,
                            mode: mode))
                }
            }
            let fullPredictions = fullRuns[0].predictions
            let teacherForced = try GemmaTeacherForcedReport.make(
                expected: partition.targetTokens,
                predicted: fullPredictions,
                assistantStart: partition.assistantStart,
                allLogitsFinite: fullRuns[0].allLogitsFinite,
                firstNonFiniteLogitTargetOffset:
                    fullRuns[0].firstNonFiniteLogitTargetOffset,
                tokenText: tokenText)

            var cacheComparisons = [GemmaCachePredictionReport]()
            for mode in [GemmaPrefillMode.balanced, .unchunked] {
                for (index, run) in cacheRuns[mode, default: []].enumerated() {
                    cacheComparisons.append(
                        try GemmaCachePredictionReport.make(
                            mode: mode,
                            repetition: index + 1,
                            expected: partition.targetTokens,
                            fullSequencePredictions: fullPredictions,
                            cachedPredictions: run.predictions,
                            assistantStart: partition.assistantStart,
                            allLogitsFinite: run.allLogitsFinite,
                            firstNonFiniteLogitTargetOffset:
                                run.firstNonFiniteLogitTargetOffset,
                            allCacheStatesFinite: run.allCacheStatesFinite,
                            firstNonFiniteCacheTargetOffset:
                                run.firstNonFiniteCacheTargetOffset,
                            tokenText: tokenText))
                }
            }
            let predictionRepeatability = try [
                GemmaPredictionRepeatabilityReport.make(
                    context: .fullSequence,
                    predictions: fullRuns.map(\.predictions),
                    assistantStart: partition.assistantStart,
                    allLogitsFiniteByRepetition: fullRuns.map(\.allLogitsFinite),
                    tokenText: tokenText),
                cacheRepeatability(
                    mode: .balanced,
                    runs: cacheRuns[.balanced, default: []],
                    assistantStart: partition.assistantStart,
                    tokenText: tokenText),
                cacheRepeatability(
                    mode: .unchunked,
                    runs: cacheRuns[.unchunked, default: []],
                    assistantStart: partition.assistantStart,
                    tokenText: tokenText),
            ]

            return GemmaParityDiagnosticReport(
                schemaVersion: 1,
                modelDirectory: values.modelDirectory,
                loadedModelType: String(reflecting: type(of: context.model)),
                adapterDirectory: values.adapterDirectory,
                adapterScale: values.adapterScale,
                mlxExecutionBackend: MLXExecutionBackend.compiled.rawValue,
                mlxUseCUDAGraphs: environment["MLX_USE_CUDA_GRAPHS"] ?? "unset",
                mlxCUDAGraphsEnabled: effectiveCUDAGraphSetting(environment),
                repetitions: repetitions,
                datasetPath: values.datasetPath,
                datasetSHA256: values.datasetSHA256,
                datasetName: values.record.datasetName,
                recordID: values.record.id,
                instructionSHA256: GemmaParityHashes.string(values.record.instruction),
                responseSHA256: GemmaParityHashes.string(values.record.response),
                promptParity: promptParity,
                teacherForced: teacherForced,
                cacheComparisons: cacheComparisons,
                predictionRepeatability: predictionRepeatability,
                diagnosis: diagnose(
                    promptParity: promptParity,
                    teacherForced: teacherForced,
                    cacheComparisons: cacheComparisons,
                    predictionRepeatability: predictionRepeatability))
        }
    }

    public static func diagnose(
        promptParity: GemmaPromptParityReport,
        teacherForced: GemmaTeacherForcedReport,
        cacheComparisons: [GemmaCachePredictionReport],
        predictionRepeatability: [GemmaPredictionRepeatabilityReport] = []
    ) -> GemmaParityDiagnosis {
        guard promptParity.exact else { return .promptPrefixMismatch }
        if !teacherForced.allLogitsFinite
            || cacheComparisons.contains(where: {
                !$0.allLogitsFinite || !$0.allCacheStatesFinite
            })
            || predictionRepeatability.contains(where: {
                $0.allLogitsFiniteByRepetition.contains(false)
                    || $0.allCacheStatesFiniteByRepetition?.contains(false) == true
            })
        {
            return .nonFiniteNumerics
        }
        if predictionRepeatability.first(where: {
            $0.context == .fullSequence
        })?.predictionsIdentical == false {
            return .fullSequenceNondeterminism
        }
        if predictionRepeatability.contains(where: {
            $0.context != .fullSequence && !$0.predictionsIdentical
        }) {
            return .cachedPredictionNondeterminism
        }
        if cacheComparisons.contains(where: {
            $0.prefillMode == .unchunked && !$0.predictionsMatchFullSequence
        }) {
            return .incrementalCacheDivergence
        }
        if cacheComparisons.contains(where: {
            $0.prefillMode == .balanced && !$0.predictionsMatchFullSequence
        }) {
            return .balancedPrefillDivergence
        }
        guard teacherForced.exact else { return .teacherForcedGreedyMismatch }
        return .noDetectedDivergence
    }

    private struct DiagnosticInputs: Sendable {
        let record: GemmaParityRecord
        let modelDirectory: String
        let adapterDirectory: String?
        let adapterScale: Float?
        let datasetPath: String
        let datasetSHA256: String
    }

    private struct PredictionRun {
        let predictions: [Int]
        let allLogitsFinite: Bool
        let firstNonFiniteLogitTargetOffset: Int?
    }

    private struct CachedPredictionRun {
        let predictions: [Int]
        let allLogitsFinite: Bool
        let firstNonFiniteLogitTargetOffset: Int?
        let allCacheStatesFinite: Bool
        let firstNonFiniteCacheTargetOffset: Int?
    }

    private static func fullSequencePredictions(
        model: any LanguageModel,
        partition: GemmaTrainingTokenPartition
    ) throws -> PredictionRun {
        let assistantStart = partition.assistantStart
        let assistantEnd = partition.assistantEnd
        guard assistantStart > 0, assistantEnd > assistantStart else {
            throw GemmaParityDiagnosticError.emptyAssistantSpan
        }
        // Match the trainer's exact input shape, including template tokens
        // after the supervised assistant span. Causal logits inside the span
        // should be unaffected, while shape-sensitive backend defects remain
        // observable instead of being hidden by a shortened diagnostic input.
        let inputTokens = Array(partition.completeTokens.dropLast())
        let logits = model(
            MLXArray(inputTokens).expandedDimensions(axis: 0), cache: nil)[0]
        let predictionLogits = logits[(assistantStart - 1)..<(assistantEnd - 1)]
        let predictions = argMax(predictionLogits, axis: -1)
        let finitePositions = isFinite(predictionLogits).all(axis: -1)
        eval(predictions, finitePositions)
        let result = predictions.asArray(Int.self)
        let positionHealth = finitePositions.asArray(Bool.self)
        guard result.count == partition.targetTokens.count else {
            throw GemmaParityDiagnosticError.predictionCountMismatch(
                label: "full-sequence", expected: partition.targetTokens.count,
                actual: result.count)
        }
        return PredictionRun(
            predictions: result,
            allLogitsFinite: positionHealth.allSatisfy({ $0 }),
            firstNonFiniteLogitTargetOffset:
                positionHealth.firstIndex(where: { !$0 }))
    }

    /// Mirrors the pinned TokenIterator prefill and decode shapes, but feeds
    /// the expected assistant token at each step. That isolates KV-cache
    /// numerical behavior from autoregressive exposure after a wrong token.
    private static func cachedPredictions(
        model: any LanguageModel,
        prefixTokens: [Int],
        targetTokens: [Int],
        mode: GemmaPrefillMode
    ) throws -> CachedPredictionRun {
        guard !prefixTokens.isEmpty, !targetTokens.isEmpty else {
            throw GemmaParityDiagnosticError.emptyAssistantSpan
        }
        var parameters = GenerateParameters(
            maxTokens: targetTokens.count, temperature: 0)
        if mode == .unchunked {
            parameters.prefill.chunking = .unchunked
        }
        let cache = try model.newCache(parameters: parameters)
        let input = LMInput(tokens: MLXArray(prefixTokens))
        var state: LMOutput.State?
        let firstLogits: MLXArray
        switch try model.prepare(
            input, cache: cache, state: nil, prefill: parameters.prefill)
        {
        case .tokens(let remaining):
            let output = withPreparedCache(cache, lengths: remaining.sequenceLengths) {
                model(remaining[text: .newAxis], cache: cache, state: state)
            }
            state = output.state
            firstLogits = output.logits
        case .logits(let output):
            state = output.state
            firstLogits = output.logits
        }

        var predictions = [Int]()
        predictions.reserveCapacity(targetTokens.count)
        var firstNonFiniteLogitTargetOffset: Int?
        var firstNonFiniteCacheTargetOffset: Int?
        var logits = firstLogits
        for targetOffset in targetTokens.indices {
            if targetOffset > 0 {
                let previous = LMInput.Text(tokens: MLXArray([targetTokens[targetOffset - 1]]))
                let output = withPreparedCache(cache, lengths: previous.sequenceLengths) {
                    model(previous[text: .newAxis], cache: cache, state: state)
                }
                state = output.state
                logits = output.logits
            }
            let selectedLogits = logits[0, -1]
            let prediction = argMax(selectedLogits, axis: -1)
            let finiteLogits = isFinite(selectedLogits).all()
            let finiteCacheStates = cache.flatMap(\.state).map {
                isFinite($0).all()
            }
            eval([prediction, finiteLogits] + finiteCacheStates)
            if !finiteLogits.item(Bool.self),
                firstNonFiniteLogitTargetOffset == nil
            {
                firstNonFiniteLogitTargetOffset = targetOffset
            }
            if finiteCacheStates.contains(where: { !$0.item(Bool.self) }),
                firstNonFiniteCacheTargetOffset == nil
            {
                firstNonFiniteCacheTargetOffset = targetOffset
            }
            predictions.append(prediction.item(Int.self))
        }
        return CachedPredictionRun(
            predictions: predictions,
            allLogitsFinite: firstNonFiniteLogitTargetOffset == nil,
            firstNonFiniteLogitTargetOffset: firstNonFiniteLogitTargetOffset,
            allCacheStatesFinite: firstNonFiniteCacheTargetOffset == nil,
            firstNonFiniteCacheTargetOffset: firstNonFiniteCacheTargetOffset)
    }

    private static func cacheRepeatability(
        mode: GemmaPrefillMode,
        runs: [CachedPredictionRun],
        assistantStart: Int,
        tokenText: (Int) -> String
    ) throws -> GemmaPredictionRepeatabilityReport {
        try GemmaPredictionRepeatabilityReport.make(
            context: mode == .balanced ? .balancedCache : .unchunkedCache,
            predictions: runs.map(\.predictions),
            assistantStart: assistantStart,
            allLogitsFiniteByRepetition: runs.map(\.allLogitsFinite),
            allCacheStatesFiniteByRepetition: runs.map(\.allCacheStatesFinite),
            tokenText: tokenText)
    }

    private static func effectiveCUDAGraphSetting(
        _ environment: [String: String]
    ) -> Bool? {
        guard MLXExecutionBackend.compiled == .cuda else { return nil }
        guard let raw = environment["MLX_USE_CUDA_GRAPHS"] else { return true }
        // MLX reads this setting with C `atoi`: unset means enabled and every
        // explicit value whose integer prefix is zero disables graph replay.
        return Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0 != 0
    }
}

public enum GemmaParityDiagnosticError: LocalizedError, Equatable {
    case versionedDatasetRequired
    case emptyRecordID
    case recordNotFound(String)
    case emptyTemplateTokens
    case emptyAssistantSpan
    case predictionCountMismatch(label: String, expected: Int, actual: Int)
    case invalidRepetitionEvidence(String)
    case incompatibleModel(String)

    public var errorDescription: String? {
        switch self {
        case .versionedDatasetRequired:
            "Gemma parity diagnostics require a versioned harness training envelope."
        case .emptyRecordID:
            "The requested training record id is empty."
        case .recordNotFound(let id):
            "Training record '\(id)' was not found."
        case .emptyTemplateTokens:
            "The tokenizer produced an empty chat-template sequence."
        case .emptyAssistantSpan:
            "The trainer's chat-template partition produced no supervised assistant tokens."
        case .predictionCountMismatch(let label, let expected, let actual):
            "\(label) produced \(actual) predictions; expected \(expected)."
        case .invalidRepetitionEvidence(let context):
            "Repeated prediction evidence for \(context) is empty or inconsistent."
        case .incompatibleModel(let type):
            "Gemma parity diagnostics require Gemma4Model or Gemma4TextModel, not \(type)."
        }
    }
}

private enum GemmaParityHashes {
    static func string(_ value: String) -> String {
        ScreeningReviewProvenance.sha256(value)
    }

    static func data(_ value: Data) -> String {
        ScreeningReviewProvenance.sha256(value)
    }

    static func tokenIDs(_ values: [Int]) -> String {
        string(values.map(String.init).joined(separator: ","))
    }
}

private func firstDifference(_ lhs: [Int], _ rhs: [Int]) -> Int? {
    let overlap = min(lhs.count, rhs.count)
    if let index = (0..<overlap).first(where: { lhs[$0] != rhs[$0] }) {
        return index
    }
    return lhs.count == rhs.count ? nil : overlap
}
