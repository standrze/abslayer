#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import ProbeCore
import Tokenizers

private enum PrefixTrainError: LocalizedError {
    case usage
    case emptyDataset
    case incompatibleModel
    case incompatibleRetentionData

    var errorDescription: String? {
        switch self {
        case .usage:
            "usage: abslayer-prefix-train MODEL_FOLDER PROMPTS_JSON ADAPTER_DIR [ITERS] [LR]"
        case .emptyDataset: "The training prompt array is empty."
        case .incompatibleModel: "The model does not expose trainable LoRA layers."
        case .incompatibleRetentionData:
            "The exact-KL fingerprint and prompt file are missing or incompatible."
        }
    }
}

@main
enum ABSlayerPrefixTrain {
    private struct TokenizedCompletion {
        let groupKey: String
        let tokens: [Int]
        let assistantStart: Int
        let assistantEnd: Int
        let rejectedTokens: [Int]?
        let rejectedStart: Int?
        let rejectedEnd: Int?
        let referenceChosenNLL: Float?
        let referenceRejectedNLL: Float?
    }

    private struct ExactKLCase {
        let inputTokens: [Int]
        let baselineLogProbabilities: [Float]
    }

    static func main() async throws {
        let args = CommandLine.arguments
        guard (4 ... 6).contains(args.count) else { throw PrefixTrainError.usage }
        let modelURL = URL(fileURLWithPath: args[1]).standardizedFileURL
        let promptsURL = URL(fileURLWithPath: args[2]).standardizedFileURL
        let adapterURL = URL(fileURLWithPath: args[3]).standardizedFileURL
        let iterations = args.count >= 5 ? (Int(args[4]) ?? 80) : 80
        let learningRate = args.count >= 6 ? (Float(args[5]) ?? 1e-5) : 1e-5
        let environment = ProcessInfo.processInfo.environment
        let klFingerprint = try environment["ABSLAYER_KL_FINGERPRINT"].map {
            try SequenceLogitFingerprint.read(from: $0)
        }
        let klWeight = Float(environment["ABSLAYER_KL_WEIGHT"] ?? "0") ?? 0
        let exactKLFingerprint = try environment["ABSLAYER_EXACT_KL_FINGERPRINT"].map {
            try LogitFingerprint.read(from: $0)
        }
        let exactKLPrompts = try environment["ABSLAYER_EXACT_KL_PROMPTS"].map {
            try PromptFile.load($0)
        }
        let exactKLWeight = Float(
            environment["ABSLAYER_EXACT_KL_WEIGHT"] ?? "0") ?? 0
        let continuationKLOptions = try ContinuationKLRegularizerOptions.parse(
            environment)
        let continuationKLPairs: [PromptPair]
        if let path = continuationKLOptions.promptPath {
            continuationKLPairs = try PromptFile.load(path)
        } else {
            continuationKLPairs = []
        }
        let resumeAdapter = environment["ABSLAYER_RESUME_ADAPTER"].map {
            URL(fileURLWithPath: $0).standardizedFileURL
        }
        let checkpointEvery = Int(environment["ABSLAYER_CHECKPOINT_EVERY"] ?? "250") ?? 250
        let maxAssistantTokens = Int(environment["ABSLAYER_MAX_ASSISTANT_TOKENS"] ?? "0") ?? 0
        let batchSize = Int(environment["ABSLAYER_BATCH_SIZE"] ?? "1") ?? 1
        let loraRank = Int(environment["ABSLAYER_LORA_RANK"] ?? "16") ?? 16
        let loraScale = Float(environment["ABSLAYER_LORA_SCALE"] ?? "32") ?? 32
        let requestedLoRALayers = environment["ABSLAYER_LORA_LAYERS"].flatMap(Int.init)
        let loraDropout = Float(environment["ABSLAYER_LORA_DROPOUT"] ?? "0.05") ?? 0.05
        let defaultKeys = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ]
        let loraKeys = environment["ABSLAYER_LORA_KEYS"].map {
            $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        } ?? defaultKeys
        let preferenceBeta = Float(environment["ABSLAYER_PREFERENCE_BETA"] ?? "2") ?? 2
        let preferenceGamma = Float(environment["ABSLAYER_PREFERENCE_GAMMA"] ?? "0.5") ?? 0.5
        let preferenceSFTWeight = Float(environment["ABSLAYER_PREFERENCE_SFT_WEIGHT"] ?? "0.2") ?? 0.2
        let loraWeightDecay = Float(environment["ABSLAYER_LORA_WEIGHT_DECAY"] ?? "0") ?? 0
        let preferenceObjective = try PairwisePreferenceObjectiveMode.parse(
            environment["ABSLAYER_PREFERENCE_OBJECTIVE"])
        guard let randomSeed = UInt64(environment["ABSLAYER_RANDOM_SEED"] ?? "1729")
        else { throw PrefixTrainError.usage }
        guard klWeight >= 0, exactKLWeight >= 0, batchSize > 0,
              loraRank > 0, requestedLoRALayers.map({ $0 > 0 }) != false,
              !loraKeys.isEmpty, preferenceBeta.isFinite, preferenceBeta > 0,
              preferenceGamma.isFinite, preferenceGamma >= 0,
              preferenceSFTWeight.isFinite, preferenceSFTWeight >= 0,
              loraWeightDecay.isFinite, loraWeightDecay >= 0
        else { throw PrefixTrainError.usage }
        guard (exactKLFingerprint == nil) == (exactKLPrompts == nil) else {
            throw PrefixTrainError.incompatibleRetentionData
        }
        if let fingerprint = exactKLFingerprint, let pairs = exactKLPrompts {
            guard !pairs.isEmpty,
                  fingerprint.promptNames == pairs.map(\.name),
                  fingerprint.logProbabilities.count == pairs.count,
                  fingerprint.logProbabilities.allSatisfy({
                      $0.count == fingerprint.vocabularySize
                  })
            else { throw PrefixTrainError.incompatibleRetentionData }
        }
        let usesSequenceKL = klFingerprint != nil && klWeight > 0
        let usesExactKL = exactKLFingerprint != nil && exactKLWeight > 0
        let usesContinuationKL = continuationKLOptions.isEnabled
        let completions = try HarnessTrainingDatasetLoader.load(promptsURL.path)
        guard !completions.isEmpty else { throw PrefixTrainError.emptyDataset }
        let hasPreferences = completions.allSatisfy { $0.rejected != nil }
        guard hasPreferences || completions.allSatisfy({ $0.rejected == nil }) else {
            throw PrefixTrainError.incompatibleModel
        }
        guard !preferenceObjective.usesReferenceScores || hasPreferences else {
            throw PrefixTrainError.incompatibleModel
        }

        try MLXResourceGuard.apply(environment: environment)
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(
                directory: modelURL, extraEOSTokens: ["<end_of_turn>"]))
        try FileManager.default.createDirectory(
            at: adapterURL, withIntermediateDirectories: true)

        try await container.perform { context in
            guard let loraModel = context.model as? LoRAModel else {
                throw PrefixTrainError.incompatibleModel
            }
            // Do not inherit a model-specific suffix length. By default the
            // repair adapter exposes every decoder layer; callers can still
            // request a smaller suffix explicitly for compute experiments.
            let loraLayers = requestedLoRALayers ?? loraModel.loraLayers.count
            let layerSelection = try LoRASuffixLayerSelection.resolve(
                requestedCount: loraLayers,
                availableCount: loraModel.loraLayers.count)
            print(
                "LoRA layers are zero-based and half-open: "
                    + "\(layerSelection.startIndex)..<\(layerSelection.endIndex) "
                    + "(last \(layerSelection.requestedCount) of "
                    + "\(layerSelection.availableCount))")
            let configuration = LoRAConfiguration(
                numLayers: loraLayers,
                loraParameters: .init(
                    rank: loraRank, scale: loraScale, dropout: loraDropout,
                    keys: loraKeys))

            // Capture the fixed teacher distributions while this is still the
            // untouched base model. Materialize every row as Swift values
            // before replacing any Linear modules with LoRA wrappers.
            let continuationKLCases: [ContinuationKLCaseFingerprint]
            if usesContinuationKL {
                let selected = try ContinuationKLReferenceSelection.select(
                    continuationKLPairs,
                    maximum: continuationKLOptions.maximumCases)
                context.model.train(false)
                var captured = [ContinuationKLCaseFingerprint]()
                captured.reserveCapacity(selected.count)
                for (index, pair) in selected.enumerated() {
                    guard let referenceResponse = pair.controlReferenceResponse else {
                        throw ContinuationKLRegularizerError.missingReferenceResponses
                    }
                    let promptTokens = try context.tokenizer.applyChatTemplate(
                        messages: [["role": "user", "content": pair.control]])
                    guard !promptTokens.isEmpty, promptTokens.count <= 512 else {
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
                        derived.prefix(continuationKLOptions.maximumTokens))
                    guard !continuation.isEmpty else {
                        throw ContinuationKLRegularizerError.missingReferenceResponses
                    }
                    let inputTokens = Array(
                        (promptTokens + continuation).dropLast())
                    let input = MLXArray(inputTokens).expandedDimensions(axis: 0)
                    let allLogits = context.model(input, cache: nil)[0]
                        .asType(.float32)
                    guard let assistantLogitRange =
                        ContinuationKLPositionSelection.suffixRange(
                            totalLogitCount: allLogits.dim(0),
                            continuationPositionCount: continuation.count)
                    else {
                        throw ContinuationKLRegularizerError.malformedPartition
                    }
                    let baselineLogs = MLXNN.logSoftmax(
                        allLogits[assistantLogitRange],
                        axis: -1)
                    let vocabulary = baselineLogs.dim(-1)
                    guard continuationKLOptions.topK < vocabulary else {
                        throw ContinuationKLRegularizerError.invalidValue(
                            key: ContinuationKLRegularizerOptions.topKKey,
                            value: String(continuationKLOptions.topK))
                    }
                    let supportIDs = argPartition(
                        baselineLogs,
                        kth: -continuationKLOptions.topK,
                        axis: -1
                    )[.ellipsis, (-continuationKLOptions.topK)...]
                    let supportLogs = takeAlong(
                        baselineLogs, supportIDs, axis: -1)
                    let retainedMass = exp(supportLogs).sum(axis: -1)
                    let tailLogs = log(clip(
                        1 - retainedMass, min: Float(1e-30)))
                    eval(supportIDs, supportLogs, tailLogs)

                    // Converting to Swift arrays is intentional: retained MLX
                    // graphs must not be evaluated after LoRA installation.
                    let flatIDs = supportIDs.asArray(Int.self)
                    let flatLogs = supportLogs.asArray(Float.self)
                    let tailValues = tailLogs.asArray(Float.self)
                    let positions = continuation.indices.map { position in
                        let range = position * continuationKLOptions.topK
                            ..< (position + 1) * continuationKLOptions.topK
                        return ContinuationKLPositionFingerprint(
                            supportTokenIDs: Array(flatIDs[range]),
                            supportLogProbabilities: Array(flatLogs[range]),
                            tailLogProbability: tailValues[position])
                    }
                    captured.append(ContinuationKLCaseFingerprint(
                        name: pair.name,
                        inputTokenIDs: inputTokens,
                        positions: positions))
                    print(
                        "untouched continuation-KL reference "
                            + "\(index + 1)/\(selected.count): \(pair.name) "
                            + "(\(continuation.count) assistant tokens)")
                }
                continuationKLCases = captured
            } else {
                continuationKLCases = []
            }

            _ = try LoRAContainer.from(model: context.model, configuration: configuration)
            // Reference-relative preference training must anchor to the policy
            // at optimizer initialization. For a resumed run that is the
            // resumed adapter, while untouched-model KL remains a separate
            // retention constraint.
            if let resumeAdapter {
                let resumeWeights = resumeAdapter.appendingPathComponent("adapters.safetensors")
                try LoRATrain.loadLoRAWeights(model: context.model, url: resumeWeights)
                print("Resuming LoRA weights from \(resumeWeights.path)")
            }

            var examples = [TokenizedCompletion]()
            examples.reserveCapacity(completions.count)
            func tokenize(_ prompt: String, _ target: String) throws -> ([Int], Int, Int) {
                // Match the deployed ChatSession prefix exactly: the tokenizer's
                // short overload renders a user turn with the assistant
                // generation header. Render the completed turn without a second
                // generation header, then supervise the full completion suffix,
                // including the template's end-of-turn token. The old
                // full-vs-empty common-suffix heuristic deliberately masked that
                // stop token and gave the model no explicit termination target.
                let promptTokens = try context.tokenizer.applyChatTemplate(messages: [
                    ["role": "user", "content": prompt]
                ])
                let completeTokens = try context.tokenizer.applyChatTemplate(messages: [
                    ["role": "user", "content": prompt],
                    ["role": "assistant", "content": target],
                ], tools: nil, additionalContext: ["add_generation_prompt": false])
                let assistant = try SupervisedCompletionBoundary.resolve(
                    promptTokens: promptTokens, completedTokens: completeTokens)
                return (completeTokens, assistant.lowerBound, assistant.upperBound)
            }
            func referenceNLL(
                _ tokens: [Int], _ start: Int, _ end: Int
            ) -> Float {
                let inputs = MLXArray(Array(tokens.dropLast()))
                    .expandedDimensions(axis: 0)
                let targets = MLXArray(Array(tokens.dropFirst()))
                    .expandedDimensions(axis: 0)
                var mask = [Float](repeating: 0, count: tokens.count - 1)
                let firstLogit = max(0, start - 1)
                let lastLogit = min(mask.count, end - 1)
                if firstLogit < lastLogit {
                    for index in firstLogit ..< lastLogit { mask[index] = 1 }
                }
                let binaryMask = MLXArray(mask).expandedDimensions(axis: 0)
                let scoringMask = PairwisePreferenceTensorMath.onsetWeights(
                    mask: binaryMask)
                let logits = (context.model as! any LLMModel)(
                    inputs, cache: nil).asType(.float32)
                let tokenLosses = crossEntropy(
                    logits: logits, targets: targets) * scoringMask
                let nll = tokenLosses.sum()
                    / maximum(scoringMask.sum(), MLXArray(1))
                eval(nll)
                return nll.item(Float.self)
            }
            context.model.train(false)
            for (completionIndex, completion) in completions.enumerated() {
                var chosen = try tokenize(completion.prompt, completion.target)
                if maxAssistantTokens > 0,
                   chosen.2 - chosen.1 > maxAssistantTokens
                {
                    let truncatedEnd = chosen.1 + maxAssistantTokens
                    chosen = (Array(chosen.0[..<truncatedEnd]), chosen.1, truncatedEnd)
                }
                var rejected = try completion.rejected.map { try tokenize(completion.prompt, $0) }
                if maxAssistantTokens > 0, let value = rejected,
                   value.2 - value.1 > maxAssistantTokens
                {
                    let truncatedEnd = value.1 + maxAssistantTokens
                    rejected = (Array(value.0[..<truncatedEnd]), value.1, truncatedEnd)
                }
                let referenceChosenNLL: Float?
                let referenceRejectedNLL: Float?
                if preferenceObjective.usesReferenceScores {
                    referenceChosenNLL = referenceNLL(chosen.0, chosen.1, chosen.2)
                    referenceRejectedNLL = rejected.map {
                        referenceNLL($0.0, $0.1, $0.2)
                    }
                    if completionIndex == 0 || (completionIndex + 1) % 25 == 0 {
                        print(
                            "reference-scored \(completionIndex + 1)/\(completions.count) pairs")
                    }
                } else {
                    referenceChosenNLL = nil
                    referenceRejectedNLL = nil
                }
                examples.append(TokenizedCompletion(
                    groupKey: completion.prompt,
                    tokens: chosen.0,
                    assistantStart: chosen.1,
                    assistantEnd: chosen.2,
                    rejectedTokens: rejected?.0,
                    rejectedStart: rejected?.1,
                    rejectedEnd: rejected?.2,
                    referenceChosenNLL: referenceChosenNLL,
                    referenceRejectedNLL: referenceRejectedNLL))
            }
            if preferenceObjective.usesReferenceScores {
                let chosen = examples.compactMap(\.referenceChosenNLL)
                let rejected = examples.compactMap(\.referenceRejectedNLL)
                let count = Float(max(chosen.count, 1))
                let basePreference = zip(chosen, rejected).filter {
                    $0.0 < $0.1
                }.count
                print(String(
                    format: "fixed reference onset NLL: chosen %.6f rejected %.6f, raw chosen preference %.2f%%",
                    chosen.reduce(0, +) / count,
                    rejected.reduce(0, +) / count,
                    100 * Float(basePreference) / count))
            }
            let exactKLCases: [ExactKLCase]
            if let fingerprint = exactKLFingerprint, let pairs = exactKLPrompts {
                exactKLCases = try zip(pairs, fingerprint.logProbabilities).map { entry in
                    let (pair, baseline) = entry
                    let tokens = try context.tokenizer.applyChatTemplate(messages: [
                        ["role": "user", "content": pair.control]
                    ])
                    guard tokens.count >= 2, tokens.count <= 512 else {
                        throw PrefixTrainError.incompatibleRetentionData
                    }
                    return ExactKLCase(
                        inputTokens: tokens,
                        baselineLogProbabilities: baseline)
                }
            } else {
                exactKLCases = []
            }
            // Split complete prompt groups so repeated/weighted rows cannot
            // leak into validation, while retaining deterministic assignment.
            let split = try PreferenceGroupSplit.make(
                groupKeys: examples.map(\.groupKey), seed: randomSeed)
            let train = split.trainingIndices.map { examples[$0] }
            let valid = split.validationIndices.map { examples[$0] }
            print(
                "exact token training: train=\(train.count), valid=\(valid.count), "
                    + "train-groups=\(split.trainingGroupCount), "
                    + "valid-groups=\(split.validationGroupCount), "
                    + "assistant tokens=\(examples.map { $0.assistantEnd - $0.assistantStart }.min() ?? 0)..."
                    + "\(examples.map { $0.assistantEnd - $0.assistantStart }.max() ?? 0)")
            if usesSequenceKL, let klFingerprint {
                print(
                    "KL retention: cases=\(klFingerprint.cases.count), "
                        + "topK=\(klFingerprint.topK), weight=\(klWeight)")
            }
            if usesExactKL, let exactKLFingerprint {
                print(
                    "Exact first-token KL retention: cases="
                        + "\(exactKLFingerprint.logProbabilities.count), "
                        + "vocabulary=\(exactKLFingerprint.vocabularySize), "
                        + "weight=\(exactKLWeight)")
            }
            if usesContinuationKL {
                print(
                    "Teacher-forced continuation KL retention: cases="
                        + "\(continuationKLCases.count), "
                        + "max-tokens=\(continuationKLOptions.maximumTokens), "
                        + "topK=\(continuationKLOptions.topK), "
                        + "weight=\(continuationKLOptions.weight), "
                        + "tail-weight=\(continuationKLOptions.tailWeight), "
                        + "tail-threshold=\(continuationKLOptions.tailThreshold)")
            }
            if hasPreferences {
                print(
                    "preference objective: mode=\(preferenceObjective.rawValue), "
                        + "beta=\(preferenceBeta), gamma=\(preferenceGamma), "
                        + "SFT-weight=\(preferenceSFTWeight), seed=\(randomSeed)")
            }
            if loraWeightDecay > 0 {
                print("LoRA AdamW weight decay: \(loraWeightDecay)")
            }

            func arrays(for example: TokenizedCompletion) -> [MLXArray] {
                func sequenceArrays(
                    _ tokens: [Int], _ start: Int, _ end: Int
                ) -> [MLXArray] {
                    let inputs = MLXArray(Array(tokens.dropLast()))
                        .expandedDimensions(axis: 0)
                    let targets = MLXArray(Array(tokens.dropFirst()))
                        .expandedDimensions(axis: 0)
                    var mask = [Float](repeating: 0, count: tokens.count - 1)
                    let firstLogit = max(0, start - 1)
                    let lastLogit = min(mask.count, end - 1)
                    if firstLogit < lastLogit {
                        for index in firstLogit ..< lastLogit { mask[index] = 1 }
                    }
                    return [inputs, targets, MLXArray(mask).expandedDimensions(axis: 0)]
                }
                var result = sequenceArrays(
                    example.tokens, example.assistantStart, example.assistantEnd)
                if let tokens = example.rejectedTokens,
                   let start = example.rejectedStart, let end = example.rejectedEnd
                {
                    result.append(contentsOf: sequenceArrays(tokens, start, end))
                }
                if preferenceObjective.usesReferenceScores {
                    result.append(MLXArray([example.referenceChosenNLL!]))
                    result.append(MLXArray([example.referenceRejectedNLL!]))
                }
                return result
            }
            func arrays(for batch: [TokenizedCompletion]) -> [MLXArray] {
                guard batch.count > 1 else {
                    return arrays(for: batch[0])
                }

                func paddedSequenceArrays(
                    _ sequences: [([Int], Int, Int)]
                ) -> [MLXArray] {
                    let maximumLength = sequences.map { $0.0.count - 1 }.max() ?? 1
                    var allInputs = [Int]()
                    var allTargets = [Int]()
                    var allMasks = [Float]()
                    allInputs.reserveCapacity(sequences.count * maximumLength)
                    allTargets.reserveCapacity(sequences.count * maximumLength)
                    allMasks.reserveCapacity(sequences.count * maximumLength)
                    for (tokens, start, end) in sequences {
                        let length = tokens.count - 1
                        var inputs = Array(tokens.dropLast())
                        var targets = Array(tokens.dropFirst())
                        var mask = [Float](repeating: 0, count: length)
                        let firstLogit = max(0, start - 1)
                        let lastLogit = min(length, end - 1)
                        if firstLogit < lastLogit {
                            for index in firstLogit ..< lastLogit { mask[index] = 1 }
                        }
                        if length < maximumLength {
                            inputs += [Int](repeating: 0, count: maximumLength - length)
                            targets += [Int](repeating: 0, count: maximumLength - length)
                            mask += [Float](repeating: 0, count: maximumLength - length)
                        }
                        allInputs += inputs
                        allTargets += targets
                        allMasks += mask
                    }
                    return [
                        MLXArray(allInputs, [sequences.count, maximumLength]),
                        MLXArray(allTargets, [sequences.count, maximumLength]),
                        MLXArray(allMasks, [sequences.count, maximumLength]),
                    ]
                }

                let chosen = paddedSequenceArrays(batch.map {
                    ($0.tokens, $0.assistantStart, $0.assistantEnd)
                })
                guard hasPreferences else { return chosen }
                let rejected = paddedSequenceArrays(batch.map { example in
                    (
                        example.rejectedTokens!,
                        example.rejectedStart!,
                        example.rejectedEnd!
                    )
                })
                var result = chosen + rejected
                if preferenceObjective.usesReferenceScores {
                    result.append(MLXArray(batch.map { $0.referenceChosenNLL! }))
                    result.append(MLXArray(batch.map { $0.referenceRejectedNLL! }))
                }
                return result
            }
            func preferenceTerms(
                languageModel: any LLMModel, arrays: [MLXArray]
            ) -> [MLXArray] {
                let inputs = arrays[0]
                let targets = arrays[1]
                let mask = arrays[2]
                let chosenScoringMask = preferenceObjective == .dpoOnset
                    ? PairwisePreferenceTensorMath.onsetWeights(mask: mask)
                    : mask
                let logits = languageModel(inputs, cache: nil).asType(.float32)
                let chosenTokenLosses = crossEntropy(
                    logits: logits, targets: targets) * chosenScoringMask
                let chosenTokenCounts = maximum(
                    chosenScoringMask.sum(), MLXArray(1))
                var completionLoss = chosenTokenLosses.sum() / chosenTokenCounts
                var objective = completionLoss
                var preferenceLoss = MLXArray(0)
                var pairwiseAccuracy = MLXArray(0)
                if hasPreferences {
                    let rejectedInputs = arrays[3]
                    let rejectedTargets = arrays[4]
                    let rejectedMask = arrays[5]
                    let rejectedScoringMask = preferenceObjective == .dpoOnset
                        ? PairwisePreferenceTensorMath.onsetWeights(mask: rejectedMask)
                        : rejectedMask
                    let rejectedLogits = languageModel(
                        rejectedInputs, cache: nil).asType(.float32)
                    let rejectedTokenLosses = crossEntropy(
                        logits: rejectedLogits, targets: rejectedTargets)
                        * rejectedScoringMask
                    let metrics = PairwisePreferenceTensorMath.evaluate(
                        chosenTokenLosses: chosenTokenLosses,
                        chosenMask: chosenScoringMask,
                        rejectedTokenLosses: rejectedTokenLosses,
                        rejectedMask: rejectedScoringMask,
                        mode: preferenceObjective,
                        beta: preferenceBeta,
                        gamma: preferenceGamma,
                        sftWeight: preferenceSFTWeight,
                        referenceChosenNegativeLogLikelihood:
                            preferenceObjective.usesReferenceScores ? arrays[6] : nil,
                        referenceRejectedNegativeLogLikelihood:
                            preferenceObjective.usesReferenceScores ? arrays[7] : nil)
                    completionLoss = metrics.chosenNegativeLogLikelihood
                    preferenceLoss = metrics.preferenceLoss
                    pairwiseAccuracy = metrics.pairwiseAccuracy
                    objective = metrics.loss
                }
                return [
                    objective, completionLoss, preferenceLoss,
                    pairwiseAccuracy,
                ]
            }

            let lossValueGrad = valueAndGrad(model: context.model) { model, arrays in
                let languageModel = model as! any LLMModel
                let preference = preferenceTerms(
                    languageModel: languageModel, arrays: arrays)
                let objective = preference[0]
                let completionLoss = preference[1]
                let preferenceLoss = preference[2]
                var nextIndex = hasPreferences
                    ? (preferenceObjective.usesReferenceScores ? 8 : 6)
                    : 3
                var sequenceRetentionLoss = MLXArray(0)
                if usesSequenceKL {
                    let klInputs = arrays[nextIndex]
                    let supportIDs = arrays[nextIndex + 1]
                    let baseSupportLogs = arrays[nextIndex + 2]
                    let baseTailLogs = arrays[nextIndex + 3]
                    let positionCount = supportIDs.dim(0)
                    let candidateLogits = languageModel(
                        klInputs, cache: nil)[0, 0 ..< positionCount]
                        .asType(.float32)
                    let candidateLogs = MLXNN.logSoftmax(candidateLogits, axis: -1)
                    let candidateSupportLogs = takeAlong(
                        candidateLogs, supportIDs, axis: -1)
                    let retainedMass = exp(candidateSupportLogs).sum(axis: -1)
                    let candidateTailLogs = log(clip(
                        1 - retainedMass, min: Float(1e-30)))
                    let supportKL = (exp(baseSupportLogs)
                        * (baseSupportLogs - candidateSupportLogs)).sum(axis: -1)
                    let tailKL = exp(baseTailLogs) * (baseTailLogs - candidateTailLogs)
                    sequenceRetentionLoss = maximum(
                        (supportKL + tailKL).mean(), MLXArray(0))
                    nextIndex += 4
                }
                var exactRetentionLoss = MLXArray(0)
                if usesExactKL {
                    let exactInputs = arrays[nextIndex]
                    let baselineLogs = arrays[nextIndex + 1]
                    let candidateLogs = MLXNN.logSoftmax(
                        languageModel(exactInputs, cache: nil)[0, -1]
                            .asType(.float32),
                        axis: -1)
                    exactRetentionLoss = maximum(
                        (exp(baselineLogs) * (baselineLogs - candidateLogs)).sum(),
                        MLXArray(0))
                    nextIndex += 2
                }
                var continuationRetentionLoss = MLXArray(0)
                var continuationTailExcess = MLXArray(0)
                if usesContinuationKL {
                    let continuationInputs = arrays[nextIndex]
                    let supportIDs = arrays[nextIndex + 1]
                    let baselineSupportLogs = arrays[nextIndex + 2]
                    let baselineTailLogs = arrays[nextIndex + 3]
                    let positionCount = supportIDs.dim(0)
                    let allCandidateLogits = languageModel(
                        continuationInputs, cache: nil)[0].asType(.float32)
                    // Reference-answer positions are a suffix of the deployed
                    // prompt-plus-reference input. Cropping the final N logits
                    // is the mask: prompt-prefix logits never enter this KL.
                    guard let assistantLogitRange =
                        ContinuationKLPositionSelection.suffixRange(
                            totalLogitCount: allCandidateLogits.dim(0),
                            continuationPositionCount: positionCount)
                    else {
                        preconditionFailure(
                            "validated continuation-KL case has invalid boundaries")
                    }
                    let candidateLogs = MLXNN.logSoftmax(
                        allCandidateLogits[assistantLogitRange],
                        axis: -1)
                    let components = ContinuationKLTensorMath.components(
                        candidateLogProbabilities: candidateLogs,
                        supportTokenIDs: supportIDs,
                        baselineSupportLogProbabilities: baselineSupportLogs,
                        baselineTailLogProbabilities: baselineTailLogs,
                        tailThreshold: continuationKLOptions.tailThreshold)
                    continuationRetentionLoss = components.meanPositionKL
                    continuationTailExcess = components.meanTailExcess
                }
                let loss = objective
                    + MLXArray(klWeight) * sequenceRetentionLoss
                    + MLXArray(exactKLWeight) * exactRetentionLoss
                    + MLXArray(continuationKLOptions.weight)
                        * (continuationRetentionLoss
                            + MLXArray(continuationKLOptions.tailWeight)
                                * continuationTailExcess)
                return [
                    loss, completionLoss, sequenceRetentionLoss,
                    exactRetentionLoss, continuationRetentionLoss,
                    continuationTailExcess, preferenceLoss,
                ]
            }

            func klArrays(for item: SequenceCaseFingerprint) -> [MLXArray] {
                let positionCount = item.positions.count
                let topK = item.positions.first?.supportTokenIDs.count ?? 0
                return [
                    MLXArray(Array(item.tokenIDs.dropLast())).expandedDimensions(axis: 0),
                    MLXArray(item.positions.flatMap(\.supportTokenIDs), [positionCount, topK]),
                    MLXArray(item.positions.flatMap(\.supportLogProbabilities), [positionCount, topK]),
                    MLXArray(item.positions.map(\.tailLogProbability)),
                ]
            }

            func continuationKLArrays(
                for item: ContinuationKLCaseFingerprint
            ) -> [MLXArray] {
                let positionCount = item.positions.count
                return [
                    MLXArray(item.inputTokenIDs).expandedDimensions(axis: 0),
                    MLXArray(
                        item.positions.flatMap(\.supportTokenIDs),
                        [positionCount, continuationKLOptions.topK]),
                    MLXArray(
                        item.positions.flatMap(\.supportLogProbabilities),
                        [positionCount, continuationKLOptions.topK]),
                    MLXArray(item.positions.map(\.tailLogProbability)),
                ]
            }

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            func saveAdapter(to directory: URL) throws {
                try FileManager.default.createDirectory(
                    at: directory, withIntermediateDirectories: true)
                try LoRATrain.saveLoRAWeights(
                    model: context.model,
                    url: directory.appendingPathComponent("adapters.safetensors"))
                try encoder.encode(configuration).write(
                    to: directory.appendingPathComponent("adapter_config.json"),
                    options: .atomic)
            }
            func evaluateValidation(step: Int, resumeTraining: Bool) {
                context.model.train(false)
                defer {
                    if resumeTraining { context.model.train() }
                }
                var lossTotal: Float = 0
                var assistantLossTotal: Float = 0
                var preferenceLossTotal: Float = 0
                var pairwiseCorrectTotal: Float = 0
                var evaluated = 0
                for start in stride(from: 0, to: valid.count, by: batchSize) {
                    let end = min(start + batchSize, valid.count)
                    let batch = Array(valid[start ..< end])
                    let terms = preferenceTerms(
                        languageModel: context.model as! any LLMModel,
                        arrays: arrays(for: batch))
                    eval(terms)
                    let count = Float(batch.count)
                    lossTotal += terms[0].item(Float.self) * count
                    assistantLossTotal += terms[1].item(Float.self) * count
                    preferenceLossTotal += terms[2].item(Float.self) * count
                    pairwiseCorrectTotal += terms[3].item(Float.self) * count
                    evaluated += batch.count
                }
                let denominator = Float(max(evaluated, 1))
                var line = String(
                    format: "Validation step %d: loss %.6f assistant %.6f",
                    step, lossTotal / denominator,
                    assistantLossTotal / denominator)
                if hasPreferences {
                    line += String(
                        format: " preference %.6f pairwise-accuracy %.2f%% (%d pairs)",
                        preferenceLossTotal / denominator,
                        100 * pairwiseCorrectTotal / denominator,
                        evaluated)
                }
                print(line)
            }
            let optimizer: Adam = loraWeightDecay > 0
                ? AdamW(
                    learningRate: learningRate,
                    weightDecay: loraWeightDecay)
                : Adam(learningRate: learningRate)
            context.model.train()
            var order = Array(train.indices)
            var trainingGenerator = ABSlayerSeededGenerator(
                seed: randomSeed ^ 0xD1B54A32D192ED03)
            order.shuffle(using: &trainingGenerator)
            var orderCursor = 0
            for iteration in 0 ..< iterations {
                var batch = [TokenizedCompletion]()
                batch.reserveCapacity(batchSize)
                for _ in 0 ..< batchSize {
                    if orderCursor == order.count {
                        order.shuffle(using: &trainingGenerator)
                        orderCursor = 0
                    }
                    batch.append(train[order[orderCursor]])
                    orderCursor += 1
                }
                var trainingArrays = arrays(for: batch)
                if usesSequenceKL, let klFingerprint {
                    trainingArrays.append(contentsOf: klArrays(
                        for: klFingerprint.cases[iteration % klFingerprint.cases.count]))
                }
                if usesExactKL {
                    let item = exactKLCases[iteration % exactKLCases.count]
                    trainingArrays.append(
                        MLXArray(item.inputTokens).expandedDimensions(axis: 0))
                    trainingArrays.append(MLXArray(item.baselineLogProbabilities))
                }
                if usesContinuationKL {
                    trainingArrays.append(contentsOf: continuationKLArrays(
                        for: continuationKLCases[
                            iteration % continuationKLCases.count]))
                }
                let (result, gradients) = lossValueGrad(
                    context.model, trainingArrays)
                optimizer.update(model: context.model, gradients: gradients)
                eval(context.model, optimizer, result[0])
                if iteration == 0 || (iteration + 1) % 5 == 0 {
                    print(
                        "Iteration \(iteration + 1): total \(result[0].item(Float.self)) "
                            + "assistant \(result[1].item(Float.self)) "
                            + "sequence-KL \(result[2].item(Float.self)) "
                            + "exact-KL \(result[3].item(Float.self)) "
                            + "continuation-KL \(result[4].item(Float.self)) "
                            + "continuation-tail-hinge \(result[5].item(Float.self)) "
                            + "preference \(result[6].item(Float.self))")
                }
                if checkpointEvery > 0, (iteration + 1) % checkpointEvery == 0 {
                    evaluateValidation(
                        step: iteration + 1, resumeTraining: true)
                    let checkpoint = adapterURL
                        .appendingPathComponent("checkpoints")
                        .appendingPathComponent(String(
                            format: "step-%04d", iteration + 1))
                    try saveAdapter(to: checkpoint)
                    print("Checkpoint saved at \(checkpoint.path)")
                }
            }
            context.model.train(false)
            evaluateValidation(step: iterations, resumeTraining: false)
            try saveAdapter(to: adapterURL)
            print("Saved prompt-masked cyber refusal adapter to \(adapterURL.path)")
        }
    }
}
