import Foundation

/// Configuration for the same-token, cross-condition prompt-end causal test.
///
/// The donor is the BF16 model with a saved LoRA loaded. The recipient is the
/// same freshly loaded BF16 model after the LoRA is unloaded. Because both
/// conditions consume the exact same prompt tokens, this diagnostic does not
/// inherit the response-token lexical confound of continuation-state patching.
public struct PromptEndPatchConfiguration: Codable, Sendable, Equatable {
    public static let absoluteMaximumCases = 16
    public static let absoluteMaximumLayers = 12
    public static let absoluteMaximumGenerationTokens = 192
    public static let defaultRandomSeed: UInt64 = 0xC0_A5_A1_7E

    public let layersZeroBased: [Int]
    public let caseOffset: Int
    public let maximumCases: Int
    public let maximumGenerationTokens: Int
    public let randomControlSeed: UInt64

    public init(
        layersZeroBased: [Int], caseOffset: Int = 0,
        maximumCases: Int = 4, maximumGenerationTokens: Int = 96,
        randomControlSeed: UInt64 = defaultRandomSeed
    ) throws {
        guard !layersZeroBased.isEmpty,
              layersZeroBased.count <= Self.absoluteMaximumLayers,
              layersZeroBased.allSatisfy({ $0 >= 0 }),
              Set(layersZeroBased).count == layersZeroBased.count
        else { throw PromptEndPatchError.invalidLayers(layersZeroBased) }
        guard caseOffset >= 0 else {
            throw PromptEndPatchError.invalidCaseOffset(caseOffset)
        }
        guard (1 ... Self.absoluteMaximumCases).contains(maximumCases) else {
            throw PromptEndPatchError.invalidMaximumCases(maximumCases)
        }
        guard (1 ... Self.absoluteMaximumGenerationTokens)
            .contains(maximumGenerationTokens)
        else {
            throw PromptEndPatchError.invalidGenerationTokens(
                maximumGenerationTokens)
        }
        self.layersZeroBased = layersZeroBased
        self.caseOffset = caseOffset
        self.maximumCases = maximumCases
        self.maximumGenerationTokens = maximumGenerationTokens
        self.randomControlSeed = randomControlSeed
    }

    public func validate(caseCount: Int, decoderLayerCount: Int) throws {
        guard caseOffset < caseCount else {
            throw PromptEndPatchError.caseOffsetOutsideInput(
                caseOffset, caseCount: caseCount)
        }
        guard layersZeroBased.allSatisfy({ $0 < decoderLayerCount }) else {
            throw PromptEndPatchError.layerOutsideDecoder(
                layers: layersZeroBased, decoderLayerCount: decoderLayerCount)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case layersZeroBased = "layers_zero_based"
        case caseOffset = "case_offset"
        case maximumCases = "maximum_cases"
        case maximumGenerationTokens = "maximum_generation_tokens"
        case randomControlSeed = "random_control_seed"
    }
}

public enum PromptEndPatchCondition: String, Codable, Sendable, Equatable {
    /// Exact donor prompt-end state substituted into the untouched model.
    case matchedDonor = "matched_donor"
    /// Same donor-minus-base delta with its sign reversed.
    case signReversed = "sign_reversed"
    /// An unrelated prompt's adapter delta, rescaled to the matched delta norm.
    case normMatchedRandom = "norm_matched_random"
}

public struct PromptEndPatchEffect: Codable, Sendable, Equatable {
    public let exactKLBaseToCondition: Double
    public let exactKLDonorToBase: Double
    public let exactKLDonorToCondition: Double
    public let donorClosenessGain: Double
    public let donorClosenessFraction: Double?
    public let donorTopTokenID: Int
    public let baseTopTokenID: Int
    public let conditionTopTokenID: Int

    public init(
        exactKLBaseToCondition: Double, exactKLDonorToBase: Double,
        exactKLDonorToCondition: Double, donorTopTokenID: Int,
        baseTopTokenID: Int, conditionTopTokenID: Int
    ) {
        self.exactKLBaseToCondition = exactKLBaseToCondition
        self.exactKLDonorToBase = exactKLDonorToBase
        self.exactKLDonorToCondition = exactKLDonorToCondition
        donorClosenessGain = exactKLDonorToBase - exactKLDonorToCondition
        donorClosenessFraction = exactKLDonorToBase > 1e-12
            ? donorClosenessGain / exactKLDonorToBase : nil
        self.donorTopTokenID = donorTopTokenID
        self.baseTopTokenID = baseTopTokenID
        self.conditionTopTokenID = conditionTopTokenID
    }

    private enum CodingKeys: String, CodingKey {
        case exactKLBaseToCondition = "exact_kl_base_to_condition"
        case exactKLDonorToBase = "exact_kl_donor_to_base"
        case exactKLDonorToCondition = "exact_kl_donor_to_condition"
        case donorClosenessGain = "donor_closeness_gain"
        case donorClosenessFraction = "donor_closeness_fraction"
        case donorTopTokenID = "donor_top_token_id"
        case baseTopTokenID = "base_top_token_id"
        case conditionTopTokenID = "condition_top_token_id"
    }
}

public struct PromptEndPatchResult: Codable, Sendable, Equatable {
    public let caseName: String
    public let category: String?
    public let layerZeroBased: Int
    public let promptTokenCount: Int
    public let matchedDeltaL2Norm: Double
    public let randomControlCaseName: String?
    public let matchedDonor: PromptEndPatchEffect
    public let signReversed: PromptEndPatchEffect
    public let normMatchedRandom: PromptEndPatchEffect?
    public let matchedDonorResponse: String
    public let signReversedResponse: String
    public let normMatchedRandomResponse: String?

    private enum CodingKeys: String, CodingKey {
        case caseName = "case_name"
        case category
        case layerZeroBased = "layer_zero_based"
        case promptTokenCount = "prompt_token_count"
        case matchedDeltaL2Norm = "matched_delta_l2_norm"
        case randomControlCaseName = "random_control_case_name"
        case matchedDonor = "matched_donor"
        case signReversed = "sign_reversed"
        case normMatchedRandom = "norm_matched_random"
        case matchedDonorResponse = "matched_donor_response"
        case signReversedResponse = "sign_reversed_response"
        case normMatchedRandomResponse = "norm_matched_random_response"
    }
}

public struct PromptEndPatchLayerSummary: Codable, Sendable, Equatable {
    public let layerZeroBased: Int
    public let resultCount: Int
    public let matchedMeanClosenessFraction: Double
    public let randomMeanClosenessFraction: Double?
    public let matchedMinusRandomMeanClosenessFraction: Double?
    public let matchedPositiveRate: Double
    public let matchedBeatsRandomRate: Double?

    private enum CodingKeys: String, CodingKey {
        case layerZeroBased = "layer_zero_based"
        case resultCount = "result_count"
        case matchedMeanClosenessFraction = "matched_mean_closeness_fraction"
        case randomMeanClosenessFraction = "random_mean_closeness_fraction"
        case matchedMinusRandomMeanClosenessFraction =
            "matched_minus_random_mean_closeness_fraction"
        case matchedPositiveRate = "matched_positive_rate"
        case matchedBeatsRandomRate = "matched_beats_random_rate"
    }
}

public struct PromptEndPatchUnloadValidation: Codable, Sendable, Equatable {
    public let maximumStateAbsoluteDifference: Double
    public let maximumLogProbabilityAbsoluteDifference: Double
    public let tolerance: Double
    public let passed: Bool

    public init(
        maximumStateAbsoluteDifference: Double,
        maximumLogProbabilityAbsoluteDifference: Double,
        tolerance: Double
    ) {
        self.maximumStateAbsoluteDifference = maximumStateAbsoluteDifference
        self.maximumLogProbabilityAbsoluteDifference =
            maximumLogProbabilityAbsoluteDifference
        self.tolerance = tolerance
        passed = maximumStateAbsoluteDifference <= tolerance
            && maximumLogProbabilityAbsoluteDifference <= tolerance
    }

    private enum CodingKeys: String, CodingKey {
        case maximumStateAbsoluteDifference =
            "maximum_state_absolute_difference"
        case maximumLogProbabilityAbsoluteDifference =
            "maximum_log_probability_absolute_difference"
        case tolerance
        case passed
    }
}

public struct PromptEndPatchStudy: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let status: String
    public let warning: String
    public let modelPath: String
    public let adapterPath: String
    public let adapterScale: Float
    public let inputSplit: String
    public let inputModelCondition: String?
    public let configuration: PromptEndPatchConfiguration
    public let cases: [MatchedResponsePatchCase]
    public let unloadValidation: PromptEndPatchUnloadValidation
    public let results: [PromptEndPatchResult]
    public let layerSummaries: [PromptEndPatchLayerSummary]

    public init(
        modelPath: String, adapterPath: String, adapterScale: Float,
        inputSplit: String, inputModelCondition: String?,
        configuration: PromptEndPatchConfiguration,
        cases: [MatchedResponsePatchCase],
        unloadValidation: PromptEndPatchUnloadValidation,
        results: [PromptEndPatchResult]
    ) {
        schemaVersion = 1
        status = "diagnostic_only"
        warning = "Prompt-end cross-condition patching localizes LoRA-caused state changes. It does not certify abliteration, generalization, capability preservation, or safe deployment."
        self.modelPath = modelPath
        self.adapterPath = adapterPath
        self.adapterScale = adapterScale
        self.inputSplit = inputSplit
        self.inputModelCondition = inputModelCondition
        self.configuration = configuration
        self.cases = cases
        self.unloadValidation = unloadValidation
        self.results = results
        layerSummaries = PromptEndPatchMath.summarize(results)
    }

    public func write(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(
            to: URL(fileURLWithPath: path).standardizedFileURL,
            options: .atomic)
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case status
        case warning
        case modelPath = "model_path"
        case adapterPath = "adapter_path"
        case adapterScale = "adapter_scale"
        case inputSplit = "input_split"
        case inputModelCondition = "input_model_condition"
        case configuration
        case cases
        case unloadValidation = "unload_validation"
        case results
        case layerSummaries = "layer_summaries"
    }
}

public enum PromptEndPatchMath {
    public static func adapterDelta(
        base: [Float], donor: [Float]
    ) throws -> [Float] {
        guard !base.isEmpty, base.count == donor.count else {
            throw PromptEndPatchError.incompatibleStates
        }
        return zip(donor, base).map(-)
    }

    public static func adding(
        _ delta: [Float], to base: [Float], scale: Float = 1
    ) throws -> [Float] {
        guard !base.isEmpty, base.count == delta.count,
              scale.isFinite
        else { throw PromptEndPatchError.incompatibleStates }
        return zip(base, delta).map { $0 + scale * $1 }
    }

    /// Applies an unrelated prompt's adapter delta to the target base state,
    /// rescaled to exactly the matched adapter-delta L2 norm.
    public static func normMatchedRandomReplacement(
        targetBase: [Float], matchedDelta: [Float],
        unrelatedBase: [Float], unrelatedDonor: [Float]
    ) throws -> [Float] {
        guard !targetBase.isEmpty,
              matchedDelta.count == targetBase.count,
              unrelatedBase.count == targetBase.count,
              unrelatedDonor.count == targetBase.count
        else { throw PromptEndPatchError.incompatibleStates }
        let unrelatedDelta = try adapterDelta(
            base: unrelatedBase, donor: unrelatedDonor)
        let matchedNorm = l2Norm(matchedDelta)
        let unrelatedNorm = l2Norm(unrelatedDelta)
        guard matchedNorm.isFinite, unrelatedNorm.isFinite,
              unrelatedNorm > 1e-12
        else { throw PromptEndPatchError.degenerateRandomControl }
        return try adding(
            unrelatedDelta, to: targetBase,
            scale: Float(matchedNorm / unrelatedNorm))
    }

    public static func effect(
        donorLogProbabilities: [Float], baseLogProbabilities: [Float],
        conditionLogProbabilities: [Float]
    ) throws -> PromptEndPatchEffect {
        let vocabulary = donorLogProbabilities.count
        guard vocabulary > 0,
              baseLogProbabilities.count == vocabulary,
              conditionLogProbabilities.count == vocabulary
        else { throw PromptEndPatchError.incompatibleMetrics }
        let donorToBase = try TeacherForcedContinuationMetricEngine.exactKL(
            baselineLogProbabilities: donorLogProbabilities,
            candidateLogProbabilities: baseLogProbabilities)
        let donorToCondition = try TeacherForcedContinuationMetricEngine.exactKL(
            baselineLogProbabilities: donorLogProbabilities,
            candidateLogProbabilities: conditionLogProbabilities)
        let baseToCondition = try TeacherForcedContinuationMetricEngine.exactKL(
            baselineLogProbabilities: baseLogProbabilities,
            candidateLogProbabilities: conditionLogProbabilities)
        return PromptEndPatchEffect(
            exactKLBaseToCondition: baseToCondition,
            exactKLDonorToBase: donorToBase,
            exactKLDonorToCondition: donorToCondition,
            donorTopTokenID: argmax(donorLogProbabilities),
            baseTopTokenID: argmax(baseLogProbabilities),
            conditionTopTokenID: argmax(conditionLogProbabilities))
    }

    public static func summarize(
        _ results: [PromptEndPatchResult]
    ) -> [PromptEndPatchLayerSummary] {
        Dictionary(grouping: results, by: \.layerZeroBased).keys.sorted().map {
            layer in
            let group = results.filter { $0.layerZeroBased == layer }
            let matched = group.compactMap {
                $0.matchedDonor.donorClosenessFraction
            }
            let random = group.compactMap {
                $0.normMatchedRandom?.donorClosenessFraction
            }
            let paired = group.compactMap { result -> Bool? in
                guard let random = result.normMatchedRandom?
                    .donorClosenessFraction,
                      let matched = result.matchedDonor.donorClosenessFraction
                else { return nil }
                return matched > random
            }
            let matchedMean = mean(matched)
            let randomMean = random.isEmpty ? nil : mean(random)
            return PromptEndPatchLayerSummary(
                layerZeroBased: layer,
                resultCount: group.count,
                matchedMeanClosenessFraction: matchedMean,
                randomMeanClosenessFraction: randomMean,
                matchedMinusRandomMeanClosenessFraction:
                    randomMean.map { matchedMean - $0 },
                matchedPositiveRate: rate(matched.map { $0 > 0 }),
                matchedBeatsRandomRate:
                    paired.isEmpty ? nil : rate(paired))
        }
    }

    public static func l2Norm(_ values: [Float]) -> Double {
        sqrt(values.reduce(0.0) { $0 + Double($1) * Double($1) })
    }

    public static func maximumAbsoluteDifference(
        _ lhs: [Float], _ rhs: [Float]
    ) -> Double {
        guard lhs.count == rhs.count else { return .infinity }
        return zip(lhs, rhs).reduce(0.0) {
            max($0, abs(Double($1.0 - $1.1)))
        }
    }

    private static func argmax(_ values: [Float]) -> Int {
        values.indices.max { values[$0] < values[$1] } ?? 0
    }

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func rate(_ values: [Bool]) -> Double {
        guard !values.isEmpty else { return 0 }
        return Double(values.count(where: { $0 })) / Double(values.count)
    }
}

public enum PromptEndPatchError: LocalizedError, Equatable {
    case invalidLayers([Int])
    case invalidCaseOffset(Int)
    case invalidMaximumCases(Int)
    case invalidGenerationTokens(Int)
    case caseOffsetOutsideInput(Int, caseCount: Int)
    case layerOutsideDecoder(layers: [Int], decoderLayerCount: Int)
    case unsupportedModel
    case incompatibleStates
    case incompatibleMetrics
    case degenerateRandomControl
    case promptTooLong(name: String, tokenCount: Int)
    case adapterUnloadValidationFailed(Double, Double, tolerance: Double)

    public var errorDescription: String? {
        switch self {
        case .invalidLayers(let layers):
            "Prompt-end patch layers are invalid: \(layers)."
        case .invalidCaseOffset(let value):
            "Prompt-end case offset must be non-negative, not \(value)."
        case .invalidMaximumCases(let value):
            "Prompt-end maximum cases must be 1...16, not \(value)."
        case .invalidGenerationTokens(let value):
            "Prompt-end generation tokens must be 1...192, not \(value)."
        case .caseOffsetOutsideInput(let offset, let count):
            "Prompt-end case offset \(offset) is outside \(count) input cases."
        case .layerOutsideDecoder(let layers, let count):
            "Prompt-end layers \(layers) are outside the \(count)-layer decoder."
        case .unsupportedModel:
            "Prompt-end patching currently requires Gemma 4."
        case .incompatibleStates:
            "Prompt-end patch states have incompatible shapes or values."
        case .incompatibleMetrics:
            "Prompt-end patch logits are incompatible."
        case .degenerateRandomControl:
            "The unrelated adapter delta is degenerate."
        case .promptTooLong(let name, let count):
            "Prompt-end case '\(name)' has \(count) tokens; maximum is 512."
        case .adapterUnloadValidationFailed(let state, let logits, let tolerance):
            "Adapter unload validation failed (state=\(state), logits=\(logits), tolerance=\(tolerance))."
        }
    }
}
