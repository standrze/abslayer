import Foundation
import MLX

/// Strict configuration for the optional teacher-forced continuation-KL
/// regularizer in `abslayer-prefix-train`.
///
/// The feature is disabled when both the prompt path and weight are absent.
/// Supplying only one is an error so a misspelled or forgotten setting cannot
/// silently turn a preservation constraint off.
public struct ContinuationKLRegularizerOptions: Equatable, Sendable {
    public static let promptPathKey = "ABSLAYER_CONTINUATION_KL_PROMPTS"
    public static let weightKey = "ABSLAYER_CONTINUATION_KL_WEIGHT"
    public static let maximumCasesKey = "ABSLAYER_CONTINUATION_KL_MAX_CASES"
    public static let maximumTokensKey = "ABSLAYER_CONTINUATION_KL_MAX_TOKENS"
    public static let topKKey = "ABSLAYER_CONTINUATION_KL_TOP_K"
    public static let tailWeightKey = "ABSLAYER_CONTINUATION_KL_TAIL_WEIGHT"
    public static let tailThresholdKey = "ABSLAYER_CONTINUATION_KL_TAIL_THRESHOLD"

    public let promptPath: String?
    public let weight: Float
    public let maximumCases: Int
    public let maximumTokens: Int
    public let topK: Int
    public let tailWeight: Float
    public let tailThreshold: Float

    public var isEnabled: Bool { promptPath != nil }

    public static func parse(
        _ environment: [String: String]
    ) throws -> Self {
        let promptPath = environment[promptPathKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if environment[promptPathKey] != nil, promptPath?.isEmpty != false {
            throw ContinuationKLRegularizerError.invalidValue(
                key: promptPathKey, value: environment[promptPathKey] ?? "")
        }
        let weight = try finiteFloat(
            environment, key: weightKey, fallback: 0)
        let maximumCases = try positiveInteger(
            environment, key: maximumCasesKey, fallback: 16)
        let maximumTokens = try positiveInteger(
            environment, key: maximumTokensKey, fallback: 16)
        let topK = try positiveInteger(
            environment, key: topKKey, fallback: 256)
        let tailWeight = try finiteFloat(
            environment, key: tailWeightKey, fallback: 0)
        let tailThreshold = try finiteFloat(
            environment, key: tailThresholdKey, fallback: 1)

        guard weight >= 0 else {
            throw ContinuationKLRegularizerError.invalidValue(
                key: weightKey, value: environment[weightKey] ?? String(weight))
        }
        guard tailWeight >= 0 else {
            throw ContinuationKLRegularizerError.invalidValue(
                key: tailWeightKey,
                value: environment[tailWeightKey] ?? String(tailWeight))
        }
        guard tailThreshold >= 0 else {
            throw ContinuationKLRegularizerError.invalidValue(
                key: tailThresholdKey,
                value: environment[tailThresholdKey] ?? String(tailThreshold))
        }
        if promptPath == nil {
            guard weight == 0, tailWeight == 0 else {
                throw ContinuationKLRegularizerError.promptPathRequired
            }
        } else if weight <= 0 {
            throw ContinuationKLRegularizerError.positiveWeightRequired
        }
        return Self(
            promptPath: promptPath,
            weight: weight,
            maximumCases: maximumCases,
            maximumTokens: maximumTokens,
            topK: topK,
            tailWeight: tailWeight,
            tailThreshold: tailThreshold)
    }

    private static func positiveInteger(
        _ environment: [String: String], key: String, fallback: Int
    ) throws -> Int {
        guard let raw = environment[key] else { return fallback }
        guard let value = Int(raw), value > 0 else {
            throw ContinuationKLRegularizerError.invalidValue(
                key: key, value: raw)
        }
        return value
    }

    private static func finiteFloat(
        _ environment: [String: String], key: String, fallback: Float
    ) throws -> Float {
        guard let raw = environment[key] else { return fallback }
        guard let value = Float(raw), value.isFinite else {
            throw ContinuationKLRegularizerError.invalidValue(
                key: key, value: raw)
        }
        return value
    }
}

public enum ContinuationKLRegularizerError: LocalizedError, Equatable {
    case invalidValue(key: String, value: String)
    case promptPathRequired
    case positiveWeightRequired
    case missingReferenceResponses
    case malformedPartition

    public var errorDescription: String? {
        switch self {
        case .invalidValue(let key, let value):
            "\(key) has an invalid value '\(value)'."
        case .promptPathRequired:
            "ABSLAYER_CONTINUATION_KL_PROMPTS is required when continuation-KL or its tail hinge has a positive weight."
        case .positiveWeightRequired:
            "ABSLAYER_CONTINUATION_KL_WEIGHT must be positive when ABSLAYER_CONTINUATION_KL_PROMPTS is set."
        case .missingReferenceResponses:
            "The continuation-KL prompt file has no non-empty controlReferenceResponse values."
        case .malformedPartition:
            "A top-K-plus-tail probability partition is empty, mismatched, non-finite, or not normalized."
        }
    }
}

/// One untouched-base top-K-plus-tail sketch over only the logits that predict
/// a fixed reference assistant continuation. `inputTokenIDs` still contains
/// the deployed prompt prefix because the candidate must receive identical
/// context; the final `positions.count` logits are the only scored positions.
public struct ContinuationKLCaseFingerprint: Equatable, Sendable {
    public let name: String
    public let inputTokenIDs: [Int]
    public let positions: [ContinuationKLPositionFingerprint]

    public init(
        name: String,
        inputTokenIDs: [Int],
        positions: [ContinuationKLPositionFingerprint]
    ) {
        self.name = name
        self.inputTokenIDs = inputTokenIDs
        self.positions = positions
    }
}

public struct ContinuationKLPositionFingerprint: Equatable, Sendable {
    public let supportTokenIDs: [Int]
    public let supportLogProbabilities: [Float]
    public let tailLogProbability: Float

    public init(
        supportTokenIDs: [Int],
        supportLogProbabilities: [Float],
        tailLogProbability: Float
    ) {
        self.supportTokenIDs = supportTokenIDs
        self.supportLogProbabilities = supportLogProbabilities
        self.tailLogProbability = tailLogProbability
    }
}

public enum ContinuationKLPositionSelection {
    /// The input is `deployedPrompt + referenceContinuation`, with the final
    /// token dropped to form next-token inputs. Consequently, exactly the
    /// final `continuationPositionCount` logits predict reference-assistant
    /// tokens; all earlier logits predict prompt-prefix tokens.
    public static func suffixRange(
        totalLogitCount: Int, continuationPositionCount: Int
    ) -> Range<Int>? {
        guard continuationPositionCount > 0,
              totalLogitCount >= continuationPositionCount
        else { return nil }
        return (totalLogitCount - continuationPositionCount) ..< totalLogitCount
    }
}

public enum ContinuationKLReferenceSelection {
    /// Selects evenly across all eligible development controls after filtering
    /// out rows without a fixed reference response.
    public static func select(
        _ pairs: [PromptPair], maximum: Int
    ) throws -> [PromptPair] {
        guard maximum > 0 else {
            throw ContinuationKLRegularizerError.invalidValue(
                key: ContinuationKLRegularizerOptions.maximumCasesKey,
                value: String(maximum))
        }
        let eligible = pairs.filter {
            $0.controlReferenceResponse?.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty == false
        }
        guard !eligible.isEmpty else {
            throw ContinuationKLRegularizerError.missingReferenceResponses
        }
        guard eligible.count > maximum else { return eligible }
        return (0 ..< maximum).map {
            eligible[$0 * eligible.count / maximum]
        }
    }
}

public struct ContinuationKLScalarComponents: Equatable, Sendable {
    public let meanPositionKL: Double
    public let meanTailExcess: Double

    public func regularizer(tailWeight: Double) -> Double {
        meanPositionKL + tailWeight * meanTailExcess
    }
}

/// Scalar specification for the differentiable top-K-plus-tail loss.
public enum ContinuationKLMath {
    public static func partitionDivergence(
        baselineSupportLogProbabilities: [Float],
        candidateSupportLogProbabilities: [Float],
        baselineTailLogProbability: Float,
        candidateTailLogProbability: Float
    ) throws -> Double {
        guard !baselineSupportLogProbabilities.isEmpty,
              baselineSupportLogProbabilities.count
                == candidateSupportLogProbabilities.count
        else { throw ContinuationKLRegularizerError.malformedPartition }
        let baselineLogs = baselineSupportLogProbabilities
            + [baselineTailLogProbability]
        let candidateLogs = candidateSupportLogProbabilities
            + [candidateTailLogProbability]
        guard baselineLogs.allSatisfy({ $0.isFinite && $0 <= 0.000_01 }),
              candidateLogs.allSatisfy({ $0.isFinite && $0 <= 0.000_01 })
        else { throw ContinuationKLRegularizerError.malformedPartition }
        let baselineMass = baselineLogs.reduce(0.0) { $0 + exp(Double($1)) }
        let candidateMass = candidateLogs.reduce(0.0) { $0 + exp(Double($1)) }
        guard abs(baselineMass - 1) <= 0.001,
              abs(candidateMass - 1) <= 0.001
        else { throw ContinuationKLRegularizerError.malformedPartition }
        let divergence = zip(baselineLogs, candidateLogs).reduce(0.0) {
            $0 + exp(Double($1.0)) * Double($1.0 - $1.1)
        }
        return max(0, divergence)
    }

    public static func components(
        positionDivergences: [Double], tailThreshold: Double
    ) throws -> ContinuationKLScalarComponents {
        guard !positionDivergences.isEmpty,
              positionDivergences.allSatisfy({ $0.isFinite && $0 >= 0 }),
              tailThreshold.isFinite, tailThreshold >= 0
        else { throw ContinuationKLRegularizerError.malformedPartition }
        let count = Double(positionDivergences.count)
        return ContinuationKLScalarComponents(
            meanPositionKL: positionDivergences.reduce(0, +) / count,
            meanTailExcess: positionDivergences.reduce(0) {
                $0 + max($1 - tailThreshold, 0)
            } / count)
    }
}

/// Differentiable implementation used by the trainer. The baseline support is
/// fixed; candidate tail mass is the exact complement of candidate mass on
/// those same support IDs.
public enum ContinuationKLTensorMath {
    public static func components(
        candidateLogProbabilities: MLXArray,
        supportTokenIDs: MLXArray,
        baselineSupportLogProbabilities: MLXArray,
        baselineTailLogProbabilities: MLXArray,
        tailThreshold: Float
    ) -> (meanPositionKL: MLXArray, meanTailExcess: MLXArray) {
        let candidateSupportLogs = takeAlong(
            candidateLogProbabilities, supportTokenIDs, axis: -1)
        let retainedMass = exp(candidateSupportLogs).sum(axis: -1)
        let candidateTailLogs = log(clip(
            1 - retainedMass, min: Float(1e-30)))
        let supportKL = (
            exp(baselineSupportLogProbabilities)
                * (baselineSupportLogProbabilities - candidateSupportLogs)
        ).sum(axis: -1)
        let tailKL = exp(baselineTailLogProbabilities)
            * (baselineTailLogProbabilities - candidateTailLogs)
        let positionKL = maximum(supportKL + tailKL, MLXArray(0))
        return (
            positionKL.mean(),
            maximum(positionKL - MLXArray(tailThreshold), MLXArray(0)).mean())
    }
}
