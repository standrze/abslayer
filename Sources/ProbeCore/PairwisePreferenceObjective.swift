import Foundation
import MLX
import MLXNN

/// Preference loss used by `abslayer-prefix-train`.
///
/// `simpoPairwise` is the preservation-aware default: every chosen/rejected
/// pair contributes one length-normalized margin before the batch mean is
/// taken. `legacyPooled` reproduces the original implementation, which pooled
/// all chosen tokens and all rejected tokens before constructing one margin.
public enum PairwisePreferenceObjectiveMode: String, Codable, Sendable {
    case simpoPairwise = "simpo-pairwise"
    case dpoOnset = "dpo-onset"
    case legacyPooled = "legacy-pooled"

    public var usesReferenceScores: Bool { self == .dpoOnset }

    public static func parse(_ value: String?) throws -> Self {
        switch value?.lowercased() {
        case nil, "", "simpo", "simpo-pairwise", "pairwise":
            .simpoPairwise
        case "dpo", "dpo-onset", "onset-dpo", "reference-relative-onset":
            .dpoOnset
        case "legacy", "legacy-pooled", "pooled":
            .legacyPooled
        default:
            throw PairwisePreferenceObjectiveError.unknownMode(value ?? "")
        }
    }
}

public enum PairwisePreferenceObjectiveError: LocalizedError, Equatable {
    case unknownMode(String)
    case mismatchedPairCounts
    case emptyPairs
    case insufficientGroups
    case invalidHyperparameter

    public var errorDescription: String? {
        switch self {
        case .unknownMode(let value):
            "Unknown preference objective '\(value)'; use simpo-pairwise, dpo-onset, or legacy-pooled."
        case .mismatchedPairCounts:
            "Chosen and rejected preference scores must have the same count."
        case .emptyPairs:
            "At least one chosen/rejected preference pair is required."
        case .insufficientGroups:
            "At least two distinct prompt groups are required for leak-free validation."
        case .invalidHyperparameter:
            "Preference beta must be positive and gamma/SFT weight must be non-negative."
        }
    }
}

public struct PairwisePreferenceScalarMetrics: Equatable, Sendable {
    public let loss: Float
    public let preferenceLoss: Float
    public let chosenNegativeLogLikelihood: Float
    public let pairwiseAccuracy: Float
}

/// Scalar reference implementation used to specify and test the differentiable
/// MLX implementation in the trainer.
public enum PairwisePreferenceMath {
    public static func simpo(
        chosenNegativeLogLikelihood: [Float],
        rejectedNegativeLogLikelihood: [Float],
        beta: Float,
        gamma: Float,
        sftWeight: Float
    ) throws -> PairwisePreferenceScalarMetrics {
        guard chosenNegativeLogLikelihood.count
                == rejectedNegativeLogLikelihood.count
        else { throw PairwisePreferenceObjectiveError.mismatchedPairCounts }
        guard !chosenNegativeLogLikelihood.isEmpty else {
            throw PairwisePreferenceObjectiveError.emptyPairs
        }
        guard beta.isFinite, beta > 0, gamma.isFinite, gamma >= 0,
              sftWeight.isFinite, sftWeight >= 0
        else { throw PairwisePreferenceObjectiveError.invalidHyperparameter }

        var preferenceLoss: Float = 0
        var correct = 0
        for (chosen, rejected) in zip(
            chosenNegativeLogLikelihood, rejectedNegativeLogLikelihood)
        {
            let margin = beta * (rejected - chosen) - gamma
            preferenceLoss += softplus(-margin)
            if chosen < rejected { correct += 1 }
        }
        let count = Float(chosenNegativeLogLikelihood.count)
        preferenceLoss /= count
        let chosenLoss = chosenNegativeLogLikelihood.reduce(0, +) / count
        return PairwisePreferenceScalarMetrics(
            loss: preferenceLoss + sftWeight * chosenLoss,
            preferenceLoss: preferenceLoss,
            chosenNegativeLogLikelihood: chosenLoss,
            pairwiseAccuracy: Float(correct) / count)
    }

    /// Reference-relative DPO over length- or onset-normalized sequence
    /// negative log likelihoods. At the untouched reference policy the
    /// preference margin is exactly zero, regardless of answer length or the
    /// reference model's pre-existing refusal preference.
    public static func dpo(
        chosenNegativeLogLikelihood: [Float],
        rejectedNegativeLogLikelihood: [Float],
        referenceChosenNegativeLogLikelihood: [Float],
        referenceRejectedNegativeLogLikelihood: [Float],
        beta: Float,
        gamma: Float,
        sftWeight: Float
    ) throws -> PairwisePreferenceScalarMetrics {
        let count = chosenNegativeLogLikelihood.count
        guard count == rejectedNegativeLogLikelihood.count,
              count == referenceChosenNegativeLogLikelihood.count,
              count == referenceRejectedNegativeLogLikelihood.count
        else { throw PairwisePreferenceObjectiveError.mismatchedPairCounts }
        guard count > 0 else { throw PairwisePreferenceObjectiveError.emptyPairs }
        guard beta.isFinite, beta > 0, gamma.isFinite, gamma >= 0,
              sftWeight.isFinite, sftWeight >= 0
        else { throw PairwisePreferenceObjectiveError.invalidHyperparameter }

        var preferenceLoss: Float = 0
        var correct = 0
        for index in 0 ..< count {
            let policyAdvantage = rejectedNegativeLogLikelihood[index]
                - chosenNegativeLogLikelihood[index]
            let referenceAdvantage = referenceRejectedNegativeLogLikelihood[index]
                - referenceChosenNegativeLogLikelihood[index]
            let margin = beta * (policyAdvantage - referenceAdvantage) - gamma
            preferenceLoss += softplus(-margin)
            if policyAdvantage > referenceAdvantage { correct += 1 }
        }
        let denominator = Float(count)
        preferenceLoss /= denominator
        let chosenLoss = chosenNegativeLogLikelihood.reduce(0, +) / denominator
        return PairwisePreferenceScalarMetrics(
            loss: preferenceLoss + sftWeight * chosenLoss,
            preferenceLoss: preferenceLoss,
            chosenNegativeLogLikelihood: chosenLoss,
            pairwiseAccuracy: Float(correct) / denominator)
    }

    private static func softplus(_ value: Float) -> Float {
        if value > 0 {
            value + log1p(exp(-value))
        } else {
            log1p(exp(value))
        }
    }
}

public struct PairwisePreferenceTensorMetrics {
    public let loss: MLXArray
    public let preferenceLoss: MLXArray
    public let chosenNegativeLogLikelihood: MLXArray
    public let pairwiseAccuracy: MLXArray
}

/// Differentiable MLX implementation used directly by the trainer. Token
/// losses have shape `[pairs, tokens]`; masks prevent padding and prompt tokens
/// from changing either sequence's normalized score.
public enum PairwisePreferenceTensorMath {
    /// Converts a binary assistant-token mask into an onset-weighted scoring
    /// mask: the first token receives 8x weight, tokens 2...4 receive 4x,
    /// and later tokens receive 1x. Prompt and padding positions remain zero.
    public static func onsetWeights(mask: MLXArray) -> MLXArray {
        let positions = mask.cumsum(axis: -1)
        let weights = MLX.where(
            positions .<= 1,
            MLXArray(Float(8)),
            MLX.where(
                positions .<= 4,
                MLXArray(Float(4)),
                MLXArray(Float(1))))
        return weights * mask
    }

    public static func evaluate(
        chosenTokenLosses: MLXArray,
        chosenMask: MLXArray,
        rejectedTokenLosses: MLXArray,
        rejectedMask: MLXArray,
        mode: PairwisePreferenceObjectiveMode,
        beta: Float,
        gamma: Float,
        sftWeight: Float,
        referenceChosenNegativeLogLikelihood: MLXArray? = nil,
        referenceRejectedNegativeLogLikelihood: MLXArray? = nil
    ) -> PairwisePreferenceTensorMetrics {
        let chosenCounts = maximum(
            chosenMask.sum(axis: -1), MLXArray(1))
        let rejectedCounts = maximum(
            rejectedMask.sum(axis: -1), MLXArray(1))
        let chosenNLL = chosenTokenLosses.sum(axis: -1) / chosenCounts
        let rejectedNLL = rejectedTokenLosses.sum(axis: -1) / rejectedCounts
        let pooledChosen = chosenTokenLosses.sum()
            / maximum(chosenMask.sum(), MLXArray(1))
        let pooledRejected = rejectedTokenLosses.sum()
            / maximum(rejectedMask.sum(), MLXArray(1))

        let chosenLoss: MLXArray
        let preferenceLoss: MLXArray
        switch mode {
        case .simpoPairwise:
            chosenLoss = chosenNLL.mean()
            let margins = MLXArray(beta) * (rejectedNLL - chosenNLL)
                - MLXArray(gamma)
            preferenceLoss = (-MLXNN.logSigmoid(margins)).mean()
        case .dpoOnset:
            guard let referenceChosenNegativeLogLikelihood,
                  let referenceRejectedNegativeLogLikelihood
            else {
                preconditionFailure("dpo-onset requires fixed reference scores")
            }
            chosenLoss = chosenNLL.mean()
            let policyAdvantage = rejectedNLL - chosenNLL
            let referenceAdvantage = referenceRejectedNegativeLogLikelihood
                - referenceChosenNegativeLogLikelihood
            let margins = MLXArray(beta)
                * (policyAdvantage - referenceAdvantage)
                - MLXArray(gamma)
            preferenceLoss = (-MLXNN.logSigmoid(margins)).mean()
        case .legacyPooled:
            chosenLoss = pooledChosen
            let margin = MLXArray(beta) * (pooledRejected - pooledChosen)
                - MLXArray(gamma)
            preferenceLoss = -MLXNN.logSigmoid(margin)
        }
        let pairwiseAccuracy: MLXArray
        if mode == .dpoOnset {
            pairwiseAccuracy = ((rejectedNLL - chosenNLL) .>
                (referenceRejectedNegativeLogLikelihood!
                    - referenceChosenNegativeLogLikelihood!))
                .asType(.float32).mean()
        } else {
            pairwiseAccuracy = less(chosenNLL, rejectedNLL)
                .asType(.float32).mean()
        }
        return PairwisePreferenceTensorMetrics(
            loss: preferenceLoss + MLXArray(sftWeight) * chosenLoss,
            preferenceLoss: preferenceLoss,
            chosenNegativeLogLikelihood: chosenLoss,
            pairwiseAccuracy: pairwiseAccuracy)
    }
}

/// SplitMix64 gives deterministic shuffles on every supported Swift platform;
/// the standard library's default RNG is intentionally nondeterministic.
public struct ABSlayerSeededGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }
}

public struct PreferenceGroupSplit: Equatable, Sendable {
    public let trainingIndices: [Int]
    public let validationIndices: [Int]
    public let trainingGroupCount: Int
    public let validationGroupCount: Int

    /// Splits whole prompt groups, not rows. Repeated/weighted preference rows
    /// for one prompt therefore cannot leak from training into validation.
    public static func make(
        groupKeys: [String], validationFraction: Double = 0.18,
        seed: UInt64
    ) throws -> Self {
        guard !groupKeys.isEmpty else {
            throw PairwisePreferenceObjectiveError.emptyPairs
        }
        guard validationFraction.isFinite,
              validationFraction > 0, validationFraction < 1
        else { throw PairwisePreferenceObjectiveError.invalidHyperparameter }

        var seen = Set<String>()
        var groups = [String]()
        for key in groupKeys where seen.insert(key).inserted {
            groups.append(key)
        }
        guard groups.count > 1 else {
            throw PairwisePreferenceObjectiveError.insufficientGroups
        }
        var generator = ABSlayerSeededGenerator(seed: seed)
        groups.shuffle(using: &generator)
        let validationCount = min(
            groups.count - 1,
            max(1, Int(ceil(Double(groups.count) * validationFraction))))
        let validationGroups = Set(groups.prefix(validationCount))
        let trainingIndices = groupKeys.indices.filter {
            !validationGroups.contains(groupKeys[$0])
        }
        let validationIndices = groupKeys.indices.filter {
            validationGroups.contains(groupKeys[$0])
        }
        return Self(
            trainingIndices: trainingIndices,
            validationIndices: validationIndices,
            trainingGroupCount: groups.count - validationCount,
            validationGroupCount: validationCount)
    }
}
