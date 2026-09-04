import Foundation

public struct SequencePositionFingerprint: Codable, Sendable, Equatable {
    public let targetTokenID: Int
    public let supportTokenIDs: [Int]
    public let supportLogProbabilities: [Float]
    public let tailLogProbability: Float
    public let targetLogProbability: Float

    public init(
        targetTokenID: Int, supportTokenIDs: [Int],
        supportLogProbabilities: [Float], tailLogProbability: Float,
        targetLogProbability: Float
    ) {
        self.targetTokenID = targetTokenID
        self.supportTokenIDs = supportTokenIDs
        self.supportLogProbabilities = supportLogProbabilities
        self.tailLogProbability = tailLogProbability
        self.targetLogProbability = targetLogProbability
    }
}

public struct SequenceCaseFingerprint: Codable, Sendable, Equatable {
    public let name: String
    public let tokenIDs: [Int]
    public let positions: [SequencePositionFingerprint]

    public init(
        name: String, tokenIDs: [Int], positions: [SequencePositionFingerprint]
    ) {
        self.name = name
        self.tokenIDs = tokenIDs
        self.positions = positions
    }
}

/// A compact control-prompt-prefix distribution sketch. Every position keeps
/// the baseline model's top-K tokens and aggregates the rest of the vocabulary
/// into one tail bucket. Comparing two sketches gives exact KL on that
/// partition, which is a lower bound on full-vocabulary KL by the log-sum
/// inequality. It does not measure an assistant response trajectory.
public struct SequenceLogitFingerprint: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let vocabularySize: Int
    public let topK: Int
    public let cases: [SequenceCaseFingerprint]

    public init(
        vocabularySize: Int, topK: Int, cases: [SequenceCaseFingerprint],
        schemaVersion: Int = 1
    ) {
        self.schemaVersion = schemaVersion
        self.vocabularySize = vocabularySize
        self.topK = topK
        self.cases = cases
    }

    public func write(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(self).write(
            to: URL(fileURLWithPath: path), options: .atomic)
    }

    public static func read(from path: String) throws -> Self {
        let value = try JSONDecoder().decode(
            Self.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        guard value.schemaVersion == 1, value.topK > 0,
              value.vocabularySize > value.topK, !value.cases.isEmpty
        else { throw FingerprintError.malformed(path) }
        return value
    }
}

public struct MeanConfidenceInterval: Codable, Sendable, Equatable {
    public let estimate: Double
    public let lowerBound: Double
    public let upperBound: Double
    public let confidenceLevel: Double
    public let sampleCount: Int

    public static func normal95(_ samples: [Double]) -> Self {
        guard !samples.isEmpty else {
            return Self(
                estimate: 0, lowerBound: 0, upperBound: 0,
                confidenceLevel: 0.95, sampleCount: 0)
        }
        let mean = samples.reduce(0, +) / Double(samples.count)
        guard samples.count > 1 else {
            return Self(
                estimate: mean, lowerBound: mean, upperBound: mean,
                confidenceLevel: 0.95, sampleCount: 1)
        }
        let variance = samples.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
            / Double(samples.count - 1)
        let halfWidth = 1.959_963_984_540_054 * sqrt(variance / Double(samples.count))
        return Self(
            estimate: mean, lowerBound: mean - halfWidth, upperBound: mean + halfWidth,
            confidenceLevel: 0.95, sampleCount: samples.count)
    }
}

public struct SequenceCaseMetrics: Sendable, Equatable {
    public let name: String
    public let tokenCount: Int
    public let coarsenedKLLowerBound: Double
    public let baselinePerplexity: Double
    public let candidatePerplexity: Double
    public let baselineSupportMass: Double
}

public struct SequenceMetricSummary: Sendable, Equatable {
    public let cases: [SequenceCaseMetrics]
    public let tokenCount: Int
    public let coarsenedKLLowerBound: Double
    public let caseMeanKL95: MeanConfidenceInterval
    public let baselinePerplexity: Double
    public let candidatePerplexity: Double
    public let perplexityRatio: Double
    public let baselineSupportMass: Double

    /// Honest name for the legacy `coarsenedKLLowerBound` field. The legacy
    /// spelling remains source-compatible with existing studies and callers.
    public var controlPromptPrefixKLLowerBound: Double {
        coarsenedKLLowerBound
    }

    public var rendered: String {
        [
            String(
                format: "Control prompt-prefix KL lower bound (top-K + tail): %.6f over %d tokens",
                coarsenedKLLowerBound, tokenCount),
            String(
                format: "Case-mean control prompt-prefix KL: %.6f (95%% CI %.6f–%.6f; n=%d)",
                caseMeanKL95.estimate, caseMeanKL95.lowerBound,
                caseMeanKL95.upperBound, caseMeanKL95.sampleCount),
            String(
                format: "Baseline/candidate perplexity: %.4f / %.4f (ratio %.4f)",
                baselinePerplexity, candidatePerplexity, perplexityRatio),
            String(
                format: "Baseline top-K support mass: %.2f%%", 100 * baselineSupportMass),
            "NOTE: this covers teacher-forced control prompt tokens only; it is a coarsened lower bound, not assistant-continuation or full-vocabulary KL.",
        ].joined(separator: "\n")
    }
}

public enum SequenceMetricEngine {
    public static func compare(
        baseline: SequenceLogitFingerprint,
        candidate: SequenceLogitFingerprint
    ) throws -> SequenceMetricSummary {
        guard baseline.schemaVersion == candidate.schemaVersion,
              baseline.vocabularySize == candidate.vocabularySize,
              baseline.topK == candidate.topK,
              baseline.cases.count == candidate.cases.count,
              !baseline.cases.isEmpty
        else { throw FingerprintError.incompatible }

        var caseMetrics = [SequenceCaseMetrics]()
        var totalKL = 0.0
        var baselineNLL = 0.0
        var candidateNLL = 0.0
        var supportMass = 0.0
        var tokenCount = 0

        for (baseCase, candidateCase) in zip(baseline.cases, candidate.cases) {
            guard baseCase.name == candidateCase.name,
                  baseCase.tokenIDs == candidateCase.tokenIDs,
                  baseCase.positions.count == candidateCase.positions.count,
                  !baseCase.positions.isEmpty
            else { throw FingerprintError.incompatible }

            var caseKL = 0.0
            var caseBaselineNLL = 0.0
            var caseCandidateNLL = 0.0
            var caseSupportMass = 0.0
            for (base, trial) in zip(baseCase.positions, candidateCase.positions) {
                guard base.targetTokenID == trial.targetTokenID,
                      base.supportTokenIDs == trial.supportTokenIDs,
                      base.supportLogProbabilities.count == baseline.topK,
                      trial.supportLogProbabilities.count == baseline.topK
                else { throw FingerprintError.incompatible }

                for (logPFloat, logQFloat) in zip(
                    base.supportLogProbabilities, trial.supportLogProbabilities
                ) {
                    let logP = Double(logPFloat)
                    caseKL += exp(logP) * (logP - Double(logQFloat))
                }
                let logTailP = Double(base.tailLogProbability)
                caseKL += exp(logTailP)
                    * (logTailP - Double(trial.tailLogProbability))
                caseBaselineNLL -= Double(base.targetLogProbability)
                caseCandidateNLL -= Double(trial.targetLogProbability)
                caseSupportMass += 1 - exp(logTailP)
            }
            // Floating-point tail aggregation can produce tiny negative values.
            caseKL = max(0, caseKL / Double(baseCase.positions.count))
            let count = baseCase.positions.count
            let metrics = SequenceCaseMetrics(
                name: baseCase.name,
                tokenCount: count,
                coarsenedKLLowerBound: caseKL,
                baselinePerplexity: exp(caseBaselineNLL / Double(count)),
                candidatePerplexity: exp(caseCandidateNLL / Double(count)),
                baselineSupportMass: caseSupportMass / Double(count))
            caseMetrics.append(metrics)
            totalKL += caseKL * Double(count)
            baselineNLL += caseBaselineNLL
            candidateNLL += caseCandidateNLL
            supportMass += caseSupportMass
            tokenCount += count
        }
        guard tokenCount > 0 else { throw FingerprintError.incompatible }
        let baselinePerplexity = exp(baselineNLL / Double(tokenCount))
        let candidatePerplexity = exp(candidateNLL / Double(tokenCount))
        return SequenceMetricSummary(
            cases: caseMetrics,
            tokenCount: tokenCount,
            coarsenedKLLowerBound: totalKL / Double(tokenCount),
            caseMeanKL95: .normal95(caseMetrics.map(\.coarsenedKLLowerBound)),
            baselinePerplexity: baselinePerplexity,
            candidatePerplexity: candidatePerplexity,
            perplexityRatio: candidatePerplexity / baselinePerplexity,
            baselineSupportMass: supportMass / Double(tokenCount))
    }
}

/// Per-case samples for exact teacher-forced preservation measurement.
///
/// `tokenKLDivergences` contains one exact full-vocabulary KL value for each
/// token in the reference assistant continuation. Target log probabilities are
/// retained only to report baseline and candidate reference perplexity.
public struct TeacherForcedContinuationCaseSamples: Sendable, Equatable {
    public let name: String
    public let tokenKLDivergences: [Double]
    public let baselineTargetLogProbabilities: [Double]
    public let candidateTargetLogProbabilities: [Double]

    public init(
        name: String,
        tokenKLDivergences: [Double],
        baselineTargetLogProbabilities: [Double],
        candidateTargetLogProbabilities: [Double]
    ) {
        self.name = name
        self.tokenKLDivergences = tokenKLDivergences
        self.baselineTargetLogProbabilities = baselineTargetLogProbabilities
        self.candidateTargetLogProbabilities = candidateTargetLogProbabilities
    }
}

public struct TeacherForcedContinuationCaseMetrics: Codable, Sendable, Equatable {
    public let name: String
    public let tokenCount: Int
    public let exactMeanKL: Double
    public let exactP95KL: Double
    public let exactMaximumKL: Double
    public let baselineReferencePerplexity: Double
    public let candidateReferencePerplexity: Double
}

/// Exact full-vocabulary preservation over benign reference assistant answers.
///
/// The reference continuation is fed token by token through independent
/// baseline and candidate KV caches. This follows the candidate's real cached
/// generation schedule while holding the continuation fixed (teacher forcing).
public struct TeacherForcedContinuationMetricSummary: Codable, Sendable, Equatable {
    public let cases: [TeacherForcedContinuationCaseMetrics]
    public let tokenCount: Int
    public let exactMeanKL: Double
    public let exactP95KL: Double
    public let exactMaximumKL: Double
    public let caseMeanKL95: MeanConfidenceInterval
    public let baselineReferencePerplexity: Double
    public let candidateReferencePerplexity: Double
    public let referencePerplexityRatio: Double

    public var rendered: String {
        [
            String(
                format: "Teacher-forced control continuation exact full-vocabulary KL: mean %.6f, p95 %.6f, max %.6f over %d tokens",
                exactMeanKL, exactP95KL, exactMaximumKL, tokenCount),
            String(
                format: "Case-mean exact KL: %.6f (95%% CI %.6f–%.6f; n=%d)",
                caseMeanKL95.estimate, caseMeanKL95.lowerBound,
                caseMeanKL95.upperBound, caseMeanKL95.sampleCount),
            String(
                format: "Baseline/candidate reference perplexity: %.4f / %.4f (ratio %.4f)",
                baselineReferencePerplexity, candidateReferencePerplexity,
                referencePerplexityRatio),
            "NOTE: this is exact over the full vocabulary at each fixed reference-continuation token; teacher forcing does not estimate free-running response drift.",
        ].joined(separator: "\n")
    }
}

public enum TeacherForcedContinuationMetricEngine {
    /// Exact `KL(P || Q)` for one next-token distribution. Both inputs must be
    /// normalized log probabilities over the same complete vocabulary.
    public static func exactKL(
        baselineLogProbabilities: [Float],
        candidateLogProbabilities: [Float]
    ) throws -> Double {
        guard !baselineLogProbabilities.isEmpty,
              baselineLogProbabilities.count == candidateLogProbabilities.count
        else { throw FingerprintError.incompatible }
        let value = zip(baselineLogProbabilities, candidateLogProbabilities)
            .reduce(0.0) { total, values in
                let logP = Double(values.0)
                return total + exp(logP) * (logP - Double(values.1))
            }
        // Float normalization can leave a tiny negative rounding residue.
        return max(0, value)
    }

    public static func summarize(
        _ samples: [TeacherForcedContinuationCaseSamples]
    ) throws -> TeacherForcedContinuationMetricSummary {
        guard !samples.isEmpty else { throw FingerprintError.incompatible }

        var caseMetrics = [TeacherForcedContinuationCaseMetrics]()
        var allKL = [Double]()
        var baselineNLL = 0.0
        var candidateNLL = 0.0

        for sample in samples {
            let count = sample.tokenKLDivergences.count
            guard !sample.name.isEmpty, count > 0,
                  sample.baselineTargetLogProbabilities.count == count,
                  sample.candidateTargetLogProbabilities.count == count,
                  sample.tokenKLDivergences.allSatisfy({ $0.isFinite && $0 >= 0 }),
                  sample.baselineTargetLogProbabilities.allSatisfy(\.isFinite),
                  sample.candidateTargetLogProbabilities.allSatisfy(\.isFinite)
            else { throw FingerprintError.incompatible }

            let caseBaselineNLL = -sample.baselineTargetLogProbabilities.reduce(0, +)
            let caseCandidateNLL = -sample.candidateTargetLogProbabilities.reduce(0, +)
            let mean = sample.tokenKLDivergences.reduce(0, +) / Double(count)
            caseMetrics.append(TeacherForcedContinuationCaseMetrics(
                name: sample.name,
                tokenCount: count,
                exactMeanKL: mean,
                exactP95KL: nearestRankPercentile(
                    sample.tokenKLDivergences, probability: 0.95),
                exactMaximumKL: sample.tokenKLDivergences.max() ?? 0,
                baselineReferencePerplexity: exp(caseBaselineNLL / Double(count)),
                candidateReferencePerplexity: exp(caseCandidateNLL / Double(count))))
            allKL += sample.tokenKLDivergences
            baselineNLL += caseBaselineNLL
            candidateNLL += caseCandidateNLL
        }

        let count = allKL.count
        guard count > 0 else { throw FingerprintError.incompatible }
        let baselinePerplexity = exp(baselineNLL / Double(count))
        let candidatePerplexity = exp(candidateNLL / Double(count))
        return TeacherForcedContinuationMetricSummary(
            cases: caseMetrics,
            tokenCount: count,
            exactMeanKL: allKL.reduce(0, +) / Double(count),
            exactP95KL: nearestRankPercentile(allKL, probability: 0.95),
            exactMaximumKL: allKL.max() ?? 0,
            caseMeanKL95: .normal95(caseMetrics.map(\.exactMeanKL)),
            baselineReferencePerplexity: baselinePerplexity,
            candidateReferencePerplexity: candidatePerplexity,
            referencePerplexityRatio: candidatePerplexity / baselinePerplexity)
    }

    /// Deterministic nearest-rank percentile. At p95 this selects rank
    /// `ceil(0.95 * n)` in the sorted token-level values.
    static func nearestRankPercentile(
        _ values: [Double], probability: Double
    ) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let rank = max(1, Int(ceil(probability * Double(sorted.count))))
        return sorted[min(sorted.count - 1, rank - 1)]
    }
}

/// Derives reference-answer tokens without assuming that a tokenizer renders
/// a user-only generation prompt as a raw prefix of a completed chat turn.
public enum TeacherForcedContinuationTokenDerivation {
    /// Prefer `promptPlusReferenceTokens`, which is produced by decoding the
    /// exact deployed generation prompt, appending the reference text, and
    /// re-encoding it. A fully templated conversation is accepted only as a
    /// verified prefix-compatible fallback.
    public static func continuation(
        promptTokens: [Int],
        promptPlusReferenceTokens: [Int],
        templatedConversationTokens: [Int]? = nil
    ) throws -> [Int] {
        guard !promptTokens.isEmpty else { throw FingerprintError.incompatible }
        if promptPlusReferenceTokens.starts(with: promptTokens) {
            let tokens = Array(
                promptPlusReferenceTokens.dropFirst(promptTokens.count))
            guard !tokens.isEmpty else { throw FingerprintError.incompatible }
            return tokens
        }
        if let templatedConversationTokens,
           templatedConversationTokens.starts(with: promptTokens)
        {
            let tokens = Array(
                templatedConversationTokens.dropFirst(promptTokens.count))
            guard !tokens.isEmpty else { throw FingerprintError.incompatible }
            return tokens
        }
        throw FingerprintError.incompatible
    }
}
