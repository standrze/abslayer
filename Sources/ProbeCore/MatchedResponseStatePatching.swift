import Foundation

/// One same-prompt response pair for causal response-state localization.
///
/// `donorContinuation` should be an independently verified substantive answer
/// and `recipientContinuation` an independently verified refusal (or the
/// response behavior being diagnosed). The patcher deliberately does not infer
/// either label from strings.
public struct MatchedResponsePatchCase: Codable, Sendable, Equatable {
    public let name: String
    public let prompt: String
    public let donorContinuation: String
    public let recipientContinuation: String
    public let category: String?

    public init(
        name: String, prompt: String, donorContinuation: String,
        recipientContinuation: String, category: String? = nil
    ) {
        self.name = name
        self.prompt = prompt
        self.donorContinuation = donorContinuation
        self.recipientContinuation = recipientContinuation
        self.category = category
    }

    private enum CodingKeys: String, CodingKey {
        case name
        case prompt
        case donorContinuation = "donor_continuation"
        case recipientContinuation = "recipient_continuation"
        case category
    }
}

public struct MatchedResponsePatchProvenance: Codable, Sendable, Equatable {
    public let recipientResponsesPath: String
    public let recipientJudgmentsPath: String
    public let donorResponsesPath: String
    public let donorJudgmentsPath: String
    public let selectionRule: String

    public init(
        recipientResponsesPath: String,
        recipientJudgmentsPath: String,
        donorResponsesPath: String,
        donorJudgmentsPath: String,
        selectionRule: String =
            "same-name exact-prompt contrast; recipient=refusal; donor=compliance"
    ) {
        self.recipientResponsesPath = recipientResponsesPath
        self.recipientJudgmentsPath = recipientJudgmentsPath
        self.donorResponsesPath = donorResponsesPath
        self.donorJudgmentsPath = donorJudgmentsPath
        self.selectionRule = selectionRule
    }

    private enum CodingKeys: String, CodingKey {
        case recipientResponsesPath = "recipient_responses_path"
        case recipientJudgmentsPath = "recipient_judgments_path"
        case donorResponsesPath = "donor_responses_path"
        case donorJudgmentsPath = "donor_judgments_path"
        case selectionRule = "selection_rule"
    }
}

/// Strict dev-only input document. A single prompt owns both continuations, so
/// the API cannot accidentally patch across different prompts in its matched
/// condition.
public struct MatchedResponsePatchDocument: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let split: String
    public let modelCondition: String?
    public let provenance: MatchedResponsePatchProvenance?
    public let cases: [MatchedResponsePatchCase]

    public init(
        schemaVersion: Int = 1, split: String,
        modelCondition: String? = nil,
        provenance: MatchedResponsePatchProvenance? = nil,
        cases: [MatchedResponsePatchCase]
    ) {
        self.schemaVersion = schemaVersion
        self.split = split
        self.modelCondition = modelCondition
        self.provenance = provenance
        self.cases = cases
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case split
        case modelCondition = "model_condition"
        case provenance
        case cases
    }

    public static func read(from path: String) throws -> Self {
        let document = try JSONDecoder().decode(
            Self.self,
            from: Data(contentsOf: URL(fileURLWithPath: path).standardizedFileURL))
        try document.validate()
        return document
    }

    public func validate() throws {
        guard schemaVersion == 1 else {
            throw MatchedResponsePatchError.unsupportedSchema(schemaVersion)
        }
        guard split.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() == "dev"
        else { throw MatchedResponsePatchError.devSplitRequired(split) }
        guard !cases.isEmpty else { throw MatchedResponsePatchError.emptyCases }
        guard cases.count <= MatchedResponsePatchConfiguration.absoluteMaximumCases else {
            throw MatchedResponsePatchError.tooManyCases(
                cases.count,
                maximum: MatchedResponsePatchConfiguration.absoluteMaximumCases)
        }
        var names = Set<String>()
        for item in cases {
            let name = item.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else {
                throw MatchedResponsePatchError.emptyField(
                    caseName: item.name, field: "name")
            }
            guard names.insert(name).inserted else {
                throw MatchedResponsePatchError.duplicateCaseName(name)
            }
            for (field, value) in [
                ("prompt", item.prompt),
                ("donor_continuation", item.donorContinuation),
                ("recipient_continuation", item.recipientContinuation),
            ] where value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                throw MatchedResponsePatchError.emptyField(
                    caseName: name, field: field)
            }
            guard item.donorContinuation != item.recipientContinuation else {
                throw MatchedResponsePatchError.identicalContinuations(name)
            }
        }
    }

    /// Joins two already-generated response artifacts using independent
    /// semantic judgments. Only exact same-name/same-prompt rows with a
    /// refused recipient and compliant donor are retained.
    public static func build(
        recipientResponses: [PromptResult],
        recipientJudgments: [RecordedOutcomeJudgment],
        donorResponses: [PromptResult],
        donorJudgments: [RecordedOutcomeJudgment],
        modelCondition: String? = nil,
        provenance: MatchedResponsePatchProvenance? = nil
    ) throws -> Self {
        let recipientByName = try uniqueResponses(
            recipientResponses, label: "recipient")
        let donorByName = try uniqueResponses(
            donorResponses, label: "donor")
        let recipientOutcomes = try uniqueContrastOutcomes(
            recipientJudgments, label: "recipient")
        let donorOutcomes = try uniqueContrastOutcomes(
            donorJudgments, label: "donor")

        var result = [MatchedResponsePatchCase]()
        for recipient in recipientResponses {
            guard recipientOutcomes[recipient.name] == .refusal,
                  donorOutcomes[recipient.name] == .compliance,
                  let canonicalRecipient = recipientByName[recipient.name],
                  let donor = donorByName[recipient.name]
            else { continue }
            guard let recipientPrompt = canonicalRecipient.contrastPrompt,
                  !recipientPrompt.trimmingCharacters(
                    in: .whitespacesAndNewlines).isEmpty,
                  let donorPrompt = donor.contrastPrompt,
                  !donorPrompt.trimmingCharacters(
                    in: .whitespacesAndNewlines).isEmpty
            else {
                throw MatchedResponsePatchError.missingRecordedPrompt(
                    recipient.name)
            }
            guard recipientPrompt == donorPrompt else {
                throw MatchedResponsePatchError.recordedPromptMismatch(
                    recipient.name)
            }
            result.append(MatchedResponsePatchCase(
                name: recipient.name,
                prompt: recipientPrompt,
                donorContinuation: donor.contrastResponse,
                recipientContinuation:
                    canonicalRecipient.contrastResponse,
                category: canonicalRecipient.category ?? donor.category))
        }
        guard !result.isEmpty else {
            throw MatchedResponsePatchError.noEligibleJudgedCases
        }
        let document = Self(
            split: "dev", modelCondition: modelCondition,
            provenance: provenance, cases: result)
        try document.validate()
        return document
    }

    private static func uniqueResponses(
        _ values: [PromptResult], label: String
    ) throws -> [String: PromptResult] {
        var result = [String: PromptResult]()
        for value in values {
            guard result.updateValue(value, forKey: value.name) == nil else {
                throw MatchedResponsePatchError.duplicateRecordedResponse(
                    label: label, name: value.name)
            }
        }
        return result
    }

    private static func uniqueContrastOutcomes(
        _ values: [RecordedOutcomeJudgment], label: String
    ) throws -> [String: BehaviorOutcome] {
        var result = [String: BehaviorOutcome]()
        for value in values where value.channel == .contrast {
            guard result.updateValue(
                value.classification.outcome, forKey: value.name) == nil
            else {
                throw MatchedResponsePatchError.duplicateRecordedJudgment(
                    label: label, name: value.name)
            }
        }
        return result
    }
}

/// Bounded search configuration. Layer and response-token indices are
/// canonical zero-based values.
public struct MatchedResponsePatchConfiguration: Codable, Sendable, Equatable {
    public static let absoluteMaximumCases = 16
    public static let absoluteMaximumLayers = 12
    public static let absoluteMaximumPositions = 8
    public static let absoluteMaximumSiteEvaluations = 128
    public static let absoluteMaximumSuffixTokens = 32
    public static let defaultRandomSeed: UInt64 = 0xA8_51_A7_E2

    public let layersZeroBased: [Int]
    public let responseTokenPositionsZeroBased: [Int]
    public let maximumCases: Int
    public let maximumSuffixTokens: Int
    public let randomControlSeed: UInt64

    public init(
        layersZeroBased: [Int],
        responseTokenPositionsZeroBased: [Int],
        maximumCases: Int = 4,
        maximumSuffixTokens: Int = 8,
        randomControlSeed: UInt64 = defaultRandomSeed
    ) throws {
        guard !layersZeroBased.isEmpty,
              layersZeroBased.count <= Self.absoluteMaximumLayers,
              layersZeroBased.allSatisfy({ $0 >= 0 }),
              Set(layersZeroBased).count == layersZeroBased.count
        else { throw MatchedResponsePatchError.invalidLayers(layersZeroBased) }
        guard !responseTokenPositionsZeroBased.isEmpty,
              responseTokenPositionsZeroBased.count
                <= Self.absoluteMaximumPositions,
              responseTokenPositionsZeroBased.allSatisfy({ $0 >= 0 }),
              Set(responseTokenPositionsZeroBased).count
                == responseTokenPositionsZeroBased.count
        else {
            throw MatchedResponsePatchError.invalidResponsePositions(
                responseTokenPositionsZeroBased)
        }
        guard (1 ... Self.absoluteMaximumCases).contains(maximumCases) else {
            throw MatchedResponsePatchError.invalidMaximumCases(maximumCases)
        }
        guard (1 ... Self.absoluteMaximumSuffixTokens).contains(
            maximumSuffixTokens)
        else {
            throw MatchedResponsePatchError.invalidMaximumSuffixTokens(
                maximumSuffixTokens)
        }
        self.layersZeroBased = layersZeroBased.sorted()
        self.responseTokenPositionsZeroBased =
            responseTokenPositionsZeroBased.sorted()
        self.maximumCases = maximumCases
        self.maximumSuffixTokens = maximumSuffixTokens
        self.randomControlSeed = randomControlSeed
    }

    public func validate(caseCount: Int, decoderLayerCount: Int) throws {
        guard decoderLayerCount > 0,
              layersZeroBased.allSatisfy({ $0 < decoderLayerCount })
        else {
            throw MatchedResponsePatchError.layerOutsideDecoder(
                layers: layersZeroBased, decoderLayerCount: decoderLayerCount)
        }
        let selectedCases = min(caseCount, maximumCases)
        let count = selectedCases * layersZeroBased.count
            * responseTokenPositionsZeroBased.count
        guard count <= Self.absoluteMaximumSiteEvaluations else {
            throw MatchedResponsePatchError.tooManySiteEvaluations(
                count, maximum: Self.absoluteMaximumSiteEvaluations)
        }
    }
}

public struct MatchedResponsePatchSite: Codable, Sendable, Equatable,
    Hashable
{
    public let layerZeroBased: Int
    public let responseTokenPositionZeroBased: Int

    public init(layerZeroBased: Int, responseTokenPositionZeroBased: Int) {
        self.layerZeroBased = layerZeroBased
        self.responseTokenPositionZeroBased = responseTokenPositionZeroBased
    }
}

/// Exact next-token and bounded teacher-forced suffix effects for one patch.
/// A positive `sourceDistributionClosenessGain` means the patched distribution
/// moved toward the matched source trajectory. It is a localization result,
/// not a behavioral-success certificate.
public struct MatchedResponsePatchEffect: Codable, Sendable, Equatable {
    public let exactKLTargetBaselineToPatched: Double
    public let exactKLSourceBaselineToTargetBaseline: Double
    public let exactKLSourceBaselineToPatched: Double
    public let sourceDistributionClosenessGain: Double
    /// Fraction of the original source→target KL gap closed by the patch.
    /// Nil when the two baseline distributions are already identical.
    public let sourceDistributionClosenessFraction: Double?
    public let nextTokenLogOddsShiftTowardSource: Double
    public let sourceAndTargetNextTokensDiffer: Bool
    public let targetContinuationMeanLogProbabilityDelta: Double
    public let targetContinuationTokenCount: Int

    public init(
        exactKLTargetBaselineToPatched: Double,
        exactKLSourceBaselineToTargetBaseline: Double,
        exactKLSourceBaselineToPatched: Double,
        sourceDistributionClosenessGain: Double,
        sourceDistributionClosenessFraction: Double?,
        nextTokenLogOddsShiftTowardSource: Double,
        sourceAndTargetNextTokensDiffer: Bool,
        targetContinuationMeanLogProbabilityDelta: Double,
        targetContinuationTokenCount: Int
    ) {
        self.exactKLTargetBaselineToPatched =
            exactKLTargetBaselineToPatched
        self.exactKLSourceBaselineToTargetBaseline =
            exactKLSourceBaselineToTargetBaseline
        self.exactKLSourceBaselineToPatched =
            exactKLSourceBaselineToPatched
        self.sourceDistributionClosenessGain =
            sourceDistributionClosenessGain
        self.sourceDistributionClosenessFraction =
            sourceDistributionClosenessFraction
        self.nextTokenLogOddsShiftTowardSource =
            nextTokenLogOddsShiftTowardSource
        self.sourceAndTargetNextTokensDiffer =
            sourceAndTargetNextTokensDiffer
        self.targetContinuationMeanLogProbabilityDelta =
            targetContinuationMeanLogProbabilityDelta
        self.targetContinuationTokenCount = targetContinuationTokenCount
    }
}

public struct MatchedResponsePatchResult: Codable, Sendable, Equatable {
    public let caseName: String
    public let category: String?
    public let site: MatchedResponsePatchSite
    public let donorNextTokenID: Int
    public let recipientNextTokenID: Int
    public let donorPatchedTokenID: Int
    public let recipientPatchedTokenID: Int
    /// False means the patch also swaps token-identity information; those rows
    /// cannot by themselves isolate a refusal mechanism from lexical content.
    public let responsePrefixesMatchThroughPatchedToken: Bool
    public let donorRecipientStateL2Distance: Double
    public let donorRecipientStateCosineSimilarity: Double
    public let donorIntoRecipient: MatchedResponsePatchEffect
    public let recipientIntoDonor: MatchedResponsePatchEffect
    /// A different case's donor direction, rescaled to the exact norm of the
    /// matched donor-minus-recipient state change before insertion.
    public let normMatchedRandomControlIntoRecipient:
        MatchedResponsePatchEffect?
    public let randomControlCaseName: String?

    public init(
        caseName: String, category: String?, site: MatchedResponsePatchSite,
        donorNextTokenID: Int, recipientNextTokenID: Int,
        donorPatchedTokenID: Int, recipientPatchedTokenID: Int,
        responsePrefixesMatchThroughPatchedToken: Bool,
        donorRecipientStateL2Distance: Double,
        donorRecipientStateCosineSimilarity: Double,
        donorIntoRecipient: MatchedResponsePatchEffect,
        recipientIntoDonor: MatchedResponsePatchEffect,
        normMatchedRandomControlIntoRecipient: MatchedResponsePatchEffect?,
        randomControlCaseName: String?
    ) {
        self.caseName = caseName
        self.category = category
        self.site = site
        self.donorNextTokenID = donorNextTokenID
        self.recipientNextTokenID = recipientNextTokenID
        self.donorPatchedTokenID = donorPatchedTokenID
        self.recipientPatchedTokenID = recipientPatchedTokenID
        self.responsePrefixesMatchThroughPatchedToken =
            responsePrefixesMatchThroughPatchedToken
        self.donorRecipientStateL2Distance =
            donorRecipientStateL2Distance
        self.donorRecipientStateCosineSimilarity =
            donorRecipientStateCosineSimilarity
        self.donorIntoRecipient = donorIntoRecipient
        self.recipientIntoDonor = recipientIntoDonor
        self.normMatchedRandomControlIntoRecipient =
            normMatchedRandomControlIntoRecipient
        self.randomControlCaseName = randomControlCaseName
    }
}

public struct MatchedResponsePatchSiteSummary: Codable, Sendable, Equatable {
    public let site: MatchedResponsePatchSite
    public let caseCount: Int
    public let matchedMeanClosenessGain: Double
    public let matchedMeanClosenessFraction: Double?
    public let reverseMeanClosenessGain: Double
    public let randomControlMeanClosenessGain: Double?
    public let randomControlMeanClosenessFraction: Double?
    public let matchedMinusRandomMeanClosenessGain: Double?
    public let matchedPositiveClosenessRate: Double
    public let reversePositiveClosenessRate: Double
    public let randomControlPositiveClosenessRate: Double?
    public let matchedMeanNextTokenLogOddsShift: Double
    public let matchedMeanTargetBaselineToPatchedKL: Double
    public let randomControlMeanTargetBaselineToPatchedKL: Double?
    public let matchedMeanTargetContinuationLogProbabilityDelta: Double
    public let lexicallyMatchedCaseCount: Int
}

public struct MatchedResponsePatchSummary: Codable, Sendable, Equatable {
    public let resultCount: Int
    public let lexicallyMatchedResultCount: Int
    public let matchedMeanClosenessGain: Double
    public let matchedMeanClosenessFraction: Double?
    public let reverseMeanClosenessGain: Double
    public let randomControlMeanClosenessGain: Double?
    public let randomControlMeanClosenessFraction: Double?
    public let matchedMinusRandomMeanClosenessGain: Double?
    public let matchedPositiveClosenessRate: Double
    public let reversePositiveClosenessRate: Double
    public let randomControlPositiveClosenessRate: Double?
    public let siteSummaries: [MatchedResponsePatchSiteSummary]
}

public struct MatchedResponsePatchStudy: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let status: String
    public let warning: String
    public let modelPath: String
    public let inputSplit: String
    public let inputModelCondition: String?
    public let inputProvenance: MatchedResponsePatchProvenance?
    public let configuration: MatchedResponsePatchConfiguration
    public let cases: [MatchedResponsePatchCase]
    public let results: [MatchedResponsePatchResult]
    public let summary: MatchedResponsePatchSummary

    public init(
        modelPath: String, inputSplit: String,
        inputModelCondition: String?,
        inputProvenance: MatchedResponsePatchProvenance? = nil,
        configuration: MatchedResponsePatchConfiguration,
        cases: [MatchedResponsePatchCase],
        results: [MatchedResponsePatchResult]
    ) {
        schemaVersion = 1
        status = "diagnostic_only"
        warning = "Activation patching localizes causal response-state sites; it does not establish behavioral abliteration, capability preservation, or deployment safety."
        self.modelPath = modelPath
        self.inputSplit = inputSplit
        self.inputModelCondition = inputModelCondition
        self.inputProvenance = inputProvenance
        self.configuration = configuration
        self.cases = cases
        self.results = results
        summary = MatchedResponsePatchMath.summarize(results)
    }

    public func write(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(
            to: URL(fileURLWithPath: path).standardizedFileURL,
            options: .atomic)
    }
}

public enum MatchedResponsePatchMath {
    public static func effect(
        sourceBaselineLogProbabilities: [Float],
        targetBaselineLogProbabilities: [Float],
        patchedLogProbabilities: [Float],
        sourceNextTokenID: Int,
        targetNextTokenID: Int,
        targetBaselineContinuationLogProbabilities: [Double],
        patchedTargetContinuationLogProbabilities: [Double]
    ) throws -> MatchedResponsePatchEffect {
        let vocabulary = sourceBaselineLogProbabilities.count
        guard vocabulary > 0,
              targetBaselineLogProbabilities.count == vocabulary,
              patchedLogProbabilities.count == vocabulary,
              (0 ..< vocabulary).contains(sourceNextTokenID),
              (0 ..< vocabulary).contains(targetNextTokenID),
              !targetBaselineContinuationLogProbabilities.isEmpty,
              targetBaselineContinuationLogProbabilities.count
                == patchedTargetContinuationLogProbabilities.count,
              sourceBaselineLogProbabilities.allSatisfy(\.isFinite),
              targetBaselineLogProbabilities.allSatisfy(\.isFinite),
              patchedLogProbabilities.allSatisfy(\.isFinite),
              targetBaselineContinuationLogProbabilities.allSatisfy(\.isFinite),
              patchedTargetContinuationLogProbabilities.allSatisfy(\.isFinite)
        else { throw MatchedResponsePatchError.incompatibleMetrics }

        let targetToPatch = try TeacherForcedContinuationMetricEngine.exactKL(
            baselineLogProbabilities: targetBaselineLogProbabilities,
            candidateLogProbabilities: patchedLogProbabilities)
        let sourceToTarget = try TeacherForcedContinuationMetricEngine.exactKL(
            baselineLogProbabilities: sourceBaselineLogProbabilities,
            candidateLogProbabilities: targetBaselineLogProbabilities)
        let sourceToPatch = try TeacherForcedContinuationMetricEngine.exactKL(
            baselineLogProbabilities: sourceBaselineLogProbabilities,
            candidateLogProbabilities: patchedLogProbabilities)
        let baselineOdds = Double(
            targetBaselineLogProbabilities[sourceNextTokenID]
                - targetBaselineLogProbabilities[targetNextTokenID])
        let patchedOdds = Double(
            patchedLogProbabilities[sourceNextTokenID]
                - patchedLogProbabilities[targetNextTokenID])
        let continuationDelta = zip(
            targetBaselineContinuationLogProbabilities,
            patchedTargetContinuationLogProbabilities
        ).reduce(0.0) { total, values in
            total + values.1 - values.0
        } / Double(targetBaselineContinuationLogProbabilities.count)

        return MatchedResponsePatchEffect(
            exactKLTargetBaselineToPatched: targetToPatch,
            exactKLSourceBaselineToTargetBaseline: sourceToTarget,
            exactKLSourceBaselineToPatched: sourceToPatch,
            sourceDistributionClosenessGain: sourceToTarget - sourceToPatch,
            sourceDistributionClosenessFraction: sourceToTarget > 0
                ? (sourceToTarget - sourceToPatch) / sourceToTarget
                : nil,
            nextTokenLogOddsShiftTowardSource: patchedOdds - baselineOdds,
            sourceAndTargetNextTokensDiffer:
                sourceNextTokenID != targetNextTokenID,
            targetContinuationMeanLogProbabilityDelta: continuationDelta,
            targetContinuationTokenCount:
                targetBaselineContinuationLogProbabilities.count)
    }

    /// Builds an unrelated-direction corruption whose injected L2 norm equals
    /// the matched donor-minus-recipient patch norm.
    public static func normMatchedControlReplacement(
        recipient: [Float], unrelatedDonor: [Float],
        matchedDonor: [Float]
    ) throws -> [Float] {
        guard !recipient.isEmpty,
              unrelatedDonor.count == recipient.count,
              matchedDonor.count == recipient.count
        else { throw MatchedResponsePatchError.incompatibleStates }
        let matchedNorm = l2Distance(recipient, matchedDonor)
        let unrelatedNorm = l2Distance(recipient, unrelatedDonor)
        guard matchedNorm.isFinite, unrelatedNorm.isFinite,
              unrelatedNorm > 1e-12
        else { throw MatchedResponsePatchError.degenerateRandomControl }
        let scale = Float(matchedNorm / unrelatedNorm)
        return zip(recipient, unrelatedDonor).map { base, other in
            base + scale * (other - base)
        }
    }

    /// Seeded SplitMix64 selection over all cases except the matched case.
    public static func randomControlIndex(
        caseIndex: Int, caseCount: Int, seed: UInt64
    ) -> Int? {
        guard caseCount > 1, (0 ..< caseCount).contains(caseIndex) else {
            return nil
        }
        var state = seed &+ UInt64(caseIndex) &* 0x9E3779B97F4A7C15
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        value ^= value >> 31
        let offset = 1 + Int(value % UInt64(caseCount - 1))
        return (caseIndex + offset) % caseCount
    }

    public static func l2Distance(_ lhs: [Float], _ rhs: [Float]) -> Double {
        guard lhs.count == rhs.count else { return .nan }
        return sqrt(zip(lhs, rhs).reduce(0.0) { total, values in
            let delta = Double(values.0 - values.1)
            return total + delta * delta
        })
    }

    public static func summarize(
        _ results: [MatchedResponsePatchResult]
    ) -> MatchedResponsePatchSummary {
        let groups = Dictionary(grouping: results, by: \.site)
        let siteSummaries = groups.keys.sorted {
            if $0.layerZeroBased != $1.layerZeroBased {
                return $0.layerZeroBased < $1.layerZeroBased
            }
            return $0.responseTokenPositionZeroBased
                < $1.responseTokenPositionZeroBased
        }.map { site in
            siteSummary(site: site, results: groups[site] ?? [])
        }
        let matched = results.map {
            $0.donorIntoRecipient.sourceDistributionClosenessGain
        }
        let reverse = results.map {
            $0.recipientIntoDonor.sourceDistributionClosenessGain
        }
        let random = results.compactMap {
            $0.normMatchedRandomControlIntoRecipient?
                .sourceDistributionClosenessGain
        }
        let matchedFractions = results.compactMap {
            $0.donorIntoRecipient.sourceDistributionClosenessFraction
        }
        let randomFractions = results.compactMap {
            $0.normMatchedRandomControlIntoRecipient?
                .sourceDistributionClosenessFraction
        }
        let matchedMean = mean(matched)
        let randomMean = random.isEmpty ? nil : mean(random)
        return MatchedResponsePatchSummary(
            resultCount: results.count,
            lexicallyMatchedResultCount: results.count(where:
                \.responsePrefixesMatchThroughPatchedToken),
            matchedMeanClosenessGain: matchedMean,
            matchedMeanClosenessFraction:
                matchedFractions.isEmpty ? nil : mean(matchedFractions),
            reverseMeanClosenessGain: mean(reverse),
            randomControlMeanClosenessGain: randomMean,
            randomControlMeanClosenessFraction:
                randomFractions.isEmpty ? nil : mean(randomFractions),
            matchedMinusRandomMeanClosenessGain:
                randomMean.map { matchedMean - $0 },
            matchedPositiveClosenessRate: positiveRate(matched),
            reversePositiveClosenessRate: positiveRate(reverse),
            randomControlPositiveClosenessRate:
                random.isEmpty ? nil : positiveRate(random),
            siteSummaries: siteSummaries)
    }

    private static func siteSummary(
        site: MatchedResponsePatchSite,
        results: [MatchedResponsePatchResult]
    ) -> MatchedResponsePatchSiteSummary {
        let matched = results.map {
            $0.donorIntoRecipient.sourceDistributionClosenessGain
        }
        let reverse = results.map {
            $0.recipientIntoDonor.sourceDistributionClosenessGain
        }
        let random = results.compactMap {
            $0.normMatchedRandomControlIntoRecipient?
                .sourceDistributionClosenessGain
        }
        let matchedFractions = results.compactMap {
            $0.donorIntoRecipient.sourceDistributionClosenessFraction
        }
        let randomFractions = results.compactMap {
            $0.normMatchedRandomControlIntoRecipient?
                .sourceDistributionClosenessFraction
        }
        let randomTargetKL = results.compactMap {
            $0.normMatchedRandomControlIntoRecipient?
                .exactKLTargetBaselineToPatched
        }
        let randomMean = random.isEmpty ? nil : mean(random)
        let matchedMean = mean(matched)
        return MatchedResponsePatchSiteSummary(
            site: site,
            caseCount: results.count,
            matchedMeanClosenessGain: matchedMean,
            matchedMeanClosenessFraction:
                matchedFractions.isEmpty ? nil : mean(matchedFractions),
            reverseMeanClosenessGain: mean(reverse),
            randomControlMeanClosenessGain: randomMean,
            randomControlMeanClosenessFraction:
                randomFractions.isEmpty ? nil : mean(randomFractions),
            matchedMinusRandomMeanClosenessGain:
                randomMean.map { matchedMean - $0 },
            matchedPositiveClosenessRate: positiveRate(matched),
            reversePositiveClosenessRate: positiveRate(reverse),
            randomControlPositiveClosenessRate:
                random.isEmpty ? nil : positiveRate(random),
            matchedMeanNextTokenLogOddsShift: mean(results.map {
                $0.donorIntoRecipient.nextTokenLogOddsShiftTowardSource
            }),
            matchedMeanTargetBaselineToPatchedKL: mean(results.map {
                $0.donorIntoRecipient.exactKLTargetBaselineToPatched
            }),
            randomControlMeanTargetBaselineToPatchedKL:
                randomTargetKL.isEmpty ? nil : mean(randomTargetKL),
            matchedMeanTargetContinuationLogProbabilityDelta: mean(
                results.map {
                    $0.donorIntoRecipient
                        .targetContinuationMeanLogProbabilityDelta
                }),
            lexicallyMatchedCaseCount: results.count(where:
                \.responsePrefixesMatchThroughPatchedToken))
    }

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func positiveRate(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return Double(values.count(where: { $0 > 0 })) / Double(values.count)
    }
}

public enum MatchedResponsePatchError: LocalizedError, Equatable, Sendable {
    case unsupportedSchema(Int)
    case devSplitRequired(String)
    case emptyCases
    case tooManyCases(Int, maximum: Int)
    case emptyField(caseName: String, field: String)
    case duplicateCaseName(String)
    case identicalContinuations(String)
    case duplicateRecordedResponse(label: String, name: String)
    case duplicateRecordedJudgment(label: String, name: String)
    case missingRecordedPrompt(String)
    case recordedPromptMismatch(String)
    case noEligibleJudgedCases
    case invalidLayers([Int])
    case invalidResponsePositions([Int])
    case invalidMaximumCases(Int)
    case invalidMaximumSuffixTokens(Int)
    case layerOutsideDecoder(layers: [Int], decoderLayerCount: Int)
    case tooManySiteEvaluations(Int, maximum: Int)
    case responsePositionUnavailable(
        caseName: String, position: Int, donorTokens: Int,
        recipientTokens: Int)
    case promptTooLong(caseName: String, tokenCount: Int, maximum: Int)
    case promptTokenizationMismatch(String)
    case unsupportedModel
    case incompatibleStates
    case degenerateRandomControl
    case incompatibleMetrics

    public var errorDescription: String? {
        switch self {
        case .unsupportedSchema(let version):
            "Matched patch input schema must be version 1, not \(version)."
        case .devSplitRequired(let split):
            "Matched patching is dev-only; document split must be 'dev', not '\(split)'."
        case .emptyCases:
            "Matched patch input contains no cases."
        case .tooManyCases(let count, let maximum):
            "Matched patch input has \(count) cases; the bounded maximum is \(maximum)."
        case .emptyField(let name, let field):
            "Matched patch case '\(name)' has an empty \(field)."
        case .duplicateCaseName(let name):
            "Matched patch input repeats case name '\(name)'."
        case .identicalContinuations(let name):
            "Matched patch case '\(name)' has identical donor and recipient continuations."
        case .duplicateRecordedResponse(let label, let name):
            "The \(label) response artifact repeats case '\(name)'."
        case .duplicateRecordedJudgment(let label, let name):
            "The \(label) judgment artifact repeats a contrast judgment for '\(name)'."
        case .missingRecordedPrompt(let name):
            "Matched response artifacts do not record a contrast prompt for '\(name)'."
        case .recordedPromptMismatch(let name):
            "Donor and recipient artifacts record different prompts for '\(name)'."
        case .noEligibleJudgedCases:
            "No exact-prompt cases satisfy recipient=refusal and donor=compliance."
        case .invalidLayers(let layers):
            "Matched patch layers must be 1-\(MatchedResponsePatchConfiguration.absoluteMaximumLayers) unique zero-based integers, not \(layers)."
        case .invalidResponsePositions(let positions):
            "Matched patch response positions must be 1-\(MatchedResponsePatchConfiguration.absoluteMaximumPositions) unique zero-based integers, not \(positions)."
        case .invalidMaximumCases(let count):
            "Matched patch maximum cases must be 1-\(MatchedResponsePatchConfiguration.absoluteMaximumCases), not \(count)."
        case .invalidMaximumSuffixTokens(let count):
            "Matched patch suffix tokens must be 1-\(MatchedResponsePatchConfiguration.absoluteMaximumSuffixTokens), not \(count)."
        case .layerOutsideDecoder(let layers, let count):
            "Matched patch layers \(layers) are outside the \(count)-layer decoder."
        case .tooManySiteEvaluations(let count, let maximum):
            "Matched patch request contains \(count) case/layer/position sites; the bounded maximum is \(maximum)."
        case .responsePositionUnavailable(
            let name, let position, let donor, let recipient):
            "Matched patch case '\(name)' cannot test response position \(position): donor/recipient have \(donor)/\(recipient) continuation tokens and each needs a following token."
        case .promptTooLong(let name, let count, let maximum):
            "Matched patch case '\(name)' has a \(count)-token teacher-forced prefix; maximum is \(maximum)."
        case .promptTokenizationMismatch(let name):
            "Matched patch case '\(name)' could not preserve the exact rendered prompt boundary while tokenizing a continuation."
        case .unsupportedModel:
            "Matched response-state patching currently requires a freshly loaded Gemma 4 BF16 text backbone."
        case .incompatibleStates:
            "Matched patch residual states have incompatible dimensions."
        case .degenerateRandomControl:
            "The mismatched random-control state has zero displacement and cannot be norm matched."
        case .incompatibleMetrics:
            "Matched patch distributions or continuation metrics are incompatible."
        }
    }
}
