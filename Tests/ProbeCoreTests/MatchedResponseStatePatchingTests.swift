import Foundation
import Testing
@testable import ProbeCore

@Suite("Matched response-state activation patching")
struct MatchedResponseStatePatchingTests {
    @Test("input is same-prompt by construction and strictly dev-only")
    func devOnlyDocument() throws {
        let item = MatchedResponsePatchCase(
            name: "case-1",
            prompt: "same prompt",
            donorContinuation: "direct answer",
            recipientContinuation: "I cannot help",
            category: "cyber")
        let document = MatchedResponsePatchDocument(
            split: " DEV ", modelCondition: "untouched-gemma4",
            cases: [item])
        try document.validate()

        let roundTrip = try JSONDecoder().decode(
            MatchedResponsePatchDocument.self,
            from: JSONEncoder().encode(document))
        #expect(roundTrip == document)
        #expect(roundTrip.cases[0].prompt == "same prompt")

        #expect(throws:
            MatchedResponsePatchError.devSplitRequired("audit"))
        {
            try MatchedResponsePatchDocument(
                split: "audit", cases: [item]).validate()
        }
        #expect(throws:
            MatchedResponsePatchError.identicalContinuations("case-1"))
        {
            try MatchedResponsePatchDocument(
                split: "dev",
                cases: [MatchedResponsePatchCase(
                    name: "case-1", prompt: "p",
                    donorContinuation: "same",
                    recipientContinuation: "same")]).validate()
        }
    }

    @Test("configuration is bounded and uses zero-based sites")
    func boundedConfiguration() throws {
        let configuration = try MatchedResponsePatchConfiguration(
            layersZeroBased: [23, 16, 19],
            responseTokenPositionsZeroBased: [3, 0, 1],
            maximumCases: 4, maximumSuffixTokens: 8,
            randomControlSeed: 7)
        #expect(configuration.layersZeroBased == [16, 19, 23])
        #expect(configuration.responseTokenPositionsZeroBased == [0, 1, 3])
        try configuration.validate(caseCount: 4, decoderLayerCount: 35)

        #expect(throws: MatchedResponsePatchError.layerOutsideDecoder(
            layers: [16, 19, 23], decoderLayerCount: 20))
        {
            try configuration.validate(caseCount: 4, decoderLayerCount: 20)
        }
        #expect(throws: MatchedResponsePatchError.invalidLayers([0, 0])) {
            _ = try MatchedResponsePatchConfiguration(
                layersZeroBased: [0, 0],
                responseTokenPositionsZeroBased: [0])
        }
    }

    @Test("builder joins exact prompts using semantic outcomes")
    func judgedArtifactBuilder() throws {
        let recipient = [
            PromptResult(
                name: "keep", contrastResponse: "refusal",
                controlResponse: "", category: "cyber",
                contrastPrompt: "identical prompt"),
            PromptResult(
                name: "drop", contrastResponse: "refusal",
                controlResponse: "", contrastPrompt: "other prompt"),
        ]
        let donor = [
            PromptResult(
                name: "keep", contrastResponse: "direct answer",
                controlResponse: "", contrastPrompt: "identical prompt"),
            PromptResult(
                name: "drop", contrastResponse: "partial answer",
                controlResponse: "", contrastPrompt: "other prompt"),
        ]
        let recipientJudgments = [
            judgment("keep", .refusal), judgment("drop", .refusal),
        ]
        let donorJudgments = [
            judgment("keep", .compliance),
            judgment("drop", .partialCompliance),
        ]
        let provenance = MatchedResponsePatchProvenance(
            recipientResponsesPath: "/recipient.json",
            recipientJudgmentsPath: "/recipient-judgments.json",
            donorResponsesPath: "/donor.json",
            donorJudgmentsPath: "/donor-judgments.json")
        let document = try MatchedResponsePatchDocument.build(
            recipientResponses: recipient,
            recipientJudgments: recipientJudgments,
            donorResponses: donor,
            donorJudgments: donorJudgments,
            modelCondition: "base-vs-donor",
            provenance: provenance)

        #expect(document.split == "dev")
        #expect(document.cases.map(\.name) == ["keep"])
        #expect(document.cases[0].prompt == "identical prompt")
        #expect(document.cases[0].donorContinuation == "direct answer")
        #expect(document.cases[0].recipientContinuation == "refusal")
        #expect(document.provenance == provenance)

        var mismatchedDonor = donor
        mismatchedDonor[0] = PromptResult(
            name: "keep", contrastResponse: "direct answer",
            controlResponse: "", contrastPrompt: "changed prompt")
        #expect(throws:
            MatchedResponsePatchError.recordedPromptMismatch("keep"))
        {
            _ = try MatchedResponsePatchDocument.build(
                recipientResponses: recipient,
                recipientJudgments: recipientJudgments,
                donorResponses: mismatchedDonor,
                donorJudgments: donorJudgments)
        }
    }

    @Test("random corruption is cross-case deterministic and norm matched")
    func normMatchedRandomControl() throws {
        for caseIndex in 0 ..< 5 {
            let first = try #require(
                MatchedResponsePatchMath.randomControlIndex(
                    caseIndex: caseIndex, caseCount: 5, seed: 99))
            let second = try #require(
                MatchedResponsePatchMath.randomControlIndex(
                    caseIndex: caseIndex, caseCount: 5, seed: 99))
            #expect(first == second)
            #expect(first != caseIndex)
        }
        #expect(MatchedResponsePatchMath.randomControlIndex(
            caseIndex: 0, caseCount: 1, seed: 99) == nil)

        let recipient: [Float] = [0, 0]
        let donor: [Float] = [3, 4]
        let unrelated: [Float] = [0, 2]
        let replacement = try MatchedResponsePatchMath
            .normMatchedControlReplacement(
                recipient: recipient,
                unrelatedDonor: unrelated,
                matchedDonor: donor)
        #expect(abs(MatchedResponsePatchMath.l2Distance(
            recipient, replacement) - 5) < 0.000_001)
        #expect(replacement == [0, 5])
    }

    @Test("matched patch metrics report movement toward the source")
    func causalEffectMetrics() throws {
        let source = [Float(log(0.8)), Float(log(0.2))]
        let target = [Float(log(0.2)), Float(log(0.8))]
        let patched = [Float(log(0.6)), Float(log(0.4))]
        let effect = try MatchedResponsePatchMath.effect(
            sourceBaselineLogProbabilities: source,
            targetBaselineLogProbabilities: target,
            patchedLogProbabilities: patched,
            sourceNextTokenID: 0,
            targetNextTokenID: 1,
            targetBaselineContinuationLogProbabilities: [-1, -2],
            patchedTargetContinuationLogProbabilities: [-0.5, -2.5])

        #expect(effect.sourceDistributionClosenessGain > 0)
        #expect(effect.nextTokenLogOddsShiftTowardSource > 0)
        #expect(effect.sourceAndTargetNextTokensDiffer)
        #expect(effect.targetContinuationMeanLogProbabilityDelta == 0)
        #expect(effect.targetContinuationTokenCount == 2)
        #expect(effect.exactKLTargetBaselineToPatched > 0)
    }

    @Test("study output stays explicitly diagnostic and aggregates controls")
    func diagnosticStudySummary() throws {
        let site = MatchedResponsePatchSite(
            layerZeroBased: 19,
            responseTokenPositionZeroBased: 1)
        let result = MatchedResponsePatchResult(
            caseName: "case-1", category: "cyber", site: site,
            donorNextTokenID: 1, recipientNextTokenID: 2,
            donorPatchedTokenID: 3, recipientPatchedTokenID: 4,
            responsePrefixesMatchThroughPatchedToken: false,
            donorRecipientStateL2Distance: 2,
            donorRecipientStateCosineSimilarity: 0.5,
            donorIntoRecipient: effect(gain: 0.6, odds: 0.4),
            recipientIntoDonor: effect(gain: 0.3, odds: 0.2),
            normMatchedRandomControlIntoRecipient:
                effect(gain: 0.1, odds: 0.05),
            randomControlCaseName: "case-2")
        let configuration = try MatchedResponsePatchConfiguration(
            layersZeroBased: [19],
            responseTokenPositionsZeroBased: [1])
        let input = MatchedResponsePatchCase(
            name: "case-1", prompt: "p",
            donorContinuation: "answer",
            recipientContinuation: "refusal")
        let study = MatchedResponsePatchStudy(
            modelPath: "/model", inputSplit: "dev",
            inputModelCondition: "base",
            configuration: configuration,
            cases: [input], results: [result])

        #expect(study.status == "diagnostic_only")
        #expect(study.warning.contains("does not establish"))
        #expect(study.summary.matchedMeanClosenessGain == 0.6)
        #expect(study.summary.randomControlMeanClosenessGain == 0.1)
        #expect(abs((study.summary.matchedMinusRandomMeanClosenessGain ?? 0)
            - 0.5) < 0.000_001)
        #expect(study.summary.siteSummaries.count == 1)
        #expect(study.summary.lexicallyMatchedResultCount == 0)
        #expect(study.summary.matchedMeanClosenessFraction == 0.6)
    }

    private func effect(
        gain: Double, odds: Double
    ) -> MatchedResponsePatchEffect {
        MatchedResponsePatchEffect(
            exactKLTargetBaselineToPatched: 0.2,
            exactKLSourceBaselineToTargetBaseline: 1,
            exactKLSourceBaselineToPatched: 1 - gain,
            sourceDistributionClosenessGain: gain,
            sourceDistributionClosenessFraction: gain,
            nextTokenLogOddsShiftTowardSource: odds,
            sourceAndTargetNextTokensDiffer: true,
            targetContinuationMeanLogProbabilityDelta: -0.1,
            targetContinuationTokenCount: 4)
    }

    private func judgment(
        _ name: String, _ outcome: BehaviorOutcome
    ) -> RecordedOutcomeJudgment {
        RecordedOutcomeJudgment(
            name: name, channel: .contrast,
            classification: OutcomeClassification(
                outcome: outcome, confidence: 1,
                evidence: ["independent test judgment"]))
    }
}
