import Foundation
import Testing
@testable import ProbeCore

@Suite("Prompt-end cross-condition patching")
struct PromptEndCrossConditionPatchingTests {
    @Test("adapter delta reconstructs donor and reverse is directional")
    func adapterDelta() throws {
        let base: [Float] = [1, 2, 3]
        let donor: [Float] = [2, 0, 7]
        let delta = try PromptEndPatchMath.adapterDelta(
            base: base, donor: donor)
        #expect(delta == [1, -2, 4])
        #expect(try PromptEndPatchMath.adding(delta, to: base) == donor)
        #expect(
            try PromptEndPatchMath.adding(delta, to: base, scale: -1)
                == [0, 4, -1])
    }

    @Test("unrelated adapter control matches matched delta norm")
    func normMatchedRandom() throws {
        let target: [Float] = [10, 20]
        let matched: [Float] = [3, 4]
        let replacement = try PromptEndPatchMath
            .normMatchedRandomReplacement(
                targetBase: target, matchedDelta: matched,
                unrelatedBase: [1, 1], unrelatedDonor: [1, 3])
        let injected = zip(replacement, target).map(-)
        #expect(
            abs(
                PromptEndPatchMath.l2Norm(injected)
                    - PromptEndPatchMath.l2Norm(matched)) < 1e-6)
        #expect(replacement[0] == 10)
        #expect(abs(replacement[1] - 25) < 1e-6)
    }

    @Test("donor closeness is exact and normalized")
    func effect() throws {
        let donor = normalizedLogs([0.8, 0.2])
        let base = normalizedLogs([0.2, 0.8])
        let exact = try PromptEndPatchMath.effect(
            donorLogProbabilities: donor,
            baseLogProbabilities: base,
            conditionLogProbabilities: donor)
        #expect(exact.exactKLDonorToCondition < 1e-6)
        #expect(exact.donorClosenessFraction != nil)
        #expect(abs((exact.donorClosenessFraction ?? 0) - 1) < 1e-6)
        #expect(exact.donorTopTokenID == 0)
        #expect(exact.baseTopTokenID == 1)
        #expect(exact.conditionTopTokenID == 0)
    }

    @Test("case offset supports disjoint discovery and validation")
    func configurationOffset() throws {
        let discovery = try PromptEndPatchConfiguration(
            layersZeroBased: [16, 19, 23], caseOffset: 0,
            maximumCases: 8)
        let validation = try PromptEndPatchConfiguration(
            layersZeroBased: [23], caseOffset: 8,
            maximumCases: 8)
        try discovery.validate(caseCount: 16, decoderLayerCount: 40)
        try validation.validate(caseCount: 16, decoderLayerCount: 40)
        #expect(discovery.caseOffset == 0)
        #expect(validation.caseOffset == 8)
    }

    @Test("maximum absolute difference is shape strict")
    func maximumDifference() {
        #expect(
            PromptEndPatchMath.maximumAbsoluteDifference(
                [1, 2], [1, 2.25]) == 0.25)
        #expect(
            PromptEndPatchMath.maximumAbsoluteDifference(
                [1], [1, 2]).isInfinite)
    }

    private func normalizedLogs(_ probabilities: [Float]) -> [Float] {
        probabilities.map { log($0) }
    }
}
