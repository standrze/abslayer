import Foundation
import MLX
import Testing

@testable import ProbeCore

@Suite("Pairwise preference objective")
struct PairwisePreferenceObjectiveTests {
    @Test("SimPO constructs and averages one margin per pair")
    func perPairMargins() throws {
        let metrics = try PairwisePreferenceMath.simpo(
            chosenNegativeLogLikelihood: [1, 4],
            rejectedNegativeLogLikelihood: [3, 2],
            beta: 1, gamma: 0, sftWeight: 0.25)

        let expectedPreference = (
            log1p(exp(Float(-2))) + log1p(exp(Float(2)))) / 2
        #expect(abs(metrics.preferenceLoss - expectedPreference) < 0.000_001)
        #expect(abs(metrics.chosenNegativeLogLikelihood - 2.5) < 0.000_001)
        #expect(abs(metrics.loss - (expectedPreference + 0.625)) < 0.000_001)
        #expect(metrics.pairwiseAccuracy == 0.5)
    }

    @Test("opposing pairs cannot cancel into a misleading pooled margin")
    func noPooledCancellation() throws {
        let metrics = try PairwisePreferenceMath.simpo(
            chosenNegativeLogLikelihood: [0, 10],
            rejectedNegativeLogLikelihood: [10, 0],
            beta: 1, gamma: 0, sftWeight: 0)

        // A pooled implementation sees equal means and reports log(2). The
        // correct pairwise loss strongly penalizes the second, reversed pair.
        #expect(metrics.preferenceLoss > 4.9)
        #expect(metrics.pairwiseAccuracy == 0.5)
    }

    @Test("MLX objective length-normalizes each pair before reduction")
    func mlxPerPairReduction() {
        let metrics = PairwisePreferenceTensorMath.evaluate(
            chosenTokenLosses: MLXArray([0, 0, 10, 0] as [Float], [2, 2]),
            chosenMask: MLXArray([1, 0, 1, 1] as [Float], [2, 2]),
            rejectedTokenLosses: MLXArray([10, 0, 0, 0] as [Float], [2, 2]),
            rejectedMask: MLXArray([1, 0, 1, 1] as [Float], [2, 2]),
            mode: .simpoPairwise, beta: 1, gamma: 0, sftWeight: 0)
        eval(
            metrics.loss, metrics.preferenceLoss,
            metrics.chosenNegativeLogLikelihood, metrics.pairwiseAccuracy)

        let expected = (log1p(exp(Float(-10))) + log1p(exp(Float(5)))) / 2
        #expect(abs(metrics.preferenceLoss.item(Float.self) - expected) < 0.000_01)
        #expect(abs(metrics.chosenNegativeLogLikelihood.item(Float.self) - 2.5) < 0.000_01)
        #expect(metrics.pairwiseAccuracy.item(Float.self) == 0.5)
    }

    @Test("reference-relative DPO starts at log two and measures improvement")
    func referenceRelativeDPO() throws {
        let atReference = try PairwisePreferenceMath.dpo(
            chosenNegativeLogLikelihood: [4],
            rejectedNegativeLogLikelihood: [1],
            referenceChosenNegativeLogLikelihood: [4],
            referenceRejectedNegativeLogLikelihood: [1],
            beta: 2, gamma: 0, sftWeight: 0)
        #expect(abs(atReference.preferenceLoss - log(Float(2))) < 0.000_001)
        #expect(atReference.pairwiseAccuracy == 0)

        let improved = try PairwisePreferenceMath.dpo(
            chosenNegativeLogLikelihood: [2],
            rejectedNegativeLogLikelihood: [2],
            referenceChosenNegativeLogLikelihood: [4],
            referenceRejectedNegativeLogLikelihood: [1],
            beta: 1, gamma: 0, sftWeight: 0.25)
        #expect(improved.preferenceLoss < 0.05)
        #expect(improved.pairwiseAccuracy == 1)
        #expect(abs(improved.loss - (improved.preferenceLoss + 0.5)) < 0.000_001)
    }

    @Test("onset weights emphasize only the first four assistant tokens")
    func onsetWeights() {
        let mask = MLXArray(
            [0, 0, 1, 1, 1, 1, 1, 0] as [Float], [1, 8])
        let weights = PairwisePreferenceTensorMath.onsetWeights(mask: mask)
        eval(weights)
        #expect(weights.asArray(Float.self) == [0, 0, 8, 4, 4, 4, 1, 0])
    }

    @Test("MLX DPO uses fixed reference advantages")
    func mlxReferenceRelativeDPO() {
        let metrics = PairwisePreferenceTensorMath.evaluate(
            chosenTokenLosses: MLXArray([2, 2] as [Float], [1, 2]),
            chosenMask: MLXArray([1, 1] as [Float], [1, 2]),
            rejectedTokenLosses: MLXArray([4, 4] as [Float], [1, 2]),
            rejectedMask: MLXArray([1, 1] as [Float], [1, 2]),
            mode: .dpoOnset, beta: 1, gamma: 0, sftWeight: 0,
            referenceChosenNegativeLogLikelihood: MLXArray([4] as [Float]),
            referenceRejectedNegativeLogLikelihood: MLXArray([1] as [Float]))
        eval(metrics.preferenceLoss, metrics.pairwiseAccuracy)

        let expected = log1p(exp(Float(-5)))
        #expect(abs(metrics.preferenceLoss.item(Float.self) - expected) < 0.000_001)
        #expect(metrics.pairwiseAccuracy.item(Float.self) == 1)
    }

    @Test("new and legacy modes are explicit and aliases remain stable")
    func objectiveModeParsing() throws {
        #expect(try PairwisePreferenceObjectiveMode.parse(nil) == .simpoPairwise)
        #expect(try PairwisePreferenceObjectiveMode.parse("simpo") == .simpoPairwise)
        #expect(try PairwisePreferenceObjectiveMode.parse("dpo") == .dpoOnset)
        #expect(try PairwisePreferenceObjectiveMode.parse("legacy-pooled") == .legacyPooled)
        #expect(throws: PairwisePreferenceObjectiveError.unknownMode("dpo-ish")) {
            try PairwisePreferenceObjectiveMode.parse("dpo-ish")
        }
    }

    @Test("seeded shuffles reproduce exactly and vary across seeds")
    func deterministicShuffle() {
        var first = Array(0 ..< 32)
        var second = first
        var third = first
        var firstGenerator = ABSlayerSeededGenerator(seed: 1729)
        var secondGenerator = ABSlayerSeededGenerator(seed: 1729)
        var thirdGenerator = ABSlayerSeededGenerator(seed: 1730)
        first.shuffle(using: &firstGenerator)
        second.shuffle(using: &secondGenerator)
        third.shuffle(using: &thirdGenerator)

        #expect(first == second)
        #expect(first != third)
        #expect(first != Array(0 ..< 32))
    }

    @Test("validation split keeps repeated prompt groups on one side")
    func groupAwareValidationSplit() throws {
        let keys = ["weighted", "other", "weighted", "third", "other", "fourth"]
        let first = try PreferenceGroupSplit.make(
            groupKeys: keys, validationFraction: 0.25, seed: 1729)
        let second = try PreferenceGroupSplit.make(
            groupKeys: keys, validationFraction: 0.25, seed: 1729)

        #expect(first == second)
        let trainingGroups = Set(first.trainingIndices.map { keys[$0] })
        let validationGroups = Set(first.validationIndices.map { keys[$0] })
        #expect(trainingGroups.isDisjoint(with: validationGroups))
        #expect(first.trainingIndices.count + first.validationIndices.count == keys.count)
        #expect(first.trainingGroupCount + first.validationGroupCount == 4)
    }

    @Test("validation refuses to leak a single prompt group")
    func singleGroupCannotSplit() {
        #expect(throws: PairwisePreferenceObjectiveError.insufficientGroups) {
            try PreferenceGroupSplit.make(
                groupKeys: ["same", "same"], seed: 1729)
        }
    }
}
