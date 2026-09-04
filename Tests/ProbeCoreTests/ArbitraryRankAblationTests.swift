import MLX
import Testing
@testable import ProbeCore

@Test func araDefaultUsesFullModelStackWithoutInheritingReferenceWindow() {
    let defaults = ARAParameters.fullStackDefault(layerCount: 35)
    #expect(defaults.startLayerIndex == 0)
    #expect(defaults.endLayerIndex == 35)
    #expect(defaults.preserveGoodBehaviorWeight
        == ARAParameters.gemma4E2BHereticSeed.preserveGoodBehaviorWeight)
    #expect(ARAParameters.gemma4E2BHereticSeed.startLayerIndex == 17)
    #expect(ARAParameters.gemma4E2BHereticSeed.endLayerIndex == 24)
}

@Test func araNearestNeighborDistancesMatchHandCalculation() {
    let queries = MLXArray([Float(0), 0, 3, 4]).reshaped(2, 2)
    let references = MLXArray([Float(0), 0, 0, 2, 6, 8]).reshaped(3, 2)
    let distances = ArbitraryRankAblation.meanDistancesToKNearestNeighbors(
        queries, among: references, k: 2)
    eval(distances)
    let values = distances.asArray(Float.self)
    #expect(abs(values[0] - 1) < 0.0001)
    let expected = Float(((13.0 as Double).squareRoot() + 5) / 2)
    #expect(abs(values[1] - expected) < 0.0001)
}

@Test func araLimitedMemoryBFGSLowersObjective() {
    let original = MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2)
    let samples = ARAModuleSamples(
        goodInput: MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2),
        goodOutput: MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2),
        badInput: MLXArray([Float(2), 2, 2, 1]).reshaped(2, 2),
        badOutput: MLXArray([Float(2), 2, 2, 1]).reshaped(2, 2))
    let parameters = ARAParameters(
        startLayerIndex: 0, endLayerIndex: 1,
        preserveGoodBehaviorWeight: 0.25,
        steerBadBehaviorWeight: 1,
        overcorrectRelativeWeight: 0,
        neighborCount: 1)
    let initial = ArbitraryRankAblation.objective(
        weight: original, samples: samples, parameters: parameters)
    let result = ArbitraryRankAblation.optimize(
        originalWeight: original, samples: samples, parameters: parameters,
        maximumIterations: 30)
    let final = ArbitraryRankAblation.objective(
        weight: result.weight, samples: samples, parameters: parameters)
    eval(initial, final)
    #expect(!result.steps.isEmpty)
    #expect(final.item(Float.self) < initial.item(Float.self))
}

@Test func araFullNormalizationPreservesEverySourceRowNorm() {
    let original = MLXArray([Float(3), 4, 0, 2]).reshaped(2, 2)
    let samples = ARAModuleSamples(
        goodInput: MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2),
        goodOutput: matmul(
            MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2), original.T),
        badInput: MLXArray([Float(2), 1, 1, 2]).reshaped(2, 2),
        badOutput: matmul(
            MLXArray([Float(2), 1, 1, 2]).reshaped(2, 2), original.T))
    let parameters = ARAParameters(
        startLayerIndex: 0, endLayerIndex: 1,
        preserveGoodBehaviorWeight: 0.1,
        steerBadBehaviorWeight: 1,
        overcorrectRelativeWeight: 0,
        neighborCount: 1,
        rowNormalization: .full)
    let result = ArbitraryRankAblation.optimize(
        originalWeight: original, samples: samples, parameters: parameters,
        maximumIterations: 20)
    let rowNorms = sqrt((result.weight * result.weight).sum(axis: 1))
    eval(rowNorms)
    let values = rowNorms.asArray(Float.self)
    #expect(abs(values[0] - 5) < 0.001)
    #expect(abs(values[1] - 2) < 0.001)
}

@Test func araTrustRegionConstrainsTheOptimizationPath() {
    let original = MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2)
    let samples = ARAModuleSamples(
        goodInput: MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2),
        goodOutput: MLXArray([Float(1), 0, 0, 1]).reshaped(2, 2),
        badInput: MLXArray([Float(2), 2, 2, 1]).reshaped(2, 2),
        badOutput: MLXArray([Float(2), 2, 2, 1]).reshaped(2, 2))
    let parameters = ARAParameters(
        startLayerIndex: 0, endLayerIndex: 1,
        preserveGoodBehaviorWeight: 0.25,
        steerBadBehaviorWeight: 1,
        overcorrectRelativeWeight: 0,
        neighborCount: 1)
    let result = ArbitraryRankAblation.optimize(
        originalWeight: original, samples: samples, parameters: parameters,
        maximumIterations: 30, maximumRelativeChange: 0.02)
    let delta = result.weight.asType(.float32) - original
    let relativeChange = sqrt((delta * delta).sum())
        / sqrt((original * original).sum())
    eval(relativeChange)
    #expect(!result.steps.isEmpty)
    #expect(relativeChange.item(Float.self) <= 0.02001)
}
