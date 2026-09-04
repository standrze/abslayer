import Testing
@testable import ProbeCore

@Test func identicalVectorsHaveZeroDistance() {
    #expect(LayerMath.cosineDistance([1, 2, 3], [1, 2, 3]) < 0.000_001)
}

@Test func orthogonalVectorsHaveUnitDistance() {
    #expect(abs(LayerMath.cosineDistance([1, 0], [0, 1]) - 1) < 0.000_001)
}

@Test func matchingDeltasHaveFullAgreement() {
    let agreement = LayerMath.meanPairwiseAgreement([[1, 2], [2, 4], [0.5, 1]])
    #expect(abs(agreement - 1) < 0.000_001)
}

@Test func geometricMedianResistsSingleOutlier() {
    let median = LayerMath.geometricMedian([[0, 0], [0, 0.1], [0.1, 0], [100, 100]])
    #expect(median[0] < 0.2)
    #expect(median[1] < 0.2)
}

@Test func silhouetteFindsSeparatedClusters() {
    let score = LayerMath.binarySilhouette(
        [[1, 0], [0.99, 0.01]],
        [[0, 1], [0.01, 0.99]])
    #expect(score > 0.9)
}

@Test func silhouetteIsLowForMixedClusters() {
    let score = LayerMath.binarySilhouette(
        [[1, 0], [0, 1]],
        [[1, 0], [0, 1]])
    #expect(score < 0.1)
}
