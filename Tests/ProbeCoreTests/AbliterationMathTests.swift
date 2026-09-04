import Foundation
import Testing
@testable import ProbeCore

@Test func projectedDirectionIsOrthogonalToControl() {
    let direction = AbliterationMath.direction(
        contrast: [[1, 2]], control: [[1, 0]], projectAwayFromControl: true)
    #expect(abs(direction[0]) < 0.0001)
    #expect(abs(direction[1] - 1) < 0.0001)
}

@Test func winsorizationClampsOutlier() {
    #expect(AbliterationMath.winsorized([1, -2, 100], quantile: 0.5) == [1, -2, 2])
}

@Test func fractionalLayerDirectionInterpolates() {
    let direction = AbliterationMath.interpolatedDirection([[1, 0], [0, 1]], layer: 0.5)
    #expect(abs(direction[0] - direction[1]) < 0.0001)
}

@Test func blendedDirectionSpansGlobalAndPerLayerEndpoints() {
    let directions: [[Float]] = [[1, 0], [0, 1]]
    let global = AbliterationMath.blendedDirection(
        directions, globalLayer: 0, localLayer: 1, perLayerFraction: 0)
    let local = AbliterationMath.blendedDirection(
        directions, globalLayer: 0, localLayer: 1, perLayerFraction: 1)
    let middle = AbliterationMath.blendedDirection(
        directions, globalLayer: 0, localLayer: 1, perLayerFraction: 0.5)
    #expect(global == [1, 0])
    #expect(local == [0, 1])
    #expect(abs(middle[0] - middle[1]) < 0.0001)
}

@Test func blendedSubspacePreservesRankOrthonormalityAndEndpoints() {
    let subspaces: [[[Float]]] = [
        [[1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 1]],
    ]
    let global = AbliterationMath.blendedSubspace(
        subspaces, globalLayer: 0, localLayer: 1, perLayerFraction: 0)
    let local = AbliterationMath.blendedSubspace(
        subspaces, globalLayer: 0, localLayer: 1, perLayerFraction: 1)
    let middle = AbliterationMath.blendedSubspace(
        subspaces, globalLayer: 0, localLayer: 1, perLayerFraction: 0.5)
    #expect(global == subspaces[0])
    #expect(local == subspaces[1])
    #expect(middle.count == 2)
    let dot = zip(middle[0], middle[1]).reduce(Float.zero) { $0 + $1.0 * $1.1 }
    #expect(abs(dot) < 0.0001)
    for vector in middle {
        let norm = sqrt(vector.reduce(Float.zero) { $0 + $1 * $1 })
        #expect(abs(norm - 1) < 0.0001)
    }
}

@Test func sequentialGlobalScopePreservesOrderedNonorthogonalSourceBasis() {
    let subspaces: [[[Float]]] = [
        [[2, 0, 0], [2, 2, 0]],
        [[0, 3, 0], [0, 3, 3]],
    ]
    let directions = subspaces.map { $0[0] }
    let sourceAtFirstTarget = AbliterationMath.resolvedBasis(
        directions: directions, subspaces: subspaces,
        scope: .global(layer: 0), targetLayer: 0,
        composition: .sequential)
    let sourceAtDistantTarget = AbliterationMath.resolvedBasis(
        directions: directions, subspaces: subspaces,
        scope: .global(layer: 0), targetLayer: 34,
        composition: .sequential)

    #expect(sourceAtFirstTarget == sourceAtDistantTarget)
    #expect(sourceAtFirstTarget.count == 2)
    #expect(sourceAtFirstTarget[0] == [1, 0, 0])
    #expect(sourceAtFirstTarget[1][0] > 0.7)
    #expect(sourceAtFirstTarget[1][1] > 0.7)
    let orderedDot = zip(sourceAtFirstTarget[0], sourceAtFirstTarget[1])
        .reduce(Float.zero) { $0 + $1.0 * $1.1 }
    #expect(orderedDot > 0.7)

    let legacy = AbliterationMath.resolvedBasis(
        directions: directions, subspaces: subspaces,
        scope: .global(layer: 0), targetLayer: 34,
        composition: .simultaneous)
    #expect(legacy == AbliterationMath.interpolatedSubspace(subspaces, layer: 0))
    let legacyDot = zip(legacy[0], legacy[1])
        .reduce(Float.zero) { $0 + $1.0 * $1.1 }
    #expect(abs(legacyDot) < 0.0001)
}

@Test func sequentialBlendedScopePreservesOrderAtBothEndpoints() {
    let subspaces: [[[Float]]] = [
        [[2, 0, 0], [2, 2, 0]],
        [[0, 3, 0], [0, 3, 3]],
    ]
    let directions = subspaces.map { $0[0] }
    let global = AbliterationMath.resolvedBasis(
        directions: directions, subspaces: subspaces,
        scope: .blended(globalLayer: 0, perLayerFraction: 0), targetLayer: 1,
        composition: .sequential)
    let local = AbliterationMath.resolvedBasis(
        directions: directions, subspaces: subspaces,
        scope: .blended(globalLayer: 0, perLayerFraction: 1), targetLayer: 1,
        composition: .sequential)
    #expect(global == AbliterationMath.interpolatedOrderedDirections(subspaces, layer: 0))
    #expect(local == subspaces[1].map(AbliterationMath.normalized))
    let localDot = zip(local[0], local[1]).reduce(Float.zero) {
        $0 + $1.0 * $1.1
    }
    #expect(localDot > 0.7)

    let legacy = AbliterationMath.resolvedBasis(
        directions: directions, subspaces: subspaces,
        scope: .blended(globalLayer: 0, perLayerFraction: 0.5), targetLayer: 1,
        composition: .simultaneous)
    #expect(legacy == AbliterationMath.blendedSubspace(
        subspaces, globalLayer: 0, localLayer: 1, perLayerFraction: 0.5))
}

@Test func layerKernelTapersAndStops() {
    let kernel = LayerAblationKernel(maximum: 1, peakLayer: 5, minimum: 0.2, radius: 2)
    #expect(kernel.weight(at: 5) == 1)
    #expect(abs(kernel.weight(at: 4) - 0.6) < 0.0001)
    #expect(kernel.weight(at: 2) == 0)
}

@Test func configurationDefaultsToHistoricalSimultaneousComposition() {
    let zero = LayerAblationKernel(maximum: 0, peakLayer: 0, minimum: 0, radius: 0)
    let configuration = AbliterationConfiguration(attention: zero, mlp: zero)
    #expect(configuration.composition == .simultaneous)
}

@Test func fullNormalizationPreservesRowNorms() {
    let edited = AbliterationMath.edit(
        matrix: [3, 4, 0, 2], rows: 2, columns: 2,
        direction: [1, 0], strength: 0.5, normalization: .full)
    let firstNorm = sqrt(edited[0] * edited[0] + edited[1] * edited[1])
    let secondNorm = sqrt(edited[2] * edited[2] + edited[3] * edited[3])
    #expect(abs(firstNorm - 5) < 0.0001)
    #expect(abs(secondNorm - 2) < 0.0001)
}

@Test func sequentialCompositionDiffersForNonorthogonalDirections() {
    let matrix: [Float] = [
        1, 0,
        0, 1,
    ]
    let directions: [[Float]] = [
        [1, 0],
        [1, 1],
    ]
    let simultaneous = AbliterationMath.edit(
        matrix: matrix, rows: 2, columns: 2,
        directions: directions.map(AbliterationMath.normalized), strength: 1,
        composition: .simultaneous)
    let sequential = AbliterationMath.edit(
        matrix: matrix, rows: 2, columns: 2,
        directions: directions, strength: 1,
        composition: .sequential)

    // The second projector sees the result of the first one. For these
    // non-orthogonal axes the ordered composition is not the summed projector.
    #expect(zip(simultaneous, sequential).contains { abs($0 - $1) > 0.1 })
    #expect(abs(sequential[0]) < 0.0001)
    #expect(abs(sequential[1] + 0.5) < 0.0001)
    #expect(abs(sequential[2]) < 0.0001)
    #expect(abs(sequential[3] - 0.5) < 0.0001)
}
