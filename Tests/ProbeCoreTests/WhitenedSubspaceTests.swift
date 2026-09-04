import Foundation
import Testing
@testable import ProbeCore

private func absoluteDot(_ lhs: [Float], _ rhs: [Float]) -> Float {
    abs(zip(lhs, rhs).reduce(Float.zero) { $0 + $1.0 * $1.1 })
}

@Test func whitenedSubspacePrefersSignalRelativeToBenignVariance() {
    // The first feature has large ordinary benign variance and a similarly
    // large nuisance delta.  The second feature has low benign variance but a
    // consistent class delta, so covariance whitening should select it.
    let control: [[Float]] = [
        [-20, -1], [-10, 1], [0, -1], [10, 1], [20, -1], [0, 1],
    ]
    let nuisance: [Float] = [8, -8, 8, -8, 8, -8]
    let contrast = zip(control, nuisance).map { row, noise in
        [row[0] + noise, row[1] + 2]
    }

    let basis = whitenedSubspace(
        contrast: contrast, control: control, rank: 1,
        minimumVarianceRatio: 0)

    #expect(basis.count == 1)
    #expect(absoluteDot(basis[0], [0, 1]) > 0.90)
}

@Test func whitenedSubspaceIsTranslationInvariantAndOrthonormal() {
    let control: [[Float]] = [
        [-2, -1, 0.5], [-1, 0, -0.5], [0, 1, 1.5],
        [1, -2, -1.5], [2, 2, 0],
    ]
    let delta: [[Float]] = [
        [1.25, -0.5, 0.75], [0.8, 0.25, -0.1], [1.6, -0.75, 0.35],
        [0.45, 0.9, -0.6], [1.1, -0.2, 0.95],
    ]
    let contrast = zip(control, delta).map { zip($0, $1).map(+) }
    let offset: [Float] = [8, -3, 11]
    let translatedControl = control.map { zip($0, offset).map(+) }
    let translatedContrast = contrast.map { zip($0, offset).map(+) }

    let original = whitenedSubspace(
        contrast: contrast, control: control, rank: 2,
        minimumVarianceRatio: 0)
    let translated = whitenedSubspace(
        contrast: translatedContrast, control: translatedControl, rank: 2,
        minimumVarianceRatio: 0)

    #expect(original.count == 2)
    #expect(translated.count == 2)
    for vector in original {
        #expect(abs(absoluteDot(vector, vector) - 1) < 0.0001)
    }
    #expect(absoluteDot(original[0], original[1]) < 0.0001)
    // SVD signs are arbitrary; compare corresponding axes sign-invariantly.
    #expect(absoluteDot(original[0], translated[0]) > 0.999)
    #expect(absoluteDot(original[1], translated[1]) > 0.999)
}

@Test func whitenedSubspaceRetainsSignalOutsideObservedBenignSpan() {
    // The harmless sample covariance spans only x. A ridge-whitened method
    // must retain the y refusal delta rather than project it away merely
    // because the small harmless sample never varied along y.
    let control: [[Float]] = [
        [-3, 0, 0], [-1, 0, 0], [1, 0, 0], [3, 0, 0],
    ]
    let contrast = control.map { [$0[0], 2, 0] }

    let basis = whitenedSubspace(
        contrast: contrast, control: control, rank: 1,
        minimumVarianceRatio: 0)

    #expect(basis.count == 1)
    #expect(absoluteDot(basis[0], [0, 1, 0]) > 0.99)
}
