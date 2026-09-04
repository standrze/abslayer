import Foundation
import Testing
@testable import ProbeCore

private func sequenceFingerprint(
    name: String = "case", support: Double, tail: Double, target: Double
) -> SequenceLogitFingerprint {
    SequenceLogitFingerprint(
        vocabularySize: 3, topK: 1,
        cases: [SequenceCaseFingerprint(
            name: name, tokenIDs: [10, 11],
            positions: [SequencePositionFingerprint(
                targetTokenID: 11,
                supportTokenIDs: [0],
                supportLogProbabilities: [Float(log(support))],
                tailLogProbability: Float(log(tail)),
                targetLogProbability: Float(log(target)))])])
}

@Test func sequenceMetricsComputeCoarsenedKLLowerBoundAndPerplexity() throws {
    let baseline = sequenceFingerprint(support: 0.8, tail: 0.2, target: 0.8)
    let candidate = sequenceFingerprint(support: 0.4, tail: 0.6, target: 0.4)
    let result = try SequenceMetricEngine.compare(baseline: baseline, candidate: candidate)
    let expectedKL = 0.8 * log(0.8 / 0.4) + 0.2 * log(0.2 / 0.6)
    #expect(abs(result.coarsenedKLLowerBound - expectedKL) < 0.000_001)
    #expect(abs(result.baselinePerplexity - 1.25) < 0.000_001)
    #expect(abs(result.candidatePerplexity - 2.5) < 0.000_001)
    #expect(abs(result.perplexityRatio - 2) < 0.000_001)
    #expect(abs(result.baselineSupportMass - 0.8) < 0.000_001)
}

@Test func identicalSequenceFingerprintsHaveZeroKL() throws {
    let fingerprint = sequenceFingerprint(support: 0.7, tail: 0.3, target: 0.7)
    let result = try SequenceMetricEngine.compare(
        baseline: fingerprint, candidate: fingerprint)
    #expect(result.coarsenedKLLowerBound == 0)
    #expect(result.perplexityRatio == 1)
}

@Test func sequenceFingerprintRoundTrips() throws {
    let fingerprint = sequenceFingerprint(support: 0.7, tail: 0.3, target: 0.7)
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("abslayer-\(UUID().uuidString).abslseq")
    defer { try? FileManager.default.removeItem(at: url) }
    try fingerprint.write(to: url.path)
    #expect(try SequenceLogitFingerprint.read(from: url.path) == fingerprint)
}

@Test func sequenceMetricsRejectDifferentSupports() {
    let baseline = sequenceFingerprint(support: 0.7, tail: 0.3, target: 0.7)
    let mismatched = SequenceLogitFingerprint(
        vocabularySize: 3, topK: 1,
        cases: [SequenceCaseFingerprint(
            name: "case", tokenIDs: [10, 11],
            positions: [SequencePositionFingerprint(
                targetTokenID: 11, supportTokenIDs: [2],
                supportLogProbabilities: [Float(log(0.7))],
                tailLogProbability: Float(log(0.3)),
                targetLogProbability: Float(log(0.7)))])])
    #expect(throws: FingerprintError.self) {
        try SequenceMetricEngine.compare(baseline: baseline, candidate: mismatched)
    }
}
