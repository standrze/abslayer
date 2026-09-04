import Foundation
import Testing
@testable import ProbeCore

@Test func identicalFingerprintsHaveZeroKL() throws {
    let fingerprint = LogitFingerprint(
        promptNames: ["one"], vocabularySize: 2,
        logProbabilities: [[Float(log(0.25)), Float(log(0.75))]])
    let divergence = try LogitFingerprintEngine.divergence(
        baseline: fingerprint, candidate: fingerprint)
    #expect(abs(divergence) < 0.000_001)
}

@Test func fingerprintRoundTrips() throws {
    let fingerprint = LogitFingerprint(
        promptNames: ["one", "two"], vocabularySize: 2,
        logProbabilities: [[-1, -2], [-3, -4]])
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("abslayer-\(UUID().uuidString).abslkl")
    defer { try? FileManager.default.removeItem(at: url) }
    try fingerprint.write(to: url.path)
    let restored = try LogitFingerprint.read(from: url.path)
    #expect(restored.promptNames == fingerprint.promptNames)
    #expect(restored.vocabularySize == 2)
    #expect(restored.logProbabilities == fingerprint.logProbabilities)
}
