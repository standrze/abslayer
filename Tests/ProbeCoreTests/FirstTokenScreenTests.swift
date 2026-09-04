import Foundation
import Testing
@testable import ProbeCore

private func firstTokenFingerprint(
    names: [String] = ["one"],
    probabilities: [[Double]],
    topTokenIDs: [Int]
) -> FirstTokenChannelFingerprint {
    FirstTokenChannelFingerprint(
        promptNames: names,
        vocabularySize: probabilities[0].count,
        logProbabilities: probabilities.map { $0.map { Float(log($0)) } },
        topTokenIDs: topTokenIDs,
        topTokenTexts: topTokenIDs.map { "token-\($0)" })
}

private let starterVocabulary = FirstTokenStarterVocabulary(
    refusalPhrases: ["I"],
    compliancePhrases: ["Sure"],
    refusalTokens: [FirstTokenStarterToken(
        tokenID: 0, tokenText: "I", phrases: ["I"])],
    complianceTokens: [FirstTokenStarterToken(
        tokenID: 1, tokenText: "Sure", phrases: ["Sure"])])

@Test func firstTokenExactKLUsesTheFullVocabulary() throws {
    let baseline = firstTokenFingerprint(
        probabilities: [[0.8, 0.15, 0.05]], topTokenIDs: [0])
    let candidate = firstTokenFingerprint(
        probabilities: [[0.5, 0.25, 0.25]], topTokenIDs: [0])
    let expected = 0.8 * log(0.8 / 0.5)
        + 0.15 * log(0.15 / 0.25)
        + 0.05 * log(0.05 / 0.25)
    let actual = try FirstTokenScreenMath.exactMeanKL(
        baseline: baseline, candidate: candidate)
    #expect(abs(actual - expected) < 0.000_001)
}

@Test func firstTokenCandidateReportsChannelSpecificStarterAndTop1Shifts() throws {
    let baseline = firstTokenFingerprint(
        names: ["one", "two"],
        probabilities: [[0.6, 0.3, 0.1], [0.5, 0.2, 0.3]],
        topTokenIDs: [0, 0])
    let candidate = firstTokenFingerprint(
        names: ["one", "two"],
        probabilities: [[0.2, 0.7, 0.1], [0.4, 0.3, 0.3]],
        topTokenIDs: [1, 0])
    let report = try FirstTokenScreenMath.candidateReport(
        baseline: baseline, candidate: candidate,
        vocabulary: starterVocabulary)
    #expect(report.top1ChangeCount == 1)
    #expect(report.top1ChangeFraction == 0.5)
    #expect(report.top1Changes.map(\.caseName) == ["one"])
    #expect(report.meanRefusalStarterMassDeltaFromBaseline < 0)
    #expect(report.meanComplianceStarterMassDeltaFromBaseline > 0)
    #expect(report.meanComplianceToRefusalLogOddsDeltaFromBaseline > 0)
}

@Test func starterVocabularyDeduplicatesFirstTokensAndRejectsOverlap() throws {
    let configuration = FirstTokenStarterConfiguration(
        refusalPhrases: ["I", "I'm sorry"],
        compliancePhrases: ["Sure", "Here"])
    let vocabulary = try FirstTokenScreenMath.makeStarterVocabulary(
        configuration: configuration,
        refusalResolutions: [
            FirstTokenPhraseResolution(phrase: "I", tokenID: 4, tokenText: "I"),
            FirstTokenPhraseResolution(
                phrase: "I'm sorry", tokenID: 4, tokenText: "I"),
        ],
        complianceResolutions: [
            FirstTokenPhraseResolution(phrase: "Sure", tokenID: 8, tokenText: "Sure"),
            FirstTokenPhraseResolution(phrase: "Here", tokenID: 9, tokenText: "Here"),
        ])
    #expect(vocabulary.refusalTokenIDs == [4])
    #expect(vocabulary.refusalTokens[0].phrases == ["I", "I'm sorry"])

    #expect(throws: FirstTokenScreenError.overlappingStarterTokens([4])) {
        try FirstTokenScreenMath.makeStarterVocabulary(
            configuration: configuration,
            refusalResolutions: [FirstTokenPhraseResolution(
                phrase: "I", tokenID: 4, tokenText: "I")],
            complianceResolutions: [FirstTokenPhraseResolution(
                phrase: "Sure", tokenID: 4, tokenText: "I")])
    }
}

@Test func firstTokenSelectionIsEvenAndStable() throws {
    let pairs = (0 ..< 10).map {
        PromptPair(name: "p\($0)", contrast: "c\($0)", control: "b\($0)")
    }
    let selected = try FirstTokenScreenMath.evenlySpaced(pairs, maximum: 4)
    #expect(selected.map(\.name) == ["p0", "p2", "p5", "p7"])
}

@Test func unloadValidationDetectsResidualAdapterEffects() throws {
    let baselineChannel = firstTokenFingerprint(
        probabilities: [[0.7, 0.2, 0.1]], topTokenIDs: [0])
    let restoredChannel = firstTokenFingerprint(
        probabilities: [[0.69, 0.21, 0.1]], topTokenIDs: [0])
    let baseline = FirstTokenDualFingerprint(
        contrast: baselineChannel, control: baselineChannel)
    let restored = FirstTokenDualFingerprint(
        contrast: restoredChannel, control: baselineChannel)
    let validation = try FirstTokenScreenMath.unloadValidation(
        baseline: baseline, restored: restored, tolerance: 0.001)
    #expect(!validation.passed)
    #expect(validation.contrastMaximumAbsoluteLogProbabilityDifference > 0.001)
    #expect(validation.controlMaximumAbsoluteLogProbabilityDifference == 0)
}
