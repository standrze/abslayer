import Foundation
import Testing
@testable import ProbeCore

@Test func exactTeacherForcedKLUsesTheCompleteVocabulary() throws {
    let baseline = [Float(log(0.75)), Float(log(0.25))]
    let candidate = [Float(log(0.5)), Float(log(0.5))]
    let measured = try TeacherForcedContinuationMetricEngine.exactKL(
        baselineLogProbabilities: baseline,
        candidateLogProbabilities: candidate)
    let expected = 0.75 * log(0.75 / 0.5) + 0.25 * log(0.25 / 0.5)
    #expect(abs(measured - expected) < 0.000_001)
}

@Test func teacherForcedSummaryReportsWeightedMeanP95MaximumAndPerplexity() throws {
    let summary = try TeacherForcedContinuationMetricEngine.summarize([
        TeacherForcedContinuationCaseSamples(
            name: "short",
            tokenKLDivergences: [0.1],
            baselineTargetLogProbabilities: [log(0.5)],
            candidateTargetLogProbabilities: [log(0.25)]),
        TeacherForcedContinuationCaseSamples(
            name: "long",
            tokenKLDivergences: [0.2, 0.3, 0.4],
            baselineTargetLogProbabilities: Array(repeating: log(0.5), count: 3),
            candidateTargetLogProbabilities: Array(repeating: log(0.25), count: 3)),
    ])

    #expect(summary.tokenCount == 4)
    #expect(abs(summary.exactMeanKL - 0.25) < 0.000_001)
    #expect(summary.exactP95KL == 0.4)
    #expect(summary.exactMaximumKL == 0.4)
    #expect(abs(summary.baselineReferencePerplexity - 2) < 0.000_001)
    #expect(abs(summary.candidateReferencePerplexity - 4) < 0.000_001)
    #expect(abs(summary.referencePerplexityRatio - 2) < 0.000_001)
    #expect(summary.cases.map(\.tokenCount) == [1, 3])
}

@Test func teacherForcedMetricsRejectMismatchedRows() {
    #expect(throws: FingerprintError.self) {
        try TeacherForcedContinuationMetricEngine.exactKL(
            baselineLogProbabilities: [0, -1],
            candidateLogProbabilities: [0])
    }
    #expect(throws: FingerprintError.self) {
        try TeacherForcedContinuationMetricEngine.summarize([
            TeacherForcedContinuationCaseSamples(
                name: "bad",
                tokenKLDivergences: [0.1, 0.2],
                baselineTargetLogProbabilities: [0],
                candidateTargetLogProbabilities: [0, 0]),
        ])
    }
}

@Test func promptPairReferenceResponseIsOptionalAndCodable() throws {
    let legacy = Data(#"{"name":"legacy","contrast":"c","control":"b"}"#.utf8)
    let decoded = try JSONDecoder().decode(PromptPair.self, from: legacy)
    #expect(decoded.controlReferenceResponse == nil)

    let pair = PromptPair(
        name: "reference", contrast: "c", control: "b",
        controlReferenceResponse: "known-good assistant answer")
    let roundTripped = try JSONDecoder().decode(
        PromptPair.self, from: JSONEncoder().encode(pair))
    #expect(roundTripped == pair)
}

@Test func continuationDerivationPrefersTheExactDeployedPromptBoundary() throws {
    let tokens = try TeacherForcedContinuationTokenDerivation.continuation(
        promptTokens: [2, 105, 4368, 107],
        promptPlusReferenceTokens: [2, 105, 4368, 107, 41, 42],
        // A completed chat can contain a different header or a turn suffix;
        // it must not override the direct deployed-prompt continuation.
        templatedConversationTokens: [2, 105, 999, 41, 42, 106])
    #expect(tokens == [41, 42])
}

@Test func continuationDerivationUsesOnlyAPrefixVerifiedTemplateFallback() throws {
    let tokens = try TeacherForcedContinuationTokenDerivation.continuation(
        promptTokens: [2, 105, 4368, 107],
        promptPlusReferenceTokens: [2, 105, 999, 41],
        templatedConversationTokens: [2, 105, 4368, 107, 41, 42, 106])
    #expect(tokens == [41, 42, 106])

    #expect(throws: FingerprintError.self) {
        try TeacherForcedContinuationTokenDerivation.continuation(
            promptTokens: [1, 2],
            promptPlusReferenceTokens: [1, 9, 3],
            templatedConversationTokens: [1, 8, 3])
    }
}

@Test func continuationKLCaseSelectionIsDeterministicAndEvenlySpaced() {
    let pairs = (0 ..< 10).map {
        PromptPair(
            name: "case-\($0)", contrast: "c", control: "b",
            controlReferenceResponse: "answer")
    }
    let selected = LoRAContinuationKLEngine.evenlySpaced(pairs, maximum: 3)
    #expect(selected.map(\.name) == ["case-0", "case-3", "case-6"])
}

@Test func continuationKLScaleZeroUsesTheUntouchedModelPath() {
    #expect(!LoRAContinuationKLEngine.shouldInstallAdapter(effectiveScale: 0))
    #expect(LoRAContinuationKLEngine.shouldInstallAdapter(effectiveScale: 16))
    #expect(LoRAContinuationKLEngine.shouldInstallAdapter(effectiveScale: -1))
}

@Test func continuationKLReportRoundTripsItsProvenanceAndSummary() throws {
    let summary = try TeacherForcedContinuationMetricEngine.summarize([
        TeacherForcedContinuationCaseSamples(
            name: "control-1",
            tokenKLDivergences: [0.01, 0.02],
            baselineTargetLogProbabilities: [log(0.5), log(0.25)],
            candidateTargetLogProbabilities: [log(0.4), log(0.2)]),
    ])
    let report = LoRAContinuationKLReport(
        modelDirectory: "/models/base",
        adapterDirectory: "/adapters/candidate",
        adapterScale: 16,
        requestedMaximumCases: 32,
        maximumReferenceTokens: 64,
        summary: summary)

    let restored = try JSONDecoder().decode(
        LoRAContinuationKLReport.self,
        from: JSONEncoder().encode(report))
    #expect(restored == report)
    #expect(restored.schemaVersion == 1)
    #expect(restored.summary.tokenCount == 2)
}
