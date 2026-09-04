import Foundation
import Testing

@testable import ProbeCore

@Suite("Gemma prompt and KV-cache parity diagnostics")
struct GemmaParityDiagnosticTests {
    @Test func recordSelectionUsesTrainerNormalizedTextAndStableID() throws {
        let data = try JSONEncoder().encode(
            HarnessTrainingDataset(
                datasetName: " exact-copy ",
                records: [
                    HarnessTrainingRecord(
                        id: " first ", instruction: " prompt one ", response: " answer one "),
                    HarnessTrainingRecord(
                        id: "second", instruction: " prompt two", response: "answer two "),
                ]))

        let selected = try GemmaParityRecordLoader.load(data, recordID: " second ")
        #expect(
            selected
                == GemmaParityRecord(
                    datasetName: "exact-copy", id: "second",
                    instruction: "prompt two", response: "answer two"))
    }

    @Test func recordSelectionRejectsAnonymousLegacyDataAndUnknownIDs() throws {
        let legacy = try JSONEncoder().encode([
            LoRATrainingExample(prompt: "one", target: "answer one"),
            LoRATrainingExample(prompt: "two", target: "answer two"),
        ])
        #expect(throws: GemmaParityDiagnosticError.versionedDatasetRequired) {
            try GemmaParityRecordLoader.load(legacy)
        }

        let envelope = try JSONEncoder().encode(
            HarnessTrainingDataset(
                datasetName: "copy",
                records: [
                    HarnessTrainingRecord(id: "one", instruction: "p1", response: "r1"),
                    HarnessTrainingRecord(id: "two", instruction: "p2", response: "r2"),
                ]))
        #expect(throws: GemmaParityDiagnosticError.recordNotFound("missing")) {
            try GemmaParityRecordLoader.load(envelope, recordID: "missing")
        }
    }

    @Test func derivesTheExactTrainerPrefixAndSupervisesTheCompletedSuffix() throws {
        let partition = try GemmaTrainingTokenPartition.derive(
            completeTokens: [1, 2, 10, 11, 90, 91],
            deployedPromptTokens: [1, 2])

        #expect(partition.trainingPrefixTokens == [1, 2])
        #expect(partition.targetTokens == [10, 11, 90, 91])
        #expect(partition.trailingTemplateTokens.isEmpty)
        #expect(partition.assistantStart == 2)
        #expect(partition.assistantEnd == 6)
    }

    @Test func promptParityLocatesValueAndLengthDivergences() throws {
        let partition = try GemmaTrainingTokenPartition.derive(
            completeTokens: [1, 2, 10, 90],
            deployedPromptTokens: [1, 2])
        let exact = GemmaPromptParityReport.make(
            partition: partition, inferencePrefixTokens: [1, 2],
            tokenText: { "t\($0)" })
        #expect(exact.exact)
        #expect(exact.firstDivergence == nil)

        let changed = GemmaPromptParityReport.make(
            partition: partition, inferencePrefixTokens: [1, 3],
            tokenText: { "t\($0)" })
        #expect(changed.firstDivergence?.tokenIndex == 1)
        #expect(changed.firstDivergence?.trainingTokenID == 2)
        #expect(changed.firstDivergence?.inferenceTokenID == 3)

        let shortened = GemmaPromptParityReport.make(
            partition: partition, inferencePrefixTokens: [1],
            tokenText: { "t\($0)" })
        #expect(shortened.firstDivergence?.tokenIndex == 1)
        #expect(shortened.firstDivergence?.trainingTokenID == 2)
        #expect(shortened.firstDivergence?.inferenceTokenID == nil)
    }

    @Test func predictionReportsLocateTheFirstTeacherAndCacheMismatch() throws {
        let teacher = try GemmaTeacherForcedReport.make(
            expected: [10, 11, 12], predicted: [10, 99, 12], assistantStart: 7,
            tokenText: { "t\($0)" })
        #expect(teacher.greedyMatchCount == 2)
        #expect(abs(teacher.greedyAccuracy - 2.0 / 3.0) < 0.000_001)
        #expect(teacher.firstMismatch?.targetOffset == 1)
        #expect(teacher.firstMismatch?.completeTokenIndex == 8)

        let cache = try GemmaCachePredictionReport.make(
            mode: .balanced,
            expected: [10, 11, 12],
            fullSequencePredictions: [10, 11, 12],
            cachedPredictions: [10, 11, 77],
            assistantStart: 7,
            tokenText: { "t\($0)" })
        #expect(cache.fullSequenceMatchCount == 2)
        #expect(!cache.predictionsMatchFullSequence)
        #expect(cache.firstDivergence?.targetOffset == 2)
        #expect(cache.firstDivergence?.completeTokenIndex == 9)
    }

    @Test func diagnosisSeparatesPromptCacheChunkingAndModelFailures() throws {
        let partition = try GemmaTrainingTokenPartition.derive(
            completeTokens: [1, 2, 10, 90],
            deployedPromptTokens: [1, 2])
        let prompt = GemmaPromptParityReport.make(
            partition: partition, inferencePrefixTokens: [1, 2],
            tokenText: { "t\($0)" })
        let teacher = try GemmaTeacherForcedReport.make(
            expected: [10], predicted: [10], assistantStart: 2,
            tokenText: { "t\($0)" })
        let balanced = try GemmaCachePredictionReport.make(
            mode: .balanced, expected: [10], fullSequencePredictions: [10],
            cachedPredictions: [99], assistantStart: 2,
            tokenText: { "t\($0)" })
        let unchunked = try GemmaCachePredictionReport.make(
            mode: .unchunked, expected: [10], fullSequencePredictions: [10],
            cachedPredictions: [10], assistantStart: 2,
            tokenText: { "t\($0)" })
        #expect(
            GemmaParityDiagnosticEngine.diagnose(
                promptParity: prompt, teacherForced: teacher,
                cacheComparisons: [balanced, unchunked]) == .balancedPrefillDivergence)

        let brokenUnchunked = try GemmaCachePredictionReport.make(
            mode: .unchunked, expected: [10], fullSequencePredictions: [10],
            cachedPredictions: [99], assistantStart: 2,
            tokenText: { "t\($0)" })
        #expect(
            GemmaParityDiagnosticEngine.diagnose(
                promptParity: prompt, teacherForced: teacher,
                cacheComparisons: [balanced, brokenUnchunked]) == .incrementalCacheDivergence)
    }

    @Test func repeatabilityEvidenceLocatesTheFirstCrossRunDivergence() throws {
        let repeatability = try GemmaPredictionRepeatabilityReport.make(
            context: .balancedCache,
            predictions: [
                [10, 11, 12, 13],
                [10, 99, 12, 13],
                [10, 11, 88, 13],
            ],
            assistantStart: 20,
            allLogitsFiniteByRepetition: [true, true, true],
            allCacheStatesFiniteByRepetition: [true, true, true],
            tokenText: { "t\($0)" })

        #expect(!repeatability.predictionsIdentical)
        #expect(repeatability.firstDivergence?.targetOffset == 1)
        #expect(repeatability.firstDivergence?.completeTokenIndex == 21)
        #expect(repeatability.firstDivergence?.referenceRepetition == 1)
        #expect(repeatability.firstDivergence?.differingRepetition == 2)
        #expect(Set(repeatability.predictionTokenSHA256s).count == 3)
    }

    @Test func diagnosisPrioritizesNonFiniteValuesAndCrossRunInstability() throws {
        let partition = try GemmaTrainingTokenPartition.derive(
            completeTokens: [1, 2, 10, 90],
            deployedPromptTokens: [1, 2])
        let prompt = GemmaPromptParityReport.make(
            partition: partition, inferencePrefixTokens: [1, 2],
            tokenText: { "t\($0)" })
        let teacher = try GemmaTeacherForcedReport.make(
            expected: [10], predicted: [10], assistantStart: 2,
            tokenText: { "t\($0)" })
        let healthy = try GemmaCachePredictionReport.make(
            mode: .balanced, expected: [10], fullSequencePredictions: [10],
            cachedPredictions: [10], assistantStart: 2,
            tokenText: { "t\($0)" })
        let nonFinite = try GemmaCachePredictionReport.make(
            mode: .unchunked, expected: [10], fullSequencePredictions: [10],
            cachedPredictions: [10], assistantStart: 2,
            allLogitsFinite: false, firstNonFiniteLogitTargetOffset: 0,
            tokenText: { "t\($0)" })
        #expect(
            GemmaParityDiagnosticEngine.diagnose(
                promptParity: prompt, teacherForced: teacher,
                cacheComparisons: [healthy, nonFinite]) == .nonFiniteNumerics)

        let unstable = try GemmaPredictionRepeatabilityReport.make(
            context: .balancedCache,
            predictions: [[10], [99]],
            assistantStart: 2,
            allLogitsFiniteByRepetition: [true, true],
            allCacheStatesFiniteByRepetition: [true, true],
            tokenText: { "t\($0)" })
        #expect(
            GemmaParityDiagnosticEngine.diagnose(
                promptParity: prompt, teacherForced: teacher,
                cacheComparisons: [healthy],
                predictionRepeatability: [unstable]) == .cachedPredictionNondeterminism)
    }
}
