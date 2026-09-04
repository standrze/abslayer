import Foundation
@testable import ProbeCore
import Testing

@Suite("Split-safe PromptFile subtraction")
struct PromptPairSubtractionTests {
    @Test("subtracts dev names while retaining train order and exact content")
    func subtractsByNameDeterministically() throws {
        let training = [
            pair("case-a", split: "train", marker: "A"),
            pair("case-b", split: "train", marker: "B"),
            pair("case-c", split: "train", marker: "C"),
            pair("case-d", split: "train", marker: "D"),
        ]
        let dev = [
            pair("case-c", split: "dev", marker: "ignored-C"),
            pair("case-a", split: "dev", marker: "ignored-A"),
        ]

        let first = try subtract(training: training, dev: dev)
        let second = try subtract(training: training, dev: dev)

        #expect(first == second)
        #expect(first.pairs == [training[1], training[3]])
        #expect(first.manifest.schemaVersion == 1)
        #expect(first.manifest.artifactRole
            == "split-safe-training-prompt-subtraction")
        #expect(first.manifest.operation == "subtract_names")
        #expect(first.manifest.inputTrainingPairs == 4)
        #expect(first.manifest.inputExclusionPairs == 2)
        #expect(first.manifest.removedPairs == 2)
        #expect(first.manifest.outputPairs == 2)
        #expect(first.manifest.removedNames == ["case-a", "case-c"])
        #expect(first.manifest.preservesTrainingOrder)
        #expect(first.manifest.preservesRetainedPairContent)
        #expect(!first.manifest.frozenAuditAccessPermitted)
        #expect(first.manifest.trainingSourcePath == "/tmp/train.json")
        #expect(first.manifest.exclusionSourcePath == "/tmp/dev.json")
        #expect(!first.manifest.trainingNamesSHA256.isEmpty)
        #expect(!first.manifest.exclusionNamesSHA256.isEmpty)
        #expect(!first.manifest.outputNamesSHA256.isEmpty)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        #expect(try encoder.encode(first.manifest)
            == encoder.encode(second.manifest))
    }

    @Test("legacy PromptFiles without split metadata remain supported")
    func acceptsLegacySplitlessInputs() throws {
        let training = [
            pair("holdout-1", split: nil, marker: "one"),
            pair("holdout-2", split: nil, marker: "two"),
            pair("holdout-3", split: nil, marker: "three"),
        ]
        let dev = [pair("holdout-2", split: nil, marker: "dev")]
        let result = try subtract(training: training, dev: dev)
        #expect(result.pairs.map(\.name) == ["holdout-1", "holdout-3"])
    }

    @Test("empty and duplicate names fail closed in both inputs")
    func rejectsInvalidNames() {
        #expect(throws: PromptPairSubtractionError
            .duplicateOrEmptyTrainingName("  ")) {
                try subtract(
                    training: [pair("  ", split: "train")],
                    dev: [pair("case-a", split: "dev")])
        }
        #expect(throws: PromptPairSubtractionError
            .duplicateOrEmptyTrainingName("case-a")) {
                try subtract(
                    training: [
                        pair("case-a", split: "train"),
                        pair("case-a", split: "train"),
                    ],
                    dev: [pair("case-a", split: "dev")])
        }
        #expect(throws: PromptPairSubtractionError
            .duplicateOrEmptyExclusionName(" case-a ")) {
                try subtract(
                    training: [
                        pair("case-a", split: "train"),
                        pair("case-b", split: "train"),
                    ],
                    dev: [
                        pair("case-a", split: "dev"),
                        pair(" case-a ", split: "dev"),
                    ])
        }
        #expect(throws: PromptPairSubtractionError
            .duplicateOrEmptyExclusionName("")) {
                try subtract(
                    training: [pair("case-a", split: "train")],
                    dev: [pair("", split: "dev")])
        }
    }

    @Test("missing dev names and total subtraction fail closed")
    func rejectsIncompleteOrEmptyResults() {
        #expect(throws: PromptPairSubtractionError
            .exclusionNamesMissingFromTraining(["case-z"])) {
                try subtract(
                    training: [pair("case-a", split: "train")],
                    dev: [pair("case-z", split: "dev")])
        }
        #expect(throws: PromptPairSubtractionError.emptyOutput) {
            try subtract(
                training: [pair("case-a", split: "train")],
                dev: [pair("case-a", split: "dev")])
        }
    }

    @Test("recorded split roles must be train and dev")
    func rejectsWrongSplitRoles() {
        #expect(throws: PromptPairSubtractionError.invalidTrainingSplit(
            name: "case-a", split: "dev")) {
                try subtract(
                    training: [pair("case-a", split: "dev")],
                    dev: [pair("case-a", split: "dev")])
        }
        #expect(throws: PromptPairSubtractionError.invalidExclusionSplit(
            name: "case-a", split: "test")) {
                try subtract(
                    training: [pair("case-a", split: "train")],
                    dev: [pair("case-a", split: "test")])
        }
    }

    private func subtract(
        training: [PromptPair], dev: [PromptPair]
    ) throws -> PromptPairSubtractionResult {
        try PromptPairSubtractor.subtract(
            trainingPairs: training,
            excludingNamesIn: dev,
            trainingSourcePath: "/tmp/train.json",
            trainingSourceSHA256: "train-sha256",
            exclusionSourcePath: "/tmp/dev.json",
            exclusionSourceSHA256: "dev-sha256")
    }

    private func pair(
        _ name: String, split: String?, marker: String = "value"
    ) -> PromptPair {
        PromptPair(
            name: name,
            contrast: "contrast-\(marker)",
            control: "control-\(marker)",
            category: "cyber",
            source: "source-\(marker)",
            controlSource: "control-source-\(marker)",
            split: split,
            requestType: "request-\(marker)",
            controlReferenceResponse: "reference-\(marker)")
    }
}
