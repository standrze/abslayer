import Foundation
import Testing
@testable import ProbeCore

@Suite("ABSlayer preflight")
struct ABSlayerPreflightTests {
    @Test("parses the exact preflight CLI and rejects expansions")
    func parsesCLI() throws {
        #expect(try ABSlayerPreflightInvocation.parse(arguments: [
            "/models/gemma",
            "--measurement", "/run/measurement.jsonl",
            "--evaluation", "/run/evaluation.jsonl",
            "--max-sequence-length", "128",
            "--report", "/run/preflight.json",
        ]) == ABSlayerPreflightInvocation(
            model: "/models/gemma",
            measurementPath: "/run/measurement.jsonl",
            evaluationPath: "/run/evaluation.jsonl",
            maximumSequenceLength: 128,
            reportPath: "/run/preflight.json"))
        #expect(try ABSlayerPreflightInvocation.parse(arguments: [
            "model", "--measurement", "m", "--evaluation", "e",
            "--max-sequence-length", "1", "--report", "r",
        ]).maximumSequenceLength == 1)
        #expect(try ABSlayerPreflightInvocation.parse(arguments: [
            "model", "--measurement", "m", "--evaluation", "e",
            "--max-sequence-length", "32768", "--report", "r",
        ]).maximumSequenceLength == 32_768)

        let invalid: [[String]] = [
            ["--measurement", "m", "--evaluation", "e",
             "--max-sequence-length", "128", "--report", "r"],
            ["model", "extra", "--measurement", "m", "--evaluation", "e",
             "--max-sequence-length", "128", "--report", "r"],
            ["model", "--measurement", "m", "--measurement", "m2",
             "--evaluation", "e", "--max-sequence-length", "128", "--report", "r"],
            ["model", "--measurement", "m", "--evaluation", "e",
             "--max-sequence-length", "0", "--report", "r"],
            ["model", "--measurement", "m", "--evaluation", "e",
             "--max-sequence-length", "32769", "--report", "r"],
            ["model", "--measurement", "m", "--evaluation", "e",
             "--max-sequence-length", "nan", "--report", "r"],
            ["model", "--measurement", "m", "--evaluation", "e",
             "--max-sequence-length", "128", "--report", "r", "--unknown", "x"],
        ]
        for arguments in invalid {
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerPreflightInvocation.parse(arguments: arguments)
            }
        }
    }

    @Test("validates both measurement channels and exact utility continuation limits")
    func validatesTokenBudgets() throws {
        let measurement = [ABSlayerMeasurementPair(
            name: "pair-000001", contrast: "measurement-long", control: "measurement-short")]
        let evaluation = [
            ABSlayerEvaluationCase(
                name: "case-000001", kind: .refusal,
                prompt: "evaluation-long", reference: nil),
            ABSlayerEvaluationCase(
                name: "case-000002", kind: .utility,
                prompt: "utility", reference: "reference"),
        ]
        let tokenizer = FakePreflightTokenizer(
            promptCounts: [
                "measurement-long": 128,
                "measurement-short": 7,
                "evaluation-long": 512,
                "utility": 510,
            ],
            continuationCounts: ["reference": 128])
        let result = try ABSlayerPreflight.validateTokenBudgets(
            measurement: measurement, evaluation: evaluation,
            maximumSequenceLength: 128, tokenizer: tokenizer)
        #expect(result.measurementMaximumPromptTokens == 128)
        #expect(result.evaluationMaximumPromptTokens == 512)
        #expect(result.utilityMaximumContinuationTokens == 128)
        #expect(result.utilityMaximumTotalTokens == 638)
        #expect(result.refusalCount == 1)
        #expect(result.utilityCount == 1)

        let fallback = FakePreflightTokenizer(
            promptCounts: tokenizer.promptCounts,
            continuationCounts: tokenizer.continuationCounts,
            fallbackReferences: ["reference"])
        #expect(try ABSlayerPreflight.validateTokenBudgets(
            measurement: measurement, evaluation: evaluation,
            maximumSequenceLength: 128, tokenizer: fallback) == result)
    }

    @Test("fails closed on every token boundary violation")
    func rejectsTokenBudgetViolations() throws {
        let baseMeasurement = [ABSlayerMeasurementPair(
            name: "pair", contrast: "contrast", control: "control")]
        let baseEvaluation = [
            ABSlayerEvaluationCase(
                name: "refusal", kind: .refusal, prompt: "refusal", reference: nil),
            ABSlayerEvaluationCase(
                name: "utility", kind: .utility,
                prompt: "utility", reference: "reference"),
        ]

        func rejects(
            _ tokenizer: FakePreflightTokenizer,
            measurement: [ABSlayerMeasurementPair] = baseMeasurement,
            maximumSequenceLength: Int = 128
        ) {
            #expect(throws: ABSlayerPreflightError.self) {
                try ABSlayerPreflight.validateTokenBudgets(
                    measurement: measurement, evaluation: baseEvaluation,
                    maximumSequenceLength: maximumSequenceLength,
                    tokenizer: tokenizer)
            }
        }

        rejects(FakePreflightTokenizer(
            promptCounts: [
                "contrast": 129, "control": 1, "refusal": 1, "utility": 1,
            ], continuationCounts: ["reference": 1]))
        rejects(FakePreflightTokenizer(
            promptCounts: [
                "contrast": 1, "control": 1, "refusal": 513, "utility": 1,
            ], continuationCounts: ["reference": 1]))
        rejects(FakePreflightTokenizer(
            promptCounts: [
                "contrast": 1, "control": 1, "refusal": 1, "utility": 1,
            ], continuationCounts: ["reference": 0]))
        rejects(FakePreflightTokenizer(
            promptCounts: [
                "contrast": 1, "control": 1, "refusal": 1, "utility": 1,
            ], continuationCounts: ["reference": 129]))
        rejects(FakePreflightTokenizer(
            promptCounts: [
                "contrast": 1, "control": 0, "refusal": 1, "utility": 1,
            ], continuationCounts: ["reference": 1]))
        rejects(FakePreflightTokenizer(
            promptCounts: [
                "contrast": 1, "control": 1, "refusal": 1, "utility": 1,
            ], continuationCounts: ["reference": 1],
            incompatibleReferences: ["reference"]))
    }

    @Test("strict JSONL rejects unknown, non-string, nonfinite, duplicate, and symlink input")
    func rejectsUnsafeJSONL() throws {
        try withTemporaryDirectory { root in
            let input = root.appendingPathComponent("input.jsonl")
            let invalidRows = [
                "{\"contrast\":\"a\",\"control\":\"b\",\"unknown\":\"x\"}\n",
                "{\"contrast\":\"a\",\"control\":1}\n",
                "{\"contrast\":\"a\",\"control\":NaN}\n",
                "{\"contrast\":\"a\",\"contrast\":\"b\",\"control\":\"c\"}\n",
                "{\"contrast\":\"a\",\"control\":\"b\"}\n"
                    + "{\"control\":\"b\",\"contrast\":\"a\"}\n",
            ]
            for row in invalidRows {
                try Data(row.utf8).write(to: input, options: .atomic)
                #expect(throws: ABSlayerBackendContractError.self) {
                    try ABSlayerBackendJSONL.loadMeasurementPairs(path: input.path)
                }
            }

            let target = root.appendingPathComponent("target.jsonl")
            let link = root.appendingPathComponent("link.jsonl")
            try Data("{\"contrast\":\"a\",\"control\":\"b\"}\n".utf8)
                .write(to: target)
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.loadMeasurementPairs(path: link.path)
            }
        }
    }

    @Test("production model inspector binds tiny Gemma4 identity and rejects unsupported or symlinked models")
    func modelInspection() throws {
        try withTemporaryDirectory { root in
            let model = root.appendingPathComponent("model", isDirectory: true)
            try makeTinyModel(model, modelType: "gemma4", textModelType: "gemma4_text")
            let binding = try ABSlayerProductionModelInspector.inspect(
                identifier: model.path)
            #expect(binding.canonicalPath == model.standardizedFileURL.path)
            #expect(binding.decoderLayerCount == 35)
            #expect(binding.hiddenSize == 1_536)
            #expect(binding.metadataSHA256.count == 64)
            #expect(binding.weightsSHA256.count == 64)
            try ABSlayerProductionModelInspector.validateUnchanged(binding)
            try Data("changed\n".utf8).write(
                to: model.appendingPathComponent("tokenizer_config.json"),
                options: .atomic)
            #expect(throws: ABSlayerBackendRuntimeError.self) {
                try ABSlayerProductionModelInspector.validateUnchanged(binding)
            }

            let unsupported = root.appendingPathComponent(
                "unsupported", isDirectory: true)
            try makeTinyModel(
                unsupported, modelType: "gemma3", textModelType: "gemma3_text")
            #expect(throws: ABSlayerBackendRuntimeError.self) {
                try ABSlayerProductionModelInspector.inspect(
                    identifier: unsupported.path)
            }

            let link = root.appendingPathComponent("model-link")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: model)
            #expect(throws: ModelFolderValidationError.self) {
                try ABSlayerProductionModelInspector.inspect(identifier: link.path)
            }
        }
    }

    @Test("report destination cannot invalidate its model binding")
    func reportOutsideModel() throws {
        let model = ABSlayerModelBinding(
            identifier: "/models/gemma", canonicalPath: "/models/gemma",
            revision: nil, metadataSHA256: String(repeating: "a", count: 64),
            weightsSHA256: String(repeating: "b", count: 64),
            decoderLayerCount: 35, hiddenSize: 1_536)
        #expect(throws: ABSlayerPreflightError.self) {
            try ABSlayerPreflight.validateReportDestination(
                "/models/gemma/preflight.json", model: model)
        }
        #expect(try ABSlayerPreflight.validateReportDestination(
            "/models/gemma-other/preflight.json", model: model)
            == "/models/gemma-other/preflight.json")

        try withTemporaryDirectory { root in
            let realModel = root.appendingPathComponent(
                "real-model", isDirectory: true)
            let reports = realModel.appendingPathComponent(
                "reports", isDirectory: true)
            try FileManager.default.createDirectory(
                at: reports, withIntermediateDirectories: true)
            let alias = root.appendingPathComponent("model-alias")
            try FileManager.default.createSymbolicLink(
                at: alias, withDestinationURL: realModel)
            let boundModel = ABSlayerModelBinding(
                identifier: realModel.path,
                canonicalPath: realModel.path,
                revision: nil,
                metadataSHA256: String(repeating: "a", count: 64),
                weightsSHA256: String(repeating: "b", count: 64),
                decoderLayerCount: 35,
                hiddenSize: 1_536)
            #expect(throws: ABSlayerPreflightError.self) {
                try ABSlayerPreflight.validateReportDestination(
                    alias.appendingPathComponent(
                        "reports/preflight.json").path,
                    model: boundModel)
            }
            #expect(throws: ABSlayerPreflightError.self) {
                try ABSlayerPreflight.validateReportDestination(
                    alias.appendingPathComponent(
                        "missing/reports/preflight.json").path,
                    model: boundModel)
            }
        }
    }

    @Test("report publication is atomic, regular, and never overwrites")
    func atomicNoOverwriteReport() throws {
        try withTemporaryDirectory { root in
            let report = root.appendingPathComponent("nested/preflight.json")
            let value = ABSlayerPreflightLimits(
                measurementMaximumSequenceLength: 128,
                evaluationMaximumPromptTokens: 512,
                utilityMaximumContinuationTokens: 128,
                utilityMaximumTotalTokens: 640)
            try ABSlayerAtomicReportWriter.write(value, to: report.path)
            #expect(ABSlayerFileSystem.isRegularFileWithoutFollowingSymlink(report.path))
            let first = try Data(contentsOf: report)
            #expect(first.last == 0x0a)
            #expect(try JSONDecoder().decode(
                ABSlayerPreflightLimits.self, from: first) == value)

            #expect(throws: ABSlayerPreflightError.self) {
                try ABSlayerAtomicReportWriter.write(value, to: report.path)
            }
            #expect(try Data(contentsOf: report) == first)
            let entries = try FileManager.default.contentsOfDirectory(
                atPath: report.deletingLastPathComponent().path)
            #expect(!entries.contains { $0.contains(".staging-") })

            let symlinkReport = root.appendingPathComponent("report-link.json")
            try FileManager.default.createSymbolicLink(
                at: symlinkReport, withDestinationURL: report)
            #expect(throws: ABSlayerPreflightError.self) {
                try ABSlayerAtomicReportWriter.write(value, to: symlinkReport.path)
            }

            let excluded = root.appendingPathComponent(
                "excluded", isDirectory: true)
            try FileManager.default.createDirectory(
                at: excluded, withIntermediateDirectories: false)
            let excludedAlias = root.appendingPathComponent("excluded-alias")
            try FileManager.default.createSymbolicLink(
                at: excludedAlias, withDestinationURL: excluded)
            #expect(throws: ABSlayerPreflightError.self) {
                try ABSlayerAtomicReportWriter.write(
                    value,
                    to: excludedAlias.appendingPathComponent(
                        "preflight.json").path,
                    excludingDirectory: excluded.path)
            }
            #expect(!FileManager.default.fileExists(
                atPath: excluded.appendingPathComponent(
                    "preflight.json").path))
        }
    }

    @Test("concurrent report publishers have exactly one winner")
    func concurrentNoOverwriteReport() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "abslayer-preflight-race-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let report = root.appendingPathComponent("preflight.json")
        let value = ABSlayerPreflightLimits(
            measurementMaximumSequenceLength: 128,
            evaluationMaximumPromptTokens: 512,
            utilityMaximumContinuationTokens: 128,
            utilityMaximumTotalTokens: 640)
        let successes = await withTaskGroup(of: Bool.self) { group in
            for _ in 0 ..< 8 {
                group.addTask {
                    do {
                        try ABSlayerAtomicReportWriter.write(value, to: report.path)
                        return true
                    } catch {
                        return false
                    }
                }
            }
            var values = [Bool]()
            for await result in group { values.append(result) }
            return values.count(where: { $0 })
        }
        #expect(successes == 1)
        #expect(try JSONDecoder().decode(
            ABSlayerPreflightLimits.self, from: Data(contentsOf: report)) == value)
    }

    @Test("failed post-publication verification retracts only the owned inode")
    func failedPublicationVerificationCleanup() throws {
        try withTemporaryDirectory { root in
            let staged = root.appendingPathComponent("staged.json")
            let published = root.appendingPathComponent("published.json")
            let expected = Data("{\"status\":\"ok\"}\n".utf8)
            try expected.write(to: staged)
            guard let stagedIdentity = ABSlayerFileSystem.regularFileIdentity(
                staged.path)
            else {
                Issue.record("staged test report has no regular-file identity")
                return
            }
            try FileManager.default.linkItem(at: staged, to: published)
            #expect(throws: ABSlayerPreflightError.self) {
                try ABSlayerAtomicReportWriter.verifyPublishedReport(
                    destination: published,
                    temporary: staged,
                    stagedBefore: stagedIdentity,
                    expectedData: Data("different\n".utf8),
                    excludingDirectory: nil)
            }
            #expect(!FileManager.default.fileExists(atPath: published.path))
            #expect(try Data(contentsOf: staged) == expected)

            try Data("unrelated\n".utf8).write(to: published)
            #expect(throws: ABSlayerPreflightError.self) {
                try ABSlayerAtomicReportWriter.verifyPublishedReport(
                    destination: published,
                    temporary: staged,
                    stagedBefore: stagedIdentity,
                    expectedData: expected,
                    excludingDirectory: nil)
            }
            #expect(try Data(contentsOf: published)
                == Data("unrelated\n".utf8))
        }
    }

    @Test("backend runtime and probe paths share preflight limits and empty-token guards")
    func productionReuseContract() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let runtime = try String(
            contentsOf: packageRoot.appendingPathComponent(
                "Sources/ProbeCore/ABSlayerBackendRuntime.swift"),
            encoding: .utf8)
        #expect(runtime.contains("ABSlayerProductionModelInspector.inspect"))
        #expect(runtime.contains("ABSlayerReferenceTokenization.continuation"))
        #expect(runtime.contains("ABSlayerPreflight.evaluationMaximumPromptTokens"))
        #expect(runtime.contains("ABSlayerPreflight.utilityMaximumContinuationTokens"))
        #expect(runtime.contains("ABSlayerPreflight.utilityMaximumTotalTokens"))

        for name in ["Gemma3Probe.swift", "Gemma4Probe.swift"] {
            let probe = try String(
                contentsOf: packageRoot.appendingPathComponent(
                    "Sources/ProbeCore/\(name)"), encoding: .utf8)
            #expect(probe.contains("emptyPromptTokenization"))
        }
    }

    private func withTemporaryDirectory(
        _ body: (URL) throws -> Void
    ) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "abslayer-preflight-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }

    private func makeTinyModel(
        _ directory: URL, modelType: String, textModelType: String
    ) throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: false)
        let config: [String: Any] = [
            "model_type": modelType,
            "text_config": [
                "model_type": textModelType,
                "num_hidden_layers": 35,
                "hidden_size": 1_536,
            ],
            "torch_dtype": "bfloat16",
        ]
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json"))
        try Data("{}\n".utf8).write(
            to: directory.appendingPathComponent("tokenizer_config.json"))

        let tensorName = "model.layers.0.weight"
        let header = try JSONSerialization.data(withJSONObject: [
            tensorName: [
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            ],
        ])
        var headerLength = UInt64(header.count).littleEndian
        var safetensors = withUnsafeBytes(of: &headerLength) { Data($0) }
        safetensors.append(header)
        safetensors.append(contentsOf: [0, 0])
        try safetensors.write(
            to: directory.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: [
            "weight_map": [tensorName: "model.safetensors"],
        ]).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
    }
}

private struct FakePreflightTokenizer: ABSlayerPromptTokenizing {
    let promptCounts: [String: Int]
    let continuationCounts: [String: Int]
    var fallbackReferences: Set<String> = []
    var incompatibleReferences: Set<String> = []

    func userPromptTokens(_ prompt: String) throws -> [Int] {
        makeTokens(count: promptCounts[prompt] ?? 1, start: 1)
    }

    func promptPlusReferenceTokens(
        promptTokens: [Int], reference: String
    ) -> [Int] {
        if fallbackReferences.contains(reference)
            || incompatibleReferences.contains(reference)
        {
            return [Int.max]
        }
        return promptTokens + makeTokens(
            count: continuationCounts[reference] ?? 1,
            start: promptTokens.count + 1)
    }

    func completedConversationTokens(
        prompt: String, reference: String
    ) throws -> [Int] {
        guard !incompatibleReferences.contains(reference) else { return [Int.max] }
        let promptTokens = try userPromptTokens(prompt)
        return promptTokens + makeTokens(
            count: continuationCounts[reference] ?? 1,
            start: promptTokens.count + 1)
    }

    private func makeTokens(count: Int, start: Int) -> [Int] {
        guard count > 0 else { return [] }
        return Array(start ..< start + count)
    }
}
