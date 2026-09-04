import Foundation
import MLX
import Testing
@testable import ProbeCore

@Suite("ABSlayer harness backend contract")
struct ABSlayerBackendContractTests {
    @Test("parses every exact controller argv shape")
    func parsesControllerCommands() throws {
        #expect(try ABSlayerBackendInvocation.parse(arguments: [
            "--json", "doctor", "--model", "/models/gemma", "--role", "measure",
            "--revision", "rev-a",
        ]) == .doctor(ABSlayerDoctorInvocation(
            model: "/models/gemma", role: .measure, revision: "rev-a")))

        #expect(try ABSlayerBackendInvocation.parse(arguments: [
            "--json", "measure", "/models/gemma",
            "--pairs", "/run/inputs/measurement.jsonl",
            "--artifact", "/run/artifacts/directions",
            "--strength", "1", "--rank", "4",
            "--max-layer-fraction", "0.5",
            "--max-sequence-length", "1024",
            "--gpu-memory-utilization", "0.8",
            "--revision", "rev-a", "--prompt-count", "20",
        ]) == .measure(ABSlayerMeasureInvocation(
            model: "/models/gemma",
            pairsPath: "/run/inputs/measurement.jsonl",
            artifactPath: "/run/artifacts/directions",
            strength: 1,
            rank: 4,
            maximumLayerFraction: 0.5,
            maximumSequenceLength: 1024,
            gpuMemoryUtilization: 0.8,
            revision: "rev-a",
            promptCount: 20,
            temporaryDirectory: nil)))

        #expect(try ABSlayerBackendInvocation.parse(arguments: [
            "--json", "apply", "/models/gemma",
            "--artifact", "/run/artifacts/directions",
            "--output", "/run/candidates/one",
            "--strength", "0.5", "--revision", "rev-a",
        ]) == .apply(ABSlayerApplyInvocation(
            model: "/models/gemma",
            artifactPath: "/run/artifacts/directions",
            outputPath: "/run/candidates/one",
            strength: 0.5,
            revision: "rev-a")))

        #expect(try ABSlayerBackendInvocation.parse(arguments: [
            "--json", "verify", "/models/gemma", "/run/candidates/one",
            "--cases", "/run/inputs/evaluation.jsonl",
            "--report", "/run/reports/one.json",
            "--gpu-memory-utilization", "0.8",
            "--source-revision", "rev-a",
        ]) == .verify(ABSlayerVerifyInvocation(
            sourceModel: "/models/gemma",
            candidateModel: "/run/candidates/one",
            casesPath: "/run/inputs/evaluation.jsonl",
            reportPath: "/run/reports/one.json",
            gpuMemoryUtilization: 0.8,
            sourceRevision: "rev-a")))
    }

    @Test("rejects expanded, duplicated, nonfinite, and out-of-range argv")
    func rejectsInvalidArguments() {
        let invalid: [[String]] = [
            ["doctor", "--model", "m", "--role", "measure"],
            ["--json", "erase"],
            ["--json", "doctor", "--model", "m", "--role", "measure", "--role", "apply"],
            ["--json", "doctor", "--model", "m", "--role", "measure", "--shell", "x"],
            ["--json", "measure", "m", "--pairs", "p", "--artifact", "a",
             "--strength", "nan", "--rank", "4", "--max-layer-fraction", "0.5",
             "--max-sequence-length", "100", "--gpu-memory-utilization", "0.8"],
            ["--json", "measure", "m", "--pairs", "p", "--artifact", "a",
             "--strength", "1", "--rank", "0", "--max-layer-fraction", "0.5",
             "--max-sequence-length", "100", "--gpu-memory-utilization", "0.8"],
            ["--json", "measure", "m", "--pairs", "p", "--artifact", "a",
             "--strength", "1", "--rank", "4", "--max-layer-fraction", "0.001",
             "--max-sequence-length", "100", "--gpu-memory-utilization", "0.8"],
            ["--json", "apply", "m", "extra", "--artifact", "a", "--output", "o",
             "--strength", "1"],
            ["--json", "verify", "source", "candidate", "--cases", "c", "--report", "r",
             "--gpu-memory-utilization", "inf"],
            ["--json", "measure", "m", "--pairs", "p", "--artifact", "a",
             "--strength", "1", "--rank", "4", "--max-layer-fraction", "0.5",
             "--max-sequence-length", "100", "--gpu-memory-utilization", "0.8",
             "--temp-dir", "/run/tmp"],
        ]
        for arguments in invalid {
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendInvocation.parse(arguments: arguments)
            }
        }
    }

    @Test("measurement JSONL is closed, duplicate-safe, and preserves exact strings")
    func measurementJSONL() throws {
        try withTemporaryDirectory { root in
            let path = root.appendingPathComponent("pairs.jsonl")
            try Data(
                "{\"contrast\":\"Decline \\u2603\",\"control\":\"Answer \\nnow\"}\n"
                    .utf8).write(to: path)
            let rows = try ABSlayerBackendJSONL.loadMeasurementPairs(path: path.path)
            #expect(rows == [ABSlayerMeasurementPair(
                name: "pair-000001", contrast: "Decline ☃", control: "Answer \nnow")])
            let bound = try ABSlayerBackendJSONL.loadMeasurementPairsBound(
                path: path.path)
            #expect(bound.records == rows)
            try ABSlayerBackendJSONL.validateUnchanged(
                path: path.path, sha256: bound.sha256)
            try Data(
                "{\"contrast\":\"changed\",\"control\":\"bytes\"}\n".utf8
            ).write(to: path)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.validateUnchanged(
                    path: path.path, sha256: bound.sha256)
            }

            try Data(
                "{\"contrast\":\"a\",\"contrast\":\"b\",\"control\":\"c\"}\n"
                    .utf8).write(to: path)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.loadMeasurementPairs(path: path.path)
            }

            try Data(
                "{\"contrast\":\"a\",\"\\u0063ontrast\":\"b\",\"control\":\"c\"}\n"
                    .utf8).write(to: path)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.loadMeasurementPairs(path: path.path)
            }

            try Data(
                "{\"contrast\":\"a\",\"control\":\"b\"}\n{\"control\":\"b\",\"contrast\":\"a\"}\n"
                    .utf8).write(to: path)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.loadMeasurementPairs(path: path.path)
            }
        }
    }

    @Test("evaluation JSONL requires refusal and utility strata with exact fields")
    func evaluationJSONL() throws {
        try withTemporaryDirectory { root in
            let path = root.appendingPathComponent("cases.jsonl")
            let refusal = (0 ..< 10).map {
                "{\"kind\":\"refusal\",\"prompt\":\"refusal \($0)\"}"
            }
            let utility = (0 ..< 10).map {
                "{\"kind\":\"utility\",\"prompt\":\"utility \($0)\",\"reference\":\"answer \($0)\"}"
            }
            try Data((refusal + utility).joined(separator: "\n").appending("\n").utf8)
                .write(to: path)
            let rows = try ABSlayerBackendJSONL.loadEvaluationCases(path: path.path)
            #expect(rows.count == 20)
            #expect(rows[0].kind == .refusal)
            #expect(rows[0].reference == nil)
            #expect(rows[10].kind == .utility)
            #expect(rows[10].reference == "answer 0")

            try Data((refusal + [utility[0]]).joined(separator: "\n")
                .appending("\n").utf8).write(to: path)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.loadEvaluationCases(path: path.path)
            }

            try Data("{\"kind\":\"refusal\",\"prompt\":\"only\"}\n".utf8)
                .write(to: path)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.loadEvaluationCases(path: path.path)
            }

            try Data(
                "{\"kind\":\"refusal\",\"prompt\":\"one\",\"reference\":\"forbidden\"}\n"
                    .appending("{\"kind\":\"utility\",\"prompt\":\"two\",\"reference\":\"answer\"}\n")
                    .utf8).write(to: path)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.loadEvaluationCases(path: path.path)
            }
        }
    }

    @Test("JSONL loader rejects blank records and symlink inputs")
    func rejectsBlankAndSymlinkJSONL() throws {
        try withTemporaryDirectory { root in
            let source = root.appendingPathComponent("source.jsonl")
            let link = root.appendingPathComponent("link.jsonl")
            try Data("\n".utf8).write(to: source)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.loadMeasurementPairs(path: source.path)
            }
            try FileManager.default.removeItem(at: source)
            try Data("{\"contrast\":\"a\",\"control\":\"b\"}\n".utf8).write(to: source)
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
            #expect(throws: ABSlayerBackendContractError.self) {
                try ABSlayerBackendJSONL.loadMeasurementPairs(path: link.path)
            }
        }
    }

    @Test("layer count never exceeds the declared fraction and selection is stable")
    func layerSelection() throws {
        #expect(try ABSlayerBackendLayerSelection.maximumCount(
            layerCount: 35, fraction: 0.5) == 17)
        #expect(try ABSlayerBackendLayerSelection.maximumCount(
            layerCount: 35, fraction: 0.01) == 1)
        let metrics = (0 ..< 8).map { layer in
            ABSlayerLayerMetric(
                zeroBasedLayer: layer,
                cosineDistance: Double(layer + 1),
                directionAgreement: 1,
                medianDirectionAgreement: 1,
                silhouette: 1)
        }
        let first = try ABSlayerBackendLayerSelection.select(
            metrics: metrics, layerCount: 8, fraction: 0.5)
        let second = try ABSlayerBackendLayerSelection.select(
            metrics: metrics, layerCount: 8, fraction: 0.5)
        #expect(first == second)
        #expect(first.count == 4)
        #expect(first == first.sorted())
    }

    @Test("balanced-v1 requires behavior improvement, bounded utility loss, and nonempty output")
    func balancedVerification() throws {
        let binding = ABSlayerDatasetBinding(
            path: "/run/evaluation.jsonl", sha256: String(repeating: "a", count: 64),
            recordCount: 20, promptCount: nil)
        let passing = try ABSlayerBalancedVerification.makeReport(
            ABSlayerVerificationInput(
                sourceRefusals: Array(repeating: true, count: 10),
                candidateRefusals: Array(repeating: false, count: 10),
                candidateNonemptyGenerations: Array(repeating: true, count: 20),
                sourceUtilityNLL: Array(repeating: 1, count: 10),
                candidateUtilityNLL: Array(repeating: 1.2, count: 10),
                sourceModel: "/models/base", candidateModel: "/run/candidate",
                sourceRevision: "rev", cases: binding))
        #expect(passing.policy == "balanced-v1")
        #expect(passing.passed)
        #expect(passing.refusalPassed)
        #expect(passing.utilityPassed)
        #expect(passing.nonemptyGenerationPassed)
        #expect(!passing.certifiesSemanticCompliance)
        #expect(passing.refusalMetricBasis == "explicit-marker-proxy-v1")

        let noImprovement = try ABSlayerBalancedVerification.makeReport(
            ABSlayerVerificationInput(
                sourceRefusals: Array(repeating: false, count: 10),
                candidateRefusals: Array(repeating: false, count: 10),
                candidateNonemptyGenerations: Array(repeating: true, count: 20),
                sourceUtilityNLL: Array(repeating: 1, count: 10),
                candidateUtilityNLL: Array(repeating: 1.2, count: 10),
                sourceModel: "base", candidateModel: "candidate",
                sourceRevision: nil, cases: binding))
        #expect(!noImprovement.refusalPassed)
        #expect(!noImprovement.passed)

        let utilityFailure = try ABSlayerBalancedVerification.makeReport(
            ABSlayerVerificationInput(
                sourceRefusals: Array(repeating: true, count: 10),
                candidateRefusals: Array(repeating: false, count: 10),
                candidateNonemptyGenerations: Array(repeating: true, count: 20),
                sourceUtilityNLL: Array(repeating: 1, count: 10),
                candidateUtilityNLL: Array(repeating: 2.01, count: 10),
                sourceModel: "base", candidateModel: "candidate",
                sourceRevision: nil, cases: binding))
        #expect(!utilityFailure.utilityPassed)
        #expect(!utilityFailure.passed)

        let emptyFailure = try ABSlayerBalancedVerification.makeReport(
            ABSlayerVerificationInput(
                sourceRefusals: Array(repeating: true, count: 10),
                candidateRefusals: Array(repeating: false, count: 10),
                candidateNonemptyGenerations:
                    Array(repeating: true, count: 19) + [false],
                sourceUtilityNLL: Array(repeating: 1, count: 10),
                candidateUtilityNLL: Array(repeating: 1, count: 10),
                sourceModel: "base", candidateModel: "candidate",
                sourceRevision: nil, cases: binding))
        #expect(!emptyFailure.nonemptyGenerationPassed)
        #expect(!emptyFailure.passed)

        #expect(throws: ABSlayerBackendContractError.self) {
            try ABSlayerBalancedVerification.makeReport(
                ABSlayerVerificationInput(
                    sourceRefusals: [true], candidateRefusals: [false],
                    candidateNonemptyGenerations: [true, true],
                    sourceUtilityNLL: [1], candidateUtilityNLL: [1],
                    sourceModel: "base", candidateModel: "candidate",
                    sourceRevision: nil, cases: binding))
        }
    }

    @Test("verification report JSON exposes exact harness gates without raw content")
    func reportJSONContract() throws {
        let report = try ABSlayerBalancedVerification.makeReport(
            ABSlayerVerificationInput(
                sourceRefusals: Array(repeating: true, count: 10),
                candidateRefusals: Array(repeating: false, count: 10),
                candidateNonemptyGenerations: Array(repeating: true, count: 20),
                sourceUtilityNLL: Array(repeating: 1, count: 10),
                candidateUtilityNLL: Array(repeating: 1, count: 10),
                sourceModel: "base", candidateModel: "candidate",
                sourceRevision: nil,
                cases: ABSlayerDatasetBinding(
                    path: "/private/input", sha256: String(repeating: "d", count: 64),
                    recordCount: 20,
                    promptCount: nil)))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(decoding: try encoder.encode(report), as: UTF8.self)
        #expect(json.contains("\"policy\":\"balanced-v1\""))
        #expect(json.contains("\"refusal_passed\":true"))
        #expect(json.contains("\"utility_passed\":true"))
        #expect(json.contains("\"nonempty_generation_passed\":true"))
        #expect(!json.contains("prompt"))
        #expect(!json.contains("response"))
        #expect(!json.contains("reference"))
    }

    @Test("disk preflight reserves two GiB and rejects insufficient capacity")
    func diskPreflight() throws {
        let checkpoint = Int64(10_000_000_000)
        let required = try ABSlayerDiskPreflight.requiredFreeBytes(
            checkpointBytes: checkpoint)
        #expect(required == checkpoint + Int64(2 * 1_024 * 1_024 * 1_024))
        #expect(throws: Never.self) {
            try ABSlayerDiskPreflight.validate(
                checkpointBytes: checkpoint, availableBytes: required)
        }
        #expect(throws: ABSlayerBackendContractError.self) {
            try ABSlayerDiskPreflight.validate(
                checkpointBytes: checkpoint, availableBytes: required - 1)
        }
    }

    @Test("direction artifact is atomic, hash-bound, layer-scoped, and no-overwrite")
    func directionArtifactRoundTrip() throws {
        try withTemporaryDirectory { root in
            let artifact = root.appendingPathComponent("directions", isDirectory: true)
            let model = ABSlayerModelBinding(
                identifier: "/models/gemma4", canonicalPath: "/models/gemma4",
                revision: "rev-a",
                metadataSHA256: String(repeating: "a", count: 64),
                weightsSHA256: String(repeating: "b", count: 64),
                decoderLayerCount: 3, hiddenSize: 4)
            let dataset = ABSlayerDatasetBinding(
                path: "/run/pairs.jsonl", sha256: String(repeating: "c", count: 64),
                recordCount: 5, promptCount: 5)
            let algorithm = ABSlayerDirectionAlgorithm(
                name: "paired-centroid-rank-k/v1", strength: 1, rank: 2,
                maximumLayerFraction: 0.67, selectedLayers: [0, 2],
                tokenPosition: "post-instruction")
            let runtime = ABSlayerDirectionRuntime(
                backend: "mlx-swift-cuda", maximumSequenceLength: 128,
                gpuMemoryUtilization: 0.8, temporaryDirectory: nil)
            let subspaces: [[[Float]]] = [
                [[1, 0, 0, 0], [0, 1, 0, 0]],
                [],
                [[0, 0, 1, 0]],
            ]
            _ = try ABSlayerDirectionArtifactStore.write(
                subspaces: subspaces, model: model, dataset: dataset,
                algorithm: algorithm, runtime: runtime, to: artifact.path)
            #expect(ABSlayerFileSystem.isDirectoryWithoutFollowingSymlink(artifact.path))
            #expect(ABSlayerFileSystem.isRegularFileWithoutFollowingSymlink(
                artifact.appendingPathComponent("manifest.json").path))
            #expect(ABSlayerFileSystem.isRegularFileWithoutFollowingSymlink(
                artifact.appendingPathComponent("directions.safetensors").path))

            let loaded = try ABSlayerDirectionArtifactStore.load(from: artifact.path)
            #expect(loaded.manifest == ABSlayerDirectionArtifactManifest(
                format: "abslayer.artifact/v1", model: model, dataset: dataset,
                algorithm: algorithm,
                tensors: loaded.manifest.tensors, runtime: runtime))
            #expect(loaded.subspaces == subspaces)
            #expect(loaded.directions[0] == [1, 0, 0, 0])
            #expect(loaded.directions[1].isEmpty)
            #expect(loaded.directionsSHA256.count == 64)
            #expect(loaded.manifestSHA256.count == 64)

            #expect(throws: ABSlayerDirectionArtifactError.self) {
                try ABSlayerDirectionArtifactStore.write(
                    subspaces: subspaces, model: model, dataset: dataset,
                    algorithm: algorithm, runtime: runtime, to: artifact.path)
            }
        }
    }

    @Test("direction artifact rejects manifest expansion and tensor tampering")
    func directionArtifactTamperDetection() throws {
        try withTemporaryDirectory { root in
            func makeArtifact(_ name: String) throws -> URL {
                let artifact = root.appendingPathComponent(name, isDirectory: true)
                _ = try ABSlayerDirectionArtifactStore.write(
                    subspaces: [[[1, 0]]],
                    model: ABSlayerModelBinding(
                        identifier: "m", canonicalPath: "/m", revision: nil,
                        metadataSHA256: String(repeating: "a", count: 64),
                        weightsSHA256: String(repeating: "b", count: 64),
                        decoderLayerCount: 1,
                        hiddenSize: 2),
                    dataset: ABSlayerDatasetBinding(
                        path: "p", sha256: String(repeating: "c", count: 64),
                        recordCount: 1,
                        promptCount: 1),
                    algorithm: ABSlayerDirectionAlgorithm(
                        name: "paired-centroid-rank-k/v1", strength: 1,
                        rank: 1, maximumLayerFraction: 1,
                        selectedLayers: [0], tokenPosition: "post-instruction"),
                    runtime: ABSlayerDirectionRuntime(
                        backend: "mlx-swift-cuda", maximumSequenceLength: 8,
                        gpuMemoryUtilization: 0.5, temporaryDirectory: nil),
                    to: artifact.path)
                return artifact
            }

            let expanded = try makeArtifact("expanded")
            let manifestURL = expanded.appendingPathComponent("manifest.json")
            var object = try #require(
                JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL))
                    as? [String: Any])
            object["unexpected"] = true
            try JSONSerialization.data(withJSONObject: object).write(
                to: manifestURL, options: .atomic)
            #expect(throws: ABSlayerDirectionArtifactError.self) {
                try ABSlayerDirectionArtifactStore.load(from: expanded.path)
            }

            let tampered = try makeArtifact("tampered")
            let tensorURL = tampered.appendingPathComponent("directions.safetensors")
            var bytes = try Data(contentsOf: tensorURL)
            bytes[bytes.count - 1] ^= 0x01
            try bytes.write(to: tensorURL, options: .atomic)
            #expect(throws: ABSlayerDirectionArtifactError.self) {
                try ABSlayerDirectionArtifactStore.load(from: tampered.path)
            }

            let duplicate = try makeArtifact("duplicate")
            let duplicateManifest = duplicate.appendingPathComponent("manifest.json")
            var raw = try String(contentsOf: duplicateManifest, encoding: .utf8)
            raw = raw.replacingOccurrences(
                of: "{", with: "{\"f\\u006frmat\":\"abslayer.artifact/v1\",",
                options: [], range: raw.range(of: "{"))
            try Data(raw.utf8).write(to: duplicateManifest, options: .atomic)
            #expect(throws: ABSlayerDirectionArtifactError.self) {
                try ABSlayerDirectionArtifactStore.load(from: duplicate.path)
            }
        }
    }

    @Test("direction artifact enforces portable bounds and orthonormal bases")
    func directionArtifactValidationBounds() throws {
        try withTemporaryDirectory { root in
            let model = ABSlayerModelBinding(
                identifier: "m", canonicalPath: "/m", revision: nil,
                metadataSHA256: String(repeating: "a", count: 64),
                weightsSHA256: String(repeating: "b", count: 64),
                decoderLayerCount: 1, hiddenSize: 2)
            let dataset = ABSlayerDatasetBinding(
                path: "p", sha256: String(repeating: "c", count: 64),
                recordCount: 5, promptCount: 5)
            let runtime = ABSlayerDirectionRuntime(
                backend: "mlx-swift-cuda", maximumSequenceLength: 128,
                gpuMemoryUtilization: 0.8, temporaryDirectory: nil)
            func algorithm(strength: Double = 1, rank: Int = 1)
                -> ABSlayerDirectionAlgorithm
            {
                ABSlayerDirectionAlgorithm(
                    name: "paired-centroid-rank-k/v1", strength: strength,
                    rank: rank, maximumLayerFraction: 1,
                    selectedLayers: [0], tokenPosition: "post-instruction")
            }
            func rejects(
                _ name: String, subspaces: [[[Float]]],
                artifactModel: ABSlayerModelBinding? = nil,
                artifactDataset: ABSlayerDatasetBinding? = nil,
                artifactAlgorithm: ABSlayerDirectionAlgorithm? = nil,
                artifactRuntime: ABSlayerDirectionRuntime? = nil
            ) {
                #expect(throws: ABSlayerDirectionArtifactError.self) {
                    try ABSlayerDirectionArtifactStore.write(
                        subspaces: subspaces,
                        model: artifactModel ?? model,
                        dataset: artifactDataset ?? dataset,
                        algorithm: artifactAlgorithm ?? algorithm(),
                        runtime: artifactRuntime ?? runtime,
                        to: root.appendingPathComponent(name).path)
                }
            }

            rejects("scaled", subspaces: [[[2, 0]]])
            rejects(
                "duplicate", subspaces: [[[1, 0], [1, 0]]],
                artifactAlgorithm: algorithm(rank: 2))
            rejects(
                "strength", subspaces: [[[1, 0]]],
                artifactAlgorithm: algorithm(strength: 2.01))
            rejects(
                "sequence", subspaces: [[[1, 0]]],
                artifactRuntime: ABSlayerDirectionRuntime(
                    backend: "mlx-swift-cuda", maximumSequenceLength: 32_769,
                    gpuMemoryUtilization: 0.8, temporaryDirectory: nil))
            rejects(
                "temporary", subspaces: [[[1, 0]]],
                artifactRuntime: ABSlayerDirectionRuntime(
                    backend: "mlx-swift-cuda", maximumSequenceLength: 128,
                    gpuMemoryUtilization: 0.8, temporaryDirectory: "/tmp"))
            rejects(
                "records", subspaces: [[[1, 0]]],
                artifactDataset: ABSlayerDatasetBinding(
                    path: "p", sha256: dataset.sha256,
                    recordCount: 10_001, promptCount: 1))
            rejects(
                "prompt-count", subspaces: [[[1, 0]]],
                artifactDataset: ABSlayerDatasetBinding(
                    path: "p", sha256: dataset.sha256,
                    recordCount: 5, promptCount: 6))
            rejects(
                "hash", subspaces: [[[1, 0]]],
                artifactModel: ABSlayerModelBinding(
                    identifier: "m", canonicalPath: "/m", revision: nil,
                    metadataSHA256: "not-a-hash",
                    weightsSHA256: model.weightsSHA256,
                    decoderLayerCount: 1, hiddenSize: 2))
        }
    }

    @Test("checkpoint provenance binds exact index, shards, and marker-free metadata")
    func checkpointProvenance() throws {
        try withTemporaryDirectory { root in
            let model = root.appendingPathComponent("model", isDirectory: true)
            try FileManager.default.createDirectory(
                at: model, withIntermediateDirectories: false)
            let shardA = model.appendingPathComponent("a.safetensors")
            let shardB = model.appendingPathComponent("b.safetensors")
            try Data([1, 2, 3]).write(to: shardA)
            try Data([4, 5]).write(to: shardB)
            try Data("{\"model_type\":\"gemma4\"}\n".utf8).write(
                to: model.appendingPathComponent("config.json"))
            try Data("{{ bos_token }}".utf8).write(
                to: model.appendingPathComponent("chat_template.jinja"))
            try Data("a b\nc d\n".utf8).write(
                to: model.appendingPathComponent("merges.txt"))
            let index = "{\"weight_map\":{\"z\":\"b.safetensors\",\"a\":\"a.safetensors\"}}\n"
            try Data(index.utf8).write(
                to: model.appendingPathComponent("model.safetensors.index.json"))

            let firstWeights = try ABSlayerCheckpointProvenance.weightsSHA256(
                directory: model.path)
            let firstMetadata = try ABSlayerCheckpointProvenance.metadataSHA256(
                directory: model.path)
            #expect(firstWeights.count == 64)
            #expect(firstMetadata.count == 64)
            #expect(try ABSlayerCheckpointProvenance.weightsSHA256(
                directory: model.path) == firstWeights)

            try Data("ignored by the matching copy/hash policy".utf8).write(
                to: model.appendingPathComponent(".gitattributes"))
            #expect(try ABSlayerCheckpointProvenance.metadataSHA256(
                directory: model.path) == firstMetadata)

            try Data("{\"owned\":true}\n".utf8).write(
                to: model.appendingPathComponent("abslayer.json"))
            #expect(try ABSlayerCheckpointProvenance.metadataSHA256(
                directory: model.path) == firstMetadata)

            try Data("{{ bos_token }} changed".utf8).write(
                to: model.appendingPathComponent("chat_template.jinja"), options: .atomic)
            #expect(try ABSlayerCheckpointProvenance.metadataSHA256(
                directory: model.path) != firstMetadata)
            try Data("{{ bos_token }}".utf8).write(
                to: model.appendingPathComponent("chat_template.jinja"), options: .atomic)
            #expect(try ABSlayerCheckpointProvenance.metadataSHA256(
                directory: model.path) == firstMetadata)

            try Data([1, 2, 4]).write(to: shardA, options: .atomic)
            #expect(try ABSlayerCheckpointProvenance.weightsSHA256(
                directory: model.path) != firstWeights)
            #expect(try ABSlayerCheckpointProvenance.metadataSHA256(
                directory: model.path) == firstMetadata)

            try Data("{\"model_type\":\"changed\"}\n".utf8).write(
                to: model.appendingPathComponent("config.json"), options: .atomic)
            #expect(try ABSlayerCheckpointProvenance.metadataSHA256(
                directory: model.path) != firstMetadata)

            try Data("{\"weight_map\":{\"a\":\"../escape.safetensors\"}}".utf8)
                .write(to: model.appendingPathComponent(
                    "model.safetensors.index.json"), options: .atomic)
            #expect(throws: ABSlayerCheckpointProvenanceError.self) {
                try ABSlayerCheckpointProvenance.weightsSHA256(directory: model.path)
            }

            let external = root.appendingPathComponent("external.txt")
            try Data("external".utf8).write(to: external)
            try FileManager.default.createSymbolicLink(
                at: model.appendingPathComponent("vocab.txt"),
                withDestinationURL: external)
            #expect(throws: ABSlayerCheckpointProvenanceError.self) {
                try ABSlayerCheckpointProvenance.metadataSHA256(directory: model.path)
            }
        }
    }

    @Test("strict JSON rejects nested and escape-equivalent duplicate keys")
    func strictJSONDuplicateKeys() {
        #expect(throws: ABSlayerStrictJSON.Error.self) {
            try ABSlayerStrictJSON.validateNoDuplicateKeys(
                Data("{\"outer\":{\"format\":1,\"f\\u006frmat\":2}}".utf8))
        }
    }

    @Test("candidate marker is closed and bound to source, artifact, and strength")
    func candidateMarkerBinding() throws {
        try withTemporaryDirectory { root in
            let source = root.appendingPathComponent("source", isDirectory: true)
            let candidate = root.appendingPathComponent("candidate", isDirectory: true)
            let artifact = root.appendingPathComponent("artifact", isDirectory: true)
            try FileManager.default.createDirectory(
                at: source, withIntermediateDirectories: false)
            try FileManager.default.createDirectory(
                at: candidate, withIntermediateDirectories: false)
            let model = ABSlayerModelBinding(
                identifier: source.path, canonicalPath: source.path,
                revision: "rev-a",
                metadataSHA256: String(repeating: "a", count: 64),
                weightsSHA256: String(repeating: "b", count: 64),
                decoderLayerCount: 1, hiddenSize: 2)
            _ = try ABSlayerDirectionArtifactStore.write(
                subspaces: [[[1, 0]]],
                model: model,
                dataset: ABSlayerDatasetBinding(
                    path: "/inputs/pairs.jsonl",
                    sha256: String(repeating: "c", count: 64),
                    recordCount: 5, promptCount: 5),
                algorithm: ABSlayerDirectionAlgorithm(
                    name: "paired-centroid-rank-k/v1", strength: 0.5,
                    rank: 1, maximumLayerFraction: 1,
                    selectedLayers: [0], tokenPosition: "post-instruction"),
                runtime: ABSlayerDirectionRuntime(
                    backend: "mlx-swift-cuda", maximumSequenceLength: 128,
                    gpuMemoryUtilization: 0.8, temporaryDirectory: nil),
                to: artifact.path)
            let loaded = try ABSlayerDirectionArtifactStore.load(from: artifact.path)
            let markerURL = candidate.appendingPathComponent("abslayer.json")

            func writeMarker(
                effectiveStrength: Double,
                addUnexpectedField: Bool = false
            ) throws {
                let marker = ABSlayerApplyMarker(
                    format: "abslayer.candidate/v1",
                    source: source.path, sourceRevision: "rev-a",
                    sourceMetadataSHA256: model.metadataSHA256,
                    sourceWeightsSHA256: model.weightsSHA256,
                    candidateWeightsSHA256: String(repeating: "d", count: 64),
                    artifact: artifact.path,
                    artifactManifestSHA256: loaded.manifestSHA256,
                    directionsSHA256: loaded.directionsSHA256,
                    strength: 1.5, effectiveStrength: effectiveStrength,
                    modifiedTensors: 2, selectedLayers: [0])
                var object = try #require(
                    JSONSerialization.jsonObject(
                        with: JSONEncoder().encode(marker)) as? [String: Any])
                if addUnexpectedField { object["unexpected"] = true }
                try JSONSerialization.data(withJSONObject: object).write(
                    to: markerURL, options: .atomic)
            }

            try writeMarker(effectiveStrength: 0.75)
            let candidateModel = ABSlayerModelBinding(
                identifier: candidate.path, canonicalPath: candidate.path,
                revision: nil, metadataSHA256: model.metadataSHA256,
                weightsSHA256: String(repeating: "d", count: 64),
                decoderLayerCount: 1, hiddenSize: 2)
            try ABSlayerBackendRuntime.validateCandidateMarker(
                candidateDirectory: candidate.path, source: model,
                candidate: candidateModel,
                sourceIdentifier: source.path)

            try writeMarker(effectiveStrength: 0.76)
            #expect(throws: ABSlayerBackendRuntimeError.self) {
                try ABSlayerBackendRuntime.validateCandidateMarker(
                    candidateDirectory: candidate.path, source: model,
                    candidate: candidateModel,
                    sourceIdentifier: source.path)
            }

            try writeMarker(effectiveStrength: 0.75, addUnexpectedField: true)
            #expect(throws: ABSlayerBackendRuntimeError.self) {
                try ABSlayerBackendRuntime.validateCandidateMarker(
                    candidateDirectory: candidate.path, source: model,
                    candidate: candidateModel,
                    sourceIdentifier: source.path)
            }

            try writeMarker(effectiveStrength: 0.75)
            var rawMarker = try String(contentsOf: markerURL, encoding: .utf8)
            rawMarker = rawMarker.replacingOccurrences(
                of: "{", with: "{\"f\\u006frmat\":\"abslayer.candidate/v1\",",
                options: [], range: rawMarker.range(of: "{"))
            try Data(rawMarker.utf8).write(to: markerURL, options: .atomic)
            #expect(throws: ABSlayerBackendRuntimeError.self) {
                try ABSlayerBackendRuntime.validateCandidateMarker(
                    candidateDirectory: candidate.path, source: model,
                    candidate: candidateModel, sourceIdentifier: source.path)
            }

            let mismatchedSource = ABSlayerModelBinding(
                identifier: source.path, canonicalPath: source.path,
                revision: "rev-a",
                metadataSHA256: String(repeating: "e", count: 64),
                weightsSHA256: model.weightsSHA256,
                decoderLayerCount: 1, hiddenSize: 2)
            try writeMarker(effectiveStrength: 0.75)
            #expect(throws: ABSlayerBackendRuntimeError.self) {
                try ABSlayerBackendRuntime.validateCandidateMarker(
                    candidateDirectory: candidate.path, source: mismatchedSource,
                    candidate: candidateModel,
                    sourceIdentifier: source.path)
            }
        }
    }

    @Test("backend process lock fails closed under contention")
    func processLockContention() throws {
        try withTemporaryDirectory { root in
            let path = root.appendingPathComponent("backend.lock").path
            do {
                let first = try ABSlayerProcessLock(path: path)
                #expect(throws: ABSlayerDirectionArtifactError.self) {
                    try ABSlayerProcessLock(path: path)
                }
                withExtendedLifetime(first) {}
            }
            let reacquired = try ABSlayerProcessLock(path: path)
            withExtendedLifetime(reacquired) {}

            let target = root.appendingPathComponent("lock-target")
            let link = root.appendingPathComponent("lock-link")
            try Data().write(to: target)
            try FileManager.default.createSymbolicLink(
                at: link, withDestinationURL: target)
            #expect(throws: ABSlayerDirectionArtifactError.self) {
                try ABSlayerProcessLock(path: link.path)
            }
        }
    }

    @Test("checkpoint sizing rejects symbolic links instead of following them")
    func checkpointSizeRejectsSymlinks() throws {
        try withTemporaryDirectory { root in
            let model = root.appendingPathComponent("model", isDirectory: true)
            try FileManager.default.createDirectory(
                at: model, withIntermediateDirectories: false)
            let weight = model.appendingPathComponent("weight.bin")
            try Data([1, 2, 3]).write(to: weight)
            #expect(try ABSlayerFileSystem.recursiveRegularFileBytes(at: model) == 3)
            try FileManager.default.createSymbolicLink(
                at: model.appendingPathComponent("weight-link.bin"),
                withDestinationURL: weight)
            #expect(throws: ABSlayerDirectionArtifactError.self) {
                try ABSlayerFileSystem.recursiveRegularFileBytes(at: model)
            }
        }
    }

    @Test("RTX utilization maps into the guarded one-to-22 GiB envelope")
    func cudaUtilizationEnvelope() throws {
        #expect(try MLXResourceLimits.cudaMemoryLimitGiB(utilization: 0.01) == 1)
        #expect(try MLXResourceLimits.cudaMemoryLimitGiB(utilization: 0.5) == 12)
        #expect(try MLXResourceLimits.cudaMemoryLimitGiB(utilization: 1) == 22)
        #expect(throws: MLXResourceGuardError.self) {
            try MLXResourceLimits.cudaMemoryLimitGiB(utilization: .infinity)
        }
    }

    @Test("verification uses refusal-channel markers and never truncates references")
    func verificationMarkerAndReferenceBounds() throws {
        #expect(ABSlayerBackendRuntime.refusalMarker(
            "Sorry, that would be harmful."))
        #expect(!RefusalEvaluator.hasControlRefusalMarker(
            "Sorry, that would be harmful."))
        let maximum = Array(0 ..< 128)
        #expect(try ABSlayerBackendRuntime.boundedReferenceContinuation(
            maximum, caseName: "maximum") == maximum)
        #expect(throws: ABSlayerBackendRuntimeError.self) {
            try ABSlayerBackendRuntime.boundedReferenceContinuation(
                Array(0 ..< 129), caseName: "too-long")
        }
    }

    @Test("BF16 editor changes only explicitly selected decoder layers")
    func selectedLayerBF16Edit() throws {
        try withTemporaryDirectory { root in
            let source = root.appendingPathComponent("source", isDirectory: true)
            let output = root.appendingPathComponent("output", isDirectory: true)
            try FileManager.default.createDirectory(
                at: source, withIntermediateDirectories: false)
            let layer0 = BF16WeightEditor.attentionOutputProjectionKey(layer: 0)
            let layer1 = BF16WeightEditor.attentionOutputProjectionKey(layer: 1)
            let mlp0 = BF16WeightEditor.mlpDownProjectionKey(layer: 0)
            let mlp1 = BF16WeightEditor.mlpDownProjectionKey(layer: 1)
            let shard = "model-00001-of-00001.safetensors"
            let original = MLXArray([
                Float(1), 2,
                3, 4,
                5, 6,
                7, 8,
            ]).reshaped(4, 2).asType(.bfloat16)
            try MLX.save(
                arrays: [
                    layer0: original, layer1: original,
                    mlp0: original, mlp1: original,
                ],
                url: source.appendingPathComponent(shard))
            let index: [String: Any] = [
                "metadata": ["total_size": 32],
                "weight_map": [
                    layer0: shard, layer1: shard,
                    mlp0: shard, mlp1: shard,
                ],
            ]
            try JSONSerialization.data(withJSONObject: index).write(
                to: source.appendingPathComponent("model.safetensors.index.json"))
            try Data("{}".utf8).write(
                to: source.appendingPathComponent("config.json"))

            let kernel = LayerAblationKernel(
                maximum: 1, peakLayer: 0, minimum: 1, radius: 2)
            let summary = try BF16WeightEditor.edit(
                sourcePath: source.path, outputPath: output.path,
                directions: [[1, 0, 0, 0], [1, 0, 0, 0]],
                subspaces: [[[1, 0, 0, 0]], [[1, 0, 0, 0]]],
                configuration: AbliterationConfiguration(
                    attention: kernel, mlp: kernel, normalization: .none),
                selectedLayers: [1])
            #expect(summary.editedAttentionMatrices == 1)
            #expect(summary.editedMLPMatrices == 1)
            let (arrays, _) = try loadArraysAndMetadata(
                url: output.appendingPathComponent(shard))
            let untouched = arrays[layer0]!.asType(.float32)
            let edited = arrays[layer1]!.asType(.float32)
            eval(untouched, edited)
            #expect(untouched.asArray(Float.self)
                == original.asType(.float32).asArray(Float.self))
            #expect(edited.asArray(Float.self)
                != original.asType(.float32).asArray(Float.self))
        }
    }

    @Test("BF16 edit requires every selected tensor in its index-declared shard")
    func selectedLayerBF16EditRejectsMissingOrMisplacedTargets() throws {
        try withTemporaryDirectory { root in
            let attention = BF16WeightEditor.attentionOutputProjectionKey(layer: 0)
            let mlp = BF16WeightEditor.mlpDownProjectionKey(layer: 0)
            let matrix = MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2)
                .asType(.bfloat16)
            let kernel = LayerAblationKernel(
                maximum: 1, peakLayer: 0, minimum: 1, radius: 1)
            let configuration = AbliterationConfiguration(
                attention: kernel, mlp: kernel, normalization: .none)

            let missing = root.appendingPathComponent("missing", isDirectory: true)
            try FileManager.default.createDirectory(
                at: missing, withIntermediateDirectories: false)
            let oneShard = "model-00001-of-00001.safetensors"
            try MLX.save(
                arrays: [attention: matrix],
                url: missing.appendingPathComponent(oneShard))
            try JSONSerialization.data(withJSONObject: [
                "weight_map": [attention: oneShard, mlp: oneShard],
            ]).write(to: missing.appendingPathComponent(
                "model.safetensors.index.json"))
            #expect(throws: EditorError.self) {
                try BF16WeightEditor.edit(
                    sourcePath: missing.path,
                    outputPath: root.appendingPathComponent("missing-output").path,
                    directions: [[1, 0]], subspaces: [[[1, 0]]],
                    configuration: configuration, selectedLayers: [0])
            }

            let misplaced = root.appendingPathComponent("misplaced", isDirectory: true)
            try FileManager.default.createDirectory(
                at: misplaced, withIntermediateDirectories: false)
            let shardA = "model-00001-of-00002.safetensors"
            let shardB = "model-00002-of-00002.safetensors"
            try MLX.save(
                arrays: [mlp: matrix],
                url: misplaced.appendingPathComponent(shardA))
            try MLX.save(
                arrays: [attention: matrix],
                url: misplaced.appendingPathComponent(shardB))
            try JSONSerialization.data(withJSONObject: [
                "weight_map": [attention: shardA, mlp: shardB],
            ]).write(to: misplaced.appendingPathComponent(
                "model.safetensors.index.json"))
            #expect(throws: EditorError.self) {
                try BF16WeightEditor.edit(
                    sourcePath: misplaced.path,
                    outputPath: root.appendingPathComponent("misplaced-output").path,
                    directions: [[1, 0]], subspaces: [[[1, 0]]],
                    configuration: configuration, selectedLayers: [0])
            }
        }
    }

    @Test("BF16 edit layout requires both targets and safe regular shards")
    func validatesBF16EditLayout() throws {
        try withTemporaryDirectory { root in
            let source = root.appendingPathComponent("source", isDirectory: true)
            try FileManager.default.createDirectory(
                at: source, withIntermediateDirectories: false)
            let attention = BF16WeightEditor.attentionOutputProjectionKey(layer: 0)
            let mlp = BF16WeightEditor.mlpDownProjectionKey(layer: 0)
            let shard = "model-00001-of-00001.safetensors"
            let shardURL = source.appendingPathComponent(shard)
            try Data([1]).write(to: shardURL)
            let indexURL = source.appendingPathComponent(
                "model.safetensors.index.json")

            func writeIndex(_ weightMap: [String: String]) throws {
                try JSONSerialization.data(withJSONObject: [
                    "metadata": ["total_size": 1],
                    "weight_map": weightMap,
                ]).write(to: indexURL)
            }

            try writeIndex([attention: shard, mlp: shard])
            try BF16WeightEditor.validateEditableLayout(
                sourcePath: source.path, layers: [0])

            try writeIndex([attention: shard])
            #expect(throws: EditorError.self) {
                try BF16WeightEditor.validateEditableLayout(
                    sourcePath: source.path, layers: [0])
            }

            try writeIndex([attention: "../escaped.safetensors", mlp: shard])
            #expect(throws: EditorError.self) {
                try BF16WeightEditor.validateEditableLayout(
                    sourcePath: source.path, layers: [0])
            }
            #expect(throws: EditorError.self) {
                try BF16WeightEditor.loadMatrices(
                    sourcePath: source.path, keys: [attention])
            }

            try writeIndex([attention: shard, mlp: shard])
            try FileManager.default.removeItem(at: shardURL)
            let target = source.appendingPathComponent("target.safetensors")
            try Data([1]).write(to: target)
            try FileManager.default.createSymbolicLink(
                at: shardURL, withDestinationURL: target)
            #expect(throws: EditorError.self) {
                try BF16WeightEditor.validateEditableLayout(
                    sourcePath: source.path, layers: [0])
            }
        }
    }

    @Test("backend entry point disables graphs before parsing and emits no print noise")
    func backendEntrypointInitializationOrder() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let entrypoint = packageRoot.appendingPathComponent(
            "Sources/ABSlayerBackend/main.swift")
        let text = try String(contentsOf: entrypoint, encoding: .utf8)
        let graph = try #require(text.range(
            of: "setenv(\"MLX_USE_CUDA_GRAPHS\", \"0\", 1)"))
        let parse = try #require(text.range(
            of: "ABSlayerBackendInvocation.parse"))
        let runtime = try #require(text.range(
            of: "ABSlayerBackendRuntime.doctor"))
        #expect(graph.lowerBound < parse.lowerBound)
        #expect(parse.lowerBound < runtime.lowerBound)
        #expect(!text.contains("print("))
        #expect(text.contains("if !report.passed { exit(5) }"))
    }

    private func withTemporaryDirectory(
        _ body: (URL) throws -> Void
    ) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "abslayer-backend-contract-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }
}
