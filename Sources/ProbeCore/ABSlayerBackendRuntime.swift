import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

public struct ABSlayerDoctorResult: Codable, Sendable {
    public let status: String
    public let command: String
    public let role: ABSlayerBackendRole
    public let backend: String
    public let model: ABSlayerModelBinding
}

public struct ABSlayerMeasureResult: Codable, Sendable {
    public let status: String
    public let command: String
    public let artifact: String
    public let selectedLayers: [Int]
    public let promptCount: Int

    enum CodingKeys: String, CodingKey {
        case status, command, artifact
        case selectedLayers = "selected_layers"
        case promptCount = "prompt_count"
    }
}

public struct ABSlayerApplyResult: Codable, Sendable {
    public let status: String
    public let command: String
    public let output: String
    public let modifiedTensors: Int

    enum CodingKeys: String, CodingKey {
        case status, command, output
        case modifiedTensors = "modified_tensors"
    }
}

public enum ABSlayerBackendRuntime {
    public static func doctor(
        _ invocation: ABSlayerDoctorInvocation
    ) throws -> ABSlayerDoctorResult {
        try requireCUDA()
        let model = try ABSlayerProductionModelInspector.inspect(
            identifier: invocation.model, revision: invocation.revision)
        if invocation.role == .apply {
            try BF16WeightEditor.validateEditableLayout(
                sourcePath: model.canonicalPath,
                layers: Set(0 ..< model.decoderLayerCount))
        }
        return ABSlayerDoctorResult(
            status: "ok", command: "doctor", role: invocation.role,
            backend: "mlx-swift-cuda", model: model)
    }

    public static func measure(
        _ invocation: ABSlayerMeasureInvocation
    ) async throws -> ABSlayerMeasureResult {
        try requireCUDA()
        let model = try ABSlayerProductionModelInspector.inspect(
            identifier: invocation.model, revision: invocation.revision)
        let artifactDestination = URL(fileURLWithPath: invocation.artifactPath)
            .standardizedFileURL
        guard !ABSlayerFileSystem.pathExistsWithoutFollowingSymlink(
            artifactDestination.path)
        else {
            throw ABSlayerDirectionArtifactError.outputExists(
                artifactDestination.path)
        }
        let measurementInput = try ABSlayerBackendJSONL.loadMeasurementPairsBound(
            path: invocation.pairsPath)
        let allPairs = measurementInput.records
        let datasetSHA256 = measurementInput.sha256
        let pairs = try selectedMeasurementPairs(
            allPairs, requested: invocation.promptCount)
        let promptPairs = pairs.map {
            PromptPair(name: $0.name, contrast: $0.contrast, control: $0.control)
        }
        let report = try await ProbeEngine(
            modelDirectory: model.canonicalPath,
            pairs: promptPairs,
            subspaceRank: invocation.rank,
            extractionMethod: .centroidDifference,
            generateResponses: false,
            tokenPosition: .postInstruction,
            maximumSequenceLength: invocation.maximumSequenceLength,
            gpuMemoryUtilization: invocation.gpuMemoryUtilization,
            emitResourceStatus: false).run()
        guard report.layers.count == model.decoderLayerCount,
              report.subspaces.count == model.decoderLayerCount
        else {
            throw ABSlayerBackendRuntimeError.modelShapeMismatch(
                expectedLayers: model.decoderLayerCount,
                actualLayers: report.subspaces.count)
        }
        let metrics = report.layers.map {
            ABSlayerLayerMetric(
                zeroBasedLayer: $0.layer - 1,
                cosineDistance: $0.cosineDistance,
                directionAgreement: $0.directionAgreement,
                medianDirectionAgreement: $0.medianDirectionAgreement,
                silhouette: $0.silhouette)
        }
        let selectedLayers = try ABSlayerBackendLayerSelection.select(
            metrics: metrics, layerCount: model.decoderLayerCount,
            fraction: invocation.maximumLayerFraction)
        try ABSlayerBackendJSONL.validateUnchanged(
            path: invocation.pairsPath, sha256: datasetSHA256)
        try ABSlayerProductionModelInspector.validateUnchanged(model)
        let dataset = ABSlayerDatasetBinding(
            path: invocation.pairsPath,
            sha256: datasetSHA256,
            recordCount: allPairs.count,
            promptCount: pairs.count)
        let algorithm = ABSlayerDirectionAlgorithm(
            name: "paired-centroid-rank-k/v1",
            strength: invocation.strength,
            rank: invocation.rank,
            maximumLayerFraction: invocation.maximumLayerFraction,
            selectedLayers: selectedLayers,
            tokenPosition: ActivationTokenPosition.postInstruction.rawValue)
        _ = try ABSlayerDirectionArtifactStore.write(
            subspaces: report.subspaces,
            model: model,
            dataset: dataset,
            algorithm: algorithm,
            runtime: ABSlayerDirectionRuntime(
                backend: "mlx-swift-cuda",
                maximumSequenceLength: invocation.maximumSequenceLength,
                gpuMemoryUtilization: invocation.gpuMemoryUtilization,
                temporaryDirectory: invocation.temporaryDirectory),
            to: invocation.artifactPath)
        return ABSlayerMeasureResult(
            status: "ok", command: "measure",
            artifact: URL(fileURLWithPath: invocation.artifactPath)
                .standardizedFileURL.path,
            selectedLayers: selectedLayers, promptCount: pairs.count)
    }

    public static func apply(
        _ invocation: ABSlayerApplyInvocation
    ) throws -> ABSlayerApplyResult {
        try requireCUDA()
        try MLXResourceGuard.apply(emitStatus: false)
        let model = try ABSlayerProductionModelInspector.inspect(
            identifier: invocation.model, revision: invocation.revision)
        let artifact = try ABSlayerDirectionArtifactStore.load(
            from: invocation.artifactPath)
        try validateArtifactModelBinding(artifact.manifest.model, against: model)
        try BF16WeightEditor.validateEditableLayout(
            sourcePath: model.canonicalPath,
            layers: Set(artifact.manifest.algorithm.selectedLayers))

        let destination = URL(fileURLWithPath: invocation.outputPath).standardizedFileURL
        guard !ABSlayerFileSystem.pathExistsWithoutFollowingSymlink(destination.path) else {
            throw ABSlayerBackendRuntimeError.outputExists(destination.path)
        }
        let source = URL(fileURLWithPath: model.canonicalPath).standardizedFileURL
        guard destination.path != source.path,
              !destination.path.hasPrefix(source.path + "/")
        else { throw ABSlayerBackendRuntimeError.outputInsideSource(destination.path) }
        let parent = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: true)
        guard ABSlayerFileSystem.isDirectoryWithoutFollowingSymlink(parent.path) else {
            throw ABSlayerBackendRuntimeError.invalidOutputParent(parent.path)
        }
        let checkpointBytes = try ABSlayerFileSystem.recursiveRegularFileBytes(at: source)
        let availableBytes = try ABSlayerFileSystem.availableCapacity(at: parent)
        try ABSlayerDiskPreflight.validate(
            checkpointBytes: checkpointBytes, availableBytes: availableBytes)

        let staging = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).staging-\(UUID().uuidString)",
            isDirectory: true)
        let effectiveStrength = invocation.strength
            * artifact.manifest.algorithm.strength
        guard effectiveStrength.isFinite, effectiveStrength > 0,
              effectiveStrength <= 4
        else { throw ABSlayerBackendRuntimeError.invalidEffectiveStrength }
        do {
            let uniform = Float(effectiveStrength)
            let kernel = LayerAblationKernel(
                maximum: uniform, peakLayer: 0, minimum: uniform,
                radius: Float(model.decoderLayerCount))
            let summary = try BF16WeightEditor.edit(
                sourcePath: source.path,
                outputPath: staging.path,
                directions: artifact.directions,
                subspaces: artifact.subspaces,
                configuration: AbliterationConfiguration(
                    attention: kernel, mlp: kernel,
                    directionScope: .perLayer,
                    normalization: .full,
                    composition: .simultaneous),
                selectedLayers: Set(artifact.manifest.algorithm.selectedLayers))
            let modified = summary.editedAttentionMatrices + summary.editedMLPMatrices
            let expectedModified = artifact.manifest.algorithm.selectedLayers.count * 2
            guard modified == expectedModified else {
                throw ABSlayerBackendRuntimeError.unexpectedModifiedTensorCount(
                    expected: expectedModified, actual: modified)
            }
            try ABSlayerProductionModelInspector.validateUnchanged(model)
            let candidateMetadataSHA256 = try ABSlayerCheckpointProvenance
                .metadataSHA256(directory: staging.path)
            guard candidateMetadataSHA256 == model.metadataSHA256 else {
                throw ABSlayerBackendRuntimeError.candidateMetadataMismatch
            }
            let candidateWeightsSHA256 = try ABSlayerCheckpointProvenance
                .weightsSHA256(directory: staging.path)
            let candidateBinding = ABSlayerModelBinding(
                identifier: staging.path, canonicalPath: staging.path,
                revision: nil, metadataSHA256: candidateMetadataSHA256,
                weightsSHA256: candidateWeightsSHA256,
                decoderLayerCount: model.decoderLayerCount,
                hiddenSize: model.hiddenSize)
            let marker = ABSlayerApplyMarker(
                format: "abslayer.candidate/v1",
                source: invocation.model,
                sourceRevision: invocation.revision,
                sourceMetadataSHA256: model.metadataSHA256,
                sourceWeightsSHA256: model.weightsSHA256,
                candidateWeightsSHA256: candidateWeightsSHA256,
                artifact: artifact.path,
                artifactManifestSHA256: artifact.manifestSHA256,
                directionsSHA256: artifact.directionsSHA256,
                strength: invocation.strength,
                effectiveStrength: effectiveStrength,
                modifiedTensors: modified,
                selectedLayers: artifact.manifest.algorithm.selectedLayers)
            try canonicalJSON(marker).write(
                to: staging.appendingPathComponent("abslayer.json"),
                options: .withoutOverwriting)
            try ABSlayerProductionModelInspector.validateUnchanged(candidateBinding)
            try FileManager.default.moveItem(at: staging, to: destination)
            return ABSlayerApplyResult(
                status: "ok", command: "apply", output: destination.path,
                modifiedTensors: modified)
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw error
        }
    }

    public static func verify(
        _ invocation: ABSlayerVerifyInvocation
    ) async throws -> ABSlayerBalancedVerificationReport {
        try requireCUDA()
        let source = try ABSlayerProductionModelInspector.inspect(
            identifier: invocation.sourceModel,
            revision: invocation.sourceRevision)
        let candidate = try ABSlayerProductionModelInspector.inspect(
            identifier: invocation.candidateModel, revision: nil)
        guard candidate.metadataSHA256 == source.metadataSHA256,
              candidate.decoderLayerCount == source.decoderLayerCount,
              candidate.hiddenSize == source.hiddenSize
        else { throw ABSlayerBackendRuntimeError.candidateMetadataMismatch }
        try validateCandidateMarker(
            candidateDirectory: candidate.canonicalPath,
            source: source, candidate: candidate,
            sourceIdentifier: invocation.sourceModel)
        let evaluationInput = try ABSlayerBackendJSONL.loadEvaluationCasesBound(
            path: invocation.casesPath)
        let cases = evaluationInput.records
        let casesSHA256 = evaluationInput.sha256
        let reportDestination = URL(fileURLWithPath: invocation.reportPath)
            .standardizedFileURL
        guard !ABSlayerFileSystem.pathExistsWithoutFollowingSymlink(
            reportDestination.path)
        else { throw ABSlayerBackendRuntimeError.outputExists(reportDestination.path) }
        let sourceSamples = try await evaluate(
            modelDirectory: source.canonicalPath, cases: cases,
            gpuMemoryUtilization: invocation.gpuMemoryUtilization)
        try ABSlayerProductionModelInspector.validateUnchanged(source)
        let candidateSamples = try await evaluate(
            modelDirectory: candidate.canonicalPath, cases: cases,
            gpuMemoryUtilization: invocation.gpuMemoryUtilization)
        try ABSlayerProductionModelInspector.validateUnchanged(candidate)
        let refusalIndices = cases.indices.filter { cases[$0].kind == .refusal }
        let utilityIndices = cases.indices.filter { cases[$0].kind == .utility }
        let sourceUtilityNLL = utilityIndices.compactMap {
            sourceSamples[$0].referenceNLL
        }
        let candidateUtilityNLL = utilityIndices.compactMap {
            candidateSamples[$0].referenceNLL
        }
        guard sourceUtilityNLL.count == utilityIndices.count,
              candidateUtilityNLL.count == utilityIndices.count
        else { throw ABSlayerBackendRuntimeError.missingReferenceNLL }
        try ABSlayerBackendJSONL.validateUnchanged(
            path: invocation.casesPath, sha256: casesSHA256)
        let report = try ABSlayerBalancedVerification.makeReport(
            ABSlayerVerificationInput(
                sourceRefusals: refusalIndices.map { sourceSamples[$0].refused },
                candidateRefusals: refusalIndices.map { candidateSamples[$0].refused },
                candidateNonemptyGenerations: candidateSamples.map(\.nonempty),
                sourceUtilityNLL: sourceUtilityNLL,
                candidateUtilityNLL: candidateUtilityNLL,
                sourceModel: invocation.sourceModel,
                candidateModel: invocation.candidateModel,
                sourceRevision: invocation.sourceRevision,
                cases: ABSlayerDatasetBinding(
                    path: invocation.casesPath,
                    sha256: casesSHA256,
                    recordCount: cases.count, promptCount: cases.count)))
        try writeJSONFileNoOverwrite(report, to: invocation.reportPath)
        return report
    }

    private struct EvaluationSample: Sendable {
        let refused: Bool
        let nonempty: Bool
        let referenceNLL: Double?
    }

    private static func evaluate(
        modelDirectory: String, cases: [ABSlayerEvaluationCase],
        gpuMemoryUtilization: Double
    ) async throws -> [EvaluationSample] {
        do {
            let result = try await evaluateLoaded(
                modelDirectory: modelDirectory, cases: cases,
                gpuMemoryUtilization: gpuMemoryUtilization)
            // The model container has left scope before cached CUDA allocations
            // are released, so verify never intentionally retains both full
            // checkpoints at once.
            Memory.clearCache()
            return result
        } catch {
            Memory.clearCache()
            throw error
        }
    }

    private static func evaluateLoaded(
        modelDirectory: String, cases: [ABSlayerEvaluationCase],
        gpuMemoryUtilization: Double
    ) async throws -> [EvaluationSample] {
        let configuration = ModelConfiguration(
            directory: URL(fileURLWithPath: modelDirectory).standardizedFileURL,
            extraEOSTokens: ["<end_of_turn>"])
        try MLXResourceGuard.apply(gpuMemoryUtilization: gpuMemoryUtilization, emitStatus: false)
        let container = try await #huggingFaceLoadModelContainer(
            configuration: configuration)
        try await container.perform { context in
            guard context.model is Gemma4Model || context.model is Gemma4TextModel else {
                throw ABSlayerBackendRuntimeError.unsupportedLoadedModel(
                    String(describing: type(of: context.model)))
            }
            let tokenizer = ABSlayerProductionPromptTokenizer(
                tokenizer: context.tokenizer)
            for item in cases {
                let tokens = try tokenizer.userPromptTokens(item.prompt)
                guard !tokens.isEmpty else {
                    throw ProbeError.emptyPromptTokenization(name: item.name)
                }
                guard tokens.count <= ABSlayerPreflight.evaluationMaximumPromptTokens
                else {
                    throw ProbeError.promptTooLong(
                        name: item.name, tokenCount: tokens.count)
                }
            }
        }
        var result = [EvaluationSample]()
        result.reserveCapacity(cases.count)
        let parameters = GenerateParameters(
            maxTokens: EvaluationGenerationOptions.defaultMaximumTokens,
            temperature: 0)
        for item in cases {
            let response = try await ChatSession(
                container, generateParameters: parameters).respond(to: item.prompt)
            let referenceNLL: Double?
            if let reference = item.reference {
                referenceNLL = try await container.perform { context in
                    try meanReferenceNLL(
                        context: context, prompt: item.prompt,
                        reference: reference, caseName: item.name)
                }
            } else {
                referenceNLL = nil
            }
            result.append(EvaluationSample(
                refused: refusalMarker(response),
                nonempty: !response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                referenceNLL: referenceNLL))
        }
        return result
    }

    private static func meanReferenceNLL(
        context: ModelContext, prompt: String, reference: String, caseName: String
    ) throws -> Double {
        let tokenizer = ABSlayerProductionPromptTokenizer(
            tokenizer: context.tokenizer)
        let promptTokens = try tokenizer.userPromptTokens(prompt)
        guard !promptTokens.isEmpty else {
            throw ProbeError.emptyPromptTokenization(name: caseName)
        }
        guard promptTokens.count <= ABSlayerPreflight.evaluationMaximumPromptTokens else {
            throw ProbeError.promptTooLong(name: caseName, tokenCount: promptTokens.count)
        }
        let derived: [Int]
        do {
            derived = try ABSlayerReferenceTokenization.continuation(
                promptTokens: promptTokens,
                prompt: prompt, reference: reference, tokenizer: tokenizer)
        } catch {
            throw ProbeError.referenceContinuationTokenizationMismatch(name: caseName)
        }
        let continuation = try boundedReferenceContinuation(
            derived, caseName: caseName)
        guard promptTokens.count + continuation.count
            <= ABSlayerPreflight.utilityMaximumTotalTokens
        else { throw ABSlayerBackendRuntimeError.invalidReference(caseName) }
        let cache = try context.model.newCache(parameters: nil)
        var targetLogProbability = 0.0
        for position in continuation.indices {
            let inputTokens = position == 0
                ? promptTokens : [continuation[position - 1]]
            let logs = MLXNN.logSoftmax(
                context.model(
                    MLXArray(inputTokens).expandedDimensions(axis: 0),
                    cache: cache)[0, -1].asType(.float32),
                axis: -1)
            let target = logs[continuation[position]]
            eval(target)
            let value = Double(target.item(Float.self))
            guard value.isFinite else {
                throw ABSlayerBackendRuntimeError.nonfiniteReferenceNLL(caseName)
            }
            targetLogProbability += value
        }
        return -targetLogProbability / Double(continuation.count)
    }

    static func refusalMarker(_ response: String) -> Bool {
        RefusalEvaluator.hasContrastRefusalMarker(response)
    }

    static func boundedReferenceContinuation(
        _ derived: [Int], caseName: String
    ) throws -> [Int] {
        guard !derived.isEmpty,
              derived.count <= ABSlayerPreflight.utilityMaximumContinuationTokens
        else {
            throw ABSlayerBackendRuntimeError.invalidReference(caseName)
        }
        return derived
    }

    private static func validateArtifactModelBinding(
        _ artifact: ABSlayerModelBinding, against model: ABSlayerModelBinding
    ) throws {
        guard artifact.canonicalPath == model.canonicalPath,
              artifact.revision == model.revision,
              artifact.metadataSHA256 == model.metadataSHA256,
              artifact.weightsSHA256 == model.weightsSHA256,
              artifact.decoderLayerCount == model.decoderLayerCount,
              artifact.hiddenSize == model.hiddenSize
        else { throw ABSlayerBackendRuntimeError.artifactModelMismatch }
    }

    static func validateCandidateMarker(
        candidateDirectory: String, source: ABSlayerModelBinding,
        candidate: ABSlayerModelBinding,
        sourceIdentifier: String
    ) throws {
        let url = URL(fileURLWithPath: candidateDirectory)
            .appendingPathComponent("abslayer.json")
        guard let markerIdentity = ABSlayerFileSystem.regularFileIdentity(url.path),
              markerIdentity.size >= 0, markerIdentity.size <= 1_048_576
        else {
            throw ABSlayerBackendRuntimeError.invalidCandidateMarker
        }
        let mappedMarker = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard Int64(mappedMarker.count) == markerIdentity.size,
              ABSlayerFileSystem.regularFileIdentity(url.path) == markerIdentity
        else { throw ABSlayerBackendRuntimeError.invalidCandidateMarker }
        let markerData = mappedMarker.withUnsafeBytes { Data($0) }
        guard ABSlayerFileSystem.regularFileIdentity(url.path) == markerIdentity,
              (try? ABSlayerStrictJSON.validateNoDuplicateKeys(markerData)) != nil,
              let markerObject = try JSONSerialization.jsonObject(with: markerData)
                  as? [String: Any]
        else { throw ABSlayerBackendRuntimeError.invalidCandidateMarker }
        let requiredMarkerKeys: Set<String> = [
            "format", "source", "source_metadata_sha256", "artifact",
            "source_weights_sha256", "candidate_weights_sha256",
            "artifact_manifest_sha256", "directions_sha256", "strength",
            "effective_strength", "modified_tensors", "selected_layers",
        ]
        let markerKeys = Set(markerObject.keys)
        guard requiredMarkerKeys.isSubset(of: markerKeys),
              markerKeys.isSubset(of: requiredMarkerKeys.union(["source_revision"]))
        else { throw ABSlayerBackendRuntimeError.invalidCandidateMarker }
        let marker: ABSlayerApplyMarker
        do { marker = try JSONDecoder().decode(
            ABSlayerApplyMarker.self, from: markerData) }
        catch { throw ABSlayerBackendRuntimeError.invalidCandidateMarker }
        let markerSource = URL(fileURLWithPath: marker.source).standardizedFileURL
            .resolvingSymlinksInPath().path
        let requestedSource = URL(fileURLWithPath: sourceIdentifier)
            .standardizedFileURL.resolvingSymlinksInPath().path
        let artifact: ABSlayerDirectionArtifact
        do { artifact = try ABSlayerDirectionArtifactStore.load(from: marker.artifact) }
        catch { throw ABSlayerBackendRuntimeError.invalidCandidateMarker }
        do { try validateArtifactModelBinding(artifact.manifest.model, against: source) }
        catch { throw ABSlayerBackendRuntimeError.invalidCandidateMarker }
        let expectedEffectiveStrength = marker.strength
            * artifact.manifest.algorithm.strength
        guard marker.format == "abslayer.candidate/v1",
              markerSource == requestedSource,
              markerSource == source.canonicalPath,
              marker.sourceRevision == source.revision,
              marker.sourceMetadataSHA256 == source.metadataSHA256,
              marker.sourceWeightsSHA256 == source.weightsSHA256,
              marker.candidateWeightsSHA256 == candidate.weightsSHA256,
              marker.artifactManifestSHA256 == artifact.manifestSHA256,
              marker.directionsSHA256 == artifact.directionsSHA256,
              marker.selectedLayers == artifact.manifest.algorithm.selectedLayers,
              marker.modifiedTensors == marker.selectedLayers.count * 2,
              marker.strength.isFinite, marker.strength > 0,
              marker.strength <= 2,
              marker.effectiveStrength.isFinite, marker.effectiveStrength > 0,
              marker.effectiveStrength <= 4,
              marker.effectiveStrength == expectedEffectiveStrength
        else { throw ABSlayerBackendRuntimeError.invalidCandidateMarker }
    }

    private static func selectedMeasurementPairs(
        _ values: [ABSlayerMeasurementPair], requested: Int?
    ) throws -> [ABSlayerMeasurementPair] {
        guard let requested else { return values }
        guard requested <= values.count else {
            throw ABSlayerBackendRuntimeError.promptCountExceedsDataset(
                requested: requested, available: values.count)
        }
        guard requested < values.count else { return values }
        return (0 ..< requested).map { values[$0 * values.count / requested] }
    }

    private static func requireCUDA() throws {
        guard MLXExecutionBackend.compiled == .cuda else {
            throw ABSlayerBackendRuntimeError.cudaRequired(
                MLXExecutionBackend.compiled.rawValue)
        }
    }

    private static func writeJSONFileNoOverwrite<T: Encodable>(
        _ value: T, to path: String
    ) throws {
        let destination = URL(fileURLWithPath: path).standardizedFileURL
        guard !ABSlayerFileSystem.pathExistsWithoutFollowingSymlink(destination.path) else {
            throw ABSlayerBackendRuntimeError.outputExists(destination.path)
        }
        let parent = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        guard ABSlayerFileSystem.isDirectoryWithoutFollowingSymlink(parent.path) else {
            throw ABSlayerBackendRuntimeError.invalidOutputParent(parent.path)
        }
        let temporary = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).staging-\(UUID().uuidString)")
        do {
            try canonicalJSON(value).write(to: temporary, options: .withoutOverwriting)
            try FileManager.default.moveItem(at: temporary, to: destination)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    private static func canonicalJSON<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var data = try encoder.encode(value)
        data.append(0x0a)
        return data
    }
}

public enum ABSlayerBackendRuntimeError: LocalizedError {
    case cudaRequired(String)
    case unsupportedModelMetadata(String, String?, String?)
    case unsupportedLoadedModel(String)
    case modelShapeMismatch(expectedLayers: Int, actualLayers: Int)
    case promptCountExceedsDataset(requested: Int, available: Int)
    case artifactModelMismatch
    case candidateMetadataMismatch
    case checkpointChanged(String)
    case outputExists(String)
    case outputInsideSource(String)
    case invalidOutputParent(String)
    case invalidEffectiveStrength
    case unexpectedModifiedTensorCount(expected: Int, actual: Int)
    case invalidCandidateMarker
    case invalidReference(String)
    case nonfiniteReferenceNLL(String)
    case missingReferenceNLL

    public var errorDescription: String? {
        switch self {
        case .cudaRequired(let backend):
            "The production harness backend requires an MLX CUDA build, not \(backend)."
        case .unsupportedModelMetadata(let path, let rootType, let textType):
            "The backend requires a local full-BF16 Gemma 4 checkpoint with decoder metadata: \(path) (model_type=\(rootType ?? "missing"), text model_type=\(textType ?? "missing"))."
        case .unsupportedLoadedModel(let type):
            "MLX loaded an unsupported model implementation: \(type)."
        case .modelShapeMismatch(let expected, let actual):
            "Measured \(actual) decoder layers but model metadata declares \(expected)."
        case .promptCountExceedsDataset(let requested, let available):
            "Requested \(requested) measurement pairs, but only \(available) are available."
        case .artifactModelMismatch:
            "Direction artifact is not bound to the exact source checkpoint metadata."
        case .candidateMetadataMismatch:
            "Candidate model metadata or decoder shape differs from the source checkpoint."
        case .checkpointChanged(let path):
            "Checkpoint changed across an operation boundary: \(path)"
        case .outputExists(let path): "Output exists; refusing to overwrite: \(path)"
        case .outputInsideSource(let path):
            "Candidate output cannot be the source checkpoint or a child of it: \(path)"
        case .invalidOutputParent(let path):
            "Candidate output parent is not a regular directory: \(path)"
        case .invalidEffectiveStrength: "Combined measure/apply strength is invalid."
        case .unexpectedModifiedTensorCount(let expected, let actual):
            "Expected to modify exactly \(expected) checkpoint tensors, but modified \(actual)."
        case .invalidCandidateMarker:
            "Candidate is missing a valid abslayer.json binding to the requested source."
        case .invalidReference(let name):
            "Utility case \(name) has no usable bounded reference continuation."
        case .nonfiniteReferenceNLL(let name):
            "Utility case \(name) produced a non-finite reference NLL."
        case .missingReferenceNLL:
            "A utility case did not produce a reference NLL sample."
        }
    }
}
