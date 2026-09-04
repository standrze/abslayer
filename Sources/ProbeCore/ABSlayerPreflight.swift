#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

public struct ABSlayerPreflightInvocation: Equatable, Sendable {
    public let model: String
    public let measurementPath: String
    public let evaluationPath: String
    public let maximumSequenceLength: Int
    public let reportPath: String

    public init(
        model: String, measurementPath: String, evaluationPath: String,
        maximumSequenceLength: Int, reportPath: String
    ) {
        self.model = model
        self.measurementPath = measurementPath
        self.evaluationPath = evaluationPath
        self.maximumSequenceLength = maximumSequenceLength
        self.reportPath = reportPath
    }

    public static func parse(arguments: [String]) throws -> Self {
        guard let model = arguments.first, !model.hasPrefix("--") else {
            throw ABSlayerBackendContractError.missingPositional("MODEL")
        }
        let flags = try StrictFlags.parse(
            Array(arguments.dropFirst()),
            allowed: [
                "--measurement", "--evaluation", "--max-sequence-length",
                "--report",
            ])
        try flags.requireExactly([
            "--measurement", "--evaluation", "--max-sequence-length", "--report",
        ])
        return Self(
            model: try checkedText(model, field: "MODEL"),
            measurementPath: try checkedText(
                flags.value("--measurement"), field: "--measurement"),
            evaluationPath: try checkedText(
                flags.value("--evaluation"), field: "--evaluation"),
            maximumSequenceLength: try boundedInteger(
                flags.value("--max-sequence-length"),
                field: "--max-sequence-length", minimum: 1, maximum: 32_768),
            reportPath: try checkedText(flags.value("--report"), field: "--report"))
    }
}

/// The exact BF16 Gemma 4 metadata and checkpoint binding used by the production
/// backend. Keeping this inspection in one helper prevents preflight and runtime
/// support policy from drifting apart.
public enum ABSlayerProductionModelInspector {
    public static func inspect(
        identifier: String, revision: String? = nil
    ) throws -> ABSlayerModelBinding {
        let inspection = try ModelFolderValidator.validateFullBF16(path: identifier)
        guard inspection.modelType == "gemma4",
              inspection.textModelType == "gemma4_text",
              let layerCount = inspection.decoderLayerCount,
              let hiddenSize = inspection.hiddenSize
        else {
            throw ABSlayerBackendRuntimeError.unsupportedModelMetadata(
                inspection.path, inspection.modelType, inspection.textModelType)
        }
        return ABSlayerModelBinding(
            identifier: identifier,
            canonicalPath: inspection.path,
            revision: revision,
            metadataSHA256: try ABSlayerCheckpointProvenance.metadataSHA256(
                directory: inspection.path),
            weightsSHA256: try ABSlayerCheckpointProvenance.weightsSHA256(
                directory: inspection.path),
            decoderLayerCount: layerCount,
            hiddenSize: hiddenSize)
    }

    public static func validateUnchanged(_ binding: ABSlayerModelBinding) throws {
        let metadata = try ABSlayerCheckpointProvenance.metadataSHA256(
            directory: binding.canonicalPath)
        let weights = try ABSlayerCheckpointProvenance.weightsSHA256(
            directory: binding.canonicalPath)
        guard metadata == binding.metadataSHA256,
              weights == binding.weightsSHA256
        else {
            throw ABSlayerBackendRuntimeError.checkpointChanged(binding.canonicalPath)
        }
    }
}

/// Narrow adapter boundary so token-budget validation can be tested without
/// loading model weights or depending on a synthetic tokenizer implementation.
public protocol ABSlayerPromptTokenizing: Sendable {
    func userPromptTokens(_ prompt: String) throws -> [Int]
    func promptPlusReferenceTokens(
        promptTokens: [Int], reference: String
    ) -> [Int]
    func completedConversationTokens(
        prompt: String, reference: String
    ) throws -> [Int]
}

struct ABSlayerProductionPromptTokenizer: ABSlayerPromptTokenizing {
    let tokenizer: any MLXLMCommon.Tokenizer

    static func load(modelDirectory: String) async throws -> Self {
        let directory = URL(fileURLWithPath: modelDirectory).standardizedFileURL
        let tokenizer = try await #huggingFaceTokenizerLoader().load(from: directory)
        return Self(tokenizer: tokenizer)
    }

    func userPromptTokens(_ prompt: String) throws -> [Int] {
        try tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": prompt],
        ])
    }

    func promptPlusReferenceTokens(
        promptTokens: [Int], reference: String
    ) -> [Int] {
        let renderedPrompt = tokenizer.decode(
            tokenIds: promptTokens, skipSpecialTokens: false)
        return tokenizer.encode(
            text: renderedPrompt + reference, addSpecialTokens: false)
    }

    func completedConversationTokens(
        prompt: String, reference: String
    ) throws -> [Int] {
        try tokenizer.applyChatTemplate(
            messages: [
                ["role": "user", "content": prompt],
                ["role": "assistant", "content": reference],
            ],
            tools: nil,
            additionalContext: ["add_generation_prompt": false])
    }
}

/// Shared continuation derivation for preflight and the production NLL path.
public enum ABSlayerReferenceTokenization {
    public static func continuation(
        promptTokens: [Int], prompt: String, reference: String,
        tokenizer: any ABSlayerPromptTokenizing
    ) throws -> [Int] {
        let promptPlusReference = tokenizer.promptPlusReferenceTokens(
            promptTokens: promptTokens, reference: reference)
        let templatedConversation: [Int]? =
            promptPlusReference.starts(with: promptTokens)
            ? nil
            : try tokenizer.completedConversationTokens(
                prompt: prompt, reference: reference)
        return try TeacherForcedContinuationTokenDerivation.continuation(
            promptTokens: promptTokens,
            promptPlusReferenceTokens: promptPlusReference,
            templatedConversationTokens: templatedConversation)
    }
}

public struct ABSlayerPreflightLimits: Codable, Equatable, Sendable {
    public let measurementMaximumSequenceLength: Int
    public let evaluationMaximumPromptTokens: Int
    public let utilityMaximumContinuationTokens: Int
    public let utilityMaximumTotalTokens: Int

    enum CodingKeys: String, CodingKey {
        case measurementMaximumSequenceLength = "measurement_max_sequence_length"
        case evaluationMaximumPromptTokens = "evaluation_max_prompt_tokens"
        case utilityMaximumContinuationTokens = "utility_max_continuation_tokens"
        case utilityMaximumTotalTokens = "utility_max_total_tokens"
    }
}

public struct ABSlayerPreflightMeasurement: Codable, Equatable, Sendable {
    public let path: String
    public let sha256: String
    public let recordCount: Int
    public let tokenizedPromptCount: Int
    public let maximumPromptTokens: Int

    enum CodingKeys: String, CodingKey {
        case path, sha256
        case recordCount = "record_count"
        case tokenizedPromptCount = "tokenized_prompt_count"
        case maximumPromptTokens = "maximum_prompt_tokens"
    }
}

public struct ABSlayerPreflightEvaluation: Codable, Equatable, Sendable {
    public let path: String
    public let sha256: String
    public let recordCount: Int
    public let refusalCount: Int
    public let utilityCount: Int
    public let maximumPromptTokens: Int
    public let maximumUtilityContinuationTokens: Int
    public let maximumUtilityTotalTokens: Int

    enum CodingKeys: String, CodingKey {
        case path, sha256
        case recordCount = "record_count"
        case refusalCount = "refusal_count"
        case utilityCount = "utility_count"
        case maximumPromptTokens = "maximum_prompt_tokens"
        case maximumUtilityContinuationTokens =
            "maximum_utility_continuation_tokens"
        case maximumUtilityTotalTokens = "maximum_utility_total_tokens"
    }
}

public struct ABSlayerPreflightReport: Codable, Equatable, Sendable {
    public let format: String
    public let status: String
    public let command: String
    public let tokenizer: String
    public let model: ABSlayerModelBinding
    public let limits: ABSlayerPreflightLimits
    public let measurement: ABSlayerPreflightMeasurement
    public let evaluation: ABSlayerPreflightEvaluation
}

public struct ABSlayerPreflightTokenSummary: Equatable, Sendable {
    public let measurementMaximumPromptTokens: Int
    public let evaluationMaximumPromptTokens: Int
    public let utilityMaximumContinuationTokens: Int
    public let utilityMaximumTotalTokens: Int
    public let refusalCount: Int
    public let utilityCount: Int
}

public enum ABSlayerPreflight {
    public static let evaluationMaximumPromptTokens = 512
    public static let utilityMaximumContinuationTokens = 128
    public static let utilityMaximumTotalTokens = 640

    public static func run(
        _ invocation: ABSlayerPreflightInvocation
    ) async throws -> ABSlayerPreflightReport {
        try ABSlayerAtomicReportWriter.requireAvailable(path: invocation.reportPath)
        let model = try ABSlayerProductionModelInspector.inspect(
            identifier: invocation.model)
        _ = try validateReportDestination(
            invocation.reportPath, model: model)
        let measurement = try ABSlayerBackendJSONL.loadMeasurementPairsBound(
            path: invocation.measurementPath)
        let evaluation = try ABSlayerBackendJSONL.loadEvaluationCasesBound(
            path: invocation.evaluationPath)
        let tokenizer = try await ABSlayerProductionPromptTokenizer.load(
            modelDirectory: model.canonicalPath)
        let tokens = try validateTokenBudgets(
            measurement: measurement.records,
            evaluation: evaluation.records,
            maximumSequenceLength: invocation.maximumSequenceLength,
            tokenizer: tokenizer)

        try ABSlayerBackendJSONL.validateUnchanged(
            path: invocation.measurementPath, sha256: measurement.sha256)
        try ABSlayerBackendJSONL.validateUnchanged(
            path: invocation.evaluationPath, sha256: evaluation.sha256)
        try ABSlayerProductionModelInspector.validateUnchanged(model)

        let report = ABSlayerPreflightReport(
            format: "abslayer.preflight/v1",
            status: "ok",
            command: "preflight",
            tokenizer: "mlx-huggingface-chat-template/v1",
            model: model,
            limits: ABSlayerPreflightLimits(
                measurementMaximumSequenceLength: invocation.maximumSequenceLength,
                evaluationMaximumPromptTokens: evaluationMaximumPromptTokens,
                utilityMaximumContinuationTokens: utilityMaximumContinuationTokens,
                utilityMaximumTotalTokens: utilityMaximumTotalTokens),
            measurement: ABSlayerPreflightMeasurement(
                path: standardizedPath(invocation.measurementPath),
                sha256: measurement.sha256,
                recordCount: measurement.records.count,
                tokenizedPromptCount: measurement.records.count * 2,
                maximumPromptTokens: tokens.measurementMaximumPromptTokens),
            evaluation: ABSlayerPreflightEvaluation(
                path: standardizedPath(invocation.evaluationPath),
                sha256: evaluation.sha256,
                recordCount: evaluation.records.count,
                refusalCount: tokens.refusalCount,
                utilityCount: tokens.utilityCount,
                maximumPromptTokens: tokens.evaluationMaximumPromptTokens,
                maximumUtilityContinuationTokens:
                    tokens.utilityMaximumContinuationTokens,
                maximumUtilityTotalTokens: tokens.utilityMaximumTotalTokens))
        try ABSlayerAtomicReportWriter.write(
            report, to: invocation.reportPath,
            excludingDirectory: model.canonicalPath)
        return report
    }

    public static func validateTokenBudgets(
        measurement: [ABSlayerMeasurementPair],
        evaluation: [ABSlayerEvaluationCase],
        maximumSequenceLength: Int,
        tokenizer: any ABSlayerPromptTokenizing
    ) throws -> ABSlayerPreflightTokenSummary {
        guard (1 ... 32_768).contains(maximumSequenceLength),
              !measurement.isEmpty, !evaluation.isEmpty
        else { throw ABSlayerPreflightError.invalidValidationInput }

        var measurementMaximum = 0
        for pair in measurement {
            for (channel, prompt) in [
                ("contrast", pair.contrast), ("control", pair.control),
            ] {
                let promptTokens = try tokenizer.userPromptTokens(prompt)
                guard !promptTokens.isEmpty else {
                    throw ABSlayerPreflightError.emptyPromptTokenization(
                        name: pair.name, channel: channel)
                }
                guard promptTokens.count <= maximumSequenceLength else {
                    throw ABSlayerPreflightError.measurementPromptTooLong(
                        name: pair.name, channel: channel,
                        tokenCount: promptTokens.count,
                        maximum: maximumSequenceLength)
                }
                measurementMaximum = max(measurementMaximum, promptTokens.count)
            }
        }

        var evaluationMaximum = 0
        var continuationMaximum = 0
        var totalMaximum = 0
        var refusalCount = 0
        var utilityCount = 0
        for item in evaluation {
            let promptTokens = try tokenizer.userPromptTokens(item.prompt)
            guard !promptTokens.isEmpty else {
                throw ABSlayerPreflightError.emptyPromptTokenization(
                    name: item.name, channel: item.kind.rawValue)
            }
            guard promptTokens.count <= evaluationMaximumPromptTokens else {
                throw ABSlayerPreflightError.evaluationPromptTooLong(
                    name: item.name, tokenCount: promptTokens.count,
                    maximum: evaluationMaximumPromptTokens)
            }
            evaluationMaximum = max(evaluationMaximum, promptTokens.count)
            switch item.kind {
            case .refusal:
                refusalCount += 1
            case .utility:
                utilityCount += 1
                guard let reference = item.reference else {
                    throw ABSlayerPreflightError.invalidUtilityReference(item.name)
                }
                let continuation: [Int]
                do {
                    continuation = try ABSlayerReferenceTokenization.continuation(
                        promptTokens: promptTokens, prompt: item.prompt,
                        reference: reference, tokenizer: tokenizer)
                } catch {
                    throw ABSlayerPreflightError.invalidUtilityReference(item.name)
                }
                guard (1 ... utilityMaximumContinuationTokens)
                    .contains(continuation.count)
                else {
                    throw ABSlayerPreflightError.utilityContinuationOutOfRange(
                        name: item.name, tokenCount: continuation.count,
                        maximum: utilityMaximumContinuationTokens)
                }
                let total = promptTokens.count + continuation.count
                guard total <= utilityMaximumTotalTokens else {
                    throw ABSlayerPreflightError.utilityTotalTooLong(
                        name: item.name, tokenCount: total,
                        maximum: utilityMaximumTotalTokens)
                }
                continuationMaximum = max(continuationMaximum, continuation.count)
                totalMaximum = max(totalMaximum, total)
            }
        }
        guard refusalCount > 0, utilityCount > 0 else {
            throw ABSlayerPreflightError.invalidValidationInput
        }
        return ABSlayerPreflightTokenSummary(
            measurementMaximumPromptTokens: measurementMaximum,
            evaluationMaximumPromptTokens: evaluationMaximum,
            utilityMaximumContinuationTokens: continuationMaximum,
            utilityMaximumTotalTokens: totalMaximum,
            refusalCount: refusalCount,
            utilityCount: utilityCount)
    }

    public static func validateReportDestination(
        _ path: String, model: ABSlayerModelBinding
    ) throws -> String {
        try validateReportDestination(
            path, modelDirectory: model.canonicalPath)
    }

    static func validateReportDestination(
        _ path: String, modelDirectory: String
    ) throws -> String {
        let reportPath = standardizedPath(path)
        let resolvedReportPath = resolvedPathIncludingMissingTail(reportPath)
        let resolvedModelPath = resolvedPathIncludingMissingTail(
            modelDirectory)
        guard !isSameOrDescendant(
            resolvedReportPath, of: resolvedModelPath)
        else { throw ABSlayerPreflightError.outputInsideModel(reportPath) }
        return reportPath
    }

    private static func standardizedPath(_ path: String) -> String {
        URL(fileURLWithPath: path).standardizedFileURL.path
    }

    /// Resolves every existing path component, then restores a possibly missing
    /// suffix. This catches an ancestor alias even when the report itself (or
    /// one of its trailing directories) does not exist yet.
    private static func resolvedPathIncludingMissingTail(_ path: String) -> String {
        var existing = URL(fileURLWithPath: path).standardizedFileURL
        var missingComponents = [String]()
        while existing.path != "/",
              !FileManager.default.fileExists(atPath: existing.path)
        {
            missingComponents.append(existing.lastPathComponent)
            existing.deleteLastPathComponent()
        }
        var resolved = existing.resolvingSymlinksInPath().standardizedFileURL
        for component in missingComponents.reversed() {
            resolved.appendPathComponent(component)
        }
        return resolved.standardizedFileURL.path
    }

    private static func isSameOrDescendant(
        _ candidate: String, of directory: String
    ) -> Bool {
        candidate == directory
            || candidate.hasPrefix(directory == "/" ? "/" : directory + "/")
    }
}

/// Same-directory hard-link publication is atomic and has no replacement
/// semantics: `link(2)` fails with EEXIST if any destination entry appears.
public enum ABSlayerAtomicReportWriter {
    public static func requireAvailable(path: String) throws {
        let destination = URL(fileURLWithPath: path).standardizedFileURL
        guard !ABSlayerFileSystem.pathExistsWithoutFollowingSymlink(destination.path)
        else { throw ABSlayerPreflightError.outputExists(destination.path) }
    }

    public static func write<T: Encodable>(
        _ value: T, to path: String,
        excludingDirectory: String? = nil
    ) throws {
        let destination = URL(fileURLWithPath: path).standardizedFileURL
        try requireAvailable(path: destination.path)
        try validateExclusion(
            destination: destination.path,
            excludingDirectory: excludingDirectory)
        let parent = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: parent, withIntermediateDirectories: true)
        guard ABSlayerFileSystem.isDirectoryWithoutFollowingSymlink(parent.path) else {
            throw ABSlayerPreflightError.invalidReportParent(parent.path)
        }
        let temporary = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).staging-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var data = try encoder.encode(value)
        data.append(0x0a)
        try data.write(to: temporary, options: .withoutOverwriting)
        guard let stagedBefore = ABSlayerFileSystem.regularFileIdentity(
            temporary.path)
        else { throw ABSlayerPreflightError.invalidTemporaryReport(temporary.path) }

        // Re-resolve after creating the parent and immediately before publish;
        // this closes ancestor aliases introduced since the initial preflight
        // check without weakening no-overwrite behavior.
        try validateExclusion(
            destination: destination.path,
            excludingDirectory: excludingDirectory)

        let result = temporary.path.withCString { source in
            destination.path.withCString { target in
                #if canImport(Darwin)
                Darwin.link(source, target)
                #elseif canImport(Glibc)
                Glibc.link(source, target)
                #else
                -1
                #endif
            }
        }
        let failure = errno
        guard result == 0 else {
            if failure == EEXIST {
                throw ABSlayerPreflightError.outputExists(destination.path)
            }
            throw ABSlayerPreflightError.reportPublicationFailed(
                path: destination.path, code: failure)
        }
        try verifyPublishedReport(
            destination: destination,
            temporary: temporary,
            stagedBefore: stagedBefore,
            expectedData: data,
            excludingDirectory: excludingDirectory)
    }

    /// Verifies the just-published hard link and retracts only the directory
    /// entry that still names our staged inode if any post-publication check
    /// fails. A concurrently replaced destination is deliberately left alone.
    static func verifyPublishedReport(
        destination: URL,
        temporary: URL,
        stagedBefore: ABSlayerFileSystem.RegularFileIdentity,
        expectedData: Data,
        excludingDirectory: String?
    ) throws {
        do {
            guard let stagedAfter = ABSlayerFileSystem.regularFileIdentity(
                      temporary.path),
                  let publishedBefore = ABSlayerFileSystem.regularFileIdentity(
                      destination.path),
                  sameFile(stagedBefore, stagedAfter),
                  sameFile(stagedAfter, publishedBefore),
                  let publishedData = try? Data(
                      contentsOf: destination, options: [.mappedIfSafe]),
                  publishedData == expectedData,
                  ABSlayerFileSystem.regularFileIdentity(destination.path)
                      == publishedBefore,
                  ABSlayerFileSystem.regularFileIdentity(temporary.path)
                      == stagedAfter
            else {
                throw ABSlayerPreflightError.invalidPublishedReport(
                    destination.path)
            }
            try validateExclusion(
                destination: destination.path,
                excludingDirectory: excludingDirectory)
        } catch {
            removePublishedDestinationIfOwned(
                destination.path, stagedIdentity: stagedBefore)
            throw error
        }
    }

    private static func validateExclusion(
        destination: String, excludingDirectory: String?
    ) throws {
        guard let excludingDirectory else { return }
        _ = try ABSlayerPreflight.validateReportDestination(
            destination, modelDirectory: excludingDirectory)
    }

    private static func sameFile(
        _ left: ABSlayerFileSystem.RegularFileIdentity,
        _ right: ABSlayerFileSystem.RegularFileIdentity
    ) -> Bool {
        left.device == right.device
            && left.inode == right.inode
            && left.size == right.size
    }

    private static func removePublishedDestinationIfOwned(
        _ path: String,
        stagedIdentity: ABSlayerFileSystem.RegularFileIdentity
    ) {
        guard let current = ABSlayerFileSystem.regularFileIdentity(path),
              current.device == stagedIdentity.device,
              current.inode == stagedIdentity.inode
        else { return }
        _ = path.withCString { target in
            #if canImport(Darwin)
            Darwin.unlink(target)
            #elseif canImport(Glibc)
            Glibc.unlink(target)
            #else
            -1
            #endif
        }
    }
}

public enum ABSlayerPreflightError: LocalizedError, Equatable {
    case invalidValidationInput
    case emptyPromptTokenization(name: String, channel: String)
    case measurementPromptTooLong(
        name: String, channel: String, tokenCount: Int, maximum: Int)
    case evaluationPromptTooLong(name: String, tokenCount: Int, maximum: Int)
    case invalidUtilityReference(String)
    case utilityContinuationOutOfRange(
        name: String, tokenCount: Int, maximum: Int)
    case utilityTotalTooLong(name: String, tokenCount: Int, maximum: Int)
    case outputExists(String)
    case outputInsideModel(String)
    case invalidReportParent(String)
    case invalidTemporaryReport(String)
    case reportPublicationFailed(path: String, code: Int32)
    case invalidPublishedReport(String)

    public var errorDescription: String? {
        switch self {
        case .invalidValidationInput:
            "Preflight token validation input is empty or out of range."
        case .emptyPromptTokenization(let name, let channel):
            "\(name) \(channel) prompt tokenized to an empty sequence."
        case .measurementPromptTooLong(
            let name, let channel, let count, let maximum):
            "\(name) \(channel) prompt has \(count) tokens; measurement maximum is \(maximum)."
        case .evaluationPromptTooLong(let name, let count, let maximum):
            "\(name) prompt has \(count) tokens; evaluation maximum is \(maximum)."
        case .invalidUtilityReference(let name):
            "\(name) utility reference is not a prefix-compatible continuation."
        case .utilityContinuationOutOfRange(let name, let count, let maximum):
            "\(name) utility continuation has \(count) tokens; required range is 1...\(maximum)."
        case .utilityTotalTooLong(let name, let count, let maximum):
            "\(name) prompt plus utility continuation has \(count) tokens; maximum is \(maximum)."
        case .outputExists(let path):
            "Preflight report already exists and will not be overwritten: \(path)"
        case .outputInsideModel(let path):
            "Preflight report must be outside the bound model directory: \(path)"
        case .invalidReportParent(let path):
            "Preflight report parent must be a regular non-symlink directory: \(path)"
        case .invalidTemporaryReport(let path):
            "Preflight temporary report is not a regular file: \(path)"
        case .reportPublicationFailed(let path, let code):
            "Could not atomically publish preflight report \(path) (errno \(code))."
        case .invalidPublishedReport(let path):
            "Published preflight report is not a regular file: \(path)"
        }
    }
}
