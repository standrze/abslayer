#if canImport(CryptoKit)
import CryptoKit
#elseif canImport(Crypto)
import Crypto
#endif
import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Generation state is separate from the response text so an actually empty
/// model response is distinguishable from work that has not run yet.
public enum ScreeningReviewGenerationStatus: String, Codable, Sendable {
    case pending
    case complete
}

/// A review label is never inferred from refusal substrings. `unjudged` is an
/// explicit, fail-closed placeholder written by `collect-screening`.
public enum ScreeningReviewJudgmentStatus: String, Codable, Sendable {
    case unjudged
    case reviewed
}

/// Deliberately has no marker/heuristic case. Marker matches may help a reviewer
/// find examples, but cannot become semantic certification through this schema.
public enum ScreeningReviewJudgmentMethod: String, Codable, Sendable {
    case unjudged
    case human
    case semanticModel = "semantic_model"
}

public struct ScreeningReviewJudgment: Codable, Equatable, Sendable {
    public var status: ScreeningReviewJudgmentStatus
    public var outcome: BehaviorOutcome
    public var confidence: Double
    public var evidence: [String]
    /// Optional for backward compatibility with review artifacts written
    /// before semantic-judge audit evidence was captured.
    public var semanticJudgeEvidence: SemanticJudgeEvidence?

    public init(
        status: ScreeningReviewJudgmentStatus,
        outcome: BehaviorOutcome,
        confidence: Double,
        evidence: [String],
        semanticJudgeEvidence: SemanticJudgeEvidence? = nil
    ) {
        self.status = status
        self.outcome = outcome
        self.confidence = confidence
        self.evidence = evidence
        self.semanticJudgeEvidence = semanticJudgeEvidence
    }

    public static let unjudged = ScreeningReviewJudgment(
        status: .unjudged,
        outcome: .unclassified,
        confidence: 0,
        evidence: [
            "UNJUDGED: assign a human or semantic-model verdict after reviewing this exact response"
        ])
}

public struct ScreeningReviewChannel: Codable, Equatable, Sendable {
    public let prompt: String
    public let promptSha256: String
    public var generationStatus: ScreeningReviewGenerationStatus
    public var response: String
    public var responseSha256: String
    public var judgment: ScreeningReviewJudgment

    public init(
        prompt: String,
        promptSha256: String,
        generationStatus: ScreeningReviewGenerationStatus = .pending,
        response: String = "",
        responseSha256: String = "",
        judgment: ScreeningReviewJudgment = .unjudged
    ) {
        self.prompt = prompt
        self.promptSha256 = promptSha256
        self.generationStatus = generationStatus
        self.response = response
        self.responseSha256 = responseSha256
        self.judgment = judgment
    }
}

public struct ScreeningReviewRecord: Codable, Equatable, Sendable {
    public let candidate: PromptPair
    public let candidateSha256: String
    public var contrast: ScreeningReviewChannel
    public var control: ScreeningReviewChannel
}

public struct ScreeningReviewGenerationConfiguration: Codable, Equatable, Sendable {
    public let maximumTokens: Int
    public let temperature: Double
    public let earlyStopHeuristicsEnabled: Bool

    public init(maximumTokens: Int) {
        self.maximumTokens = maximumTokens
        temperature = 0
        // Collection retains the full capped response. Refusal markers are not
        // used to truncate output or create a label.
        earlyStopHeuristicsEnabled = false
    }
}

/// Model-conditioned, resumable source of truth for semantic pair screening.
/// The response digests bind later judgments to the exact generated text.
public struct ScreeningReviewArtifact: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 1
    public static let currentArtifactRole =
        "target-model-screening-review"
    public static let unjudgedReviewerIdentifier = "UNJUDGED"

    public let schemaVersion: Int
    public let artifactRole: String
    public let targetModel: String
    public let targetModelMetadataSha256: String
    public let sourceCatalog: String
    public let sourceCatalogSha256: String
    public let generationTool: String
    public let generation: ScreeningReviewGenerationConfiguration
    public let markerHeuristicsCanCertifySemanticCompliance: Bool
    public let semanticEligibilityRule: String
    public var judgmentMethod: ScreeningReviewJudgmentMethod
    public var reviewerIdentifier: String
    public var records: [ScreeningReviewRecord]

    public var completedChannelCount: Int {
        records.reduce(0) { partial, record in
            partial
                + (record.contrast.generationStatus == .complete ? 1 : 0)
                + (record.control.generationStatus == .complete ? 1 : 0)
        }
    }

    public var totalChannelCount: Int { records.count * 2 }

    public var reviewedChannelCount: Int {
        records.reduce(0) { partial, record in
            partial
                + (record.contrast.judgment.status == .reviewed ? 1 : 0)
                + (record.control.judgment.status == .reviewed ? 1 : 0)
        }
    }

    public static func make(
        targetModel: String,
        targetModelMetadataSha256: String,
        sourceCatalog: String,
        sourceCatalogSha256: String,
        candidates: [PromptPair],
        maximumTokens: Int
    ) throws -> ScreeningReviewArtifact {
        guard !candidates.isEmpty else { throw ScreeningReviewError.emptyCandidates }
        guard maximumTokens > 0 else {
            throw ScreeningReviewError.invalidMaximumTokens(maximumTokens)
        }
        var names = Set<String>()
        let records = try candidates.map { candidate in
            guard names.insert(candidate.name).inserted else {
                throw ScreeningReviewError.duplicateCandidate(candidate.name)
            }
            return ScreeningReviewRecord(
                candidate: candidate,
                candidateSha256: ScreeningReviewProvenance.candidateSha256(candidate),
                contrast: ScreeningReviewChannel(
                    prompt: candidate.contrast,
                    promptSha256: ScreeningReviewProvenance.sha256(candidate.contrast)),
                control: ScreeningReviewChannel(
                    prompt: candidate.control,
                    promptSha256: ScreeningReviewProvenance.sha256(candidate.control)))
        }
        return ScreeningReviewArtifact(
            schemaVersion: currentSchemaVersion,
            artifactRole: currentArtifactRole,
            targetModel: URL(fileURLWithPath: targetModel).standardizedFileURL.path,
            targetModelMetadataSha256: targetModelMetadataSha256,
            sourceCatalog: URL(fileURLWithPath: sourceCatalog).standardizedFileURL.path,
            sourceCatalogSha256: sourceCatalogSha256,
            generationTool: "abslayer-dataset collect-screening-v1",
            generation: ScreeningReviewGenerationConfiguration(maximumTokens: maximumTokens),
            markerHeuristicsCanCertifySemanticCompliance: false,
            semanticEligibilityRule: "target base response=refusal AND authorized same-task control response=direct substantive compliance; lecture, redirect, defensive substitution, refusal, partial answer, or clarifying question fails",
            judgmentMethod: .unjudged,
            reviewerIdentifier: unjudgedReviewerIdentifier,
            records: records)
    }

    /// Validates an existing checkpoint before any generation is skipped.
    public func validateResume(
        targetModel: String,
        targetModelMetadataSha256: String,
        sourceCatalog: String,
        sourceCatalogSha256: String,
        candidates: [PromptPair],
        maximumTokens: Int
    ) throws {
        let expected = try Self.make(
            targetModel: targetModel,
            targetModelMetadataSha256: targetModelMetadataSha256,
            sourceCatalog: sourceCatalog,
            sourceCatalogSha256: sourceCatalogSha256,
            candidates: candidates,
            maximumTokens: maximumTokens)
        guard schemaVersion == expected.schemaVersion else {
            throw ScreeningReviewError.resumeMismatch("schema_version")
        }
        guard artifactRole == expected.artifactRole else {
            throw ScreeningReviewError.resumeMismatch("artifact_role")
        }
        guard self.targetModel == expected.targetModel else {
            throw ScreeningReviewError.resumeMismatch("target_model")
        }
        guard self.targetModelMetadataSha256 == expected.targetModelMetadataSha256 else {
            throw ScreeningReviewError.resumeMismatch("target_model_metadata_sha256")
        }
        guard self.sourceCatalog == expected.sourceCatalog else {
            throw ScreeningReviewError.resumeMismatch("source_catalog")
        }
        guard self.sourceCatalogSha256 == expected.sourceCatalogSha256 else {
            throw ScreeningReviewError.resumeMismatch("source_catalog_sha256")
        }
        guard generation == expected.generation else {
            throw ScreeningReviewError.resumeMismatch("generation")
        }
        guard generationTool == expected.generationTool else {
            throw ScreeningReviewError.resumeMismatch("generation_tool")
        }
        guard semanticEligibilityRule == expected.semanticEligibilityRule else {
            throw ScreeningReviewError.resumeMismatch("semantic_eligibility_rule")
        }
        guard !markerHeuristicsCanCertifySemanticCompliance else {
            throw ScreeningReviewError.markerCertificationForbidden
        }
        guard records.count == expected.records.count else {
            throw ScreeningReviewError.resumeMismatch("record_count")
        }
        for (record, expectedRecord) in zip(records, expected.records) {
            guard record.candidate == expectedRecord.candidate,
                  record.candidateSha256 == expectedRecord.candidateSha256,
                  record.contrast.prompt == expectedRecord.contrast.prompt,
                  record.contrast.promptSha256 == expectedRecord.contrast.promptSha256,
                  record.control.prompt == expectedRecord.control.prompt,
                  record.control.promptSha256 == expectedRecord.control.promptSha256
            else {
                throw ScreeningReviewError.resumeMismatch(
                    "candidate_or_prompt:\(expectedRecord.candidate.name)")
            }
            try Self.validateResponseBinding(
                record.contrast, name: record.candidate.name, channel: .contrast)
            try Self.validateResponseBinding(
                record.control, name: record.candidate.name, channel: .control)
        }
    }

    public mutating func recordResponse(
        _ response: String, name: String, channel: EvaluationChannel
    ) throws {
        guard let index = records.firstIndex(where: { $0.candidate.name == name }) else {
            throw ScreeningReviewError.unknownCandidate(name)
        }
        var value = channel == .contrast ? records[index].contrast : records[index].control
        guard value.generationStatus == .pending else {
            throw ScreeningReviewError.responseAlreadyRecorded(name: name, channel: channel)
        }
        value.response = response
        value.responseSha256 = ScreeningReviewProvenance.sha256(response)
        value.generationStatus = .complete
        // A judgment cannot predate the response it claims to assess.
        value.judgment = .unjudged
        if channel == .contrast {
            records[index].contrast = value
        } else {
            records[index].control = value
        }
    }

    /// Revalidates all provenance that can be established from the artifact
    /// alone. This is the trust boundary used before a local semantic judge sees
    /// any text and again before each returned verdict is persisted.
    public func validateStoredBindings() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw ScreeningReviewError.resumeMismatch("schema_version")
        }
        guard artifactRole == Self.currentArtifactRole else {
            throw ScreeningReviewError.resumeMismatch("artifact_role")
        }
        guard !markerHeuristicsCanCertifySemanticCompliance else {
            throw ScreeningReviewError.markerCertificationForbidden
        }
        guard !records.isEmpty else { throw ScreeningReviewError.emptyCandidates }

        var names = Set<String>()
        for record in records {
            let name = record.candidate.name
            guard names.insert(name).inserted else {
                throw ScreeningReviewError.duplicateCandidate(name)
            }
            guard record.candidateSha256
                    == ScreeningReviewProvenance.candidateSha256(record.candidate),
                  record.contrast.prompt == record.candidate.contrast,
                  record.control.prompt == record.candidate.control,
                  record.contrast.promptSha256
                    == ScreeningReviewProvenance.sha256(record.contrast.prompt),
                  record.control.promptSha256
                    == ScreeningReviewProvenance.sha256(record.control.prompt)
            else {
                throw ScreeningReviewError.resumeMismatch("candidate_or_prompt:\(name)")
            }
            for (channel, value) in [
                (EvaluationChannel.contrast, record.contrast),
                (EvaluationChannel.control, record.control),
            ] {
                try Self.validateResponseBinding(value, name: name, channel: channel)
                switch value.judgment.status {
                case .unjudged:
                    guard value.judgment == .unjudged else {
                        throw ScreeningReviewError.invalidJudgment(
                            name: name, channel: channel)
                    }
                case .reviewed:
                    guard value.generationStatus == .complete else {
                        throw ScreeningReviewError.incompleteGeneration(
                            name: name, channel: channel)
                    }
                    try Self.validateCompletedJudgment(
                        value.judgment, name: name, channel: channel)
                    if let evidence = value.judgment.semanticJudgeEvidence {
                        try Self.validateSemanticJudgeEvidence(
                            evidence,
                            prompt: value.prompt,
                            response: value.response,
                            reviewerIdentifier: reviewerIdentifier,
                            name: name,
                            channel: channel)
                    }
                }
            }
        }

        switch judgmentMethod {
        case .unjudged:
            guard reviewerIdentifier == Self.unjudgedReviewerIdentifier,
                  reviewedChannelCount == 0
            else {
                throw ScreeningReviewError.incompleteReview(
                    "unjudged review contains reviewer metadata or completed verdicts")
            }
        case .human, .semanticModel:
            let reviewer = reviewerIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !reviewer.isEmpty, reviewer != Self.unjudgedReviewerIdentifier else {
                throw ScreeningReviewError.incompleteReview("reviewer_identifier is unjudged")
            }
        }
    }

    /// Starts or resumes one judge-model identity. Mixing verdicts from another
    /// model (or converting a human review in place) is rejected.
    public mutating func beginSemanticReview(reviewerIdentifier: String) throws {
        try validateStoredBindings()
        let reviewer = reviewerIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !reviewer.isEmpty, reviewer != Self.unjudgedReviewerIdentifier else {
            throw ScreeningReviewError.incompleteReview("semantic reviewer is empty")
        }
        switch judgmentMethod {
        case .unjudged:
            judgmentMethod = .semanticModel
            self.reviewerIdentifier = reviewer
        case .semanticModel:
            guard self.reviewerIdentifier == reviewer else {
                throw ScreeningReviewError.semanticReviewerMismatch(
                    expected: self.reviewerIdentifier, received: reviewer)
            }
        case .human:
            throw ScreeningReviewError.incompatibleJudgmentMethod(judgmentMethod)
        }
    }

    /// Returns only completed, still-unjudged channels. Pending response slots
    /// are deliberately absent and therefore cannot acquire a verdict.
    public func semanticJudgeWorkItems() throws -> [SemanticJudgeWorkItem] {
        try validateStoredBindings()
        var items = [SemanticJudgeWorkItem]()
        items.reserveCapacity(totalChannelCount - reviewedChannelCount)
        for record in records {
            let result = PromptResult(
                name: record.candidate.name,
                contrastResponse: record.contrast.response,
                controlResponse: record.control.response,
                category: record.candidate.category,
                contrastPrompt: record.contrast.prompt,
                controlPrompt: record.control.prompt)
            if record.contrast.generationStatus == .complete,
               record.contrast.judgment.status == .unjudged
            {
                items.append(SemanticJudgeWorkItem(result: result, channel: .contrast))
            }
            if record.control.generationStatus == .complete,
               record.control.judgment.status == .unjudged
            {
                items.append(SemanticJudgeWorkItem(result: result, channel: .control))
            }
        }
        return items
    }

    public mutating func recordSemanticJudgment(
        _ judgment: RecordedOutcomeJudgment,
        reviewerIdentifier: String
    ) throws {
        try validateStoredBindings()
        guard judgmentMethod == .semanticModel else {
            throw ScreeningReviewError.incompatibleJudgmentMethod(judgmentMethod)
        }
        guard self.reviewerIdentifier == reviewerIdentifier else {
            throw ScreeningReviewError.semanticReviewerMismatch(
                expected: self.reviewerIdentifier, received: reviewerIdentifier)
        }
        guard let index = records.firstIndex(where: {
            $0.candidate.name == judgment.name
        }) else { throw ScreeningReviewError.unknownCandidate(judgment.name) }

        var value = judgment.channel == .contrast
            ? records[index].contrast : records[index].control
        guard value.generationStatus == .complete else {
            throw ScreeningReviewError.incompleteGeneration(
                name: judgment.name, channel: judgment.channel)
        }
        guard value.judgment.status == .unjudged else {
            throw ScreeningReviewError.judgmentAlreadyRecorded(
                name: judgment.name, channel: judgment.channel)
        }
        let reviewed = ScreeningReviewJudgment(
            status: .reviewed,
            outcome: judgment.classification.outcome,
            confidence: judgment.classification.confidence,
            evidence: judgment.classification.evidence,
            semanticJudgeEvidence: judgment.semanticJudgeEvidence)
        try Self.validateCompletedJudgment(
            reviewed, name: judgment.name, channel: judgment.channel)
        if let evidence = reviewed.semanticJudgeEvidence {
            try Self.validateSemanticJudgeEvidence(
                evidence,
                prompt: value.prompt,
                response: value.response,
                reviewerIdentifier: reviewerIdentifier,
                name: judgment.name,
                channel: judgment.channel)
        }
        value.judgment = reviewed
        if judgment.channel == .contrast {
            records[index].contrast = value
        } else {
            records[index].control = value
        }
        try validateStoredBindings()
    }

    /// Fails closed unless every response is bound and every label is an
    /// explicit human or semantic-model judgment.
    public func screeningPayload(
        targetModel: String,
        targetModelMetadataSha256: String,
        sourceCatalog: String,
        sourceCatalogSha256: String,
        candidates: [PromptPair]
    ) throws -> ScreeningReviewPayload {
        try validateStoredBindings()
        try validateResume(
            targetModel: targetModel,
            targetModelMetadataSha256: targetModelMetadataSha256,
            sourceCatalog: sourceCatalog,
            sourceCatalogSha256: sourceCatalogSha256,
            candidates: candidates,
            maximumTokens: generation.maximumTokens)
        guard judgmentMethod != .unjudged else {
            throw ScreeningReviewError.incompleteReview("judgment_method is unjudged")
        }
        let reviewer = reviewerIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !reviewer.isEmpty, reviewer != Self.unjudgedReviewerIdentifier else {
            throw ScreeningReviewError.incompleteReview("reviewer_identifier is unjudged")
        }
        let normalizedReviewer = reviewer.lowercased()
        guard !["nonsemantic", "non-semantic", "substring", "marker", "heuristic"]
            .contains(where: normalizedReviewer.contains)
        else { throw ScreeningReviewError.markerCertificationForbidden }

        var responses = [PromptResult]()
        var judgments = [RecordedOutcomeJudgment]()
        responses.reserveCapacity(records.count)
        judgments.reserveCapacity(records.count * 2)
        for record in records {
            for (channel, value) in [
                (EvaluationChannel.contrast, record.contrast),
                (EvaluationChannel.control, record.control),
            ] {
                guard value.generationStatus == .complete else {
                    throw ScreeningReviewError.incompleteGeneration(
                        name: record.candidate.name, channel: channel)
                }
                try Self.validateResponseBinding(
                    value, name: record.candidate.name, channel: channel)
                try Self.validateCompletedJudgment(
                    value.judgment, name: record.candidate.name, channel: channel)
                judgments.append(RecordedOutcomeJudgment(
                    name: record.candidate.name,
                    channel: channel,
                    classification: OutcomeClassification(
                        outcome: value.judgment.outcome,
                        confidence: value.judgment.confidence,
                        evidence: value.judgment.evidence),
                    semanticJudgeEvidence: value.judgment.semanticJudgeEvidence))
            }
            responses.append(PromptResult(
                name: record.candidate.name,
                contrastResponse: record.contrast.response,
                controlResponse: record.control.response,
                category: record.candidate.category,
                contrastPrompt: record.contrast.prompt,
                controlPrompt: record.control.prompt))
        }
        return ScreeningReviewPayload(
            classifierIdentifier: "\(judgmentMethod.rawValue):\(reviewer)",
            responses: responses,
            judgments: judgments)
    }

    private static func validateResponseBinding(
        _ value: ScreeningReviewChannel,
        name: String,
        channel: EvaluationChannel
    ) throws {
        switch value.generationStatus {
        case .pending:
            guard value.response.isEmpty,
                  value.responseSha256.isEmpty,
                  value.judgment.status == .unjudged,
                  value.judgment.outcome == .unclassified
            else {
                throw ScreeningReviewError.invalidResponseBinding(name: name, channel: channel)
            }
        case .complete:
            guard value.responseSha256 == ScreeningReviewProvenance.sha256(value.response) else {
                throw ScreeningReviewError.invalidResponseBinding(name: name, channel: channel)
            }
        }
    }

    private static func validateCompletedJudgment(
        _ judgment: ScreeningReviewJudgment,
        name: String,
        channel: EvaluationChannel
    ) throws {
        guard judgment.status == .reviewed, judgment.outcome != .unclassified else {
            throw ScreeningReviewError.incompleteJudgment(name: name, channel: channel)
        }
        guard judgment.confidence.isFinite,
              judgment.confidence > 0,
              judgment.confidence <= 1,
              judgment.evidence.contains(where: {
                  !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
              }),
              !judgment.evidence.contains(where: {
                  $0.localizedCaseInsensitiveContains("UNJUDGED")
              })
        else { throw ScreeningReviewError.invalidJudgment(name: name, channel: channel) }
    }

    private static func validateSemanticJudgeEvidence(
        _ evidence: SemanticJudgeEvidence,
        prompt: String,
        response: String,
        reviewerIdentifier: String,
        name: String,
        channel: EvaluationChannel
    ) throws {
        let canonicalPath = URL(fileURLWithPath: evidence.judgeModelPath)
            .standardizedFileURL.path
        let judgePrompt = SemanticJudgeEngine.classificationPrompt(
            request: prompt, response: response)
        guard evidence.judgeModelIdentifier == reviewerIdentifier,
              !evidence.judgeModelIdentifier.isEmpty,
              evidence.judgeModelPath == canonicalPath,
              !canonicalPath.isEmpty,
              evidence.promptVersion == SemanticJudgeEngine.promptVersion,
              evidence.rawOutputSha256
                == ScreeningReviewProvenance.sha256(evidence.rawOutput),
              evidence.judgePromptSha256
                == ScreeningReviewProvenance.sha256(judgePrompt),
              evidence.requestSha256 == ScreeningReviewProvenance.sha256(prompt),
              evidence.responseSha256 == ScreeningReviewProvenance.sha256(response),
              evidence.maximumTokens == SemanticJudgeEngine.maximumTokens,
              evidence.temperature == SemanticJudgeEngine.temperature
        else { throw ScreeningReviewError.invalidJudgment(name: name, channel: channel) }
    }
}

public struct ScreeningReviewPayload: Sendable {
    public let classifierIdentifier: String
    public let responses: [PromptResult]
    public let judgments: [RecordedOutcomeJudgment]
}

public enum ScreeningReviewProvenance {
    public static func sha256(_ value: String) -> String {
        sha256(Data(value.utf8))
    }

    public static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func fileSHA256(_ path: String) throws -> String {
        try sha256(Data(contentsOf: URL(fileURLWithPath: path).standardizedFileURL))
    }

    /// Hashes the small model identity/configuration files, never multi-GB
    /// weight shards. The canonical model path remains part of the condition.
    public static func modelMetadataSha256(_ directory: String) throws -> String {
        let url = URL(fileURLWithPath: directory).standardizedFileURL
        let names = try FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil)
            .filter {
                let name = $0.lastPathComponent
                return $0.pathExtension.lowercased() == "json"
                    || name == "tokenizer.model" || name == "tokenizer_config"
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !names.isEmpty else {
            throw ScreeningReviewError.missingModelMetadata(url.path)
        }
        var aggregate = Data()
        for file in names {
            let name = Data(file.lastPathComponent.utf8)
            let data = try Data(contentsOf: file)
            aggregate.append(Data("\(name.count):".utf8))
            aggregate.append(name)
            aggregate.append(Data("\(data.count):".utf8))
            aggregate.append(data)
        }
        return sha256(aggregate)
    }

    /// Stable reviewer identity bound to the judge checkpoint metadata rather
    /// than to a user-editable display name.
    public static func semanticReviewerIdentifier(_ modelDirectory: String) throws -> String {
        let path = URL(fileURLWithPath: modelDirectory).standardizedFileURL.path
        let metadata = try modelMetadataSha256(path)
        let identity = sha256("\(path)\n\(metadata)")
        return "semantic-model-identity-sha256:\(identity)"
    }

    public static func candidateSha256(_ candidate: PromptPair) -> String {
        let fields = [
            candidate.name, candidate.contrast, candidate.control,
            candidate.category ?? "", candidate.source ?? "",
            candidate.controlSource ?? "", candidate.split ?? "",
            candidate.requestType ?? "", candidate.controlReferenceResponse ?? "",
        ]
        return sha256(fields.map { "\(Data($0.utf8).count):\($0)" }.joined())
    }
}

/// Generates every pending channel with deterministic decoding and checkpoints
/// after each channel. An interrupted invocation can safely resume without
/// regenerating text that may already have been semantically reviewed.
public enum ScreeningReviewCollectionEngine {
    public static func collect(
        artifact input: ScreeningReviewArtifact,
        checkpoint: (ScreeningReviewArtifact) throws -> Void,
        progress: ((String) -> Void)? = nil
    ) async throws -> ScreeningReviewArtifact {
        var artifact = input
        guard artifact.completedChannelCount < artifact.totalChannelCount else {
            progress?("screening response collection already complete")
            return artifact
        }
        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(
                directory: URL(fileURLWithPath: artifact.targetModel),
                extraEOSTokens: ["<end_of_turn>"]))
        let parameters = GenerateParameters(
            maxTokens: artifact.generation.maximumTokens,
            temperature: Float(artifact.generation.temperature))

        for index in artifact.records.indices {
            let name = artifact.records[index].candidate.name
            if artifact.records[index].contrast.generationStatus == .pending {
                let response = try await response(
                    container: container,
                    prompt: artifact.records[index].contrast.prompt,
                    parameters: parameters)
                try artifact.recordResponse(response, name: name, channel: .contrast)
                try checkpoint(artifact)
                progress?(
                    "collected \(artifact.completedChannelCount)/\(artifact.totalChannelCount) channels")
            }
            if artifact.records[index].control.generationStatus == .pending {
                let response = try await response(
                    container: container,
                    prompt: artifact.records[index].control.prompt,
                    parameters: parameters)
                try artifact.recordResponse(response, name: name, channel: .control)
                try checkpoint(artifact)
                progress?(
                    "collected \(artifact.completedChannelCount)/\(artifact.totalChannelCount) channels")
            }
        }
        return artifact
    }

    private static func response(
        container: ModelContainer,
        prompt: String,
        parameters: GenerateParameters
    ) async throws -> String {
        let session = ChatSession(container, generateParameters: parameters)
        var output = ""
        for try await chunk in session.streamResponse(to: prompt) {
            output += chunk
        }
        return output
    }
}

/// Applies the local semantic judge to only the unjudged, response-complete
/// channels of a review artifact. Each channel verdict is checkpointed before
/// another model generation begins, so rerunning after interruption resumes at
/// the first genuinely unjudged channel.
public enum ScreeningReviewSemanticJudgeEngine {
    public static func review(
        artifact input: ScreeningReviewArtifact,
        judgeModelDirectory: String,
        checkpoint: @escaping (ScreeningReviewArtifact) throws -> Void,
        progress: ((String) -> Void)? = nil
    ) async throws -> ScreeningReviewArtifact {
        let reviewer = try ScreeningReviewProvenance.semanticReviewerIdentifier(
            judgeModelDirectory)
        var artifact = input
        try artifact.beginSemanticReview(reviewerIdentifier: reviewer)
        // Persist the judge identity before its first verdict. If model loading
        // fails, the response-bound artifact remains safely resumable.
        try checkpoint(artifact)
        let workItems = try artifact.semanticJudgeWorkItems()
        guard !workItems.isEmpty else {
            progress?("screening semantic review has no completed unjudged channels")
            return artifact
        }

        _ = try await SemanticJudgeEngine.judge(
            modelDirectory: judgeModelDirectory,
            workItems: workItems,
            onJudgment: { judgment in
                try artifact.recordSemanticJudgment(
                    judgment, reviewerIdentifier: reviewer)
                try checkpoint(artifact)
            },
            progress: { completed, total in
                progress?("semantically judged \(completed)/\(total) pending channels")
            })
        try artifact.validateStoredBindings()
        return artifact
    }
}

public enum ScreeningReviewError: LocalizedError, Equatable {
    case emptyCandidates
    case invalidMaximumTokens(Int)
    case duplicateCandidate(String)
    case unknownCandidate(String)
    case responseAlreadyRecorded(name: String, channel: EvaluationChannel)
    case resumeMismatch(String)
    case invalidResponseBinding(name: String, channel: EvaluationChannel)
    case incompleteGeneration(name: String, channel: EvaluationChannel)
    case incompleteReview(String)
    case incompleteJudgment(name: String, channel: EvaluationChannel)
    case invalidJudgment(name: String, channel: EvaluationChannel)
    case judgmentAlreadyRecorded(name: String, channel: EvaluationChannel)
    case semanticReviewerMismatch(expected: String, received: String)
    case incompatibleJudgmentMethod(ScreeningReviewJudgmentMethod)
    case markerCertificationForbidden
    case missingModelMetadata(String)
    case reviewArtifactRequiredForCounterfactualCandidates

    public var errorDescription: String? {
        switch self {
        case .emptyCandidates:
            "No screening candidates were provided."
        case .invalidMaximumTokens(let value):
            "The screening response token cap must be positive; received \(value)."
        case .duplicateCandidate(let name):
            "Screening candidates contain duplicate name '\(name)'."
        case .unknownCandidate(let name):
            "The screening review has no candidate named '\(name)'."
        case .responseAlreadyRecorded(let name, let channel):
            "A \(channel.rawValue) response is already recorded for '\(name)'."
        case .resumeMismatch(let field):
            "The existing screening review cannot be resumed because '\(field)' changed."
        case .invalidResponseBinding(let name, let channel):
            "The \(channel.rawValue) response digest/status is invalid for '\(name)'."
        case .incompleteGeneration(let name, let channel):
            "The \(channel.rawValue) response has not been collected for '\(name)'."
        case .incompleteReview(let detail):
            "Semantic screening review is incomplete: \(detail)."
        case .incompleteJudgment(let name, let channel):
            "The \(channel.rawValue) response for '\(name)' is still explicitly unjudged."
        case .invalidJudgment(let name, let channel):
            "The reviewed \(channel.rawValue) judgment for '\(name)' lacks valid confidence/evidence."
        case .judgmentAlreadyRecorded(let name, let channel):
            "A semantic judgment is already recorded for '\(name)' \(channel.rawValue)."
        case .semanticReviewerMismatch(let expected, let received):
            "The screening review is bound to semantic reviewer '\(expected)', not '\(received)'."
        case .incompatibleJudgmentMethod(let method):
            "Cannot apply a semantic-model reviewer to a \(method.rawValue) screening review."
        case .markerCertificationForbidden:
            "Marker/substring heuristics cannot certify semantic screening. Use completed human or semantic-model judgments."
        case .missingModelMetadata(let directory):
            "No model identity/configuration metadata was found in '\(directory)'."
        case .reviewArtifactRequiredForCounterfactualCandidates:
            "Counterfactual controls require a model-conditioned collect-screening review artifact; detached legacy judgments are not accepted."
        }
    }
}
