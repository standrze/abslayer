import Foundation
import Testing
@testable import ProbeCore

@Test func screeningReviewStartsExplicitlyUnjudgedAndNonSemantic() throws {
    let artifact = try makeReview()

    #expect(artifact.judgmentMethod == .unjudged)
    #expect(artifact.reviewerIdentifier == ScreeningReviewArtifact.unjudgedReviewerIdentifier)
    #expect(!artifact.markerHeuristicsCanCertifySemanticCompliance)
    #expect(artifact.completedChannelCount == 0)
    #expect(artifact.totalChannelCount == 2)
    #expect(artifact.records[0].contrast.generationStatus == .pending)
    #expect(artifact.records[0].contrast.judgment.status == .unjudged)
    #expect(artifact.records[0].contrast.judgment.outcome == .unclassified)

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let json = String(decoding: try encoder.encode(artifact), as: UTF8.self)
    #expect(json.contains("\"judgment_method\":\"unjudged\""))
    #expect(json.contains("\"status\":\"unjudged\""))
    #expect(json.contains("\"outcome\":\"unclassified\""))
    #expect(json.contains("\"response\":\"\""))
    #expect(json.contains("\"marker_heuristics_can_certify_semantic_compliance\":false"))
}

@Test func screeningReviewCLISnakeCaseJSONRoundTrips() throws {
    let artifact = try generatedReview()
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(artifact)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase

    let restored = try decoder.decode(ScreeningReviewArtifact.self, from: data)
    #expect(restored == artifact)
}

@Test func screeningReviewResumePreservesResponsesAndRejectsChangedInputs() throws {
    var artifact = try makeReview()
    try artifact.recordResponse("base response", name: "dev-1", channel: .contrast)

    try artifact.validateResume(
        targetModel: "/models/target", targetModelMetadataSha256: "model-hash",
        sourceCatalog: "/data/candidates.json", sourceCatalogSha256: "source-hash",
        candidates: candidates, maximumTokens: 192)
    #expect(artifact.completedChannelCount == 1)
    #expect(artifact.records[0].contrast.response == "base response")
    #expect(artifact.records[0].control.generationStatus == .pending)

    #expect(throws: ScreeningReviewError.self) {
        try artifact.validateResume(
            targetModel: "/models/a-different-target",
            targetModelMetadataSha256: "model-hash",
            sourceCatalog: "/data/candidates.json",
            sourceCatalogSha256: "source-hash",
            candidates: candidates, maximumTokens: 192)
    }
}

@Test func semanticScreeningRejectsUnjudgedOrMarkerCertifiedReview() throws {
    var artifact = try generatedReview()

    #expect(throws: ScreeningReviewError.self) {
        try artifact.screeningPayload(
            targetModel: "/models/target", targetModelMetadataSha256: "model-hash",
            sourceCatalog: "/data/candidates.json", sourceCatalogSha256: "source-hash",
            candidates: candidates)
    }

    artifact.judgmentMethod = .human
    artifact.reviewerIdentifier = "explicit-marker-fallback-v1-nonsemantic"
    artifact.records[0].contrast.judgment = reviewed(.refusal)
    artifact.records[0].control.judgment = reviewed(.compliance)
    #expect(throws: ScreeningReviewError.self) {
        try artifact.screeningPayload(
            targetModel: "/models/target", targetModelMetadataSha256: "model-hash",
            sourceCatalog: "/data/candidates.json", sourceCatalogSha256: "source-hash",
            candidates: candidates)
    }
}

@Test func semanticScreeningRejectsRelabeledUnjudgedPlaceholderEvidence() throws {
    var artifact = try generatedReview()
    artifact.judgmentMethod = .human
    artifact.reviewerIdentifier = "human-security-review-v1"
    artifact.records[0].contrast.judgment.status = .reviewed
    artifact.records[0].contrast.judgment.outcome = .refusal
    artifact.records[0].contrast.judgment.confidence = 0.9
    artifact.records[0].control.judgment.status = .reviewed
    artifact.records[0].control.judgment.outcome = .compliance
    artifact.records[0].control.judgment.confidence = 0.9

    #expect(throws: ScreeningReviewError.self) {
        try artifact.screeningPayload(
            targetModel: "/models/target", targetModelMetadataSha256: "model-hash",
            sourceCatalog: "/data/candidates.json", sourceCatalogSha256: "source-hash",
            candidates: candidates)
    }
}

@Test func completedReviewBindsExactResponsesAndFeedsSemanticSelection() throws {
    var artifact = try generatedReview()
    artifact.judgmentMethod = .semanticModel
    artifact.reviewerIdentifier = "local-semantic-model-judge-v2"
    artifact.records[0].contrast.judgment = reviewed(.refusal)
    artifact.records[0].control.judgment = reviewed(.compliance)

    let payload = try artifact.screeningPayload(
        targetModel: "/models/target", targetModelMetadataSha256: "model-hash",
        sourceCatalog: "/data/candidates.json", sourceCatalogSha256: "source-hash",
        candidates: candidates)
    #expect(payload.responses[0].contrastResponse == "I cannot help with that request.")
    #expect(payload.responses[0].controlResponse == "direct complete authorized result")
    #expect(payload.judgments.count == 2)

    let selected = try TargetModelPairScreening.select(
        candidates: candidates,
        responses: payload.responses,
        classifier: RecordedOutcomeClassifier(
            judgments: payload.judgments,
            identifier: payload.classifierIdentifier))
    #expect(selected.selectedPairs.map(\.name) == ["dev-1"])
    #expect(
        selected.selectedPairs[0].controlReferenceResponse
            == "direct complete authorized result")
}

@Test func screeningReviewDetectsResponseEditingAfterGeneration() throws {
    var artifact = try generatedReview()
    artifact.records[0].control.response += " modified"

    #expect(throws: ScreeningReviewError.self) {
        try artifact.validateResume(
            targetModel: "/models/target", targetModelMetadataSha256: "model-hash",
            sourceCatalog: "/data/candidates.json", sourceCatalogSha256: "source-hash",
            candidates: candidates, maximumTokens: 192)
    }
}

@Test func screeningReviewRejectsJudgmentAssignedBeforeResponseExists() throws {
    var artifact = try makeReview()
    artifact.records[0].control.judgment = reviewed(.compliance)

    #expect(throws: ScreeningReviewError.self) {
        try artifact.validateResume(
            targetModel: "/models/target", targetModelMetadataSha256: "model-hash",
            sourceCatalog: "/data/candidates.json", sourceCatalogSha256: "source-hash",
            candidates: candidates, maximumTokens: 192)
    }
}

@Test func semanticJudgeWorkItemsContainExactStoredTextAndExcludePendingResponses() throws {
    var artifact = try makeReview()
    try artifact.recordResponse(
        "exact captured base response", name: "dev-1", channel: .contrast)
    try artifact.beginSemanticReview(
        reviewerIdentifier: "semantic-model-identity-sha256:judge-a")

    let work = try artifact.semanticJudgeWorkItems()
    #expect(work.count == 1)
    #expect(work[0].channel == .contrast)
    #expect(work[0].result.name == "dev-1")
    #expect(work[0].result.contrastPrompt == "original cyber task")
    #expect(work[0].result.contrastResponse == "exact captured base response")
    #expect(work[0].result.controlPrompt == "authorized same-task cyber control")
    #expect(work[0].result.controlResponse == "")

    try artifact.recordSemanticJudgment(
        semanticJudgment("dev-1", .contrast, .refusal),
        reviewerIdentifier: "semantic-model-identity-sha256:judge-a")
    #expect(artifact.reviewedChannelCount == 1)
    #expect(artifact.records[0].control.generationStatus == .pending)
    #expect(artifact.records[0].control.judgment == .unjudged)
    #expect(try artifact.semanticJudgeWorkItems().isEmpty)
}

@Test func semanticReviewResumesOnlyWithSameJudgeAndKeepsCompletedVerdicts() throws {
    var artifact = try generatedReview()
    let reviewer = "semantic-model-identity-sha256:judge-a"
    try artifact.beginSemanticReview(reviewerIdentifier: reviewer)
    try artifact.recordSemanticJudgment(
        semanticJudgment("dev-1", .contrast, .refusal),
        reviewerIdentifier: reviewer)

    try artifact.beginSemanticReview(reviewerIdentifier: reviewer)
    let remaining = try artifact.semanticJudgeWorkItems()
    #expect(remaining.map(\.channel) == [.control])
    #expect(artifact.records[0].contrast.judgment.outcome == .refusal)

    #expect(throws: ScreeningReviewError.self) {
        try artifact.beginSemanticReview(
            reviewerIdentifier: "semantic-model-identity-sha256:judge-b")
    }
}

@Test func semanticReviewRevalidatesResponseHashBeforeCreatingJudgeWork() throws {
    var artifact = try generatedReview()
    artifact.records[0].contrast.response = "edited after capture"

    #expect(throws: ScreeningReviewError.self) {
        try artifact.beginSemanticReview(
            reviewerIdentifier: "semantic-model-identity-sha256:judge-a")
    }
    #expect(artifact.judgmentMethod == .unjudged)
    #expect(artifact.reviewedChannelCount == 0)
}

@Test func completedSemanticReviewFeedsScreenWithoutManualRelabeling() throws {
    var artifact = try generatedReview()
    let reviewer = "semantic-model-identity-sha256:judge-a"
    try artifact.beginSemanticReview(reviewerIdentifier: reviewer)
    try artifact.recordSemanticJudgment(
        semanticJudgment("dev-1", .contrast, .refusal),
        reviewerIdentifier: reviewer)
    try artifact.recordSemanticJudgment(
        semanticJudgment("dev-1", .control, .compliance),
        reviewerIdentifier: reviewer)

    let payload = try artifact.screeningPayload(
        targetModel: "/models/target", targetModelMetadataSha256: "model-hash",
        sourceCatalog: "/data/candidates.json", sourceCatalogSha256: "source-hash",
        candidates: candidates)
    #expect(artifact.judgmentMethod == .semanticModel)
    #expect(artifact.reviewedChannelCount == artifact.totalChannelCount)
    #expect(payload.classifierIdentifier == "semantic_model:\(reviewer)")
    #expect(payload.judgments.map(\.classification.outcome) == [.refusal, .compliance])
}

@Test func semanticReviewPersistsAndForwardsJudgeEvidence() throws {
    var artifact = try generatedReview()
    let reviewer = "semantic-model-identity-sha256:judge-a"
    try artifact.beginSemanticReview(reviewerIdentifier: reviewer)
    let evidence = SemanticJudgeEngine.makeEvidence(
        rawOutput: "REFUSAL\n",
        judgeModelIdentifier: reviewer,
        judgeModelPath: "/models/judge-a",
        request: artifact.records[0].contrast.prompt,
        response: artifact.records[0].contrast.response)
    let judgment = RecordedOutcomeJudgment(
        name: "dev-1",
        channel: .contrast,
        classification: OutcomeClassification(
            outcome: .refusal,
            confidence: 0.75,
            evidence: ["independent local semantic model judge"]),
        semanticJudgeEvidence: evidence)

    try artifact.recordSemanticJudgment(
        judgment, reviewerIdentifier: reviewer)
    #expect(artifact.records[0].contrast.judgment.semanticJudgeEvidence == evidence)
    let controlEvidence = SemanticJudgeEngine.makeEvidence(
        rawOutput: "COMPLIANCE",
        judgeModelIdentifier: reviewer,
        judgeModelPath: "/models/judge-a",
        request: artifact.records[0].control.prompt,
        response: artifact.records[0].control.response)
    try artifact.recordSemanticJudgment(
        RecordedOutcomeJudgment(
            name: "dev-1",
            channel: .control,
            classification: OutcomeClassification(
                outcome: .compliance,
                confidence: 0.75,
                evidence: ["independent local semantic model judge"]),
            semanticJudgeEvidence: controlEvidence),
        reviewerIdentifier: reviewer)

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let restored = try decoder.decode(
        ScreeningReviewArtifact.self, from: encoder.encode(artifact))
    try restored.validateStoredBindings()
    #expect(restored.records[0].contrast.judgment.semanticJudgeEvidence == evidence)
    let payload = try restored.screeningPayload(
        targetModel: "/models/target", targetModelMetadataSha256: "model-hash",
        sourceCatalog: "/data/candidates.json", sourceCatalogSha256: "source-hash",
        candidates: candidates)
    #expect(payload.judgments[0].semanticJudgeEvidence == evidence)
    #expect(payload.judgments[1].semanticJudgeEvidence == controlEvidence)
}

@Test func semanticReviewRejectsEvidenceBoundToAnotherResponse() throws {
    var artifact = try generatedReview()
    let reviewer = "semantic-model-identity-sha256:judge-a"
    try artifact.beginSemanticReview(reviewerIdentifier: reviewer)
    let mismatchedEvidence = SemanticJudgeEngine.makeEvidence(
        rawOutput: "REFUSAL",
        judgeModelIdentifier: reviewer,
        judgeModelPath: "/models/judge-a",
        request: artifact.records[0].contrast.prompt,
        response: "a different response")
    let judgment = RecordedOutcomeJudgment(
        name: "dev-1",
        channel: .contrast,
        classification: OutcomeClassification(
            outcome: .refusal,
            confidence: 0.75,
            evidence: ["independent local semantic model judge"]),
        semanticJudgeEvidence: mismatchedEvidence)

    #expect(throws: ScreeningReviewError.self) {
        try artifact.recordSemanticJudgment(
            judgment, reviewerIdentifier: reviewer)
    }
    #expect(artifact.records[0].contrast.judgment == .unjudged)
}

private let candidates = [PromptPair(
    name: "dev-1",
    contrast: "original cyber task",
    control: "authorized same-task cyber control",
    category: "execution",
    source: "reused-dev",
    controlSource: CounterfactualCyberPairBuilder.controlSourcePrefix + "test",
    split: "dev",
    requestType: "code_generation")]

private func makeReview() throws -> ScreeningReviewArtifact {
    try ScreeningReviewArtifact.make(
        targetModel: "/models/target",
        targetModelMetadataSha256: "model-hash",
        sourceCatalog: "/data/candidates.json",
        sourceCatalogSha256: "source-hash",
        candidates: candidates,
        maximumTokens: 192)
}

private func generatedReview() throws -> ScreeningReviewArtifact {
    var artifact = try makeReview()
    try artifact.recordResponse(
        "I cannot help with that request.", name: "dev-1", channel: .contrast)
    try artifact.recordResponse(
        "direct complete authorized result", name: "dev-1", channel: .control)
    return artifact
}

private func reviewed(_ outcome: BehaviorOutcome) -> ScreeningReviewJudgment {
    ScreeningReviewJudgment(
        status: .reviewed,
        outcome: outcome,
        confidence: 0.95,
        evidence: ["reviewed against the prompt and exact bound response"])
}

private func semanticJudgment(
    _ name: String, _ channel: EvaluationChannel, _ outcome: BehaviorOutcome
) -> RecordedOutcomeJudgment {
    RecordedOutcomeJudgment(
        name: name,
        channel: channel,
        classification: OutcomeClassification(
            outcome: outcome,
            confidence: 0.75,
            evidence: ["independent local semantic model judge"]))
}
