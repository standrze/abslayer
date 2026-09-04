import Foundation
import Testing
@testable import ProbeCore

@Test func semanticJudgeAcceptsOnlyExactTrimmedLabels() throws {
    #expect(try SemanticJudgeEngine.parseLabel("COMPLIANCE") == .compliance)
    #expect(try SemanticJudgeEngine.parseLabel(" \nPARTIAL\t") == .partialCompliance)
    #expect(try SemanticJudgeEngine.parseLabel("REFUSAL\n") == .refusal)
    #expect(try SemanticJudgeEngine.parseLabel("\tINCOHERENT ") == .incoherent)
    #expect(try SemanticJudgeEngine.parseLabel("PARTIAL<pad><pad>") == .partialCompliance)
    #expect(try SemanticJudgeEngine.parseLabel("COMPLIANCE <end_of_turn>") == .compliance)
    #expect(try SemanticJudgeEngine.parseLabel("REFUSAL</s>") == .refusal)

    for invalid in [
        "The label is COMPLIANCE",
        "PARTIAL REFUSAL",
        "REFUSAL\nCOMPLIANCE",
        "INCOHERENT because the response is empty",
        "NONCOMPLIANCE",
        "compliance",
        "INCORRECT",
        "PARTIAL<pad>explanation",
        "<pad>COMPLIANCE",
        "",
    ] {
        #expect(throws: SemanticJudgeError.self) {
            try SemanticJudgeEngine.parseLabel(invalid)
        }
    }
}

@Test func semanticJudgeEvidenceRetainsRawOutputAndReproducibilityBindings() throws {
    let request = "exact request"
    let response = "exact assistant response"
    let raw = "  REFUSAL\n"
    let identifier = "semantic-model-identity-sha256:judge-a"
    let evidence = SemanticJudgeEngine.makeEvidence(
        rawOutput: raw,
        judgeModelIdentifier: identifier,
        judgeModelPath: "/models/../models/judge-a",
        request: request,
        response: response)
    let judgment = RecordedOutcomeJudgment(
        name: "case",
        channel: .contrast,
        classification: OutcomeClassification(
            outcome: .refusal,
            confidence: 0.75,
            evidence: ["independent local semantic model judge"]),
        semanticJudgeEvidence: evidence)

    #expect(evidence.rawOutput == raw)
    #expect(evidence.rawOutputSha256 == ScreeningReviewProvenance.sha256(raw))
    #expect(evidence.judgeModelIdentifier == identifier)
    #expect(evidence.judgeModelPath == "/models/judge-a")
    #expect(evidence.promptVersion == SemanticJudgeEngine.promptVersion)
    #expect(evidence.requestSha256 == ScreeningReviewProvenance.sha256(request))
    #expect(evidence.responseSha256 == ScreeningReviewProvenance.sha256(response))
    #expect(evidence.maximumTokens == SemanticJudgeEngine.maximumTokens)
    #expect(evidence.temperature == SemanticJudgeEngine.temperature)

    let restored = try JSONDecoder().decode(
        RecordedOutcomeJudgment.self,
        from: JSONEncoder().encode(judgment))
    #expect(restored == judgment)
    #expect(restored.semanticJudgeEvidence?.rawOutput == raw)
}

@Test func legacyRecordedJudgmentWithoutSemanticEvidenceStillDecodes() throws {
    let legacyJSON = #"""
    {
      "name": "legacy-case",
      "channel": "control",
      "classification": {
        "outcome": "compliance",
        "confidence": 0.75,
        "evidence": ["independent local semantic model judge"]
      }
    }
    """#

    let judgment = try JSONDecoder().decode(
        RecordedOutcomeJudgment.self, from: Data(legacyJSON.utf8))
    #expect(judgment.name == "legacy-case")
    #expect(judgment.classification.outcome == .compliance)
    #expect(judgment.semanticJudgeEvidence == nil)
}
