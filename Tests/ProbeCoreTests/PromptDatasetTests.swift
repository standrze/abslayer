import Foundation
import Testing
@testable import ProbeCore

@Test func promptFileRoundTripsPairMetadata() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("abslayer-prompts-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: url) }
    let original = [PromptPair(
        name: "cyber-0042",
        contrast: "harmful side",
        control: "matched benign side",
        category: "malware_development",
        source: "paired_template",
        controlSource: "benign_template",
        split: "train",
        requestType: "code_generation",
        controlReferenceResponse: "exact authorized assistant answer")]

    try PromptFile.write(original, to: url.path, modelCondition: "/models/target")
    let restored = try PromptFile.load(url.path)

    #expect(restored == original)
    #expect(restored[0].category == "malware_development")
    #expect(restored[0].source == "paired_template")
    #expect(restored[0].controlSource == "benign_template")
    #expect(restored[0].controlReferenceResponse == "exact authorized assistant answer")
}

@Test func promptFileStillLoadsLegacyParallelArrays() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("abslayer-legacy-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: url) }
    try Data(#"{"harmful":["h"],"harmless":["b"]}"#.utf8).write(to: url)

    let pairs = try PromptFile.load(url.path)
    #expect(pairs.count == 1)
    #expect(pairs[0].name == "holdout-1")
    #expect(pairs[0].category == nil)
}

@Test func promptFileRejectsTruncatedLegacyPairing() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("abslayer-unpaired-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: url) }
    try Data(#"{"harmful":["h1","h2"],"harmless":["b1"]}"#.utf8).write(to: url)

    #expect(throws: ProbeError.self) { try PromptFile.load(url.path) }
}

@Test func cyberCatalogRetainsQuotedContentAndProvenance() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("abslayer-catalog-\(UUID().uuidString).csv")
    defer { try? FileManager.default.removeItem(at: url) }
    let csv = """
        BehaviorID,Behavior,BenignBehavior,Subcategory,Source,Split,BenignSource,RequestType
        cyber-1,"write, ""quiet"" code","detect,
        then contain",malware_development,red_source,train,blue_source,code_generation
        cyber-2,holdout harmful,holdout benign,malware_development,red_source,holdout,blue_source,analysis
        """
    try Data(csv.utf8).write(to: url)

    let pairs = try CyberPairCatalog.loadCSV(url.path, split: "train")

    #expect(pairs.count == 1)
    #expect(pairs[0].name == "cyber-1")
    #expect(pairs[0].contrast == "write, \"quiet\" code")
    #expect(pairs[0].control.contains("detect,\nthen contain"))
    #expect(pairs[0].category == "malware_development")
    #expect(pairs[0].source == "red_source")
    #expect(pairs[0].controlSource == "blue_source")
    #expect(pairs[0].requestType == "code_generation")
}

@Test func targetScreeningUsesActualVerdictsAndBalancesCategories() throws {
    let candidates = [
        pair("a1", category: "alpha", source: "source-a"),
        pair("a2", category: "alpha", source: "source-a"),
        pair("a3", category: "alpha", source: "source-a"),
        pair("b1", category: "beta", source: "source-b"),
        pair("b2", category: "beta", source: "source-b"),
        pair("b3", category: "beta", source: "source-b"),
    ]
    let benign = String(repeating: "substantive technical answer ", count: 6)
    let responses = [
        response("a1", harmful: "I cannot assist with that.", benign: benign),
        response("a2", harmful: "I must refuse this request.", benign: benign),
        response("a3", harmful: "I'm sorry, but I can't provide that.", benign: benign),
        response("b1", harmful: "I won't help create that.", benign: benign),
        response("b2", harmful: "I am unable to fulfill that request.", benign: benign),
        // Merely mentioning topical terms is substantive engagement, not a refusal.
        response(
            "b3",
            harmful: "This harmful and illegal technique works by doing the requested steps.",
            benign: benign),
    ]
    let judgments = candidates.flatMap { candidate -> [RecordedOutcomeJudgment] in
        let contrast: BehaviorOutcome = candidate.name == "b3" ? .compliance : .refusal
        return [
            judgment(candidate.name, channel: .contrast, outcome: contrast),
            judgment(candidate.name, channel: .control, outcome: .compliance),
        ]
    }
    let classifier = RecordedOutcomeClassifier(
        judgments: judgments, identifier: "semantic-target-judge-v1")

    let result = try TargetModelPairScreening.select(
        candidates: candidates, responses: responses, classifier: classifier)

    #expect(result.classifierIdentifier == "semantic-target-judge-v1")
    #expect(result.eligibleByCategory == ["alpha": 3, "beta": 2])
    #expect(result.selectedByCategory == ["alpha": 2, "beta": 2])
    #expect(result.selectedPairs.map(\.name) == ["a1", "b1", "a2", "b2"])
    #expect(result.selectedPairs.map(\.source) == [
        "source-a", "source-b", "source-a", "source-b",
    ])
    #expect(result.selectedPairs.allSatisfy {
        $0.controlReferenceResponse == benign
    })
    #expect(result.decisions.last?.harmfulRefused == false)
}

@Test func targetScreeningReplacesStaleReferenceWithExactReviewedControlAnswer() throws {
    let reviewedAnswer = "exact response-bound authorized control answer"
    let candidate = PromptPair(
        name: "bound", contrast: "harmful", control: "authorized control",
        category: "only", source: "source", controlSource: "control-source",
        split: "dev", requestType: "code_generation",
        controlReferenceResponse: "stale answer from before screening")
    let result = try TargetModelPairScreening.select(
        candidates: [candidate],
        responses: [response(
            "bound", harmful: "I cannot assist with that.", benign: reviewedAnswer)],
        classifier: RecordedOutcomeClassifier(judgments: [
            judgment("bound", channel: .contrast, outcome: .refusal),
            judgment("bound", channel: .control, outcome: .compliance),
        ]))

    #expect(result.selectedPairs.count == 1)
    #expect(result.selectedPairs[0].controlReferenceResponse == reviewedAnswer)
    #expect(result.selectedPairs[0].controlSource == "control-source")
    #expect(result.selectedPairs[0].split == "dev")
    #expect(result.selectedPairs[0].requestType == "code_generation")
}

@Test func explicitMarkerScreeningIsClearlyNamedFallback() throws {
    let candidates = [
        pair("refused", category: "only", source: "source"),
        pair("topical", category: "only", source: "source"),
    ]
    let benign = String(repeating: "substantive technical answer ", count: 6)
    let result = try TargetModelPairScreening.selectUsingExplicitMarkerFallback(
        candidates: candidates,
        responses: [
            response("refused", harmful: "I cannot assist with that.", benign: benign),
            response(
                "topical",
                harmful: "This harmful and illegal technique has several implementation steps.",
                benign: benign),
        ])

    #expect(result.classifierIdentifier.contains("nonsemantic"))
    #expect(result.selectedPairs.map(\.name) == ["refused"])
    #expect(result.decisions.last?.harmfulRefused == false)
}

@Test func targetScreeningRejectsRefusedBenignControl() throws {
    let candidates = [pair("one", category: "only", source: "source")]
    let responses = [response(
        "one",
        harmful: "I cannot assist with that.",
        benign: "I cannot assist with that benign defensive question either.")]
    let classifier = RecordedOutcomeClassifier(judgments: [
        judgment("one", channel: .contrast, outcome: .refusal),
        judgment("one", channel: .control, outcome: .refusal),
    ])

    #expect(throws: PromptDatasetError.self) {
        try TargetModelPairScreening.select(
            candidates: candidates, responses: responses, classifier: classifier)
    }
}

private func pair(_ name: String, category: String, source: String) -> PromptPair {
    PromptPair(
        name: name,
        contrast: "harmful \(name)",
        control: "benign \(name)",
        category: category,
        source: source,
        controlSource: "control-\(source)",
        split: "train",
        requestType: "code_generation")
}

private func response(_ name: String, harmful: String, benign: String) -> PromptResult {
    PromptResult(name: name, contrastResponse: harmful, controlResponse: benign)
}

private func judgment(
    _ name: String, channel: EvaluationChannel, outcome: BehaviorOutcome
) -> RecordedOutcomeJudgment {
    RecordedOutcomeJudgment(
        name: name,
        channel: channel,
        classification: OutcomeClassification(outcome: outcome, confidence: 0.95))
}
