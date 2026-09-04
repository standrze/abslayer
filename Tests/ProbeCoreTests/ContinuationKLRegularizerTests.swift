import Foundation
import MLX
import MLXNN
import Testing
@testable import ProbeCore

@Test func continuationKLOptionsAreDisabledByDefault() throws {
    let options = try ContinuationKLRegularizerOptions.parse([:])
    #expect(!options.isEnabled)
    #expect(options.weight == 0)
    #expect(options.maximumCases == 16)
    #expect(options.maximumTokens == 16)
    #expect(options.topK == 256)
    #expect(options.tailWeight == 0)
    #expect(options.tailThreshold == 1)
}

@Test func continuationKLOptionsParseAllBoundedValues() throws {
    let options = try ContinuationKLRegularizerOptions.parse([
        "ABSLAYER_CONTINUATION_KL_PROMPTS": " controls.json ",
        "ABSLAYER_CONTINUATION_KL_WEIGHT": "4",
        "ABSLAYER_CONTINUATION_KL_MAX_CASES": "7",
        "ABSLAYER_CONTINUATION_KL_MAX_TOKENS": "9",
        "ABSLAYER_CONTINUATION_KL_TOP_K": "64",
        "ABSLAYER_CONTINUATION_KL_TAIL_WEIGHT": "0.25",
        "ABSLAYER_CONTINUATION_KL_TAIL_THRESHOLD": "0.5",
    ])
    #expect(options.isEnabled)
    #expect(options.promptPath == "controls.json")
    #expect(options.weight == 4)
    #expect(options.maximumCases == 7)
    #expect(options.maximumTokens == 9)
    #expect(options.topK == 64)
    #expect(options.tailWeight == 0.25)
    #expect(options.tailThreshold == 0.5)
}

@Test(arguments: [
    ("ABSLAYER_CONTINUATION_KL_MAX_CASES", "0"),
    ("ABSLAYER_CONTINUATION_KL_MAX_TOKENS", "-1"),
    ("ABSLAYER_CONTINUATION_KL_TOP_K", "nope"),
    ("ABSLAYER_CONTINUATION_KL_WEIGHT", "nan"),
    ("ABSLAYER_CONTINUATION_KL_TAIL_WEIGHT", "-0.1"),
    ("ABSLAYER_CONTINUATION_KL_TAIL_THRESHOLD", "-1"),
])
func continuationKLOptionsRejectInvalidBoundaries(key: String, value: String) {
    #expect(throws: ContinuationKLRegularizerError.self) {
        var environment = [
            "ABSLAYER_CONTINUATION_KL_PROMPTS": "controls.json",
            "ABSLAYER_CONTINUATION_KL_WEIGHT": "1",
        ]
        environment[key] = value
        _ = try ContinuationKLRegularizerOptions.parse(environment)
    }
}

@Test func continuationKLOptionsRejectSilentPartialConfiguration() {
    #expect(throws: ContinuationKLRegularizerError.promptPathRequired) {
        try ContinuationKLRegularizerOptions.parse([
            "ABSLAYER_CONTINUATION_KL_WEIGHT": "1"
        ])
    }
    #expect(throws: ContinuationKLRegularizerError.positiveWeightRequired) {
        try ContinuationKLRegularizerOptions.parse([
            "ABSLAYER_CONTINUATION_KL_PROMPTS": "controls.json"
        ])
    }
}

@Test func continuationKLReferenceSelectionFiltersThenSpacesEligibleRows() throws {
    let pairs = (0 ..< 9).map { index in
        PromptPair(
            name: "case-\(index)", contrast: "c", control: "b",
            controlReferenceResponse: index.isMultiple(of: 2) ? "answer" : nil)
    }
    let selected = try ContinuationKLReferenceSelection.select(pairs, maximum: 3)
    #expect(selected.map(\.name) == ["case-0", "case-2", "case-6"])

    #expect(throws: ContinuationKLRegularizerError.missingReferenceResponses) {
        try ContinuationKLReferenceSelection.select(
            [PromptPair(name: "none", contrast: "c", control: "b")],
            maximum: 1)
    }
}

@Test func continuationKLPositionBoundarySelectsOnlyReferenceAssistantLogits() {
    // Four deployed-prompt tokens and two continuation tokens produce five
    // next-token logits. Only logits 3 and 4 predict assistant-reference tokens.
    #expect(ContinuationKLPositionSelection.suffixRange(
        totalLogitCount: 5, continuationPositionCount: 2) == (3 ..< 5))
    #expect(ContinuationKLPositionSelection.suffixRange(
        totalLogitCount: 1, continuationPositionCount: 1) == (0 ..< 1))
    #expect(ContinuationKLPositionSelection.suffixRange(
        totalLogitCount: 1, continuationPositionCount: 2) == nil)
    #expect(ContinuationKLPositionSelection.suffixRange(
        totalLogitCount: 1, continuationPositionCount: 0) == nil)
}

@Test func continuationKLComputesTheTopKPlusTailPartition() throws {
    let measured = try ContinuationKLMath.partitionDivergence(
        baselineSupportLogProbabilities: [Float(log(0.6)), Float(log(0.2))],
        candidateSupportLogProbabilities: [Float(log(0.3)), Float(log(0.1))],
        baselineTailLogProbability: Float(log(0.2)),
        candidateTailLogProbability: Float(log(0.6)))
    let expected = 0.6 * log(0.6 / 0.3)
        + 0.2 * log(0.2 / 0.1)
        + 0.2 * log(0.2 / 0.6)
    #expect(abs(measured - expected) < 0.000_001)
}

@Test func continuationKLTailHingeAveragesOnlyExcessAboveThreshold() throws {
    let components = try ContinuationKLMath.components(
        positionDivergences: [0.1, 0.5, 1.5], tailThreshold: 0.5)
    #expect(abs(components.meanPositionKL - 0.7) < 0.000_001)
    #expect(abs(components.meanTailExcess - (1.0 / 3.0)) < 0.000_001)
    #expect(abs(components.regularizer(tailWeight: 0.25)
        - (0.7 + 0.25 / 3.0)) < 0.000_001)
}

@Test func continuationKLTensorMathMatchesScalarPartitionAndHinge() throws {
    let candidateLogs = MLXNN.logSoftmax(MLXArray([
        Float(log(0.3)), Float(log(0.1)), Float(log(0.6)),
        Float(log(0.2)), Float(log(0.2)), Float(log(0.6)),
    ], [2, 3]), axis: -1)
    let supportIDs = MLXArray([0, 1, 0, 1], [2, 2])
    let baselineSupport = MLXArray([
        Float(log(0.6)), Float(log(0.2)),
        Float(log(0.2)), Float(log(0.2)),
    ], [2, 2])
    let baselineTail = MLXArray([Float(log(0.2)), Float(log(0.6))])
    let tensor = ContinuationKLTensorMath.components(
        candidateLogProbabilities: candidateLogs,
        supportTokenIDs: supportIDs,
        baselineSupportLogProbabilities: baselineSupport,
        baselineTailLogProbabilities: baselineTail,
        tailThreshold: 0.1)
    eval(tensor.meanPositionKL, tensor.meanTailExcess)

    let first = try ContinuationKLMath.partitionDivergence(
        baselineSupportLogProbabilities: [Float(log(0.6)), Float(log(0.2))],
        candidateSupportLogProbabilities: [Float(log(0.3)), Float(log(0.1))],
        baselineTailLogProbability: Float(log(0.2)),
        candidateTailLogProbability: Float(log(0.6)))
    let scalar = try ContinuationKLMath.components(
        positionDivergences: [first, 0], tailThreshold: 0.1)
    #expect(abs(Double(tensor.meanPositionKL.item(Float.self))
        - scalar.meanPositionKL) < 0.000_01)
    #expect(abs(Double(tensor.meanTailExcess.item(Float.self))
        - scalar.meanTailExcess) < 0.000_01)
}

@Test func continuationKLMathRejectsMalformedPartitions() {
    #expect(throws: ContinuationKLRegularizerError.malformedPartition) {
        try ContinuationKLMath.partitionDivergence(
            baselineSupportLogProbabilities: [Float(log(0.8))],
            candidateSupportLogProbabilities: [Float(log(0.5))],
            baselineTailLogProbability: Float(log(0.5)),
            candidateTailLogProbability: Float(log(0.5)))
    }
    #expect(throws: ContinuationKLRegularizerError.malformedPartition) {
        try ContinuationKLMath.components(
            positionDivergences: [], tailThreshold: 1)
    }
}
