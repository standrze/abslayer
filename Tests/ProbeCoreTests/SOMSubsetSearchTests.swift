import Foundation
import Testing
@testable import ProbeCore

@Test func somCandidateArchiveRequiresAllSixteenFiniteNonzeroDirections() throws {
    let valid = SOMCandidateDirectionArchive(
        schemaVersion: 1,
        sourceLayerZeroBased: 20,
        trainingMode: "officialMiniSom235",
        candidateDirections: (0 ..< 16).map { id in
            [Float(id + 1), Float(id + 2)]
        })
    try valid.validate()

    let short = SOMCandidateDirectionArchive(
        schemaVersion: 1, sourceLayerZeroBased: 20,
        trainingMode: "officialMiniSom235",
        candidateDirections: Array(valid.candidateDirections.prefix(15)))
    #expect(throws: SOMSubsetSearchError.invalidCandidateCount(expected: 16, actual: 15)) {
        try short.validate()
    }
}

@Test func somCandidateArchiveBundleRetainsEveryTrainedLayerAndSelectedAlias() throws {
    func result(layer: Int) -> SOMDirectionResult {
        let candidates = (0 ..< 16).map { id in
            [Float(layer + id + 1), Float(layer + id + 2)]
        }
        return SOMDirectionResult(
            directions: Array(candidates.prefix(2)),
            selectedNeuronIndices: [0, 1],
            candidateDirections: candidates,
            neurons: candidates,
            occupancy: Array(repeating: 1, count: 16),
            quantizationErrorByNeuron: Array(repeating: 0.1, count: 16),
            quantizationError: 0.1)
    }

    let files = try SOMCandidateArchiveBundle.encode(
        resultsByLayer: [19: result(layer: 19), 23: result(layer: 23),
                         24: result(layer: 24)],
        selectedSourceLayerZeroBased: 23,
        trainingMode: SOMTrainingMode.officialMiniSom235.rawValue)

    #expect(Set(files.keys) == [
        "abslayer_som_candidates.json",
        "abslayer_som_candidates_layer_19.json",
        "abslayer_som_candidates_layer_23.json",
        "abslayer_som_candidates_layer_24.json",
    ])
    let decoder = JSONDecoder()
    let selected = try decoder.decode(
        SOMCandidateDirectionArchive.self,
        from: try #require(files["abslayer_som_candidates.json"]))
    let layer19 = try decoder.decode(
        SOMCandidateDirectionArchive.self,
        from: try #require(files["abslayer_som_candidates_layer_19.json"]))
    #expect(selected.sourceLayerZeroBased == 23)
    #expect(layer19.sourceLayerZeroBased == 19)
    #expect(files["abslayer_som_candidates.json"]
        == files["abslayer_som_candidates_layer_23.json"])
}

@Test func somBeamExpansionEvaluatesSinglesThenOrderedUnusedAppends() throws {
    let singles = try SOMSubsetSearchMath.expandedSequences(
        candidateCount: 4, previousBeam: nil)
    #expect(singles == [[0], [1], [2], [3]])
    let expanded = try SOMSubsetSearchMath.expandedSequences(
        candidateCount: 4, previousBeam: [[0], [2]])
    #expect(expanded == [
        [0, 1], [0, 2], [0, 3],
        [2, 0], [2, 1], [2, 3],
    ])
    #expect(Set(expanded.map { $0.map(String.init).joined(separator: ",") }).count == 6)
}

@Test func defaultFourWideBeamHasBoundedThreeHundredSixteenCandidateBudget() throws {
    var beam: [[Int]]?
    var counts = [Int]()
    for _ in 1 ... 7 {
        let expanded = try SOMSubsetSearchMath.expandedSequences(
            candidateCount: 16, previousBeam: beam)
        counts.append(expanded.count)
        beam = Array(expanded.prefix(4))
    }
    #expect(counts == [16, 60, 56, 52, 48, 44, 40])
    #expect(counts.reduce(0, +) == 316)
}

@Test func somObjectiveIsExplicitAndUsesBothUtilityPenalties() {
    let objective = SOMSubsetSearchObjective(
        controlKLPenalty: 2,
        controlStarterShiftPenalty: 3,
        contrastKLPenalty: 4)
    let metrics = SOMSubsetRankingMetrics(
        contrastStarterLogOddsDelta: 10,
        contrastExactMeanKL: 0.5,
        controlExactMeanKL: 0.25,
        controlStarterLogOddsDelta: -1)
    #expect(abs(objective.score(metrics) - 4.5) < 0.000_001)
    #expect(objective.definition.contains("first-token KL"))
}

@Test func somBeamEnforcesControlCeilingAndPrefersUsefulParetoCandidates() throws {
    func candidate(
        _ ids: [Int], _ contrastDelta: Double, _ contrastKL: Double,
        _ controlKL: Double, _ controlDelta: Double, _ score: Double,
        _ feasible: Bool = true
    ) -> SOMSubsetSearchMath.RankedCandidate {
        SOMSubsetSearchMath.RankedCandidate(
            orderedLatticeIDs: ids,
            metrics: SOMSubsetRankingMetrics(
                contrastStarterLogOddsDelta: contrastDelta,
                contrastExactMeanKL: contrastKL,
                controlExactMeanKL: controlKL,
                controlStarterLogOddsDelta: controlDelta),
            objectiveScore: score,
            passesContrastKLCeiling: feasible,
            passesControlKLCeiling: feasible)
    }
    let values = [
        candidate([0], 2, 0.2, 0.2, 0.1, 1.6),
        candidate([1], 1, 0.1, 0.1, 0.05, 1.7),
        candidate([2], 9, 2, 2, 2, 99, false),
        candidate([3], -1, 0.01, 0.01, 0.01, 5),
    ]
    let pareto = SOMSubsetSearchMath.paretoSequences(values)
    #expect(pareto.contains("0"))
    #expect(pareto.contains("1"))
    #expect(!pareto.contains("2"))
    let beam = try SOMSubsetSearchMath.selectBeam(values, width: 2)
    #expect(beam == [[1], [0]])
    #expect(!beam.contains([2]))
}

@Test func somBeamRejectsContrastKLGateFailuresBeforeParetoRetention() throws {
    func candidate(
        _ id: Int, contrastKL: Double, passesContrast: Bool
    ) -> SOMSubsetSearchMath.RankedCandidate {
        SOMSubsetSearchMath.RankedCandidate(
            orderedLatticeIDs: [id],
            metrics: SOMSubsetRankingMetrics(
                contrastStarterLogOddsDelta: Double(10 - id),
                contrastExactMeanKL: contrastKL,
                controlExactMeanKL: 0.1,
                controlStarterLogOddsDelta: 0),
            objectiveScore: Double(10 - id),
            passesContrastKLCeiling: passesContrast,
            passesControlKLCeiling: true)
    }
    let values = [
        candidate(0, contrastKL: 1.25, passesContrast: false),
        candidate(1, contrastKL: 0.49, passesContrast: true),
    ]
    #expect(SOMSubsetSearchMath.paretoSequences(values) == ["1"])
    #expect(try SOMSubsetSearchMath.selectBeam(values, width: 4) == [[1]])
}

@Test func somSearchConfigurationNeverInfersSourceOrApplicationScope() throws {
    let configuration = SOMSubsetSearchConfiguration(
        sourceLayerZeroBased: 24,
        applicationScope: .local,
        maximumCases: 10)
    try configuration.validate(candidateCount: 16)
    #expect(configuration.sourceLayerZeroBased == 24)
    #expect(configuration.applicationScope == .local)
    #expect(configuration.maximumDepth == 7)
    #expect(configuration.beamWidth == 4)
    #expect(configuration.maximumFinalists == 8)
    #expect(configuration.contrastKLCeiling == 1)
}

@Test func somFinalistsFromShallowDepthsSurviveWhenDepthThreeBeamDies() throws {
    func candidate(
        _ ids: [Int], score: Double, contrastDelta: Double,
        contrastKL: Double, controlKL: Double,
        passesContrast: Bool = true, passesControl: Bool = true
    ) -> SOMSubsetSearchMath.RankedCandidate {
        SOMSubsetSearchMath.RankedCandidate(
            orderedLatticeIDs: ids,
            metrics: SOMSubsetRankingMetrics(
                contrastStarterLogOddsDelta: contrastDelta,
                contrastExactMeanKL: contrastKL,
                controlExactMeanKL: controlKL,
                controlStarterLogOddsDelta: 0.05),
            objectiveScore: score,
            passesContrastKLCeiling: passesContrast,
            passesControlKLCeiling: passesControl)
    }

    let depthOne = [
        candidate([0], score: 2, contrastDelta: 2,
                  contrastKL: 0.2, controlKL: 0.2),
        candidate([1], score: 1, contrastDelta: 1,
                  contrastKL: 0.1, controlKL: 0.1),
    ]
    let depthTwo = [
        candidate([0, 1], score: 3, contrastDelta: 3,
                  contrastKL: 0.3, controlKL: 0.25),
        candidate([1, 0], score: 99, contrastDelta: 9,
                  contrastKL: 1.5, controlKL: 0.2,
                  passesContrast: false),
    ]
    let depthThree = [
        candidate([0, 1, 2], score: 100, contrastDelta: 10,
                  contrastKL: 1.8, controlKL: 0.2,
                  passesContrast: false),
        candidate([0, 1, 3], score: 101, contrastDelta: 11,
                  contrastKL: 0.2, controlKL: 1.8,
                  passesControl: false),
    ]

    #expect(try SOMSubsetSearchMath.selectBeam(depthThree, width: 4).isEmpty)
    let finalists = try SOMSubsetSearchMath.selectFinalists([
        .init(depth: 1, candidates: depthOne),
        .init(depth: 2, candidates: depthTwo),
        .init(depth: 3, candidates: depthThree),
    ], maximumCount: 8)

    let depthOneRepresentative = finalists.first {
        $0.depth == 1 && $0.selectionReason == .bestFeasibleAtDepth
    }
    let depthTwoRepresentative = finalists.first {
        $0.depth == 2 && $0.selectionReason == .bestFeasibleAtDepth
    }
    #expect(depthOneRepresentative?.candidate.orderedLatticeIDs == [0])
    #expect(depthTwoRepresentative?.candidate.orderedLatticeIDs == [0, 1])
    #expect(finalists.allSatisfy {
        $0.candidate.passesContrastKLCeiling
            && $0.candidate.passesControlKLCeiling
    })
    #expect(!finalists.contains { $0.depth == 3 })
    #expect(finalists.count <= 8)
}

@Test func somFinalistLimitIsValidatedAndBoundsDepthRepresentatives() throws {
    func candidate(_ ids: [Int], score: Double) -> SOMSubsetSearchMath.RankedCandidate {
        SOMSubsetSearchMath.RankedCandidate(
            orderedLatticeIDs: ids,
            metrics: SOMSubsetRankingMetrics(
                contrastStarterLogOddsDelta: score,
                contrastExactMeanKL: 0.1,
                controlExactMeanKL: 0.1,
                controlStarterLogOddsDelta: 0),
            objectiveScore: score,
            passesContrastKLCeiling: true,
            passesControlKLCeiling: true)
    }
    let reached = (1 ... 4).map { depth in
        SOMSubsetSearchMath.DepthCandidates(
            depth: depth,
            candidates: [candidate(Array(0 ..< depth), score: Double(depth))])
    }
    let finalists = try SOMSubsetSearchMath.selectFinalists(
        reached, maximumCount: 2)
    #expect(finalists.map(\.depth) == [1, 2])
    #expect(throws: SOMSubsetSearchError.invalidMaximumFinalists(0)) {
        try SOMSubsetSearchMath.selectFinalists(reached, maximumCount: 0)
    }
}
