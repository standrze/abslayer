import Testing
@testable import ProbeCore

private func multiSourceCandidate(
    _ plan: SOMMultiSourcePlan,
    score: Double,
    contrastKL: Double = 0.1,
    controlKL: Double = 0.1,
    passes: Bool = true
) -> SOMMultiSourceRankedCandidate {
    SOMMultiSourceRankedCandidate(
        plan: plan,
        metrics: SOMSubsetRankingMetrics(
            contrastStarterLogOddsDelta: score,
            contrastExactMeanKL: contrastKL,
            controlExactMeanKL: controlKL,
            controlStarterLogOddsDelta: 0),
        objectiveScore: score,
        passesContrastKLCeiling: passes,
        passesControlKLCeiling: passes)
}

private func multiSourcePlan(_ values: [(Int, [Int])]) -> SOMMultiSourcePlan {
    SOMMultiSourcePlan(selections: values.map {
        SOMLayerSpecificSelection(
            sourceLayerZeroBased: $0.0, orderedLatticeIDs: $0.1)
    })
}

@Test func multiSourceDefaultsEnforceBothHalfNatKLGates() throws {
    let configuration = SOMMultiSourceSearchConfiguration(
        maximumCases: 30, minimumSourceLayersPerFinalist: 3)
    try configuration.validate(sourceLayerCount: 3, candidateCount: 16)
    #expect(configuration.contrastKLCeiling == 0.5)
    #expect(configuration.controlKLCeiling == 0.5)
    #expect(configuration.beamWidth >= 3)

    let relaxed = SOMMultiSourceSearchConfiguration(
        maximumCases: 30, minimumSourceLayersPerFinalist: 3,
        contrastKLCeiling: 0.500_001)
    #expect(throws: SOMSubsetSearchError.invalidContrastKLCeiling(0.500_001)) {
        try relaxed.validate(sourceLayerCount: 3, candidateCount: 16)
    }
}

@Test func multiSourceSingletonStageCoversAllThreeRequestedLayers() throws {
    let plans = try SOMMultiSourceSearchMath.singletonPlans(
        sourceLayers: [23, 19, 20], candidateCount: 16)
    #expect(plans.count == 48)
    #expect(Set(plans.flatMap(\.sourceLayersZeroBased)) == [19, 20, 23])
    #expect(plans.first?.stableKey == "19:0")
    #expect(plans.last?.stableKey == "23:15")
    #expect(!plans.contains { $0.sourceLayersZeroBased.contains(24) })
}

@Test func multiSourcePreselectionUsesOnlySafeSingletonsPerLayer() throws {
    var candidates = [SOMMultiSourceRankedCandidate]()
    for layer in [19, 20, 23] {
        candidates.append(multiSourceCandidate(
            multiSourcePlan([(layer, [0])]), score: Double(layer)))
        candidates.append(multiSourceCandidate(
            multiSourcePlan([(layer, [1])]), score: Double(layer + 10)))
        candidates.append(multiSourceCandidate(
            multiSourcePlan([(layer, [2])]), score: 999,
            contrastKL: 0.8, passes: false))
    }
    let selected = try SOMMultiSourceSearchMath.preselectedIDs(
        singletonCandidates: candidates,
        sourceLayers: [19, 20, 23], perLayer: 2)
    #expect(selected[19] == [1, 0])
    #expect(selected[20] == [1, 0])
    #expect(selected[23] == [1, 0])
    #expect(selected.values.allSatisfy { !$0.contains(2) })
}

@Test func multiSourceExpansionPreservesOrderWithinEachSourceAndDeduplicates() throws {
    let beam = [
        multiSourcePlan([(19, [1])]),
        multiSourcePlan([(20, [2])]),
    ]
    let expanded = try SOMMultiSourceSearchMath.expandedPlans(
        previousBeam: beam,
        candidateIDsByLayer: [19: [1, 3], 20: [2, 4], 23: [5]],
        maximumDirectionsPerLayer: 2)
    let keys = expanded.map(\.stableKey)
    #expect(keys.contains("19:1,3"))
    #expect(keys.contains("19:1|20:2"))
    #expect(keys.contains("19:1|23:5"))
    #expect(keys.contains("20:2,4"))
    #expect(Set(keys).count == keys.count)
    #expect(!keys.contains("19:3,1"))
}

@Test func coverageStratifiedBeamKeepsEveryTwoSourceBridge() throws {
    let candidates = [
        multiSourceCandidate(multiSourcePlan([(19, [0]), (20, [0])]), score: 1),
        multiSourceCandidate(multiSourcePlan([(19, [0]), (23, [0])]), score: 2),
        multiSourceCandidate(multiSourcePlan([(20, [0]), (23, [0])]), score: 3),
        multiSourceCandidate(multiSourcePlan([(20, [0, 1])]), score: 100),
        multiSourceCandidate(
            multiSourcePlan([(19, [1]), (20, [1])]), score: 999,
            contrastKL: 0.9, passes: false),
    ]
    let beam = try SOMMultiSourceSearchMath.selectBeam(candidates, width: 4)
    let masks = Set(beam.map { $0.sourceLayersZeroBased.map(String.init)
        .joined(separator: ",") })
    #expect(masks.contains("19,20"))
    #expect(masks.contains("19,23"))
    #expect(masks.contains("20,23"))
    #expect(!beam.contains { $0.stableKey == "19:1|20:1" })
}

@Test func multiSourceFinalistsRequireAllConfiguredSources() throws {
    let pair = multiSourceCandidate(
        multiSourcePlan([(19, [0]), (20, [0])]), score: 100)
    let triple = multiSourceCandidate(
        multiSourcePlan([(19, [0]), (20, [0]), (23, [0])]), score: 2)
    let finalists = try SOMMultiSourceSearchMath.selectFinalists([
        .init(depth: 2, candidates: [pair]),
        .init(depth: 3, candidates: [triple]),
    ], minimumSourceLayers: 3, maximumCount: 4)
    #expect(finalists.count == 1)
    #expect(finalists[0].candidate.plan.sourceLayersZeroBased == [19, 20, 23])
}

@Test func multiSourceDefaultSearchHasAStaticBound() {
    let budget = SOMMultiSourceSearchMath.theoreticalCandidateBudget(
        sourceLayerCount: 3, candidateCount: 16,
        preselectionPerLayer: 4, maximumDepth: 7, beamWidth: 4)
    #expect(budget == 252)
}
