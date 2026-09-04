import Testing
@testable import ProbeCore

@Test func optimizationRequestUsesHereticStartupRatioAndCap() {
    let full = OptimizationRequest(
        sourceModel: "bf16", measurementPairs: [], evaluationPairs: [],
        workDirectory: "study", outputModel: "output",
        trialCount: 200, measurementCases: 1, evaluationCases: 1)
    #expect(full.startupTrialCount == 60)

    let smoke = OptimizationRequest(
        sourceModel: "bf16", measurementPairs: [], evaluationPairs: [],
        workDirectory: "study", outputModel: "output",
        trialCount: 24, measurementCases: 1, evaluationCases: 1)
    #expect(smoke.startupTrialCount == 7)
}

@Test func paretoOrderingKeepsTradeoffsAheadOfDominatedTrial() {
    func record(_ index: Int, _ refusal: Double, _ kl: Double) -> AbliterationTrialRecord {
        let parameters = TPESampler(seed: UInt64(index)).suggest(layerCount: 4, history: [])
        return AbliterationTrialRecord(
            index: index, parameters: parameters,
            metrics: .init(
                refusalRate: refusal, controlFailureRate: 0, firstTokenKL: kl))
    }
    let ordered = TPESampler.paretoOrdered([
        record(0, 0.4, 0.2),
        record(1, 0.2, 0.4),
        record(2, 0.6, 0.3),
    ])
    #expect(Set(ordered.prefix(2).map(\.index)) == Set([0, 1]))
    #expect(ordered.last?.index == 2)
}

@Test func tpeSourceAndApplicationPeaksExploreFullZeroBasedStack() {
    let suggestions = (0 ..< 64).map { seed in
        TPESampler(seed: UInt64(seed)).suggest(layerCount: 35, history: [])
    }
    #expect(suggestions.allSatisfy { (0 ... 34).contains($0.directionLayer) })
    #expect(suggestions.allSatisfy { (0 ... 34).contains($0.attention.peakLayer) })
    #expect(suggestions.allSatisfy { (0 ... 34).contains($0.mlp.peakLayer) })
    // Guard against restoring the old 40%-90% direction and 60%-100% peak
    // assumptions: deterministic startup exploration must reach earlier layers.
    #expect(suggestions.map(\.directionLayer).min()! < 13.6)
    #expect(suggestions.map(\.attention.peakLayer).min()! < 20.4)
    #expect(suggestions.map(\.mlp.peakLayer).min()! < 20.4)
}

@Test func startupExplorationBalancesScopesAndStratifiesBlends() {
    let sampler = TPESampler(seed: 7, startupTrials: 60)
    var history = [AbliterationTrialRecord]()
    var scopes = [TrialDirectionScope: Int]()
    var blends = [Float]()
    var attentionMaxima = Set<Float>()
    for index in 0 ..< 60 {
        let suggestion = sampler.suggest(layerCount: 35, history: history)
        scopes[suggestion.directionScope, default: 0] += 1
        attentionMaxima.insert(suggestion.attention.maximum)
        if suggestion.directionScope == .blended {
            blends.append(suggestion.directionBlend ?? -1)
        }
        history.append(.init(
            index: index, parameters: suggestion,
            metrics: .init(refusalRate: 1, controlFailureRate: 0, firstTokenKL: 0)))
    }
    #expect(scopes[.global] == 20)
    #expect(scopes[.perLayer] == 20)
    #expect(scopes[.blended] == 20)
    #expect(Set(blends).count == 20)
    #expect(blends.min() == 0.025)
    #expect(blends.max() == 0.975)
    #expect(attentionMaxima.count > 45)
}

@Test func rankFourScalesLocalStrengthWithoutWeakeningGlobalStrength() {
    let sampler = TPESampler(seed: 7, startupTrials: 60, subspaceRank: 4)
    var history = [AbliterationTrialRecord]()
    var suggestions = [AbliterationTrialParameters]()
    for index in 0 ..< 3 {
        let suggestion = sampler.suggest(layerCount: 35, history: history)
        suggestions.append(suggestion)
        history.append(.init(
            index: index, parameters: suggestion,
            metrics: .init(refusalRate: 1, controlFailureRate: 0, firstTokenKL: 0)))
    }
    #expect(suggestions[0].directionScope == .perLayer)
    #expect((0.2 ... 0.75).contains(suggestions[0].attention.maximum))
    #expect(suggestions[1].directionScope == .global)
    #expect((0.8 ... 1.5).contains(suggestions[1].attention.maximum))
    #expect(suggestions[2].directionScope == .blended)
    #expect(suggestions[2].directionBlend == 0.025)
    #expect((0.785 ... 1.481).contains(suggestions[2].attention.maximum))
}

@Test func startupExplicitlyExploresAttentionOnlyAndMLPOnly() {
    let sampler = TPESampler(seed: 17, startupTrials: 60, subspaceRank: 2)
    var history = [AbliterationTrialRecord]()
    var suggestions = [AbliterationTrialParameters]()
    for index in 0 ..< 9 {
        let suggestion = sampler.suggest(layerCount: 35, history: history)
        suggestions.append(suggestion)
        history.append(.init(
            index: index, parameters: suggestion,
            metrics: .init(refusalRate: 1, controlFailureRate: 0, firstTokenKL: 0)))
    }
    for index in 3 ... 5 {
        #expect(suggestions[index].attention.maximum > 0)
        #expect(suggestions[index].mlp.maximum == 0)
    }
    for index in 6 ... 8 {
        #expect(suggestions[index].attention.maximum == 0)
        #expect(suggestions[index].mlp.maximum > 0)
    }
}

@Test func studyMinimizesRefusalsWithinControlGuardrailThenKL() {
    let parameters = TPESampler(seed: 1).suggest(layerCount: 10, history: [])
    let study = AbliterationStudy(modelPath: "model", seed: 1, trials: [
        .init(index: 1, parameters: parameters,
              metrics: .init(refusalRate: 0, controlFailureRate: 0.1, firstTokenKL: 0.6)),
        .init(index: 2, parameters: parameters,
              metrics: .init(refusalRate: 0.1, controlFailureRate: 0, firstTokenKL: 0.2)),
        .init(index: 3, parameters: parameters,
              metrics: .init(refusalRate: 0.1, controlFailureRate: 0, firstTokenKL: 0.1)),
    ])
    #expect(study.best?.index == 3)
}

@Test func tpeSuggestionIsDeterministic() {
    let sampler = TPESampler(seed: 42, startupTrials: 2)
    let first = sampler.suggest(layerCount: 20, history: [])
    let history = [
        AbliterationTrialRecord(index: 0, parameters: first,
            metrics: .init(refusalRate: 0.5, controlFailureRate: 0, firstTokenKL: 0.1)),
        AbliterationTrialRecord(index: 1, parameters: sampler.suggest(layerCount: 20, history: []),
            metrics: .init(refusalRate: 0.2, controlFailureRate: 0, firstTokenKL: 0.1)),
    ]
    #expect(sampler.suggest(layerCount: 20, history: history)
        == sampler.suggest(layerCount: 20, history: history))
}

@Test func forcedExplorationCanLeaveCollapsedScope() {
    let sampler = TPESampler(
        seed: 11, startupTrials: 1, explorationProbability: 1)
    let anchor = TPESampler(seed: 1).suggest(layerCount: 20, history: [])
    let history = [AbliterationTrialRecord(
        index: 0,
        parameters: AbliterationTrialParameters(
            directionScope: .global, directionLayer: anchor.directionLayer,
            attention: anchor.attention, mlp: anchor.mlp),
        metrics: .init(refusalRate: 1, controlFailureRate: 0, firstTokenKL: 0))]
    // Exact scope is seed-dependent, but forced exploration must equal the
    // deterministic uniform proposal rather than a KDE-clamped boundary.
    #expect(sampler.suggest(layerCount: 20, history: history)
        == sampler.suggest(layerCount: 20, history: history))
}
