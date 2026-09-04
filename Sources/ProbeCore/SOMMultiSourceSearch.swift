import Foundation

/// Ordered SOM lattice IDs attached to the zero-based decoder layer where the
/// corresponding directions were measured.
public struct SOMLayerSpecificSelection: Codable, Equatable, Hashable, Sendable {
    public let sourceLayerZeroBased: Int
    public let orderedLatticeIDs: [Int]

    public init(sourceLayerZeroBased: Int, orderedLatticeIDs: [Int]) {
        self.sourceLayerZeroBased = sourceLayerZeroBased
        self.orderedLatticeIDs = orderedLatticeIDs
    }
}

/// Canonical multi-source plan.  Layer entries are sorted, while direction
/// order inside one layer is semantically significant and is never sorted.
public struct SOMMultiSourcePlan: Codable, Equatable, Hashable, Sendable {
    public let selections: [SOMLayerSpecificSelection]

    public init(selections: [SOMLayerSpecificSelection]) {
        self.selections = selections
            .filter { !$0.orderedLatticeIDs.isEmpty }
            .sorted { $0.sourceLayerZeroBased < $1.sourceLayerZeroBased }
    }

    public var totalDirectionCount: Int {
        selections.reduce(0) { $0 + $1.orderedLatticeIDs.count }
    }

    public var sourceLayerCount: Int { selections.count }
    public var sourceLayersZeroBased: [Int] {
        selections.map(\.sourceLayerZeroBased)
    }

    public var stableKey: String {
        selections.map {
            "\($0.sourceLayerZeroBased):"
                + $0.orderedLatticeIDs.map(String.init).joined(separator: ",")
        }.joined(separator: "|")
    }

    public func validate(
        sourceLayers: Set<Int>, candidateCount: Int,
        maximumDirectionsPerLayer: Int
    ) throws {
        guard !selections.isEmpty,
              Set(sourceLayersZeroBased).count == selections.count,
              selections.allSatisfy({ sourceLayers.contains($0.sourceLayerZeroBased) }),
              selections.allSatisfy({ selection in
                  !selection.orderedLatticeIDs.isEmpty
                      && selection.orderedLatticeIDs.count <= maximumDirectionsPerLayer
                      && Set(selection.orderedLatticeIDs).count
                          == selection.orderedLatticeIDs.count
                      && selection.orderedLatticeIDs.allSatisfy {
                          $0 >= 0 && $0 < candidateCount
                      }
              })
        else { throw SOMMultiSourceSearchError.invalidPlan(stableKey) }
    }
}

public struct SOMMultiSourceSearchConfiguration: Codable, Equatable, Sendable {
    public let components: SOMApplicationComponents
    public let maximumCases: Int
    public let preselectionPerLayer: Int
    public let maximumDepth: Int
    public let maximumDirectionsPerLayer: Int
    public let beamWidth: Int
    public let minimumSourceLayersPerFinalist: Int
    public let maximumFinalists: Int
    public let contrastKLCeiling: Double
    public let controlKLCeiling: Double
    public let unloadTolerance: Double
    public let objective: SOMSubsetSearchObjective

    public init(
        components: SOMApplicationComponents = .omlp,
        maximumCases: Int,
        preselectionPerLayer: Int = 4,
        maximumDepth: Int = 7,
        maximumDirectionsPerLayer: Int = 3,
        beamWidth: Int = 4,
        minimumSourceLayersPerFinalist: Int,
        maximumFinalists: Int = 8,
        contrastKLCeiling: Double = 0.5,
        controlKLCeiling: Double = 0.5,
        unloadTolerance: Double = 1e-5,
        objective: SOMSubsetSearchObjective = .default
    ) {
        self.components = components
        self.maximumCases = maximumCases
        self.preselectionPerLayer = preselectionPerLayer
        self.maximumDepth = maximumDepth
        self.maximumDirectionsPerLayer = maximumDirectionsPerLayer
        self.beamWidth = beamWidth
        self.minimumSourceLayersPerFinalist = minimumSourceLayersPerFinalist
        self.maximumFinalists = maximumFinalists
        self.contrastKLCeiling = contrastKLCeiling
        self.controlKLCeiling = controlKLCeiling
        self.unloadTolerance = unloadTolerance
        self.objective = objective
    }

    public func validate(sourceLayerCount: Int, candidateCount: Int) throws {
        guard sourceLayerCount >= 2 else {
            throw SOMMultiSourceSearchError.insufficientSourceLayers(sourceLayerCount)
        }
        guard maximumCases > 0 else {
            throw FirstTokenScreenError.invalidMaximumCases(maximumCases)
        }
        guard preselectionPerLayer > 0,
              preselectionPerLayer <= candidateCount
        else {
            throw SOMMultiSourceSearchError.invalidPreselectionPerLayer(
                preselectionPerLayer)
        }
        guard maximumDepth > 0, maximumDepth <= 7,
              maximumDepth <= sourceLayerCount * preselectionPerLayer
        else { throw SOMSubsetSearchError.invalidMaximumDepth(maximumDepth) }
        guard maximumDirectionsPerLayer > 0,
              maximumDirectionsPerLayer <= maximumDepth
        else {
            throw SOMMultiSourceSearchError.invalidMaximumDirectionsPerLayer(
                maximumDirectionsPerLayer)
        }
        guard beamWidth >= sourceLayerCount else {
            throw SOMMultiSourceSearchError.beamTooNarrow(
                beamWidth: beamWidth, sourceLayerCount: sourceLayerCount)
        }
        guard minimumSourceLayersPerFinalist >= 2,
              minimumSourceLayersPerFinalist <= sourceLayerCount,
              minimumSourceLayersPerFinalist <= maximumDepth
        else {
            throw SOMMultiSourceSearchError.invalidMinimumSourceLayers(
                minimumSourceLayersPerFinalist)
        }
        guard maximumFinalists > 0 else {
            throw SOMSubsetSearchError.invalidMaximumFinalists(maximumFinalists)
        }
        // This search is intentionally the strict utility-preserving path. A
        // caller may tighten either gate, but cannot silently relax it above
        // the project-wide 0.5 boundary.
        guard contrastKLCeiling.isFinite,
              (0 ... 0.5).contains(contrastKLCeiling)
        else {
            throw SOMSubsetSearchError.invalidContrastKLCeiling(
                contrastKLCeiling)
        }
        guard controlKLCeiling.isFinite,
              (0 ... 0.5).contains(controlKLCeiling)
        else {
            throw SOMSubsetSearchError.invalidControlKLCeiling(controlKLCeiling)
        }
        guard unloadTolerance.isFinite, unloadTolerance >= 0 else {
            throw FirstTokenScreenError.invalidUnloadTolerance(unloadTolerance)
        }
        try objective.validate()
    }
}

public struct SOMMultiSourceRankedCandidate: Equatable, Sendable {
    public let plan: SOMMultiSourcePlan
    public let metrics: SOMSubsetRankingMetrics
    public let objectiveScore: Double
    public let passesContrastKLCeiling: Bool
    public let passesControlKLCeiling: Bool

    public init(
        plan: SOMMultiSourcePlan,
        metrics: SOMSubsetRankingMetrics,
        objectiveScore: Double,
        passesContrastKLCeiling: Bool,
        passesControlKLCeiling: Bool
    ) {
        self.plan = plan
        self.metrics = metrics
        self.objectiveScore = objectiveScore
        self.passesContrastKLCeiling = passesContrastKLCeiling
        self.passesControlKLCeiling = passesControlKLCeiling
    }
}

public struct SOMMultiSourceCandidateReport: Codable, Equatable, Sendable {
    public let plan: SOMMultiSourcePlan
    public let objectiveScore: Double
    public let passesContrastKLCeiling: Bool
    public let passesControlKLCeiling: Bool
    public let passesKLCeilings: Bool
    public let isParetoAtDepth: Bool
    public let retainedForExpansion: Bool
    public let rankingMetrics: SOMSubsetRankingMetrics
    public let contrast: FirstTokenCandidateChannelReport
    public let control: FirstTokenCandidateChannelReport
    public let unloadValidation: FirstTokenUnloadValidation
}

public struct SOMMultiSourceDepthReport: Codable, Equatable, Sendable {
    public let depth: Int
    public let generatedCandidateCount: Int
    public let retainedPlans: [SOMMultiSourcePlan]
    public let candidates: [SOMMultiSourceCandidateReport]
}

public struct SOMMultiSourceFinalist: Codable, Equatable, Sendable {
    public let depth: Int
    public let plan: SOMMultiSourcePlan
    public let objectiveScore: Double
    public let rankingMetrics: SOMSubsetRankingMetrics
    public let isParetoAtDepth: Bool
    public let selectionReason: SOMSubsetFinalistSelectionReason
}

public struct SOMMultiSourceMaterializedFinalist: Codable, Equatable, Sendable {
    public let depth: Int
    public let plan: SOMMultiSourcePlan
    public let objectiveScore: Double
    public let rankingMetrics: SOMSubsetRankingMetrics
    public let selectionReason: SOMSubsetFinalistSelectionReason
    public let adapterDirectory: String
}

public struct SOMMultiSourceSearchReport: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: String
    public let modelDirectory: String
    public let promptFile: String
    public let candidateArchiveFilesByLayer: [Int: String]
    public let candidateTrainingModesByLayer: [Int: String]
    public let sourceLayersZeroBased: [Int]
    public let selectedCaseNames: [String]
    public let vocabularySize: Int
    public let configuration: SOMMultiSourceSearchConfiguration
    public let preselectedLatticeIDsByLayer: [Int: [Int]]
    public let composition: AblationComposition
    public let normalization: WeightNormalization
    public let layerSpecificNoBroadcast: Bool
    public let runtimeAdapterScale: Float
    public let exactKLGates: Bool
    public let proxyNotice: String
    public let objectiveDefinition: String
    public let starterMetricDefinition: String
    public let starterVocabulary: FirstTokenStarterVocabulary
    public let baselineContrast: FirstTokenBaselineChannelReport
    public let baselineControl: FirstTokenBaselineChannelReport
    public let theoreticalCandidateBudget: Int
    public let screenedCandidateCount: Int
    public let depths: [SOMMultiSourceDepthReport]
    public let selectedFinalists: [SOMMultiSourceFinalist]
    public let materializedFinalists: [SOMMultiSourceMaterializedFinalist]

    public init(
        schemaVersion: Int = 1, createdAt: String,
        modelDirectory: String, promptFile: String,
        candidateArchiveFilesByLayer: [Int: String],
        candidateTrainingModesByLayer: [Int: String],
        sourceLayersZeroBased: [Int], selectedCaseNames: [String],
        vocabularySize: Int, configuration: SOMMultiSourceSearchConfiguration,
        preselectedLatticeIDsByLayer: [Int: [Int]],
        composition: AblationComposition = .sequential,
        normalization: WeightNormalization = .none,
        layerSpecificNoBroadcast: Bool = true,
        runtimeAdapterScale: Float = 1,
        exactKLGates: Bool = true,
        proxyNotice: String = FirstTokenScreenEngine.proxyNotice,
        objectiveDefinition: String,
        starterMetricDefinition: String = FirstTokenScreenEngine.starterMetricDefinition,
        starterVocabulary: FirstTokenStarterVocabulary,
        baselineContrast: FirstTokenBaselineChannelReport,
        baselineControl: FirstTokenBaselineChannelReport,
        theoreticalCandidateBudget: Int, screenedCandidateCount: Int,
        depths: [SOMMultiSourceDepthReport],
        selectedFinalists: [SOMMultiSourceFinalist],
        materializedFinalists: [SOMMultiSourceMaterializedFinalist]
    ) {
        self.schemaVersion = schemaVersion
        self.createdAt = createdAt
        self.modelDirectory = modelDirectory
        self.promptFile = promptFile
        self.candidateArchiveFilesByLayer = candidateArchiveFilesByLayer
        self.candidateTrainingModesByLayer = candidateTrainingModesByLayer
        self.sourceLayersZeroBased = sourceLayersZeroBased
        self.selectedCaseNames = selectedCaseNames
        self.vocabularySize = vocabularySize
        self.configuration = configuration
        self.preselectedLatticeIDsByLayer = preselectedLatticeIDsByLayer
        self.composition = composition
        self.normalization = normalization
        self.layerSpecificNoBroadcast = layerSpecificNoBroadcast
        self.runtimeAdapterScale = runtimeAdapterScale
        self.exactKLGates = exactKLGates
        self.proxyNotice = proxyNotice
        self.objectiveDefinition = objectiveDefinition
        self.starterMetricDefinition = starterMetricDefinition
        self.starterVocabulary = starterVocabulary
        self.baselineContrast = baselineContrast
        self.baselineControl = baselineControl
        self.theoreticalCandidateBudget = theoreticalCandidateBudget
        self.screenedCandidateCount = screenedCandidateCount
        self.depths = depths
        self.selectedFinalists = selectedFinalists
        self.materializedFinalists = materializedFinalists
    }
}

/// Deterministic bounded planning and ranking, separated from Metal execution.
public enum SOMMultiSourceSearchMath {
    public struct DepthCandidates: Equatable, Sendable {
        public let depth: Int
        public let candidates: [SOMMultiSourceRankedCandidate]

        public init(depth: Int, candidates: [SOMMultiSourceRankedCandidate]) {
            self.depth = depth
            self.candidates = candidates
        }
    }

    public struct FinalistSelection: Equatable, Sendable {
        public let depth: Int
        public let candidate: SOMMultiSourceRankedCandidate
        public let isParetoAtDepth: Bool
        public let selectionReason: SOMSubsetFinalistSelectionReason
    }

    public static func singletonPlans(
        sourceLayers: [Int], candidateCount: Int
    ) throws -> [SOMMultiSourcePlan] {
        guard sourceLayers.count >= 2,
              Set(sourceLayers).count == sourceLayers.count,
              sourceLayers.allSatisfy({ $0 >= 0 }), candidateCount > 0
        else { throw SOMMultiSourceSearchError.invalidSources(sourceLayers) }
        return sourceLayers.sorted().flatMap { layer in
            (0 ..< candidateCount).map { id in
                SOMMultiSourcePlan(selections: [SOMLayerSpecificSelection(
                    sourceLayerZeroBased: layer, orderedLatticeIDs: [id])])
            }
        }
    }

    public static func preselectedIDs(
        singletonCandidates: [SOMMultiSourceRankedCandidate],
        sourceLayers: [Int], perLayer: Int
    ) throws -> [Int: [Int]] {
        guard perLayer > 0 else {
            throw SOMMultiSourceSearchError.invalidPreselectionPerLayer(perLayer)
        }
        var result = [Int: [Int]]()
        for layer in sourceLayers.sorted() {
            let eligible = singletonCandidates.filter { candidate in
                candidate.plan.selections.count == 1
                    && candidate.plan.selections[0].sourceLayerZeroBased == layer
                    && candidate.passesContrastKLCeiling
                    && candidate.passesControlKLCeiling
                    && candidate.metrics.isFinite
                    && candidate.objectiveScore.isFinite
            }.sorted(by: rankedPrecedes)
            let ids = eligible.prefix(perLayer).compactMap {
                $0.plan.selections.first?.orderedLatticeIDs.first
            }
            guard !ids.isEmpty else {
                throw SOMMultiSourceSearchError.noFeasibleSingleton(layer: layer)
            }
            result[layer] = ids
        }
        return result
    }

    public static func expandedPlans(
        previousBeam: [SOMMultiSourcePlan],
        candidateIDsByLayer: [Int: [Int]],
        maximumDirectionsPerLayer: Int
    ) throws -> [SOMMultiSourcePlan] {
        guard maximumDirectionsPerLayer > 0,
              !previousBeam.isEmpty, !candidateIDsByLayer.isEmpty
        else {
            throw SOMMultiSourceSearchError.invalidMaximumDirectionsPerLayer(
                maximumDirectionsPerLayer)
        }
        let sourceLayers = Set(candidateIDsByLayer.keys)
        let candidateCount = (candidateIDsByLayer.values.flatMap { $0 }.max() ?? -1) + 1
        var seen = Set<String>()
        var result = [SOMMultiSourcePlan]()
        for plan in previousBeam {
            try plan.validate(
                sourceLayers: sourceLayers,
                candidateCount: candidateCount,
                maximumDirectionsPerLayer: maximumDirectionsPerLayer)
            let current = Dictionary(uniqueKeysWithValues: plan.selections.map {
                ($0.sourceLayerZeroBased, $0.orderedLatticeIDs)
            })
            for layer in candidateIDsByLayer.keys.sorted() {
                let existing = current[layer] ?? []
                guard existing.count < maximumDirectionsPerLayer else { continue }
                let used = Set(existing)
                for id in candidateIDsByLayer[layer] ?? [] where !used.contains(id) {
                    var next = current
                    next[layer] = existing + [id]
                    let expanded = SOMMultiSourcePlan(selections: next.map {
                        SOMLayerSpecificSelection(
                            sourceLayerZeroBased: $0.key,
                            orderedLatticeIDs: $0.value)
                    })
                    if seen.insert(expanded.stableKey).inserted {
                        result.append(expanded)
                    }
                }
            }
        }
        return result.sorted { $0.stableKey < $1.stableKey }
    }

    /// Upper bound includes all 16 singleton screens, then assumes every beam
    /// member can append every remaining preselected candidate at each depth.
    public static func theoreticalCandidateBudget(
        sourceLayerCount: Int, candidateCount: Int,
        preselectionPerLayer: Int, maximumDepth: Int, beamWidth: Int
    ) -> Int {
        let pool = sourceLayerCount * min(candidateCount, preselectionPerLayer)
        var total = sourceLayerCount * candidateCount
        if maximumDepth >= 2 {
            for depth in 2 ... maximumDepth {
                total += beamWidth * max(0, pool - (depth - 1))
            }
        }
        return total
    }

    public static func paretoPlanKeys(
        _ candidates: [SOMMultiSourceRankedCandidate]
    ) -> Set<String> {
        let feasible = candidates.filter(isEligible)
        return Set(feasible.compactMap { candidate in
            let dominated = feasible.contains { other in
                other.plan != candidate.plan
                    && dominates(other.metrics, candidate.metrics)
            }
            return dominated ? nil : candidate.plan.stableKey
        })
    }

    /// Coverage-stratified beam selection keeps a path from every source layer
    /// alive at depth one and favors broader source coverage thereafter.  This
    /// prevents a strong z20 singleton from silently collapsing a requested
    /// z19+z20+z23 search back into a one-source experiment.
    public static func selectBeam(
        _ candidates: [SOMMultiSourceRankedCandidate], width: Int
    ) throws -> [SOMMultiSourcePlan] {
        guard width > 0 else { throw SOMSubsetSearchError.invalidBeamWidth(width) }
        let feasible = candidates.filter(isEligible)
        let pareto = paretoPlanKeys(feasible)
        let sorted = feasible.sorted {
            rankedPrecedes($0, $1, paretoKeys: pareto)
        }
        let grouped = Dictionary(grouping: sorted) {
            $0.plan.sourceLayersZeroBased.map(String.init).joined(separator: ",")
        }
        let representatives = grouped.values.compactMap(\.first).sorted {
            rankedPrecedes($0, $1, paretoKeys: pareto)
        }
        var selected = [SOMMultiSourcePlan]()
        var seen = Set<String>()
        for candidate in representatives where selected.count < width {
            if seen.insert(candidate.plan.stableKey).inserted {
                selected.append(candidate.plan)
            }
        }
        for candidate in sorted where selected.count < width {
            if seen.insert(candidate.plan.stableKey).inserted {
                selected.append(candidate.plan)
            }
        }
        return selected
    }

    public static func selectFinalists(
        _ depths: [DepthCandidates],
        minimumSourceLayers: Int,
        maximumCount: Int
    ) throws -> [FinalistSelection] {
        guard minimumSourceLayers > 0 else {
            throw SOMMultiSourceSearchError.invalidMinimumSourceLayers(
                minimumSourceLayers)
        }
        guard maximumCount > 0 else {
            throw SOMSubsetSearchError.invalidMaximumFinalists(maximumCount)
        }
        struct Annotated {
            let depth: Int
            let candidate: SOMMultiSourceRankedCandidate
            let pareto: Bool
        }
        var byDepth = [Annotated]()
        var all = [Annotated]()
        for depth in depths.sorted(by: { $0.depth < $1.depth }) {
            let eligible = depth.candidates.filter {
                isEligible($0)
                    && $0.plan.sourceLayerCount >= minimumSourceLayers
            }
            let pareto = paretoPlanKeys(eligible)
            let sorted = eligible.sorted {
                rankedPrecedes($0, $1, paretoKeys: pareto)
            }
            let annotated = sorted.map {
                Annotated(
                    depth: depth.depth, candidate: $0,
                    pareto: pareto.contains($0.plan.stableKey))
            }
            if let first = annotated.first { byDepth.append(first) }
            all.append(contentsOf: annotated)
        }
        var result = [FinalistSelection]()
        var seen = Set<String>()
        func append(_ value: Annotated, reason: SOMSubsetFinalistSelectionReason) {
            let key = "\(value.depth):\(value.candidate.plan.stableKey)"
            guard result.count < maximumCount, seen.insert(key).inserted else { return }
            result.append(FinalistSelection(
                depth: value.depth, candidate: value.candidate,
                isParetoAtDepth: value.pareto, selectionReason: reason))
        }
        for value in byDepth { append(value, reason: .bestFeasibleAtDepth) }
        let global = all.sorted {
            if $0.candidate.plan.sourceLayerCount != $1.candidate.plan.sourceLayerCount {
                return $0.candidate.plan.sourceLayerCount
                    > $1.candidate.plan.sourceLayerCount
            }
            return rankedPrecedes($0.candidate, $1.candidate)
        }
        for value in global { append(value, reason: .globalTop) }
        return result
    }

    private static func isEligible(_ candidate: SOMMultiSourceRankedCandidate) -> Bool {
        candidate.passesContrastKLCeiling
            && candidate.passesControlKLCeiling
            && candidate.metrics.isFinite
            && candidate.objectiveScore.isFinite
    }

    private static func rankedPrecedes(
        _ lhs: SOMMultiSourceRankedCandidate,
        _ rhs: SOMMultiSourceRankedCandidate
    ) -> Bool {
        rankedPrecedes(lhs, rhs, paretoKeys: [])
    }

    private static func rankedPrecedes(
        _ lhs: SOMMultiSourceRankedCandidate,
        _ rhs: SOMMultiSourceRankedCandidate,
        paretoKeys: Set<String>
    ) -> Bool {
        if lhs.plan.sourceLayerCount != rhs.plan.sourceLayerCount {
            return lhs.plan.sourceLayerCount > rhs.plan.sourceLayerCount
        }
        let lhsPareto = paretoKeys.contains(lhs.plan.stableKey)
        let rhsPareto = paretoKeys.contains(rhs.plan.stableKey)
        let lhsUseful = lhsPareto && lhs.metrics.contrastStarterLogOddsDelta > 0
        let rhsUseful = rhsPareto && rhs.metrics.contrastStarterLogOddsDelta > 0
        if lhsUseful != rhsUseful { return lhsUseful }
        if lhsPareto != rhsPareto { return lhsPareto }
        if lhs.objectiveScore != rhs.objectiveScore {
            return lhs.objectiveScore > rhs.objectiveScore
        }
        if lhs.metrics.controlExactMeanKL != rhs.metrics.controlExactMeanKL {
            return lhs.metrics.controlExactMeanKL < rhs.metrics.controlExactMeanKL
        }
        if lhs.metrics.contrastExactMeanKL != rhs.metrics.contrastExactMeanKL {
            return lhs.metrics.contrastExactMeanKL < rhs.metrics.contrastExactMeanKL
        }
        return lhs.plan.stableKey < rhs.plan.stableKey
    }

    private static func dominates(
        _ lhs: SOMSubsetRankingMetrics, _ rhs: SOMSubsetRankingMetrics
    ) -> Bool {
        let noWorse = lhs.contrastStarterLogOddsDelta
                >= rhs.contrastStarterLogOddsDelta
            && lhs.contrastExactMeanKL <= rhs.contrastExactMeanKL
            && lhs.controlExactMeanKL <= rhs.controlExactMeanKL
            && abs(lhs.controlStarterLogOddsDelta)
                <= abs(rhs.controlStarterLogOddsDelta)
        let strict = lhs.contrastStarterLogOddsDelta
                > rhs.contrastStarterLogOddsDelta
            || lhs.contrastExactMeanKL < rhs.contrastExactMeanKL
            || lhs.controlExactMeanKL < rhs.controlExactMeanKL
            || abs(lhs.controlStarterLogOddsDelta)
                < abs(rhs.controlStarterLogOddsDelta)
        return noWorse && strict
    }
}

public enum SOMMultiSourceSearchEngine {
    private struct LoadedArchive {
        let path: String
        let data: Data
        let archive: SOMCandidateDirectionArchive
    }

    private struct CapturedCandidate {
        let ranked: SOMMultiSourceRankedCandidate
        let contrast: FirstTokenCandidateChannelReport
        let control: FirstTokenCandidateChannelReport
        let unload: FirstTokenUnloadValidation
    }

    public static func run(
        modelDirectory: String,
        promptFile: String,
        pairs: [PromptPair],
        candidateArchiveFiles: [String],
        configuration: SOMMultiSourceSearchConfiguration,
        starterConfiguration: FirstTokenStarterConfiguration = .default,
        finalistDirectory: String? = nil,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> SOMMultiSourceSearchReport {
        let loaded = try loadArchives(candidateArchiveFiles)
        let layers = loaded.map(\.archive.sourceLayerZeroBased).sorted()
        let candidateCount = loaded[0].archive.candidateDirections.count
        try configuration.validate(
            sourceLayerCount: layers.count, candidateCount: candidateCount)
        let archives = Dictionary(uniqueKeysWithValues: loaded.map {
            ($0.archive.sourceLayerZeroBased, $0.archive)
        })

        let selected = try FirstTokenScreenMath.evenlySpaced(
            pairs, maximum: configuration.maximumCases)
        let runtime = try await FirstTokenScreenRuntime(modelDirectory: modelDirectory)
        let starters = try await runtime.resolveStarterVocabulary(starterConfiguration)
        let baseline = try await runtime.capture(pairs: selected)
        let sentinel = FirstTokenScreenMath.prefix(baseline, count: 1)
        let baselineContrast = try FirstTokenScreenMath.baselineReport(
            fingerprint: baseline.contrast, vocabulary: starters)
        let baselineControl = try FirstTokenScreenMath.baselineReport(
            fingerprint: baseline.control, vocabulary: starters)
        progress?("captured untouched baseline over \(selected.count) matched pairs")

        let singletonPlans = try SOMMultiSourceSearchMath.singletonPlans(
            sourceLayers: layers, candidateCount: candidateCount)
        var singletonCaptured = [CapturedCandidate]()
        singletonCaptured.reserveCapacity(singletonPlans.count)
        for (index, plan) in singletonPlans.enumerated() {
            let value = try await screen(
                plan: plan, archives: archives, configuration: configuration,
                runtime: runtime, selectedPairs: selected,
                baseline: baseline, baselineSentinel: sentinel,
                starters: starters)
            singletonCaptured.append(value)
            progress?(screenProgress(
                prefix: "multi-source singleton \(index + 1)/\(singletonPlans.count)",
                value: value))
        }
        let pools = try SOMMultiSourceSearchMath.preselectedIDs(
            singletonCandidates: singletonCaptured.map(\.ranked),
            sourceLayers: layers,
            perLayer: configuration.preselectionPerLayer)
        progress?("safe singleton pools: \(stablePoolDescription(pools))")

        let poolSets = pools.mapValues(Set.init)
        let pooledSingletons = singletonCaptured.filter { value in
            guard let selection = value.ranked.plan.selections.first,
                  let id = selection.orderedLatticeIDs.first
            else { return false }
            return poolSets[selection.sourceLayerZeroBased]?.contains(id) == true
        }
        var previousBeam = try SOMMultiSourceSearchMath.selectBeam(
            pooledSingletons.map(\.ranked), width: configuration.beamWidth)
        let singletonPareto = SOMMultiSourceSearchMath.paretoPlanKeys(
            singletonCaptured.map(\.ranked))
        var depthReports = [makeDepthReport(
            depth: 1, captured: singletonCaptured,
            retained: previousBeam, pareto: singletonPareto)]
        var reached = [SOMMultiSourceSearchMath.DepthCandidates(
            depth: 1, candidates: singletonCaptured.map(\.ranked))]
        var screenedCount = singletonCaptured.count

        if configuration.maximumDepth >= 2 {
            for depth in 2 ... configuration.maximumDepth {
                let plans = try SOMMultiSourceSearchMath.expandedPlans(
                    previousBeam: previousBeam,
                    candidateIDsByLayer: pools,
                    maximumDirectionsPerLayer:
                        configuration.maximumDirectionsPerLayer)
                guard !plans.isEmpty else { break }
                var captured = [CapturedCandidate]()
                captured.reserveCapacity(plans.count)
                for (index, plan) in plans.enumerated() {
                    let value = try await screen(
                        plan: plan, archives: archives,
                        configuration: configuration,
                        runtime: runtime, selectedPairs: selected,
                        baseline: baseline, baselineSentinel: sentinel,
                        starters: starters)
                    captured.append(value)
                    progress?(screenProgress(
                        prefix: "multi-source depth \(depth) \(index + 1)/\(plans.count)",
                        value: value))
                }
                screenedCount += captured.count
                let ranked = captured.map(\.ranked)
                reached.append(.init(depth: depth, candidates: ranked))
                let pareto = SOMMultiSourceSearchMath.paretoPlanKeys(ranked)
                previousBeam = try SOMMultiSourceSearchMath.selectBeam(
                    ranked, width: configuration.beamWidth)
                depthReports.append(makeDepthReport(
                    depth: depth, captured: captured,
                    retained: previousBeam, pareto: pareto))
                progress?("multi-source depth \(depth) retained \(previousBeam.count)/\(plans.count)")
                if previousBeam.isEmpty { break }
            }
        }

        let finalists = try SOMMultiSourceSearchMath.selectFinalists(
            reached,
            minimumSourceLayers: configuration.minimumSourceLayersPerFinalist,
            maximumCount: configuration.maximumFinalists)
        let selectedFinalists = finalists.map {
            SOMMultiSourceFinalist(
                depth: $0.depth, plan: $0.candidate.plan,
                objectiveScore: $0.candidate.objectiveScore,
                rankingMetrics: $0.candidate.metrics,
                isParetoAtDepth: $0.isParetoAtDepth,
                selectionReason: $0.selectionReason)
        }
        let materialized = try await materialize(
            finalists: finalists, loaded: loaded, archives: archives,
            modelDirectory: modelDirectory, configuration: configuration,
            runtime: runtime, rootDirectory: finalistDirectory,
            progress: progress)
        let budget = SOMMultiSourceSearchMath.theoreticalCandidateBudget(
            sourceLayerCount: layers.count, candidateCount: candidateCount,
            preselectionPerLayer: configuration.preselectionPerLayer,
            maximumDepth: configuration.maximumDepth,
            beamWidth: configuration.beamWidth)
        return SOMMultiSourceSearchReport(
            createdAt: ISO8601DateFormatter().string(from: Date()),
            modelDirectory: URL(fileURLWithPath: modelDirectory).standardizedFileURL.path,
            promptFile: URL(fileURLWithPath: promptFile).standardizedFileURL.path,
            candidateArchiveFilesByLayer: Dictionary(uniqueKeysWithValues: loaded.map {
                ($0.archive.sourceLayerZeroBased, $0.path)
            }),
            candidateTrainingModesByLayer: Dictionary(uniqueKeysWithValues: loaded.map {
                ($0.archive.sourceLayerZeroBased, $0.archive.trainingMode)
            }),
            sourceLayersZeroBased: layers,
            selectedCaseNames: selected.map(\.name),
            vocabularySize: baseline.contrast.vocabularySize,
            configuration: configuration,
            preselectedLatticeIDsByLayer: pools,
            objectiveDefinition: configuration.objective.definition,
            starterVocabulary: starters,
            baselineContrast: baselineContrast,
            baselineControl: baselineControl,
            theoreticalCandidateBudget: budget,
            screenedCandidateCount: screenedCount,
            depths: depthReports,
            selectedFinalists: selectedFinalists,
            materializedFinalists: materialized)
    }

    private static func loadArchives(_ paths: [String]) throws -> [LoadedArchive] {
        guard paths.count >= 2 else {
            throw SOMMultiSourceSearchError.insufficientSourceLayers(paths.count)
        }
        var result = [LoadedArchive]()
        var seen = Set<Int>()
        var width: Int?
        for rawPath in paths {
            let path = URL(fileURLWithPath: rawPath).standardizedFileURL.path
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let archive = try JSONDecoder().decode(
                SOMCandidateDirectionArchive.self, from: data)
            try archive.validate()
            guard seen.insert(archive.sourceLayerZeroBased).inserted else {
                throw SOMMultiSourceSearchError.duplicateSourceLayer(
                    archive.sourceLayerZeroBased)
            }
            let candidateWidth = archive.candidateDirections[0].count
            guard width == nil || width == candidateWidth else {
                throw SOMMultiSourceSearchError.inconsistentDirectionWidths
            }
            width = candidateWidth
            result.append(LoadedArchive(path: path, data: data, archive: archive))
        }
        return result.sorted {
            $0.archive.sourceLayerZeroBased < $1.archive.sourceLayerZeroBased
        }
    }

    private static func screen(
        plan: SOMMultiSourcePlan,
        archives: [Int: SOMCandidateDirectionArchive],
        configuration: SOMMultiSourceSearchConfiguration,
        runtime: FirstTokenScreenRuntime,
        selectedPairs: [PromptPair],
        baseline: FirstTokenDualFingerprint,
        baselineSentinel: FirstTokenDualFingerprint,
        starters: FirstTokenStarterVocabulary
    ) async throws -> CapturedCandidate {
        let bases = try basesByLayer(plan: plan, archives: archives)
        let adapter = try await runtime.makeLayerSpecificSequentialSOMAdapter(
            basesByLayer: bases, components: configuration.components)
        do {
            try await runtime.loadAdapter(adapter)
        } catch {
            await runtime.unloadAdapter(adapter)
            throw error
        }
        let candidate: FirstTokenDualFingerprint
        do {
            candidate = try await runtime.capture(pairs: selectedPairs)
        } catch {
            await runtime.unloadAdapter(adapter)
            throw error
        }
        await runtime.unloadAdapter(adapter)
        let restored = try await runtime.capture(pairs: [selectedPairs[0]])
        let unload = try FirstTokenScreenMath.unloadValidation(
            baseline: baselineSentinel, restored: restored,
            tolerance: configuration.unloadTolerance)
        guard unload.passed else {
            throw FirstTokenScreenError.adapterUnloadDidNotRestore(
                directory: "in-memory layer-specific SOM \(plan.stableKey)",
                maximumDifference: max(
                    unload.contrastMaximumAbsoluteLogProbabilityDifference,
                    unload.controlMaximumAbsoluteLogProbabilityDifference),
                tolerance: configuration.unloadTolerance)
        }
        let contrast = try FirstTokenScreenMath.candidateReport(
            baseline: baseline.contrast, candidate: candidate.contrast,
            vocabulary: starters)
        let control = try FirstTokenScreenMath.candidateReport(
            baseline: baseline.control, candidate: candidate.control,
            vocabulary: starters)
        let metrics = SOMSubsetRankingMetrics(
            contrastStarterLogOddsDelta:
                contrast.meanComplianceToRefusalLogOddsDeltaFromBaseline,
            contrastExactMeanKL: contrast.exactMeanKLFromBaseline,
            controlExactMeanKL: control.exactMeanKLFromBaseline,
            controlStarterLogOddsDelta:
                control.meanComplianceToRefusalLogOddsDeltaFromBaseline)
        return CapturedCandidate(
            ranked: SOMMultiSourceRankedCandidate(
                plan: plan, metrics: metrics,
                objectiveScore: configuration.objective.score(metrics),
                passesContrastKLCeiling:
                    metrics.contrastExactMeanKL <= configuration.contrastKLCeiling,
                passesControlKLCeiling:
                    metrics.controlExactMeanKL <= configuration.controlKLCeiling),
            contrast: contrast, control: control, unload: unload)
    }

    private static func basesByLayer(
        plan: SOMMultiSourcePlan,
        archives: [Int: SOMCandidateDirectionArchive]
    ) throws -> [Int: [[Float]]] {
        var result = [Int: [[Float]]]()
        for selection in plan.selections {
            guard let archive = archives[selection.sourceLayerZeroBased],
                  selection.orderedLatticeIDs.allSatisfy({
                      $0 >= 0 && $0 < archive.candidateDirections.count
                  })
            else { throw SOMMultiSourceSearchError.invalidPlan(plan.stableKey) }
            result[selection.sourceLayerZeroBased] =
                selection.orderedLatticeIDs.map {
                    archive.candidateDirections[$0]
                }
        }
        return result
    }

    private static func makeDepthReport(
        depth: Int, captured: [CapturedCandidate],
        retained: [SOMMultiSourcePlan], pareto: Set<String>
    ) -> SOMMultiSourceDepthReport {
        let retainedKeys = Set(retained.map(\.stableKey))
        return SOMMultiSourceDepthReport(
            depth: depth,
            generatedCandidateCount: captured.count,
            retainedPlans: retained,
            candidates: captured.map { value in
                SOMMultiSourceCandidateReport(
                    plan: value.ranked.plan,
                    objectiveScore: value.ranked.objectiveScore,
                    passesContrastKLCeiling:
                        value.ranked.passesContrastKLCeiling,
                    passesControlKLCeiling:
                        value.ranked.passesControlKLCeiling,
                    passesKLCeilings:
                        value.ranked.passesContrastKLCeiling
                            && value.ranked.passesControlKLCeiling,
                    isParetoAtDepth: pareto.contains(
                        value.ranked.plan.stableKey),
                    retainedForExpansion: retainedKeys.contains(
                        value.ranked.plan.stableKey),
                    rankingMetrics: value.ranked.metrics,
                    contrast: value.contrast,
                    control: value.control,
                    unloadValidation: value.unload)
            })
    }

    private static func materialize(
        finalists: [SOMMultiSourceSearchMath.FinalistSelection],
        loaded: [LoadedArchive],
        archives: [Int: SOMCandidateDirectionArchive],
        modelDirectory: String,
        configuration: SOMMultiSourceSearchConfiguration,
        runtime: FirstTokenScreenRuntime,
        rootDirectory: String?,
        progress: (@Sendable (String) -> Void)?
    ) async throws -> [SOMMultiSourceMaterializedFinalist] {
        guard let rootDirectory else { return [] }
        let root = URL(fileURLWithPath: rootDirectory).standardizedFileURL
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var outputs = [SOMMultiSourceMaterializedFinalist]()
        for finalist in finalists {
            let plan = finalist.candidate.plan
            let name = "d\(plan.totalDirectionCount)-" + plan.selections.map {
                "z\($0.sourceLayerZeroBased)-"
                    + $0.orderedLatticeIDs.map(String.init).joined(separator: "-")
            }.joined(separator: "__")
            let destination = root.appendingPathComponent(name, isDirectory: true)
            let adapter = try await runtime.makeLayerSpecificSequentialSOMAdapter(
                basesByLayer: try basesByLayer(plan: plan, archives: archives),
                components: configuration.components)
            let manifest = SOMMultiSourceFinalistManifest(
                schemaVersion: 1,
                sourceModel: URL(fileURLWithPath: modelDirectory).standardizedFileURL.path,
                plan: plan,
                objectiveScore: finalist.candidate.objectiveScore,
                rankingMetrics: finalist.candidate.metrics,
                selectionReason: finalist.selectionReason,
                searchConfiguration: configuration,
                composition: .sequential,
                normalization: .none,
                layerSpecificNoBroadcast: true,
                requiredRuntimeAdapterScale: 1,
                exactContrastKLCeiling: configuration.contrastKLCeiling,
                exactControlKLCeiling: configuration.controlKLCeiling)
            var files = Dictionary(uniqueKeysWithValues: loaded.map {
                (SOMCandidateArchiveBundle.filename(
                    sourceLayerZeroBased: $0.archive.sourceLayerZeroBased), $0.data)
            })
            files["abslayer_som_multisource_manifest.json"] = try encoder.encode(manifest)
            try LoRAAdapterPersistence.write(
                adapter, to: destination.path, additionalFiles: files)
            outputs.append(SOMMultiSourceMaterializedFinalist(
                depth: finalist.depth, plan: plan,
                objectiveScore: finalist.candidate.objectiveScore,
                rankingMetrics: finalist.candidate.metrics,
                selectionReason: finalist.selectionReason,
                adapterDirectory: destination.path))
            progress?("materialized layer-specific finalist \(plan.stableKey) -> \(destination.path)")
        }
        return outputs
    }

    private static func stablePoolDescription(_ pools: [Int: [Int]]) -> String {
        pools.keys.sorted().map { layer in
            "z\(layer)=[\((pools[layer] ?? []).map(String.init).joined(separator: ","))]"
        }.joined(separator: " ")
    }

    private static func screenProgress(
        prefix: String, value: CapturedCandidate
    ) -> String {
        let metrics = value.ranked.metrics
        let gates = value.ranked.passesContrastKLCeiling
            && value.ranked.passesControlKLCeiling ? "PASS" : "FAIL"
        return prefix + " " + value.ranked.plan.stableKey + " "
            + String(
                format: "score=%+.5f contrastKL=%.5f controlKL=%.5f gates=%@",
                value.ranked.objectiveScore,
                metrics.contrastExactMeanKL,
                metrics.controlExactMeanKL,
                gates)
    }
}

private struct SOMMultiSourceFinalistManifest: Codable {
    let schemaVersion: Int
    let sourceModel: String
    let plan: SOMMultiSourcePlan
    let objectiveScore: Double
    let rankingMetrics: SOMSubsetRankingMetrics
    let selectionReason: SOMSubsetFinalistSelectionReason
    let searchConfiguration: SOMMultiSourceSearchConfiguration
    let composition: AblationComposition
    let normalization: WeightNormalization
    let layerSpecificNoBroadcast: Bool
    let requiredRuntimeAdapterScale: Float
    let exactContrastKLCeiling: Double
    let exactControlKLCeiling: Double
}

public enum SOMMultiSourceSearchError: LocalizedError, Equatable {
    case insufficientSourceLayers(Int)
    case invalidSources([Int])
    case duplicateSourceLayer(Int)
    case inconsistentDirectionWidths
    case invalidPlan(String)
    case invalidPreselectionPerLayer(Int)
    case invalidMaximumDirectionsPerLayer(Int)
    case beamTooNarrow(beamWidth: Int, sourceLayerCount: Int)
    case invalidMinimumSourceLayers(Int)
    case noFeasibleSingleton(layer: Int)

    public var errorDescription: String? {
        switch self {
        case .insufficientSourceLayers(let count):
            "Multi-source SOM search requires at least two distinct source layers, found \(count)."
        case .invalidSources(let layers):
            "SOM source layers must be distinct non-negative indices: \(layers)."
        case .duplicateSourceLayer(let layer):
            "More than one SOM archive claims zero-based source layer \(layer)."
        case .inconsistentDirectionWidths:
            "All layer-specific SOM archives must use the same hidden width."
        case .invalidPlan(let key):
            "Invalid layer-specific SOM plan '\(key)'."
        case .invalidPreselectionPerLayer(let value):
            "SOM preselection per layer must be positive and no larger than the lattice, not \(value)."
        case .invalidMaximumDirectionsPerLayer(let value):
            "Maximum SOM directions per source layer must be positive and no larger than search depth, not \(value)."
        case .beamTooNarrow(let beam, let count):
            "Beam width \(beam) cannot preserve one path for each of \(count) source layers."
        case .invalidMinimumSourceLayers(let value):
            "Minimum source-layer coverage per finalist is invalid: \(value)."
        case .noFeasibleSingleton(let layer):
            "Zero-based source layer \(layer) has no singleton candidate passing both exact KL gates."
        }
    }
}
