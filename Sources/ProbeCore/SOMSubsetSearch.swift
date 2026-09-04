import Foundation
import MLXLMCommon

/// Stable on-disk handoff from SOM extraction to resident subset search.
/// Candidate lattice IDs are the zero-based indices in `candidateDirections`.
public struct SOMCandidateDirectionArchive: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let sourceLayerZeroBased: Int
    public let trainingMode: String
    public let candidateDirections: [[Float]]

    public init(
        schemaVersion: Int, sourceLayerZeroBased: Int, trainingMode: String,
        candidateDirections: [[Float]]
    ) {
        self.schemaVersion = schemaVersion
        self.sourceLayerZeroBased = sourceLayerZeroBased
        self.trainingMode = trainingMode
        self.candidateDirections = candidateDirections
    }

    public static func load(path: String) throws -> Self {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let archive = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        try archive.validate()
        return archive
    }

    public func validate(expectedCandidateCount: Int = 16) throws {
        guard schemaVersion == 1 else {
            throw SOMSubsetSearchError.unsupportedArchiveSchema(schemaVersion)
        }
        guard sourceLayerZeroBased >= 0 else {
            throw SOMSubsetSearchError.invalidSourceLayer(sourceLayerZeroBased)
        }
        guard candidateDirections.count == expectedCandidateCount else {
            throw SOMSubsetSearchError.invalidCandidateCount(
                expected: expectedCandidateCount, actual: candidateDirections.count)
        }
        guard let width = candidateDirections.first?.count, width > 0,
              candidateDirections.allSatisfy({ direction in
                  direction.count == width
                      && direction.allSatisfy(\.isFinite)
                      && direction.contains(where: { $0 != 0 })
              })
        else { throw SOMSubsetSearchError.invalidCandidateDirections }
    }
}

/// Encodes the raw 4x4 SOM candidates from every trained zero-based source
/// layer while preserving the historical filename for the selected source.
///
/// A candidate archive is intrinsically source-layer-specific: the resident
/// subset search rejects an archive whose embedded layer does not match its
/// explicit source-layer argument.  Multi-layer extraction therefore needs
/// one file per trained layer rather than silently retaining only the selected
/// peak's candidates.
public enum SOMCandidateArchiveBundle {
    public static let selectedSourceFilename = "abslayer_som_candidates.json"

    public static func filename(sourceLayerZeroBased: Int) -> String {
        "abslayer_som_candidates_layer_\(sourceLayerZeroBased).json"
    }

    public static func encode(
        resultsByLayer: [Int: SOMDirectionResult],
        selectedSourceLayerZeroBased: Int,
        trainingMode: String
    ) throws -> [String: Data] {
        guard resultsByLayer[selectedSourceLayerZeroBased] != nil else {
            throw SOMCandidateArchiveBundleError.missingSelectedSourceLayer(
                selected: selectedSourceLayerZeroBased,
                trained: resultsByLayer.keys.sorted())
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var files = [String: Data]()
        for layer in resultsByLayer.keys.sorted() {
            guard let result = resultsByLayer[layer] else { continue }
            let archive = SOMCandidateDirectionArchive(
                schemaVersion: 1,
                sourceLayerZeroBased: layer,
                trainingMode: trainingMode,
                candidateDirections: result.candidateDirections)
            try archive.validate()
            let data = try encoder.encode(archive)
            files[filename(sourceLayerZeroBased: layer)] = data
            if layer == selectedSourceLayerZeroBased {
                files[selectedSourceFilename] = data
            }
        }
        return files
    }
}

public enum SOMCandidateArchiveBundleError: LocalizedError, Equatable {
    case missingSelectedSourceLayer(selected: Int, trained: [Int])

    public var errorDescription: String? {
        switch self {
        case .missingSelectedSourceLayer(let selected, let trained):
            "Selected zero-based SOM source layer \(selected) was not trained; trained layers are \(trained)."
        }
    }
}

/// Where the one measured SOM source-layer basis is applied.
public enum SOMApplicationScope: String, Codable, CaseIterable, Sendable {
    /// Apply the source basis to every decoder layer's selected matrices.
    case global
    /// Apply the source basis only at the source decoder layer.
    case local
}

public enum SOMApplicationComponents: String, Codable, CaseIterable, Sendable {
    case attention
    case mlp
    case omlp

    var includesAttention: Bool { self != .mlp }
    var includesMLP: Bool { self != .attention }
}

/// Explicit scalar used only to order the bounded search beam. Its starter
/// terms are deliberately reported as a proxy rather than semantic compliance.
public struct SOMSubsetSearchObjective: Codable, Equatable, Sendable {
    public let controlKLPenalty: Double
    public let controlStarterShiftPenalty: Double
    public let contrastKLPenalty: Double

    public init(
        controlKLPenalty: Double = 1,
        controlStarterShiftPenalty: Double = 0.5,
        contrastKLPenalty: Double = 0.25
    ) {
        self.controlKLPenalty = controlKLPenalty
        self.controlStarterShiftPenalty = controlStarterShiftPenalty
        self.contrastKLPenalty = contrastKLPenalty
    }

    public static let `default` = Self()

    public var definition: String {
        "contrast starter log-odds delta"
            + " - \(controlKLPenalty) * control exact first-token KL"
            + " - \(controlStarterShiftPenalty) * abs(control starter log-odds delta)"
            + " - \(contrastKLPenalty) * contrast exact first-token KL"
    }

    public func validate() throws {
        guard controlKLPenalty.isFinite, controlKLPenalty >= 0,
              controlStarterShiftPenalty.isFinite, controlStarterShiftPenalty >= 0,
              contrastKLPenalty.isFinite, contrastKLPenalty >= 0
        else { throw SOMSubsetSearchError.invalidObjective }
    }

    public func score(_ metrics: SOMSubsetRankingMetrics) -> Double {
        metrics.contrastStarterLogOddsDelta
            - controlKLPenalty * metrics.controlExactMeanKL
            - controlStarterShiftPenalty * abs(metrics.controlStarterLogOddsDelta)
            - contrastKLPenalty * metrics.contrastExactMeanKL
    }
}

public struct SOMSubsetSearchConfiguration: Codable, Equatable, Sendable {
    public let sourceLayerZeroBased: Int
    public let applicationScope: SOMApplicationScope
    public let components: SOMApplicationComponents
    public let maximumCases: Int
    public let maximumDepth: Int
    public let beamWidth: Int
    public let maximumFinalists: Int
    public let contrastKLCeiling: Double
    public let controlKLCeiling: Double
    public let unloadTolerance: Double
    public let objective: SOMSubsetSearchObjective

    public init(
        sourceLayerZeroBased: Int,
        applicationScope: SOMApplicationScope,
        components: SOMApplicationComponents = .omlp,
        maximumCases: Int,
        maximumDepth: Int = 7,
        beamWidth: Int = 4,
        maximumFinalists: Int = 8,
        contrastKLCeiling: Double = 1,
        controlKLCeiling: Double = 1,
        unloadTolerance: Double = 1e-5,
        objective: SOMSubsetSearchObjective = .default
    ) {
        self.sourceLayerZeroBased = sourceLayerZeroBased
        self.applicationScope = applicationScope
        self.components = components
        self.maximumCases = maximumCases
        self.maximumDepth = maximumDepth
        self.beamWidth = beamWidth
        self.maximumFinalists = maximumFinalists
        self.contrastKLCeiling = contrastKLCeiling
        self.controlKLCeiling = controlKLCeiling
        self.unloadTolerance = unloadTolerance
        self.objective = objective
    }

    public func validate(candidateCount: Int) throws {
        guard sourceLayerZeroBased >= 0 else {
            throw SOMSubsetSearchError.invalidSourceLayer(sourceLayerZeroBased)
        }
        guard maximumCases > 0 else {
            throw FirstTokenScreenError.invalidMaximumCases(maximumCases)
        }
        guard maximumDepth > 0, maximumDepth <= 7,
              maximumDepth <= candidateCount
        else { throw SOMSubsetSearchError.invalidMaximumDepth(maximumDepth) }
        guard beamWidth > 0 else { throw SOMSubsetSearchError.invalidBeamWidth(beamWidth) }
        guard maximumFinalists > 0 else {
            throw SOMSubsetSearchError.invalidMaximumFinalists(maximumFinalists)
        }
        guard contrastKLCeiling.isFinite, contrastKLCeiling >= 0 else {
            throw SOMSubsetSearchError.invalidContrastKLCeiling(contrastKLCeiling)
        }
        guard controlKLCeiling.isFinite, controlKLCeiling >= 0 else {
            throw SOMSubsetSearchError.invalidControlKLCeiling(controlKLCeiling)
        }
        guard unloadTolerance.isFinite, unloadTolerance >= 0 else {
            throw FirstTokenScreenError.invalidUnloadTolerance(unloadTolerance)
        }
        try objective.validate()
    }
}

/// Compact values used for Pareto membership and scalar beam ordering.
public struct SOMSubsetRankingMetrics: Codable, Equatable, Sendable {
    public let contrastStarterLogOddsDelta: Double
    public let contrastExactMeanKL: Double
    public let controlExactMeanKL: Double
    public let controlStarterLogOddsDelta: Double

    public init(
        contrastStarterLogOddsDelta: Double,
        contrastExactMeanKL: Double,
        controlExactMeanKL: Double,
        controlStarterLogOddsDelta: Double
    ) {
        self.contrastStarterLogOddsDelta = contrastStarterLogOddsDelta
        self.contrastExactMeanKL = contrastExactMeanKL
        self.controlExactMeanKL = controlExactMeanKL
        self.controlStarterLogOddsDelta = controlStarterLogOddsDelta
    }

    var isFinite: Bool {
        contrastStarterLogOddsDelta.isFinite
            && contrastExactMeanKL.isFinite
            && controlExactMeanKL.isFinite
            && controlStarterLogOddsDelta.isFinite
    }
}

public struct SOMSubsetSearchCandidateReport: Codable, Equatable, Sendable {
    public let orderedLatticeIDs: [Int]
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

    public init(
        orderedLatticeIDs: [Int], objectiveScore: Double,
        passesContrastKLCeiling: Bool, passesControlKLCeiling: Bool,
        passesKLCeilings: Bool, isParetoAtDepth: Bool,
        retainedForExpansion: Bool,
        rankingMetrics: SOMSubsetRankingMetrics,
        contrast: FirstTokenCandidateChannelReport,
        control: FirstTokenCandidateChannelReport,
        unloadValidation: FirstTokenUnloadValidation
    ) {
        self.orderedLatticeIDs = orderedLatticeIDs
        self.objectiveScore = objectiveScore
        self.passesContrastKLCeiling = passesContrastKLCeiling
        self.passesControlKLCeiling = passesControlKLCeiling
        self.passesKLCeilings = passesKLCeilings
        self.isParetoAtDepth = isParetoAtDepth
        self.retainedForExpansion = retainedForExpansion
        self.rankingMetrics = rankingMetrics
        self.contrast = contrast
        self.control = control
        self.unloadValidation = unloadValidation
    }
}

public struct SOMSubsetSearchDepthReport: Codable, Equatable, Sendable {
    public let depth: Int
    public let generatedCandidateCount: Int
    public let retainedOrderedLatticeIDs: [[Int]]
    public let candidates: [SOMSubsetSearchCandidateReport]

    public init(
        depth: Int, generatedCandidateCount: Int,
        retainedOrderedLatticeIDs: [[Int]],
        candidates: [SOMSubsetSearchCandidateReport]
    ) {
        self.depth = depth
        self.generatedCandidateCount = generatedCandidateCount
        self.retainedOrderedLatticeIDs = retainedOrderedLatticeIDs
        self.candidates = candidates
    }
}

public enum SOMSubsetFinalistSelectionReason: String, Codable, Equatable, Sendable {
    /// The highest-ranked feasible candidate retained to represent this depth.
    case bestFeasibleAtDepth
    /// An additional high-ranked feasible candidate selected after depth coverage.
    case globalTop
}

public struct SOMSubsetSearchFinalist: Codable, Equatable, Sendable {
    public let depth: Int
    public let orderedLatticeIDs: [Int]
    public let objectiveScore: Double
    public let rankingMetrics: SOMSubsetRankingMetrics
    public let isParetoAtDepth: Bool
    public let passesContrastKLCeiling: Bool
    public let passesControlKLCeiling: Bool
    public let selectionReason: SOMSubsetFinalistSelectionReason
}

public struct SOMSubsetMaterializedFinalist: Codable, Equatable, Sendable {
    public let depth: Int
    public let orderedLatticeIDs: [Int]
    public let objectiveScore: Double
    public let rankingMetrics: SOMSubsetRankingMetrics
    public let isParetoAtDepth: Bool
    public let passesContrastKLCeiling: Bool
    public let passesControlKLCeiling: Bool
    public let selectionReason: SOMSubsetFinalistSelectionReason
    public let adapterDirectory: String
}

public struct SOMSubsetSearchReport: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: String
    public let modelDirectory: String
    public let promptFile: String
    public let candidateArchiveFile: String
    public let candidateTrainingMode: String
    public let selectedCaseNames: [String]
    public let vocabularySize: Int
    public let configuration: SOMSubsetSearchConfiguration
    public let composition: AblationComposition
    public let runtimeAdapterScale: Float
    public let proxyNotice: String
    public let objectiveDefinition: String
    public let paretoDefinition: String
    public let starterMetricDefinition: String
    public let starterVocabulary: FirstTokenStarterVocabulary
    public let baselineContrast: FirstTokenBaselineChannelReport
    public let baselineControl: FirstTokenBaselineChannelReport
    public let depths: [SOMSubsetSearchDepthReport]
    public let selectedFinalists: [SOMSubsetSearchFinalist]
    public let materializedFinalists: [SOMSubsetMaterializedFinalist]

    public init(
        schemaVersion: Int = 2, createdAt: String,
        modelDirectory: String, promptFile: String,
        candidateArchiveFile: String, candidateTrainingMode: String,
        selectedCaseNames: [String], vocabularySize: Int,
        configuration: SOMSubsetSearchConfiguration,
        composition: AblationComposition = .sequential,
        runtimeAdapterScale: Float = 1,
        proxyNotice: String = FirstTokenScreenEngine.proxyNotice,
        objectiveDefinition: String,
        paretoDefinition: String = SOMSubsetSearchMath.paretoDefinition,
        starterMetricDefinition: String = FirstTokenScreenEngine.starterMetricDefinition,
        starterVocabulary: FirstTokenStarterVocabulary,
        baselineContrast: FirstTokenBaselineChannelReport,
        baselineControl: FirstTokenBaselineChannelReport,
        depths: [SOMSubsetSearchDepthReport],
        selectedFinalists: [SOMSubsetSearchFinalist],
        materializedFinalists: [SOMSubsetMaterializedFinalist]
    ) {
        self.schemaVersion = schemaVersion
        self.createdAt = createdAt
        self.modelDirectory = modelDirectory
        self.promptFile = promptFile
        self.candidateArchiveFile = candidateArchiveFile
        self.candidateTrainingMode = candidateTrainingMode
        self.selectedCaseNames = selectedCaseNames
        self.vocabularySize = vocabularySize
        self.configuration = configuration
        self.composition = composition
        self.runtimeAdapterScale = runtimeAdapterScale
        self.proxyNotice = proxyNotice
        self.objectiveDefinition = objectiveDefinition
        self.paretoDefinition = paretoDefinition
        self.starterMetricDefinition = starterMetricDefinition
        self.starterVocabulary = starterVocabulary
        self.baselineContrast = baselineContrast
        self.baselineControl = baselineControl
        self.depths = depths
        self.selectedFinalists = selectedFinalists
        self.materializedFinalists = materializedFinalists
    }
}

/// Pure deterministic beam/Pareto operations, separated for fast tests.
public enum SOMSubsetSearchMath {
    public static let paretoDefinition =
        "Within the candidates that pass both contrast- and control-KL ceilings at one depth, a candidate is Pareto if no other candidate has at least as large a contrast starter log-odds delta and no larger contrast KL, control KL, or absolute control starter log-odds delta, with at least one strict improvement. This is still a first-token proxy."

    public struct RankedCandidate: Equatable, Sendable {
        public let orderedLatticeIDs: [Int]
        public let metrics: SOMSubsetRankingMetrics
        public let objectiveScore: Double
        public let passesContrastKLCeiling: Bool
        public let passesControlKLCeiling: Bool

        public init(
            orderedLatticeIDs: [Int], metrics: SOMSubsetRankingMetrics,
            objectiveScore: Double, passesContrastKLCeiling: Bool,
            passesControlKLCeiling: Bool
        ) {
            self.orderedLatticeIDs = orderedLatticeIDs
            self.metrics = metrics
            self.objectiveScore = objectiveScore
            self.passesContrastKLCeiling = passesContrastKLCeiling
            self.passesControlKLCeiling = passesControlKLCeiling
        }
    }

    public struct DepthCandidates: Equatable, Sendable {
        public let depth: Int
        public let candidates: [RankedCandidate]

        public init(depth: Int, candidates: [RankedCandidate]) {
            self.depth = depth
            self.candidates = candidates
        }
    }

    public struct FinalistSelection: Equatable, Sendable {
        public let depth: Int
        public let candidate: RankedCandidate
        public let isParetoAtDepth: Bool
        public let selectionReason: SOMSubsetFinalistSelectionReason

        public init(
            depth: Int, candidate: RankedCandidate, isParetoAtDepth: Bool,
            selectionReason: SOMSubsetFinalistSelectionReason
        ) {
            self.depth = depth
            self.candidate = candidate
            self.isParetoAtDepth = isParetoAtDepth
            self.selectionReason = selectionReason
        }
    }

    public static func expandedSequences(
        candidateCount: Int, previousBeam: [[Int]]?
    ) throws -> [[Int]] {
        guard candidateCount > 0 else {
            throw SOMSubsetSearchError.invalidCandidateCount(
                expected: 1, actual: candidateCount)
        }
        guard let previousBeam else { return (0 ..< candidateCount).map { [$0] } }
        var seen = Set<String>()
        var result = [[Int]]()
        for prefix in previousBeam {
            guard !prefix.isEmpty, Set(prefix).count == prefix.count,
                  prefix.allSatisfy({ $0 >= 0 && $0 < candidateCount })
            else { throw SOMSubsetSearchError.invalidBeamSequence(prefix) }
            let used = Set(prefix)
            for id in 0 ..< candidateCount where !used.contains(id) {
                let sequence = prefix + [id]
                let key = sequence.map(String.init).joined(separator: ",")
                if seen.insert(key).inserted { result.append(sequence) }
            }
        }
        return result
    }

    public static func paretoSequences(
        _ candidates: [RankedCandidate]
    ) -> Set<String> {
        let feasible = candidates.filter {
            $0.passesContrastKLCeiling && $0.passesControlKLCeiling
                && $0.metrics.isFinite
        }
        return Set(feasible.compactMap { candidate in
            let dominated = feasible.contains { other in
                guard other.orderedLatticeIDs != candidate.orderedLatticeIDs else {
                    return false
                }
                return dominates(other.metrics, candidate.metrics)
            }
            return dominated ? nil : sequenceKey(candidate.orderedLatticeIDs)
        })
    }

    public static func selectBeam(
        _ candidates: [RankedCandidate], width: Int
    ) throws -> [[Int]] {
        guard width > 0 else { throw SOMSubsetSearchError.invalidBeamWidth(width) }
        let feasible = candidates.filter {
            $0.passesContrastKLCeiling
                && $0.passesControlKLCeiling
                && $0.metrics.isFinite
                && $0.objectiveScore.isFinite
        }
        let pareto = paretoSequences(feasible)
        let sorted = feasible.sorted {
            rankedPrecedes($0, $1, paretoKeys: pareto)
        }
        var seen = Set<String>()
        return sorted.compactMap { candidate in
            let key = sequenceKey(candidate.orderedLatticeIDs)
            return seen.insert(key).inserted ? candidate.orderedLatticeIDs : nil
        }.prefix(width).map { $0 }
    }

    /// Selects a bounded set of safe finalists without assuming the deepest
    /// reached beam is viable. One best feasible candidate from each reached
    /// depth is considered first; remaining capacity is filled by the strongest
    /// feasible candidates globally. Both KL gates are hard filters in both
    /// stages, so an attractive scalar score cannot admit a destructive result.
    public static func selectFinalists(
        _ reachedDepths: [DepthCandidates], maximumCount: Int
    ) throws -> [FinalistSelection] {
        guard maximumCount > 0 else {
            throw SOMSubsetSearchError.invalidMaximumFinalists(maximumCount)
        }

        struct Annotated {
            let depth: Int
            let candidate: RankedCandidate
            let isParetoAtDepth: Bool
        }

        let orderedDepths = reachedDepths.sorted { lhs, rhs in
            if lhs.depth != rhs.depth { return lhs.depth < rhs.depth }
            let lhsKey = lhs.candidates.map { sequenceKey($0.orderedLatticeIDs) }
                .sorted().joined(separator: ";")
            let rhsKey = rhs.candidates.map { sequenceKey($0.orderedLatticeIDs) }
                .sorted().joined(separator: ";")
            return lhsKey < rhsKey
        }
        var bestByDepth = [Annotated]()
        var globallyEligible = [Annotated]()
        for reached in orderedDepths {
            let feasible = reached.candidates.filter(isFinalistEligible)
            let paretoKeys = paretoSequences(feasible)
            let sorted = feasible.sorted {
                rankedPrecedes($0, $1, paretoKeys: paretoKeys)
            }
            let annotated = sorted.map { candidate in
                Annotated(
                    depth: reached.depth,
                    candidate: candidate,
                    isParetoAtDepth: paretoKeys.contains(
                        sequenceKey(candidate.orderedLatticeIDs)))
            }
            if let best = annotated.first { bestByDepth.append(best) }
            globallyEligible.append(contentsOf: annotated)
        }

        var selected = [FinalistSelection]()
        var selectedKeys = Set<String>()
        func key(_ value: Annotated) -> String {
            "\(value.depth):\(sequenceKey(value.candidate.orderedLatticeIDs))"
        }
        for value in bestByDepth where selected.count < maximumCount {
            guard selectedKeys.insert(key(value)).inserted else { continue }
            selected.append(FinalistSelection(
                depth: value.depth,
                candidate: value.candidate,
                isParetoAtDepth: value.isParetoAtDepth,
                selectionReason: .bestFeasibleAtDepth))
        }

        let global = globallyEligible.sorted { lhs, rhs in
            let lhsUseful = lhs.isParetoAtDepth
                && lhs.candidate.metrics.contrastStarterLogOddsDelta > 0
            let rhsUseful = rhs.isParetoAtDepth
                && rhs.candidate.metrics.contrastStarterLogOddsDelta > 0
            if lhsUseful != rhsUseful { return lhsUseful }
            if lhs.isParetoAtDepth != rhs.isParetoAtDepth {
                return lhs.isParetoAtDepth
            }
            if lhs.candidate.objectiveScore != rhs.candidate.objectiveScore {
                return lhs.candidate.objectiveScore > rhs.candidate.objectiveScore
            }
            if lhs.candidate.metrics.controlExactMeanKL
                != rhs.candidate.metrics.controlExactMeanKL
            {
                return lhs.candidate.metrics.controlExactMeanKL
                    < rhs.candidate.metrics.controlExactMeanKL
            }
            if lhs.candidate.metrics.contrastExactMeanKL
                != rhs.candidate.metrics.contrastExactMeanKL
            {
                return lhs.candidate.metrics.contrastExactMeanKL
                    < rhs.candidate.metrics.contrastExactMeanKL
            }
            if lhs.depth != rhs.depth { return lhs.depth < rhs.depth }
            return lexicographicallyPrecedes(
                lhs.candidate.orderedLatticeIDs,
                rhs.candidate.orderedLatticeIDs)
        }
        for value in global where selected.count < maximumCount {
            guard selectedKeys.insert(key(value)).inserted else { continue }
            selected.append(FinalistSelection(
                depth: value.depth,
                candidate: value.candidate,
                isParetoAtDepth: value.isParetoAtDepth,
                selectionReason: .globalTop))
        }
        return selected
    }

    private static func isFinalistEligible(_ candidate: RankedCandidate) -> Bool {
        candidate.passesContrastKLCeiling
            && candidate.passesControlKLCeiling
            && candidate.metrics.isFinite
            && candidate.objectiveScore.isFinite
    }

    private static func rankedPrecedes(
        _ lhs: RankedCandidate, _ rhs: RankedCandidate,
        paretoKeys: Set<String>
    ) -> Bool {
        let lhsPareto = paretoKeys.contains(sequenceKey(lhs.orderedLatticeIDs))
        let rhsPareto = paretoKeys.contains(sequenceKey(rhs.orderedLatticeIDs))
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
        return lexicographicallyPrecedes(
            lhs.orderedLatticeIDs, rhs.orderedLatticeIDs)
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
        let strictlyBetter = lhs.contrastStarterLogOddsDelta
                > rhs.contrastStarterLogOddsDelta
            || lhs.contrastExactMeanKL < rhs.contrastExactMeanKL
            || lhs.controlExactMeanKL < rhs.controlExactMeanKL
            || abs(lhs.controlStarterLogOddsDelta)
                < abs(rhs.controlStarterLogOddsDelta)
        return noWorse && strictlyBetter
    }

    static func sequenceKey(_ sequence: [Int]) -> String {
        sequence.map(String.init).joined(separator: ",")
    }

    private static func lexicographicallyPrecedes(
        _ lhs: [Int], _ rhs: [Int]
    ) -> Bool {
        for (left, right) in zip(lhs, rhs) where left != right { return left < right }
        return lhs.count < rhs.count
    }
}

public enum SOMSubsetSearchEngine {
    public static func run(
        modelDirectory: String,
        promptFile: String,
        pairs: [PromptPair],
        candidateArchiveFile: String,
        configuration: SOMSubsetSearchConfiguration,
        starterConfiguration: FirstTokenStarterConfiguration = .default,
        finalistDirectory: String? = nil,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> SOMSubsetSearchReport {
        let archivePath = URL(fileURLWithPath: candidateArchiveFile).standardizedFileURL.path
        let archiveData = try Data(contentsOf: URL(fileURLWithPath: archivePath))
        let archive = try JSONDecoder().decode(
            SOMCandidateDirectionArchive.self, from: archiveData)
        try archive.validate()
        try configuration.validate(candidateCount: archive.candidateDirections.count)
        guard configuration.sourceLayerZeroBased == archive.sourceLayerZeroBased else {
            throw SOMSubsetSearchError.sourceLayerMismatch(
                requested: configuration.sourceLayerZeroBased,
                archive: archive.sourceLayerZeroBased)
        }

        let selected = try FirstTokenScreenMath.evenlySpaced(
            pairs, maximum: configuration.maximumCases)
        let runtime = try await FirstTokenScreenRuntime(modelDirectory: modelDirectory)
        let starters = try await runtime.resolveStarterVocabulary(starterConfiguration)
        let baseline = try await runtime.capture(pairs: selected)
        let baselineSentinel = FirstTokenScreenMath.prefix(baseline, count: 1)
        let baselineContrast = try FirstTokenScreenMath.baselineReport(
            fingerprint: baseline.contrast, vocabulary: starters)
        let baselineControl = try FirstTokenScreenMath.baselineReport(
            fingerprint: baseline.control, vocabulary: starters)
        progress?("captured untouched baseline over \(selected.count) matched pairs")

        var depthReports = [SOMSubsetSearchDepthReport]()
        var reachedDepthCandidates = [SOMSubsetSearchMath.DepthCandidates]()
        var previousBeam: [[Int]]?
        for depth in 1 ... configuration.maximumDepth {
            let sequences = try SOMSubsetSearchMath.expandedSequences(
                candidateCount: archive.candidateDirections.count,
                previousBeam: previousBeam)
            guard !sequences.isEmpty else { break }
            var captured = [(ranked: SOMSubsetSearchMath.RankedCandidate,
                             contrast: FirstTokenCandidateChannelReport,
                             control: FirstTokenCandidateChannelReport,
                             unload: FirstTokenUnloadValidation)]()
            captured.reserveCapacity(sequences.count)

            for (index, sequence) in sequences.enumerated() {
                let sourceDirections = sequence.map { archive.candidateDirections[$0] }
                let adapter = try await runtime.makeSequentialSOMAdapter(
                    sourceDirections: sourceDirections,
                    sourceLayerZeroBased: configuration.sourceLayerZeroBased,
                    applicationScope: configuration.applicationScope,
                    components: configuration.components)
                do {
                    try await runtime.loadAdapter(adapter)
                } catch {
                    await runtime.unloadAdapter(adapter)
                    throw error
                }
                let candidate: FirstTokenDualFingerprint
                do {
                    candidate = try await runtime.capture(pairs: selected)
                } catch {
                    await runtime.unloadAdapter(adapter)
                    throw error
                }
                await runtime.unloadAdapter(adapter)

                let restored = try await runtime.capture(pairs: [selected[0]])
                let unload = try FirstTokenScreenMath.unloadValidation(
                    baseline: baselineSentinel, restored: restored,
                    tolerance: configuration.unloadTolerance)
                guard unload.passed else {
                    throw FirstTokenScreenError.adapterUnloadDidNotRestore(
                        directory: "in-memory SOM IDs [\(SOMSubsetSearchMath.sequenceKey(sequence))]",
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
                let ranked = SOMSubsetSearchMath.RankedCandidate(
                    orderedLatticeIDs: sequence,
                    metrics: metrics,
                    objectiveScore: configuration.objective.score(metrics),
                    passesContrastKLCeiling:
                        contrast.exactMeanKLFromBaseline
                            <= configuration.contrastKLCeiling,
                    passesControlKLCeiling:
                        control.exactMeanKLFromBaseline <= configuration.controlKLCeiling)
                captured.append((ranked, contrast, control, unload))
                progress?(
                    "depth \(depth) screened \(index + 1)/\(sequences.count) "
                        + "IDs=[\(SOMSubsetSearchMath.sequenceKey(sequence))] "
                        + String(
                            format: "score=%+.5f contrastKL=%.5f controlKL=%.5f",
                            ranked.objectiveScore,
                            metrics.contrastExactMeanKL,
                            metrics.controlExactMeanKL))
            }

            let ranked = captured.map(\.ranked)
            reachedDepthCandidates.append(.init(depth: depth, candidates: ranked))
            let retained = try SOMSubsetSearchMath.selectBeam(
                ranked, width: configuration.beamWidth)
            let retainedKeys = Set(retained.map(SOMSubsetSearchMath.sequenceKey))
            let paretoKeys = SOMSubsetSearchMath.paretoSequences(ranked)
            let reports = captured.map { value in
                let key = SOMSubsetSearchMath.sequenceKey(value.ranked.orderedLatticeIDs)
                return SOMSubsetSearchCandidateReport(
                    orderedLatticeIDs: value.ranked.orderedLatticeIDs,
                    objectiveScore: value.ranked.objectiveScore,
                    passesContrastKLCeiling:
                        value.ranked.passesContrastKLCeiling,
                    passesControlKLCeiling: value.ranked.passesControlKLCeiling,
                    passesKLCeilings:
                        value.ranked.passesContrastKLCeiling
                            && value.ranked.passesControlKLCeiling,
                    isParetoAtDepth: paretoKeys.contains(key),
                    retainedForExpansion: retainedKeys.contains(key),
                    rankingMetrics: value.ranked.metrics,
                    contrast: value.contrast,
                    control: value.control,
                    unloadValidation: value.unload)
            }
            depthReports.append(SOMSubsetSearchDepthReport(
                depth: depth,
                generatedCandidateCount: sequences.count,
                retainedOrderedLatticeIDs: retained,
                candidates: reports))
            previousBeam = retained
            progress?("depth \(depth) retained \(retained.count)/\(sequences.count)")
            if retained.isEmpty { break }
        }

        let finalists = try SOMSubsetSearchMath.selectFinalists(
            reachedDepthCandidates,
            maximumCount: configuration.maximumFinalists)
        let finalistReports = finalists.map { finalist in
            SOMSubsetSearchFinalist(
                depth: finalist.depth,
                orderedLatticeIDs: finalist.candidate.orderedLatticeIDs,
                objectiveScore: finalist.candidate.objectiveScore,
                rankingMetrics: finalist.candidate.metrics,
                isParetoAtDepth: finalist.isParetoAtDepth,
                passesContrastKLCeiling:
                    finalist.candidate.passesContrastKLCeiling,
                passesControlKLCeiling:
                    finalist.candidate.passesControlKLCeiling,
                selectionReason: finalist.selectionReason)
        }
        progress?("selected \(finalists.count) safe finalists across reached depths")
        let materialized = try await materializeFinalists(
            finalists: finalists,
            archive: archive,
            archiveData: archiveData,
            archivePath: archivePath,
            modelDirectory: modelDirectory,
            configuration: configuration,
            runtime: runtime,
            rootDirectory: finalistDirectory,
            progress: progress)
        return SOMSubsetSearchReport(
            createdAt: ISO8601DateFormatter().string(from: Date()),
            modelDirectory: URL(fileURLWithPath: modelDirectory).standardizedFileURL.path,
            promptFile: URL(fileURLWithPath: promptFile).standardizedFileURL.path,
            candidateArchiveFile: archivePath,
            candidateTrainingMode: archive.trainingMode,
            selectedCaseNames: selected.map(\.name),
            vocabularySize: baseline.contrast.vocabularySize,
            configuration: configuration,
            objectiveDefinition: configuration.objective.definition,
            starterVocabulary: starters,
            baselineContrast: baselineContrast,
            baselineControl: baselineControl,
            depths: depthReports,
            selectedFinalists: finalistReports,
            materializedFinalists: materialized)
    }

    private static func materializeFinalists(
        finalists: [SOMSubsetSearchMath.FinalistSelection],
        archive: SOMCandidateDirectionArchive,
        archiveData: Data,
        archivePath: String,
        modelDirectory: String,
        configuration: SOMSubsetSearchConfiguration,
        runtime: FirstTokenScreenRuntime,
        rootDirectory: String?,
        progress: (@Sendable (String) -> Void)?
    ) async throws -> [SOMSubsetMaterializedFinalist] {
        guard let rootDirectory else { return [] }
        let root = URL(fileURLWithPath: rootDirectory).standardizedFileURL
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var outputs = [SOMSubsetMaterializedFinalist]()
        for finalist in finalists {
            let candidate = finalist.candidate
            let ids = candidate.orderedLatticeIDs
            let destination = root.appendingPathComponent(
                "k\(ids.count)-ids-\(ids.map(String.init).joined(separator: "-"))",
                isDirectory: true)
            let adapter = try await runtime.makeSequentialSOMAdapter(
                sourceDirections: ids.map { archive.candidateDirections[$0] },
                sourceLayerZeroBased: configuration.sourceLayerZeroBased,
                applicationScope: configuration.applicationScope,
                components: configuration.components)
            let manifest = SOMSubsetFinalistManifest(
                schemaVersion: 2,
                sourceModel: URL(fileURLWithPath: modelDirectory).standardizedFileURL.path,
                candidateArchiveFile: archivePath,
                sourceLayerZeroBased: configuration.sourceLayerZeroBased,
                applicationScope: configuration.applicationScope,
                components: configuration.components,
                depth: finalist.depth,
                orderedLatticeIDs: ids,
                objectiveScore: candidate.objectiveScore,
                rankingMetrics: candidate.metrics,
                isParetoAtDepth: finalist.isParetoAtDepth,
                passesContrastKLCeiling: candidate.passesContrastKLCeiling,
                passesControlKLCeiling: candidate.passesControlKLCeiling,
                selectionReason: finalist.selectionReason,
                searchConfiguration: configuration,
                composition: .sequential,
                normalization: .none,
                requiredRuntimeAdapterScale: 1)
            try LoRAAdapterPersistence.write(
                adapter, to: destination.path,
                additionalFiles: [
                    "abslayer_som_candidates.json": archiveData,
                    "abslayer_som_search_manifest.json": try encoder.encode(manifest),
                ])
            outputs.append(SOMSubsetMaterializedFinalist(
                depth: finalist.depth,
                orderedLatticeIDs: ids,
                objectiveScore: candidate.objectiveScore,
                rankingMetrics: candidate.metrics,
                isParetoAtDepth: finalist.isParetoAtDepth,
                passesContrastKLCeiling: candidate.passesContrastKLCeiling,
                passesControlKLCeiling: candidate.passesControlKLCeiling,
                selectionReason: finalist.selectionReason,
                adapterDirectory: destination.path))
            progress?("materialized finalist IDs=[\(SOMSubsetSearchMath.sequenceKey(ids))] -> \(destination.path)")
        }
        return outputs
    }
}

private struct SOMSubsetFinalistManifest: Codable {
    let schemaVersion: Int
    let sourceModel: String
    let candidateArchiveFile: String
    let sourceLayerZeroBased: Int
    let applicationScope: SOMApplicationScope
    let components: SOMApplicationComponents
    let depth: Int
    let orderedLatticeIDs: [Int]
    let objectiveScore: Double
    let rankingMetrics: SOMSubsetRankingMetrics
    let isParetoAtDepth: Bool
    let passesContrastKLCeiling: Bool
    let passesControlKLCeiling: Bool
    let selectionReason: SOMSubsetFinalistSelectionReason
    let searchConfiguration: SOMSubsetSearchConfiguration
    let composition: AblationComposition
    let normalization: WeightNormalization
    let requiredRuntimeAdapterScale: Float
}

public enum SOMSubsetSearchError: LocalizedError, Equatable {
    case unsupportedArchiveSchema(Int)
    case invalidCandidateCount(expected: Int, actual: Int)
    case invalidCandidateDirections
    case invalidSourceLayer(Int)
    case sourceLayerMismatch(requested: Int, archive: Int)
    case sourceLayerOutsideModel(requested: Int, layerCount: Int)
    case invalidMaximumDepth(Int)
    case invalidBeamWidth(Int)
    case invalidMaximumFinalists(Int)
    case invalidContrastKLCeiling(Double)
    case invalidControlKLCeiling(Double)
    case invalidObjective
    case invalidBeamSequence([Int])

    public var errorDescription: String? {
        switch self {
        case .unsupportedArchiveSchema(let version):
            "Unsupported SOM candidate archive schema version \(version)."
        case .invalidCandidateCount(let expected, let actual):
            "SOM subset search expected \(expected) candidate directions, found \(actual)."
        case .invalidCandidateDirections:
            "SOM candidate directions must be non-empty, finite, nonzero, and have one common width."
        case .invalidSourceLayer(let value):
            "The SOM source layer must be zero-based and non-negative, not \(value)."
        case .sourceLayerMismatch(let requested, let archive):
            "The requested zero-based source layer \(requested) does not match archive layer \(archive)."
        case .sourceLayerOutsideModel(let requested, let count):
            "The requested zero-based source layer \(requested) is outside the model's 0..<\(count) layers."
        case .invalidMaximumDepth(let value):
            "SOM beam depth must be within 1...7 and no larger than the candidate count, not \(value)."
        case .invalidBeamWidth(let value):
            "SOM beam width must be positive, not \(value)."
        case .invalidMaximumFinalists(let value):
            "The maximum number of SOM finalists must be positive, not \(value)."
        case .invalidContrastKLCeiling(let value):
            "Contrast exact first-token KL ceiling must be finite and non-negative, not \(value)."
        case .invalidControlKLCeiling(let value):
            "Control exact first-token KL ceiling must be finite and non-negative, not \(value)."
        case .invalidObjective:
            "SOM search objective penalties must be finite and non-negative."
        case .invalidBeamSequence(let sequence):
            "Invalid SOM beam prefix: \(sequence)."
        }
    }
}
