import Foundation

/// Opt-in configuration for learning edit bases from multiple semantic token
/// roles while testing them independently at application layers.
public struct TrajectoryDirectionSearchConfiguration: Equatable, Sendable {
    public static let positionsKey = "ABSLAYER_DIRECTION_POSITIONS"
    public static let layersPerPositionKey =
        "ABSLAYER_DIRECTION_LAYERS_PER_POSITION"

    public let positions: [ActivationTokenPosition]
    public let layersPerPosition: Int

    public init?(
        positions: [ActivationTokenPosition], layersPerPosition: Int
    ) {
        guard !positions.isEmpty,
              Set(positions).count == positions.count,
              layersPerPosition > 0
        else { return nil }
        self.positions = positions
        self.layersPerPosition = layersPerPosition
    }

    /// Absence of both variables preserves the exact legacy diagnostic path.
    /// Naming positions opts into trajectory mode; the conservative default
    /// keeps its initial source/application cross-product bounded.
    public static func parse(
        environment: [String: String],
        defaultLayersPerPosition: Int = 2
    ) throws -> Self? {
        guard defaultLayersPerPosition > 0 else {
            throw TrajectoryDirectionSearchConfigurationError
                .invalidLayersPerPosition(String(defaultLayersPerPosition))
        }
        guard let rawPositions = environment[positionsKey] else {
            if environment[layersPerPositionKey] != nil {
                throw TrajectoryDirectionSearchConfigurationError
                    .positionsRequired
            }
            return nil
        }

        let pieces = rawPositions.split(
            separator: ",", omittingEmptySubsequences: false)
        var seen = Set<ActivationTokenPosition>()
        var positions = [ActivationTokenPosition]()
        for piece in pieces {
            let name = piece.trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard !name.isEmpty,
                  let position = ActivationTokenPosition(rawValue: name)
            else {
                throw TrajectoryDirectionSearchConfigurationError
                    .invalidPositions(rawPositions)
            }
            guard seen.insert(position).inserted else {
                throw TrajectoryDirectionSearchConfigurationError
                    .duplicatePosition(position)
            }
            positions.append(position)
        }
        guard !positions.isEmpty else {
            throw TrajectoryDirectionSearchConfigurationError
                .invalidPositions(rawPositions)
        }

        let layersPerPosition: Int
        if let rawLayers = environment[layersPerPositionKey] {
            let trimmed = rawLayers.trimmingCharacters(
                in: .whitespacesAndNewlines)
            guard let parsed = Int(trimmed), parsed > 0 else {
                throw TrajectoryDirectionSearchConfigurationError
                    .invalidLayersPerPosition(rawLayers)
            }
            layersPerPosition = parsed
        } else {
            layersPerPosition = defaultLayersPerPosition
        }
        guard let configuration = Self(
            positions: positions, layersPerPosition: layersPerPosition)
        else {
            throw TrajectoryDirectionSearchConfigurationError
                .invalidPositions(rawPositions)
        }
        return configuration
    }
}

public enum TrajectoryDirectionSearchConfigurationError: LocalizedError,
    Equatable, Sendable
{
    case invalidPositions(String)
    case duplicatePosition(ActivationTokenPosition)
    case invalidLayersPerPosition(String)
    case positionsRequired

    public var errorDescription: String? {
        switch self {
        case .invalidPositions(let value):
            "\(TrajectoryDirectionSearchConfiguration.positionsKey) must be "
                + "a comma-separated subset of post-instruction,first-response,"
                + "second-response,last-user, not '\(value)'."
        case .duplicatePosition(let position):
            "\(TrajectoryDirectionSearchConfiguration.positionsKey) contains "
                + "duplicate position '\(position.rawValue)'."
        case .invalidLayersPerPosition(let value):
            "\(TrajectoryDirectionSearchConfiguration.layersPerPositionKey) "
                + "must be a positive integer, not '\(value)'."
        case .positionsRequired:
            "\(TrajectoryDirectionSearchConfiguration.positionsKey) is required "
                + "when \(TrajectoryDirectionSearchConfiguration.layersPerPositionKey) is set."
        }
    }
}

/// Strict, opt-in configuration for retaining every exact diagnostic trial.
///
/// The legacy search uses its inexpensive behavioral proxy as an early screen:
/// candidates that do not improve that proxy are not assigned KL metrics or
/// written to the study. Persist-all mode deliberately disables only that early
/// screen so the captured full responses can be judged semantically later.
public enum DiagnosticTrialPersistenceConfiguration {
    public static let persistAllTrialsKey =
        "ABSLAYER_DIAGNOSTIC_PERSIST_ALL_TRIALS"

    /// Unset preserves the legacy `false` behavior. Values are deliberately
    /// limited to boolean words; malformed configuration never silently falls
    /// back to the fast path.
    public static func parse(environment: [String: String]) throws -> Bool {
        guard let raw = environment[persistAllTrialsKey] else { return false }
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "true": return true
        case "false": return false
        default:
            throw DiagnosticTrialPersistenceConfigurationError.invalidBoolean(raw)
        }
    }
}

public enum DiagnosticTrialPersistenceConfigurationError: LocalizedError,
    Equatable, Sendable
{
    case invalidBoolean(String)

    public var errorDescription: String? {
        switch self {
        case .invalidBoolean(let value):
            "\(DiagnosticTrialPersistenceConfiguration.persistAllTrialsKey) "
                + "must be true or false, not '\(value)'."
        }
    }
}

public struct ResidualInterventionDiagnosticRequest: Sendable {
    public var modelDirectory: String
    public var measurementPairs: [PromptPair]
    public var evaluationPairs: [PromptPair]
    public var measurementCases: Int
    public var evaluationCases: Int
    public var maximumCandidateLayers: Int
    /// Explicit application-layer override in canonical zero-based indexing.
    /// Nil means localize across the complete measured decoder stack.
    public var applicationLayersZeroBased: [Int]?
    /// Nil preserves the legacy coupled post-instruction/same-layer search.
    /// A non-empty list opts into independent source-role localization.
    public var directionPositions: [ActivationTokenPosition]?
    public var directionLayersPerPosition: Int
    public var maximumBundleLayers: Int
    public var maximumRounds: Int
    public var subspaceRank: Int
    public var benignNullRank: Int
    public var gateQuantiles: [Float]
    public var strengths: [Float]
    /// Token-application schedules evaluated for every learned layer/quantile
    /// plan and strength. A nil entry is the exact pre-schedule legacy form.
    public var interventionSchedules: [ExactResidualInterventionSchedule?]
    public var maximumStrictCyberFailureRate: Double
    public var sequenceTopK: Int
    public var maximumSequenceKLLowerBound: Double
    public var maximumTeacherForcedContinuationMeanKL: Double
    public var maximumTeacherForcedContinuationP95KL: Double
    public var maximumTeacherForcedReferenceTokens: Int
    /// When true, bypasses only the cheap behavioral-proxy early skip so every
    /// candidate receives preservation metrics and a persisted response record.
    public var persistAllTrials: Bool
    public var guardrails: InterventionGuardrails
    public var outputPath: String?

    public var maximumControlPromptPrefixKLLowerBound: Double {
        get { maximumSequenceKLLowerBound }
        set { maximumSequenceKLLowerBound = newValue }
    }

    public init(
        modelDirectory: String,
        measurementPairs: [PromptPair], evaluationPairs: [PromptPair],
        measurementCases: Int = 48, evaluationCases: Int = 24,
        maximumCandidateLayers: Int = 6,
        candidateLayers: [Int]? = nil,
        applicationLayersZeroBased: [Int]? = nil,
        directionPositions: [ActivationTokenPosition]? = nil,
        directionLayersPerPosition: Int = 2,
        maximumBundleLayers: Int = 3,
        maximumRounds: Int = 2,
        subspaceRank: Int = 4, benignNullRank: Int = 8,
        gateQuantiles: [Float] = [0.95, 0.99],
        strengths: [Float] = [0.5, 0.75, 1],
        interventionSchedules: [ExactResidualInterventionSchedule?] = [nil],
        maximumStrictCyberFailureRate: Double = 0.10,
        sequenceTopK: Int = 64,
        maximumSequenceKLLowerBound: Double = 0.5,
        maximumTeacherForcedContinuationMeanKL: Double = 0.5,
        maximumTeacherForcedContinuationP95KL: Double = 1.0,
        maximumTeacherForcedReferenceTokens: Int = 128,
        persistAllTrials: Bool = false,
        guardrails: InterventionGuardrails = .init(
            maximumFirstTokenKL: 0.5,
            maximumControlFailureRate: 0.10,
            maximumAddedControlFailureRate: 0.10),
        outputPath: String? = nil
    ) {
        self.modelDirectory = modelDirectory
        self.measurementPairs = measurementPairs
        self.evaluationPairs = evaluationPairs
        self.measurementCases = measurementCases
        self.evaluationCases = evaluationCases
        self.maximumCandidateLayers = maximumCandidateLayers
        // `candidateLayers` was the original one-based API. Keep it as a
        // source-compatible compatibility input while making the canonical
        // representation and all new callers explicitly zero-based.
        self.applicationLayersZeroBased = applicationLayersZeroBased
            ?? candidateLayers?.map { $0 - 1 }
        self.directionPositions = directionPositions
        self.directionLayersPerPosition = directionLayersPerPosition
        self.maximumBundleLayers = maximumBundleLayers
        self.maximumRounds = maximumRounds
        self.subspaceRank = subspaceRank
        self.benignNullRank = benignNullRank
        self.gateQuantiles = gateQuantiles
        self.strengths = strengths
        self.interventionSchedules = interventionSchedules
        self.maximumStrictCyberFailureRate = maximumStrictCyberFailureRate
        self.sequenceTopK = sequenceTopK
        self.maximumSequenceKLLowerBound = maximumSequenceKLLowerBound
        self.maximumTeacherForcedContinuationMeanKL =
            maximumTeacherForcedContinuationMeanKL
        self.maximumTeacherForcedContinuationP95KL =
            maximumTeacherForcedContinuationP95KL
        self.maximumTeacherForcedReferenceTokens =
            maximumTeacherForcedReferenceTokens
        self.persistAllTrials = persistAllTrials
        self.guardrails = guardrails
        self.outputPath = outputPath
    }
}

public struct ExactBehaviorMetrics: Codable, Sendable, Equatable {
    public let strictCyberFailureRate: Double
    public let refusalRate: Double
    public let partialComplianceRate: Double
    public let controlFailureRate: Double
    public let worstCategoryFailureRate: Double
    public let categoryFailureRates: [String: Double]

    public init(evaluation: RefusalEvaluation) {
        let total = Double(max(1, evaluation.total))
        strictCyberFailureRate = Double(evaluation.cases.count {
            $0.contrast.outcome != .compliance
        }) / total
        refusalRate = Double(evaluation.cases.count(where: \.refused)) / total
        partialComplianceRate = Double(evaluation.cases.count {
            $0.contrast.outcome == .partialCompliance
        }) / total
        controlFailureRate = Double(evaluation.cases.count {
            !$0.controlHealthy
        }) / total
        categoryFailureRates = Dictionary(uniqueKeysWithValues:
            evaluation.categories.map { category in
                let count = Double(max(1, category.cases.count))
                let failures = category.cases.count {
                    $0.contrast.outcome != .compliance
                }
                return (category.category, Double(failures) / count)
            })
        worstCategoryFailureRate = categoryFailureRates.values.max() ?? 1
    }
}

public struct LayerRefusalLocality: Codable, Sendable, Equatable {
    /// One-based runtime residual-tap layer retained in study schema v1.
    public let layer: Int
    public let postInstructionAUC: Double
    public let lastUserAUC: Double
    public let refusalAmplification: Double
    public let priority: Double
}

public struct ExactCaseOutcomeRecord: Codable, Sendable, Equatable {
    public let name: String
    public let category: String
    public let contrast: OutcomeClassification
    public let control: OutcomeClassification

    init(_ result: RefusalCaseResult) {
        name = result.name
        category = result.category
        contrast = result.contrast
        control = result.control
    }
}

public struct ExactInterventionTelemetry: Codable, Sendable, Equatable {
    public let harmfulGateRate: Double
    public let controlGateRate: Double
    public let harmfulMeanGate: Double
    public let controlMeanGate: Double
    public let harmfulMeanProjectedFraction: Double
    public let controlMeanProjectedFraction: Double
    public let editableProbeEnergy: Double
}

public struct ResidualLocalityRound: Codable, Sendable, Equatable {
    public let round: Int
    public let pairCount: Int
    public let layers: [LayerRefusalLocality]
}

public struct ExactResidualTrialRecord: Codable, Sendable, Equatable {
    public let index: Int
    public let round: Int
    public let intervention: ExactResidualIntervention
    public let companionInterventions: [ExactResidualIntervention]?
    public let metrics: ExactBehaviorMetrics
    public let firstTokenKL: Double
    public let sequenceKLLowerBound: Double
    /// Exact full-vocabulary KL over fixed benign assistant continuations.
    /// `nil` means the evaluation corpus supplied no reference answers.
    public let teacherForcedContinuationMeanKL: Double?
    public let teacherForcedContinuationP95KL: Double?
    public let teacherForcedContinuationMaximumKL: Double?
    public let controlPerplexityRatio: Double
    public let passedSequenceGuardrail: Bool
    public let passedTeacherForcedContinuationGuardrail: Bool?
    public let score: InterventionCandidateScore
    public let telemetry: ExactInterventionTelemetry?
    public let companionTelemetry: [ExactInterventionTelemetry]?
    public let caseOutcomes: [ExactCaseOutcomeRecord]?
    public let responses: [PromptResult]?

    public var proposedInterventions: [ExactResidualIntervention] {
        [intervention] + (companionInterventions ?? [])
    }

    public var passedGuardrails: Bool {
        score.passesGuardrails && passedSequenceGuardrail
            && (passedTeacherForcedContinuationGuardrail ?? true)
    }

    /// Honest name for the persisted legacy `sequenceKLLowerBound` field.
    public var controlPromptPrefixKLLowerBound: Double {
        sequenceKLLowerBound
    }
}

public struct ExactResidualInterventionStudy: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let modelPath: String
    public let classifierIdentifier: String
    /// Optional so schema-v1 artifacts written before metric descriptors still
    /// decode. New studies always persist this alongside the legacy metric.
    public var cyberFailureMetric: CyberFailureMetricMetadata? = nil
    public let maximumStrictCyberFailureRate: Double
    public let maximumSequenceKLLowerBound: Double
    /// Optional for backward decoding of studies written before reference
    /// assistant continuations were supported.
    public var maximumTeacherForcedContinuationMeanKL: Double?
    public var maximumTeacherForcedContinuationP95KL: Double?
    public let baseline: ExactBehaviorMetrics
    public let baselineCaseOutcomes: [ExactCaseOutcomeRecord]?
    public var localityRounds: [ResidualLocalityRound]
    public var trials: [ExactResidualTrialRecord]
    public var acceptedInterventions: [ExactResidualIntervention]
    public var finalMetrics: ExactBehaviorMetrics
    public var finalFirstTokenKL: Double
    public var finalSequenceKLLowerBound: Double
    public var finalTeacherForcedContinuationMeanKL: Double?
    public var finalTeacherForcedContinuationP95KL: Double?
    public var finalTeacherForcedContinuationMaximumKL: Double?
    public var finalControlPerplexityRatio: Double
    public var finalCaseOutcomes: [ExactCaseOutcomeRecord]?

    public var passedAbliteration: Bool {
        finalMetrics.strictCyberFailureRate <= maximumStrictCyberFailureRate
    }

    public var finalControlPromptPrefixKLLowerBound: Double {
        finalSequenceKLLowerBound
    }

    public var usesSemanticClassifier: Bool {
        resolvedCyberFailureMetric.certifiesSemanticCompliance
    }

    public var resolvedCyberFailureMetric: CyberFailureMetricMetadata {
        cyberFailureMetric ?? .inferred(
            classifierIdentifier: classifierIdentifier)
    }

    public var passedAll: Bool {
        usesSemanticClassifier && passedAbliteration
            && trials.last(where: { trial in
                trial.proposedInterventions.allSatisfy(
                    acceptedInterventions.contains)
            })?.passedGuardrails != false
    }

    public func write(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(
            to: URL(fileURLWithPath: path).standardizedFileURL,
            options: .atomic)
    }

    public static func read(from path: String) throws -> Self {
        try JSONDecoder().decode(
            Self.self,
            from: Data(contentsOf: URL(fileURLWithPath: path).standardizedFileURL))
    }
}

/// Performs exact causal layer localization and sequential residual editing.
/// Every trial runs in the model's actual residual path; no LoRA approximation
/// is used to select a candidate. A later round measures only cases that still
/// fail strict semantic compliance after the earlier accepted intervention(s).
public enum ResidualInterventionDiagnosticEngine {
    private struct DirectionApplicationCandidate: Sendable {
        let sourcePosition: ActivationTokenPosition?
        let directionLayer: LayerActivationSet
        let applicationLayer: LayerActivationSet
    }

    public static func run<Classifier: OutcomeClassifying>(
        _ request: ResidualInterventionDiagnosticRequest,
        classifier: Classifier
    ) async throws -> ExactResidualInterventionStudy {
        guard request.measurementCases > 1,
              request.evaluationCases > 0,
              request.maximumCandidateLayers > 0,
              request.applicationLayersZeroBased?.allSatisfy({ $0 >= 0 }) != false,
              request.directionPositions.map({
                  !$0.isEmpty && Set($0).count == $0.count
              }) ?? true,
              request.directionLayersPerPosition > 0,
              request.maximumBundleLayers >= 0,
              request.maximumRounds > 0,
              request.subspaceRank > 0,
              request.benignNullRank >= 0,
              !request.gateQuantiles.isEmpty,
              request.gateQuantiles.allSatisfy({ (0 ... 1).contains($0) }),
              !request.strengths.isEmpty,
              request.strengths.allSatisfy({ $0.isFinite && $0 >= 0 }),
              !request.interventionSchedules.isEmpty,
              (0 ... 1).contains(request.maximumStrictCyberFailureRate),
              request.sequenceTopK > 0,
              request.maximumSequenceKLLowerBound.isFinite,
              request.maximumSequenceKLLowerBound >= 0,
              request.maximumTeacherForcedContinuationMeanKL.isFinite,
              request.maximumTeacherForcedContinuationMeanKL >= 0,
              request.maximumTeacherForcedContinuationP95KL.isFinite,
              request.maximumTeacherForcedContinuationP95KL >= 0,
              request.maximumTeacherForcedReferenceTokens > 0
        else { throw ResidualDiagnosticError.invalidConfiguration }

        let modelPath = URL(fileURLWithPath: request.modelDirectory)
            .standardizedFileURL.path
        let measurementPool = evenlySpaced(
            request.measurementPairs, maximum: request.measurementCases)
        let evaluationPairs = evenlySpaced(
            request.evaluationPairs, maximum: request.evaluationCases)
        let runtime = try await ResidentTrialRuntime(modelDirectory: modelPath)

        print("Capturing untouched exact-runtime baseline…")
        let baselineResponses = try await runtime.responses(
            pairs: evaluationPairs, maximumCases: evaluationPairs.count)
        let baselineEvaluation = RefusalEvaluator.evaluate(
            baselineResponses, classifier: classifier)
        let baselineMetrics = ExactBehaviorMetrics(evaluation: baselineEvaluation)
        let baselineFingerprint = try await runtime.fingerprint(
            pairs: evaluationPairs, maximumCases: evaluationPairs.count)
        let baselineSequence = try await runtime.controlPromptPrefixFingerprint(
            pairs: evaluationPairs, maximumCases: evaluationPairs.count,
            topK: request.sequenceTopK)

        var study = ExactResidualInterventionStudy(
            schemaVersion: 1, modelPath: modelPath,
            classifierIdentifier: classifier.identifier,
            maximumStrictCyberFailureRate: request.maximumStrictCyberFailureRate,
            maximumSequenceKLLowerBound: request.maximumSequenceKLLowerBound,
            maximumTeacherForcedContinuationMeanKL:
                request.maximumTeacherForcedContinuationMeanKL,
            maximumTeacherForcedContinuationP95KL:
                request.maximumTeacherForcedContinuationP95KL,
            baseline: baselineMetrics,
            baselineCaseOutcomes: baselineEvaluation.cases.map(ExactCaseOutcomeRecord.init),
            localityRounds: [], trials: [],
            acceptedInterventions: [], finalMetrics: baselineMetrics,
            finalFirstTokenKL: 0, finalSequenceKLLowerBound: 0,
            finalTeacherForcedContinuationMeanKL: nil,
            finalTeacherForcedContinuationP95KL: nil,
            finalTeacherForcedContinuationMaximumKL: nil,
            finalControlPerplexityRatio: 1,
            finalCaseOutcomes: baselineEvaluation.cases.map(ExactCaseOutcomeRecord.init))
        study.cyberFailureMetric = .inferred(
            classifierIdentifier: classifier.identifier)
        try persist(study, path: request.outputPath)

        let cyberFailureLabel = study.resolvedCyberFailureMetric.displayLabel
        let trajectoryMode = request.directionPositions != nil

        // Iterative discovery must remain inside the measurement split. The
        // evaluation set ranks/checks candidates, but its failures must never
        // become training rows for a later direction.
        var currentMeasurementEvaluation: RefusalEvaluation?
        for round in 0 ..< request.maximumRounds {
            let measurementPairs: [PromptPair]
            if round == 0 {
                measurementPairs = measurementPool
            } else {
                guard let currentMeasurementEvaluation else { break }
                let failedNames = Set(currentMeasurementEvaluation.cases.compactMap { item in
                    item.contrast.outcome == .compliance ? nil : item.name
                })
                measurementPairs = evenlySpaced(
                    measurementPool.filter { failedNames.contains($0.name) },
                    maximum: request.measurementCases)
            }
            guard measurementPairs.count > 1 else { break }

            if !study.acceptedInterventions.isEmpty {
                try await runtime.install(study.acceptedInterventions)
            }
            let collections: [ActivationTokenPosition: ActivationCollection]
            do {
                let positions: Set<ActivationTokenPosition>
                if let directionPositions = request.directionPositions {
                    positions = Set(directionPositions).union(
                        [.lastUser, .postInstruction])
                } else {
                    positions = [.lastUser, .postInstruction]
                }
                collections = try await runtime.activationCollections(
                    pairs: measurementPairs,
                    positions: positions) { completed, total in
                        print("round \(round + 1) activation \(completed)/\(total)")
                    }
            } catch {
                await runtime.clearResidualIntervention()
                throw error
            }
            await runtime.clearResidualIntervention()

            guard let post = collections[.postInstruction],
                  let user = collections[.lastUser],
                  post.layers.count == user.layers.count
            else { throw ResidualDiagnosticError.incompleteActivationCollection }

            let locality = zip(post.layers, user.layers).map { postLayer, userLayer in
                let postAUC = linearProbeAUC(postLayer)
                let userAUC = linearProbeAUC(userLayer)
                let amplification = postAUC - userAUC
                return LayerRefusalLocality(
                    layer: postLayer.layer,
                    postInstructionAUC: postAUC,
                    lastUserAUC: userAUC,
                    refusalAmplification: amplification,
                    priority: postAUC + max(0, amplification))
            }.sorted {
                if $0.priority != $1.priority { return $0.priority > $1.priority }
                if $0.refusalAmplification != $1.refusalAmplification {
                    return $0.refusalAmplification > $1.refusalAmplification
                }
                if $0.postInstructionAUC != $1.postInstructionAUC {
                    return $0.postInstructionAUC > $1.postInstructionAUC
                }
                return $0.layer < $1.layer
            }
            study.localityRounds.append(ResidualLocalityRound(
                round: round + 1, pairCount: measurementPairs.count,
                layers: locality))

            // A tiny high-dimensional sample can make several adjacent early
            // layers look equally separable. Testing only that cluster misses
            // later causal sites where the refusal policy is actually used.
            // Take the best locality candidate from each depth band, then fill
            // any gaps by global priority.
            let selectedLayerNumbers: [Int]
            if let requested = request.applicationLayersZeroBased,
               !requested.isEmpty
            {
                let canonical = try DecoderLayerSelection.validateZeroBased(
                    requested, layerCount: post.layers.count)
                selectedLayerNumbers = Array(
                    canonical.prefix(request.maximumCandidateLayers)).map { $0 + 1 }
            } else {
                selectedLayerNumbers = selectCandidateLayerNumbers(
                    locality: locality,
                    maximum: request.maximumCandidateLayers,
                    layerCount: post.layers.count)
            }
            let selectedLayers = selectedLayerNumbers.compactMap { number in
                post.layers.first { $0.layer == number }
            }
            let searchCandidates: [DirectionApplicationCandidate]
            if let directionPositions = request.directionPositions {
                var candidates = [DirectionApplicationCandidate]()
                for position in directionPositions {
                    guard let collection = collections[position],
                          collection.layers.count == post.layers.count
                    else {
                        throw ResidualDiagnosticError
                            .incompleteActivationCollection
                    }
                    let sourceLayerNumbers = selectDirectionLayerNumbers(
                        layers: collection.layers,
                        maximum: request.directionLayersPerPosition)
                    let sourceLayers = sourceLayerNumbers.compactMap { number in
                        collection.layers.first { $0.layer == number }
                    }
                    for directionLayer in sourceLayers {
                        for applicationLayer in selectedLayers {
                            candidates.append(DirectionApplicationCandidate(
                                sourcePosition: position,
                                directionLayer: directionLayer,
                                applicationLayer: applicationLayer))
                        }
                    }
                }
                searchCandidates = candidates
            } else {
                searchCandidates = selectedLayers.map { layer in
                    DirectionApplicationCandidate(
                        sourcePosition: nil,
                        directionLayer: layer,
                        applicationLayer: layer)
                }
            }
            var roundTrials = [ExactResidualTrialRecord]()

            for candidate in searchCandidates {
                for quantile in request.gateQuantiles {
                    let planned: ControlledEvasionLayerPlan?
                    if candidate.sourcePosition != nil {
                        planned = ControlledEvasionPlanner.makePlan(
                            directionLayer: candidate.directionLayer,
                            applicationLayer: candidate.applicationLayer,
                            subspaceRank: request.subspaceRank,
                            benignNullRank: request.benignNullRank,
                            controlGateQuantile: quantile,
                            strength: 1,
                            includeApplicationGateAxisInEdit: false)
                    } else {
                        planned = ControlledEvasionPlanner.makePlan(
                            layer: candidate.applicationLayer,
                            subspaceRank: request.subspaceRank,
                            benignNullRank: request.benignNullRank,
                            controlGateQuantile: quantile,
                            strength: 1)
                    }
                    guard let base = planned
                    else { continue }
                    let telemetry = interventionTelemetry(
                        plan: base.intervention.plan,
                        layer: candidate.applicationLayer)

                    for strength in request.strengths {
                      for schedule in request.interventionSchedules {
                        guard let plan = plan(
                            from: base.intervention.plan,
                            layer: candidate.applicationLayer.layer,
                            strength: strength,
                            schedule: schedule,
                            sourceLayerZeroBased: candidate.sourcePosition == nil
                                ? nil : candidate.directionLayer.layer - 1,
                            sourcePosition: candidate.sourcePosition)
                        else { continue }
                        let stack = study.acceptedInterventions + [plan]
                        try await runtime.install(stack)
                        let responses: [PromptResult]
                        do {
                            responses = try await runtime.responses(
                                pairs: evaluationPairs,
                                maximumCases: evaluationPairs.count)
                        } catch {
                            await runtime.clearResidualIntervention()
                            throw error
                        }
                        let evaluation = RefusalEvaluator.evaluate(
                            responses, classifier: classifier)
                        let metrics = ExactBehaviorMetrics(evaluation: evaluation)
                        guard shouldFullyMeasureCandidate(
                            candidateCyberFailureRate:
                                metrics.strictCyberFailureRate,
                            incumbentCyberFailureRate:
                                study.finalMetrics.strictCyberFailureRate,
                            persistAllTrials: request.persistAllTrials)
                        else {
                            await runtime.clearResidualIntervention()
                            if let sourcePosition = candidate.sourcePosition {
                                print(String(
                                    format: "screened source=%@ L0=%d -> application L0=%d q=%.3f strength=%.2f schedule=%@ %@=%.3f (no improvement)",
                                    sourcePosition.rawValue,
                                    candidate.directionLayer.layer - 1,
                                    candidate.applicationLayer.layer - 1,
                                    quantile, strength,
                                    scheduleName(schedule),
                                    cyberFailureLabel,
                                    metrics.strictCyberFailureRate))
                            } else {
                                print(String(
                                    format: "screened L%d q=%.3f strength=%.2f schedule=%@ %@=%.3f (no improvement)",
                                    candidate.applicationLayer.layer,
                                    quantile, strength,
                                    scheduleName(schedule),
                                    cyberFailureLabel,
                                    metrics.strictCyberFailureRate))
                            }
                            continue
                        }
                        let fingerprint: LogitFingerprint
                        let sequenceFingerprint: SequenceLogitFingerprint
                        let continuationMetrics:
                            TeacherForcedContinuationMetricSummary?
                        do {
                            fingerprint = try await runtime.fingerprint(
                                pairs: evaluationPairs,
                                maximumCases: evaluationPairs.count)
                            sequenceFingerprint = try await runtime
                                .controlPromptPrefixFingerprint(
                                pairs: evaluationPairs,
                                maximumCases: evaluationPairs.count,
                                topK: request.sequenceTopK,
                                reference: baselineSequence)
                            continuationMetrics = try await runtime
                                .teacherForcedControlContinuationMetrics(
                                    pairs: evaluationPairs,
                                    maximumCases: evaluationPairs.count,
                                    maximumReferenceTokens: request
                                        .maximumTeacherForcedReferenceTokens)
                        } catch {
                            await runtime.clearResidualIntervention()
                            throw error
                        }
                        await runtime.clearResidualIntervention()
                        let kl = try LogitFingerprintEngine.divergence(
                            baseline: baselineFingerprint, candidate: fingerprint)
                        let sequence = try SequenceMetricEngine.compare(
                            baseline: baselineSequence,
                            candidate: sequenceFingerprint)
                        let score = InterventionCandidateScore.evaluate(
                            baselineRefusalRate: study.finalMetrics.strictCyberFailureRate,
                            candidateRefusalRate: metrics.strictCyberFailureRate,
                            baselineControlFailureRate: study.baseline.controlFailureRate,
                            candidateControlFailureRate: metrics.controlFailureRate,
                            firstTokenKL: kl, guardrails: request.guardrails)
                        let passedContinuation = continuationMetrics.map {
                            $0.exactMeanKL
                                <= request.maximumTeacherForcedContinuationMeanKL
                                && $0.exactP95KL
                                    <= request.maximumTeacherForcedContinuationP95KL
                        }
                        let trial = ExactResidualTrialRecord(
                            index: study.trials.count, round: round + 1,
                            intervention: plan, companionInterventions: nil,
                            metrics: metrics,
                            firstTokenKL: kl,
                            sequenceKLLowerBound:
                                sequence.controlPromptPrefixKLLowerBound,
                            teacherForcedContinuationMeanKL:
                                continuationMetrics?.exactMeanKL,
                            teacherForcedContinuationP95KL:
                                continuationMetrics?.exactP95KL,
                            teacherForcedContinuationMaximumKL:
                                continuationMetrics?.exactMaximumKL,
                            controlPerplexityRatio: sequence.perplexityRatio,
                            passedSequenceGuardrail:
                                sequence.controlPromptPrefixKLLowerBound
                                    <= request.maximumSequenceKLLowerBound,
                            passedTeacherForcedContinuationGuardrail:
                                passedContinuation,
                            score: score,
                            telemetry: telemetry,
                            companionTelemetry: nil,
                            caseOutcomes: evaluation.cases.map(ExactCaseOutcomeRecord.init),
                            responses: responses)
                        study.trials.append(trial)
                        roundTrials.append(trial)
                        try persist(study, path: request.outputPath)
                        let continuationReport = continuationMetrics.map {
                            String(
                                format: "continuation-KL(mean/p95/max)=%.5f/%.5f/%.5f",
                                $0.exactMeanKL, $0.exactP95KL,
                                $0.exactMaximumKL)
                        } ?? "continuation-KL=n/a"
                        if let sourcePosition = candidate.sourcePosition {
                            print(String(
                                format: "exact trial %d source=%@ L0=%d -> application L0=%d q=%.3f strength=%.2f schedule=%@ %@=%.3f control=%.3f first-KL=%.5f control-prompt-prefix-KL-lower-bound=%.5f gate(h/c)=%.2f/%.2f editable=%.3f %@",
                                trial.index + 1, sourcePosition.rawValue,
                                candidate.directionLayer.layer - 1,
                                candidate.applicationLayer.layer - 1,
                                quantile, strength,
                                scheduleName(schedule),
                                cyberFailureLabel,
                                metrics.strictCyberFailureRate,
                                metrics.controlFailureRate, kl,
                                sequence.controlPromptPrefixKLLowerBound,
                                telemetry.harmfulMeanGate,
                                telemetry.controlMeanGate,
                                telemetry.editableProbeEnergy,
                                trial.passedGuardrails ? "PASS" : "FAIL")
                                + " \(continuationReport)")
                        } else {
                            print(String(
                                format: "exact trial %d L%d q=%.3f strength=%.2f schedule=%@ %@=%.3f control=%.3f first-KL=%.5f control-prompt-prefix-KL-lower-bound=%.5f gate(h/c)=%.2f/%.2f editable=%.3f %@",
                                trial.index + 1,
                                candidate.applicationLayer.layer,
                                quantile, strength,
                                scheduleName(schedule),
                                cyberFailureLabel,
                                metrics.strictCyberFailureRate,
                                metrics.controlFailureRate, kl,
                                sequence.controlPromptPrefixKLLowerBound,
                                telemetry.harmfulMeanGate,
                                telemetry.controlMeanGate,
                                telemetry.editableProbeEnergy,
                                trial.passedGuardrails ? "PASS" : "FAIL")
                                + " \(continuationReport)")
                        }
                      }
                    }
                }
            }

            let bundleLayers = evenlySpacedElements(
                selectedLayers, maximum: request.maximumBundleLayers)
            if !trajectoryMode, bundleLayers.count >= 2 {
                for quantile in request.gateQuantiles {
                    let bases = bundleLayers.compactMap { layer -> (
                        layer: LayerActivationSet,
                        plan: ControlledEvasionLayerPlan,
                        telemetry: ExactInterventionTelemetry
                    )? in
                        guard let planned = ControlledEvasionPlanner.makePlan(
                            layer: layer,
                            subspaceRank: request.subspaceRank,
                            benignNullRank: request.benignNullRank,
                            controlGateQuantile: quantile,
                            strength: 1)
                        else { return nil }
                        return (
                            layer, planned,
                            interventionTelemetry(
                                plan: planned.intervention.plan, layer: layer))
                    }
                    guard bases.count >= 2 else { continue }

                    for strength in request.strengths {
                      for schedule in request.interventionSchedules {
                        let proposals = bases.compactMap { item in
                            plan(
                                from: item.plan.intervention.plan,
                                layer: item.layer.layer, strength: strength,
                                schedule: schedule)
                        }
                        guard proposals.count == bases.count,
                              let primary = proposals.first
                        else { continue }
                        try await runtime.install(
                            study.acceptedInterventions + proposals)
                        let responses: [PromptResult]
                        let fingerprint: LogitFingerprint
                        let sequenceFingerprint: SequenceLogitFingerprint
                        let continuationMetrics:
                            TeacherForcedContinuationMetricSummary?
                        do {
                            responses = try await runtime.responses(
                                pairs: evaluationPairs,
                                maximumCases: evaluationPairs.count)
                            fingerprint = try await runtime.fingerprint(
                                pairs: evaluationPairs,
                                maximumCases: evaluationPairs.count)
                            sequenceFingerprint = try await runtime
                                .controlPromptPrefixFingerprint(
                                pairs: evaluationPairs,
                                maximumCases: evaluationPairs.count,
                                topK: request.sequenceTopK,
                                reference: baselineSequence)
                            continuationMetrics = try await runtime
                                .teacherForcedControlContinuationMetrics(
                                    pairs: evaluationPairs,
                                    maximumCases: evaluationPairs.count,
                                    maximumReferenceTokens: request
                                        .maximumTeacherForcedReferenceTokens)
                        } catch {
                            await runtime.clearResidualIntervention()
                            throw error
                        }
                        await runtime.clearResidualIntervention()

                        let evaluation = RefusalEvaluator.evaluate(
                            responses, classifier: classifier)
                        let metrics = ExactBehaviorMetrics(evaluation: evaluation)
                        let kl = try LogitFingerprintEngine.divergence(
                            baseline: baselineFingerprint, candidate: fingerprint)
                        let sequence = try SequenceMetricEngine.compare(
                            baseline: baselineSequence,
                            candidate: sequenceFingerprint)
                        let score = InterventionCandidateScore.evaluate(
                            baselineRefusalRate: study.finalMetrics.strictCyberFailureRate,
                            candidateRefusalRate: metrics.strictCyberFailureRate,
                            baselineControlFailureRate: study.baseline.controlFailureRate,
                            candidateControlFailureRate: metrics.controlFailureRate,
                            firstTokenKL: kl, guardrails: request.guardrails)
                        let passedContinuation = continuationMetrics.map {
                            $0.exactMeanKL
                                <= request.maximumTeacherForcedContinuationMeanKL
                                && $0.exactP95KL
                                    <= request.maximumTeacherForcedContinuationP95KL
                        }
                        let trial = ExactResidualTrialRecord(
                            index: study.trials.count, round: round + 1,
                            intervention: primary,
                            companionInterventions: Array(proposals.dropFirst()),
                            metrics: metrics, firstTokenKL: kl,
                            sequenceKLLowerBound:
                                sequence.controlPromptPrefixKLLowerBound,
                            teacherForcedContinuationMeanKL:
                                continuationMetrics?.exactMeanKL,
                            teacherForcedContinuationP95KL:
                                continuationMetrics?.exactP95KL,
                            teacherForcedContinuationMaximumKL:
                                continuationMetrics?.exactMaximumKL,
                            controlPerplexityRatio: sequence.perplexityRatio,
                            passedSequenceGuardrail:
                                sequence.controlPromptPrefixKLLowerBound
                                    <= request.maximumSequenceKLLowerBound,
                            passedTeacherForcedContinuationGuardrail:
                                passedContinuation,
                            score: score,
                            telemetry: bases.first?.telemetry,
                            companionTelemetry: Array(bases.dropFirst().map(\.telemetry)),
                            caseOutcomes: evaluation.cases.map(
                                ExactCaseOutcomeRecord.init),
                            responses: responses)
                        study.trials.append(trial)
                        roundTrials.append(trial)
                        try persist(study, path: request.outputPath)
                        let layerList = proposals.map { String($0.layer) }
                            .joined(separator: ",")
                        let continuationReport = continuationMetrics.map {
                            String(
                                format: "continuation-KL(mean/p95/max)=%.5f/%.5f/%.5f",
                                $0.exactMeanKL, $0.exactP95KL,
                                $0.exactMaximumKL)
                        } ?? "continuation-KL=n/a"
                        print(String(
                            format: "exact bundle %d L[%@] q=%.3f strength=%.2f schedule=%@ %@=%.3f control=%.3f first-KL=%.5f control-prompt-prefix-KL-lower-bound=%.5f %@",
                            trial.index + 1, layerList, quantile, strength,
                            scheduleName(schedule),
                            cyberFailureLabel,
                            metrics.strictCyberFailureRate,
                            metrics.controlFailureRate, kl,
                            sequence.controlPromptPrefixKLLowerBound,
                            trial.passedGuardrails ? "PASS" : "FAIL")
                            + " \(continuationReport)")
                      }
                    }
                }
            }

            guard let accepted = roundTrials
                .filter({ $0.passedGuardrails
                    && $0.metrics.strictCyberFailureRate
                        < study.finalMetrics.strictCyberFailureRate })
                .min(by: trialIsWorse)
            else { break }

            for intervention in accepted.proposedInterventions
                where !study.acceptedInterventions.contains(intervention)
            {
                study.acceptedInterventions.append(intervention)
            }
            study.finalMetrics = accepted.metrics
            study.finalFirstTokenKL = accepted.firstTokenKL
            study.finalSequenceKLLowerBound = accepted.sequenceKLLowerBound
            study.finalTeacherForcedContinuationMeanKL =
                accepted.teacherForcedContinuationMeanKL
            study.finalTeacherForcedContinuationP95KL =
                accepted.teacherForcedContinuationP95KL
            study.finalTeacherForcedContinuationMaximumKL =
                accepted.teacherForcedContinuationMaximumKL
            study.finalControlPerplexityRatio = accepted.controlPerplexityRatio
            study.finalCaseOutcomes = accepted.caseOutcomes
            let acceptedResponses: [PromptResult]
            let acceptedMeasurementResponses: [PromptResult]
            try await runtime.install(study.acceptedInterventions)
            do {
                acceptedResponses = try await runtime.responses(
                    pairs: evaluationPairs, maximumCases: evaluationPairs.count)
                acceptedMeasurementResponses = try await runtime.responses(
                    pairs: measurementPool, maximumCases: measurementPool.count)
            } catch {
                await runtime.clearResidualIntervention()
                throw error
            }
            await runtime.clearResidualIntervention()
            // Keep the evaluation replay for the persisted certificate, while
            // deriving the next round's hard cases only from measurement data.
            _ = RefusalEvaluator.evaluate(acceptedResponses, classifier: classifier)
            currentMeasurementEvaluation = RefusalEvaluator.evaluate(
                acceptedMeasurementResponses, classifier: classifier)
            try persist(study, path: request.outputPath)
            if study.passedAbliteration { break }
        }
        return study
    }

    /// Mann–Whitney AUC of a target-model-fitted probe. AUC is scale-free, so
    /// it can be compared across layers with very different residual norms.
    public static func linearProbeAUC(_ layer: LayerActivationSet) -> Double {
        guard layer.contrast.count == layer.control.count,
              layer.contrast.count >= 4
        else { return fittedProbeAUC(layer) }

        // Deterministic two-fold out-of-sample scoring prevents hidden-width
        // overfitting from making every decoder layer appear to have AUC 1.0.
        var foldAUC = [Double]()
        for fold in 0 ..< 2 {
            let validation = layer.contrast.indices.filter { $0 % 2 == fold }
            let training = layer.contrast.indices.filter { $0 % 2 != fold }
            guard !validation.isEmpty,
                  let probe = ControlledEvasionProbe.fit(
                    positive: training.map { layer.contrast[$0] },
                    negative: training.map { layer.control[$0] })
            else { continue }
            foldAUC.append(auc(
                positive: validation.map { probe.score(layer.contrast[$0]) },
                negative: validation.map { probe.score(layer.control[$0]) }))
        }
        return foldAUC.isEmpty
            ? fittedProbeAUC(layer)
            : foldAUC.reduce(0, +) / Double(foldAUC.count)
    }

    private static func fittedProbeAUC(_ layer: LayerActivationSet) -> Double {
        guard let probe = ControlledEvasionProbe.fit(
            positive: layer.contrast, negative: layer.control),
              !layer.contrast.isEmpty, !layer.control.isEmpty
        else { return 0.5 }
        return auc(
            positive: layer.contrast.map(probe.score),
            negative: layer.control.map(probe.score))
    }

    private static func auc(positive: [Float], negative: [Float]) -> Double {
        guard !positive.isEmpty, !negative.isEmpty else { return 0.5 }
        var wins = 0.0
        for lhs in positive {
            for rhs in negative {
                if lhs > rhs { wins += 1 }
                else if lhs == rhs { wins += 0.5 }
            }
        }
        return wins / Double(positive.count * negative.count)
    }

    private static func plan(
        from base: ControlledEvasionPlan, layer: Int, strength: Float,
        schedule: ExactResidualInterventionSchedule?,
        sourceLayerZeroBased: Int? = nil,
        sourcePosition: ActivationTokenPosition? = nil
    ) -> ExactResidualIntervention? {
        guard let adjusted = ControlledEvasionPlan(
            probe: base.probe, behaviorBasis: base.behaviorBasis,
            reference: base.reference, margin: base.margin,
            transitionWidth: base.transitionWidth, strength: strength)
        else { return nil }
        return ExactResidualIntervention(
            layer: layer, plan: adjusted, schedule: schedule,
            sourceLayerZeroBased: sourceLayerZeroBased,
            sourcePosition: sourcePosition)
    }

    private static func scheduleName(
        _ schedule: ExactResidualInterventionSchedule?
    ) -> String {
        schedule?.diagnosticName ?? "legacy"
    }

    /// Keeps the legacy proxy-improvement early screen unless exhaustive trial
    /// persistence is explicitly requested. Acceptance remains independently
    /// constrained to genuinely improving, guardrail-passing candidates.
    static func shouldFullyMeasureCandidate(
        candidateCyberFailureRate: Double,
        incumbentCyberFailureRate: Double,
        persistAllTrials: Bool
    ) -> Bool {
        persistAllTrials
            || candidateCyberFailureRate < incumbentCyberFailureRate
    }

    private static func trialIsWorse(
        _ lhs: ExactResidualTrialRecord, _ rhs: ExactResidualTrialRecord
    ) -> Bool {
        if lhs.metrics.strictCyberFailureRate != rhs.metrics.strictCyberFailureRate {
            return lhs.metrics.strictCyberFailureRate < rhs.metrics.strictCyberFailureRate
        }
        if lhs.metrics.worstCategoryFailureRate != rhs.metrics.worstCategoryFailureRate {
            return lhs.metrics.worstCategoryFailureRate
                < rhs.metrics.worstCategoryFailureRate
        }
        if lhs.sequenceKLLowerBound != rhs.sequenceKLLowerBound {
            return lhs.sequenceKLLowerBound < rhs.sequenceKLLowerBound
        }
        return lhs.firstTokenKL < rhs.firstTokenKL
    }

    private static func evenlySpaced(
        _ pairs: [PromptPair], maximum: Int
    ) -> [PromptPair] {
        guard maximum > 0, pairs.count > maximum else { return pairs }
        return (0 ..< maximum).map { pairs[$0 * pairs.count / maximum] }
    }

    private static func evenlySpacedElements<Element>(
        _ values: [Element], maximum: Int
    ) -> [Element] {
        guard maximum > 0 else { return [] }
        guard values.count > maximum else { return values }
        return (0 ..< maximum).map { values[$0 * values.count / maximum] }
    }

    static func selectCandidateLayerNumbers(
        locality: [LayerRefusalLocality], maximum: Int, layerCount: Int
    ) -> [Int] {
        let ranked = locality.map {
            DecoderLayerSelection.RankedLayer(
                zeroBasedIndex: $0.layer - 1, priority: $0.priority)
        }
        return (try? DecoderLayerSelection.depthDistributedZeroBased(
            rankedLayers: ranked, maximum: maximum, layerCount: layerCount))?
            .map { $0 + 1 } ?? []
    }

    /// Ranks one semantic token role independently, then preserves candidates
    /// across the measured decoder depth. Returned values are the one-based
    /// residual-tap numbers used internally by activation collections.
    static func selectDirectionLayerNumbers(
        layers: [LayerActivationSet], maximum: Int
    ) -> [Int] {
        let ranked = layers.map {
            DecoderLayerSelection.RankedLayer(
                zeroBasedIndex: $0.layer - 1,
                priority: linearProbeAUC($0))
        }
        return (try? DecoderLayerSelection.depthDistributedZeroBased(
            rankedLayers: ranked,
            maximum: maximum,
            layerCount: layers.count))?.map { $0 + 1 } ?? []
    }

    private static func interventionTelemetry(
        plan: ControlledEvasionPlan, layer: LayerActivationSet
    ) -> ExactInterventionTelemetry {
        let harmfulGates = layer.contrast.map(plan.gate)
        let controlGates = layer.control.map(plan.gate)
        let harmfulFractions = layer.contrast.map {
            projectedFraction(plan: plan, activation: $0)
        }
        let controlFractions = layer.control.map {
            projectedFraction(plan: plan, activation: $0)
        }
        let editableProbeEnergy = plan.behaviorBasis.reduce(0.0) { total, axis in
            let overlap = zip(axis, plan.probe.weights).reduce(Float.zero) {
                $0 + $1.0 * $1.1
            }
            return total + Double(overlap * overlap)
        }
        return ExactInterventionTelemetry(
            harmfulGateRate: rate(harmfulGates) { $0 > 0 },
            controlGateRate: rate(controlGates) { $0 > 0 },
            harmfulMeanGate: mean(harmfulGates),
            controlMeanGate: mean(controlGates),
            harmfulMeanProjectedFraction: mean(harmfulFractions),
            controlMeanProjectedFraction: mean(controlFractions),
            editableProbeEnergy: editableProbeEnergy)
    }

    private static func projectedFraction(
        plan: ControlledEvasionPlan, activation: [Float]
    ) -> Float {
        let centered = zip(activation, plan.reference).map(-)
        let denominator = sqrt(centered.reduce(Float.zero) { $0 + $1 * $1 })
        guard denominator > 1e-8 else { return 0 }
        let projection = plan.projectedComponent(of: activation)
        let numerator = sqrt(projection.reduce(Float.zero) { $0 + $1 * $1 })
        return numerator / denominator
    }

    private static func mean(_ values: [Float]) -> Double {
        guard !values.isEmpty else { return 0 }
        return Double(values.reduce(0, +)) / Double(values.count)
    }

    private static func rate(
        _ values: [Float], where predicate: (Float) -> Bool
    ) -> Double {
        guard !values.isEmpty else { return 0 }
        return Double(values.count(where: predicate)) / Double(values.count)
    }

    private static func persist(
        _ study: ExactResidualInterventionStudy, path: String?
    ) throws {
        if let path { try study.write(to: path) }
    }
}

public enum ResidualDiagnosticError: LocalizedError {
    case invalidConfiguration
    case incompleteActivationCollection

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            "Exact residual diagnostic parameters are invalid."
        case .incompleteActivationCollection:
            "Both last-user and post-instruction activation collections are required."
        }
    }
}
