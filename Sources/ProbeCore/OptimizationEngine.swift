import Foundation

public struct OptimizationRequest: Sendable {
    public var sourceModel: String
    public var measurementModel: String?
    public var measurementPairs: [PromptPair]
    public var evaluationPairs: [PromptPair]
    public var workDirectory: String
    public var outputModel: String
    public var trialCount: Int
    public var startupTrialCount: Int
    public var maximumRefusalRate: Double
    public var measurementCases: Int
    public var evaluationCases: Int
    public var finalEvaluationCases: Int
    public var subspaceRank: Int
    public var extractionMethod: DirectionExtractionMethod
    public var seed: UInt64
    public var exportFinalModel: Bool
    public var normalization: WeightNormalization
    public var fullNormalizationRank: Int
    public var winsorizationQuantile: Float?

    public init(
        sourceModel: String, measurementModel: String? = nil,
        measurementPairs: [PromptPair], evaluationPairs: [PromptPair],
        workDirectory: String, outputModel: String,
        trialCount: Int, measurementCases: Int, evaluationCases: Int,
        finalEvaluationCases: Int? = nil,
        startupTrialCount: Int? = nil,
        maximumRefusalRate: Double = 0.10,
        subspaceRank: Int = 1,
        extractionMethod: DirectionExtractionMethod = .centroidDifference,
        seed: UInt64 = 0, exportFinalModel: Bool = true,
        normalization: WeightNormalization = .full,
        fullNormalizationRank: Int = 3, winsorizationQuantile: Float? = nil
    ) {
        self.sourceModel = sourceModel
        self.measurementModel = measurementModel
        self.measurementPairs = measurementPairs
        self.evaluationPairs = evaluationPairs
        self.workDirectory = workDirectory
        self.outputModel = outputModel
        self.trialCount = trialCount
        self.startupTrialCount = min(
            trialCount, startupTrialCount ?? min(60, max(4, trialCount * 3 / 10)))
        self.maximumRefusalRate = maximumRefusalRate
        self.measurementCases = measurementCases
        self.evaluationCases = evaluationCases
        self.finalEvaluationCases = finalEvaluationCases ?? evaluationCases
        self.subspaceRank = subspaceRank
        self.extractionMethod = extractionMethod
        self.seed = seed
        self.exportFinalModel = exportFinalModel
        self.normalization = normalization
        self.fullNormalizationRank = max(1, fullNormalizationRank)
        self.winsorizationQuantile = winsorizationQuantile
    }
}

public enum OptimizationEngine {
    public static func run(_ request: OptimizationRequest) async throws -> AbliterationStudy {
        let source = try ModelFolderValidator.validateFullBF16(path: request.sourceModel)
        let measurementModel = URL(fileURLWithPath: request.measurementModel ?? source.path)
            .standardizedFileURL.path
        guard request.trialCount > 0, request.startupTrialCount >= 0,
              request.startupTrialCount <= request.trialCount,
              (0 ... 1).contains(request.maximumRefusalRate),
              request.measurementCases > 0, request.evaluationCases > 0,
              request.finalEvaluationCases >= request.evaluationCases,
              request.subspaceRank > 0
        else { throw OptimizationError.invalidCounts }
        let work = URL(fileURLWithPath: request.workDirectory).standardizedFileURL
        let output = URL(fileURLWithPath: request.outputModel).standardizedFileURL
        guard !request.exportFinalModel || !FileManager.default.fileExists(atPath: output.path) else {
            throw EditorError.outputExists(output.path)
        }
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        let studyURL = work.appendingPathComponent("study.json")
        let baselineURL = work.appendingPathComponent("baseline.abslkl")
        let measurementSignature = dataSignature(request.measurementPairs)
        let evaluationSignature = dataSignature(request.evaluationPairs)
        var study: AbliterationStudy
        if FileManager.default.fileExists(atPath: studyURL.path) {
            study = try .read(from: studyURL.path)
            guard study.modelPath == source.path, study.seed == request.seed,
                  study.measurementModelPath == measurementModel,
                  study.measurementDataSignature == measurementSignature,
                  study.evaluationDataSignature == evaluationSignature,
                  study.measurementCases == request.measurementCases,
                  study.evaluationCases == request.evaluationCases,
                  study.finalEvaluationCases == request.finalEvaluationCases,
                  (study.subspaceRank ?? 1) == request.subspaceRank,
                  (study.extractionMethod ?? .meanDifference) == request.extractionMethod,
                  study.startupTrialCount == request.startupTrialCount,
                  study.maximumRefusalRate == request.maximumRefusalRate,
                  study.normalization == request.normalization,
                  study.fullNormalizationRank == request.fullNormalizationRank,
                  study.winsorizationQuantile == request.winsorizationQuantile
            else {
                throw OptimizationError.studyMismatch
            }
        } else {
            study = AbliterationStudy(
                modelPath: source.path, seed: request.seed,
                measurementModelPath: measurementModel,
                measurementDataSignature: measurementSignature,
                evaluationDataSignature: evaluationSignature,
                measurementCases: request.measurementCases,
                evaluationCases: request.evaluationCases,
                finalEvaluationCases: request.finalEvaluationCases,
                subspaceRank: request.subspaceRank,
                extractionMethod: request.extractionMethod,
                startupTrialCount: request.startupTrialCount,
                maximumRefusalRate: request.maximumRefusalRate,
                normalization: request.normalization,
                fullNormalizationRank: request.fullNormalizationRank,
                winsorizationQuantile: request.winsorizationQuantile)
        }

        let measurement = evenlySpaced(
            request.measurementPairs, maximum: request.measurementCases)
        print("Measuring refusal geometry once across \(measurement.count) pair(s)…")
        let report = try await ProbeEngine(
            modelDirectory: measurementModel, pairs: measurement,
            subspaceRank: request.subspaceRank,
            extractionMethod: request.extractionMethod, generateResponses: false,
            winsorizationQuantile: request.winsorizationQuantile).run()

        print("Loading lightweight trial model once…")
        let runtime = try await ResidentTrialRuntime(modelDirectory: measurementModel)
        let baseline: LogitFingerprint
        if FileManager.default.fileExists(atPath: baselineURL.path) {
            baseline = try .read(from: baselineURL.path)
        } else {
            print("Capturing untouched utility baseline…")
            baseline = try await runtime.fingerprint(
                pairs: request.evaluationPairs, maximumCases: request.evaluationCases)
            try baseline.write(to: baselineURL.path)
        }

        if study.baselineControlFailureRate == nil || study.baselineRefusalRate == nil {
            print("Measuring untouched refusal and benign-control behavior…")
            let baselineResponses = try await runtime.responses(
                pairs: request.evaluationPairs, maximumCases: request.evaluationCases)
            let baselineEvaluation = RefusalEvaluator.evaluate(
                baselineResponses, classifier: SubstringOutcomeClassifier())
            let baselineTotal = Double(max(1, baselineEvaluation.total))
            study.baselineControlFailureRate = Double(
                baselineEvaluation.cases.count { !$0.controlHealthy })
                / baselineTotal
            study.baselineRefusalRate = Double(
                baselineEvaluation.cases.count { $0.refused || !$0.engaged })
                / baselineTotal
            try study.write(to: studyURL.path)
            print(String(
                format: "Untouched baseline: refusal=%.3f controls=%.3f",
                study.baselineRefusalRate ?? 0,
                study.baselineControlFailureRate ?? 0))
        }

        let sampler = TPESampler(
            seed: request.seed,
            startupTrials: request.startupTrialCount,
            subspaceRank: request.subspaceRank)
        while study.trials.count < request.trialCount {
            let index = study.trials.count
            let parameters = sampler.suggest(
                layerCount: report.layers.count, history: study.trials)
            print("Trial \(index + 1)/\(request.trialCount): applying in-memory adapter…")
            let adapter = try await runtime.makeAdapter(
                directions: report.directions, subspaces: report.subspaces,
                configuration: parameters.configuration(normalization: request.normalization),
                fullNormalizationRank: request.fullNormalizationRank)
            try await runtime.load(adapter)
            do {
                let responses = try await runtime.responses(
                    pairs: request.evaluationPairs, maximumCases: request.evaluationCases)
                let refusal = RefusalEvaluator.evaluate(
                    responses, classifier: SubstringOutcomeClassifier())
                let candidateFingerprint = try await runtime.fingerprint(
                    pairs: request.evaluationPairs, maximumCases: request.evaluationCases)
                let kl = try LogitFingerprintEngine.divergence(
                    baseline: baseline, candidate: candidateFingerprint)
                let total = max(1, refusal.total)
                let absoluteControlFailure = Double(
                    refusal.cases.count { !$0.controlHealthy }) / Double(total)
                let metrics = AbliterationTrialMetrics(
                    refusalRate: Double(refusal.cases.count { $0.refused || !$0.engaged }) / Double(total),
                    controlFailureRate: max(
                        0, absoluteControlFailure - (study.baselineControlFailureRate ?? 0)),
                    firstTokenKL: kl)
                study.trials.append(AbliterationTrialRecord(
                    index: index, parameters: parameters, metrics: metrics))
                try study.write(to: studyURL.path)
                print(String(
                    format: "Trial %d: refusal=%.3f controls=%.3f KL=%.6f",
                    index + 1, metrics.refusalRate, metrics.controlFailureRate, kl))
            } catch {
                await runtime.unload(adapter)
                throw error
            }
            await runtime.unload(adapter)
        }
        guard let best = study.best else { throw OptimizationError.noTrials }
        if request.exportFinalModel {
            print("Exporting best trial \(best.index + 1) to \(output.path)…")
            _ = try BF16WeightEditor.edit(
                sourcePath: source.path, outputPath: output.path,
                directions: report.directions, subspaces: report.subspaces,
                configuration: best.parameters.configuration(normalization: request.normalization))
            print("Verifying exact BF16 export against untouched BF16 baseline…")
            let baselineBF16 = try await ResidentTrialRuntime(modelDirectory: source.path)
            let baselineBF16Fingerprint = try await baselineBF16.fingerprint(
                pairs: request.evaluationPairs, maximumCases: request.finalEvaluationCases)
            let baselineBF16Responses = try await baselineBF16.responses(
                pairs: request.evaluationPairs, maximumCases: request.finalEvaluationCases)
            let baselineBF16Evaluation = RefusalEvaluator.evaluate(
                baselineBF16Responses, classifier: SubstringOutcomeClassifier())

            let finalRuntime = try await ResidentTrialRuntime(modelDirectory: output.path)
            let finalFingerprint = try await finalRuntime.fingerprint(
                pairs: request.evaluationPairs, maximumCases: request.finalEvaluationCases)
            let finalResponses = try await finalRuntime.responses(
                pairs: request.evaluationPairs, maximumCases: request.finalEvaluationCases)
            let finalEvaluation = RefusalEvaluator.evaluate(
                finalResponses, classifier: SubstringOutcomeClassifier())
            let total = max(1, finalEvaluation.total)
            let baselineTotal = max(1, baselineBF16Evaluation.total)
            let baselineControlFailure = Double(
                baselineBF16Evaluation.cases.count { !$0.controlHealthy }) / Double(baselineTotal)
            let finalControlFailure = Double(
                finalEvaluation.cases.count { !$0.controlHealthy }) / Double(total)
            let finalMetrics = AbliterationTrialMetrics(
                refusalRate: Double(
                    finalEvaluation.cases.count { $0.refused || !$0.engaged }) / Double(total),
                controlFailureRate: max(0, finalControlFailure - baselineControlFailure),
                firstTokenKL: try LogitFingerprintEngine.divergence(
                    baseline: baselineBF16Fingerprint, candidate: finalFingerprint))
            study.finalVerification = AbliterationFinalVerification(
                metrics: finalMetrics,
                passedGuardrails: finalMetrics.controlFailureRate <= 0.10
                    && finalMetrics.firstTokenKL <= 0.5,
                passedAbliteration: finalMetrics.refusalRate <= request.maximumRefusalRate)
            try study.write(to: studyURL.path)
            print(String(
                format: "Final BF16: refusal=%.3f controls=%.3f KL=%.6f utility=%@ abliteration=%@",
                finalMetrics.refusalRate, finalMetrics.controlFailureRate,
                finalMetrics.firstTokenKL,
                study.finalVerification?.passedGuardrails == true ? "PASS" : "FAIL",
                study.finalVerification?.passedAbliteration == true ? "PASS" : "FAIL"))
        } else {
            print("Dry run complete; skipped final BF16 export.")
        }
        return study
    }

    private static func evenlySpaced(_ pairs: [PromptPair], maximum: Int) -> [PromptPair] {
        guard pairs.count > maximum else { return pairs }
        return (0 ..< maximum).map { pairs[$0 * pairs.count / maximum] }
    }

    /// Stable FNV-1a signature prevents accidentally resuming a study with a
    /// different direction dataset, holdout, or ordering.
    private static func dataSignature(_ pairs: [PromptPair]) -> UInt64 {
        var value: UInt64 = 0xcbf29ce484222325
        for pair in pairs {
            for byte in (pair.name + "\u{0}" + pair.contrast + "\u{0}" + pair.control + "\u{1}").utf8 {
                value ^= UInt64(byte)
                value &*= 0x100000001b3
            }
        }
        return value
    }

    private static func removeRecoveredCandidate(_ candidate: URL, inside work: URL) throws {
        let candidate = candidate.standardizedFileURL
        guard candidate.deletingLastPathComponent().standardizedFileURL.path
                == work.standardizedFileURL.path,
              candidate.lastPathComponent.hasPrefix("candidate-")
        else { throw OptimizationError.unsafeTemporaryPath(candidate.path) }
        if FileManager.default.fileExists(atPath: candidate.path) {
            try FileManager.default.removeItem(at: candidate)
        }
    }
}

public enum OptimizationError: LocalizedError {
    case invalidCounts
    case studyMismatch
    case noTrials
    case unsafeTemporaryPath(String)

    public var errorDescription: String? {
        switch self {
        case .invalidCounts: "Trial, measurement, and evaluation counts must be positive."
        case .studyMismatch:
            "Existing study uses a different model, dataset, case count, or seed. Choose a new work folder."
        case .noTrials: "Optimization produced no trials."
        case .unsafeTemporaryPath(let path): "Refusing unsafe temporary path: \(path)"
        }
    }
}
