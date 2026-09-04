#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerAnswerTransport {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count >= 2 else {
            usage()
            exit(2)
        }
        switch arguments[1] {
        case "fit" where arguments.count == 6:
            try await fit(
                modelPath: arguments[2],
                measurementPath: arguments[3],
                layerText: arguments[4],
                outputPath: arguments[5])
        case "evaluate" where arguments.count == 7:
            try await evaluate(
                modelPath: arguments[2],
                artifactPath: arguments[3],
                evaluationPath: arguments[4],
                reportPath: arguments[5],
                responsesPath: arguments[6])
        case "derive" where arguments.count == 4:
            try derive(
                artifactPath: arguments[2],
                outputPath: arguments[3])
        default:
            usage()
            exit(2)
        }
    }

    private static func derive(
        artifactPath: String, outputPath: String
    ) throws {
        let environment = ProcessInfo.processInfo.environment
        let artifact = try AnswerCenteredTransportArtifact.read(
            from: artifactPath)
        let strength = try float(
            environment, "ABSLAYER_TRANSPORT_STRENGTH",
            fallback: artifact.fit.plan.strength)
        let schedule = environment["ABSLAYER_TRANSPORT_SCHEDULE"] == nil
            ? artifact.intervention.schedule
            : try applicationSchedule(environment)
        let gateOverride = try derivationGateOverride(environment)
        guard let derived = artifact.derived(
            from: artifactPath,
            strength: strength,
            schedule: schedule,
            gateOverride: gateOverride)
        else { throw CLIError.fitFailed }
        try derived.write(to: outputPath)
        print(String(
            format: "Derived %@ rank=%d L0=%d strength=%.3f gate=%@ schedule=%@ -> %@",
            derived.fit.plan.method.rawValue,
            derived.fit.plan.rank,
            derived.layerZeroBased,
            derived.fit.plan.strength,
            gateOverride.rawValue,
            derived.intervention.schedule?.diagnosticName
                ?? "legacy-compatible",
            URL(fileURLWithPath: outputPath).standardizedFileURL.path))
    }

    private static func fit(
        modelPath: String, measurementPath: String,
        layerText: String, outputPath: String
    ) async throws {
        try rejectProtectedDataPath(measurementPath)
        guard let layerZeroBased = Int(layerText), layerZeroBased >= 0 else {
            throw CLIError.invalidLayer(layerText)
        }
        let environment = ProcessInfo.processInfo.environment
        let inspection = try ModelFolderValidator.validateFullBF16(
            path: modelPath)
        if let count = inspection.decoderLayerCount,
           layerZeroBased >= count
        {
            throw CLIError.layerOutsideModel(
                layerZeroBased, decoderLayerCount: count)
        }
        let allPairs = try PromptFile.load(measurementPath)
        let maximumCases = try integer(
            environment, "ABSLAYER_TRANSPORT_MEASURE_CASES", fallback: 96)
        let pairs = evenlySpaced(allPairs, maximum: maximumCases)
        let method = try transportMethod(environment)
        let rank = try integer(
            environment, "ABSLAYER_TRANSPORT_RANK", fallback: 2)
        let ridge = try float(
            environment, "ABSLAYER_TRANSPORT_RIDGE", fallback: 1e-3)
        let strength = try float(
            environment, "ABSLAYER_TRANSPORT_STRENGTH", fallback: 1)
        let gateQuantile = try optionalQuantile(environment)
        let schedule = try applicationSchedule(environment)

        FileHandle.standardError.write(Data(
            "DEVELOPMENT ONLY: fit only on outcome-screened development pairs. Dataset intent does not prove the contrast was refused or the control was substantively answered. Never pass a frozen audit artifact.\n".utf8))
        print(
            "Capturing \(pairs.count) post-instruction residual pairs at "
                + "zero-based layer \(layerZeroBased)")
        let collections = try await ActivationCollectionEngine(
            modelDirectory: inspection.path, pairs: pairs
        ).run(positions: [.postInstruction]) { completed, total in
            print("captured \(completed)/\(total)")
        }
        guard let collection = collections[.postInstruction],
              collection.layers.indices.contains(layerZeroBased)
        else { throw CLIError.captureMissing(layerZeroBased) }
        let layer = collection.layers[layerZeroBased]
        guard let fit = AnswerCenteredTransportFitter.fit(
            refusedActivations: layer.contrast,
            answeredActivations: layer.control,
            method: method,
            rank: rank,
            covarianceRidge: ridge,
            controlGateQuantile: gateQuantile,
            strength: strength),
              let intervention = AnswerCenteredResidualIntervention(
                layer: layerZeroBased + 1,
                plan: fit.plan,
                schedule: schedule)
        else { throw CLIError.fitFailed }
        let artifact = AnswerCenteredTransportArtifact(
            modelPath: inspection.path,
            measurementPath: measurementPath,
            pairNames: collection.pairs.map(\.name),
            tokenPosition: .postInstruction,
            layerZeroBased: layerZeroBased,
            fit: fit,
            intervention: intervention)
        try artifact.write(to: outputPath)
        let sourceGate = fit.diagnostics.meanRefusedGate.map {
            String(format: "%.3f", $0)
        } ?? "unavailable"
        let targetGate = fit.diagnostics.meanAnsweredGate.map {
            String(format: "%.3f", $0)
        } ?? "unavailable"
        print(String(
            format: "Fitted %@ rank=%d L0=%d explained-variance=%.4f covariance-RMSE=%.6f source-gate=%@ target-gate=%@ schedule=%@ -> %@",
            method.rawValue,
            fit.plan.rank,
            layerZeroBased,
            fit.diagnostics.explainedVarianceFraction,
            fit.diagnostics.projectedCovarianceAlignmentRMSE,
            sourceGate,
            targetGate,
            schedule?.diagnosticName ?? "legacy-compatible",
            URL(fileURLWithPath: outputPath).standardizedFileURL.path))
        print(artifact.warning)
    }

    private static func evaluate(
        modelPath: String, artifactPath: String,
        evaluationPath: String, reportPath: String,
        responsesPath: String
    ) async throws {
        try rejectProtectedDataPath(evaluationPath)
        let environment = ProcessInfo.processInfo.environment
        let artifact = try AnswerCenteredTransportArtifact.read(
            from: artifactPath)
        let normalizedModel = URL(fileURLWithPath: modelPath)
            .standardizedFileURL.path
        guard normalizedModel == artifact.modelPath else {
            throw CLIError.modelMismatch(
                expected: artifact.modelPath, actual: normalizedModel)
        }
        let pairs = try PromptFile.load(evaluationPath)
        let maximumCases = try integer(
            environment, "ABSLAYER_TRANSPORT_EVAL_CASES",
            fallback: pairs.count)
        let topK = try integer(
            environment, "ABSLAYER_TRANSPORT_SEQUENCE_TOP_K",
            fallback: 64)
        let maximumTokens = try integer(
            environment, "ABSLAYER_TRANSPORT_MAX_TOKENS",
            fallback: 100)
        let maximumReferenceTokens = try integer(
            environment, "ABSLAYER_TRANSPORT_REFERENCE_TOKENS",
            fallback: 128)

        FileHandle.standardError.write(Data(
            "DEVELOPMENT ONLY: this command measures preservation and records responses for an independent semantic judge. It does not certify abliteration and must not consume frozen audits during tuning.\n".utf8))
        let runtime = try await ResidentTrialRuntime(
            modelDirectory: normalizedModel)
        let baselineFirst = try await runtime.fingerprint(
            pairs: pairs, maximumCases: maximumCases)
        let baselineSequence = try await runtime
            .controlPromptPrefixFingerprint(
                pairs: pairs, maximumCases: maximumCases,
                topK: topK)

        try await runtime.install(artifact.intervention)
        let candidateFirst: LogitFingerprint
        let candidateSequence: SequenceLogitFingerprint
        let continuation: TeacherForcedContinuationMetricSummary?
        let responses: [PromptResult]
        do {
            candidateFirst = try await runtime.fingerprint(
                pairs: pairs, maximumCases: maximumCases)
            candidateSequence = try await runtime
                .controlPromptPrefixFingerprint(
                    pairs: pairs, maximumCases: maximumCases,
                    topK: topK, reference: baselineSequence)
            continuation = try await runtime
                .teacherForcedControlContinuationMetrics(
                    pairs: pairs, maximumCases: maximumCases,
                    maximumReferenceTokens: maximumReferenceTokens)
            responses = try await runtime.responses(
                pairs: pairs, maximumCases: maximumCases,
                maximumTokens: maximumTokens)
        } catch {
            await runtime.clearResidualIntervention()
            throw error
        }
        await runtime.clearResidualIntervention()
        let firstTokenKL = try LogitFingerprintEngine.divergence(
            baseline: baselineFirst, candidate: candidateFirst)
        let sequence = try SequenceMetricEngine.compare(
            baseline: baselineSequence,
            candidate: candidateSequence)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(responses).write(
            to: URL(fileURLWithPath: responsesPath)
                .standardizedFileURL,
            options: .atomic)
        let report = EvaluationReport(
            artifactPath: artifactPath,
            modelPath: normalizedModel,
            evaluationPath: evaluationPath,
            promptNames: baselineFirst.promptNames,
            firstTokenKL: firstTokenKL,
            controlPromptPrefixKLLowerBound:
                sequence.controlPromptPrefixKLLowerBound,
            baselineControlPromptPerplexity:
                sequence.baselinePerplexity,
            candidateControlPromptPerplexity:
                sequence.candidatePerplexity,
            controlPromptPerplexityRatio:
                sequence.perplexityRatio,
            teacherForcedControlContinuation: continuation,
            responsesPath: responsesPath)
        try encoder.encode(report).write(
            to: URL(fileURLWithPath: reportPath)
                .standardizedFileURL,
            options: .atomic)
        print(String(
            format: "Transport evaluation: first-KL=%.6f control-prefix-KL-lower-bound=%.6f prompt-ppl-ratio=%.4f",
            firstTokenKL,
            sequence.controlPromptPrefixKLLowerBound,
            sequence.perplexityRatio))
        if let continuation {
            print(String(
                format: "Teacher-forced control continuation: mean-KL=%.6f p95-KL=%.6f max-KL=%.6f",
                continuation.exactMeanKL,
                continuation.exactP95KL,
                continuation.exactMaximumKL))
        } else {
            print("Teacher-forced control continuation: unavailable (no references)")
        }
        print("Semantic-judge input -> \(responsesPath)")
        print(report.warning)
    }

    private static func transportMethod(
        _ environment: [String: String]
    ) throws -> AnswerCenteredTransportMethod {
        let raw = environment["ABSLAYER_TRANSPORT_METHOD"]
            ?? AnswerCenteredTransportMethod.pcaGaussianOT.rawValue
        guard let value = AnswerCenteredTransportMethod(rawValue:
            raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
        else { throw CLIError.invalidMethod(raw) }
        return value
    }

    private static func optionalQuantile(
        _ environment: [String: String]
    ) throws -> Float? {
        let raw = environment["ABSLAYER_TRANSPORT_GATE_QUANTILE"]
            ?? "0.99"
        let trimmed = raw.trimmingCharacters(
            in: .whitespacesAndNewlines).lowercased()
        if trimmed == "none" { return nil }
        guard let value = Float(trimmed), (0 ... 1).contains(value)
        else { throw CLIError.invalidQuantile(raw) }
        return value
    }

    private static func applicationSchedule(
        _ environment: [String: String]
    ) throws -> ExactResidualInterventionSchedule? {
        let raw = environment["ABSLAYER_TRANSPORT_SCHEDULE"] ?? "0..7"
        let trimmed = raw.trimmingCharacters(
            in: .whitespacesAndNewlines).lowercased()
        if trimmed == "legacy-compatible" { return nil }
        guard let preset = ExactResidualInterventionSchedulePreset(
            rawValue: trimmed)
        else { throw CLIError.invalidSchedule(raw) }
        return preset.schedule
    }

    private static func derivationGateOverride(
        _ environment: [String: String]
    ) throws -> AnswerCenteredTransportGateOverride {
        let raw = environment["ABSLAYER_TRANSPORT_GATE"] ?? "keep"
        let normalized = raw.trimmingCharacters(
            in: .whitespacesAndNewlines).lowercased()
        guard let value = AnswerCenteredTransportGateOverride(
            rawValue: normalized)
        else { throw CLIError.invalidGateOverride(raw) }
        return value
    }

    private static func integer(
        _ environment: [String: String], _ key: String, fallback: Int
    ) throws -> Int {
        guard let raw = environment[key] else {
            guard fallback > 0 else { throw CLIError.invalidNumber(key, "") }
            return fallback
        }
        guard let value = Int(raw.trimmingCharacters(
            in: .whitespacesAndNewlines)), value > 0
        else { throw CLIError.invalidNumber(key, raw) }
        return value
    }

    private static func float(
        _ environment: [String: String], _ key: String, fallback: Float
    ) throws -> Float {
        guard let raw = environment[key] else { return fallback }
        guard let value = Float(raw.trimmingCharacters(
            in: .whitespacesAndNewlines)), value.isFinite, value > 0
        else { throw CLIError.invalidNumber(key, raw) }
        return value
    }

    private static func evenlySpaced<T>(
        _ values: [T], maximum: Int
    ) -> [T] {
        guard maximum > 0, values.count > maximum else { return values }
        return (0 ..< maximum).map {
            values[$0 * values.count / maximum]
        }
    }

    private static func rejectProtectedDataPath(_ path: String) throws {
        let normalized = URL(fileURLWithPath: path).standardizedFileURL.path
            .lowercased()
        let protectedMarkers = [
            "audit-primary", "audit-confirmation", "frozen-audit",
            "sealed-audit",
        ]
        if let marker = protectedMarkers.first(where: normalized.contains) {
            throw CLIError.protectedData(path: path, marker: marker)
        }
    }

    private static func usage() {
        FileHandle.standardError.write(Data(
            """
            usage:
              abslayer-answer-transport fit \
                MODEL_BF16 MEASUREMENT_DEV_JSON LAYER_ZERO_BASED ARTIFACT_JSON

              abslayer-answer-transport evaluate \
                MODEL_BF16 ARTIFACT_JSON EVALUATION_DEV_JSON REPORT_JSON RESPONSES_JSON

              abslayer-answer-transport derive \
                INPUT_ARTIFACT_JSON OUTPUT_ARTIFACT_JSON

            fit environment:
              ABSLAYER_TRANSPORT_METHOD=pca-gaussian-ot|affine-centroid
              ABSLAYER_TRANSPORT_RANK=2
              ABSLAYER_TRANSPORT_RIDGE=0.001
              ABSLAYER_TRANSPORT_GATE_QUANTILE=0.99|none
              ABSLAYER_TRANSPORT_STRENGTH=1.0
              ABSLAYER_TRANSPORT_SCHEDULE=post|token0|0..1|0..7|legacy
              ABSLAYER_TRANSPORT_MEASURE_CASES=96

            evaluate environment:
              ABSLAYER_TRANSPORT_EVAL_CASES=30
              ABSLAYER_TRANSPORT_SEQUENCE_TOP_K=64
              ABSLAYER_TRANSPORT_MAX_TOKENS=100
              ABSLAYER_TRANSPORT_REFERENCE_TOKENS=128

            derive environment:
              ABSLAYER_TRANSPORT_STRENGTH=0.75
              ABSLAYER_TRANSPORT_SCHEDULE=post|token0|0..1|0..7|legacy
              ABSLAYER_TRANSPORT_GATE=keep|none

            Fit and tune on development data only. Layer indices are zero-based.
            """.utf8))
    }
}

private struct EvaluationReport: Codable {
    let schemaVersion = 1
    let status = "development_evaluation"
    let warning = "Preservation metrics and raw responses do not certify abliteration. Require independent semantic judgments and all configured gates before promotion."
    let artifactPath: String
    let modelPath: String
    let evaluationPath: String
    let promptNames: [String]
    let firstTokenKL: Double
    let controlPromptPrefixKLLowerBound: Double
    let baselineControlPromptPerplexity: Double
    let candidateControlPromptPerplexity: Double
    let controlPromptPerplexityRatio: Double
    let teacherForcedControlContinuation:
        TeacherForcedContinuationMetricSummary?
    let responsesPath: String

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case status
        case warning
        case artifactPath = "artifact_path"
        case modelPath = "model_path"
        case evaluationPath = "evaluation_path"
        case promptNames = "prompt_names"
        case firstTokenKL = "first_token_kl"
        case controlPromptPrefixKLLowerBound =
            "control_prompt_prefix_kl_lower_bound"
        case baselineControlPromptPerplexity =
            "baseline_control_prompt_perplexity"
        case candidateControlPromptPerplexity =
            "candidate_control_prompt_perplexity"
        case controlPromptPerplexityRatio =
            "control_prompt_perplexity_ratio"
        case teacherForcedControlContinuation =
            "teacher_forced_control_continuation"
        case responsesPath = "responses_path"
    }
}

private enum CLIError: LocalizedError {
    case invalidLayer(String)
    case layerOutsideModel(Int, decoderLayerCount: Int)
    case captureMissing(Int)
    case fitFailed
    case modelMismatch(expected: String, actual: String)
    case invalidMethod(String)
    case invalidQuantile(String)
    case invalidSchedule(String)
    case invalidNumber(String, String)
    case protectedData(path: String, marker: String)
    case invalidGateOverride(String)

    var errorDescription: String? {
        switch self {
        case .invalidLayer(let value):
            "Layer must be a non-negative zero-based integer, not '\(value)'."
        case .layerOutsideModel(let layer, let count):
            "Zero-based layer \(layer) is outside the model's 0..<\(count) decoder layers."
        case .captureMissing(let layer):
            "Post-instruction activation capture did not include zero-based layer \(layer)."
        case .fitFailed:
            "The selected states could not produce a stable answer-centered transport map."
        case .modelMismatch(let expected, let actual):
            "The artifact was fitted for '\(expected)', not '\(actual)'."
        case .invalidMethod(let value):
            "ABSLAYER_TRANSPORT_METHOD must be pca-gaussian-ot or affine-centroid, not '\(value)'."
        case .invalidQuantile(let value):
            "ABSLAYER_TRANSPORT_GATE_QUANTILE must be in [0,1] or none, not '\(value)'."
        case .invalidSchedule(let value):
            "ABSLAYER_TRANSPORT_SCHEDULE must be post, token0, 0..1, 0..7, legacy, or legacy-compatible, not '\(value)'."
        case .invalidNumber(let key, let value):
            "\(key) must be a positive number, not '\(value)'."
        case .protectedData(let path, let marker):
            "Refusing protected dataset path '\(path)' (matched '\(marker)'). Fit and tune on development data only."
        case .invalidGateOverride(let value):
            "ABSLAYER_TRANSPORT_GATE must be keep or none for derive, not '\(value)'."
        }
    }
}
