#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerOptimize {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 9,
              let trials = Int(args[6]), let measurement = Int(args[7]),
              let evaluation = Int(args[8])
        else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-optimize BF16_MODEL MEASUREMENT_JSON HOLDOUT_JSON WORK_DIR OUTPUT_MODEL TRIALS MEASURE_CASES EVAL_CASES\n".utf8))
            exit(2)
        }
        let environment = ProcessInfo.processInfo.environment
        let normalizationName = environment["ABSLAYER_ROW_NORMALIZATION"] ?? "full"
        guard let normalization = WeightNormalization(rawValue: normalizationName) else {
            FileHandle.standardError.write(Data(
                "ABSLAYER_ROW_NORMALIZATION must be none, pre, or full\n".utf8))
            exit(2)
        }
        let fullRank = Int(environment["ABSLAYER_FULL_NORMALIZATION_RANK"] ?? "3") ?? 3
        let winsorValue = Float(environment["ABSLAYER_WINSOR_QUANTILE"] ?? "1.0") ?? 1
        let startupTrials = Int(environment["ABSLAYER_STARTUP_TRIALS"] ?? "")
        let maximumRefusalRate = Double(
            environment["ABSLAYER_MAX_REFUSAL_RATE"] ?? "0.10") ?? 0.10
        let finalEvaluationCases = Int(
            environment["ABSLAYER_FINAL_EVAL_CASES"] ?? "")
        let subspaceRank = Int(environment["ABSLAYER_SUBSPACE_RANK"] ?? "1") ?? 1
        let extractionName = environment["ABSLAYER_DIRECTION_METHOD"] ?? "centroidDifference"
        guard let extractionMethod = DirectionExtractionMethod(rawValue: extractionName) else {
            FileHandle.standardError.write(Data(
                "ABSLAYER_DIRECTION_METHOD must be centroidDifference, meanDifference, or whitenedSVD\n".utf8))
            exit(2)
        }
        let study = try await OptimizationEngine.run(OptimizationRequest(
            sourceModel: args[1],
            measurementModel: environment["ABSLAYER_MEASUREMENT_MODEL"],
            measurementPairs: try PromptFile.load(args[2]),
            evaluationPairs: try PromptFile.load(args[3]),
            workDirectory: args[4], outputModel: args[5], trialCount: trials,
            measurementCases: measurement, evaluationCases: evaluation,
            finalEvaluationCases: finalEvaluationCases,
            startupTrialCount: startupTrials,
            maximumRefusalRate: maximumRefusalRate,
            subspaceRank: subspaceRank,
            extractionMethod: extractionMethod,
            exportFinalModel: environment["ABSLAYER_SKIP_EXPORT"] != "1",
            normalization: normalization, fullNormalizationRank: fullRank,
            winsorizationQuantile: winsorValue < 1 ? winsorValue : nil))
        if let best = study.best {
            print(String(
                format: "Best trial %d: refusal=%.3f controls=%.3f KL=%.6f",
                best.index + 1, best.metrics.refusalRate,
                best.metrics.controlFailureRate, best.metrics.firstTokenKL))
        }
        if let final = study.finalVerification {
            print(String(
                format: "Exact BF16 overall %@ (utility=%@, abliteration=%@): refusal=%.3f controls=%.3f KL=%.6f",
                final.passedAll ? "PASS" : "FAIL",
                final.passedGuardrails ? "PASS" : "FAIL",
                final.passedAbliteration == true ? "PASS" : "FAIL",
                final.metrics.refusalRate, final.metrics.controlFailureRate,
                final.metrics.firstTokenKL))
        }
    }
}
