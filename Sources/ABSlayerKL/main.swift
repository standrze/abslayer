#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerKL {
    static func main() async throws {
        let args = CommandLine.arguments
        if args.count == 4, args[1] == "compare" {
            let baseline = try LogitFingerprint.read(from: args[2])
            let candidate = try LogitFingerprint.read(from: args[3])
            let divergence = try LogitFingerprintEngine.divergence(
                baseline: baseline, candidate: candidate)
            print(String(format: "First-token KL divergence: %.6f", divergence))
            return
        }
        guard (args.count == 5 || args.count == 6), let maximum = Int(args[3]) else {
            let usage =
                "usage: abslayer-kl MODEL_FOLDER PROMPTS_JSON MAX_CASES OUTPUT_FINGERPRINT [BASELINE_FINGERPRINT]\n"
                + "       abslayer-kl compare BASELINE_FINGERPRINT CANDIDATE_FINGERPRINT\n"
            FileHandle.standardError.write(Data(usage.utf8))
            exit(2)
        }
        let adapter: AdapterRuntimeOptions
        do {
            adapter = try AdapterRuntimeOptions.parse(
                environment: ProcessInfo.processInfo.environment)
        } catch {
            FileHandle.standardError.write(Data("Error: \(error.localizedDescription)\n".utf8))
            exit(2)
        }
        let model = try ModelFolderValidator.validateFullBF16(path: args[1])
        let pairs = try PromptFile.load(args[2])
        if let rawTopK = ProcessInfo.processInfo.environment["ABSLAYER_SEQUENCE_TOP_K"] {
            guard let topK = Int(rawTopK), topK > 0 else {
                FileHandle.standardError.write(Data(
                    "ABSLAYER_SEQUENCE_TOP_K must be a positive integer.\n".utf8))
                exit(2)
            }
            let baseline = args.count == 6
                ? try SequenceLogitFingerprint.read(from: args[5]) : nil
            guard baseline == nil || baseline?.topK == topK else {
                FileHandle.standardError.write(Data(
                    "ABSLAYER_SEQUENCE_TOP_K must match the baseline fingerprint.\n".utf8))
                exit(2)
            }
            let fingerprint = try await LogitFingerprintEngine.captureSequence(
                modelDirectory: model.path, pairs: pairs, maximumCases: maximum,
                topK: topK, reference: baseline,
                adapterDirectory: adapter.directory,
                adapterScaleOverride: adapter.scaleOverride)
            try fingerprint.write(to: args[4])
            print("Saved sequence fingerprint \(args[4])")
            if let baseline {
                let metrics = try SequenceMetricEngine.compare(
                    baseline: baseline, candidate: fingerprint)
                print(metrics.rendered)
            }
            return
        }
        let fingerprint = try await LogitFingerprintEngine.capture(
            modelDirectory: model.path, pairs: pairs, maximumCases: maximum,
            adapterDirectory: adapter.directory,
            adapterScaleOverride: adapter.scaleOverride)
        try fingerprint.write(to: args[4])
        print("Saved \(args[4])")
        if args.count == 6 {
            let baseline = try LogitFingerprint.read(from: args[5])
            let divergence = try LogitFingerprintEngine.divergence(
                baseline: baseline, candidate: fingerprint)
            print(String(format: "First-token KL divergence: %.6f", divergence))
        }
    }
}
