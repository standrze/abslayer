#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerContinuationKL {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 6,
              let maximumCases = Int(args[3]), maximumCases > 0,
              let maximumReferenceTokens = Int(args[4]), maximumReferenceTokens > 0
        else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-continuation-kl MODEL_FOLDER PROMPTS_JSON MAX_CASES MAX_REFERENCE_TOKENS OUTPUT_JSON\n".utf8))
            exit(2)
        }

        let options: AdapterRuntimeOptions
        do {
            options = try AdapterRuntimeOptions.parse(
                environment: ProcessInfo.processInfo.environment)
            try MLXResourceGuard.apply(
                environment: ProcessInfo.processInfo.environment)
        } catch {
            FileHandle.standardError.write(Data(
                "Error: \(error.localizedDescription)\n".utf8))
            exit(2)
        }
        guard let adapterDirectory = options.directory else {
            FileHandle.standardError.write(Data(
                "Error: ABSLAYER_ADAPTER_DIR is required; this command compares a LoRA adapter with the untouched base model.\n".utf8))
            exit(2)
        }

        let model = try ModelFolderValidator.validateFullBF16(path: args[1])
        let pairs = try PromptFile.load(args[2])
        let report = try await LoRAContinuationKLEngine.run(
            modelDirectory: model.path,
            adapterDirectory: adapterDirectory,
            adapterScaleOverride: options.scaleOverride,
            pairs: pairs,
            maximumCases: maximumCases,
            maximumReferenceTokens: maximumReferenceTokens)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(
            to: URL(fileURLWithPath: args[5]).standardizedFileURL,
            options: .atomic)
        print(report.summary.rendered)
        print("Saved exact continuation-KL report to \(args[5])")
    }
}
