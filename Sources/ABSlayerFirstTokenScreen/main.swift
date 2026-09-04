#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerFirstTokenScreen {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count >= 5,
              let maximumCases = Int(arguments[3]),
              maximumCases > 0
        else {
            FileHandle.standardError.write(Data(usage.utf8))
            exit(2)
        }

        let environment = ProcessInfo.processInfo.environment
        let starterConfiguration: FirstTokenStarterConfiguration
        let unloadTolerance: Double
        do {
            starterConfiguration = FirstTokenStarterConfiguration(
                refusalPhrases: try phrases(
                    environment: environment,
                    key: "ABSLAYER_REFUSAL_STARTERS_JSON",
                    fallback: FirstTokenStarterConfiguration.default.refusalPhrases),
                compliancePhrases: try phrases(
                    environment: environment,
                    key: "ABSLAYER_COMPLIANCE_STARTERS_JSON",
                    fallback: FirstTokenStarterConfiguration.default.compliancePhrases))
            if let raw = environment["ABSLAYER_UNLOAD_TOLERANCE"] {
                guard let value = Double(raw), value.isFinite, value >= 0 else {
                    throw CLIError.invalidUnloadTolerance(raw)
                }
                unloadTolerance = value
            } else {
                unloadTolerance = 1e-5
            }
        } catch {
            FileHandle.standardError.write(Data("Error: \(error.localizedDescription)\n".utf8))
            exit(2)
        }

        let model = try ModelFolderValidator.validateFullBF16(path: arguments[1])
        let pairs = try PromptFile.load(arguments[2])
        let adapters = Array(arguments.dropFirst(5))
        print(FirstTokenScreenEngine.proxyNotice)
        let report = try await FirstTokenScreenEngine.run(
            modelDirectory: model.path,
            promptFile: arguments[2],
            pairs: pairs,
            maximumCases: maximumCases,
            adapterDirectories: adapters,
            starterConfiguration: starterConfiguration,
            unloadTolerance: unloadTolerance,
            progress: { print($0) })

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(report).write(
            to: URL(fileURLWithPath: arguments[4]).standardizedFileURL,
            options: .atomic)

        print(String(
            format: "baseline contrast starter log-odds=%+.4f control=%+.4f",
            report.baselineContrast.starterSummary.meanComplianceToRefusalLogOdds,
            report.baselineControl.starterSummary.meanComplianceToRefusalLogOdds))
        for adapter in report.adapters {
            let name = URL(fileURLWithPath: adapter.adapterDirectory).lastPathComponent
            print(String(
                format: "%@ | contrast KL=%.6f top1=%d/%d starter-delta=%+.4f | control KL=%.6f top1=%d/%d starter-delta=%+.4f | unload=pass",
                name,
                adapter.contrast.exactMeanKLFromBaseline,
                adapter.contrast.top1ChangeCount,
                report.selectedCaseNames.count,
                adapter.contrast.meanComplianceToRefusalLogOddsDeltaFromBaseline,
                adapter.control.exactMeanKLFromBaseline,
                adapter.control.top1ChangeCount,
                report.selectedCaseNames.count,
                adapter.control.meanComplianceToRefusalLogOddsDeltaFromBaseline))
        }
        print("Saved machine-readable screen to \(arguments[4])")
    }

    private static func phrases(
        environment: [String: String], key: String, fallback: [String]
    ) throws -> [String] {
        guard let raw = environment[key] else { return fallback }
        guard let data = raw.data(using: .utf8),
              let values = try? JSONDecoder().decode([String].self, from: data),
              !values.isEmpty
        else { throw CLIError.invalidPhraseJSON(key) }
        return values
    }

    private static let usage = """
        usage: abslayer-first-token-screen MODEL_FOLDER PROMPTS_JSON MAX_CASES OUTPUT_JSON [ADAPTER_DIR ...]

        The BF16 model is loaded once. An untouched baseline is captured, then every
        adapter is loaded at runtime scale 1, captured, unloaded, and checked against
        an untouched sentinel before the next adapter. This is a first-token causal
        screening proxy, not a semantic compliance verdict.

        Optional environment:
          ABSLAYER_REFUSAL_STARTERS_JSON='["I","Sorry","Unfortunately","As"]'
          ABSLAYER_COMPLIANCE_STARTERS_JSON='["Sure","Certainly","Here","Below"]'
          ABSLAYER_UNLOAD_TOLERANCE=1e-5
        """
}

private enum CLIError: LocalizedError {
    case invalidPhraseJSON(String)
    case invalidUnloadTolerance(String)

    var errorDescription: String? {
        switch self {
        case .invalidPhraseJSON(let key):
            "\(key) must be a non-empty JSON array of strings."
        case .invalidUnloadTolerance(let value):
            "ABSLAYER_UNLOAD_TOLERANCE must be finite and non-negative, not '\(value)'."
        }
    }
}
