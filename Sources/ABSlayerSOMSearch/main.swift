#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerSOMSearch {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 8,
              let sourceLayer = Int(arguments[4]), sourceLayer >= 0,
              let applicationScope = SOMApplicationScope(rawValue: arguments[5]),
              let maximumCases = Int(arguments[6]), maximumCases > 0
        else {
            FileHandle.standardError.write(Data(usage.utf8))
            exit(2)
        }
        let environment = ProcessInfo.processInfo.environment
        let configuration: SOMSubsetSearchConfiguration
        let starterConfiguration: FirstTokenStarterConfiguration
        let finalistDirectory: String?
        do {
            guard let components = SOMApplicationComponents(
                rawValue: environment["ABSLAYER_SOM_COMPONENTS"] ?? "omlp")
            else {
                throw CLIError.invalidEnum(
                    key: "ABSLAYER_SOM_COMPONENTS",
                    value: environment["ABSLAYER_SOM_COMPONENTS"] ?? "")
            }
            configuration = SOMSubsetSearchConfiguration(
                sourceLayerZeroBased: sourceLayer,
                applicationScope: applicationScope,
                components: components,
                maximumCases: maximumCases,
                maximumDepth: try integer(
                    environment, key: "ABSLAYER_SOM_MAX_DEPTH", fallback: 7),
                beamWidth: try integer(
                    environment, key: "ABSLAYER_SOM_BEAM_WIDTH", fallback: 4),
                maximumFinalists: try integer(
                    environment, key: "ABSLAYER_SOM_MAX_FINALISTS", fallback: 8),
                contrastKLCeiling: try double(
                    environment, key: "ABSLAYER_SOM_CONTRAST_KL_CEILING", fallback: 1),
                controlKLCeiling: try double(
                    environment, key: "ABSLAYER_SOM_CONTROL_KL_CEILING", fallback: 1),
                unloadTolerance: try double(
                    environment, key: "ABSLAYER_UNLOAD_TOLERANCE", fallback: 1e-5),
                objective: SOMSubsetSearchObjective(
                    controlKLPenalty: try double(
                        environment, key: "ABSLAYER_SOM_CONTROL_KL_PENALTY",
                        fallback: 1),
                    controlStarterShiftPenalty: try double(
                        environment, key: "ABSLAYER_SOM_CONTROL_STARTER_PENALTY",
                        fallback: 0.5),
                    contrastKLPenalty: try double(
                        environment, key: "ABSLAYER_SOM_CONTRAST_KL_PENALTY",
                        fallback: 0.25)))
            starterConfiguration = FirstTokenStarterConfiguration(
                refusalPhrases: try phrases(
                    environment: environment,
                    key: "ABSLAYER_REFUSAL_STARTERS_JSON",
                    fallback: FirstTokenStarterConfiguration.default.refusalPhrases),
                compliancePhrases: try phrases(
                    environment: environment,
                    key: "ABSLAYER_COMPLIANCE_STARTERS_JSON",
                    fallback: FirstTokenStarterConfiguration.default.compliancePhrases))
            finalistDirectory = environment["ABSLAYER_SOM_FINALIST_DIR"]
        } catch {
            FileHandle.standardError.write(Data("Error: \(error.localizedDescription)\n".utf8))
            exit(2)
        }

        let model = try ModelFolderValidator.validateFullBF16(path: arguments[1])
        let pairs = try PromptFile.load(arguments[2])
        print(FirstTokenScreenEngine.proxyNotice)
        print("Search objective: \(configuration.objective.definition)")
        let report = try await SOMSubsetSearchEngine.run(
            modelDirectory: model.path,
            promptFile: arguments[2],
            pairs: pairs,
            candidateArchiveFile: arguments[3],
            configuration: configuration,
            starterConfiguration: starterConfiguration,
            finalistDirectory: finalistDirectory,
            progress: { print($0) })
        let output = URL(fileURLWithPath: arguments[7]).standardizedFileURL
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(report).write(to: output, options: .atomic)
        if let depth = report.depths.last {
            print("Final depth \(depth.depth) retained ordered IDs: \(depth.retainedOrderedLatticeIDs)")
        }
        for finalist in report.selectedFinalists {
            print(
                "Finalist depth=\(finalist.depth) "
                    + "IDs=[\(finalist.orderedLatticeIDs.map(String.init).joined(separator: ","))] "
                    + String(
                        format: "score=%+.5f contrastKL=%.5f controlKL=%.5f reason=%@",
                        finalist.objectiveScore,
                        finalist.rankingMetrics.contrastExactMeanKL,
                        finalist.rankingMetrics.controlExactMeanKL,
                        finalist.selectionReason.rawValue))
        }
        print("Saved machine-readable SOM subset search -> \(output.path)")
    }

    private static func phrases(
        environment: [String: String], key: String, fallback: [String]
    ) throws -> [String] {
        guard let raw = environment[key] else { return fallback }
        guard let data = raw.data(using: .utf8),
              let values = try? JSONDecoder().decode([String].self, from: data),
              !values.isEmpty
        else { throw CLIError.invalidJSON(key) }
        return values
    }

    private static func integer(
        _ environment: [String: String], key: String, fallback: Int
    ) throws -> Int {
        guard let raw = environment[key] else { return fallback }
        guard let value = Int(raw) else { throw CLIError.invalidNumber(key: key, value: raw) }
        return value
    }

    private static func double(
        _ environment: [String: String], key: String, fallback: Double
    ) throws -> Double {
        guard let raw = environment[key] else { return fallback }
        guard let value = Double(raw), value.isFinite else {
            throw CLIError.invalidNumber(key: key, value: raw)
        }
        return value
    }

    private static let usage = """
        usage: abslayer-som-search BF16_MODEL PROMPTS_JSON SOM_CANDIDATES_JSON SOURCE_LAYER_ZERO_BASED APPLICATION_SCOPE MAX_CASES OUTPUT_JSON

        APPLICATION_SCOPE is `global` (source basis on all decoder layers) or
        `local` (source basis only on SOURCE_LAYER_ZERO_BASED). The BF16 model is
        loaded once. Ordered, nonorthogonal subsets are authored in memory at
        runtime scale 1. Every candidate is unloaded and checked against an exact
        full-vocabulary untouched sentinel before the next candidate. Results are
        first-token screening proxies, not semantic-compliance verdicts.

        Optional environment:
          ABSLAYER_SOM_BEAM_WIDTH=4
          ABSLAYER_SOM_MAX_DEPTH=7
          ABSLAYER_SOM_MAX_FINALISTS=8
          ABSLAYER_SOM_CONTRAST_KL_CEILING=1.0
          ABSLAYER_SOM_CONTROL_KL_CEILING=1.0
          ABSLAYER_SOM_COMPONENTS=omlp          # omlp, attention, or mlp
          ABSLAYER_SOM_FINALIST_DIR=/path       # persist bounded cross-depth finalists
          ABSLAYER_SOM_CONTROL_KL_PENALTY=1.0
          ABSLAYER_SOM_CONTROL_STARTER_PENALTY=0.5
          ABSLAYER_SOM_CONTRAST_KL_PENALTY=0.25
          ABSLAYER_REFUSAL_STARTERS_JSON='["I","Sorry","Unfortunately","As"]'
          ABSLAYER_COMPLIANCE_STARTERS_JSON='["Sure","Certainly","Here","Below"]'
          ABSLAYER_UNLOAD_TOLERANCE=1e-5
        """
}

private enum CLIError: LocalizedError {
    case invalidJSON(String)
    case invalidNumber(key: String, value: String)
    case invalidEnum(key: String, value: String)

    var errorDescription: String? {
        switch self {
        case .invalidJSON(let key):
            "\(key) must be a non-empty JSON array of strings."
        case .invalidNumber(let key, let value):
            "\(key) has invalid numeric value '\(value)'."
        case .invalidEnum(let key, let value):
            "\(key) has invalid value '\(value)'."
        }
    }
}
