#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerSOMMultiSourceSearch {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count >= 7,
              let maximumCases = Int(arguments[3]), maximumCases > 0
        else {
            FileHandle.standardError.write(Data(usage.utf8))
            exit(2)
        }
        let archiveFiles = Array(arguments[5...])
        let environment = ProcessInfo.processInfo.environment
        let output = URL(fileURLWithPath: arguments[4]).standardizedFileURL
        let finalistDirectory = environment["ABSLAYER_SOM_FINALIST_DIR"]
            ?? output.deletingPathExtension().path + "-finalists"

        let configuration: SOMMultiSourceSearchConfiguration
        let starterConfiguration: FirstTokenStarterConfiguration
        do {
            guard let components = SOMApplicationComponents(
                rawValue: environment["ABSLAYER_SOM_COMPONENTS"] ?? "omlp")
            else {
                throw CLIError.invalidEnum(
                    key: "ABSLAYER_SOM_COMPONENTS",
                    value: environment["ABSLAYER_SOM_COMPONENTS"] ?? "")
            }
            configuration = SOMMultiSourceSearchConfiguration(
                components: components,
                maximumCases: maximumCases,
                preselectionPerLayer: try integer(
                    environment, key: "ABSLAYER_SOM_PRESELECT_PER_LAYER",
                    fallback: 4),
                maximumDepth: try integer(
                    environment, key: "ABSLAYER_SOM_MAX_DEPTH", fallback: 7),
                maximumDirectionsPerLayer: try integer(
                    environment, key: "ABSLAYER_SOM_MAX_PER_LAYER", fallback: 3),
                beamWidth: try integer(
                    environment, key: "ABSLAYER_SOM_BEAM_WIDTH",
                    fallback: max(4, archiveFiles.count)),
                minimumSourceLayersPerFinalist: try integer(
                    environment, key: "ABSLAYER_SOM_MIN_SOURCE_LAYERS",
                    fallback: archiveFiles.count),
                maximumFinalists: try integer(
                    environment, key: "ABSLAYER_SOM_MAX_FINALISTS", fallback: 8),
                contrastKLCeiling: try double(
                    environment, key: "ABSLAYER_SOM_CONTRAST_KL_CEILING",
                    fallback: 0.5),
                controlKLCeiling: try double(
                    environment, key: "ABSLAYER_SOM_CONTROL_KL_CEILING",
                    fallback: 0.5),
                unloadTolerance: try double(
                    environment, key: "ABSLAYER_UNLOAD_TOLERANCE",
                    fallback: 1e-5),
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
        } catch {
            FileHandle.standardError.write(
                Data("Error: \(error.localizedDescription)\n".utf8))
            exit(2)
        }

        let model = try ModelFolderValidator.validateFullBF16(path: arguments[1])
        let pairs = try PromptFile.load(arguments[2])
        print(FirstTokenScreenEngine.proxyNotice)
        print("Layer-specific search; source archives are never broadcast globally.")
        print("Hard exact KL gates: contrast <= \(configuration.contrastKLCeiling), control <= \(configuration.controlKLCeiling)")
        let report = try await SOMMultiSourceSearchEngine.run(
            modelDirectory: model.path,
            promptFile: arguments[2],
            pairs: pairs,
            candidateArchiveFiles: archiveFiles,
            configuration: configuration,
            starterConfiguration: starterConfiguration,
            finalistDirectory: finalistDirectory,
            progress: { print($0) })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(report).write(to: output, options: .atomic)

        print("Searched zero-based source layers \(report.sourceLayersZeroBased); runtime scale=1")
        print("Screened \(report.screenedCandidateCount)/<=\(report.theoreticalCandidateBudget) bounded candidates")
        for finalist in report.selectedFinalists {
            print(
                "Finalist \(finalist.plan.stableKey) "
                    + String(
                        format: "score=%+.5f contrastKL=%.5f controlKL=%.5f",
                        finalist.objectiveScore,
                        finalist.rankingMetrics.contrastExactMeanKL,
                        finalist.rankingMetrics.controlExactMeanKL))
        }
        print("Materialized \(report.materializedFinalists.count) adapters for semantic generation -> \(finalistDirectory)")
        print("Saved machine-readable multi-source search -> \(output.path)")
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
        guard let value = Int(raw) else {
            throw CLIError.invalidNumber(key: key, value: raw)
        }
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
        usage: abslayer-som-multisource-search BF16_MODEL DEV_PROMPTS_JSON MAX_CASES OUTPUT_JSON SOM_ARCHIVE [SOM_ARCHIVE ...]

        Supply one archive per explicitly selected zero-based source layer.
        Archive inputs are the source-layer shortlist: omitting an archive is
        what excludes that layer. Each measured basis edits only its own layer's
        O/down projections. The model is loaded once, every
        candidate runs at adapter scale 1, both exact full-vocabulary first-token
        KL gates default to 0.5, and unload restoration is checked after every
        trial. Finalists are materialized for full semantic generation. Do not
        use frozen primary/confirmation prompts during search.

        Optional environment:
          ABSLAYER_SOM_PRESELECT_PER_LAYER=4
          ABSLAYER_SOM_BEAM_WIDTH=4
          ABSLAYER_SOM_MAX_DEPTH=7
          ABSLAYER_SOM_MAX_PER_LAYER=3
          ABSLAYER_SOM_MIN_SOURCE_LAYERS=3
          ABSLAYER_SOM_MAX_FINALISTS=8
          ABSLAYER_SOM_CONTRAST_KL_CEILING=0.5  # cannot exceed 0.5
          ABSLAYER_SOM_CONTROL_KL_CEILING=0.5   # cannot exceed 0.5
          ABSLAYER_SOM_COMPONENTS=omlp          # omlp, attention, or mlp
          ABSLAYER_SOM_FINALIST_DIR=/path
          ABSLAYER_SOM_CONTROL_KL_PENALTY=1.0
          ABSLAYER_SOM_CONTROL_STARTER_PENALTY=0.5
          ABSLAYER_SOM_CONTRAST_KL_PENALTY=0.25
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
