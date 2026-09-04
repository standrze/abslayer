#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerCLI {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 4 || arguments.count == 5 else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-cli MODEL_FOLDER PROMPTS_JSON REPORT_PATH [OUTPUT_MODEL_FOLDER]\n".utf8))
            exit(2)
        }
        let model = try ModelFolderValidator.validateFullBF16(path: arguments[1])
        let allPairs = try PromptFile.load(arguments[2])
        let requestedNames = ProcessInfo.processInfo.environment["ABSLAYER_PAIR_NAMES"]
            .map { Set($0.split(separator: ",").map(String.init)) }
        let eligiblePairs: [PromptPair]
        if let requestedNames {
            eligiblePairs = allPairs.filter { requestedNames.contains($0.name) }
            let missing = requestedNames.subtracting(eligiblePairs.map(\.name))
            guard missing.isEmpty else {
                FileHandle.standardError.write(Data(
                    "unknown pair name(s): \(missing.sorted().joined(separator: ", "))\n".utf8))
                exit(2)
            }
        } else {
            eligiblePairs = allPairs
        }
        let requested = ProcessInfo.processInfo.environment["ABSLAYER_MEASURE_CASES"]
            .flatMap(Int.init) ?? eligiblePairs.count
        let pairs: [PromptPair]
        if requested > 0, requested < eligiblePairs.count {
            pairs = (0 ..< requested).map {
                eligiblePairs[$0 * eligiblePairs.count / requested]
            }
        } else {
            pairs = eligiblePairs
        }
        print("Loading \(model.path); measuring \(pairs.count) prompt pair(s)…")
        let rank = ProcessInfo.processInfo.environment["ABSLAYER_RANK"].flatMap(Int.init) ?? 4
        let tokenPosition = ProcessInfo.processInfo.environment["ABSLAYER_TOKEN_POSITION"]
            .flatMap(ActivationTokenPosition.init(rawValue:)) ?? .postInstruction
        let generateResponses = ProcessInfo.processInfo.environment[
            "ABSLAYER_GENERATE_RESPONSES"] != "0"
        print("Activation position: \(tokenPosition.rawValue)")
        let report = try await ProbeEngine(
            modelDirectory: model.path, pairs: pairs, subspaceRank: rank,
            generateResponses: generateResponses,
            tokenPosition: tokenPosition).run()
        try report.rendered.write(toFile: arguments[3], atomically: true, encoding: .utf8)
        print(report.rendered)
        print("Saved \(arguments[3])")
        if arguments.count == 5 {
            let environment = ProcessInfo.processInfo.environment
            let explicitSource = try optionalInteger(
                environment, "ABSLAYER_SOURCE_LAYER")
            let explicitApplication = try optionalInteger(
                environment, "ABSLAYER_APPLICATION_LAYER")
            let legacyPeak = try optionalInteger(
                environment, "ABSLAYER_PEAK_LAYER")
            guard explicitSource == nil || legacyPeak == nil else {
                throw ABSlayerCLIError.conflictingSourceLayerSettings
            }
            if legacyPeak != nil {
                FileHandle.standardError.write(Data(
                    "WARNING: ABSLAYER_PEAK_LAYER is a legacy one-based setting; prefer zero-based ABSLAYER_SOURCE_LAYER.\n".utf8))
            }
            let automaticSource = (report.layers.max(by: {
                    $0.cosineDistance * max(0, $0.directionAgreement)
                        * max(0, $0.medianDirectionAgreement) * max(0, $0.silhouette)
                        < $1.cosineDistance * max(0, $1.directionAgreement)
                        * max(0, $1.medianDirectionAgreement) * max(0, $1.silhouette)
                })?.layer).map { $0 - 1 }
                ?? max(0, report.layers.count / 2 - 1)
            let sourceLayer = explicitSource ?? legacyPeak.map { $0 - 1 }
                ?? automaticSource
            _ = try DecoderLayerSelection.validateZeroBased(
                [sourceLayer], layerCount: report.directions.count)
            let applicationLayer = explicitApplication ?? sourceLayer
            _ = try DecoderLayerSelection.validateZeroBased(
                [applicationLayer], layerCount: report.directions.count)
            func setting(_ name: String, _ fallback: Float) -> Float {
                environment[name].flatMap(Float.init) ?? fallback
            }
            print(
                "Editing BF16 weights from zero-based source layer \(sourceLayer) "
                    + "around application layer \(applicationLayer)…")
            let summary = try BF16WeightEditor.edit(
                sourcePath: model.path,
                outputPath: arguments[4],
                directions: report.directions,
                subspaces: report.subspaces,
                configuration: AbliterationConfiguration(
                    attention: LayerAblationKernel(
                        maximum: setting("ABSLAYER_ATTN_MAX", 1.0),
                        peakLayer: Float(applicationLayer),
                        minimum: setting("ABSLAYER_ATTN_MIN", 0.2),
                        radius: setting("ABSLAYER_ATTN_RADIUS", 10)),
                    mlp: LayerAblationKernel(
                        maximum: setting("ABSLAYER_MLP_MAX", 0.15),
                        peakLayer: Float(applicationLayer),
                        minimum: setting("ABSLAYER_MLP_MIN", 0.03),
                        radius: setting("ABSLAYER_MLP_RADIUS", 6)),
                    directionScope: environment["ABSLAYER_GLOBAL_LAYER"].flatMap(Float.init)
                        .map(DirectionScope.global(layer:)) ?? .perLayer,
                    normalization: .full
                ))
            print("Edited \(summary.editedAttentionMatrices) attention and "
                + "\(summary.editedMLPMatrices) MLP matrices -> \(summary.outputPath)")
        }
    }

    private static func optionalInteger(
        _ environment: [String: String], _ key: String
    ) throws -> Int? {
        guard let raw = environment[key] else { return nil }
        guard let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw ABSlayerCLIError.invalidInteger(key: key, value: raw)
        }
        return value
    }
}

private enum ABSlayerCLIError: LocalizedError {
    case conflictingSourceLayerSettings
    case invalidInteger(key: String, value: String)

    var errorDescription: String? {
        switch self {
        case .conflictingSourceLayerSettings:
            "Set only ABSLAYER_SOURCE_LAYER (zero-based); do not combine it with legacy ABSLAYER_PEAK_LAYER (one-based)."
        case .invalidInteger(let key, let value):
            "\(key) must be an integer, not '\(value)'."
        }
    }
}
