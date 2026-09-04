#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import MLX
import ProbeCore

@main
enum ABSlayerARA {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 4 || arguments.count == 5 else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-ara BF16_MODEL MEASUREMENT_PAIRS OUTPUT_MODEL [REPORT_JSON]\n".utf8))
            exit(2)
        }
        let environment = ProcessInfo.processInfo.environment
        let model = try ModelFolderValidator.validateFullBF16(path: arguments[1])
        guard let decoderLayerCount = model.decoderLayerCount else {
            throw ARACommandError.missingDecoderLayerCount
        }
        let allPairs = try PromptFile.load(arguments[2])
        let requestedCases = integer(environment, "ABSLAYER_ARA_MEASURE_CASES", 96)
        let pairs = evenlySpaced(allPairs, maximum: requestedCases)
        let parameters = try parameters(
            from: environment, decoderLayerCount: decoderLayerCount)
        let layers = try selectedLayers(
            from: environment, parameters: parameters,
            decoderLayerCount: decoderLayerCount)
        let maximumIterations = integer(environment, "ABSLAYER_ARA_ITERATIONS", 100)
        let historySize = integer(environment, "ABSLAYER_ARA_HISTORY", 10)
        let lineSearchIterations = integer(environment, "ABSLAYER_ARA_LINE_SEARCH", 24)
        let dryRun = environment["ABSLAYER_ARA_DRY_RUN"] == "1"
        let maximumRelativeChange = environment["ABSLAYER_ARA_MAX_RELATIVE_CHANGE"]
            .flatMap(Float.init)

        print("Loading \(model.path)")
        print("Capturing \(pairs.count) cyber/control projection-I/O pairs")
        if environment["ABSLAYER_ARA_LAYERS"] == nil {
            print("ARA layers are zero-based and half-open: \(parameters.startLayerIndex)..<\(parameters.endLayerIndex)")
        } else {
            print("ARA selected zero-based layers: \(layers.map(String.init).joined(separator: ","))")
        }
        print(String(
            format: "ARA preserve=%.6f steer=%.6f overcorrect=%.6f k=%d normalization=%@",
            parameters.preserveGoodBehaviorWeight,
            parameters.steerBadBehaviorWeight,
            parameters.overcorrectRelativeWeight,
            parameters.neighborCount,
            parameters.rowNormalization.rawValue))

        let collection = try await ARAProjectionIOCollectionEngine(
            modelDirectory: model.path, pairs: pairs, layers: layers
        ).run { completed, total in
            print("captured \(completed)/\(total)")
        }
        let keys = Set(layers.map(BF16WeightEditor.attentionOutputProjectionKey(layer:)))
        let originals = try BF16WeightEditor.loadMatrices(
            sourcePath: model.path, keys: keys)
        var replacements = [String: MLXArray]()
        var records = [LayerRecord]()

        for capture in collection.layers {
            let key = BF16WeightEditor.attentionOutputProjectionKey(layer: capture.layer)
            guard let original = originals[key] else {
                throw ARACommandError.missingWeight(key)
            }
            let samples = capture.moduleSamples()
            let initialLoss = ArbitraryRankAblation.objective(
                weight: original, samples: samples, parameters: parameters)
            eval(initialLoss)
            print(String(
                format: "optimizing layer %d: initial loss %.8f",
                capture.layer, initialLoss.item(Float.self)))
            let result = ArbitraryRankAblation.optimize(
                originalWeight: original,
                samples: samples,
                parameters: parameters,
                maximumIterations: maximumIterations,
                historySize: historySize,
                maximumLineSearchIterations: lineSearchIterations,
                maximumRelativeChange: maximumRelativeChange)
            let originalFP32 = original.asType(.float32)
            let rawDelta = result.weight.asType(.float32) - originalFP32
            let originalNorm = maximum(
                sqrt((originalFP32 * originalFP32).sum()), MLXArray(1e-12 as Float))
            let rawRelativeChange = sqrt((rawDelta * rawDelta).sum()) / originalNorm
            eval(rawRelativeChange)
            let rawRelative = rawRelativeChange.item(Float.self)
            let edited = result.weight
            let finalDelta = edited.asType(.float32) - originalFP32
            let relativeChange = sqrt((finalDelta * finalDelta).sum()) / originalNorm
            let finalLoss = ArbitraryRankAblation.objective(
                weight: edited, samples: samples, parameters: parameters)
            eval(finalLoss, relativeChange, edited)
            let record = LayerRecord(
                layer: capture.layer,
                tensorKey: key,
                initialLoss: Double(initialLoss.item(Float.self)),
                finalLoss: Double(finalLoss.item(Float.self)),
                uncappedRelativeFrobeniusChange: Double(rawRelative),
                relativeFrobeniusChange: Double(relativeChange.item(Float.self)),
                steps: result.steps)
            records.append(record)
            replacements[key] = edited
            print(String(
                format: "layer %d: final loss %.8f, relative change %.5f, iterations %d",
                capture.layer, record.finalLoss,
                record.relativeFrobeniusChange, record.steps.count))
        }

        let outputPath: String
        let editedCount: Int
        if dryRun {
            outputPath = URL(fileURLWithPath: arguments[3]).standardizedFileURL.path
            editedCount = replacements.count
            print("Dry run: skipped BF16 checkpoint export")
        } else {
            let summary = try BF16WeightEditor.replaceMatrices(
                sourcePath: model.path,
                outputPath: arguments[3],
                replacements: replacements)
            outputPath = summary.outputPath
            editedCount = summary.editedAttentionMatrices
        }
        let report = ARAReport(
            schemaVersion: 2,
            sourceModel: model.path,
            outputModel: outputPath,
            measurementFile: URL(fileURLWithPath: arguments[2]).standardizedFileURL.path,
            pairNames: collection.pairNames,
            parameters: parameters,
            selectedLayers: layers,
            maximumIterations: maximumIterations,
            historySize: historySize,
            maximumRelativeChange: maximumRelativeChange,
            layers: records)
        let reportPath = arguments.count == 5
            ? arguments[4]
            : arguments[3] + "-ara-report.json"
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(
            to: URL(fileURLWithPath: reportPath).standardizedFileURL,
            options: .atomic)
        print("Prepared \(editedCount) ARA matrices -> \(outputPath)")
        print("Saved \(reportPath)")
    }

    private static func parameters(
        from environment: [String: String], decoderLayerCount: Int
    ) throws -> ARAParameters {
        let seed = ARAParameters.fullStackDefault(layerCount: decoderLayerCount)
        let start = integer(
            environment, "ABSLAYER_ARA_START_LAYER", seed.startLayerIndex)
        let end = integer(
            environment, "ABSLAYER_ARA_END_LAYER", seed.endLayerIndex)
        guard start >= 0, end > start, end <= decoderLayerCount else {
            throw ARACommandError.invalidLayerRange(
                start: start, end: end, available: decoderLayerCount)
        }
        return ARAParameters(
            startLayerIndex: start,
            endLayerIndex: end,
            preserveGoodBehaviorWeight: float(
                environment, "ABSLAYER_ARA_PRESERVE", seed.preserveGoodBehaviorWeight),
            steerBadBehaviorWeight: float(
                environment, "ABSLAYER_ARA_STEER", seed.steerBadBehaviorWeight),
            overcorrectRelativeWeight: float(
                environment, "ABSLAYER_ARA_OVERCORRECT", seed.overcorrectRelativeWeight),
            neighborCount: integer(
                environment, "ABSLAYER_ARA_NEIGHBORS", seed.neighborCount),
            rowNormalization: environment["ABSLAYER_ARA_ROW_NORMALIZATION"]
                .flatMap(ARARowNormalization.init(rawValue:)) ?? seed.rowNormalization)
    }

    /// Optional explicit layer list for sparse/synergy searches.  The range
    /// variables remain the default and retain their half-open semantics.
    private static func selectedLayers(
        from environment: [String: String], parameters: ARAParameters,
        decoderLayerCount: Int
    ) throws -> [Int] {
        guard let raw = environment["ABSLAYER_ARA_LAYERS"] else {
            return Array(parameters.startLayerIndex ..< parameters.endLayerIndex)
        }
        let pieces = raw.split(separator: ",", omittingEmptySubsequences: false)
        let parsed = pieces.compactMap { piece in
            Int(piece.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard !pieces.isEmpty, parsed.count == pieces.count,
              Set(parsed).count == parsed.count,
              parsed.allSatisfy({ $0 >= 0 && $0 < decoderLayerCount })
        else {
            throw ARACommandError.invalidLayerList(raw)
        }
        return Array(Set(parsed)).sorted()
    }

    private static func evenlySpaced<T>(_ values: [T], maximum: Int) -> [T] {
        guard maximum > 0, values.count > maximum else { return values }
        return (0 ..< maximum).map { values[$0 * values.count / maximum] }
    }

    private static func integer(
        _ environment: [String: String], _ name: String, _ fallback: Int
    ) -> Int {
        environment[name].flatMap(Int.init) ?? fallback
    }

    private static func float(
        _ environment: [String: String], _ name: String, _ fallback: Float
    ) -> Float {
        environment[name].flatMap(Float.init) ?? fallback
    }
}

private struct LayerRecord: Codable {
    let layer: Int
    let tensorKey: String
    let initialLoss: Double
    let finalLoss: Double
    let uncappedRelativeFrobeniusChange: Double
    let relativeFrobeniusChange: Double
    let steps: [ARAOptimizationStep]
}

private struct ARAReport: Codable {
    let schemaVersion: Int
    let sourceModel: String
    let outputModel: String
    let measurementFile: String
    let pairNames: [String]
    let parameters: ARAParameters
    let selectedLayers: [Int]
    let maximumIterations: Int
    let historySize: Int
    let maximumRelativeChange: Float?
    let layers: [LayerRecord]
}

private enum ARACommandError: LocalizedError {
    case missingWeight(String)
    case missingDecoderLayerCount
    case invalidLayerList(String)
    case invalidLayerRange(start: Int, end: Int, available: Int)

    var errorDescription: String? {
        switch self {
        case .missingWeight(let key): "The source model did not load \(key)."
        case .missingDecoderLayerCount:
            "The model config does not report a positive text decoder layer count."
        case .invalidLayerList(let value):
            "ABSLAYER_ARA_LAYERS must be a comma-separated list of unique, in-range zero-based decoder layers, not '\(value)'."
        case .invalidLayerRange(let start, let end, let available):
            "ARA's zero-based half-open layer range \(start)..<\(end) is outside the model's 0..<\(available) decoder layers."
        }
    }
}
