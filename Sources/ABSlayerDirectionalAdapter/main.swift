#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerDirectionalAdapter {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 4 else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-directional-adapter BF16_MODEL MEASUREMENT_PAIRS ADAPTER_DIR\n".utf8))
            exit(2)
        }
        let environment = ProcessInfo.processInfo.environment
        let model = try ModelFolderValidator.validateFullBF16(path: arguments[1])
        let allPairs = try PromptFile.load(arguments[2])
        let requested = integer(environment, "ABSLAYER_MEASURE_CASES", 96)
        let pairs = evenlySpaced(allPairs, maximum: requested)
        let rank = integer(environment, "ABSLAYER_RANK", 1)
        guard rank > 0, !pairs.isEmpty else { throw DirectionalAdapterError.invalidSetting }
        let methodName = environment["ABSLAYER_DIRECTION_METHOD"] ?? "centroidDifference"
        guard let method = DirectionExtractionMethod(rawValue: methodName) else {
            throw DirectionalAdapterError.invalidDirectionMethod(methodName)
        }
        let normalizationName = environment["ABSLAYER_ROW_NORMALIZATION"] ?? "none"
        guard let normalization = WeightNormalization(rawValue: normalizationName) else {
            throw DirectionalAdapterError.invalidNormalization(normalizationName)
        }
        let compositionName = environment["ABSLAYER_COMPOSITION"] ?? "simultaneous"
        guard let composition = AblationComposition(rawValue: compositionName) else {
            throw DirectionalAdapterError.invalidComposition(compositionName)
        }
        if method == .som, composition != .sequential {
            throw DirectionalAdapterError.somRequiresSequentialComposition
        }
        let tokenName = environment["ABSLAYER_TOKEN_POSITION"] ?? "post-instruction"
        guard let tokenPosition = ActivationTokenPosition(rawValue: tokenName) else {
            throw DirectionalAdapterError.invalidTokenPosition(tokenName)
        }
        let explicitSourceZeroBased = try optionalInteger(
            environment, "ABSLAYER_SOURCE_LAYER")
        let explicitApplicationZeroBased = try optionalInteger(
            environment, "ABSLAYER_APPLICATION_LAYER")
        let legacyPeakOneBased = try optionalInteger(
            environment, "ABSLAYER_PEAK_LAYER")
        guard explicitSourceZeroBased == nil || legacyPeakOneBased == nil else {
            throw DirectionalAdapterError.conflictingSourceLayerSettings
        }
        if let explicitSourceZeroBased, explicitSourceZeroBased < 0 {
            throw DirectionalAdapterError.invalidSourceLayer(explicitSourceZeroBased)
        }
        if let explicitApplicationZeroBased, explicitApplicationZeroBased < 0 {
            throw DirectionalAdapterError.invalidApplicationLayer(
                requested: explicitApplicationZeroBased, available: 0)
        }
        if legacyPeakOneBased != nil {
            FileHandle.standardError.write(Data(
                "WARNING: ABSLAYER_PEAK_LAYER is a legacy one-based setting; prefer zero-based ABSLAYER_SOURCE_LAYER.\n".utf8))
        }
        let explicitPeakOneBased = explicitSourceZeroBased.map { $0 + 1 }
            ?? legacyPeakOneBased
        let explicitExtractionLayers = try optionalIntegerSet(
            environment, "ABSLAYER_EXTRACTION_LAYERS")
        let extractionLayers: Set<Int>?
        if method == .som {
            if let explicitExtractionLayers {
                extractionLayers = explicitExtractionLayers
            } else if let explicitPeakOneBased {
                guard explicitPeakOneBased > 0 else {
                    throw DirectionalAdapterError.invalidPeak(
                        requested: explicitPeakOneBased, available: 0)
                }
                extractionLayers = [explicitPeakOneBased - 1]
            } else {
                // A source layer is model- and dataset-specific. With no
                // shortlist, train every decoder layer and localize from the
                // measured scores instead of inheriting an experiment window.
                extractionLayers = nil
            }
            if let explicitPeakOneBased,
               !(extractionLayers?.contains(explicitPeakOneBased - 1) ?? false)
            {
                throw DirectionalAdapterError.peakNotExtracted(
                    peak: explicitPeakOneBased - 1,
                    extracted: extractionLayers?.sorted() ?? [])
            }
        } else {
            extractionLayers = explicitExtractionLayers
        }

        print("Measuring \(pairs.count) unmatched contrast pairs on \(model.path)")
        print(
            "Direction method=\(method.rawValue) rank=\(rank) "
                + "position=\(tokenPosition.rawValue)")
        if let extractionLayers {
            print("Expensive extraction restricted to zero-based layers \(extractionLayers.sorted())")
        } else if method == .som {
            print("SOM extraction scans the full zero-based decoder stack; set ABSLAYER_EXTRACTION_LAYERS to use an explicit compute shortlist")
        }
        let report = try await ProbeEngine(
            modelDirectory: model.path, pairs: pairs, subspaceRank: rank,
            extractionMethod: method, generateResponses: false,
            tokenPosition: tokenPosition,
            extractionLayers: extractionLayers).run()
        let localizationLayers: [LayerScore]
        if method == .som, let extractionLayers {
            localizationLayers = report.layers.filter {
                extractionLayers.contains($0.layer - 1)
            }
        } else {
            localizationLayers = report.layers
        }
        let automaticPeak = localizationLayers.max { lhs, rhs in
            score(lhs) < score(rhs)
        }?.layer ?? max(1, report.layers.count / 2)
        let peakOneBased = explicitPeakOneBased ?? automaticPeak
        guard (1 ... report.layers.count).contains(peakOneBased) else {
            throw DirectionalAdapterError.invalidPeak(
                requested: peakOneBased, available: report.layers.count)
        }
        let peak = peakOneBased - 1
        let applicationPeak = explicitApplicationZeroBased ?? peak
        guard report.layers.indices.contains(applicationPeak) else {
            throw DirectionalAdapterError.invalidApplicationLayer(
                requested: applicationPeak, available: report.layers.count)
        }
        let scopeName = environment["ABSLAYER_DIRECTION_SCOPE"] ?? "perLayer"
        let directionScope: DirectionScope
        switch scopeName {
        case "perLayer": directionScope = .perLayer
        case "global": directionScope = .global(layer: Float(peak))
        default: throw DirectionalAdapterError.invalidDirectionScope(scopeName)
        }
        let variants: [DirectionalVariant]
        let somDiagnosticsData: Data?
        let somCandidateArchiveFiles: [String: Data]
        if method == .som {
            guard let result = report.somResultsByLayer[peak] else {
                throw DirectionalAdapterError.missingSOMResult(
                    peak: peak, trained: report.somResultsByLayer.keys.sorted())
            }
            let selections = try SOMCandidateSelector.resolve(
                result: result, rank: rank,
                specification: environment[SOMCandidateSelector.environmentVariable])
            variants = selections.map { selection in
                var subspaces = report.subspaces
                var directions = report.directions
                subspaces[peak] = selection.directions
                directions[peak] = selection.directions[0]
                return DirectionalVariant(
                    label: selection.label,
                    latticeIDs: selection.latticeIDs,
                    directions: directions,
                    subspaces: subspaces)
            }
            let diagnostics = try SOMSelectionDiagnostics.project(
                result: result, selections: selections)
            let diagnosticsEncoder = JSONEncoder()
            diagnosticsEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            somDiagnosticsData = try diagnosticsEncoder.encode(diagnostics)
            somCandidateArchiveFiles = try SOMCandidateArchiveBundle.encode(
                resultsByLayer: report.somResultsByLayer,
                selectedSourceLayerZeroBased: peak,
                trainingMode: SOMTrainingMode.officialMiniSom235.rawValue)
        } else {
            variants = [DirectionalVariant(
                label: nil, latticeIDs: nil,
                directions: report.directions, subspaces: report.subspaces)]
            somDiagnosticsData = nil
            somCandidateArchiveFiles = [:]
        }
        let plannedOutputs = try variants.flatMap { variant -> [PlannedOutput] in
            let basePath = variants.count > 1
                ? suffixedPath(arguments[3], suffix: variant.label!)
                : arguments[3]
            return try DirectionalAdapterStrengthPlan.parse(
                environment: environment, baseOutputPath: basePath
            ).map { PlannedOutput(variant: variant, strength: $0) }
        }
        guard Set(plannedOutputs.map(\.strength.outputPath)).count == plannedOutputs.count else {
            throw DirectionalAdapterError.outputCollision
        }
        for output in plannedOutputs
            where FileManager.default.fileExists(atPath: output.strength.outputPath)
        {
            throw LoRAAdapterPersistenceError.outputExists(output.strength.outputPath)
        }
        let attentionMaximum = float(environment, "ABSLAYER_ATTN_MAX", 1)
        let attentionMinimum = float(environment, "ABSLAYER_ATTN_MIN", 0)
        let attentionRadius = float(environment, "ABSLAYER_ATTN_RADIUS", 0)
        let mlpMaximum = float(environment, "ABSLAYER_MLP_MAX", 0)
        let mlpMinimum = float(environment, "ABSLAYER_MLP_MIN", 0)
        let mlpRadius = float(environment, "ABSLAYER_MLP_RADIUS", 0)
        guard attentionMaximum.isFinite, attentionMinimum.isFinite,
              attentionRadius.isFinite, mlpMaximum.isFinite,
              mlpMinimum.isFinite, mlpRadius.isFinite,
              attentionMaximum >= 0, attentionMinimum >= 0, attentionRadius >= 0,
              mlpMaximum >= 0, mlpMinimum >= 0, mlpRadius >= 0,
              attentionMaximum > 0 || mlpMaximum > 0
        else { throw DirectionalAdapterError.invalidSetting }
        guard plannedOutputs.allSatisfy({ output in
            (attentionMaximum * output.strength.multiplier).isFinite
                && (attentionMinimum * output.strength.multiplier).isFinite
                && (mlpMaximum * output.strength.multiplier).isFinite
                && (mlpMinimum * output.strength.multiplier).isFinite
        }) else { throw DirectionalAdapterError.invalidSetting }

        let runtimeScaleWarning: String?
        if composition == .sequential {
            let warning =
                "Sequential nonorthogonal projection is nonlinear in authored strength; "
                + "evaluate this adapter with ABSLAYER_ADAPTER_SCALE=1. "
                + "Use ABSLAYER_STRENGTHS to author other strengths."
            runtimeScaleWarning = warning
            FileHandle.standardError.write(Data("WARNING: \(warning)\n".utf8))
        } else {
            runtimeScaleWarning = nil
        }

        // Measurement happens once and the adapter-building runtime is loaded
        // once; only the cheap static low-rank factors differ between outputs.
        let runtime = try await ResidentTrialRuntime(modelDirectory: model.path)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        for output in plannedOutputs {
            let variant = output.variant
            let plan = output.strength
            let authoredAttentionMaximum = attentionMaximum * plan.multiplier
            let authoredAttentionMinimum = attentionMinimum * plan.multiplier
            let authoredMLPMaximum = mlpMaximum * plan.multiplier
            let authoredMLPMinimum = mlpMinimum * plan.multiplier
            let configuration = AbliterationConfiguration(
                attention: LayerAblationKernel(
                    maximum: authoredAttentionMaximum,
                    peakLayer: Float(applicationPeak),
                    minimum: authoredAttentionMinimum, radius: attentionRadius),
                mlp: LayerAblationKernel(
                    maximum: authoredMLPMaximum,
                    peakLayer: Float(applicationPeak),
                    minimum: authoredMLPMinimum, radius: mlpRadius),
                directionScope: directionScope,
                normalization: normalization,
                composition: composition)
            print(
                "Building zero-based source layer \(peak), application peak "
                    + "\(applicationPeak), authored multiplier="
                    + "\(plan.multiplier), attention=\(authoredAttentionMaximum), MLP="
                    + "\(authoredMLPMaximum), normalization=\(normalization.rawValue), "
                    + "composition=\(composition.rawValue), scope=\(scopeName)")
            let adapter = try await runtime.makeAdapter(
                directions: variant.directions, subspaces: variant.subspaces,
                configuration: configuration)

            let manifest = DirectionalAdapterManifest(
                schemaVersion: 7,
                sourceModel: model.path,
                measurementFile: URL(fileURLWithPath: arguments[2]).standardizedFileURL.path,
                pairNames: pairs.map(\.name),
                rank: rank,
                adapterRank: adapter.configuration.loraParameters.rank,
                adapterScale: adapter.configuration.loraParameters.scale,
                adapterNumLayers: adapter.configuration.numLayers,
                adapterKeys: adapter.configuration.loraParameters.keys ?? [],
                activeTensorNames: adapter.parameters.flattened().map(\.0).sorted(),
                directionMethod: method,
                tokenPosition: tokenPosition,
                peakLayerZeroBased: peak,
                sourceLayerZeroBased: peak,
                applicationPeakLayerZeroBased: applicationPeak,
                extractionLayersZeroBased: extractionLayers?.sorted(),
                directionScope: scopeName,
                somTrainingMode: method == .som
                    ? SOMTrainingMode.officialMiniSom235.rawValue : nil,
                somSelectionLabel: variant.label,
                somSelectedLatticeIDs: variant.latticeIDs,
                authoredStrengthMultiplier: plan.multiplier,
                attentionMaximum: authoredAttentionMaximum,
                attentionMinimum: authoredAttentionMinimum,
                attentionRadius: attentionRadius,
                mlpMaximum: authoredMLPMaximum,
                mlpMinimum: authoredMLPMinimum,
                mlpRadius: mlpRadius,
                normalization: normalization,
                composition: composition,
                requiredRuntimeAdapterScale: composition == .sequential ? 1 : nil,
                runtimeScaleWarning: runtimeScaleWarning)
            let manifestData = try encoder.encode(manifest)
            var additionalFiles = ["abslayer_manifest.json": manifestData]
            if let somDiagnosticsData {
                additionalFiles["abslayer_som_diagnostics.json"] = somDiagnosticsData
            }
            for (filename, data) in somCandidateArchiveFiles {
                additionalFiles[filename] = data
            }
            try LoRAAdapterPersistence.write(
                adapter, to: plan.outputPath,
                additionalFiles: additionalFiles)
            print("Saved static directional adapter -> \(plan.outputPath)")
        }
    }

    private static func score(_ layer: LayerScore) -> Double {
        layer.cosineDistance
            * max(0, layer.directionAgreement)
            * max(0, layer.medianDirectionAgreement)
            * max(0, layer.silhouette)
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

    private static func optionalInteger(
        _ environment: [String: String], _ name: String
    ) throws -> Int? {
        guard let raw = environment[name] else { return nil }
        let value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let parsed = Int(value) else {
            throw DirectionalAdapterError.invalidInteger(name: name, value: raw)
        }
        return parsed
    }

    private static func optionalIntegerSet(
        _ environment: [String: String], _ name: String
    ) throws -> Set<Int>? {
        guard let raw = environment[name] else { return nil }
        let tokens = raw.split(separator: ",", omittingEmptySubsequences: false)
        guard !tokens.isEmpty else {
            throw DirectionalAdapterError.invalidIntegerList(name: name, value: raw)
        }
        var result = Set<Int>()
        for token in tokens {
            let text = token.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let value = Int(text), value >= 0, result.insert(value).inserted else {
                throw DirectionalAdapterError.invalidIntegerList(name: name, value: raw)
            }
        }
        return result
    }

    private static func suffixedPath(_ rawPath: String, suffix: String) -> String {
        let base = URL(fileURLWithPath: rawPath).standardizedFileURL
        return base.deletingLastPathComponent()
            .appendingPathComponent(base.lastPathComponent + "-" + suffix, isDirectory: true)
            .path
    }
}

private struct DirectionalVariant {
    let label: String?
    let latticeIDs: [Int]?
    let directions: [[Float]]
    let subspaces: [[[Float]]]
}

private struct PlannedOutput {
    let variant: DirectionalVariant
    let strength: DirectionalAdapterStrengthPlan
}

private struct DirectionalAdapterManifest: Codable {
    let schemaVersion: Int
    let sourceModel: String
    let measurementFile: String
    let pairNames: [String]
    /// Rank requested from direction measurement.
    let rank: Int
    /// Actual persisted LoRA rank (may be larger for full row normalization).
    let adapterRank: Int
    let adapterScale: Float
    let adapterNumLayers: Int
    let adapterKeys: [String]
    let activeTensorNames: [String]
    let directionMethod: DirectionExtractionMethod
    let tokenPosition: ActivationTokenPosition
    /// Legacy schema alias for `sourceLayerZeroBased`.
    let peakLayerZeroBased: Int
    let sourceLayerZeroBased: Int
    let applicationPeakLayerZeroBased: Int
    let extractionLayersZeroBased: [Int]?
    let directionScope: String
    let somTrainingMode: String?
    let somSelectionLabel: String?
    let somSelectedLatticeIDs: [Int]?
    /// Multiplier applied while authoring every active component strength.
    let authoredStrengthMultiplier: Float
    let attentionMaximum: Float
    let attentionMinimum: Float
    let attentionRadius: Float
    let mlpMaximum: Float
    let mlpMinimum: Float
    let mlpRadius: Float
    let normalization: WeightNormalization
    let composition: AblationComposition
    /// Sequential compositions must be evaluated at one; scaling their final
    /// LoRA delta is not the same operation as changing every projector.
    let requiredRuntimeAdapterScale: Float?
    let runtimeScaleWarning: String?
}

private enum DirectionalAdapterError: LocalizedError {
    case invalidSetting
    case invalidDirectionMethod(String)
    case invalidNormalization(String)
    case invalidComposition(String)
    case invalidTokenPosition(String)
    case invalidDirectionScope(String)
    case somRequiresSequentialComposition
    case missingSOMResult(peak: Int, trained: [Int])
    case outputCollision
    case invalidInteger(name: String, value: String)
    case invalidIntegerList(name: String, value: String)
    case conflictingSourceLayerSettings
    case invalidSourceLayer(Int)
    case invalidApplicationLayer(requested: Int, available: Int)
    case peakNotExtracted(peak: Int, extracted: [Int])
    case invalidPeak(requested: Int, available: Int)

    var errorDescription: String? {
        switch self {
        case .invalidSetting: "Directional-adapter settings are invalid."
        case .invalidDirectionMethod(let value):
            "Unknown direction method: \(value)"
        case .invalidNormalization(let value):
            "Unknown row normalization: \(value)"
        case .invalidComposition(let value):
            "Unknown ablation composition: \(value)"
        case .invalidTokenPosition(let value):
            "Unknown activation token position: \(value)"
        case .invalidDirectionScope(let value):
            "Unknown direction scope: \(value). Expected perLayer or global."
        case .somRequiresSequentialComposition:
            "SOM directions are ordered and nonorthogonal; set ABSLAYER_COMPOSITION=sequential."
        case .missingSOMResult(let peak, let trained):
            "No SOM was trained at zero-based peak layer \(peak); trained layers are \(trained)."
        case .outputCollision:
            "SOM selection and strength settings resolve to duplicate output directories."
        case .invalidInteger(let name, let value):
            "\(name) must be an integer, not '\(value)'."
        case .invalidIntegerList(let name, let value):
            "\(name) must be a comma-separated list of unique zero-based non-negative integers, not '\(value)'."
        case .conflictingSourceLayerSettings:
            "Set only ABSLAYER_SOURCE_LAYER (zero-based); do not combine it with legacy ABSLAYER_PEAK_LAYER (one-based)."
        case .invalidSourceLayer(let value):
            "ABSLAYER_SOURCE_LAYER must be a zero-based non-negative integer, not \(value)."
        case .invalidApplicationLayer(let requested, let available):
            "Zero-based application layer \(requested) is outside the model's 0..<\(available) decoder stack."
        case .peakNotExtracted(let peak, let extracted):
            "Zero-based peak layer \(peak) was not among SOM extraction layers \(extracted)."
        case .invalidPeak(let requested, let available):
            "One-based peak layer \(requested) is outside 1...\(available)."
        }
    }
}
