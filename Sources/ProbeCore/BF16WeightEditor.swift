import Foundation
import MLX

public enum DirectionScope: Sendable {
    case perLayer
    case global(layer: Float)
    case blended(globalLayer: Float, perLayerFraction: Float)
}

public struct AbliterationConfiguration: Sendable {
    public var attention: LayerAblationKernel
    public var mlp: LayerAblationKernel
    public var directionScope: DirectionScope
    public var normalization: WeightNormalization
    public var composition: AblationComposition

    public init(
        attention: LayerAblationKernel,
        mlp: LayerAblationKernel,
        directionScope: DirectionScope = .perLayer,
        normalization: WeightNormalization = .full,
        composition: AblationComposition = .simultaneous
    ) {
        self.attention = attention
        self.mlp = mlp
        self.directionScope = directionScope
        self.normalization = normalization
        self.composition = composition
    }
}

public struct WeightEditSummary: Sendable {
    public let outputPath: String
    public let editedAttentionMatrices: Int
    public let editedMLPMatrices: Int
}

public enum BF16WeightEditor {
    private struct ValidatedLayout {
        let indexData: Data
        let weightMap: [String: String]
        let shardNames: [String]
        let shardIdentities: [String: ABSlayerFileSystem.RegularFileIdentity]
    }

    public static func attentionOutputProjectionKey(layer: Int) -> String {
        "language_model.model.layers.\(layer).self_attn.o_proj.weight"
    }

    public static func mlpDownProjectionKey(layer: Int) -> String {
        "language_model.model.layers.\(layer).mlp.down_proj.weight"
    }

    /// Validates every shard name before the editor uses the Hugging Face
    /// weight map as a relative path. It also proves that both deployed edit
    /// targets exist for each requested decoder layer.
    public static func validateEditableLayout(
        sourcePath: String, layers: Set<Int>
    ) throws {
        guard !layers.isEmpty, layers.allSatisfy({ $0 >= 0 }) else {
            throw EditorError.invalidSelectedLayers(layers.sorted())
        }
        let required = Set(layers.flatMap { layer in
            [attentionOutputProjectionKey(layer: layer),
             mlpDownProjectionKey(layer: layer)]
        })
        _ = try validatedLayout(sourcePath: sourcePath, requiredKeys: required)
    }

    /// Loads only the requested tensors from a sharded checkpoint and returns
    /// evaluated arrays detached from the temporary shard dictionaries.
    public static func loadMatrices(
        sourcePath: String, keys: Set<String>
    ) throws -> [String: MLXArray] {
        guard !keys.isEmpty else { throw EditorError.noReplacements }
        let source = URL(fileURLWithPath: sourcePath).standardizedFileURL
        let layout = try validatedLayout(
            sourcePath: sourcePath, requiredKeys: keys)
        let shards = Set(keys.compactMap { layout.weightMap[$0] }).sorted()
        var result = [String: MLXArray]()
        for shardName in shards {
            let (arrays, _) = try loadValidatedShard(
                source: source, shardName: shardName, layout: layout)
            for key in keys where layout.weightMap[key] == shardName {
                guard let value = arrays[key] else {
                    throw EditorError.replacementKeysNotFound([key])
                }
                let detached = stopGradient(value)
                eval(detached)
                result[key] = detached
            }
        }
        return result
    }

    /// Writes exact replacement tensors into a copy of a sharded BF16 model.
    /// This is used by full-matrix methods such as ARA; the conventional
    /// directional editor below remains unchanged.
    public static func replaceMatrices(
        sourcePath: String,
        outputPath: String,
        replacements: [String: MLXArray]
    ) throws -> WeightEditSummary {
        let source = URL(fileURLWithPath: sourcePath).standardizedFileURL
        let output = URL(fileURLWithPath: outputPath).standardizedFileURL
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw EditorError.outputExists(output.path)
        }
        guard !replacements.isEmpty else { throw EditorError.noReplacements }

        let layout = try validatedLayout(
            sourcePath: sourcePath, requiredKeys: Set(replacements.keys))
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        for item in try FileManager.default.contentsOfDirectory(
            at: source, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) where item.pathExtension != "safetensors"
            && item.lastPathComponent != "model.safetensors.index.json"
        {
            try FileManager.default.copyItem(
                at: item, to: output.appendingPathComponent(item.lastPathComponent))
        }

        var remaining = Set(replacements.keys)
        var attentionCount = 0
        var mlpCount = 0
        for shardName in layout.shardNames {
            var (arrays, metadata) = try loadValidatedShard(
                source: source, shardName: shardName, layout: layout)
            for key in arrays.keys.sorted() {
                guard let replacement = replacements[key], let original = arrays[key] else {
                    continue
                }
                guard replacement.shape == original.shape else {
                    throw EditorError.replacementShape(
                        key: key, expected: original.shape, actual: replacement.shape)
                }
                arrays[key] = replacement.asType(original.dtype)
                remaining.remove(key)
                if let target = target(for: key) {
                    switch target.component {
                    case .attention: attentionCount += 1
                    case .mlp: mlpCount += 1
                    }
                }
            }
            eval(Array(arrays.values))
            try save(
                arrays: arrays, metadata: metadata,
                url: output.appendingPathComponent(shardName))
        }
        guard remaining.isEmpty else {
            throw EditorError.replacementKeysNotFound(remaining.sorted())
        }
        try layout.indexData.write(
            to: output.appendingPathComponent("model.safetensors.index.json"))
        return WeightEditSummary(
            outputPath: output.path,
            editedAttentionMatrices: attentionCount,
            editedMLPMatrices: mlpCount)
    }

    public static func edit(
        sourcePath: String,
        outputPath: String,
        directions: [[Float]],
        subspaces: [[[Float]]]? = nil,
        configuration: AbliterationConfiguration,
        selectedLayers: Set<Int>? = nil
    ) throws -> WeightEditSummary {
        let source = URL(fileURLWithPath: sourcePath).standardizedFileURL
        let output = URL(fileURLWithPath: outputPath).standardizedFileURL
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw EditorError.outputExists(output.path)
        }
        guard !directions.isEmpty else { throw EditorError.noDirections }
        if let selectedLayers {
            guard !selectedLayers.isEmpty,
                  selectedLayers.allSatisfy(directions.indices.contains),
                  selectedLayers.allSatisfy({ !directions[$0].isEmpty })
            else { throw EditorError.invalidSelectedLayers(selectedLayers.sorted()) }
        }

        let requiredKeys: Set<String>
        if let selectedLayers {
            requiredKeys = Set(selectedLayers.flatMap { layer in
                [attentionOutputProjectionKey(layer: layer),
                 mlpDownProjectionKey(layer: layer)]
            })
        } else {
            requiredKeys = []
        }
        let layout = try validatedLayout(
            sourcePath: sourcePath, requiredKeys: requiredKeys)

        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        for item in try FileManager.default.contentsOfDirectory(
            at: source, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) where item.pathExtension != "safetensors"
            && item.lastPathComponent != "model.safetensors.index.json"
        {
            try FileManager.default.copyItem(
                at: item, to: output.appendingPathComponent(item.lastPathComponent))
        }

        var attentionCount = 0
        var mlpCount = 0
        var remainingRequiredKeys = requiredKeys
        for shardName in layout.shardNames {
            var (arrays, metadata) = try loadValidatedShard(
                source: source, shardName: shardName, layout: layout)
            for key in arrays.keys.sorted() {
                guard let target = target(for: key), target.layer < directions.count else { continue }
                if let selectedLayers, !selectedLayers.contains(target.layer) { continue }
                // The index is authoritative. A tensor found in a different
                // shard must never count as satisfying the declared mapping.
                guard layout.weightMap[key] == shardName else { continue }
                let strength: Float
                switch target.component {
                case .attention: strength = configuration.attention.weight(at: target.layer)
                case .mlp: strength = configuration.mlp.weight(at: target.layer)
                }
                guard strength != 0, let weight = arrays[key], weight.ndim == 2 else { continue }
                let basis = AbliterationMath.resolvedBasis(
                    directions: directions, subspaces: subspaces,
                    scope: configuration.directionScope, targetLayer: target.layer,
                    composition: configuration.composition)
                guard let width = basis.first?.count, width == weight.dim(0) else {
                    throw EditorError.directionShape(
                        key: key, expected: weight.dim(0), actual: basis.first?.count ?? 0)
                }
                arrays[key] = editMatrix(
                    weight, basis: basis, strength: strength,
                    normalization: configuration.normalization,
                    composition: configuration.composition)
                remainingRequiredKeys.remove(key)
                switch target.component {
                case .attention: attentionCount += 1
                case .mlp: mlpCount += 1
                }
            }
            eval(Array(arrays.values))
            try save(
                arrays: arrays, metadata: metadata,
                url: output.appendingPathComponent(shardName))
        }
        guard remainingRequiredKeys.isEmpty else {
            throw EditorError.replacementKeysNotFound(
                remainingRequiredKeys.sorted())
        }
        try layout.indexData.write(
            to: output.appendingPathComponent("model.safetensors.index.json"))
        return WeightEditSummary(
            outputPath: output.path,
            editedAttentionMatrices: attentionCount,
            editedMLPMatrices: mlpCount)
    }

    private enum Component { case attention, mlp }
    private struct Target { let layer: Int; let component: Component }

    private static func validatedLayout(
        sourcePath: String, requiredKeys: Set<String>
    ) throws -> ValidatedLayout {
        let source = URL(fileURLWithPath: sourcePath).standardizedFileURL
        let indexURL = source.appendingPathComponent("model.safetensors.index.json")
        guard let indexIdentity = ABSlayerFileSystem.regularFileIdentity(
            indexURL.path)
        else { throw EditorError.missingIndex(indexURL.path) }
        let mappedIndex = try Data(contentsOf: indexURL, options: [.mappedIfSafe])
        guard mappedIndex.count <= 64 * 1_024 * 1_024,
              ABSlayerFileSystem.regularFileIdentity(indexURL.path) == indexIdentity
        else { throw EditorError.missingIndex(indexURL.path) }
        let indexData = mappedIndex.withUnsafeBytes { Data($0) }
        guard ABSlayerFileSystem.regularFileIdentity(indexURL.path) == indexIdentity,
              (try? ABSlayerStrictJSON.validateNoDuplicateKeys(indexData)) != nil,
              let index = try JSONSerialization.jsonObject(with: indexData)
                  as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String]
        else { throw EditorError.missingIndex(indexURL.path) }
        let shardNames = Set(weightMap.values).sorted()
        guard !shardNames.isEmpty else { throw EditorError.missingIndex(indexURL.path) }
        var identities = [String: ABSlayerFileSystem.RegularFileIdentity]()
        for shardName in shardNames {
            let component = URL(fileURLWithPath: shardName).lastPathComponent
            guard !shardName.isEmpty, component == shardName,
                  shardName != ".", shardName != "..",
                  shardName.hasSuffix(".safetensors"),
                  let identity = ABSlayerFileSystem.regularFileIdentity(
                      source.appendingPathComponent(shardName).path)
            else { throw EditorError.invalidShardName(shardName) }
            identities[shardName] = identity
        }
        let missing = requiredKeys.subtracting(weightMap.keys)
        guard missing.isEmpty else {
            throw EditorError.replacementKeysNotFound(missing.sorted())
        }
        return ValidatedLayout(
            indexData: indexData, weightMap: weightMap,
            shardNames: shardNames, shardIdentities: identities)
    }

    private static func loadValidatedShard(
        source: URL, shardName: String, layout: ValidatedLayout
    ) throws -> ([String: MLXArray], [String: String]) {
        let shardURL = source.appendingPathComponent(shardName)
        guard let expected = layout.shardIdentities[shardName],
              ABSlayerFileSystem.regularFileIdentity(shardURL.path) == expected
        else { throw EditorError.shardChanged(shardName) }
        let loaded = try loadArraysAndMetadata(url: shardURL)
        // Safetensors arrays may be lazy. Materialize them before the final
        // identity comparison so subsequent consumers cannot fault bytes from
        // a shard that has already changed on disk.
        eval(Array(loaded.0.values))
        guard ABSlayerFileSystem.regularFileIdentity(shardURL.path) == expected else {
            throw EditorError.shardChanged(shardName)
        }
        return loaded
    }

    private static func target(for key: String) -> Target? {
        guard key.hasPrefix("language_model.model.layers."), key.hasSuffix(".weight") else {
            return nil
        }
        let parts = key.split(separator: ".")
        guard parts.count > 6, let layer = Int(parts[3]) else { return nil }
        if key.hasSuffix(".self_attn.o_proj.weight") {
            return Target(layer: layer, component: .attention)
        }
        if key.hasSuffix(".mlp.down_proj.weight") {
            return Target(layer: layer, component: .mlp)
        }
        return nil
    }

    static func editMatrix(
        _ source: MLXArray,
        basis: [[Float]],
        strength: Float,
        normalization: WeightNormalization,
        composition: AblationComposition = .simultaneous
    ) -> MLXArray {
        let originalType = source.dtype
        let original = source.asType(.float32)
        let rowNorms = sqrt((original * original).sum(axis: 1, keepDims: true))
        var working = original
        if normalization != .none { working = working / maximum(rowNorms, MLXArray(1e-12)) }
        switch composition {
        case .simultaneous:
            let basisArray = MLXArray(basis.flatMap { $0 })
                .reshaped(basis.count, basis[0].count).asType(.float32)
            let projected = matmul(basisArray.T, matmul(basisArray, working))
            working = working - strength * projected
        case .sequential:
            for rawDirection in basis {
                let direction = MLXArray(AbliterationMath.normalized(rawDirection))
                    .reshaped(rawDirection.count, 1).asType(.float32)
                working = working - strength * matmul(
                    direction, matmul(direction.T, working))
            }
        }
        if normalization == .pre {
            working = working * rowNorms
        } else if normalization == .full {
            let editedNorms = sqrt((working * working).sum(axis: 1, keepDims: true))
            working = working * rowNorms / maximum(editedNorms, MLXArray(1e-12))
        }
        return working.asType(originalType)
    }
}

public enum EditorError: LocalizedError {
    case outputExists(String)
    case noDirections
    case noReplacements
    case invalidSelectedLayers([Int])
    case invalidShardName(String)
    case shardChanged(String)
    case missingIndex(String)
    case directionShape(key: String, expected: Int, actual: Int)
    case replacementKeysNotFound([String])
    case replacementShape(key: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .outputExists(let path): "Output folder already exists: \(path)"
        case .noDirections: "No measured layer directions were provided."
        case .noReplacements: "No replacement weight matrices were provided."
        case .invalidSelectedLayers(let layers):
            "Selected layer indices are empty or outside the direction array: \(layers)."
        case .invalidShardName(let name):
            "Unsafe or missing safetensors shard in the model index: \(name)"
        case .shardChanged(let name):
            "Safetensors shard changed while it was being read: \(name)"
        case .missingIndex(let path): "Missing or malformed shard index: \(path)"
        case .directionShape(let key, let expected, let actual):
            "Direction size \(actual) does not match \(key)'s output size \(expected)."
        case .replacementKeysNotFound(let keys):
            "Replacement tensor key(s) are absent from the model index: \(keys.joined(separator: ", "))"
        case .replacementShape(let key, let expected, let actual):
            "Replacement shape \(actual) does not match \(key)'s source shape \(expected)."
        }
    }
}
