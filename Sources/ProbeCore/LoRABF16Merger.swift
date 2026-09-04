import Foundation
import MLX
import MLXLMCommon

public struct LoRAMergeSummary: Sendable {
    public let outputPath: String
    public let mergedMatrices: Int
}

public enum LoRABF16Merger {
    /// Merges a conventional dense LoRA into a sharded BF16 checkpoint without
    /// loading the complete model. The source and adapter are never modified.
    public static func merge(
        sourcePath: String, adapterPath: String, outputPath: String
    ) throws -> LoRAMergeSummary {
        let source = URL(fileURLWithPath: sourcePath).standardizedFileURL
        let adapter = URL(fileURLWithPath: adapterPath).standardizedFileURL
        let output = URL(fileURLWithPath: outputPath).standardizedFileURL
        let manager = FileManager.default
        guard !manager.fileExists(atPath: output.path) else {
            throw LoRAMergeError.outputExists(output.path)
        }

        let configuration = try JSONDecoder().decode(
            LoRAConfiguration.self,
            from: Data(contentsOf: adapter.appendingPathComponent("adapter_config.json")))
        guard configuration.fineTuneType == .lora else {
            throw LoRAMergeError.unsupportedFineTuneType
        }
        let adapterArrays = try MLX.loadArrays(
            url: adapter.appendingPathComponent("adapters.safetensors"))
        let pairs = try adapterPairs(adapterArrays, rank: configuration.loraParameters.rank)

        let indexURL = source.appendingPathComponent("model.safetensors.index.json")
        let indexData = try Data(contentsOf: indexURL)
        guard let index = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String]
        else { throw LoRAMergeError.missingIndex(indexURL.path) }
        let unknown = Set(pairs.keys).subtracting(weightMap.keys)
        guard unknown.isEmpty else { throw LoRAMergeError.baseWeightsMissing(unknown.sorted()) }

        try manager.createDirectory(at: output, withIntermediateDirectories: true)
        do {
            for item in try manager.contentsOfDirectory(
                at: source, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
            ) where item.pathExtension != "safetensors"
                && item.lastPathComponent != "model.safetensors.index.json"
            {
                try manager.copyItem(
                    at: item, to: output.appendingPathComponent(item.lastPathComponent))
            }

            var merged = 0
            for shardName in Set(weightMap.values).sorted() {
                var (arrays, metadata) = try loadArraysAndMetadata(
                    url: source.appendingPathComponent(shardName))
                for key in arrays.keys.sorted() {
                    guard let pair = pairs[key], let weight = arrays[key] else { continue }
                    arrays[key] = try mergedMatrix(
                        weight: weight, loraA: pair.a, loraB: pair.b,
                        scale: configuration.loraParameters.scale, key: key)
                    merged += 1
                }
                eval(Array(arrays.values))
                try MLX.save(
                    arrays: arrays, metadata: metadata,
                    url: output.appendingPathComponent(shardName))
            }
            guard merged == pairs.count else {
                throw LoRAMergeError.incompleteMerge(expected: pairs.count, actual: merged)
            }
            try indexData.write(
                to: output.appendingPathComponent("model.safetensors.index.json"), options: .atomic)
            return LoRAMergeSummary(outputPath: output.path, mergedMatrices: merged)
        } catch {
            try? manager.removeItem(at: output)
            throw error
        }
    }

    public static func mergedMatrix(
        weight: MLXArray, loraA: MLXArray, loraB: MLXArray,
        scale: Float, key: String = "weight"
    ) throws -> MLXArray {
        guard weight.ndim == 2, loraA.ndim == 2, loraB.ndim == 2,
              loraA.dim(1) == loraB.dim(0),
              weight.dim(0) == loraB.dim(1), weight.dim(1) == loraA.dim(0)
        else {
            throw LoRAMergeError.incompatibleShape(
                key: key, weight: weight.shape, a: loraA.shape, b: loraB.shape)
        }
        let dtype = weight.dtype
        let delta = scale * matmul(
            loraB.T.asType(.float32), loraA.T.asType(.float32))
        return (weight.asType(.float32) + delta).asType(dtype)
    }

    private struct Pair { let a: MLXArray; let b: MLXArray }

    private static func adapterPairs(
        _ arrays: [String: MLXArray], rank: Int
    ) throws -> [String: Pair] {
        var components = [String: (a: MLXArray?, b: MLXArray?)]()
        for (name, array) in arrays {
            let suffix: String
            if name.hasSuffix(".lora_a") { suffix = ".lora_a" }
            else if name.hasSuffix(".lora_b") { suffix = ".lora_b" }
            else { throw LoRAMergeError.unexpectedAdapterTensor(name) }
            let stem = String(name.dropLast(suffix.count))
            var pair = components[stem] ?? (nil, nil)
            if suffix == ".lora_a" { pair.a = array } else { pair.b = array }
            components[stem] = pair
        }
        guard !components.isEmpty else { throw LoRAMergeError.emptyAdapter }
        var result = [String: Pair]()
        for (stem, component) in components {
            guard let a = component.a, let b = component.b else {
                throw LoRAMergeError.unpairedAdapterTensor(stem)
            }
            guard a.ndim == 2, b.ndim == 2, a.dim(1) == rank, b.dim(0) == rank else {
                throw LoRAMergeError.rankMismatch(stem)
            }
            result[stem + ".weight"] = Pair(a: a, b: b)
        }
        return result
    }
}

public enum LoRAMergeError: LocalizedError {
    case outputExists(String)
    case missingIndex(String)
    case unsupportedFineTuneType
    case emptyAdapter
    case unexpectedAdapterTensor(String)
    case unpairedAdapterTensor(String)
    case rankMismatch(String)
    case baseWeightsMissing([String])
    case incompatibleShape(key: String, weight: [Int], a: [Int], b: [Int])
    case incompleteMerge(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .outputExists(let path): "Output model already exists: \(path)"
        case .missingIndex(let path): "Missing or invalid sharded model index: \(path)"
        case .unsupportedFineTuneType: "Only dense LoRA adapters can be merged into BF16 weights."
        case .emptyAdapter: "The adapter contains no LoRA tensors."
        case .unexpectedAdapterTensor(let name): "Unexpected adapter tensor: \(name)"
        case .unpairedAdapterTensor(let stem): "LoRA tensor pair is incomplete: \(stem)"
        case .rankMismatch(let stem): "LoRA tensor rank disagrees with adapter_config.json: \(stem)"
        case .baseWeightsMissing(let keys): "Adapter targets missing base weights: \(keys.joined(separator: ", "))"
        case .incompatibleShape(let key, let weight, let a, let b):
            "Incompatible LoRA shapes for \(key): weight=\(weight), a=\(a), b=\(b)"
        case .incompleteMerge(let expected, let actual):
            "Incomplete LoRA merge: expected \(expected) matrices, merged \(actual)."
        }
    }
}
