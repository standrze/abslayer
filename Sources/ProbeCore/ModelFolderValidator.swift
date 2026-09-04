#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import CoreFoundation

public enum ModelFolderKind: String, Sendable {
    case quantized
    case fullBF16
}

public struct ModelFolderInspection: Sendable {
    public let kind: ModelFolderKind
    public let path: String
    public let weightFiles: Int
    public let detectedDTypes: [String]
    /// Decoder-layer count reported by the model configuration. Multimodal
    /// checkpoints commonly put this under `text_config`; using the root count
    /// could accidentally select an audio or vision encoder instead.
    public let decoderLayerCount: Int?
    /// Decoder hidden width from the text configuration, when declared.
    public let hiddenSize: Int?
    /// Root Hugging Face architecture discriminator (for example `gemma4`).
    public let modelType: String?
    /// Nested text-decoder discriminator for multimodal checkpoints.
    public let textModelType: String?
}

public enum ModelFolderValidationError: LocalizedError {
    case missingDirectory(String)
    case sameDirectory
    case missingFile(folder: String, file: String)
    case missingWeights(String)
    case malformedSafeTensors(String)
    case noBF16Weights(String)
    case fullBF16ContainsQuantizedWeights(String)
    case quantizedLooksFullPrecision(String)

    public var errorDescription: String? {
        switch self {
        case .missingDirectory(let path):
            "Model directory does not exist: \(path)"
        case .sameDirectory:
            "Choose two different directories: one quantized model and one full BF16 model."
        case .missingFile(let folder, let file):
            "\(folder) is missing \(file)."
        case .missingWeights(let folder):
            "\(folder) contains no .safetensors weight files."
        case .malformedSafeTensors(let file):
            "Could not read the safetensors header in \(file)."
        case .noBF16Weights(let folder):
            "\(folder) does not contain BF16 tensors; select the full BF16 checkpoint."
        case .fullBF16ContainsQuantizedWeights(let folder):
            "\(folder) contains quantization metadata or non-floating tensors; select an unquantized BF16 checkpoint."
        case .quantizedLooksFullPrecision(let folder):
            "\(folder) looks full precision, not quantized. Select the quantized checkpoint."
        }
    }
}

public enum ModelFolderValidator {
    public static func validateFullBF16(path: String) throws -> ModelFolderInspection {
        let url = normalizedURL(path)
        let inspection = try inspect(url, kind: .fullBF16)
        guard inspection.detectedDTypes == ["BF16"] else {
            throw ModelFolderValidationError.noBF16Weights(inspection.path)
        }
        let configuration = try config(at: url)
        let floatingDTypes = Set(["BF16", "F16", "F32", "F64"])
        guard configuration["quantization"] == nil,
              configuration["quantization_config"] == nil,
              Set(inspection.detectedDTypes).isSubset(of: floatingDTypes)
        else {
            throw ModelFolderValidationError.fullBF16ContainsQuantizedWeights(
                inspection.path)
        }
        return inspection
    }

    public static func validatePair(
        quantizedPath: String,
        fullBF16Path: String
    ) throws -> (quantized: ModelFolderInspection, fullBF16: ModelFolderInspection) {
        let quantizedURL = normalizedURL(quantizedPath)
        let bf16URL = normalizedURL(fullBF16Path)
        guard quantizedURL != bf16URL else { throw ModelFolderValidationError.sameDirectory }

        let quantized = try inspect(quantizedURL, kind: .quantized)
        let bf16 = try validateFullBF16(path: bf16URL.path)

        let quantizedConfig = try config(at: quantizedURL)
        let hasQuantizationMetadata =
            quantizedConfig["quantization"] != nil
            || quantizedConfig["quantization_config"] != nil
        let quantizedDTypes = Set(quantized.detectedDTypes)
        if !hasQuantizationMetadata && quantizedDTypes.isSubset(of: ["BF16", "F16", "F32"]) {
            throw ModelFolderValidationError.quantizedLooksFullPrecision(quantized.path)
        }

        return (quantized, bf16)
    }

    private static func inspect(
        _ folder: URL,
        kind: ModelFolderKind
    ) throws -> ModelFolderInspection {
        guard ABSlayerFileSystem.isDirectoryWithoutFollowingSymlink(folder.path)
        else { throw ModelFolderValidationError.missingDirectory(folder.path) }

        for required in ["config.json", "tokenizer_config.json"] {
            guard ABSlayerFileSystem.isRegularFileWithoutFollowingSymlink(
                folder.appendingPathComponent(required).path) else {
                throw ModelFolderValidationError.missingFile(folder: folder.path, file: required)
            }
        }

        let files = try FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        let allWeights = files.filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !allWeights.isEmpty else {
            throw ModelFolderValidationError.missingWeights(folder.path)
        }
        guard allWeights.allSatisfy({
            ABSlayerFileSystem.isRegularFileWithoutFollowingSymlink($0.path)
        }) else { throw ModelFolderValidationError.malformedSafeTensors(folder.path) }

        let indexURL = folder.appendingPathComponent("model.safetensors.index.json")
        let weightMap: [String: String]?
        let weights: [URL]
        if ABSlayerFileSystem.pathExistsWithoutFollowingSymlink(indexURL.path) {
            let parsed = try loadWeightMap(indexURL)
            let byName = Dictionary(
                uniqueKeysWithValues: allWeights.map { ($0.lastPathComponent, $0) })
            let referenced = Set(parsed.values)
            guard referenced.allSatisfy({ byName[$0] != nil }) else {
                throw ModelFolderValidationError.malformedSafeTensors(indexURL.path)
            }
            weightMap = parsed
            weights = referenced.sorted().compactMap { byName[$0] }
        } else {
            guard allWeights.count == 1 else {
                throw ModelFolderValidationError.malformedSafeTensors(folder.path)
            }
            weightMap = nil
            weights = allWeights
        }

        var dtypes = Set<String>()
        var tensorsByShard = [String: Set<String>]()
        for weight in weights {
            let inspection = try inspectSafeTensors(weight)
            dtypes.formUnion(inspection.dtypes)
            tensorsByShard[weight.lastPathComponent] = inspection.tensorNames
        }
        if let weightMap {
            guard weightMap.allSatisfy({ key, shard in
                tensorsByShard[shard]?.contains(key) == true
            }), tensorsByShard.allSatisfy({ shard, names in
                names.allSatisfy { weightMap[$0] == shard }
            }) else {
                throw ModelFolderValidationError.malformedSafeTensors(indexURL.path)
            }
        }
        let modelConfig = try config(at: folder)
        return ModelFolderInspection(
            kind: kind,
            path: folder.path,
            weightFiles: weights.count,
            detectedDTypes: dtypes.sorted(),
            decoderLayerCount: decoderLayerCount(in: modelConfig),
            hiddenSize: hiddenSize(in: modelConfig),
            modelType: modelConfig["model_type"] as? String,
            textModelType: nestedTextConfig(in: modelConfig)?["model_type"] as? String
        )
    }

    private static func normalizedURL(_ path: String) -> URL {
        URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
            .standardizedFileURL
    }

    private static func config(at folder: URL) throws -> [String: Any] {
        let url = folder.appendingPathComponent("config.json")
        let data = try stableRegularFileData(url, maximumBytes: 64 * 1_024 * 1_024)
        guard (try? ABSlayerStrictJSON.validateNoDuplicateKeys(data)) != nil else {
            throw ModelFolderValidationError.missingFile(
                folder: folder.path, file: "config.json")
        }
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }

    /// Returns the text decoder depth, not an unrelated modality encoder
    /// depth. The root fallback covers text-only Hugging Face checkpoints.
    private static func decoderLayerCount(in config: [String: Any]) -> Int? {
        for key in ["text_config", "language_config"] {
            if let nested = config[key] as? [String: Any],
               let count = positiveInteger(nested["num_hidden_layers"])
            {
                return count
            }
        }
        return positiveInteger(config["num_hidden_layers"])
            ?? positiveInteger(config["num_layers"])
    }

    private static func hiddenSize(in config: [String: Any]) -> Int? {
        for key in ["text_config", "language_config"] {
            if let nested = config[key] as? [String: Any],
               let size = positiveInteger(nested["hidden_size"])
            {
                return size
            }
        }
        return positiveInteger(config["hidden_size"])
            ?? positiveInteger(config["d_model"])
    }

    private static func nestedTextConfig(
        in config: [String: Any]
    ) -> [String: Any]? {
        for key in ["text_config", "language_config"] {
            if let nested = config[key] as? [String: Any] { return nested }
        }
        return nil
    }

    private static func positiveInteger(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID()
        else { return nil }
        let count = number.intValue
        guard count > 0, number.doubleValue == Double(count) else { return nil }
        return count
    }

    private struct SafeTensorInspection {
        let dtypes: Set<String>
        let tensorNames: Set<String>
    }

    /// Safetensors starts with an unsigned 64-bit little-endian JSON-header size.
    /// Reading only that header validates tensor schemas and payload spans
    /// without materializing multi-gigabyte weights.
    private static func inspectSafeTensors(
        _ file: URL
    ) throws -> SafeTensorInspection {
        guard let expected = ABSlayerFileSystem.regularFileIdentity(file.path),
              expected.size >= 9
        else { throw ModelFolderValidationError.malformedSafeTensors(file.path) }
        let descriptor = file.path.withCString {
            open($0, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        }
        guard descriptor >= 0,
              ABSlayerFileSystem.regularFileIdentity(descriptor: descriptor) == expected
        else {
            if descriptor >= 0 { close(descriptor) }
            throw ModelFolderValidationError.malformedSafeTensors(file.path)
        }
        defer { close(descriptor) }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        guard let sizeData = try readExactly(handle, count: 8), sizeData.count == 8 else {
            throw ModelFolderValidationError.malformedSafeTensors(file.path)
        }
        let headerSize = sizeData.withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard headerSize > 0, headerSize <= 100_000_000,
              headerSize <= UInt64(expected.size - 8),
              let header = try readExactly(handle, count: Int(headerSize)),
              header.count == Int(headerSize),
              (try? ABSlayerStrictJSON.validateNoDuplicateKeys(header)) != nil,
              let object = try JSONSerialization.jsonObject(with: header) as? [String: Any]
        else { throw ModelFolderValidationError.malformedSafeTensors(file.path) }

        var result = Set<String>()
        var tensorNames = Set<String>()
        var spans = [(start: Int64, end: Int64)]()
        let payloadBytes = expected.size - 8 - Int64(headerSize)
        for (name, value) in object where name != "__metadata__" {
            guard let tensor = value as? [String: Any],
                  Set(tensor.keys) == ["dtype", "shape", "data_offsets"],
                  let dtype = tensor["dtype"] as? String, !dtype.isEmpty,
                  let shape = tensor["shape"] as? [Any],
                  let elementBytes = dataTypeByteWidth(dtype),
                  let elementCount = tensorElementCount(shape),
                  let offsets = tensor["data_offsets"] as? [Any], offsets.count == 2,
                  let start = nonnegativeInteger(offsets[0]),
                  let end = nonnegativeInteger(offsets[1]),
                  start <= end, end <= payloadBytes,
                  elementCount <= Int64.max / elementBytes,
                  end - start == elementCount * elementBytes,
                  !name.isEmpty, tensorNames.insert(name).inserted
            else { throw ModelFolderValidationError.malformedSafeTensors(file.path) }
            spans.append((start, end))
            result.insert(dtype)
        }
        if let metadata = object["__metadata__"] {
            guard metadata is [String: String] else {
                throw ModelFolderValidationError.malformedSafeTensors(file.path)
            }
        }
        let orderedSpans = spans.sorted {
            $0.start == $1.start ? $0.end < $1.end : $0.start < $1.start
        }
        var covered: Int64 = 0
        for span in orderedSpans {
            guard span.start == covered else {
                throw ModelFolderValidationError.malformedSafeTensors(file.path)
            }
            covered = span.end
        }
        guard !tensorNames.isEmpty, !result.isEmpty, covered == payloadBytes,
              ABSlayerFileSystem.regularFileIdentity(descriptor: descriptor) == expected,
              ABSlayerFileSystem.regularFileIdentity(file.path) == expected
        else { throw ModelFolderValidationError.malformedSafeTensors(file.path) }
        return SafeTensorInspection(dtypes: result, tensorNames: tensorNames)
    }

    private static func loadWeightMap(_ index: URL) throws -> [String: String] {
        let data = try stableRegularFileData(index, maximumBytes: 64 * 1_024 * 1_024)
        guard (try? ABSlayerStrictJSON.validateNoDuplicateKeys(data)) != nil,
              let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(root.keys).isSubset(of: ["metadata", "weight_map"]),
              let map = root["weight_map"] as? [String: String], !map.isEmpty,
              map.allSatisfy({ key, shard in
                  !key.isEmpty && !shard.isEmpty
                      && shard == URL(fileURLWithPath: shard).lastPathComponent
                      && shard != "." && shard != ".."
                      && shard.hasSuffix(".safetensors")
              })
        else { throw ModelFolderValidationError.malformedSafeTensors(index.path) }
        return map
    }

    private static func tensorElementCount(_ shape: [Any]) -> Int64? {
        var count: Int64 = 1
        for dimensionValue in shape {
            guard let dimension = nonnegativeInteger(dimensionValue),
                  dimension == 0 || count <= Int64.max / dimension
            else { return nil }
            count *= dimension
        }
        return count
    }

    private static func dataTypeByteWidth(_ dtype: String) -> Int64? {
        switch dtype {
        case "BOOL", "I8", "U8", "F8_E4M3", "F8_E5M2": 1
        case "I16", "U16", "F16", "BF16": 2
        case "I32", "U32", "F32": 4
        case "I64", "U64", "F64": 8
        default: nil
        }
    }

    private static func stableRegularFileData(
        _ file: URL, maximumBytes: Int
    ) throws -> Data {
        guard let expected = ABSlayerFileSystem.regularFileIdentity(file.path),
              expected.size >= 0, expected.size <= Int64(maximumBytes)
        else {
            throw ModelFolderValidationError.missingFile(
                folder: file.deletingLastPathComponent().path,
                file: file.lastPathComponent)
        }
        let descriptor = file.path.withCString {
            open($0, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        }
        guard descriptor >= 0,
              ABSlayerFileSystem.regularFileIdentity(descriptor: descriptor) == expected
        else {
            if descriptor >= 0 { close(descriptor) }
            throw ModelFolderValidationError.missingFile(
                folder: file.deletingLastPathComponent().path,
                file: file.lastPathComponent)
        }
        defer { close(descriptor) }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
        guard let data = try readExactly(handle, count: Int(expected.size)),
              ABSlayerFileSystem.regularFileIdentity(descriptor: descriptor) == expected,
              ABSlayerFileSystem.regularFileIdentity(file.path) == expected
        else { throw ModelFolderValidationError.malformedSafeTensors(file.path) }
        return data
    }

    private static func readExactly(
        _ handle: FileHandle, count: Int
    ) throws -> Data? {
        var result = Data()
        result.reserveCapacity(count)
        while result.count < count {
            guard let chunk = try handle.read(upToCount: count - result.count),
                  !chunk.isEmpty
            else { return nil }
            result.append(chunk)
        }
        return result
    }

    private static func nonnegativeInteger(_ value: Any) -> Int64? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID()
        else { return nil }
        let integer = number.int64Value
        guard integer >= 0, number.doubleValue == Double(integer) else { return nil }
        return integer
    }
}
