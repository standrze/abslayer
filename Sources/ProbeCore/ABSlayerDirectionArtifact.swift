#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import MLX

public struct ABSlayerDirectionArtifact: Sendable {
    public let path: String
    public let manifest: ABSlayerDirectionArtifactManifest
    public let directions: [[Float]]
    public let subspaces: [[[Float]]]
    public let manifestSHA256: String
    public let directionsSHA256: String

    public init(
        path: String, manifest: ABSlayerDirectionArtifactManifest,
        directions: [[Float]], subspaces: [[[Float]]],
        manifestSHA256: String, directionsSHA256: String
    ) {
        self.path = path
        self.manifest = manifest
        self.directions = directions
        self.subspaces = subspaces
        self.manifestSHA256 = manifestSHA256
        self.directionsSHA256 = directionsSHA256
    }
}

/// Owns the stable, portable direction-artifact boundary used between the
/// harness's independently retried measure and apply processes.
public enum ABSlayerDirectionArtifactStore {
    public static let format = "abslayer.artifact/v1"
    public static let tensorFile = "directions.safetensors"
    public static let manifestFile = "manifest.json"
    public static let tensorSchema = "abslayer.layer-direction/v1"
    public static let maximumManifestBytes = 1_048_576
    public static let maximumTensorBytes = 512 * 1_024 * 1_024

    @discardableResult
    public static func write(
        subspaces: [[[Float]]],
        model: ABSlayerModelBinding,
        dataset: ABSlayerDatasetBinding,
        algorithm: ABSlayerDirectionAlgorithm,
        runtime: ABSlayerDirectionRuntime,
        to path: String
    ) throws -> ABSlayerDirectionArtifactManifest {
        try validate(
            subspaces: subspaces, model: model, dataset: dataset,
            algorithm: algorithm, runtime: runtime)
        let destination = URL(fileURLWithPath: path).standardizedFileURL
        let manager = FileManager.default
        guard !ABSlayerFileSystem.pathExistsWithoutFollowingSymlink(destination.path) else {
            throw ABSlayerDirectionArtifactError.outputExists(destination.path)
        }
        let parent = destination.deletingLastPathComponent()
        try manager.createDirectory(at: parent, withIntermediateDirectories: true)
        guard ABSlayerFileSystem.isDirectoryWithoutFollowingSymlink(parent.path) else {
            throw ABSlayerDirectionArtifactError.invalidOutputParent(parent.path)
        }
        let staging = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).staging-\(UUID().uuidString)",
            isDirectory: true)
        try manager.createDirectory(at: staging, withIntermediateDirectories: false)
        do {
            var arrays = [String: MLXArray]()
            for layer in algorithm.selectedLayers {
                for (direction, vector) in subspaces[layer].enumerated() {
                    arrays[tensorKey(layer: layer, direction: direction)] =
                        MLXArray(vector).asType(.float32)
                }
            }
            eval(Array(arrays.values))
            let tensorsURL = staging.appendingPathComponent(tensorFile)
            try MLX.save(arrays: arrays, url: tensorsURL)
            let tensorHash = try ScreeningReviewProvenance.fileSHA256(tensorsURL.path)
            let manifest = ABSlayerDirectionArtifactManifest(
                format: format,
                model: model,
                dataset: dataset,
                algorithm: algorithm,
                tensors: ABSlayerDirectionTensorBinding(
                    file: tensorFile, sha256: tensorHash, schema: tensorSchema),
                runtime: runtime)
            let data = try canonicalJSON(manifest)
            try data.write(
                to: staging.appendingPathComponent(manifestFile),
                options: .withoutOverwriting)
            try manager.moveItem(at: staging, to: destination)
            return manifest
        } catch {
            try? manager.removeItem(at: staging)
            throw error
        }
    }

    public static func load(from path: String) throws -> ABSlayerDirectionArtifact {
        let root = URL(fileURLWithPath: path).standardizedFileURL
        guard ABSlayerFileSystem.isDirectoryWithoutFollowingSymlink(root.path) else {
            throw ABSlayerDirectionArtifactError.invalidArtifactDirectory(root.path)
        }
        let manifestURL = root.appendingPathComponent(manifestFile)
        let tensorsURL = root.appendingPathComponent(tensorFile)
        let manifestData = try stableFileData(
            manifestURL, maximumBytes: maximumManifestBytes)
        let tensorsData = try stableFileData(
            tensorsURL, maximumBytes: maximumTensorBytes)
        guard !manifestData.isEmpty else {
            throw ABSlayerDirectionArtifactError.invalidManifest
        }
        try validateClosedManifestJSON(manifestData)
        let decoder = JSONDecoder()
        let manifest: ABSlayerDirectionArtifactManifest
        do { manifest = try decoder.decode(ABSlayerDirectionArtifactManifest.self, from: manifestData) }
        catch { throw ABSlayerDirectionArtifactError.invalidManifest }
        guard manifest.format == format,
              manifest.tensors.file == tensorFile,
              manifest.tensors.schema == tensorSchema
        else { throw ABSlayerDirectionArtifactError.invalidManifest }
        let tensorHash = ScreeningReviewProvenance.sha256(tensorsData)
        guard tensorHash == manifest.tensors.sha256 else {
            throw ABSlayerDirectionArtifactError.tensorHashMismatch
        }
        let (arrays, _) = try loadArraysAndMetadata(data: tensorsData)
        let expectedLayers = Set(manifest.algorithm.selectedLayers)
        var byLayer = [Int: [Int: [Float]]]()
        for (key, rawArray) in arrays {
            guard let parsed = parseTensorKey(key),
                  expectedLayers.contains(parsed.layer),
                  parsed.direction < manifest.algorithm.rank,
                  rawArray.ndim == 1,
                  rawArray.dim(0) == manifest.model.hiddenSize,
                  byLayer[parsed.layer]?[parsed.direction] == nil
            else { throw ABSlayerDirectionArtifactError.invalidTensorSchema(key) }
            let array = rawArray.asType(.float32)
            eval(array)
            let vector = array.asArray(Float.self)
            guard vector.allSatisfy(\.isFinite), vector.contains(where: { $0 != 0 }) else {
                throw ABSlayerDirectionArtifactError.invalidTensorSchema(key)
            }
            byLayer[parsed.layer, default: [:]][parsed.direction] = vector
        }
        guard Set(byLayer.keys) == expectedLayers else {
            throw ABSlayerDirectionArtifactError.incompleteTensorSet
        }
        var subspaces = Array(
            repeating: [[Float]](), count: manifest.model.decoderLayerCount)
        for layer in manifest.algorithm.selectedLayers {
            guard let entries = byLayer[layer], !entries.isEmpty,
                  entries.keys.sorted() == Array(0 ..< entries.count)
            else { throw ABSlayerDirectionArtifactError.incompleteTensorSet }
            subspaces[layer] = entries.keys.sorted().compactMap { entries[$0] }
        }
        try validate(
            subspaces: subspaces, model: manifest.model,
            dataset: manifest.dataset, algorithm: manifest.algorithm,
            runtime: manifest.runtime)
        let directions = subspaces.map { $0.first ?? [] }
        return ABSlayerDirectionArtifact(
            path: root.path, manifest: manifest,
            directions: directions, subspaces: subspaces,
            manifestSHA256: ScreeningReviewProvenance.sha256(manifestData),
            directionsSHA256: tensorHash)
    }

    private static func validate(
        subspaces: [[[Float]]], model: ABSlayerModelBinding,
        dataset: ABSlayerDatasetBinding, algorithm: ABSlayerDirectionAlgorithm,
        runtime: ABSlayerDirectionRuntime
    ) throws {
        let selected = algorithm.selectedLayers
        guard (1 ... 1_000).contains(model.decoderLayerCount), model.hiddenSize > 0,
              subspaces.count == model.decoderLayerCount,
              !model.identifier.isEmpty, !model.canonicalPath.isEmpty,
              isLowercaseSHA256(model.metadataSHA256),
              isLowercaseSHA256(model.weightsSHA256),
              (1 ... 10_000).contains(dataset.recordCount),
              dataset.promptCount.map({ (1 ... dataset.recordCount).contains($0) }) ?? true,
              !dataset.path.isEmpty, isLowercaseSHA256(dataset.sha256),
              algorithm.name == "paired-centroid-rank-k/v1",
              algorithm.strength.isFinite,
              algorithm.strength > 0, algorithm.strength <= 2,
              algorithm.rank > 0, algorithm.rank <= 64,
              algorithm.maximumLayerFraction.isFinite,
              (0.01 ... 1).contains(algorithm.maximumLayerFraction),
              algorithm.tokenPosition == ActivationTokenPosition.postInstruction.rawValue,
              !selected.isEmpty, selected == selected.sorted(),
              Set(selected).count == selected.count,
              selected.allSatisfy((0 ..< model.decoderLayerCount).contains),
              selected.count <= max(
                  1, Int(floor(Double(model.decoderLayerCount)
                      * algorithm.maximumLayerFraction))),
              runtime.backend == "mlx-swift-cuda",
              (1 ... 32_768).contains(runtime.maximumSequenceLength),
              runtime.gpuMemoryUtilization.isFinite,
              (0.01 ... 1).contains(runtime.gpuMemoryUtilization),
              runtime.temporaryDirectory == nil
        else { throw ABSlayerDirectionArtifactError.invalidManifest }
        for layer in selected {
            let basis = subspaces[layer]
            guard !basis.isEmpty, basis.count <= algorithm.rank,
                  basis.allSatisfy({ vector in
                      vector.count == model.hiddenSize
                          && vector.allSatisfy(\.isFinite)
                          && vector.contains(where: { $0 != 0 })
                  }),
                  isOrthonormal(basis)
            else { throw ABSlayerDirectionArtifactError.invalidTensorSchema("layer \(layer)") }
        }
    }

    private static func isOrthonormal(
        _ basis: [[Float]], tolerance: Double = 1e-3
    ) -> Bool {
        for row in basis.indices {
            let normSquared = basis[row].reduce(0.0) {
                $0 + Double($1) * Double($1)
            }
            guard normSquared.isFinite,
                  abs(normSquared - 1) <= tolerance else { return false }
            for other in basis.indices where other < row {
                let dot = zip(basis[row], basis[other]).reduce(0.0) {
                    $0 + Double($1.0) * Double($1.1)
                }
                guard dot.isFinite, abs(dot) <= tolerance else { return false }
            }
        }
        return true
    }

    private static func tensorKey(layer: Int, direction: Int) -> String {
        String(format: "layer_%03d.direction_%03d", layer, direction)
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        value.count == 64
            && value.utf8.allSatisfy {
                (0x30 ... 0x39).contains($0) || (0x61 ... 0x66).contains($0)
            }
    }

    private static func parseTensorKey(_ key: String) -> (layer: Int, direction: Int)? {
        let pattern = /^layer_([0-9]{3})\.direction_([0-9]{3})$/
        guard let match = key.wholeMatch(of: pattern),
              let layer = Int(match.1), let direction = Int(match.2)
        else { return nil }
        return (layer, direction)
    }

    private static func canonicalJSON<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var data = try encoder.encode(value)
        data.append(0x0a)
        return data
    }

    private static func validateClosedManifestJSON(_ data: Data) throws {
        guard (try? ABSlayerStrictJSON.validateNoDuplicateKeys(data)) != nil,
              let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(root.keys) == ["format", "model", "dataset", "algorithm", "tensors", "runtime"],
              closedObject(root["model"], required: [
                  "identifier", "canonical_path", "metadata_sha256",
                  "weights_sha256", "decoder_layer_count", "hidden_size",
              ], optional: ["revision"]),
              closedObject(root["dataset"], required: [
                  "path", "sha256", "record_count",
              ], optional: ["prompt_count"]),
              closedObject(root["algorithm"], required: [
                  "name", "strength", "rank", "max_layer_fraction",
                  "selected_layers", "token_position",
              ]),
              closedObject(root["tensors"], required: ["file", "sha256", "schema"]),
              closedObject(root["runtime"], required: [
                  "backend", "max_sequence_length", "gpu_memory_utilization",
              ], optional: ["temporary_directory"])
        else { throw ABSlayerDirectionArtifactError.invalidManifest }
    }

    private static func stableFileData(
        _ url: URL, maximumBytes: Int
    ) throws -> Data {
        guard let identity = ABSlayerFileSystem.regularFileIdentity(url.path),
              identity.size >= 0, identity.size <= Int64(maximumBytes)
        else {
            throw ABSlayerDirectionArtifactError.missingArtifactFile(
                url.deletingLastPathComponent().path)
        }
        let mapped = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard Int64(mapped.count) == identity.size,
              ABSlayerFileSystem.regularFileIdentity(url.path) == identity
        else { throw ABSlayerDirectionArtifactError.artifactFileChanged(url.path) }
        let data = mapped.withUnsafeBytes { Data($0) }
        guard ABSlayerFileSystem.regularFileIdentity(url.path) == identity else {
            throw ABSlayerDirectionArtifactError.artifactFileChanged(url.path)
        }
        return data
    }

    private static func closedObject(
        _ value: Any?, required: Set<String>, optional: Set<String> = []
    ) -> Bool {
        guard let object = value as? [String: Any] else { return false }
        let keys = Set(object.keys)
        return required.isSubset(of: keys) && keys.isSubset(of: required.union(optional))
    }
}

public enum ABSlayerFileSystem {
    public struct RegularFileIdentity: Equatable, Sendable {
        public let device: UInt64
        public let inode: UInt64
        public let size: Int64
        public let modifiedSeconds: Int64
        public let modifiedNanoseconds: Int64
        public let changedSeconds: Int64
        public let changedNanoseconds: Int64
    }

    public static func pathExistsWithoutFollowingSymlink(_ path: String) -> Bool {
        var information = stat()
        return path.withCString { lstat($0, &information) } == 0
    }

    public static func isRegularFileWithoutFollowingSymlink(_ path: String) -> Bool {
        regularFileIdentity(path) != nil
    }

    public static func regularFileIdentity(
        _ path: String
    ) -> RegularFileIdentity? {
        var information = stat()
        guard path.withCString({ lstat($0, &information) }) == 0,
              (information.st_mode & S_IFMT) == S_IFREG
        else { return nil }
        return regularFileIdentity(information)
    }

    public static func regularFileIdentity(
        descriptor: Int32
    ) -> RegularFileIdentity? {
        var information = stat()
        guard fstat(descriptor, &information) == 0,
              (information.st_mode & S_IFMT) == S_IFREG
        else { return nil }
        return regularFileIdentity(information)
    }

    private static func regularFileIdentity(
        _ information: stat
    ) -> RegularFileIdentity {
        #if canImport(Darwin)
        let modifiedSeconds = Int64(information.st_mtimespec.tv_sec)
        let modifiedNanoseconds = Int64(information.st_mtimespec.tv_nsec)
        let changedSeconds = Int64(information.st_ctimespec.tv_sec)
        let changedNanoseconds = Int64(information.st_ctimespec.tv_nsec)
        #else
        let modifiedSeconds = Int64(information.st_mtim.tv_sec)
        let modifiedNanoseconds = Int64(information.st_mtim.tv_nsec)
        let changedSeconds = Int64(information.st_ctim.tv_sec)
        let changedNanoseconds = Int64(information.st_ctim.tv_nsec)
        #endif
        return RegularFileIdentity(
            device: UInt64(information.st_dev),
            inode: UInt64(information.st_ino),
            size: Int64(information.st_size),
            modifiedSeconds: modifiedSeconds,
            modifiedNanoseconds: modifiedNanoseconds,
            changedSeconds: changedSeconds,
            changedNanoseconds: changedNanoseconds)
    }

    public static func isDirectoryWithoutFollowingSymlink(_ path: String) -> Bool {
        var information = stat()
        guard path.withCString({ lstat($0, &information) }) == 0 else { return false }
        return (information.st_mode & S_IFMT) == S_IFDIR
    }

    public static func recursiveRegularFileBytes(at root: URL) throws -> Int64 {
        guard isDirectoryWithoutFollowingSymlink(root.path) else {
            throw ABSlayerDirectionArtifactError.invalidArtifactDirectory(root.path)
        }
        let keys: [URLResourceKey] = [
            .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
        ]
        guard let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants])
        else { throw ABSlayerDirectionArtifactError.cannotInspectFilesystem(root.path) }
        var total: Int64 = 0
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: Set(keys))
            guard values.isSymbolicLink != true else {
                throw ABSlayerDirectionArtifactError.symlinkInSource(url.path)
            }
            if values.isRegularFile == true {
                guard let size = values.fileSize, size >= 0,
                      total <= Int64.max - Int64(size)
                else { throw ABSlayerDirectionArtifactError.cannotInspectFilesystem(root.path) }
                total += Int64(size)
            }
        }
        return total
    }

    public static func availableCapacity(at directory: URL) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfFileSystem(
            forPath: directory.path)
        guard let number = attributes[.systemFreeSize] as? NSNumber else {
            throw ABSlayerDirectionArtifactError.cannotInspectFilesystem(directory.path)
        }
        return number.int64Value
    }
}

public final class ABSlayerProcessLock: @unchecked Sendable {
    private let descriptor: Int32

    public init(path: String = "/tmp/abslayer-mlx-backend.lock") throws {
        let descriptor = path.withCString {
            open(
                $0, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW,
                mode_t(S_IRUSR | S_IWUSR))
        }
        guard descriptor >= 0 else {
            throw ABSlayerDirectionArtifactError.processLockFailure(path)
        }
        var information = stat()
        guard fstat(descriptor, &information) == 0,
              (information.st_mode & S_IFMT) == S_IFREG,
              information.st_uid == geteuid()
        else {
            close(descriptor)
            throw ABSlayerDirectionArtifactError.processLockFailure(path)
        }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            throw ABSlayerDirectionArtifactError.processBusy(path)
        }
        self.descriptor = descriptor
    }

    deinit {
        _ = flock(descriptor, LOCK_UN)
        close(descriptor)
    }
}

public enum ABSlayerDirectionArtifactError: LocalizedError {
    case outputExists(String)
    case invalidArtifactDirectory(String)
    case invalidOutputParent(String)
    case missingArtifactFile(String)
    case invalidManifest
    case tensorHashMismatch
    case invalidTensorSchema(String)
    case incompleteTensorSet
    case cannotInspectFilesystem(String)
    case symlinkInSource(String)
    case artifactFileChanged(String)
    case processLockFailure(String)
    case processBusy(String)

    public var errorDescription: String? {
        switch self {
        case .outputExists(let path): "Output already exists; refusing to overwrite: \(path)"
        case .invalidArtifactDirectory(let path):
            "Direction artifact must be a regular non-symlink directory: \(path)"
        case .invalidOutputParent(let path):
            "Direction artifact parent must be a regular non-symlink directory: \(path)"
        case .missingArtifactFile(let path):
            "Direction artifact is missing a regular manifest or safetensors file: \(path)"
        case .invalidManifest: "Direction artifact manifest is invalid or incompatible."
        case .tensorHashMismatch: "Direction tensor checksum does not match the manifest."
        case .invalidTensorSchema(let key): "Direction tensor schema is invalid at \(key)."
        case .incompleteTensorSet: "Direction artifact does not contain every declared layer basis."
        case .cannotInspectFilesystem(let path): "Could not inspect filesystem capacity at \(path)."
        case .symlinkInSource(let path): "Model source contains a symbolic link: \(path)"
        case .artifactFileChanged(let path):
            "Direction artifact file changed while it was being read: \(path)"
        case .processLockFailure(let path): "Could not open backend process lock: \(path)"
        case .processBusy(let path): "Another ABSlayer model process holds \(path)."
        }
    }
}
