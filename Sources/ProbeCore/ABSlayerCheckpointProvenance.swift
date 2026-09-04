#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
#if canImport(CryptoKit)
import CryptoKit
#elseif canImport(Crypto)
import Crypto
#endif
import Foundation

/// Cryptographic identity for a local checkpoint. Weight identity is intentionally
/// expensive: it covers the exact index bytes and every referenced shard byte.
public enum ABSlayerCheckpointProvenance {
    private static let indexName = "model.safetensors.index.json"
    private static let markerName = "abslayer.json"
    private static let maximumIndexBytes = 64 * 1_024 * 1_024
    private static let maximumMetadataFileBytes = 512 * 1_024 * 1_024
    private static let streamBufferBytes = 8 * 1_024 * 1_024

    public static func metadataSHA256(directory: String) throws -> String {
        let root = try validatedDirectory(directory)
        let files = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])
            .filter {
                $0.lastPathComponent != markerName
                    && $0.pathExtension.lowercased() != "safetensors"
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw ABSlayerCheckpointProvenanceError.missingMetadata(root.path)
        }
        var hasher = SHA256()
        // v2 matches the editor's copy policy: every non-hidden top-level
        // non-weight file is bound, except the backend-owned candidate marker.
        update(&hasher, bytes: Data("abslayer.metadata/v2\0".utf8))
        for file in files {
            let identity = try validatedIdentity(
                file.path, maximumBytes: Int64(maximumMetadataFileBytes))
            updateFramed(&hasher, bytes: Data(file.lastPathComponent.utf8))
            updateLength(&hasher, UInt64(identity.size))
            try streamStableFile(file.path, expected: identity) { chunk in
                hasher.update(data: chunk)
            }
        }
        return hex(hasher.finalize())
    }

    public static func weightsSHA256(directory: String) throws -> String {
        let root = try validatedDirectory(directory)
        let indexURL = root.appendingPathComponent(indexName)
        let indexIdentity = try validatedIdentity(
            indexURL.path, maximumBytes: Int64(maximumIndexBytes))
        let indexData = try readStableFile(
            indexURL.path, expected: indexIdentity,
            maximumBytes: maximumIndexBytes)
        let shards = try referencedShards(indexData)

        var hasher = SHA256()
        update(&hasher, bytes: Data("abslayer.weights/v1\0".utf8))
        updateFramed(&hasher, bytes: Data(indexName.utf8))
        updateLength(&hasher, UInt64(indexIdentity.size))
        update(&hasher, bytes: indexData)
        for name in shards {
            let path = root.appendingPathComponent(name).path
            let identity = try validatedIdentity(path, maximumBytes: nil)
            updateFramed(&hasher, bytes: Data(name.utf8))
            updateLength(&hasher, UInt64(identity.size))
            try streamStableFile(path, expected: identity) { chunk in
                hasher.update(data: chunk)
            }
        }
        return hex(hasher.finalize())
    }

    private static func validatedDirectory(_ directory: String) throws -> URL {
        let root = URL(fileURLWithPath: directory).standardizedFileURL
        guard ABSlayerFileSystem.isDirectoryWithoutFollowingSymlink(root.path) else {
            throw ABSlayerCheckpointProvenanceError.invalidDirectory(root.path)
        }
        return root
    }

    private static func referencedShards(_ indexData: Data) throws -> [String] {
        guard (try? ABSlayerStrictJSON.validateNoDuplicateKeys(indexData)) != nil,
              let root = try? JSONSerialization.jsonObject(with: indexData)
                  as? [String: Any],
              let map = root["weight_map"] as? [String: String],
              !map.isEmpty
        else { throw ABSlayerCheckpointProvenanceError.invalidIndex }
        let shards = Set(map.values)
        guard !shards.isEmpty,
              shards.allSatisfy({ name in
                  !name.isEmpty
                      && name == URL(fileURLWithPath: name).lastPathComponent
                      && name.hasSuffix(".safetensors")
                      && name != "." && name != ".."
              })
        else { throw ABSlayerCheckpointProvenanceError.invalidIndex }
        return shards.sorted()
    }

    private static func validatedIdentity(
        _ path: String, maximumBytes: Int64?
    ) throws -> ABSlayerFileSystem.RegularFileIdentity {
        guard let identity = ABSlayerFileSystem.regularFileIdentity(path),
              identity.size >= 0,
              maximumBytes.map({ identity.size <= $0 }) ?? true
        else { throw ABSlayerCheckpointProvenanceError.invalidFile(path) }
        return identity
    }

    private static func readStableFile(
        _ path: String, expected: ABSlayerFileSystem.RegularFileIdentity,
        maximumBytes: Int
    ) throws -> Data {
        guard expected.size <= Int64(maximumBytes) else {
            throw ABSlayerCheckpointProvenanceError.invalidFile(path)
        }
        var result = Data()
        result.reserveCapacity(Int(expected.size))
        try streamStableFile(path, expected: expected) { result.append($0) }
        guard result.count == Int(expected.size) else {
            throw ABSlayerCheckpointProvenanceError.fileChanged(path)
        }
        return result
    }

    private static func streamStableFile(
        _ path: String, expected: ABSlayerFileSystem.RegularFileIdentity,
        consume: (Data) throws -> Void
    ) throws {
        let descriptor = path.withCString {
            open($0, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        }
        guard descriptor >= 0 else {
            throw ABSlayerCheckpointProvenanceError.invalidFile(path)
        }
        defer { close(descriptor) }
        guard ABSlayerFileSystem.regularFileIdentity(descriptor: descriptor) == expected else {
            throw ABSlayerCheckpointProvenanceError.fileChanged(path)
        }
        var total: Int64 = 0
        var buffer = [UInt8](repeating: 0, count: streamBufferBytes)
        while true {
            let count = buffer.withUnsafeMutableBytes { bytes in
                read(descriptor, bytes.baseAddress, bytes.count)
            }
            guard count >= 0 else {
                throw ABSlayerCheckpointProvenanceError.cannotRead(path)
            }
            if count == 0 { break }
            guard total <= Int64.max - Int64(count) else {
                throw ABSlayerCheckpointProvenanceError.invalidFile(path)
            }
            total += Int64(count)
            try consume(Data(buffer[0 ..< count]))
        }
        guard total == expected.size,
              ABSlayerFileSystem.regularFileIdentity(descriptor: descriptor) == expected,
              ABSlayerFileSystem.regularFileIdentity(path) == expected
        else { throw ABSlayerCheckpointProvenanceError.fileChanged(path) }
    }

    private static func update(_ hasher: inout SHA256, bytes: Data) {
        hasher.update(data: bytes)
    }

    private static func updateFramed(_ hasher: inout SHA256, bytes: Data) {
        updateLength(&hasher, UInt64(bytes.count))
        update(&hasher, bytes: bytes)
    }

    private static func updateLength(_ hasher: inout SHA256, _ value: UInt64) {
        var bigEndian = value.bigEndian
        withUnsafeBytes(of: &bigEndian) { hasher.update(bufferPointer: $0) }
    }

    private static func hex<D: Digest>(_ digest: D) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}

public enum ABSlayerCheckpointProvenanceError: LocalizedError, Equatable {
    case invalidDirectory(String)
    case missingMetadata(String)
    case invalidIndex
    case invalidFile(String)
    case cannotRead(String)
    case fileChanged(String)

    public var errorDescription: String? {
        switch self {
        case .invalidDirectory(let path):
            "Checkpoint must be a regular non-symlink directory: \(path)"
        case .missingMetadata(let path):
            "Checkpoint has no bindable metadata files: \(path)"
        case .invalidIndex:
            "Checkpoint safetensors index is missing, malformed, or unsafe."
        case .invalidFile(let path):
            "Checkpoint binding requires a regular bounded file: \(path)"
        case .cannotRead(let path):
            "Checkpoint file could not be read while binding: \(path)"
        case .fileChanged(let path):
            "Checkpoint file changed while its cryptographic identity was computed: \(path)"
        }
    }
}
