import Foundation
import MLX
import MLXLMCommon

public enum LoRAAdapterPersistence {
    /// Publishes a complete adapter directory in one same-filesystem rename.
    public static func write(
        _ adapter: LoRAContainer,
        to path: String,
        additionalFiles: [String: Data] = [:]
    ) throws {
        let destination = URL(fileURLWithPath: path).standardizedFileURL
        let manager = FileManager.default
        guard !manager.fileExists(atPath: destination.path) else {
            throw LoRAAdapterPersistenceError.outputExists(destination.path)
        }
        let parent = destination.deletingLastPathComponent()
        try manager.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).staging-\(UUID().uuidString)",
            isDirectory: true)
        try manager.createDirectory(at: staging, withIntermediateDirectories: false)
        do {
            let arrays = Dictionary(uniqueKeysWithValues: adapter.parameters.flattened())
            guard !arrays.isEmpty else { throw LoRAAdapterPersistenceError.emptyAdapter }
            eval(Array(arrays.values))
            try MLX.save(
                arrays: arrays,
                url: staging.appendingPathComponent("adapters.safetensors"))
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(adapter.configuration).write(
                to: staging.appendingPathComponent("adapter_config.json"),
                options: .atomic)
            for name in additionalFiles.keys.sorted() {
                try validateAdditionalFileName(name)
                try additionalFiles[name]?.write(
                    to: staging.appendingPathComponent(name), options: .atomic)
            }
            try manager.moveItem(at: staging, to: destination)
        } catch {
            try? manager.removeItem(at: staging)
            throw error
        }
    }

    static func validateAdditionalFileName(_ name: String) throws {
        let reserved = Set(["adapter_config.json", "adapters.safetensors"])
        let lastComponent = URL(fileURLWithPath: name).lastPathComponent
        guard !name.isEmpty, name != ".", name != "..",
              lastComponent == name, !reserved.contains(name)
        else { throw LoRAAdapterPersistenceError.invalidAdditionalFileName(name) }
    }
}

public enum LoRAAdapterPersistenceError: LocalizedError, Equatable {
    case outputExists(String)
    case emptyAdapter
    case invalidAdditionalFileName(String)

    public var errorDescription: String? {
        switch self {
        case .outputExists(let path): "Adapter output already exists: \(path)"
        case .emptyAdapter: "Refusing to publish an adapter with no tensors."
        case .invalidAdditionalFileName(let name):
            "Invalid or reserved additional adapter filename: \(name)"
        }
    }
}
