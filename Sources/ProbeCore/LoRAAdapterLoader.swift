import Foundation
import MLXLMCommon

/// Adapter settings shared by command-line entry points. Parsing lives in
/// ProbeCore so every caller applies the same strict environment validation
/// before loading a model.
public struct AdapterRuntimeOptions: Equatable, Sendable {
    public let directory: String?
    public let scaleOverride: Float?

    public init(directory: String?, scaleOverride: Float?) {
        self.directory = directory
        self.scaleOverride = scaleOverride
    }

    public static func parse(environment: [String: String]) throws -> Self {
        let rawDirectory = environment["ABSLAYER_ADAPTER_DIR"]
        let directory = rawDirectory.flatMap { value in
            value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : value
        }
        guard let rawScale = environment["ABSLAYER_ADAPTER_SCALE"] else {
            return Self(directory: directory, scaleOverride: nil)
        }

        let normalizedScale = rawScale.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let scale = Float(normalizedScale), scale.isFinite, scale >= 0 else {
            throw AdapterRuntimeOptionsError.invalidScale(rawScale)
        }
        guard directory != nil else {
            throw AdapterRuntimeOptionsError.scaleRequiresAdapterDirectory
        }
        return Self(directory: directory, scaleOverride: scale)
    }
}

public enum AdapterRuntimeOptionsError: LocalizedError, Equatable {
    case invalidScale(String)
    case scaleRequiresAdapterDirectory

    public var errorDescription: String? {
        switch self {
        case .invalidScale(let value):
            "ABSLAYER_ADAPTER_SCALE must be a finite, non-negative number, not '\(value)'."
        case .scaleRequiresAdapterDirectory:
            "ABSLAYER_ADAPTER_SCALE requires ABSLAYER_ADAPTER_DIR to be set."
        }
    }
}

/// Loads an adapter without mutating its directory and optionally substitutes
/// only the inference-time LoRA scale. This keeps scale sweeps reproducible and
/// avoids copied or mismatched sidecar configurations.
public enum LoRAAdapterLoader {
    public static func load(
        directory: String, scaleOverride: Float? = nil
    ) throws -> LoRAContainer {
        let url = URL(fileURLWithPath: directory).standardizedFileURL
        let original = try LoRAContainer.from(directory: url)
        guard let scaleOverride else { return original }
        guard scaleOverride >= 0, scaleOverride.isFinite else {
            throw LoRAAdapterLoaderError.invalidScale(scaleOverride)
        }
        let source = original.configuration
        let parameters = source.loraParameters
        let configuration = LoRAConfiguration(
            numLayers: source.numLayers,
            fineTuneType: source.fineTuneType,
            loraParameters: LoRAConfiguration.LoRAParameters(
                rank: parameters.rank,
                scale: scaleOverride,
                dropout: parameters.dropout,
                keys: parameters.keys))
        return LoRAContainer(
            configuration: configuration,
            parameters: original.parameters)
    }
}

public enum LoRAAdapterLoaderError: LocalizedError {
    case invalidScale(Float)

    public var errorDescription: String? {
        switch self {
        case .invalidScale(let value):
            "ABSLAYER_ADAPTER_SCALE must be finite and non-negative, not \(value)."
        }
    }
}
