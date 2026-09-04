import Foundation

/// Resolves MLX's `suffix(numLayers)` LoRA selection into explicit zero-based
/// bounds while rejecting counts that would otherwise silently clamp.
public struct LoRASuffixLayerSelection: Equatable, Sendable {
    public let requestedCount: Int
    public let availableCount: Int
    public let startIndex: Int
    public let endIndex: Int

    public var range: Range<Int> { startIndex ..< endIndex }

    public static func resolve(
        requestedCount: Int, availableCount: Int
    ) throws -> Self {
        guard requestedCount > 0 else {
            throw LoRASuffixLayerSelectionError.invalidRequestedCount(requestedCount)
        }
        guard availableCount > 0 else {
            throw LoRASuffixLayerSelectionError.modelHasNoLayers
        }
        guard requestedCount <= availableCount else {
            throw LoRASuffixLayerSelectionError.exceedsAvailableLayers(
                requested: requestedCount, available: availableCount)
        }
        return Self(
            requestedCount: requestedCount,
            availableCount: availableCount,
            startIndex: availableCount - requestedCount,
            endIndex: availableCount)
    }
}

public enum LoRASuffixLayerSelectionError: LocalizedError, Equatable {
    case invalidRequestedCount(Int)
    case modelHasNoLayers
    case exceedsAvailableLayers(requested: Int, available: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidRequestedCount(let requested):
            "ABSLAYER_LORA_LAYERS must be positive, not \(requested)."
        case .modelHasNoLayers:
            "The model does not expose any LoRA-capable decoder layers."
        case .exceedsAvailableLayers(let requested, let available):
            "ABSLAYER_LORA_LAYERS requests \(requested) layers, but the model exposes only \(available)."
        }
    }
}
