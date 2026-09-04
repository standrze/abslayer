import Foundation

/// Canonical decoder-layer indexing and depth-aware candidate selection.
///
/// Public configuration and persisted artifacts use zero-based indices. A few
/// residual-tap APIs inherited from MLX report layers one-based; conversions to
/// that representation are kept explicit at the runtime boundary.
public enum DecoderLayerSelection {
    public struct RankedLayer: Equatable, Sendable {
        public let zeroBasedIndex: Int
        public let priority: Double

        public init(zeroBasedIndex: Int, priority: Double) {
            self.zeroBasedIndex = zeroBasedIndex
            self.priority = priority
        }
    }

    public static func fullStackZeroBased(layerCount: Int) throws -> [Int] {
        guard layerCount > 0 else {
            throw DecoderLayerSelectionError.invalidLayerCount(layerCount)
        }
        return Array(0 ..< layerCount)
    }

    /// Validates canonical zero-based indices without silently dropping,
    /// reordering, or deduplicating an explicit user selection.
    public static func validateZeroBased(
        _ layers: [Int], layerCount: Int
    ) throws -> [Int] {
        guard layerCount > 0 else {
            throw DecoderLayerSelectionError.invalidLayerCount(layerCount)
        }
        guard !layers.isEmpty else {
            throw DecoderLayerSelectionError.emptySelection
        }
        var seen = Set<Int>()
        for layer in layers {
            guard (0 ..< layerCount).contains(layer) else {
                throw DecoderLayerSelectionError.zeroBasedLayerOutsideModel(
                    requested: layer, layerCount: layerCount)
            }
            guard seen.insert(layer).inserted else {
                throw DecoderLayerSelectionError.duplicateZeroBasedLayer(layer)
            }
        }
        return layers
    }

    /// Compatibility conversion for older APIs that exposed one-based layer
    /// numbers. New environment variables and artifacts should remain
    /// zero-based.
    public static func zeroBasedFromLegacyOneBased(
        _ layers: [Int], layerCount: Int
    ) throws -> [Int] {
        guard layerCount > 0 else {
            throw DecoderLayerSelectionError.invalidLayerCount(layerCount)
        }
        guard !layers.isEmpty else {
            throw DecoderLayerSelectionError.emptySelection
        }
        for layer in layers where !(1 ... layerCount).contains(layer) {
            throw DecoderLayerSelectionError.oneBasedLayerOutsideModel(
                requested: layer, layerCount: layerCount)
        }
        return try validateZeroBased(layers.map { $0 - 1 }, layerCount: layerCount)
    }

    public static func runtimeOneBased(
        fromZeroBased layers: [Int], layerCount: Int
    ) throws -> [Int] {
        try validateZeroBased(layers, layerCount: layerCount).map { $0 + 1 }
    }

    /// Selects candidates from the complete measured stack while preserving a
    /// path through every depth band. Ranking determines the winner *within*
    /// each band; no architecture-specific window or preferred depth is used.
    public static func depthDistributedZeroBased(
        rankedLayers: [RankedLayer], maximum: Int, layerCount: Int
    ) throws -> [Int] {
        guard maximum > 0 else {
            throw DecoderLayerSelectionError.invalidMaximum(maximum)
        }
        guard layerCount > 0 else {
            throw DecoderLayerSelectionError.invalidLayerCount(layerCount)
        }
        guard !rankedLayers.isEmpty else { return [] }

        let validated = try validateZeroBased(
            rankedLayers.map(\.zeroBasedIndex), layerCount: layerCount)
        let byIndex = Dictionary(uniqueKeysWithValues:
            zip(validated, rankedLayers.map(\.priority)))
        let ranked = validated.sorted { lhs, rhs in
            let left = byIndex[lhs] ?? -.infinity
            let right = byIndex[rhs] ?? -.infinity
            if left != right { return left > right }
            return lhs < rhs
        }
        let targetCount = min(maximum, ranked.count, layerCount)
        var selected = [Int]()
        var seen = Set<Int>()

        for band in 0 ..< targetCount {
            let lower = band * layerCount / targetCount
            let upperExclusive = (band + 1) * layerCount / targetCount
            if let best = ranked.first(where: {
                $0 >= lower && $0 < upperExclusive
            }), seen.insert(best).inserted {
                selected.append(best)
            }
        }
        for layer in ranked where selected.count < targetCount {
            if seen.insert(layer).inserted { selected.append(layer) }
        }
        return selected
    }
}

public enum DecoderLayerSelectionError: LocalizedError, Equatable, Sendable {
    case invalidLayerCount(Int)
    case emptySelection
    case invalidMaximum(Int)
    case zeroBasedLayerOutsideModel(requested: Int, layerCount: Int)
    case oneBasedLayerOutsideModel(requested: Int, layerCount: Int)
    case duplicateZeroBasedLayer(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidLayerCount(let count):
            "Decoder-layer count must be positive, not \(count)."
        case .emptySelection:
            "An explicit decoder-layer selection cannot be empty."
        case .invalidMaximum(let maximum):
            "Maximum decoder-layer candidates must be positive, not \(maximum)."
        case .zeroBasedLayerOutsideModel(let requested, let count):
            "Zero-based decoder layer \(requested) is outside the model's 0..<\(count) stack."
        case .oneBasedLayerOutsideModel(let requested, let count):
            "Legacy one-based decoder layer \(requested) is outside the model's 1...\(count) stack."
        case .duplicateZeroBasedLayer(let layer):
            "Zero-based decoder layer \(layer) was selected more than once."
        }
    }
}
