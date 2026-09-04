import Foundation

public enum WeightNormalization: String, CaseIterable, Sendable, Codable {
    case none
    case pre
    case full
}

/// How multiple refusal directions are combined into a weight edit.
///
/// `simultaneous` preserves ABSlayer's original summed-projector behavior,
/// `W' = W - s RᵀRW`. `sequential` applies each unit direction to the
/// result of the preceding projection, which is the composition used by SOM
/// multidirectional refusal ablation for deliberately non-orthogonal axes.
public enum AblationComposition: String, CaseIterable, Sendable, Codable {
    case simultaneous
    case sequential
}

public struct LayerAblationKernel: Sendable, Equatable {
    public var maximum: Float
    public var peakLayer: Float
    public var minimum: Float
    public var radius: Float

    public init(maximum: Float, peakLayer: Float, minimum: Float, radius: Float) {
        self.maximum = maximum
        self.peakLayer = peakLayer
        self.minimum = minimum
        self.radius = radius
    }

    public func weight(at layer: Int) -> Float {
        guard radius > 0 else { return Float(layer) == peakLayer ? maximum : 0 }
        let distance = abs(Float(layer) - peakLayer)
        guard distance <= radius else { return 0 }
        return maximum - (maximum - minimum) * distance / radius
    }
}

public enum AbliterationMath {
    public static func winsorized(_ values: [Float], quantile: Float?) -> [Float] {
        guard let quantile, quantile > 0, quantile < 1, !values.isEmpty else { return values }
        let magnitudes = values.map { abs($0) }.sorted()
        let index = min(magnitudes.count - 1, Int(Float(magnitudes.count - 1) * quantile))
        let limit = magnitudes[index]
        return values.map { min(max($0, -limit), limit) }
    }

    public static func mean(_ vectors: [[Float]]) -> [Float] {
        guard let first = vectors.first else { return [] }
        var result = Array(repeating: Float.zero, count: first.count)
        for vector in vectors {
            precondition(vector.count == result.count)
            for index in result.indices { result[index] += vector[index] }
        }
        let scale = 1 / Float(vectors.count)
        return result.map { $0 * scale }
    }

    public static func normalized(_ vector: [Float]) -> [Float] {
        let norm = sqrt(vector.reduce(Float.zero) { $0 + $1 * $1 })
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }

    /// Refusal direction from bad/control centroids. With projection enabled, removes
    /// the component parallel to the benign centroid before re-normalizing.
    public static func direction(
        contrast: [[Float]],
        control: [[Float]],
        projectAwayFromControl: Bool = true,
        winsorQuantile: Float? = nil
    ) -> [Float] {
        let bad = mean(contrast.map { winsorized($0, quantile: winsorQuantile) })
        let good = mean(control.map { winsorized($0, quantile: winsorQuantile) })
        var result = zip(normalized(bad), normalized(good)).map(-)
        if projectAwayFromControl {
            let goodDirection = normalized(good)
            let component = zip(result, goodDirection).reduce(Float.zero) { $0 + $1.0 * $1.1 }
            result = zip(result, goodDirection).map { $0 - component * $1 }
        }
        return normalized(result)
    }

    /// Linear interpolation supports Heretic-style fractional global layer selection.
    public static func interpolatedDirection(_ directions: [[Float]], layer: Float) -> [Float] {
        precondition(!directions.isEmpty)
        let clamped = min(max(layer, 0), Float(directions.count - 1))
        let lower = Int(floor(clamped))
        let upper = Int(ceil(clamped))
        guard lower != upper else { return directions[lower] }
        let fraction = clamped - Float(lower)
        return normalized(zip(directions[lower], directions[upper]).map {
            $0 * (1 - fraction) + $1 * fraction
        })
    }

    /// Interpolates between one shared global direction and the local layer direction.
    public static func blendedDirection(
        _ directions: [[Float]], globalLayer: Float, localLayer: Int,
        perLayerFraction: Float
    ) -> [Float] {
        let global = interpolatedDirection(directions, layer: globalLayer)
        let local = directions[min(max(localLayer, 0), directions.count - 1)]
        let fraction = min(max(perLayerFraction, 0), 1)
        return normalized(zip(global, local).map {
            $0 * (1 - fraction) + $1 * fraction
        })
    }

    /// Interpolates corresponding orthonormal refusal features between layers.
    /// Basis-vector signs are aligned before interpolation because SVD vectors
    /// are equivalent under sign inversion.
    public static func interpolatedSubspace(
        _ subspaces: [[[Float]]], layer: Float
    ) -> [[Float]] {
        precondition(!subspaces.isEmpty)
        let clamped = min(max(layer, 0), Float(subspaces.count - 1))
        let lower = Int(floor(clamped))
        let upper = Int(ceil(clamped))
        guard lower != upper else { return orthonormalized(subspaces[lower]) }
        let fraction = clamped - Float(lower)
        let count = min(subspaces[lower].count, subspaces[upper].count)
        let candidates = (0 ..< count).map { index -> [Float] in
            let lhs = normalized(subspaces[lower][index])
            var rhs = normalized(subspaces[upper][index])
            if dot(lhs, rhs) < 0 { rhs = rhs.map(-) }
            return zip(lhs, rhs).map { $0 * (1 - fraction) + $1 * fraction }
        }
        return orthonormalized(candidates)
    }

    /// Interpolates corresponding directions without Gram-Schmidt. This is the
    /// ordered/non-orthogonal counterpart to `interpolatedSubspace`, intended
    /// for sequential SOM projection composition. Integer source layers return
    /// the same direction order with each vector independently normalized.
    public static func interpolatedOrderedDirections(
        _ subspaces: [[[Float]]], layer: Float
    ) -> [[Float]] {
        precondition(!subspaces.isEmpty)
        let clamped = min(max(layer, 0), Float(subspaces.count - 1))
        let lower = Int(floor(clamped))
        let upper = Int(ceil(clamped))
        guard lower != upper else { return subspaces[lower].map(normalized) }
        let fraction = clamped - Float(lower)
        let count = min(subspaces[lower].count, subspaces[upper].count)
        return (0 ..< count).map { index -> [Float] in
            let lhs = normalized(subspaces[lower][index])
            var rhs = normalized(subspaces[upper][index])
            if dot(lhs, rhs) < 0 { rhs = rhs.map(-) }
            return normalized(zip(lhs, rhs).map {
                $0 * (1 - fraction) + $1 * fraction
            })
        }
    }

    /// Interpolates the complete global and local refusal subspaces. Rank 1 is
    /// exactly the ordinary blended-direction operation; higher ranks retain
    /// independent cyber-refusal features instead of collapsing them to one axis.
    public static func blendedSubspace(
        _ subspaces: [[[Float]]], globalLayer: Float, localLayer: Int,
        perLayerFraction: Float
    ) -> [[Float]] {
        precondition(!subspaces.isEmpty)
        let global = interpolatedSubspace(subspaces, layer: globalLayer)
        let local = orthonormalized(
            subspaces[min(max(localLayer, 0), subspaces.count - 1)])
        let fraction = min(max(perLayerFraction, 0), 1)
        let count = min(global.count, local.count)
        let candidates = (0 ..< count).map { index -> [Float] in
            let lhs = global[index]
            var rhs = local[index]
            if dot(lhs, rhs) < 0 { rhs = rhs.map(-) }
            return zip(lhs, rhs).map { $0 * (1 - fraction) + $1 * fraction }
        }
        return orthonormalized(candidates)
    }

    /// Blends corresponding global and local directions while retaining their
    /// order and non-orthogonality. This deliberately performs no basis
    /// orthogonalization and is therefore suitable for sequential SOM edits.
    public static func blendedOrderedDirections(
        _ subspaces: [[[Float]]], globalLayer: Float, localLayer: Int,
        perLayerFraction: Float
    ) -> [[Float]] {
        precondition(!subspaces.isEmpty)
        let global = interpolatedOrderedDirections(subspaces, layer: globalLayer)
        let local = subspaces[min(max(localLayer, 0), subspaces.count - 1)].map(normalized)
        let fraction = min(max(perLayerFraction, 0), 1)
        if fraction == 0 { return global }
        if fraction == 1 { return local }
        let count = min(global.count, local.count)
        return (0 ..< count).map { index -> [Float] in
            let lhs = global[index]
            var rhs = local[index]
            if dot(lhs, rhs) < 0 { rhs = rhs.map(-) }
            return normalized(zip(lhs, rhs).map {
                $0 * (1 - fraction) + $1 * fraction
            })
        }
    }

    /// Resolves the directions used at a target layer. This is the shared
    /// adapter/BF16 source-of-truth and is also a clean API for callers that
    /// author a single measured source-layer basis across many target layers.
    /// Sequential multi-direction scopes preserve direction order and
    /// non-orthogonality; simultaneous scopes retain historical behavior.
    public static func resolvedBasis(
        directions: [[Float]], subspaces: [[[Float]]]? = nil,
        scope: DirectionScope, targetLayer: Int,
        composition: AblationComposition
    ) -> [[Float]] {
        switch scope {
        case .perLayer:
            return subspaces?[targetLayer] ?? [directions[targetLayer]]
        case .global(let layer):
            if let subspaces {
                return composition == .sequential
                    ? interpolatedOrderedDirections(subspaces, layer: layer)
                    : interpolatedSubspace(subspaces, layer: layer)
            }
            return [interpolatedDirection(directions, layer: layer)]
        case let .blended(globalLayer, fraction):
            if let subspaces {
                return composition == .sequential
                    ? blendedOrderedDirections(
                        subspaces, globalLayer: globalLayer,
                        localLayer: targetLayer, perLayerFraction: fraction)
                    : blendedSubspace(
                        subspaces, globalLayer: globalLayer,
                        localLayer: targetLayer, perLayerFraction: fraction)
            }
            return [blendedDirection(
                directions, globalLayer: globalLayer, localLayer: targetLayer,
                perLayerFraction: fraction)]
        }
    }

    public static func orthonormalized(_ vectors: [[Float]]) -> [[Float]] {
        var basis = [[Float]]()
        for vector in vectors {
            var candidate = vector
            for existing in basis {
                let component = dot(candidate, existing)
                candidate = zip(candidate, existing).map { $0 - component * $1 }
            }
            let unit = normalized(candidate)
            if dot(unit, unit) > 0.99 { basis.append(unit) }
        }
        return basis
    }

    private static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { $0 + $1.0 * $1.1 }
    }

    /// Applies W' = W - strength * v(vᵀW) to a row-major [output, input] matrix.
    public static func edit(
        matrix: [Float], rows: Int, columns: Int,
        direction: [Float], strength: Float,
        normalization: WeightNormalization = .none
    ) -> [Float] {
        edit(
            matrix: matrix, rows: rows, columns: columns,
            directions: [direction], strength: strength,
            normalization: normalization, composition: .simultaneous)
    }

    /// Applies either the original summed projector or an ordered composition
    /// of projectors to a row-major [output, input] matrix. Sequential vectors
    /// are normalized independently and deliberately are not orthogonalized.
    public static func edit(
        matrix: [Float], rows: Int, columns: Int,
        directions: [[Float]], strength: Float,
        normalization: WeightNormalization = .none,
        composition: AblationComposition = .simultaneous
    ) -> [Float] {
        precondition(
            matrix.count == rows * columns
                && !directions.isEmpty
                && directions.allSatisfy { $0.count == rows })
        let originalNorms = rowNorms(matrix, rows: rows, columns: columns)
        var working = matrix
        if normalization != .none {
            for row in 0 ..< rows where originalNorms[row] > 0 {
                for column in 0 ..< columns {
                    working[row * columns + column] /= originalNorms[row]
                }
            }
        }

        switch composition {
        case .simultaneous:
            // Keep the historical behavior exactly: callers traditionally
            // supplied an orthonormal basis, so no normalization occurs here.
            let source = working
            for direction in directions {
                let projected = projection(
                    matrix: source, rows: rows, columns: columns,
                    direction: direction)
                subtractProjection(
                    from: &working, rows: rows, columns: columns,
                    direction: direction, projected: projected, strength: strength)
            }
        case .sequential:
            for rawDirection in directions {
                let direction = normalized(rawDirection)
                let projected = projection(
                    matrix: working, rows: rows, columns: columns,
                    direction: direction)
                subtractProjection(
                    from: &working, rows: rows, columns: columns,
                    direction: direction, projected: projected, strength: strength)
            }
        }

        if normalization == .pre {
            for row in 0 ..< rows {
                for column in 0 ..< columns {
                    working[row * columns + column] *= originalNorms[row]
                }
            }
        } else if normalization == .full {
            let editedNorms = rowNorms(working, rows: rows, columns: columns)
            for row in 0 ..< rows where editedNorms[row] > 0 {
                let scale = originalNorms[row] / editedNorms[row]
                for column in 0 ..< columns { working[row * columns + column] *= scale }
            }
        }
        return working
    }

    private static func projection(
        matrix: [Float], rows: Int, columns: Int, direction: [Float]
    ) -> [Float] {
        var projected = Array(repeating: Float.zero, count: columns)
        for column in 0 ..< columns {
            for row in 0 ..< rows {
                projected[column] += direction[row] * matrix[row * columns + column]
            }
        }
        return projected
    }

    private static func subtractProjection(
        from matrix: inout [Float], rows: Int, columns: Int,
        direction: [Float], projected: [Float], strength: Float
    ) {
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                matrix[row * columns + column] -=
                    strength * direction[row] * projected[column]
            }
        }
    }

    private static func rowNorms(_ matrix: [Float], rows: Int, columns: Int) -> [Float] {
        (0 ..< rows).map { row in
            sqrt((0 ..< columns).reduce(Float.zero) {
                let value = matrix[row * columns + $1]
                return $0 + value * value
            })
        }
    }
}
