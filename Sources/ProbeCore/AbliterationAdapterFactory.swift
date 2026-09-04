import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Builds an inference-only LoRA/QLoRA adapter whose low-rank branch represents
/// the same projection used by the BF16 editor. This is the lightweight trial
/// path: the quantized model stays resident and adapters are loaded/unloaded.
public enum AbliterationAdapterFactory {
    public static func make(
        model: LanguageModel,
        directions: [[Float]],
        subspaces: [[[Float]]]? = nil,
        configuration: AbliterationConfiguration,
        fullNormalizationRank: Int = 3
    ) throws -> LoRAContainer {
        guard let loraModel = model as? LoRAModel else {
            throw AbliterationAdapterError.incompatibleModel
        }
        guard !directions.isEmpty else { throw EditorError.noDirections }
        MLXRandom.seed(0xAB51_A9E2)

        let measuredRank = max(1, subspaces?.map(\.count).max() ?? 1)
        let rank: Int
        switch configuration.normalization {
        case .full:
            rank = max(measuredRank, fullNormalizationRank)
        case .none, .pre:
            switch configuration.directionScope {
            case .global, .blended, .perLayer: rank = measuredRank
            }
        }

        var parameters = [String: MLXArray]()
        var targetLayers = Set<Int>()
        var includesAttention = false
        var includesMLP = false
        let modules = Dictionary(uniqueKeysWithValues: model.namedModules())
        for (path, module) in modules {
            guard let target = target(for: path), target.layer < directions.count,
                  let linear = module as? Linear
            else { continue }
            let strength = target.component == .attention
                ? configuration.attention.weight(at: target.layer)
                : configuration.mlp.weight(at: target.layer)
            guard strength != 0 else { continue }

            let basis = AbliterationMath.resolvedBasis(
                directions: directions, subspaces: subspaces,
                scope: configuration.directionScope, targetLayer: target.layer,
                composition: configuration.composition)
            let weight = dequantizedWeight(linear).asType(.float32)
            guard let width = basis.first?.count, width == weight.dim(0) else {
                throw EditorError.directionShape(
                    key: path, expected: weight.dim(0), actual: basis.first?.count ?? 0)
            }
            let factors = factors(
                weight: weight, basis: basis, strength: strength,
                normalization: configuration.normalization,
                composition: configuration.composition, adapterRank: rank)
            eval(factors.a, factors.b)
            parameters[path + ".lora_a"] = factors.a
            parameters[path + ".lora_b"] = factors.b
            targetLayers.insert(target.layer)
            switch target.component {
            case .attention: includesAttention = true
            case .mlp: includesMLP = true
            }
        }
        guard !parameters.isEmpty else { throw AbliterationAdapterError.noTargetLayers }
        let coverage = try sparseCoverage(
            layerCount: loraModel.loraLayers.count,
            targetLayers: targetLayers,
            includesAttention: includesAttention,
            includesMLP: includesMLP)
        eval(Array(parameters.values))
        return LoRAContainer(
            configuration: LoRAConfiguration(
                numLayers: coverage.numLayers,
                loraParameters: .init(
                    rank: rank, scale: 1, dropout: 0,
                    keys: coverage.keys)),
            parameters: ModuleParameters.unflattened(parameters))
    }

    /// Builds one exact, inference-only adapter from independent ordered bases
    /// at explicitly selected decoder layers.  Unlike `make`, no source basis
    /// is interpolated or broadcast: `basesByLayer[19]` edits only layer 19,
    /// `basesByLayer[20]` edits only layer 20, and so on.  This is the resident
    /// search path for combining SOM candidates measured at different layers.
    ///
    /// Every selected matrix is projected with strength one.  Runtime adapter
    /// scale must therefore remain one; callers tune the ordered bases rather
    /// than hiding an additional scalar in the loader.
    public static func makeLayerSpecific(
        model: LanguageModel,
        basesByLayer: [Int: [[Float]]],
        components: SOMApplicationComponents = .omlp,
        normalization: WeightNormalization = .none,
        composition: AblationComposition = .sequential
    ) throws -> LoRAContainer {
        guard let loraModel = model as? LoRAModel else {
            throw AbliterationAdapterError.incompatibleModel
        }
        guard !basesByLayer.isEmpty else { throw EditorError.noDirections }
        let layerCount = loraModel.loraLayers.count
        guard basesByLayer.keys.allSatisfy({ $0 >= 0 && $0 < layerCount }) else {
            let requested = basesByLayer.keys.first { $0 < 0 || $0 >= layerCount } ?? -1
            throw SOMSubsetSearchError.sourceLayerOutsideModel(
                requested: requested, layerCount: layerCount)
        }
        guard basesByLayer.values.allSatisfy({ basis in
            guard let width = basis.first?.count, width > 0 else { return false }
            return basis.allSatisfy { direction in
                direction.count == width
                    && direction.allSatisfy(\.isFinite)
                    && direction.contains(where: { $0 != 0 })
            }
        }) else { throw SOMSubsetSearchError.invalidCandidateDirections }

        let rank = basesByLayer.values.map(\.count).max() ?? 0
        guard rank > 0 else { throw EditorError.noDirections }
        MLXRandom.seed(0xAB51_A9E2)

        var parameters = [String: MLXArray]()
        var targetLayers = Set<Int>()
        var includesAttention = false
        var includesMLP = false
        for (path, module) in Dictionary(uniqueKeysWithValues: model.namedModules()) {
            guard let target = target(for: path),
                  let basis = basesByLayer[target.layer],
                  let linear = module as? Linear
            else { continue }
            switch target.component {
            case .attention where !components.includesAttention: continue
            case .mlp where !components.includesMLP: continue
            default: break
            }

            let weight = dequantizedWeight(linear).asType(.float32)
            guard basis.allSatisfy({ $0.count == weight.dim(0) }) else {
                throw EditorError.directionShape(
                    key: path, expected: weight.dim(0),
                    actual: basis.first?.count ?? 0)
            }
            let lowRank = factors(
                weight: weight, basis: basis, strength: 1,
                normalization: normalization, composition: composition,
                adapterRank: rank)
            eval(lowRank.a, lowRank.b)
            parameters[path + ".lora_a"] = lowRank.a
            parameters[path + ".lora_b"] = lowRank.b
            targetLayers.insert(target.layer)
            switch target.component {
            case .attention: includesAttention = true
            case .mlp: includesMLP = true
            }
        }
        guard !parameters.isEmpty else { throw AbliterationAdapterError.noTargetLayers }
        let coverage = try sparseCoverage(
            layerCount: layerCount, targetLayers: targetLayers,
            includesAttention: includesAttention, includesMLP: includesMLP)
        eval(Array(parameters.values))
        return LoRAContainer(
            configuration: LoRAConfiguration(
                numLayers: coverage.numLayers,
                loraParameters: .init(
                    rank: rank, scale: 1, dropout: 0, keys: coverage.keys)),
            parameters: ModuleParameters.unflattened(parameters))
    }

    static func sparseCoverage(
        layerCount: Int,
        targetLayers: Set<Int>,
        includesAttention: Bool,
        includesMLP: Bool
    ) throws -> (numLayers: Int, keys: [String]) {
        guard let earliest = targetLayers.min(),
              earliest >= 0, targetLayers.allSatisfy({ $0 < layerCount })
        else { throw AbliterationAdapterError.noTargetLayers }
        var keys = [String]()
        if includesAttention { keys.append("self_attn.o_proj") }
        if includesMLP { keys.append("mlp.down_proj") }
        guard !keys.isEmpty else { throw AbliterationAdapterError.noTargetLayers }
        return (layerCount - earliest, keys)
    }

    private enum Component { case attention, mlp }
    private struct Target { let layer: Int; let component: Component }

    private static func target(for path: String) -> Target? {
        guard let range = path.range(of: ".layers.") else { return nil }
        let suffix = path[range.upperBound...]
        let digits = suffix.prefix { $0.isNumber }
        guard let layer = Int(digits) else { return nil }
        if path.hasSuffix(".self_attn.o_proj") {
            return Target(layer: layer, component: .attention)
        }
        if path.hasSuffix(".mlp.down_proj") {
            return Target(layer: layer, component: .mlp)
        }
        return nil
    }

    private static func dequantizedWeight(_ linear: Linear) -> MLXArray {
        guard let quantized = linear as? QuantizedLinear else { return linear.weight }
        return dequantized(
            quantized.weight, scales: quantized.scales, biases: quantized.biases,
            groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode,
            dtype: .float32)
    }

    static func factors(
        weight: MLXArray, basis: [[Float]], strength: Float,
        normalization: WeightNormalization,
        composition: AblationComposition = .simultaneous,
        adapterRank: Int
    ) -> (a: MLXArray, b: MLXArray) {
        precondition(!basis.isEmpty && adapterRank >= basis.count)
        let rowNorms = sqrt((weight * weight).sum(axis: 1, keepDims: true))
        let safeNorms = maximum(rowNorms, MLXArray(1e-12))
        let working = normalization == .none ? weight : weight / safeNorms

        let edited: MLXArray
        let exactFactors: (left: MLXArray, right: MLXArray)?
        switch composition {
        case .simultaneous:
            let basisArray = MLXArray(basis.flatMap { $0 })
                .reshaped(basis.count, basis[0].count).asType(.float32)
            let projected = matmul(basisArray, working)
            let left = -strength * basisArray.T
            // Keep the historical simultaneous expression for reproducible
            // `.full` approximation inputs.
            edited = working - strength * matmul(basisArray.T, projected)
            exactFactors = (left, projected)
        case .sequential:
            // Preserve the order and non-orthogonality of SOM directions. Each
            // rank-one term is formed from the already-edited matrix, so the
            // concatenated L/R factors reconstruct the complete composition:
            //   ΔW = Σᵢ -s rᵢ (rᵢᵀ Wᵢ₋₁) = L R.
            var current = working
            var leftColumns = [MLXArray]()
            var rightRows = [MLXArray]()
            leftColumns.reserveCapacity(basis.count)
            rightRows.reserveCapacity(basis.count)
            for rawDirection in basis {
                let direction = MLXArray(AbliterationMath.normalized(rawDirection))
                    .reshaped(rawDirection.count, 1).asType(.float32)
                let right = matmul(direction.T, current)
                let left = -strength * direction
                current = current + matmul(left, right)
                leftColumns.append(left)
                rightRows.append(right)
            }
            let left = concatenated(leftColumns, axis: 1)
            let right = concatenated(rightRows, axis: 0)
            edited = current
            exactFactors = (left, right)
        }

        if normalization != .full, var exactFactors {
            if normalization == .pre { exactFactors.left = exactFactors.left * rowNorms }
            return paddedFactors(
                left: exactFactors.left, right: exactFactors.right, rank: adapterRank)
        }

        // Exact row-norm restoration can raise the rank beyond the number of
        // directions. As before, `.full` uses the configured low-rank SVD
        // approximation; `.none` and `.pre` always take the exact path above.
        let editedNorms = sqrt((edited * edited).sum(axis: 1, keepDims: true))
        let renormalized = edited * rowNorms / maximum(editedNorms, MLXArray(1e-12))
        return randomizedFactors(delta: renormalized - weight, rank: adapterRank)
    }

    /// Converts L[output,r] × R[r,input] into MLX LoRA A[input,rank], B[rank,output].
    private static func paddedFactors(
        left: MLXArray, right: MLXArray, rank: Int
    ) -> (a: MLXArray, b: MLXArray) {
        let active = left.dim(1)
        precondition(active <= rank)
        guard active < rank else { return (right.T, left.T) }
        let a = concatenated([
            right.T, MLXArray.zeros([right.dim(1), rank - active])
        ], axis: 1)
        let b = concatenated([
            left.T, MLXArray.zeros([rank - active, left.dim(0)])
        ], axis: 0)
        return (a, b)
    }

    /// Heretic uses a low-rank approximation for exact norm preservation during
    /// lightweight trials. A randomized range finder avoids a full large SVD.
    private static func randomizedFactors(
        delta: MLXArray, rank: Int, powerIterations: Int = 6
    ) -> (a: MLXArray, b: MLXArray) {
        let sketchRank = min(min(delta.dim(0), delta.dim(1)), 2 * rank + 4)
        let omega = MLXRandom.normal([delta.dim(1), sketchRank])
        var (q, _) = MLX.qr(matmul(delta, omega), stream: .cpu)
        for _ in 0 ..< powerIterations {
            let (rightQ, _) = MLX.qr(matmul(delta.T, q), stream: .cpu)
            (q, _) = MLX.qr(matmul(delta, rightQ), stream: .cpu)
        }
        let small = matmul(q.T, delta)
        let (smallU, singular, vt) = MLX.svd(small, stream: .cpu)
        let kept = min(rank, singular.dim(0))
        let u = matmul(q, smallU[0..., 0 ..< kept])
        let root = sqrt(singular[0 ..< kept])
        let left = u * root
        let right = root.expandedDimensions(axis: 1) * vt[0 ..< kept, 0...]
        return paddedFactors(left: left, right: right, rank: rank)
    }
}

public enum AbliterationAdapterError: LocalizedError {
    case incompatibleModel
    case noTargetLayers

    public var errorDescription: String? {
        switch self {
        case .incompatibleModel: "The loaded model does not expose LoRA-compatible layers."
        case .noTargetLayers: "No attention output or MLP down-projection layers were found."
        }
    }
}
