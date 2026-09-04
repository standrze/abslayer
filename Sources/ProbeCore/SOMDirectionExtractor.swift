import Foundation
import MLX

/// The two reproducible interpretations of the SOM training procedure.
public enum SOMTrainingMode: String, Codable, Sendable, Equatable {
    /// Algorithm-level compatibility with the authors' released code, pinned
    /// to MiniSom 2.3.5: sigma 0.33, a balanced modulo schedule shuffled once,
    /// and asymptotic decay of both learning rate and sigma.
    ///
    /// The Swift implementation uses its own deterministic PRNG, so the random
    /// stream is not byte-identical to NumPy's `RandomState`; schedule counts,
    /// update ordering semantics, and decay equations are otherwise the same.
    case officialMiniSom235

    /// A literal interpretation of the paper text: iid random online samples,
    /// sigma fixed at 0.3, and only the learning rate following the stated
    /// `p(t) = p(0) / (1 + 2t/T)` schedule. This also preserves the behavior of
    /// ABSlayer's first SOM implementation, but only when explicitly selected.
    case paperText

    fileprivate var defaultSigma: Float {
        switch self {
        case .officialMiniSom235: 0.33
        case .paperText: 0.3
        }
    }
}

/// Training settings for Piras et al., "SOM Directions are Better than One"
/// (AAAI 2026), with explicit paper/released-code compatibility semantics.
public struct SOMDirectionConfiguration: Sendable, Equatable {
    public var rows: Int
    public var columns: Int
    public var iterations: Int
    public var initialLearningRate: Float
    public var sigma: Float
    public var seed: UInt64
    public var trainingMode: SOMTrainingMode

    public init(
        rows: Int = 4,
        columns: Int = 4,
        iterations: Int = 10_000,
        initialLearningRate: Float = 0.01,
        sigma: Float? = nil,
        seed: UInt64 = 0,
        trainingMode: SOMTrainingMode = .officialMiniSom235
    ) {
        self.rows = rows
        self.columns = columns
        self.iterations = iterations
        self.initialLearningRate = initialLearningRate
        self.sigma = sigma ?? trainingMode.defaultSigma
        self.seed = seed
        self.trainingMode = trainingMode
    }

    /// Defaults used by the authors' public MiniSom 2.3.5 implementation.
    public static let officialMiniSom235 = SOMDirectionConfiguration(
        trainingMode: .officialMiniSom235)

    /// Literal defaults stated in the paper. Kept as a named alternative so
    /// the sigma/schedule discrepancy can never be silent.
    public static let paperDefault = SOMDirectionConfiguration(
        trainingMode: .paperText)
}

/// Diagnostics and the deterministically selected SOM directions for one layer.
public struct SOMDirectionResult: Sendable {
    /// Individually unit-normalized `neuron - harmlessCentroid` vectors.
    /// They are deliberately not orthogonalized.
    public let directions: [[Float]]
    public let selectedNeuronIndices: [Int]
    /// Every row-major lattice candidate, individually normalized, so later
    /// dev-set subset search can address stable neuron indices without retraining.
    public let candidateDirections: [[Float]]
    public let neurons: [[Float]]
    public let occupancy: [Int]
    /// Mean Euclidean BMU error per neuron; infinity denotes an empty neuron.
    public let quantizationErrorByNeuron: [Float]
    public let quantizationError: Float

    static let empty = SOMDirectionResult(
        directions: [], selectedNeuronIndices: [], candidateDirections: [],
        neurons: [], occupancy: [],
        quantizationErrorByNeuron: [], quantizationError: .infinity)
}

/// Extracts a non-orthogonal approximation of the harmful representation
/// manifold with a hexagonal self-organizing map.
///
/// Both modes train only on harmful activations, use Euclidean BMUs, a 4x4
/// hexagonal lattice, 10,000 online updates, Gaussian neighborhoods, and
/// initialize each neuron from an actual harmful row with replacement. The
/// default `.officialMiniSom235` mode follows the released implementation's
/// balanced shuffled schedule, sigma 0.33, and decay of both alpha and sigma.
/// `.paperText` is the separately named literal paper interpretation (iid
/// sampling, fixed sigma 0.3, decayed alpha).
///
/// The only runtime optimization is to batch independent layer SOMs into the
/// leading MLX dimension. This preserves each layer's update equation while
/// avoiding `layers x iterations` Swift/Metal dispatches. Because hidden width
/// 2048 x 35 layers x 10,000 online materializations is still expensive,
/// `ProbeEngine.extractionLayers` can restrict training to selected layers.
public enum SOMDirectionExtractor {
    public static func extract(
        harmful: [[Float]],
        harmless: [[Float]],
        rank: Int,
        configuration: SOMDirectionConfiguration = .officialMiniSom235
    ) -> SOMDirectionResult {
        extractLayers(
            harmfulByLayer: [harmful], harmlessByLayer: [harmless], rank: rank,
            selectedLayers: [0], configuration: configuration)[0]
    }

    /// Returns one result per input layer. Layers outside `selectedLayers` are
    /// represented by `.empty`; ProbeEngine replaces those with an explicitly
    /// documented centroid compatibility direction.
    public static func extractLayers(
        harmfulByLayer: [[[Float]]],
        harmlessByLayer: [[[Float]]],
        rank: Int,
        selectedLayers: Set<Int>? = nil,
        configuration: SOMDirectionConfiguration = .officialMiniSom235
    ) -> [SOMDirectionResult] {
        var results = Array(
            repeating: SOMDirectionResult.empty, count: harmfulByLayer.count)
        guard rank > 0,
              harmfulByLayer.count == harmlessByLayer.count,
              configuration.rows > 0,
              configuration.columns > 0,
              configuration.iterations > 0,
              configuration.initialLearningRate > 0,
              configuration.sigma > 0
        else { return results }

        let requested = (selectedLayers ?? Set(harmfulByLayer.indices))
            .filter(harmfulByLayer.indices.contains)
            .sorted()

        // Layers normally share `(sampleCount, hiddenWidth)`, so this forms one
        // batch. Grouping by shape keeps the public helper safe for synthetic or
        // architecture-specific inputs without changing any layer's algorithm.
        var groups = [SOMShape: [Int]]()
        for layer in requested {
            guard let first = harmfulByLayer[layer].first,
                  !first.isEmpty,
                  !harmlessByLayer[layer].isEmpty,
                  harmfulByLayer[layer].allSatisfy({ $0.count == first.count }),
                  harmlessByLayer[layer].allSatisfy({ $0.count == first.count })
            else { continue }
            groups[SOMShape(samples: harmfulByLayer[layer].count, width: first.count),
                   default: []].append(layer)
        }

        for (shape, layers) in groups.sorted(by: { lhs, rhs in
            lhs.key.samples == rhs.key.samples
                ? lhs.key.width < rhs.key.width
                : lhs.key.samples < rhs.key.samples
        }) {
            let harmfulLayers = layers.map { harmfulByLayer[$0] }
            let neuronsByLayer = train(
                harmfulByLayer: harmfulLayers, shape: shape,
                configuration: configuration)
            for (batchIndex, layer) in layers.enumerated() {
                results[layer] = summarize(
                    neurons: neuronsByLayer[batchIndex],
                    harmful: harmfulByLayer[layer],
                    harmless: harmlessByLayer[layer],
                    rank: rank)
            }
        }
        return results
    }

    /// Indexes only layers that actually trained a SOM. Compatibility
    /// placeholders and unselected layers are deliberately excluded so a
    /// targeted run never retains `16 x everyLayer` candidate vectors.
    static func indexedTrainedResults(
        _ results: [SOMDirectionResult]
    ) -> [Int: SOMDirectionResult] {
        Dictionary(uniqueKeysWithValues: results.enumerated().compactMap { index, result in
            result.neurons.isEmpty ? nil : (index, result)
        })
    }

    private static func train(
        harmfulByLayer: [[[Float]]],
        shape: SOMShape,
        configuration: SOMDirectionConfiguration
    ) -> [[[Float]]] {
        let neuronCount = configuration.rows * configuration.columns
        let flat = harmfulByLayer.flatMap { $0.flatMap { $0 } }
        let harmful = MLXArray(flat)
            .reshaped(harmfulByLayer.count, shape.samples, shape.width)
            .asType(.float32)

        var random = SOMSplitMix64(seed: configuration.seed)
        let initialIndices = initializationIndices(
            sampleCount: shape.samples, neuronCount: neuronCount, random: &random)
        let sampleIndices = trainingSampleIndices(
            sampleCount: shape.samples,
            iterations: configuration.iterations,
            trainingMode: configuration.trainingMode,
            random: &random)
        var neurons = harmful.take(MLXArray(initialIndices), axis: 1)
        eval(neurons)

        let latticeDistances = MLXArray(hexagonalDistanceMatrix(
            rows: configuration.rows, columns: configuration.columns))
            .reshaped(neuronCount, neuronCount)
        let update = compile { arrays in
            let weights = arrays[0]
            let sample = arrays[1]
            let learningRate = arrays[2]
            let latticeDistances = arrays[3]
            let gaussianDenominator = arrays[4]
            let delta = sample.expandedDimensions(axis: 1) - weights
            let distances = (delta * delta).sum(axis: 2)
            let winners = distances.argMin(axis: 1)
            let neighborhood = exp(
                -latticeDistances.take(winners, axis: 0) / gaussianDenominator)
                .expandedDimensions(axis: 2)
            return [weights + learningRate * neighborhood * delta]
        }

        for (iteration, sampleIndex) in sampleIndices.enumerated() {
            let sample = harmful.take(MLXArray(sampleIndex), axis: 1)
            let parameters = trainingParameters(
                at: iteration, configuration: configuration)
            let gaussianDenominator = MLXArray(
                2 * parameters.sigma * parameters.sigma)
            neurons = update([
                neurons, sample, MLXArray(parameters.learningRate), latticeDistances,
                gaussianDenominator,
            ])[0]
            // Materialize every online update. Deferring this would create a
            // 10,000-node state chain and defeat the bounded-memory batch.
            eval(neurons)
        }

        return harmfulByLayer.indices.map { layer in
            let flatLayer = neurons[layer].asArray(Float.self)
            return stride(from: 0, to: flatLayer.count, by: shape.width).map {
                Array(flatLayer[$0 ..< $0 + shape.width])
            }
        }
    }

    private static func summarize(
        neurons: [[Float]],
        harmful: [[Float]],
        harmless: [[Float]],
        rank: Int
    ) -> SOMDirectionResult {
        guard !neurons.isEmpty, let width = neurons.first?.count, width > 0 else {
            return .empty
        }
        var occupancy = Array(repeating: 0, count: neurons.count)
        var errorSums = Array(repeating: Double.zero, count: neurons.count)
        var totalError = Double.zero
        for sample in harmful {
            var winner = 0
            var best = Double.infinity
            for neuronIndex in neurons.indices {
                var squared = Double.zero
                for feature in 0 ..< width {
                    let delta = Double(sample[feature] - neurons[neuronIndex][feature])
                    squared += delta * delta
                }
                if squared < best {
                    best = squared
                    winner = neuronIndex
                }
            }
            let error = sqrt(best)
            occupancy[winner] += 1
            errorSums[winner] += error
            totalError += error
        }
        let perNeuronError = occupancy.indices.map { index in
            occupancy[index] > 0
                ? Float(errorSums[index] / Double(occupancy[index]))
                : Float.infinity
        }

        let harmlessCentroid = AbliterationMath.mean(harmless)
        let rawDirections = neurons.map { neuron in
            zip(neuron, harmlessCentroid).map(-)
        }
        let usable = neurons.indices.filter { index in
            occupancy[index] > 0
                && rawDirections[index].reduce(Float.zero) { $0 + $1 * $1 }
                    > Float.ulpOfOne
        }
        let candidateDirections = rawDirections.map(AbliterationMath.normalized)
        let selected = selectForCoverage(
            candidates: usable, neurons: neurons, occupancy: occupancy,
            quantizationError: perNeuronError, rank: min(rank, usable.count))
        let directions = selected.map { index in
            candidateDirections[index]
        }
        return SOMDirectionResult(
            directions: directions,
            selectedNeuronIndices: selected,
            candidateDirections: candidateDirections,
            neurons: neurons,
            occupancy: occupancy,
            quantizationErrorByNeuron: perNeuronError,
            quantizationError: harmful.isEmpty
                ? .infinity : Float(totalError / Double(harmful.count)))
    }

    /// Evaluation-free deterministic subset ordering. The first direction is
    /// the densest occupied neuron (lowest quantization error, then index, break
    /// ties). Each subsequent direction greedily maximizes
    /// `occupancy * squaredDistanceToNearestSelectedPrototype`, which covers
    /// distinct harmful modes without making the directions orthogonal.
    private static func selectForCoverage(
        candidates: [Int],
        neurons: [[Float]],
        occupancy: [Int],
        quantizationError: [Float],
        rank: Int
    ) -> [Int] {
        guard rank > 0, !candidates.isEmpty else { return [] }
        let first = candidates.min { lhs, rhs in
            if occupancy[lhs] != occupancy[rhs] {
                return occupancy[lhs] > occupancy[rhs]
            }
            if quantizationError[lhs] != quantizationError[rhs] {
                return quantizationError[lhs] < quantizationError[rhs]
            }
            return lhs < rhs
        }!
        var selected = [first]
        var remaining = Set(candidates.filter { $0 != first })
        while selected.count < rank, !remaining.isEmpty {
            let next = remaining.min { lhs, rhs in
                let lhsScore = coverageScore(
                    lhs, selected: selected, neurons: neurons,
                    occupancy: occupancy)
                let rhsScore = coverageScore(
                    rhs, selected: selected, neurons: neurons,
                    occupancy: occupancy)
                if lhsScore != rhsScore { return lhsScore > rhsScore }
                if quantizationError[lhs] != quantizationError[rhs] {
                    return quantizationError[lhs] < quantizationError[rhs]
                }
                return lhs < rhs
            }!
            selected.append(next)
            remaining.remove(next)
        }
        return selected
    }

    private static func coverageScore(
        _ candidate: Int,
        selected: [Int],
        neurons: [[Float]],
        occupancy: [Int]
    ) -> Double {
        let nearest = selected.map { selectedIndex -> Double in
            zip(neurons[candidate], neurons[selectedIndex]).reduce(Double.zero) {
                let delta = Double($1.0 - $1.1)
                return $0 + delta * delta
            }
        }.min() ?? 0
        return Double(occupancy[candidate]) * nearest
    }

    private static func initializationIndices(
        sampleCount: Int,
        neuronCount: Int,
        random: inout SOMSplitMix64
    ) -> [Int] {
        (0 ..< neuronCount).map { _ in
            randomIndex(upperBound: sampleCount, random: &random)
        }
    }

    /// Deterministic schedule helper exposed internally for compatibility
    /// tests. `neuronCount` accounts for random draws consumed by MiniSom-style
    /// random-row initialization before its training schedule is shuffled.
    static func trainingSampleIndices(
        sampleCount: Int,
        neuronCount: Int,
        configuration: SOMDirectionConfiguration
    ) -> [Int] {
        guard sampleCount > 0, neuronCount >= 0, configuration.iterations > 0 else {
            return []
        }
        var random = SOMSplitMix64(seed: configuration.seed)
        _ = initializationIndices(
            sampleCount: sampleCount, neuronCount: neuronCount, random: &random)
        return trainingSampleIndices(
            sampleCount: sampleCount,
            iterations: configuration.iterations,
            trainingMode: configuration.trainingMode,
            random: &random)
    }

    private static func trainingSampleIndices(
        sampleCount: Int,
        iterations: Int,
        trainingMode: SOMTrainingMode,
        random: inout SOMSplitMix64
    ) -> [Int] {
        switch trainingMode {
        case .officialMiniSom235:
            // MiniSom 2.3.5 `_build_iteration_indexes` constructs this
            // balanced modulo schedule, then shuffles it once.
            var indices = (0 ..< iterations).map { $0 % sampleCount }
            if indices.count > 1 {
                for upper in stride(from: indices.count - 1, through: 1, by: -1) {
                    let lower = randomIndex(upperBound: upper + 1, random: &random)
                    indices.swapAt(upper, lower)
                }
            }
            return indices
        case .paperText:
            return (0 ..< iterations).map { _ in
                randomIndex(upperBound: sampleCount, random: &random)
            }
        }
    }

    /// Returns the update parameters at a zero-based online iteration. This is
    /// internal so tests can lock down the paper/code discrepancy explicitly.
    static func trainingParameters(
        at iteration: Int,
        configuration: SOMDirectionConfiguration
    ) -> (learningRate: Float, sigma: Float) {
        let denominator = 1 + 2 * Float(iteration) / Float(configuration.iterations)
        let learningRate = configuration.initialLearningRate / denominator
        let sigma: Float
        switch configuration.trainingMode {
        case .officialMiniSom235:
            // MiniSom's default `asymptotic_decay` is applied independently to
            // both learning rate and neighborhood sigma.
            sigma = configuration.sigma / denominator
        case .paperText:
            // The paper gives an alpha schedule and states sigma = 0.3 without
            // specifying a sigma schedule, so the literal mode keeps it fixed.
            sigma = configuration.sigma
        }
        return (learningRate, sigma)
    }

    private static func randomIndex(
        upperBound: Int,
        random: inout SOMSplitMix64
    ) -> Int {
        precondition(upperBound > 0)
        let bound = UInt64(upperBound)
        // Rejection avoids modulo bias while keeping every seeded run stable.
        let cutoff = UInt64.max - UInt64.max % bound
        var value = random.next()
        while value >= cutoff { value = random.next() }
        return Int(value % bound)
    }

    /// Offset-row coordinates with row spacing sqrt(3)/2 produce six
    /// equidistant unit-distance neighbors for interior hexes.
    private static func hexagonalDistanceMatrix(rows: Int, columns: Int) -> [Float] {
        let coordinates = (0 ..< rows * columns).map { index -> (Float, Float) in
            let row = index / columns
            let column = index % columns
            return (
                Float(column) + (row.isMultiple(of: 2) ? 0 : 0.5),
                Float(row) * sqrt(3) / 2)
        }
        return coordinates.flatMap { source in
            coordinates.map { target in
                let x = source.0 - target.0
                let y = source.1 - target.1
                return x * x + y * y
            }
        }
    }
}

private struct SOMShape: Hashable {
    let samples: Int
    let width: Int
}

private struct SOMSplitMix64 {
    var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}
