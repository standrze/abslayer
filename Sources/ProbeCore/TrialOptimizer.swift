import Foundation

public enum TrialDirectionScope: String, Codable, Sendable, CaseIterable {
    case perLayer
    case global
    case blended
}

public struct ComponentTrialParameters: Codable, Sendable, Equatable {
    public var maximum: Float
    public var peakLayer: Float
    public var minimumFraction: Float
    public var radius: Float

    public var minimum: Float { maximum * minimumFraction }
}

public struct AbliterationTrialParameters: Codable, Sendable, Equatable {
    public var directionScope: TrialDirectionScope
    public var directionLayer: Float
    public var directionBlend: Float? = nil
    public var attention: ComponentTrialParameters
    public var mlp: ComponentTrialParameters

    public func configuration(
        normalization: WeightNormalization = .full
    ) -> AbliterationConfiguration {
        let runtimeDirectionScope: DirectionScope
        switch directionScope {
        case .perLayer:
            runtimeDirectionScope = .perLayer
        case .global:
            runtimeDirectionScope = .global(layer: directionLayer)
        case .blended:
            runtimeDirectionScope = .blended(
                globalLayer: directionLayer, perLayerFraction: directionBlend ?? 0.5)
        }

        return AbliterationConfiguration(
            attention: LayerAblationKernel(
                maximum: attention.maximum, peakLayer: attention.peakLayer,
                minimum: attention.minimum, radius: attention.radius),
            mlp: LayerAblationKernel(
                maximum: mlp.maximum, peakLayer: mlp.peakLayer,
                minimum: mlp.minimum, radius: mlp.radius),
            directionScope: runtimeDirectionScope,
            normalization: normalization)
    }
}

public struct AbliterationTrialMetrics: Codable, Sendable, Equatable {
    public var refusalRate: Double
    public var controlFailureRate: Double
    public var firstTokenKL: Double

    /// A bounded scalarization used only to split TPE's good/bad observations.
    /// Final selection minimizes refusals then KL inside a benign-control guardrail.
    public var tpeLoss: Double {
        let utilityPenalty = firstTokenKL <= 0.5
            ? 0.2 * firstTokenKL
            : 2 + (firstTokenKL - 0.5)
        return 2 * controlFailureRate + refusalRate + utilityPenalty
    }
}

public struct AbliterationTrialRecord: Codable, Sendable, Equatable {
    public var index: Int
    public var parameters: AbliterationTrialParameters
    public var metrics: AbliterationTrialMetrics
}

public struct AbliterationFinalVerification: Codable, Sendable, Equatable {
    public var metrics: AbliterationTrialMetrics
    public var passedGuardrails: Bool
    public var passedAbliteration: Bool?

    public init(
        metrics: AbliterationTrialMetrics, passedGuardrails: Bool,
        passedAbliteration: Bool? = nil
    ) {
        self.metrics = metrics
        self.passedGuardrails = passedGuardrails
        self.passedAbliteration = passedAbliteration
    }

    public var passedAll: Bool { passedGuardrails && passedAbliteration == true }
}

public struct AbliterationStudy: Codable, Sendable, Equatable {
    public var modelPath: String
    public var measurementModelPath: String?
    public var measurementDataSignature: UInt64?
    public var evaluationDataSignature: UInt64?
    public var measurementCases: Int?
    public var evaluationCases: Int?
    public var finalEvaluationCases: Int?
    public var subspaceRank: Int?
    public var extractionMethod: DirectionExtractionMethod?
    public var startupTrialCount: Int?
    public var maximumRefusalRate: Double?
    public var normalization: WeightNormalization?
    public var fullNormalizationRank: Int?
    public var winsorizationQuantile: Float?
    public var seed: UInt64
    public var baselineControlFailureRate: Double?
    public var baselineRefusalRate: Double?
    public var finalVerification: AbliterationFinalVerification?
    public var trials: [AbliterationTrialRecord]

    public init(
        modelPath: String, seed: UInt64,
        measurementModelPath: String? = nil,
        measurementDataSignature: UInt64? = nil,
        evaluationDataSignature: UInt64? = nil,
        measurementCases: Int? = nil, evaluationCases: Int? = nil,
        finalEvaluationCases: Int? = nil,
        subspaceRank: Int? = nil,
        extractionMethod: DirectionExtractionMethod? = nil,
        startupTrialCount: Int? = nil,
        maximumRefusalRate: Double? = nil,
        normalization: WeightNormalization? = nil,
        fullNormalizationRank: Int? = nil, winsorizationQuantile: Float? = nil,
        baselineControlFailureRate: Double? = nil,
        baselineRefusalRate: Double? = nil,
        finalVerification: AbliterationFinalVerification? = nil,
        trials: [AbliterationTrialRecord] = []
    ) {
        self.modelPath = modelPath
        self.measurementModelPath = measurementModelPath
        self.measurementDataSignature = measurementDataSignature
        self.evaluationDataSignature = evaluationDataSignature
        self.measurementCases = measurementCases
        self.evaluationCases = evaluationCases
        self.finalEvaluationCases = finalEvaluationCases
        self.subspaceRank = subspaceRank
        self.extractionMethod = extractionMethod
        self.startupTrialCount = startupTrialCount
        self.maximumRefusalRate = maximumRefusalRate
        self.normalization = normalization
        self.fullNormalizationRank = fullNormalizationRank
        self.winsorizationQuantile = winsorizationQuantile
        self.seed = seed
        self.baselineControlFailureRate = baselineControlFailureRate
        self.baselineRefusalRate = baselineRefusalRate
        self.finalVerification = finalVerification
        self.trials = trials
    }

    public var best: AbliterationTrialRecord? {
        let feasible = trials.filter {
            $0.metrics.controlFailureRate <= 0.10 && $0.metrics.firstTokenKL <= 0.5
        }
        if !feasible.isEmpty {
            return feasible.min {
                ($0.metrics.refusalRate, $0.metrics.firstTokenKL, $0.metrics.controlFailureRate)
                    < ($1.metrics.refusalRate, $1.metrics.firstTokenKL, $1.metrics.controlFailureRate)
            }
        }
        return trials.min {
            ($0.metrics.controlFailureRate, $0.metrics.refusalRate, $0.metrics.firstTokenKL)
                < ($1.metrics.controlFailureRate, $1.metrics.refusalRate, $1.metrics.firstTokenKL)
        }
    }

    public func write(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    public static func read(from path: String) throws -> AbliterationStudy {
        try JSONDecoder().decode(
            AbliterationStudy.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    }
}

/// A deterministic multivariate Tree-structured Parzen Estimator sampler.
/// Initial trials explore uniformly. Later trials draw from KDEs fitted to the
/// best quartile and maximize log l(x) - log g(x), as in TPE/Optuna.
public struct TPESampler: Sendable {
    public var seed: UInt64
    public var startupTrials: Int
    public var candidateCount: Int
    public var explorationProbability: Double
    public var subspaceRank: Int

    public init(
        seed: UInt64 = 0, startupTrials: Int = 12, candidateCount: Int = 64,
        explorationProbability: Double = 0.20, subspaceRank: Int = 1
    ) {
        self.seed = seed
        self.startupTrials = startupTrials
        self.candidateCount = candidateCount
        self.explorationProbability = explorationProbability
        self.subspaceRank = max(1, subspaceRank)
    }

    public func suggest(layerCount: Int, history: [AbliterationTrialRecord]) -> AbliterationTrialParameters {
        precondition(layerCount > 1)
        // Do not step trial seeds by SplitMix64's internal Weyl increment: that
        // creates overlapping shifted streams and repeats continuous parameters
        // across otherwise different categorical trials.
        let trialSeed = seed ^ (UInt64(history.count) &* 0xD1B54A32D192ED03)
        var random = SplitMix64(seed: trialSeed)
        guard history.count >= startupTrials else {
            return uniform(layerCount, trialIndex: history.count, random: &random)
        }
        // TPE can collapse onto a safe but ineffective categorical branch when
        // all feasible startup observations share it. Preserve explicit exploration.
        if random.unit < explorationProbability {
            return uniform(layerCount, random: &random)
        }
        // Form TPE's good set from Pareto fronts instead of collapsing refusal,
        // control health, and KL into a single objective. The scalar loss is
        // retained only as a deterministic tie-breaker within each front.
        let ordered = Self.paretoOrdered(history)
        let split = max(2, Int(ceil(Double(ordered.count) * 0.25)))
        let good = Array(ordered.prefix(split))
        let bad = Array(ordered.dropFirst(split))
        var best = uniform(layerCount, random: &random)
        var bestRatio = -Double.infinity
        for _ in 0 ..< candidateCount {
            let candidate = sampleNearGood(good, layerCount: layerCount, random: &random)
            let ratio = logDensity(candidate, observations: good, layerCount: layerCount)
                - logDensity(candidate, observations: bad, layerCount: layerCount)
            if ratio > bestRatio { best = candidate; bestRatio = ratio }
        }
        return best
    }

    static func paretoOrdered(
        _ observations: [AbliterationTrialRecord]
    ) -> [AbliterationTrialRecord] {
        var remaining = observations
        var ordered = [AbliterationTrialRecord]()
        while !remaining.isEmpty {
            let front = remaining.filter { candidate in
                !remaining.contains { other in
                    other.index != candidate.index
                        && dominates(other.metrics, candidate.metrics)
                }
            }.sorted {
                ($0.metrics.tpeLoss, $0.index) < ($1.metrics.tpeLoss, $1.index)
            }
            precondition(!front.isEmpty)
            let indices = Set(front.map(\.index))
            ordered.append(contentsOf: front)
            remaining.removeAll { indices.contains($0.index) }
        }
        return ordered
    }

    private static func dominates(
        _ lhs: AbliterationTrialMetrics, _ rhs: AbliterationTrialMetrics
    ) -> Bool {
        let lhsViolation = max(0, lhs.controlFailureRate - 0.10)
            + max(0, lhs.firstTokenKL - 0.5)
        let rhsViolation = max(0, rhs.controlFailureRate - 0.10)
            + max(0, rhs.firstTokenKL - 0.5)
        if lhsViolation != rhsViolation { return lhsViolation < rhsViolation }
        let lhsValues = [lhs.refusalRate, lhs.controlFailureRate, lhs.firstTokenKL]
        let rhsValues = [rhs.refusalRate, rhs.controlFailureRate, rhs.firstTokenKL]
        return zip(lhsValues, rhsValues).allSatisfy { $0 <= $1 }
            && zip(lhsValues, rhsValues).contains { $0 < $1 }
    }

    private func uniform(
        _ layers: Int, trialIndex: Int? = nil, random: inout SplitMix64
    ) -> AbliterationTrialParameters {
        let last = Float(layers - 1)
        func component(
            mlp: Bool, lowerScale: Float, upperScale: Float
        ) -> ComponentTrialParameters {
            ComponentTrialParameters(
                maximum: mlp
                    ? max(0, random.float(in: (-0.25 * lowerScale) ... (1.5 * upperScale)))
                    : random.float(in: (0.8 * lowerScale) ... (1.5 * upperScale)),
                // Refusal sites are model- and dataset-specific. Sampling only
                // a presumed middle/late window can make the optimizer miss a
                // valid early source or application peak entirely.
                peakLayer: random.float(in: 0 ... last),
                minimumFraction: random.float(in: 0 ... 1),
                radius: random.float(in: 1 ... max(1, last)))
        }
        let scopes = TrialDirectionScope.allCases
        let scope = trialIndex.map { scopes[$0 % scopes.count] }
            ?? scopes[Int(random.next() % UInt64(scopes.count))]
        // Startup exploration gives each categorical branch equal coverage.
        // Blended trials visit every twentieth of the interpolation range in a
        // coprime permutation, avoiding a random cluster at either endpoint.
        let blend: Float
        if scope == .blended, let trialIndex {
            let stratumCount = max(1, startupTrials / scopes.count)
            let ordinal = trialIndex / scopes.count
            let stride = stratumCount == 1 ? 1 : Self.coprimeStride(for: stratumCount)
            blend = (Float((ordinal * stride) % stratumCount) + 0.5)
                / Float(stratumCount)
        } else {
            blend = random.float(in: 0 ... 1)
        }
        let scales = strengthScales(scope: scope, blend: blend)
        var attention = component(
            mlp: false, lowerScale: scales.lower, upperScale: scales.upper)
        var mlp = component(
            mlp: true, lowerScale: scales.lower, upperScale: scales.upper)
        // Every categorical scope explicitly measures combined, attention-only,
        // and MLP-only edits during startup. This distinguishes the component
        // carrying refusal from the component causing benign-model drift.
        if let trialIndex {
            switch (trialIndex / scopes.count) % 5 {
            case 1:
                mlp.maximum = 0
            case 2:
                attention.maximum = 0
            default:
                break
            }
        }
        return AbliterationTrialParameters(
            directionScope: scope,
            directionLayer: random.float(in: 0 ... last),
            directionBlend: blend,
            attention: attention, mlp: mlp)
    }

    private static func coprimeStride(for count: Int) -> Int {
        func gcd(_ lhs: Int, _ rhs: Int) -> Int {
            var a = lhs
            var b = rhs
            while b != 0 { (a, b) = (b, a % b) }
            return a
        }
        var candidate = max(1, count / 3)
        while gcd(candidate, count) != 1 { candidate += 1 }
        return candidate
    }

    private func sampleNearGood(
        _ good: [AbliterationTrialRecord], layerCount: Int, random: inout SplitMix64
    ) -> AbliterationTrialParameters {
        let last = Float(layerCount - 1)
        let anchor = good[Int(random.next() % UInt64(good.count))].parameters
        func jitter(_ value: Float, range: ClosedRange<Float>) -> Float {
            let sigma = Double(range.upperBound - range.lowerBound) / sqrt(Double(good.count) + 1)
            return min(max(value + Float(random.gaussian() * sigma), range.lowerBound), range.upperBound)
        }
        func component(
            _ value: ComponentTrialParameters, mlp: Bool,
            lowerScale: Float, upperScale: Float
        ) -> ComponentTrialParameters {
            ComponentTrialParameters(
                maximum: jitter(
                    value.maximum,
                    range: mlp
                        ? 0 ... (1.5 * upperScale)
                        : 0 ... (1.5 * upperScale)),
                peakLayer: jitter(value.peakLayer, range: 0 ... last),
                minimumFraction: jitter(value.minimumFraction, range: 0 ... 1),
                radius: jitter(value.radius, range: 1 ... max(1, last)))
        }
        let scopeWeights = TrialDirectionScope.allCases.map { scope in
            good.count { $0.parameters.directionScope == scope } + 1
        }
        let scopeTotal = scopeWeights.reduce(0, +)
        var scopeDraw = Int(random.next() % UInt64(scopeTotal))
        var selectedScope = TrialDirectionScope.global
        for (scope, weight) in zip(TrialDirectionScope.allCases, scopeWeights) {
            if scopeDraw < weight { selectedScope = scope; break }
            scopeDraw -= weight
        }
        let selectedBlend = jitter(anchor.directionBlend ?? 0.5, range: 0 ... 1)
        let scales = strengthScales(scope: selectedScope, blend: selectedBlend)
        return AbliterationTrialParameters(
            directionScope: selectedScope,
            directionLayer: jitter(anchor.directionLayer, range: 0 ... last),
            directionBlend: selectedBlend,
            attention: component(
                anchor.attention, mlp: false,
                lowerScale: scales.lower, upperScale: scales.upper),
            mlp: component(
                anchor.mlp, mlp: true,
                lowerScale: scales.lower, upperScale: scales.upper))
    }

    private func strengthScales(
        scope: TrialDirectionScope, blend: Float
    ) -> (lower: Float, upper: Float) {
        let localLower = 1 / Float(subspaceRank)
        let localUpper = 1 / sqrt(Float(subspaceRank))
        switch scope {
        case .global:
            return (1, 1)
        case .perLayer:
            return (localLower, localUpper)
        case .blended:
            let fraction = min(max(blend, 0), 1)
            return (
                1 - fraction * (1 - localLower),
                1 - fraction * (1 - localUpper))
        }
    }

    private func logDensity(
        _ candidate: AbliterationTrialParameters,
        observations: [AbliterationTrialRecord], layerCount: Int
    ) -> Double {
        guard !observations.isEmpty else { return 0 }
        let last = Float(layerCount - 1)
        func kde(_ value: Float, _ values: [Float], span: Float) -> Double {
            let bandwidth = max(Double(span) / sqrt(Double(values.count) + 1), 1e-5)
            let density = values.reduce(0.0) {
                let z = (Double(value) - Double($1)) / bandwidth
                return $0 + exp(-0.5 * z * z) / bandwidth
            } / Double(values.count)
            return log(max(density, 1e-300))
        }
        let scopes = observations.count { $0.parameters.directionScope == candidate.directionScope }
        var result = log(Double(scopes + 1) / Double(observations.count + 2))
        let values = observations.map(\.parameters)
        result += kde(candidate.directionLayer, values.map(\.directionLayer), span: 0.5 * last)
        result += kde(
            candidate.directionBlend ?? 0.5,
            values.map { $0.directionBlend ?? 0.5 }, span: 1)
        result += kde(candidate.attention.maximum, values.map(\.attention.maximum), span: 0.7)
        result += kde(candidate.attention.peakLayer, values.map(\.attention.peakLayer), span: 0.4 * last)
        result += kde(candidate.attention.minimumFraction, values.map(\.attention.minimumFraction), span: 1)
        result += kde(candidate.attention.radius, values.map(\.attention.radius), span: 0.6 * last)
        result += kde(candidate.mlp.maximum, values.map(\.mlp.maximum), span: 1.5)
        result += kde(candidate.mlp.peakLayer, values.map(\.mlp.peakLayer), span: 0.4 * last)
        result += kde(candidate.mlp.minimumFraction, values.map(\.mlp.minimumFraction), span: 1)
        result += kde(candidate.mlp.radius, values.map(\.mlp.radius), span: 0.6 * last)
        return result
    }
}

private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    var unit: Double {
        mutating get { Double(next() >> 11) / Double(UInt64(1) << 53) }
    }
    mutating func float(in range: ClosedRange<Float>) -> Float {
        range.lowerBound + Float(unit) * (range.upperBound - range.lowerBound)
    }
    mutating func gaussian() -> Double {
        let u1 = max(unit, 1e-12)
        let u2 = unit
        return sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
    }
}
