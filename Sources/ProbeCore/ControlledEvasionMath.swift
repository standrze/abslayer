import Foundation

/// A unit-normalized linear probe. Its score is the signed Euclidean distance
/// from an activation to the learned separating hyperplane.
public struct ControlledEvasionProbe: Codable, Sendable, Equatable {
    public let weights: [Float]
    public let bias: Float

    public var width: Int { weights.count }

    public init?(weights: [Float], bias: Float = 0) {
        guard !weights.isEmpty, bias.isFinite, weights.allSatisfy(\.isFinite) else {
            return nil
        }
        let norm = sqrt(ControlledEvasionMath.dot(weights, weights))
        guard norm.isFinite, norm > ControlledEvasionMath.epsilon else { return nil }
        self.weights = weights.map { $0 / norm }
        self.bias = bias / norm
    }

    /// Fits a diagonal-covariance LDA probe. The pooled variance prevents a
    /// high-variance feature from dominating a raw difference-of-means probe.
    public static func fit(
        positive: [[Float]], negative: [[Float]], varianceFloor: Float = 1e-4
    ) -> ControlledEvasionProbe? {
        guard
            let first = positive.first ?? negative.first,
            !positive.isEmpty, !negative.isEmpty, !first.isEmpty,
            varianceFloor.isFinite, varianceFloor > 0,
            (positive + negative).allSatisfy({
                $0.count == first.count && $0.allSatisfy(\.isFinite)
            })
        else { return nil }

        let positiveMean = ControlledEvasionMath.mean(positive)
        let negativeMean = ControlledEvasionMath.mean(negative)
        var pooledVariance = Array(repeating: Float.zero, count: first.count)
        for vector in positive {
            for index in vector.indices {
                let residual = vector[index] - positiveMean[index]
                pooledVariance[index] += residual * residual
            }
        }
        for vector in negative {
            for index in vector.indices {
                let residual = vector[index] - negativeMean[index]
                pooledVariance[index] += residual * residual
            }
        }
        let degreesOfFreedom = Float(max(1, positive.count + negative.count - 2))
        let rawWeights = positiveMean.indices.map { index in
            (positiveMean[index] - negativeMean[index])
                / (pooledVariance[index] / degreesOfFreedom + varianceFloor)
        }
        let midpointSum = zip(positiveMean, negativeMean).map(+)
        let rawBias = -0.5 * ControlledEvasionMath.dot(rawWeights, midpointSum)
        return ControlledEvasionProbe(weights: rawWeights, bias: rawBias)
    }

    public func score(_ activation: [Float]) -> Float {
        precondition(activation.count == width)
        return ControlledEvasionMath.dot(weights, activation) + bias
    }

    public func isTriggered(_ activation: [Float], margin: Float = 0) -> Bool {
        score(activation) > margin
    }
}

/// A prompt-conditioned controlled latent evasion (CLE) plan. The prompt probe
/// opens a continuous gate only past `margin`; the target residual is then
/// projected away from the selected behavior subspace around an affine anchor.
public struct ControlledEvasionPlan: Codable, Sendable, Equatable {
    public let probe: ControlledEvasionProbe
    public let behaviorBasis: [[Float]]
    public let reference: [Float]
    public let margin: Float
    public let transitionWidth: Float
    public let strength: Float

    public var promptWidth: Int { probe.width }
    public var targetWidth: Int { reference.count }
    public var rank: Int { behaviorBasis.count }

    public init?(
        probe: ControlledEvasionProbe, behaviorBasis: [[Float]],
        reference: [Float]? = nil, margin: Float = 0,
        transitionWidth: Float = 0, strength: Float = 1
    ) {
        guard
            let width = behaviorBasis.first?.count, width > 0,
            behaviorBasis.allSatisfy({
                $0.count == width && $0.allSatisfy(\.isFinite)
            }),
            margin.isFinite, transitionWidth.isFinite, transitionWidth >= 0,
            strength.isFinite, strength >= 0
        else { return nil }
        let anchor = reference ?? Array(repeating: Float.zero, count: width)
        guard anchor.count == width, anchor.allSatisfy(\.isFinite) else { return nil }
        let basis = ControlledEvasionMath.orthonormalized(behaviorBasis)
        guard !basis.isEmpty else { return nil }
        self.probe = probe
        self.behaviorBasis = basis
        self.reference = anchor
        self.margin = margin
        self.transitionWidth = transitionWidth
        self.strength = strength
    }

    /// A zero-to-one intervention gate. A nonzero transition width gives a
    /// linear ramp instead of a discontinuous edit at the probe margin.
    public func gate(for promptActivation: [Float]) -> Float {
        let excess = probe.score(promptActivation) - margin
        guard excess > 0 else { return 0 }
        guard transitionWidth > 0 else { return 1 }
        return min(1, excess / transitionWidth)
    }

    public func projectedComponent(of targetActivation: [Float]) -> [Float] {
        precondition(targetActivation.count == targetWidth)
        let centered = zip(targetActivation, reference).map(-)
        var projection = Array(repeating: Float.zero, count: targetWidth)
        for direction in behaviorBasis {
            let coordinate = ControlledEvasionMath.dot(direction, centered)
            for index in projection.indices {
                projection[index] += coordinate * direction[index]
            }
        }
        return projection
    }

    public func apply(
        promptActivation: [Float], to targetActivation: [Float]
    ) -> [Float] {
        precondition(targetActivation.count == targetWidth)
        let amount = strength * gate(for: promptActivation)
        guard amount != 0 else { return targetActivation }
        let projection = projectedComponent(of: targetActivation)
        return zip(targetActivation, projection).map { $0 - amount * $1 }
    }
}

/// A compact soft null-space projector learned from benign activations.
///
/// For retained covariance eigenpairs `(q, lambda)`, applying the projector is
/// `x - sum(lambda / (lambda + ridge) * q * dot(q, x))`. This is the stable,
/// ridge-regularized form of a benign null-space constraint and does not require
/// materializing a hidden-width square matrix.
public struct SoftBenignNullProjector: Codable, Sendable, Equatable {
    public let width: Int
    public let basis: [[Float]]
    public let covarianceEigenvalues: [Float]
    public let removedFractions: [Float]
    public let ridge: Float

    public var rank: Int { basis.count }

    public init?(
        benignKeys: [[Float]], rank requestedRank: Int,
        ridge: Float = 1e-3, center: Bool = false,
        iterations: Int = 48, minimumEigenvalueRatio: Float = 1e-6
    ) {
        guard
            let width = benignKeys.first?.count, width > 0, requestedRank > 0,
            ridge.isFinite, ridge > 0, iterations > 0,
            minimumEigenvalueRatio.isFinite, minimumEigenvalueRatio >= 0,
            benignKeys.allSatisfy({
                $0.count == width && $0.allSatisfy(\.isFinite)
            })
        else { return nil }

        var keys = benignKeys
        if center {
            let mean = ControlledEvasionMath.mean(keys)
            keys = keys.map { zip($0, mean).map(-) }
        }
        let requestedBlockWidth = min(
            width, min(keys.count, requestedRank + min(4, requestedRank)))
        var subspace = ControlledEvasionMath.pivotedRowBasis(
            keys, limit: requestedBlockWidth)
        guard !subspace.isEmpty else { return nil }

        for _ in 0 ..< iterations {
            let products = subspace.map {
                ControlledEvasionMath.covarianceProduct(keys: keys, vector: $0)
            }
            let next = ControlledEvasionMath.orthonormalized(
                products, limit: requestedBlockWidth)
            guard !next.isEmpty else { break }
            subspace = next
        }

        let blockWidth = subspace.count
        var rayleigh = Array(
            repeating: Double.zero, count: blockWidth * blockWidth)
        let covarianceProducts = subspace.map {
            ControlledEvasionMath.covarianceProduct(keys: keys, vector: $0)
        }
        for row in 0 ..< blockWidth {
            for column in row ..< blockWidth {
                let value = Double(ControlledEvasionMath.dot(
                    subspace[row], covarianceProducts[column]))
                rayleigh[row * blockWidth + column] = value
                rayleigh[column * blockWidth + row] = value
            }
        }
        let decomposition = ControlledEvasionMath.symmetricEigendecomposition(
            rayleigh, size: blockWidth)
        let orderedIndices = decomposition.values.indices.sorted {
            decomposition.values[$0] > decomposition.values[$1]
        }
        guard let leadingIndex = orderedIndices.first else { return nil }
        let leadingValue = max(0, decomposition.values[leadingIndex])
        guard leadingValue > Double(ControlledEvasionMath.epsilon) else { return nil }
        let threshold = leadingValue * Double(minimumEigenvalueRatio)

        var retainedBasis = [[Float]]()
        var retainedValues = [Float]()
        for eigenIndex in orderedIndices.prefix(requestedRank) {
            let eigenvalue = max(0, decomposition.values[eigenIndex])
            guard
                eigenvalue > Double(ControlledEvasionMath.epsilon),
                eigenvalue >= threshold
            else { continue }
            var vector = Array(repeating: Float.zero, count: width)
            for sourceIndex in 0 ..< blockWidth {
                let coefficient = Float(
                    decomposition.vectors[sourceIndex * blockWidth + eigenIndex])
                for coordinate in vector.indices {
                    vector[coordinate] += coefficient * subspace[sourceIndex][coordinate]
                }
            }
            let unit = ControlledEvasionMath.normalized(vector)
            guard ControlledEvasionMath.dot(unit, unit) > 0.99 else { continue }
            retainedBasis.append(unit)
            retainedValues.append(Float(eigenvalue))
        }
        guard !retainedBasis.isEmpty else { return nil }

        self.width = width
        self.basis = retainedBasis
        self.covarianceEigenvalues = retainedValues
        self.removedFractions = retainedValues.map { $0 / ($0 + ridge) }
        self.ridge = ridge
    }

    public func apply(to vector: [Float]) -> [Float] {
        precondition(vector.count == width)
        var result = vector
        for (direction, fraction) in zip(basis, removedFractions) {
            let coordinate = fraction * ControlledEvasionMath.dot(direction, vector)
            for index in result.indices { result[index] -= coordinate * direction[index] }
        }
        return result
    }
}

/// A low-rank matrix intervention `Delta W = sum(scale * output * input^T)`.
/// Input factors are first normalized and then passed through a benign-null
/// projector, so benign keys produce little or no intervention response.
public struct BenignNullLowRankIntervention: Codable, Sendable, Equatable {
    public let outputFactors: [[Float]]
    public let inputFactors: [[Float]]
    public let scales: [Float]
    public let outputWidth: Int
    public let inputWidth: Int

    public var rank: Int { scales.count }

    public init?(
        outputDirections: [[Float]], conditionDirections: [[Float]],
        scales: [Float], benignProjector: SoftBenignNullProjector
    ) {
        guard
            !scales.isEmpty, outputDirections.count == scales.count,
            conditionDirections.count == scales.count,
            let outputWidth = outputDirections.first?.count, outputWidth > 0,
            let inputWidth = conditionDirections.first?.count,
            inputWidth == benignProjector.width,
            scales.allSatisfy(\.isFinite),
            outputDirections.allSatisfy({
                $0.count == outputWidth && $0.allSatisfy(\.isFinite)
                    && ControlledEvasionMath.norm($0) > ControlledEvasionMath.epsilon
            }),
            conditionDirections.allSatisfy({
                $0.count == inputWidth && $0.allSatisfy(\.isFinite)
                    && ControlledEvasionMath.norm($0) > ControlledEvasionMath.epsilon
            })
        else { return nil }

        self.outputFactors = outputDirections.map(ControlledEvasionMath.normalized)
        self.inputFactors = conditionDirections.map {
            benignProjector.apply(to: ControlledEvasionMath.normalized($0))
        }
        self.scales = scales
        self.outputWidth = outputWidth
        self.inputWidth = inputWidth
    }

    public func delta(for input: [Float]) -> [Float] {
        precondition(input.count == inputWidth)
        var result = Array(repeating: Float.zero, count: outputWidth)
        for factorIndex in 0 ..< rank {
            let response = scales[factorIndex]
                * ControlledEvasionMath.dot(inputFactors[factorIndex], input)
            for outputIndex in result.indices {
                result[outputIndex] += response * outputFactors[factorIndex][outputIndex]
            }
        }
        return result
    }

    public func deltaMatrix() -> [Float] {
        var result = Array(
            repeating: Float.zero, count: outputWidth * inputWidth)
        for factorIndex in 0 ..< rank {
            for row in 0 ..< outputWidth {
                for column in 0 ..< inputWidth {
                    result[row * inputWidth + column] += scales[factorIndex]
                        * outputFactors[factorIndex][row]
                        * inputFactors[factorIndex][column]
                }
            }
        }
        return result
    }

    public func applying(to matrix: [Float], rows: Int, columns: Int) -> [Float] {
        precondition(
            rows == outputWidth && columns == inputWidth
                && matrix.count == rows * columns)
        return zip(matrix, deltaMatrix()).map(+)
    }
}

public struct InterventionGuardrails: Codable, Sendable, Equatable {
    public var maximumFirstTokenKL: Double
    public var maximumControlFailureRate: Double
    public var maximumAddedControlFailureRate: Double
    public var controlPenalty: Double
    public var utilityFloor: Double

    public init(
        maximumFirstTokenKL: Double = 0.5,
        maximumControlFailureRate: Double = 0.10,
        maximumAddedControlFailureRate: Double = 0.02,
        controlPenalty: Double = 1,
        utilityFloor: Double = 1e-6
    ) {
        self.maximumFirstTokenKL = maximumFirstTokenKL
        self.maximumControlFailureRate = maximumControlFailureRate
        self.maximumAddedControlFailureRate = maximumAddedControlFailureRate
        self.controlPenalty = controlPenalty
        self.utilityFloor = utilityFloor
    }
}

/// A higher-is-better candidate score that exists only when both KL and benign
/// control gates pass. Keeping rejected candidates at `nil` prevents a large
/// refusal reduction from numerically buying its way past a preservation gate.
public struct InterventionCandidateScore: Codable, Sendable, Equatable {
    public let refusalReduction: Double
    public let addedControlFailureRate: Double
    public let firstTokenKL: Double
    public let passesGuardrails: Bool
    public let value: Double?

    public static func evaluate(
        baselineRefusalRate: Double, candidateRefusalRate: Double,
        baselineControlFailureRate: Double,
        candidateControlFailureRate: Double, firstTokenKL: Double,
        guardrails: InterventionGuardrails = .init()
    ) -> InterventionCandidateScore {
        let inputs = [
            baselineRefusalRate, candidateRefusalRate,
            baselineControlFailureRate, candidateControlFailureRate, firstTokenKL,
            guardrails.maximumFirstTokenKL,
            guardrails.maximumControlFailureRate,
            guardrails.maximumAddedControlFailureRate,
            guardrails.controlPenalty, guardrails.utilityFloor,
        ]
        let rates = [
            baselineRefusalRate, candidateRefusalRate,
            baselineControlFailureRate, candidateControlFailureRate,
        ]
        let valid = inputs.allSatisfy { $0.isFinite && $0 >= 0 }
            && rates.allSatisfy { $0 <= 1 }
            && guardrails.utilityFloor > 0
        let refusalReduction = valid
            ? max(0, baselineRefusalRate - candidateRefusalRate) : 0
        let addedControl = valid
            ? max(0, candidateControlFailureRate - baselineControlFailureRate) : 0
        let passes = valid
            && firstTokenKL <= guardrails.maximumFirstTokenKL
            && candidateControlFailureRate <= guardrails.maximumControlFailureRate
            && addedControl <= guardrails.maximumAddedControlFailureRate
        let denominator = max(
            guardrails.utilityFloor,
            firstTokenKL + guardrails.controlPenalty * addedControl)
        return InterventionCandidateScore(
            refusalReduction: refusalReduction,
            addedControlFailureRate: addedControl,
            firstTokenKL: firstTokenKL,
            passesGuardrails: passes,
            value: passes ? refusalReduction / denominator : nil)
    }
}

private enum ControlledEvasionMath {
    static let epsilon: Float = 1e-7

    static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        return lhs.indices.reduce(Float.zero) { $0 + lhs[$1] * rhs[$1] }
    }

    static func norm(_ vector: [Float]) -> Float { sqrt(dot(vector, vector)) }

    static func normalized(_ vector: [Float]) -> [Float] {
        let length = norm(vector)
        guard length > epsilon else { return vector }
        return vector.map { $0 / length }
    }

    static func mean(_ vectors: [[Float]]) -> [Float] {
        guard let first = vectors.first else { return [] }
        var result = Array(repeating: Float.zero, count: first.count)
        for vector in vectors {
            for index in result.indices { result[index] += vector[index] }
        }
        let scale = 1 / Float(vectors.count)
        return result.map { $0 * scale }
    }

    /// Twice-reorthogonalized modified Gram-Schmidt avoids the loss of
    /// orthogonality that otherwise accumulates in block power iteration.
    static func orthonormalized(
        _ vectors: [[Float]], limit: Int? = nil
    ) -> [[Float]] {
        var result = [[Float]]()
        for vector in vectors {
            var candidate = vector
            for _ in 0 ..< 2 {
                for existing in result {
                    let coefficient = dot(existing, candidate)
                    for index in candidate.indices {
                        candidate[index] -= coefficient * existing[index]
                    }
                }
            }
            let length = norm(candidate)
            if length > epsilon { result.append(candidate.map { $0 / length }) }
            if let limit, result.count == limit { break }
        }
        return result
    }

    static func pivotedRowBasis(_ rows: [[Float]], limit: Int) -> [[Float]] {
        var result = [[Float]]()
        while result.count < limit {
            var bestCandidate: [Float]?
            var bestNormSquared = Float.zero
            for row in rows {
                var candidate = row
                for _ in 0 ..< 2 {
                    for existing in result {
                        let coefficient = dot(existing, candidate)
                        for index in candidate.indices {
                            candidate[index] -= coefficient * existing[index]
                        }
                    }
                }
                let normSquared = dot(candidate, candidate)
                if normSquared > bestNormSquared {
                    bestNormSquared = normSquared
                    bestCandidate = candidate
                }
            }
            guard let bestCandidate, bestNormSquared > epsilon * epsilon else { break }
            result.append(bestCandidate.map { $0 / sqrt(bestNormSquared) })
        }
        return result
    }

    static func covarianceProduct(keys: [[Float]], vector: [Float]) -> [Float] {
        var result = Array(repeating: Float.zero, count: vector.count)
        for key in keys {
            let coefficient = dot(key, vector)
            for index in result.indices { result[index] += coefficient * key[index] }
        }
        let scale = 1 / Float(keys.count)
        return result.map { $0 * scale }
    }

    /// Jacobi rotations are used only on the small Rayleigh-Ritz matrix (rank
    /// plus oversampling), keeping this path independent of MLX/LAPACK.
    static func symmetricEigendecomposition(
        _ matrix: [Double], size: Int
    ) -> (values: [Double], vectors: [Double]) {
        precondition(matrix.count == size * size)
        guard size > 1 else { return ([matrix[0]], [1]) }
        var values = matrix
        var vectors = Array(repeating: Double.zero, count: size * size)
        for index in 0 ..< size { vectors[index * size + index] = 1 }

        let maximumRotations = max(32, 12 * size * size)
        for _ in 0 ..< maximumRotations {
            var pivotRow = 0
            var pivotColumn = 1
            var largest = abs(values[pivotColumn])
            for row in 0 ..< size {
                for column in (row + 1) ..< size {
                    let magnitude = abs(values[row * size + column])
                    if magnitude > largest {
                        largest = magnitude
                        pivotRow = row
                        pivotColumn = column
                    }
                }
            }
            guard largest > 1e-12 else { break }

            let p = pivotRow
            let q = pivotColumn
            let app = values[p * size + p]
            let aqq = values[q * size + q]
            let apq = values[p * size + q]
            let tau = (aqq - app) / (2 * apq)
            let sign = tau >= 0 ? 1.0 : -1.0
            let tangent = sign / (abs(tau) + sqrt(1 + tau * tau))
            let cosine = 1 / sqrt(1 + tangent * tangent)
            let sine = tangent * cosine

            for index in 0 ..< size where index != p && index != q {
                let aip = values[index * size + p]
                let aiq = values[index * size + q]
                let rotatedP = cosine * aip - sine * aiq
                let rotatedQ = sine * aip + cosine * aiq
                values[index * size + p] = rotatedP
                values[p * size + index] = rotatedP
                values[index * size + q] = rotatedQ
                values[q * size + index] = rotatedQ
            }
            values[p * size + p] = app - tangent * apq
            values[q * size + q] = aqq + tangent * apq
            values[p * size + q] = 0
            values[q * size + p] = 0

            for row in 0 ..< size {
                let vip = vectors[row * size + p]
                let viq = vectors[row * size + q]
                vectors[row * size + p] = cosine * vip - sine * viq
                vectors[row * size + q] = sine * vip + cosine * viq
            }
        }
        return ((0 ..< size).map { values[$0 * size + $0] }, vectors)
    }
}
