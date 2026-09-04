import Foundation
import MLX
import MLXLinalg

/// Closed-form residual-stream maps that target the distribution of genuine
/// answers instead of the origin of activation space.
public enum AnswerCenteredTransportMethod: String, Codable, Sendable {
    /// Classic affine concept editing: replace the coordinate along the
    /// refusal/answer centroid axis with the answered centroid coordinate.
    case affineCentroid = "affine-centroid"
    /// Gaussian 2-Wasserstein transport in a pooled-PCA subspace.
    case pcaGaussianOT = "pca-gaussian-ot"
}

/// Derive-time gate policy. This never refits or changes transport geometry.
public enum AnswerCenteredTransportGateOverride: String, Codable, Sendable {
    /// Retain the gate encoded in the source artifact.
    case preserve = "keep"
    /// Apply the fitted affine map unconditionally.
    case disable = "none"
}

/// Optional target-model gate for an answer-centered residual map.
///
/// The probe is fitted at the application site. `margin` is normally a high
/// quantile of scores from answered/benign states, so the map remains closed
/// on almost all protected controls. A positive transition width makes the
/// gate a continuous ramp rather than a hard threshold.
public struct AnswerCenteredTransportGate: Codable, Sendable, Equatable {
    public let probe: ControlledEvasionProbe
    public let margin: Float
    public let transitionWidth: Float

    public init?(
        probe: ControlledEvasionProbe, margin: Float,
        transitionWidth: Float = 0
    ) {
        guard margin.isFinite, transitionWidth.isFinite,
              transitionWidth >= 0
        else { return nil }
        self.probe = probe
        self.margin = margin
        self.transitionWidth = transitionWidth
    }

    public func value(for activation: [Float]) -> Float {
        precondition(activation.count == probe.width)
        let excess = probe.score(activation) - margin
        guard excess > 0 else { return 0 }
        guard transitionWidth > 0 else { return 1 }
        return min(1, excess / transitionWidth)
    }
}

/// A low-rank affine transport expressed in row-orthonormal residual axes.
///
/// For row-vector hidden state `h`, row basis `B`, source/target projected
/// means `muR`/`muA`, and reduced transport matrix `A`, the full-strength map
/// is
///
/// ```
/// z  = h B^T
/// z' = (z - muR) A^T + muA
/// h' = h + (z' - z) B
/// ```
///
/// The last line is the identity-preserving lift. It is deliberately not
/// `B^T A B h + b`, which would erase the entire orthogonal complement despite
/// occasionally appearing that way in abbreviated PCA-OT descriptions.
public struct AnswerCenteredTransportPlan: Codable, Sendable, Equatable {
    public let method: AnswerCenteredTransportMethod
    public let basis: [[Float]]
    public let sourceMeanCoordinates: [Float]
    public let targetMeanCoordinates: [Float]
    /// Row-major reduced matrix. `transportMatrix[row][column]` is A[r,c].
    public let transportMatrix: [[Float]]
    public let gate: AnswerCenteredTransportGate?
    public let strength: Float

    public var rank: Int { basis.count }
    public var width: Int { basis.first?.count ?? 0 }

    public init?(
        method: AnswerCenteredTransportMethod,
        basis: [[Float]],
        sourceMeanCoordinates: [Float],
        targetMeanCoordinates: [Float],
        transportMatrix: [[Float]],
        gate: AnswerCenteredTransportGate? = nil,
        strength: Float = 1
    ) {
        guard let width = basis.first?.count, width > 0,
              !basis.isEmpty,
              basis.allSatisfy({
                  $0.count == width && $0.allSatisfy(\.isFinite)
              }),
              sourceMeanCoordinates.count == basis.count,
              targetMeanCoordinates.count == basis.count,
              sourceMeanCoordinates.allSatisfy(\.isFinite),
              targetMeanCoordinates.allSatisfy(\.isFinite),
              transportMatrix.count == basis.count,
              transportMatrix.allSatisfy({
                  $0.count == basis.count && $0.allSatisfy(\.isFinite)
              }),
              gate.map({ $0.probe.width == width }) ?? true,
              strength.isFinite, strength >= 0
        else { return nil }
        let orthonormal = AnswerCenteredTransportMath.isOrthonormal(basis)
        guard orthonormal else { return nil }
        self.method = method
        self.basis = basis
        self.sourceMeanCoordinates = sourceMeanCoordinates
        self.targetMeanCoordinates = targetMeanCoordinates
        self.transportMatrix = transportMatrix
        self.gate = gate
        self.strength = strength
    }

    public func gateValue(for activation: [Float]) -> Float {
        gate?.value(for: activation) ?? 1
    }

    /// Pure-array reference implementation used for unit tests and artifact
    /// inspection. The deployed MLX transform below implements the same map.
    public func apply(
        gateActivation: [Float], to activation: [Float]
    ) -> [Float] {
        precondition(gateActivation.count == width)
        precondition(activation.count == width)
        let amount = strength * gateValue(for: gateActivation)
        guard amount != 0 else { return activation }
        let coordinates = basis.map {
            AnswerCenteredTransportMath.dot($0, activation)
        }
        let centered = zip(coordinates, sourceMeanCoordinates).map(-)
        let mapped = transportMatrix.indices.map { row in
            targetMeanCoordinates[row]
                + AnswerCenteredTransportMath.dot(
                    transportMatrix[row], centered)
        }
        var result = activation
        for axis in basis.indices {
            let coordinateDelta = amount * (mapped[axis] - coordinates[axis])
            for index in result.indices {
                result[index] += coordinateDelta * basis[axis][index]
            }
        }
        return result
    }
}

public struct AnswerCenteredTransportFitDiagnostics: Codable, Sendable,
    Equatable
{
    public let sampleCountRefused: Int
    public let sampleCountAnswered: Int
    public let explainedVarianceFraction: Float
    public let covarianceRidge: Float
    public let projectedMeanAlignmentRMSE: Float
    public let projectedCovarianceAlignmentRMSE: Float
    /// Mean fitted gate values are absent in artifacts written before gate
    /// telemetry was introduced. `nil` means unavailable, not a closed gate.
    public let meanRefusedGate: Float?
    public let meanAnsweredGate: Float?
}

public struct AnswerCenteredTransportFit: Codable, Sendable, Equatable {
    public let plan: AnswerCenteredTransportPlan
    public let diagnostics: AnswerCenteredTransportFitDiagnostics
}

/// Reproducible development artifact for one fitted answer-centered map.
/// This is a candidate to evaluate, never an abliteration certificate.
public struct AnswerCenteredTransportArtifact: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let status: String
    public let warning: String
    public let modelPath: String
    public let measurementPath: String
    public let parentArtifactPath: String?
    public let derivationGateOverride: AnswerCenteredTransportGateOverride?
    public let pairNames: [String]
    public let tokenPosition: ActivationTokenPosition
    public let layerZeroBased: Int
    public let fit: AnswerCenteredTransportFit
    public let intervention: AnswerCenteredResidualIntervention

    public init(
        modelPath: String, measurementPath: String,
        parentArtifactPath: String? = nil,
        derivationGateOverride: AnswerCenteredTransportGateOverride? = nil,
        pairNames: [String], tokenPosition: ActivationTokenPosition,
        layerZeroBased: Int, fit: AnswerCenteredTransportFit,
        intervention: AnswerCenteredResidualIntervention
    ) {
        schemaVersion = 1
        status = "development_candidate"
        warning = "This fitted map is not an abliteration certificate. Judge dev responses semantically and enforce control KL before promotion; do not fit on frozen audits."
        self.modelPath = URL(fileURLWithPath: modelPath)
            .standardizedFileURL.path
        self.measurementPath = URL(fileURLWithPath: measurementPath)
            .standardizedFileURL.path
        self.parentArtifactPath = parentArtifactPath.map {
            URL(fileURLWithPath: $0).standardizedFileURL.path
        }
        self.derivationGateOverride = derivationGateOverride
        self.pairNames = pairNames
        self.tokenPosition = tokenPosition
        self.layerZeroBased = layerZeroBased
        self.fit = fit
        self.intervention = intervention
    }

    public func write(to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(
            to: URL(fileURLWithPath: path).standardizedFileURL,
            options: .atomic)
    }

    public static func read(from path: String) throws -> Self {
        let result = try JSONDecoder().decode(
            Self.self,
            from: Data(contentsOf:
                URL(fileURLWithPath: path).standardizedFileURL))
        guard result.schemaVersion == 1,
              result.status == "development_candidate",
              result.layerZeroBased >= 0,
              result.intervention.layer == result.layerZeroBased + 1,
              result.intervention.plan == result.fit.plan,
              !result.pairNames.isEmpty
        else { throw AnswerCenteredTransportArtifactError.malformed(path) }
        return result
    }

    /// Produces a strength/timing sibling without loading the model or
    /// recapturing activations. Geometry and exact measurement provenance
    /// remain unchanged; callers may explicitly remove only the learned gate.
    public func derived(
        from parentArtifactPath: String,
        strength: Float,
        schedule: ExactResidualInterventionSchedule?,
        gateOverride: AnswerCenteredTransportGateOverride = .preserve
    ) -> Self? {
        let derivedGate = gateOverride == .disable ? nil : fit.plan.gate
        guard let plan = AnswerCenteredTransportPlan(
            method: fit.plan.method,
            basis: fit.plan.basis,
            sourceMeanCoordinates: fit.plan.sourceMeanCoordinates,
            targetMeanCoordinates: fit.plan.targetMeanCoordinates,
            transportMatrix: fit.plan.transportMatrix,
            gate: derivedGate,
            strength: strength),
              let intervention = AnswerCenteredResidualIntervention(
                layer: layerZeroBased + 1,
                plan: plan,
                schedule: schedule)
        else { return nil }
        let diagnostics: AnswerCenteredTransportFitDiagnostics
        if gateOverride == .disable {
            diagnostics = AnswerCenteredTransportFitDiagnostics(
                sampleCountRefused: fit.diagnostics.sampleCountRefused,
                sampleCountAnswered: fit.diagnostics.sampleCountAnswered,
                explainedVarianceFraction:
                    fit.diagnostics.explainedVarianceFraction,
                covarianceRidge: fit.diagnostics.covarianceRidge,
                projectedMeanAlignmentRMSE:
                    fit.diagnostics.projectedMeanAlignmentRMSE,
                projectedCovarianceAlignmentRMSE:
                    fit.diagnostics.projectedCovarianceAlignmentRMSE,
                meanRefusedGate: 1,
                meanAnsweredGate: 1)
        } else {
            diagnostics = fit.diagnostics
        }
        return Self(
            modelPath: modelPath,
            measurementPath: measurementPath,
            parentArtifactPath: parentArtifactPath,
            derivationGateOverride: gateOverride,
            pairNames: pairNames,
            tokenPosition: tokenPosition,
            layerZeroBased: layerZeroBased,
            fit: AnswerCenteredTransportFit(
                plan: plan, diagnostics: diagnostics),
            intervention: intervention)
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case status
        case warning
        case modelPath = "model_path"
        case measurementPath = "measurement_path"
        case parentArtifactPath = "parent_artifact_path"
        case derivationGateOverride = "derivation_gate_override"
        case pairNames = "pair_names"
        case tokenPosition = "token_position"
        case layerZeroBased = "layer_zero_based"
        case fit
        case intervention
    }
}

public enum AnswerCenteredTransportArtifactError: LocalizedError, Equatable {
    case malformed(String)

    public var errorDescription: String? {
        switch self {
        case .malformed(let path):
            "Malformed answer-centered transport artifact: \(path)"
        }
    }
}

/// Fits ACE or PCA-Gaussian-OT from outcome-labeled residual states.
///
/// `refusedActivations` and `answeredActivations` must be selected from actual
/// model outcomes on the development split. Dataset intent is not a substitute
/// for observing whether the model refused or answered. Frozen audit examples
/// must never be used here.
public enum AnswerCenteredTransportFitter {
    public static func fit(
        refusedActivations: [[Float]],
        answeredActivations: [[Float]],
        method: AnswerCenteredTransportMethod,
        rank requestedRank: Int = 2,
        covarianceRidge: Float = 1e-3,
        controlGateQuantile: Float? = 0.99,
        strength: Float = 1
    ) -> AnswerCenteredTransportFit? {
        guard let width = refusedActivations.first?.count,
              width > 0,
              refusedActivations.count >= 2,
              answeredActivations.count >= 2,
              requestedRank > 0,
              covarianceRidge.isFinite, covarianceRidge > 0,
              strength.isFinite, strength >= 0,
              controlGateQuantile.map({ (0 ... 1).contains($0) }) ?? true,
              (refusedActivations + answeredActivations).allSatisfy({
                  $0.count == width && $0.allSatisfy(\.isFinite)
              })
        else { return nil }

        let refusedMean = AnswerCenteredTransportMath.mean(
            refusedActivations)
        let answeredMean = AnswerCenteredTransportMath.mean(
            answeredActivations)
        let basis: [[Float]]
        let explainedVariance: Float
        switch method {
        case .affineCentroid:
            let difference = zip(refusedMean, answeredMean).map(-)
            let axis = AnswerCenteredTransportMath.normalized(difference)
            guard AnswerCenteredTransportMath.norm(axis) > 0.99 else {
                return nil
            }
            basis = [axis]
            explainedVariance = AnswerCenteredTransportMath
                .varianceFraction(in: basis, samples:
                    refusedActivations + answeredActivations)
        case .pcaGaussianOT:
            let maximumRank = min(
                requestedRank,
                min(width, refusedActivations.count
                    + answeredActivations.count - 1))
            guard let pca = AnswerCenteredTransportMath.pooledPCA(
                refusedActivations, answeredActivations, rank: maximumRank),
                  !pca.basis.isEmpty
            else { return nil }
            basis = pca.basis
            explainedVariance = pca.explainedVarianceFraction
        }

        let refusedCoordinates = refusedActivations.map { activation in
            basis.map { AnswerCenteredTransportMath.dot($0, activation) }
        }
        let answeredCoordinates = answeredActivations.map { activation in
            basis.map { AnswerCenteredTransportMath.dot($0, activation) }
        }
        let sourceMean = AnswerCenteredTransportMath.mean(refusedCoordinates)
        let targetMean = AnswerCenteredTransportMath.mean(answeredCoordinates)
        let sourceCovariance = AnswerCenteredTransportMath.covariance(
            refusedCoordinates, mean: sourceMean)
        let targetCovariance = AnswerCenteredTransportMath.covariance(
            answeredCoordinates, mean: targetMean)
        let reducedRank = basis.count

        let transport: [[Float]]
        switch method {
        case .affineCentroid:
            transport = [[0]]
        case .pcaGaussianOT:
            let averageVariance = max(
                AnswerCenteredTransportMath.trace(sourceCovariance)
                    + AnswerCenteredTransportMath.trace(targetCovariance),
                Float.ulpOfOne) / Float(2 * reducedRank)
            let absoluteRidge = covarianceRidge * averageVariance
            let regularizedSource = AnswerCenteredTransportMath
                .addingDiagonal(sourceCovariance, value: absoluteRidge)
            let regularizedTarget = AnswerCenteredTransportMath
                .addingDiagonal(targetCovariance, value: absoluteRidge)
            guard let map = AnswerCenteredTransportMath.gaussianMap(
                sourceCovariance: regularizedSource,
                targetCovariance: regularizedTarget)
            else { return nil }
            transport = map
        }

        let gate: AnswerCenteredTransportGate?
        if let controlGateQuantile {
            guard let probe = ControlledEvasionProbe.fit(
                positive: refusedActivations,
                negative: answeredActivations)
            else { return nil }
            let answeredScores = answeredActivations.map(probe.score).sorted()
            let refusedScores = refusedActivations.map(probe.score).sorted()
            guard let margin = AnswerCenteredTransportMath.quantile(
                answeredScores, controlGateQuantile),
                  let refusedMedian = AnswerCenteredTransportMath.quantile(
                    refusedScores, 0.5),
                  let fittedGate = AnswerCenteredTransportGate(
                    probe: probe, margin: margin,
                    transitionWidth: max(
                        abs(refusedMedian - margin), 1e-4))
            else { return nil }
            gate = fittedGate
        } else {
            gate = nil
        }

        guard let plan = AnswerCenteredTransportPlan(
            method: method,
            basis: basis,
            sourceMeanCoordinates: sourceMean,
            targetMeanCoordinates: targetMean,
            transportMatrix: transport,
            gate: gate,
            strength: strength)
        else { return nil }

        let mappedMean = AnswerCenteredTransportMath.add(
            AnswerCenteredTransportMath.multiply(
                transport,
                AnswerCenteredTransportMath.subtract(sourceMean, sourceMean)),
            targetMean)
        let mappedCovariance = AnswerCenteredTransportMath.multiply(
            AnswerCenteredTransportMath.multiply(
                transport, sourceCovariance),
            AnswerCenteredTransportMath.transpose(transport))
        let meanRMSE = AnswerCenteredTransportMath.rmse(
            mappedMean, targetMean)
        let covarianceRMSE = AnswerCenteredTransportMath.rmse(
            mappedCovariance.flatMap { $0 },
            targetCovariance.flatMap { $0 })
        let meanRefusedGate = refusedActivations.map {
            plan.gateValue(for: $0)
        }
            .reduce(Float.zero, +) / Float(refusedActivations.count)
        let meanAnsweredGate = answeredActivations.map {
            plan.gateValue(for: $0)
        }
            .reduce(Float.zero, +) / Float(answeredActivations.count)
        return AnswerCenteredTransportFit(
            plan: plan,
            diagnostics: AnswerCenteredTransportFitDiagnostics(
                sampleCountRefused: refusedActivations.count,
                sampleCountAnswered: answeredActivations.count,
                explainedVarianceFraction: explainedVariance,
                covarianceRidge: covarianceRidge,
                projectedMeanAlignmentRMSE: meanRMSE,
                projectedCovarianceAlignmentRMSE: covarianceRMSE,
                meanRefusedGate: meanRefusedGate,
                meanAnsweredGate: meanAnsweredGate))
    }
}

/// One selected decoder-layer application of an answer-centered map.
public struct AnswerCenteredResidualIntervention: Codable, Sendable,
    Equatable
{
    /// One-based decoder layer, matching `ExactResidualIntervention` artifacts.
    public let layer: Int
    public let plan: AnswerCenteredTransportPlan
    public let schedule: ExactResidualInterventionSchedule?

    public init?(
        layer: Int, plan: AnswerCenteredTransportPlan,
        schedule: ExactResidualInterventionSchedule? = nil
    ) {
        guard layer > 0 else { return nil }
        self.layer = layer
        self.plan = plan
        self.schedule = schedule
    }

    public func makeTransform()
        -> @Sendable (_ zeroBasedLayer: Int, _ state: MLXArray) -> MLXArray
    {
        let selectedLayer = layer - 1
        let stickyGate = AnswerCenteredStickyPromptGate()
        let applicationSchedule = schedule ?? .backwardCompatible
        return { zeroBasedLayer, state in
            guard zeroBasedLayer == selectedLayer,
                  state.dim(-1) == plan.width
            else { return state }

            let working = state.asType(.float32)
            let tokenGate: MLXArray
            if let gate = plan.gate {
                let probe = MLXArray(gate.probe.weights)
                    .reshaped(gate.probe.width, 1).asType(.float32)
                let score = matmul(working, probe).squeezed(axis: -1)
                    + gate.probe.bias
                let excess = maximum(
                    score - gate.margin, MLXArray(Float.zero))
                if gate.transitionWidth > 0 {
                    tokenGate = clip(
                        excess / gate.transitionWidth, min: 0, max: 1)
                } else {
                    tokenGate = (excess .> 0).asType(.float32)
                }
            } else {
                tokenGate = MLXArray.ones(Array(state.shape.dropLast()))
                    .asType(.float32)
            }

            let applicationGate: MLXArray
            if state.dim(-2) > 1 {
                let latched = tokenGate[0, -1]
                eval(latched)
                stickyGate.beginPrefill(latched)
                if applicationSchedule.includePostInstruction {
                    applicationGate = concatenated([
                        MLXArray.zeros([1, state.dim(-2) - 1])
                            .asType(.float32),
                        latched.reshaped(1, 1).asType(.float32),
                    ], axis: -1)
                } else {
                    applicationGate = MLXArray.zeros(tokenGate.shape)
                        .asType(.float32)
                }
            } else {
                let (latched, generationIndex) =
                    stickyGate.beginDecodeToken()
                if applicationSchedule.appliesToGeneration(
                    index: generationIndex)
                {
                    applicationGate = latched ?? tokenGate
                } else {
                    applicationGate = MLXArray.zeros(tokenGate.shape)
                        .asType(.float32)
                }
            }

            let basis = MLXArray(plan.basis.flatMap { $0 })
                .reshaped(plan.rank, plan.width).asType(.float32)
            let sourceMean = MLXArray(plan.sourceMeanCoordinates)
                .asType(.float32)
            let targetMean = MLXArray(plan.targetMeanCoordinates)
                .asType(.float32)
            let transport = MLXArray(plan.transportMatrix.flatMap { $0 })
                .reshaped(plan.rank, plan.rank).asType(.float32)
            let coordinates = matmul(working, basis.T)
            let mapped = matmul(
                coordinates - sourceMean, transport.T) + targetMean
            let liftedDelta = matmul(mapped - coordinates, basis)
            let edited = working + plan.strength
                * applicationGate.expandedDimensions(axis: -1)
                * liftedDelta
            return edited.asType(state.dtype)
        }
    }

    public static func combinedTransform(
        _ interventions: [AnswerCenteredResidualIntervention]
    ) -> @Sendable (_ zeroBasedLayer: Int, _ state: MLXArray) -> MLXArray {
        let transforms = interventions.map { $0.makeTransform() }
        return { layer, state in
            transforms.reduce(state) { current, transform in
                transform(layer, current)
            }
        }
    }
}

extension ResidentTrialRuntime {
    public func install(
        _ intervention: AnswerCenteredResidualIntervention
    ) async throws {
        try await installResidualIntervention(intervention.makeTransform())
    }

    public func install(
        _ interventions: [AnswerCenteredResidualIntervention]
    ) async throws {
        try await installResidualIntervention(
            AnswerCenteredResidualIntervention.combinedTransform(
                interventions))
    }
}

private final class AnswerCenteredStickyPromptGate: @unchecked Sendable {
    private let lock = NSLock()
    private var value: MLXArray?
    private var generationIndex = 0

    func beginPrefill(_ value: MLXArray) {
        lock.lock()
        self.value = value
        generationIndex = 0
        lock.unlock()
    }

    func beginDecodeToken() -> (MLXArray?, Int) {
        lock.lock()
        defer { lock.unlock() }
        let index = generationIndex
        generationIndex += 1
        return (value, index)
    }
}

private enum AnswerCenteredTransportMath {
    static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        return lhs.indices.reduce(Float.zero) {
            $0 + lhs[$1] * rhs[$1]
        }
    }

    static func norm(_ vector: [Float]) -> Float {
        sqrt(max(dot(vector, vector), 0))
    }

    static func normalized(_ vector: [Float]) -> [Float] {
        let length = norm(vector)
        guard length > 1e-8 else { return vector }
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

    static func isOrthonormal(
        _ basis: [[Float]], tolerance: Float = 2e-3
    ) -> Bool {
        for row in basis.indices {
            for column in basis.indices {
                let expected: Float = row == column ? 1 : 0
                if abs(dot(basis[row], basis[column]) - expected) > tolerance {
                    return false
                }
            }
        }
        return true
    }

    static func pooledPCA(
        _ first: [[Float]], _ second: [[Float]], rank: Int
    ) -> (basis: [[Float]], explainedVarianceFraction: Float)? {
        let samples = first + second
        guard let width = samples.first?.count, !samples.isEmpty,
              rank > 0
        else { return nil }
        let pooledMean = mean(samples)
        let centered = samples.map { zip($0, pooledMean).map(-) }
        let matrix = MLXArray(centered.flatMap { $0 })
            .reshaped(centered.count, width).asType(.float32)
        let (_, singular, vt) = MLXLinalg.svd(matrix, stream: .cpu)
        let available = min(rank, min(vt.dim(0), width))
        guard available > 0 else { return nil }
        let selected = vt[0 ..< available, 0...].asType(.float32)
        eval(selected, singular)
        var basis = [[Float]]()
        for row in 0 ..< available {
            let vector = normalized(selected[row].asArray(Float.self))
            if norm(vector) > 0.99 { basis.append(vector) }
        }
        guard !basis.isEmpty else { return nil }
        let values = singular.asArray(Float.self).map { $0 * $0 }
        let total = values.reduce(Float.zero, +)
        let explained = total > 0
            ? values.prefix(basis.count).reduce(Float.zero, +) / total
            : 0
        return (basis, explained)
    }

    static func varianceFraction(
        in basis: [[Float]], samples: [[Float]]
    ) -> Float {
        let center = mean(samples)
        let centered = samples.map { zip($0, center).map(-) }
        let total = centered.reduce(Float.zero) {
            $0 + dot($1, $1)
        }
        guard total > 0 else { return 0 }
        let selected = centered.reduce(Float.zero) { total, vector in
            total + basis.reduce(Float.zero) {
                let coordinate = dot($1, vector)
                return $0 + coordinate * coordinate
            }
        }
        return selected / total
    }

    static func covariance(
        _ samples: [[Float]], mean: [Float]
    ) -> [[Float]] {
        let width = mean.count
        var result = Array(
            repeating: Array(repeating: Float.zero, count: width),
            count: width)
        let denominator = Float(max(1, samples.count - 1))
        for sample in samples {
            let centered = zip(sample, mean).map(-)
            for row in 0 ..< width {
                for column in row ..< width {
                    let value = centered[row] * centered[column]
                    result[row][column] += value
                    if row != column { result[column][row] += value }
                }
            }
        }
        return result.map { $0.map { $0 / denominator } }
    }

    static func trace(_ matrix: [[Float]]) -> Float {
        matrix.indices.reduce(Float.zero) { $0 + matrix[$1][$1] }
    }

    static func addingDiagonal(
        _ matrix: [[Float]], value: Float
    ) -> [[Float]] {
        var result = matrix
        for index in result.indices { result[index][index] += value }
        return result
    }

    static func gaussianMap(
        sourceCovariance: [[Float]], targetCovariance: [[Float]]
    ) -> [[Float]]? {
        guard sourceCovariance.count == targetCovariance.count,
              !sourceCovariance.isEmpty,
              sourceCovariance.allSatisfy({
                  $0.count == sourceCovariance.count
              }),
              targetCovariance.allSatisfy({
                  $0.count == targetCovariance.count
              }),
              let sourceRoot = symmetricPower(
                sourceCovariance, exponent: 0.5),
              let sourceInverseRoot = symmetricPower(
                sourceCovariance, exponent: -0.5)
        else { return nil }
        let middle = symmetrized(multiply(
            multiply(sourceRoot, targetCovariance), sourceRoot))
        guard let middleRoot = symmetricPower(middle, exponent: 0.5)
        else { return nil }
        let result = symmetrized(multiply(
            multiply(sourceInverseRoot, middleRoot),
            sourceInverseRoot))
        return result.flatMap { $0 }.allSatisfy(\.isFinite)
            ? result : nil
    }

    static func symmetricPower(
        _ matrix: [[Float]], exponent: Double
    ) -> [[Float]]? {
        let size = matrix.count
        guard size > 0, matrix.allSatisfy({ $0.count == size }) else {
            return nil
        }
        let decomposition = symmetricEigendecomposition(
            matrix.flatMap { $0 }.map(Double.init), size: size)
        let largest = decomposition.values.map(abs).max() ?? 0
        let floor = max(largest * 1e-10, 1e-12)
        var diagonal = Array(repeating: Double.zero, count: size)
        for index in 0 ..< size {
            let value = max(decomposition.values[index], floor)
            diagonal[index] = Foundation.pow(value, exponent)
            guard diagonal[index].isFinite else { return nil }
        }
        var result = Array(
            repeating: Array(repeating: Float.zero, count: size),
            count: size)
        // Eigenvectors are columns of V; reconstruct V D V^T.
        for row in 0 ..< size {
            for column in 0 ..< size {
                var value = Double.zero
                for axis in 0 ..< size {
                    value += decomposition.vectors[row * size + axis]
                        * diagonal[axis]
                        * decomposition.vectors[column * size + axis]
                }
                result[row][column] = Float(value)
            }
        }
        return result
    }

    static func symmetricEigendecomposition(
        _ matrix: [Double], size: Int
    ) -> (values: [Double], vectors: [Double]) {
        precondition(matrix.count == size * size)
        guard size > 1 else { return ([matrix[0]], [1]) }
        var values = matrix
        var vectors = Array(repeating: Double.zero, count: size * size)
        for index in 0 ..< size { vectors[index * size + index] = 1 }

        for _ in 0 ..< max(32, 16 * size * size) {
            var p = 0
            var q = 1
            var largest = abs(values[q])
            for row in 0 ..< size {
                for column in (row + 1) ..< size {
                    let magnitude = abs(values[row * size + column])
                    if magnitude > largest {
                        largest = magnitude
                        p = row
                        q = column
                    }
                }
            }
            if largest <= 1e-13 { break }
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

    static func multiply(
        _ lhs: [[Float]], _ rhs: [[Float]]
    ) -> [[Float]] {
        precondition(!lhs.isEmpty && !rhs.isEmpty)
        precondition(lhs[0].count == rhs.count)
        let rows = lhs.count
        let columns = rhs[0].count
        let inner = rhs.count
        var result = Array(
            repeating: Array(repeating: Float.zero, count: columns),
            count: rows)
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                for index in 0 ..< inner {
                    result[row][column] += lhs[row][index]
                        * rhs[index][column]
                }
            }
        }
        return result
    }

    static func multiply(
        _ matrix: [[Float]], _ vector: [Float]
    ) -> [Float] {
        matrix.map { dot($0, vector) }
    }

    static func transpose(_ matrix: [[Float]]) -> [[Float]] {
        guard let columns = matrix.first?.count else { return [] }
        return (0 ..< columns).map { column in
            matrix.map { $0[column] }
        }
    }

    static func symmetrized(_ matrix: [[Float]]) -> [[Float]] {
        let transposed = transpose(matrix)
        return matrix.indices.map { row in
            matrix[row].indices.map { column in
                0.5 * (matrix[row][column] + transposed[row][column])
            }
        }
    }

    static func add(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
        zip(lhs, rhs).map(+)
    }

    static func subtract(_ lhs: [Float], _ rhs: [Float]) -> [Float] {
        zip(lhs, rhs).map(-)
    }

    static func rmse(_ lhs: [Float], _ rhs: [Float]) -> Float {
        precondition(lhs.count == rhs.count)
        guard !lhs.isEmpty else { return 0 }
        let mse = zip(lhs, rhs).reduce(Float.zero) {
            let difference = $1.0 - $1.1
            return $0 + difference * difference
        } / Float(lhs.count)
        return sqrt(max(mse, 0))
    }

    static func quantile(
        _ sortedValues: [Float], _ probability: Float
    ) -> Float? {
        guard !sortedValues.isEmpty, (0 ... 1).contains(probability)
        else { return nil }
        guard sortedValues.count > 1 else { return sortedValues[0] }
        let position = Float(sortedValues.count - 1) * probability
        let lower = Int(floor(position))
        let upper = Int(ceil(position))
        let fraction = position - Float(lower)
        return sortedValues[lower] * (1 - fraction)
            + sortedValues[upper] * fraction
    }
}
