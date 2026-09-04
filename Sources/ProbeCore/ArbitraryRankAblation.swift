import Foundation
import MLX

/// Row constraint used while optimizing an ARA matrix.  `.none` matches an
/// unconstrained full-matrix edit; `.full` reparameterizes every row so its
/// L2 norm remains equal to the source checkpoint.
public enum ARARowNormalization: String, Codable, Sendable {
    case none
    case full
}

/// Heretic-style Arbitrary-Rank Ablation parameters. Layer indices are
/// zero-based and `endLayerIndex` is exclusive.
public struct ARAParameters: Codable, Equatable, Sendable {
    public var startLayerIndex: Int
    public var endLayerIndex: Int
    public var preserveGoodBehaviorWeight: Float
    public var steerBadBehaviorWeight: Float
    public var overcorrectRelativeWeight: Float
    public var neighborCount: Int
    public var rowNormalization: ARARowNormalization

    public init(
        startLayerIndex: Int,
        endLayerIndex: Int,
        preserveGoodBehaviorWeight: Float,
        steerBadBehaviorWeight: Float,
        overcorrectRelativeWeight: Float,
        neighborCount: Int,
        rowNormalization: ARARowNormalization = .none
    ) {
        precondition(startLayerIndex >= 0 && endLayerIndex > startLayerIndex)
        precondition(preserveGoodBehaviorWeight >= 0 && steerBadBehaviorWeight >= 0)
        precondition(overcorrectRelativeWeight >= 0 && neighborCount > 0)
        self.startLayerIndex = startLayerIndex
        self.endLayerIndex = endLayerIndex
        self.preserveGoodBehaviorWeight = preserveGoodBehaviorWeight
        self.steerBadBehaviorWeight = steerBadBehaviorWeight
        self.overcorrectRelativeWeight = overcorrectRelativeWeight
        self.neighborCount = neighborCount
        self.rowNormalization = rowNormalization
    }

    /// Parameters reported for the public Gemma 4 E2B Heretic checkpoint.
    /// This is a reproducibility seed, not a universal layer window.
    public static let gemma4E2BHereticSeed = ARAParameters(
        startLayerIndex: 17,
        endLayerIndex: 24,
        preserveGoodBehaviorWeight: 0.5767,
        steerBadBehaviorWeight: 0.0003,
        overcorrectRelativeWeight: 1.2505,
        neighborCount: 4,
        rowNormalization: .none)

    /// Uses a reference seed's optimizer weights without inheriting its
    /// model-specific layer window. Layer localization must be repeated for
    /// each model and dataset, so the safe default exposes the full zero-based
    /// decoder stack and lets the caller provide an explicit shortlist.
    public static func fullStackDefault(
        layerCount: Int,
        objectiveSeed: ARAParameters = gemma4E2BHereticSeed
    ) -> ARAParameters {
        precondition(layerCount > 0)
        return ARAParameters(
            startLayerIndex: 0,
            endLayerIndex: layerCount,
            preserveGoodBehaviorWeight: objectiveSeed.preserveGoodBehaviorWeight,
            steerBadBehaviorWeight: objectiveSeed.steerBadBehaviorWeight,
            overcorrectRelativeWeight: objectiveSeed.overcorrectRelativeWeight,
            neighborCount: objectiveSeed.neighborCount,
            rowNormalization: objectiveSeed.rowNormalization)
    }
}

public struct ARAModuleSamples {
    public let goodInput: MLXArray
    public let goodOutput: MLXArray
    public let badInput: MLXArray
    public let badOutput: MLXArray

    public init(
        goodInput: MLXArray, goodOutput: MLXArray,
        badInput: MLXArray, badOutput: MLXArray
    ) {
        precondition(goodInput.ndim == 2 && goodOutput.ndim == 2)
        precondition(badInput.ndim == 2 && badOutput.ndim == 2)
        precondition(goodInput.dim(0) == goodOutput.dim(0))
        precondition(badInput.dim(0) == badOutput.dim(0))
        precondition(goodInput.dim(1) == badInput.dim(1))
        precondition(goodOutput.dim(1) == badOutput.dim(1))
        self.goodInput = goodInput.asType(.float32)
        self.goodOutput = goodOutput.asType(.float32)
        self.badInput = badInput.asType(.float32)
        self.badOutput = badOutput.asType(.float32)
    }
}

public struct ARAOptimizationStep: Codable, Equatable, Sendable {
    public let iteration: Int
    public let loss: Double
    public let stepSize: Double
    public let gradientNorm: Double
}

public struct ARAOptimizationResult {
    public let weight: MLXArray
    public let steps: [ARAOptimizationStep]

    public init(weight: MLXArray, steps: [ARAOptimizationStep]) {
        self.weight = weight
        self.steps = steps
    }
}

/// Full-matrix ARA objective and a deterministic limited-memory BFGS solver.
/// The line search enforces the strong Wolfe conditions and therefore mirrors
/// the optimizer family used by Heretic without depending on Python/PyTorch.
public enum ArbitraryRankAblation {
    public static func meanDistancesToKNearestNeighbors(
        _ queries: MLXArray, among references: MLXArray, k: Int
    ) -> MLXArray {
        precondition(queries.ndim == 2 && references.ndim == 2)
        precondition(queries.dim(1) == references.dim(1))
        precondition(k > 0 && k <= references.dim(0))
        let difference = queries.expandedDimensions(axis: 1)
            - references.expandedDimensions(axis: 0)
        // MLX's raw sqrt derivative is undefined at an exact zero distance.
        // PyTorch cdist (used by Heretic) defines that coincident-point gradient
        // as zero, so a tiny floor gives the same practical behavior and avoids
        // 0 * NaN when the overcorrection coefficient is zero.
        let distances = sqrt(maximum(
            (difference * difference).sum(axis: 2), MLXArray(1e-12 as Float)))
        let indices = argPartition(distances, kth: k - 1, axis: 1)[0..., 0 ..< k]
        return takeAlong(distances, indices, axis: 1).mean(axis: 1)
    }

    public static func objective(
        weight: MLXArray,
        samples: ARAModuleSamples,
        parameters: ARAParameters,
        originalRowNorms: MLXArray? = nil
    ) -> MLXArray {
        let effective = effectiveWeight(
            weight.asType(.float32), normalization: parameters.rowNormalization,
            originalRowNorms: originalRowNorms)
        precondition(effective.ndim == 2)
        precondition(samples.goodInput.dim(1) == effective.dim(1))
        precondition(samples.goodOutput.dim(1) == effective.dim(0))
        precondition(parameters.neighborCount <= samples.goodOutput.dim(0))
        precondition(parameters.neighborCount <= samples.badOutput.dim(0))

        let newGoodOutput = matmul(samples.goodInput, effective.T)
        let newBadOutput = matmul(samples.badInput, effective.T)
        let goodError = newGoodOutput - samples.goodOutput
        let preserveGood = (goodError * goodError).mean()
        let pullToGood = meanDistancesToKNearestNeighbors(
            newBadOutput, among: samples.goodOutput,
            k: parameters.neighborCount).mean()
        let pushFromBad = meanDistancesToKNearestNeighbors(
            newBadOutput, among: samples.badOutput,
            k: parameters.neighborCount).mean()
        return parameters.preserveGoodBehaviorWeight * preserveGood
            + parameters.steerBadBehaviorWeight
                * (pullToGood - parameters.overcorrectRelativeWeight * pushFromBad)
    }

    public static func optimize(
        originalWeight: MLXArray,
        samples: ARAModuleSamples,
        parameters: ARAParameters,
        maximumIterations: Int = 100,
        historySize: Int = 10,
        initialStepSize: Float = 1,
        gradientTolerance: Float = 1e-5,
        changeTolerance: Float = 1e-9,
        maximumLineSearchIterations: Int = 24,
        maximumRelativeChange: Float? = nil
    ) -> ARAOptimizationResult {
        precondition(originalWeight.ndim == 2)
        precondition(maximumIterations > 0 && historySize > 0)
        precondition(maximumRelativeChange == nil || maximumRelativeChange! > 0)
        let sourceType = originalWeight.dtype
        let original = stopGradient(originalWeight.asType(.float32))
        var x = original
        let originalNorm = sqrt(max(dot(original, original), 1e-24))
        let trustRadius = maximumRelativeChange.map { Double($0) * originalNorm }
        let rowNorms = stopGradient(sqrt(
            (x * x).sum(axis: 1, keepDims: true)))
        let valueGradient = valueAndGrad { arrays in
            [objective(
                weight: arrays[0], samples: samples, parameters: parameters,
                originalRowNorms: rowNorms)]
        }

        func evaluate(_ point: MLXArray) -> Evaluation {
            let (values, gradients) = valueGradient([point])
            let value = values[0]
            let gradient = gradients[0]
            eval(value, gradient)
            return Evaluation(
                point: stopGradient(point), value: Double(value.item(Float.self)),
                gradient: stopGradient(gradient))
        }

        var current = evaluate(x)
        var history = [HistoryPair]()
        var trace = [ARAOptimizationStep]()
        trace.reserveCapacity(maximumIterations)

        for iteration in 0 ..< maximumIterations {
            let gradientNorm = Double(
                sqrt((current.gradient * current.gradient).sum()).item(Float.self))
            if gradientNorm <= Double(gradientTolerance) { break }

            var direction = twoLoopDirection(
                gradient: current.gradient, history: history)
            var directionalDerivative = dot(current.gradient, direction)
            if directionalDerivative >= 0 || !directionalDerivative.isFinite {
                history.removeAll(keepingCapacity: true)
                direction = -current.gradient
                directionalDerivative = dot(current.gradient, direction)
            }

            let maximumStep: Double?
            if let trustRadius {
                maximumStep = trustRegionMaximumStep(
                    origin: original, current: current.point,
                    direction: direction, radius: trustRadius)
                if maximumStep! <= 1e-12 { break }
            } else {
                maximumStep = nil
            }

            guard let accepted = strongWolfeLineSearch(
                current: current, direction: direction,
                initialStep: Double(initialStepSize),
                directionalDerivative: directionalDerivative,
                maximumIterations: maximumLineSearchIterations,
                maximumStep: maximumStep,
                evaluate: { alpha in evaluate(current.point + Float(alpha) * direction) })
            else { break }

            let s = stopGradient(accepted.evaluation.point - current.point)
            let y = stopGradient(accepted.evaluation.gradient - current.gradient)
            eval(s, y)
            let curvature = dot(s, y)
            if curvature > 1e-10 && curvature.isFinite {
                history.append(HistoryPair(
                    s: s, y: y, inverseCurvature: MLXArray(Float(1 / curvature))))
                if history.count > historySize { history.removeFirst() }
            }

            let lossChange = abs(current.value - accepted.evaluation.value)
            current = accepted.evaluation
            x = current.point
            trace.append(ARAOptimizationStep(
                iteration: iteration + 1,
                loss: current.value,
                stepSize: accepted.step,
                gradientNorm: gradientNorm))
            if let trustRadius {
                let delta = current.point - original
                let distance = sqrt(max(dot(delta, delta), 0))
                // A boundary point is the constrained result along the accepted
                // descent path. Stop here instead of optimizing far outside the
                // ball and shrinking an unrelated final solution afterward.
                if distance >= trustRadius * (1 - 1e-6) { break }
            }
            if lossChange <= Double(changeTolerance) { break }
        }

        let result = effectiveWeight(
            x, normalization: parameters.rowNormalization,
            originalRowNorms: rowNorms).asType(sourceType)
        eval(result)
        return ARAOptimizationResult(weight: result, steps: trace)
    }

    private struct Evaluation {
        let point: MLXArray
        let value: Double
        let gradient: MLXArray
    }

    private struct HistoryPair {
        let s: MLXArray
        let y: MLXArray
        let inverseCurvature: MLXArray
    }

    private struct LineSearchResult {
        let step: Double
        let evaluation: Evaluation
    }

    private static func effectiveWeight(
        _ weight: MLXArray, normalization: ARARowNormalization,
        originalRowNorms: MLXArray?
    ) -> MLXArray {
        guard normalization == .full else { return weight }
        let norms = originalRowNorms ?? sqrt(
            (weight * weight).sum(axis: 1, keepDims: true))
        let currentNorms = sqrt((weight * weight).sum(axis: 1, keepDims: true))
        return weight * norms / maximum(currentNorms, MLXArray(1e-12 as Float))
    }

    private static func dot(_ first: MLXArray, _ second: MLXArray) -> Double {
        let value = (first * second).sum()
        eval(value)
        return Double(value.item(Float.self))
    }

    private static func twoLoopDirection(
        gradient: MLXArray, history: [HistoryPair]
    ) -> MLXArray {
        guard !history.isEmpty else { return -gradient }
        var q = gradient
        var alphas = [MLXArray]()
        alphas.reserveCapacity(history.count)
        for pair in history.reversed() {
            let alpha = pair.inverseCurvature * (pair.s * q).sum()
            alphas.append(alpha)
            q = q - alpha * pair.y
        }
        let latest = history[history.count - 1]
        let scale = (latest.s * latest.y).sum()
            / maximum((latest.y * latest.y).sum(), MLXArray(1e-12 as Float))
        var r = scale * q
        for (offset, pair) in history.enumerated() {
            let beta = pair.inverseCurvature * (pair.y * r).sum()
            let alpha = alphas[history.count - 1 - offset]
            r = r + pair.s * (alpha - beta)
        }
        return stopGradient(-r)
    }

    private static func strongWolfeLineSearch(
        current: Evaluation,
        direction: MLXArray,
        initialStep: Double,
        directionalDerivative: Double,
        maximumIterations: Int,
        maximumStep: Double?,
        evaluate: (Double) -> Evaluation
    ) -> LineSearchResult? {
        let c1 = 1e-4
        let c2 = 0.9
        var previousStep = 0.0
        var previous = current
        let upperBound = maximumStep ?? .infinity
        var step = min(max(initialStep, 1e-8), upperBound)

        for iteration in 0 ..< maximumIterations {
            let candidate = evaluate(step)
            if candidate.value > current.value + c1 * step * directionalDerivative
                || (iteration > 0 && candidate.value >= previous.value)
            {
                return zoom(
                    lowerStep: previousStep, lower: previous,
                    upperStep: step, upper: candidate,
                    current: current, direction: direction,
                    directionalDerivative: directionalDerivative,
                    maximumIterations: maximumIterations,
                    evaluate: evaluate)
            }
            let candidateDerivative = dot(candidate.gradient, direction)
            if abs(candidateDerivative) <= -c2 * directionalDerivative {
                return LineSearchResult(step: step, evaluation: candidate)
            }
            if step >= upperBound * (1 - 1e-12) {
                // The unconstrained Wolfe curvature condition need not hold at
                // an active trust boundary. Armijo plus improvement is enough.
                return candidate.value < current.value
                    ? LineSearchResult(step: step, evaluation: candidate)
                    : nil
            }
            if candidateDerivative >= 0 {
                return zoom(
                    lowerStep: step, lower: candidate,
                    upperStep: previousStep, upper: previous,
                    current: current, direction: direction,
                    directionalDerivative: directionalDerivative,
                    maximumIterations: maximumIterations,
                    evaluate: evaluate)
            }
            previousStep = step
            previous = candidate
            step = min(step * 2, upperBound)
        }
        return nil
    }

    /// Largest non-negative step that keeps `current + step * direction`
    /// inside the Frobenius ball centered on `origin`.
    private static func trustRegionMaximumStep(
        origin: MLXArray, current: MLXArray,
        direction: MLXArray, radius: Double
    ) -> Double {
        let offset = current - origin
        let a = dot(direction, direction)
        guard a > 0, a.isFinite else { return 0 }
        let b = dot(offset, direction)
        let c = dot(offset, offset) - radius * radius
        let discriminant = max(b * b - a * c, 0)
        return max((-b + sqrt(discriminant)) / a, 0)
    }

    private static func zoom(
        lowerStep initialLowerStep: Double,
        lower initialLower: Evaluation,
        upperStep initialUpperStep: Double,
        upper initialUpper: Evaluation,
        current: Evaluation,
        direction: MLXArray,
        directionalDerivative: Double,
        maximumIterations: Int,
        evaluate: (Double) -> Evaluation
    ) -> LineSearchResult? {
        let c1 = 1e-4
        let c2 = 0.9
        var lowerStep = initialLowerStep
        var lower = initialLower
        var upperStep = initialUpperStep
        var upper = initialUpper
        for _ in 0 ..< maximumIterations {
            let step = 0.5 * (lowerStep + upperStep)
            let candidate = evaluate(step)
            if candidate.value > current.value + c1 * step * directionalDerivative
                || candidate.value >= lower.value
            {
                upperStep = step
                upper = candidate
            } else {
                let candidateDerivative = dot(candidate.gradient, direction)
                if abs(candidateDerivative) <= -c2 * directionalDerivative {
                    return LineSearchResult(step: step, evaluation: candidate)
                }
                if candidateDerivative * (upperStep - lowerStep) >= 0 {
                    upperStep = lowerStep
                    upper = lower
                }
                lowerStep = step
                lower = candidate
            }
            if abs(upperStep - lowerStep) <= 1e-12 {
                return lower.value <= upper.value
                    ? LineSearchResult(step: lowerStep, evaluation: lower)
                    : LineSearchResult(step: upperStep, evaluation: upper)
            }
        }
        return lower.value <= upper.value
            ? LineSearchResult(step: lowerStep, evaluation: lower)
            : LineSearchResult(step: upperStep, evaluation: upper)
    }
}
