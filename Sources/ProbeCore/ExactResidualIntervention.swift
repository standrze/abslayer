import Foundation
import MLX

/// A serializable, one-based decoder-layer intervention. The runtime converts
/// this plan to MLX arrays once, then applies it inside Gemma 4's real forward
/// pass for prompt prefill and every generated token.
public struct ExactResidualIntervention: Codable, Sendable, Equatable {
  public let layer: Int
  public let plan: ControlledEvasionPlan
  /// `nil` preserves the original application policy: edit the final prompt
  /// token and every generated token.
  public let schedule: ExactResidualInterventionSchedule?
  /// Canonical zero-based layer where the edit basis was measured. `nil`
  /// denotes legacy same-site plans written before source provenance existed.
  public let sourceLayerZeroBased: Int?
  /// Semantic token role used to measure the edit basis. `nil` denotes the
  /// legacy coupled post-instruction/same-layer search.
  public let sourcePosition: ActivationTokenPosition?

  public init?(
    layer: Int, plan: ControlledEvasionPlan,
    schedule: ExactResidualInterventionSchedule? = nil,
    sourceLayerZeroBased: Int? = nil,
    sourcePosition: ActivationTokenPosition? = nil
  ) {
    guard layer > 0,
      sourceLayerZeroBased.map({ $0 >= 0 }) ?? true
    else { return nil }
    self.layer = layer
    self.plan = plan
    self.schedule = schedule
    self.sourceLayerZeroBased = sourceLayerZeroBased
    self.sourcePosition = sourcePosition
  }

  public func makeTransform()
    -> @Sendable (_ zeroBasedLayer: Int, _ state: MLXArray) -> MLXArray
  {
    let selectedLayer = layer - 1
    let promptGate = StickyPromptGate()
    let applicationSchedule = schedule ?? .backwardCompatible

    return { zeroBasedLayer, state in
      guard zeroBasedLayer == selectedLayer,
        state.dim(-1) == plan.targetWidth
      else { return state }

      // MLXArray is intentionally not Sendable. Constants are therefore
      // materialized on the model's execution context instead of being
      // captured across concurrency domains.
      let probeColumn = MLXArray(plan.probe.weights)
        .reshaped(plan.probe.width, 1).asType(.float32)
      let basis = MLXArray(plan.behaviorBasis.flatMap { $0 })
        .reshaped(plan.rank, plan.targetWidth).asType(.float32)
      let reference = MLXArray(plan.reference).asType(.float32)
      let working = state.asType(.float32)
      let score =
        matmul(working, probeColumn).squeezed(axis: -1)
        + plan.probe.bias
      let excess = maximum(score - plan.margin, MLXArray(Float.zero))
      let tokenGate: MLXArray
      if plan.transitionWidth > 0 {
        tokenGate = clip(excess / plan.transitionWidth, min: 0, max: 1)
      } else {
        tokenGate = (excess .> 0).asType(.float32)
      }
      // Calibration is performed at the final post-instruction token.
      // During prefill, edit only that final token. Applying independently
      // computed gates to earlier prompt tokens violates the calibration
      // contract and can damage a benign prompt even when its final gate is
      // exactly closed. Latch the final gate and reuse it for one-token
      // decode calls; recomputing from the model's own refusal tokens can
      // otherwise close the intervention exactly when it is needed.
      let gate: MLXArray
      if state.dim(-2) > 1 {
        let latched = tokenGate[0, -1]
        eval(latched)
        promptGate.beginPrefill(latched)
        if applicationSchedule.includePostInstruction {
          let prefix = MLXArray.zeros([1, state.dim(-2) - 1])
            .asType(.float32)
          gate = concatenated(
            [
              prefix, latched.reshaped(1, 1).asType(.float32),
            ], axis: -1)
        } else {
          gate = MLXArray.zeros(tokenGate.shape).asType(.float32)
        }
      } else {
        let (latched, generationIndex) = promptGate.beginDecodeToken()
        if applicationSchedule.appliesToGeneration(
          index: generationIndex)
        {
          gate = latched ?? tokenGate
        } else {
          gate = MLXArray.zeros(tokenGate.shape).asType(.float32)
        }
      }
      let centered = working - reference
      let projection = matmul(matmul(centered, basis.T), basis)
      let edited =
        working
        - plan.strength * gate.expandedDimensions(axis: -1) * projection
      return edited.asType(state.dtype)
    }
  }

  public static func combinedTransform(
    _ interventions: [ExactResidualIntervention]
  ) -> @Sendable (_ zeroBasedLayer: Int, _ state: MLXArray) -> MLXArray {
    let transforms = interventions.map { $0.makeTransform() }
    return { layer, state in
      transforms.reduce(state) { current, transform in
        transform(layer, current)
      }
    }
  }
}

/// Selects the token positions where an exact residual intervention runs.
///
/// `generationStart` is the zero-based index of the first generated token to
/// edit. Set it to `nil` for a post-instruction-only intervention. When a start
/// is present, `generationEnd` is an optional inclusive upper bound; `nil`
/// means the edit continues for all later generated tokens.
public struct ExactResidualInterventionSchedule: Codable, Sendable, Equatable {
  public let includePostInstruction: Bool
  public let generationStart: Int?
  public let generationEnd: Int?

  public init?(
    includePostInstruction: Bool,
    generationStart: Int?,
    generationEnd: Int? = nil
  ) {
    guard generationStart.map({ $0 >= 0 }) ?? (generationEnd == nil),
      generationEnd.map({ end in
        generationStart.map { end >= $0 } ?? false
      }) ?? true
    else { return nil }
    self.includePostInstruction = includePostInstruction
    self.generationStart = generationStart
    self.generationEnd = generationEnd
  }

  public func appliesToGeneration(index: Int) -> Bool {
    guard index >= 0, let generationStart,
      index >= generationStart
    else { return false }
    return generationEnd.map { index <= $0 } ?? true
  }

  /// Behavior used by artifacts written before schedules were introduced.
  public static let backwardCompatible = ExactResidualInterventionSchedule(
    includePostInstruction: true,
    generationStart: 0,
    generationEnd: nil)!

  /// Stable diagnostic label for the built-in schedule presets.
  public var diagnosticName: String {
    switch (includePostInstruction, generationStart, generationEnd) {
    case (true, 0, nil): "legacy"
    case (true, nil, nil): "post"
    case (false, 0, 0): "token0"
    case (false, 0, 1): "0..1"
    case (false, 0, 7): "0..7"
    default:
      "post=\(includePostInstruction),generation="
        + generationRangeDescription
    }
  }

  private var generationRangeDescription: String {
    guard let generationStart else { return "none" }
    return generationEnd.map { "\(generationStart)..\($0)" }
      ?? "\(generationStart)..."
  }
}

/// Named application schedules accepted by the diagnostic CLI.
public enum ExactResidualInterventionSchedulePreset: String, CaseIterable,
  Sendable
{
  case legacy
  case post
  case token0
  case firstTwo = "0..1"
  case firstEight = "0..7"

  public var schedule: ExactResidualInterventionSchedule {
    switch self {
    case .legacy:
      .backwardCompatible
    case .post:
      ExactResidualInterventionSchedule(
        includePostInstruction: true,
        generationStart: nil)!
    case .token0:
      ExactResidualInterventionSchedule(
        includePostInstruction: false,
        generationStart: 0,
        generationEnd: 0)!
    case .firstTwo:
      ExactResidualInterventionSchedule(
        includePostInstruction: false,
        generationStart: 0,
        generationEnd: 1)!
    case .firstEight:
      ExactResidualInterventionSchedule(
        includePostInstruction: false,
        generationStart: 0,
        generationEnd: 7)!
    }
  }

  /// Absence preserves old request and JSON behavior (`schedule == nil`). An
  /// explicitly named `legacy` preset is materialized so the trial artifact
  /// records that the caller deliberately selected it.
  public static func diagnosticGrid(
    environment: [String: String],
    key: String = "ABSLAYER_INTERVENTION_SCHEDULES"
  ) throws -> [ExactResidualInterventionSchedule?] {
    guard let raw = environment[key] else { return [nil] }
    let pieces = raw.split(separator: ",", omittingEmptySubsequences: false)
    var seen = Set<String>()
    var result = [ExactResidualInterventionSchedule?]()
    for piece in pieces {
      let name = piece.trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
      guard !name.isEmpty,
        let preset = Self(rawValue: name)
      else { throw ExactResidualInterventionSchedulePresetError.invalid(raw) }
      guard seen.insert(preset.rawValue).inserted else {
        throw ExactResidualInterventionSchedulePresetError.duplicate(
          preset.rawValue)
      }
      result.append(preset.schedule)
    }
    guard !result.isEmpty else {
      throw ExactResidualInterventionSchedulePresetError.invalid(raw)
    }
    return result
  }
}

public enum ExactResidualInterventionSchedulePresetError: LocalizedError,
  Equatable
{
  case invalid(String)
  case duplicate(String)

  public var errorDescription: String? {
    switch self {
    case .invalid(let value):
      "ABSLAYER_INTERVENTION_SCHEDULES must be a comma-separated subset of "
        + "legacy,post,token0,0..1,0..7, not '\(value)'."
    case .duplicate(let value):
      "ABSLAYER_INTERVENTION_SCHEDULES contains duplicate preset '\(value)'."
    }
  }
}

private final class StickyPromptGate: @unchecked Sendable {
  private let lock = NSLock()
  private var value: MLXArray?
  private var generationIndex = 0

  func beginPrefill(_ value: MLXArray) {
    lock.lock()
    self.value = value
    generationIndex = 0
    lock.unlock()
  }

  func beginDecodeToken() -> (value: MLXArray?, generationIndex: Int) {
    lock.lock()
    defer { lock.unlock() }
    let currentIndex = generationIndex
    generationIndex += 1
    return (value, currentIndex)
  }
}

public struct ControlledEvasionLayerPlan: Codable, Sendable, Equatable {
  public let layer: Int
  public let controlGateQuantile: Float
  public let controlScoreAtMargin: Float
  public let harmfulScoreMedian: Float
  public let intervention: ExactResidualIntervention

  public init(
    layer: Int, controlGateQuantile: Float,
    controlScoreAtMargin: Float, harmfulScoreMedian: Float,
    intervention: ExactResidualIntervention
  ) {
    self.layer = layer
    self.controlGateQuantile = controlGateQuantile
    self.controlScoreAtMargin = controlScoreAtMargin
    self.harmfulScoreMedian = harmfulScoreMedian
    self.intervention = intervention
  }
}

/// Converts paired residuals into prompt-gated, benign-null intervention plans.
/// The gate margin is calibrated from the target model's own harmless examples,
/// instead of assuming the source model and target model share a refusal plane.
public enum ControlledEvasionPlanner {
  public static func makePlan(
    layer: LayerActivationSet,
    subspaceRank: Int = 4,
    benignNullRank: Int = 8,
    benignNullRidge: Float = 1e-3,
    controlGateQuantile: Float = 0.95,
    strength: Float = 1
  ) -> ControlledEvasionLayerPlan? {
    makePlan(
      directionLayer: layer,
      applicationLayer: layer,
      subspaceRank: subspaceRank,
      benignNullRank: benignNullRank,
      benignNullRidge: benignNullRidge,
      controlGateQuantile: controlGateQuantile,
      strength: strength,
      includeApplicationGateAxisInEdit: true)
  }

  /// Builds a plan whose refusal basis is learned at one residual site and
  /// whose gate/reference are calibrated at the site where the edit will be
  /// applied. Decoder residuals share a coordinate space, but a direction that
  /// is decodable at one layer is not necessarily causal there. Keeping these
  /// roles separate lets the exact-runtime search test that distinction rather
  /// than assuming `sourceLayer == applicationLayer`.
  ///
  /// `includeApplicationGateAxisInEdit` preserves the original same-site
  /// behavior. Cross-site trajectory experiments normally set it to `false`:
  /// a low overlap between the source direction and application gate is then a
  /// diagnostic to test causally, not a reason to silently replace the source
  /// direction with the gate normal.
  public static func makePlan(
    directionLayer: LayerActivationSet,
    applicationLayer: LayerActivationSet,
    subspaceRank: Int = 4,
    benignNullRank: Int = 8,
    benignNullRidge: Float = 1e-3,
    controlGateQuantile: Float = 0.95,
    strength: Float = 1,
    includeApplicationGateAxisInEdit: Bool = false
  ) -> ControlledEvasionLayerPlan? {
    guard
      !directionLayer.contrast.isEmpty,
      directionLayer.contrast.count == directionLayer.control.count,
      !applicationLayer.contrast.isEmpty,
      applicationLayer.contrast.count == applicationLayer.control.count,
      directionLayer.contrast.count == applicationLayer.contrast.count,
      directionLayer.contrast.first?.count
        == applicationLayer.contrast.first?.count,
      subspaceRank > 0,
      (0...1).contains(controlGateQuantile),
      let directionProbe = ControlledEvasionProbe.fit(
        positive: directionLayer.contrast,
        negative: directionLayer.control),
      let applicationProbe = ControlledEvasionProbe.fit(
        positive: applicationLayer.contrast,
        negative: applicationLayer.control)
    else { return nil }

    // The edit basis originates at the explicitly selected direction site.
    // A covariance-whitened SVD can add useful secondary features while the
    // first axis preserves the source site's diagonal-LDA refusal normal.
    var basis = [directionProbe.weights]
    if subspaceRank > 1 {
      basis += whitenedSubspace(
        contrast: directionLayer.contrast,
        control: directionLayer.control,
        rank: subspaceRank - 1)
    }
    basis = Array(
      AbliterationMath.orthonormalized(basis).prefix(subspaceRank))

    if benignNullRank > 0,
      let projector = SoftBenignNullProjector(
        benignKeys: applicationLayer.control,
        rank: min(benignNullRank, applicationLayer.control.count),
        ridge: benignNullRidge,
        center: true)
    {
      let protected = AbliterationMath.orthonormalized(
        basis.compactMap { vector in
          let projected = projector.apply(to: vector)
          let retained = sqrt(
            projected.reduce(Float.zero) {
              $0 + $1 * $1
            })
          // Do not normalize numerical residue back into a full edit
          // direction when benign protection removed essentially the
          // entire candidate axis.
          return retained >= 0.05 ? projected : nil
        })
      if !protected.isEmpty {
        basis = protected
      }
    }

    // For the original same-site mode, retain overlap between the edit and the
    // gate normal. Cross-site mode deliberately leaves low overlap intact so
    // exact causal trials can determine whether the source direction transfers
    // to the application site instead of silently changing the hypothesis.
    let editableProbeEnergy = basis.reduce(Float.zero) { total, axis in
      let overlap = zip(axis, applicationProbe.weights).reduce(Float.zero) {
        $0 + $1.0 * $1.1
      }
      return total + overlap * overlap
    }
    if includeApplicationGateAxisInEdit && editableProbeEnergy < 0.10 {
      basis = Array(
        AbliterationMath.orthonormalized(
          [applicationProbe.weights] + basis
        ).prefix(subspaceRank))
    }

    let controlScores = applicationLayer.control.map(applicationProbe.score).sorted()
    let harmfulScores = applicationLayer.contrast.map(applicationProbe.score).sorted()
    guard let margin = quantile(controlScores, controlGateQuantile),
      let harmfulMedian = quantile(harmfulScores, 0.5)
    else { return nil }
    // A smooth gate avoids a discontinuity while retaining a target-model
    // calibrated benign false-positive ceiling.
    let transitionWidth = max(abs(harmfulMedian - margin), 1e-4)
    let reference = AbliterationMath.mean(applicationLayer.control)
    guard
      let plan = ControlledEvasionPlan(
        probe: applicationProbe, behaviorBasis: basis, reference: reference,
        margin: margin, transitionWidth: transitionWidth,
        strength: strength),
      let intervention = ExactResidualIntervention(
        layer: applicationLayer.layer, plan: plan)
    else { return nil }

    return ControlledEvasionLayerPlan(
      layer: applicationLayer.layer,
      controlGateQuantile: controlGateQuantile,
      controlScoreAtMargin: margin,
      harmfulScoreMedian: harmfulMedian,
      intervention: intervention)
  }

  static func quantile(_ sortedValues: [Float], _ probability: Float) -> Float? {
    guard !sortedValues.isEmpty, (0...1).contains(probability) else { return nil }
    guard sortedValues.count > 1 else { return sortedValues[0] }
    let position = Float(sortedValues.count - 1) * probability
    let lower = Int(floor(position))
    let upper = Int(ceil(position))
    let fraction = position - Float(lower)
    return sortedValues[lower] * (1 - fraction) + sortedValues[upper] * fraction
  }
}
