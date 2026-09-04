import Foundation
import MLX
import Testing

@testable import ProbeCore

@Suite("Exact residual intervention application schedules")
struct ExactResidualInterventionScheduleTests {
  @Test("CLI schedule grid defaults to the exact legacy representation")
  func defaultGrid() throws {
    let schedules = try ExactResidualInterventionSchedulePreset.diagnosticGrid(
      environment: [:])
    #expect(schedules.count == 1)
    #expect(schedules[0] == nil)
  }

  @Test("CLI schedule names map in order and materialize explicit legacy")
  func namedGrid() throws {
    let schedules = try ExactResidualInterventionSchedulePreset.diagnosticGrid(
      environment: [
        "ABSLAYER_INTERVENTION_SCHEDULES":
          " legacy, post, token0, 0..1, 0..7 "
      ])
    #expect(schedules.compactMap { $0?.diagnosticName }
      == ["legacy", "post", "token0", "0..1", "0..7"])
    #expect(schedules[0] == .backwardCompatible)
  }

  @Test("CLI schedule grid rejects unknown and duplicate names")
  func invalidGrid() {
    #expect(throws: ExactResidualInterventionSchedulePresetError.invalid(
      "post,forever")) {
        try ExactResidualInterventionSchedulePreset.diagnosticGrid(
          environment: [
            "ABSLAYER_INTERVENTION_SCHEDULES": "post,forever"
          ])
    }
    #expect(throws: ExactResidualInterventionSchedulePresetError.duplicate(
      "token0")) {
        try ExactResidualInterventionSchedulePreset.diagnosticGrid(
          environment: [
            "ABSLAYER_INTERVENTION_SCHEDULES": "token0,TOKEN0"
          ])
    }
  }

  @Test("an explicit schedule is retained in the intervention artifact")
  func explicitScheduleArtifact() throws {
    let original = try makeIntervention(
      schedule: ExactResidualInterventionSchedulePreset.firstEight.schedule)
    let data = try JSONEncoder().encode(original)
    let object = try #require(
      JSONSerialization.jsonObject(with: data) as? [String: Any])
    let encodedSchedule = try #require(object["schedule"] as? [String: Any])
    #expect(encodedSchedule["includePostInstruction"] as? Bool == false)
    #expect(encodedSchedule["generationStart"] as? Int == 0)
    #expect(encodedSchedule["generationEnd"] as? Int == 7)
    #expect(try JSONDecoder().decode(
      ExactResidualIntervention.self, from: data) == original)
  }

  @Test("post-instruction-only schedule never edits generated tokens")
  func postInstructionOnly() throws {
    let schedule = try #require(
      ExactResidualInterventionSchedule(
        includePostInstruction: true,
        generationStart: nil))
    #expect(!schedule.appliesToGeneration(index: 0))

    let transform = try makeTransform(schedule: schedule)
    let prefill = transform(1, tensor([2, 5, 3, 7], tokens: 2))
    let token0 = transform(1, tensor([-1, 9], tokens: 1))
    eval(prefill, token0)

    #expect(prefill.asArray(Float.self) == [2, 5, 0, 7])
    #expect(token0.asArray(Float.self) == [-1, 9])
  }

  @Test("generation token zero can be selected without the prompt token")
  func generationTokenZeroOnly() throws {
    let schedule = try #require(
      ExactResidualInterventionSchedule(
        includePostInstruction: false,
        generationStart: 0,
        generationEnd: 0))
    #expect(schedule.appliesToGeneration(index: 0))
    #expect(!schedule.appliesToGeneration(index: 1))

    let transform = try makeTransform(schedule: schedule)
    let prefill = transform(1, tensor([2, 5, 3, 7], tokens: 2))
    let token0 = transform(1, tensor([-1, 9], tokens: 1))
    let token1 = transform(1, tensor([-2, 8], tokens: 1))
    eval(prefill, token0, token1)

    #expect(prefill.asArray(Float.self) == [2, 5, 3, 7])
    #expect(token0.asArray(Float.self) == [0, 9])
    #expect(token1.asArray(Float.self) == [-2, 8])
  }

  @Test("inclusive zero-through-one window resets for a new prefill")
  func firstTwoGenerationTokens() throws {
    let schedule = try #require(
      ExactResidualInterventionSchedule(
        includePostInstruction: false,
        generationStart: 0,
        generationEnd: 1))
    #expect(schedule.appliesToGeneration(index: 0))
    #expect(schedule.appliesToGeneration(index: 1))
    #expect(!schedule.appliesToGeneration(index: 2))

    let transform = try makeTransform(schedule: schedule)
    _ = transform(1, tensor([2, 5, 3, 7], tokens: 2))
    let token0 = transform(1, tensor([-1, 9], tokens: 1))
    let token1 = transform(1, tensor([-2, 8], tokens: 1))
    let token2 = transform(1, tensor([-3, 7], tokens: 1))

    // A new prefill must restart generation indexing at zero.
    let secondPrefill = transform(1, tensor([1, 4, 2, 6], tokens: 2))
    let secondToken0 = transform(1, tensor([-4, 5], tokens: 1))
    eval(token0, token1, token2, secondPrefill, secondToken0)

    #expect(token0.asArray(Float.self) == [0, 9])
    #expect(token1.asArray(Float.self) == [0, 8])
    #expect(token2.asArray(Float.self) == [-3, 7])
    #expect(secondPrefill.asArray(Float.self) == [1, 4, 2, 6])
    #expect(secondToken0.asArray(Float.self) == [0, 5])
  }

  @Test("JSON without a schedule retains prompt-plus-all-decode behavior")
  func backwardCompatibleDecodeBehavior() throws {
    let original = try makeIntervention(schedule: nil)
    let encoded = try JSONEncoder().encode(original)
    var object = try #require(
      JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    #expect(object["sourceLayerZeroBased"] == nil)
    #expect(object["sourcePosition"] == nil)
    object.removeValue(forKey: "schedule")
    let legacyJSON = try JSONSerialization.data(withJSONObject: object)
    let decoded = try JSONDecoder().decode(
      ExactResidualIntervention.self, from: legacyJSON)
    #expect(decoded.schedule == nil)

    let transform = decoded.makeTransform()
    let prefill = transform(1, tensor([2, 5, 3, 7], tokens: 2))
    let token0 = transform(1, tensor([-1, 9], tokens: 1))
    let token1 = transform(1, tensor([-2, 8], tokens: 1))
    eval(prefill, token0, token1)

    #expect(prefill.asArray(Float.self) == [2, 5, 0, 7])
    #expect(token0.asArray(Float.self) == [0, 9])
    #expect(token1.asArray(Float.self) == [0, 8])
  }

  @Test("source provenance round-trips while legacy JSON remains decodable")
  func sourceProvenanceCompatibility() throws {
    let original = try makeIntervention(
      schedule: nil,
      sourceLayerZeroBased: 4,
      sourcePosition: .firstResponse)
    let encoded = try JSONEncoder().encode(original)
    let decoded = try JSONDecoder().decode(
      ExactResidualIntervention.self, from: encoded)
    #expect(decoded.sourceLayerZeroBased == 4)
    #expect(decoded.sourcePosition == .firstResponse)

    var object = try #require(
      JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    object.removeValue(forKey: "sourceLayerZeroBased")
    object.removeValue(forKey: "sourcePosition")
    let legacyJSON = try JSONSerialization.data(withJSONObject: object)
    let legacy = try JSONDecoder().decode(
      ExactResidualIntervention.self, from: legacyJSON)
    #expect(legacy.sourceLayerZeroBased == nil)
    #expect(legacy.sourcePosition == nil)
  }

  private func makeTransform(
    schedule: ExactResidualInterventionSchedule?
  ) throws -> @Sendable (Int, MLXArray) -> MLXArray {
    try makeIntervention(schedule: schedule).makeTransform()
  }

  private func makeIntervention(
    schedule: ExactResidualInterventionSchedule?,
    sourceLayerZeroBased: Int? = nil,
    sourcePosition: ActivationTokenPosition? = nil
  ) throws -> ExactResidualIntervention {
    let probe = try #require(ControlledEvasionProbe(weights: [1, 0]))
    let plan = try #require(
      ControlledEvasionPlan(
        probe: probe,
        behaviorBasis: [[1, 0]],
        reference: [0, 0],
        margin: 0,
        transitionWidth: 0,
        strength: 1))
    return try #require(
      ExactResidualIntervention(
        layer: 2, plan: plan, schedule: schedule,
        sourceLayerZeroBased: sourceLayerZeroBased,
        sourcePosition: sourcePosition))
  }

  private func tensor(_ values: [Float], tokens: Int) -> MLXArray {
    MLXArray(values, [1, tokens, 2]).asType(.float32)
  }
}
