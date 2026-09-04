import Foundation
import Testing
@testable import ProbeCore

@Test func controlledEvasionProbeNormalizesItsSeparatingPlane() throws {
    let probe = try #require(ControlledEvasionProbe(weights: [3, 4], bias: -5))
    let norm = sqrt(probe.weights.reduce(Float.zero) { $0 + $1 * $1 })
    #expect(abs(norm - 1) < 0.0001)
    #expect(abs(probe.score([3, 4]) - 4) < 0.0001)
    #expect(abs(probe.score([0, 0]) + 1) < 0.0001)
}

@Test func diagonalLDAProbeSeparatesClassesWithNoisyNuisanceFeature() throws {
    let probe = try #require(ControlledEvasionProbe.fit(
        positive: [[2, -50], [3, 50]],
        negative: [[-2, -50], [-3, 50]]))
    #expect(probe.score([2.5, 0]) > 0)
    #expect(probe.score([-2.5, 0]) < 0)
    #expect(abs(probe.weights[0]) > abs(probe.weights[1]))
}

@Test func controlledEvasionUsesPromptMarginAndAffineProjection() throws {
    let probe = try #require(ControlledEvasionProbe(weights: [1]))
    let plan = try #require(ControlledEvasionPlan(
        probe: probe, behaviorBasis: [[1, 0]], reference: [2, 0],
        margin: 1, transitionWidth: 2, strength: 1))

    #expect(plan.apply(promptActivation: [1], to: [5, 7]) == [5, 7])
    let partiallyEdited = plan.apply(promptActivation: [2], to: [5, 7])
    #expect(abs(partiallyEdited[0] - 3.5) < 0.0001)
    #expect(partiallyEdited[1] == 7)
    let fullyEdited = plan.apply(promptActivation: [3], to: [5, 7])
    #expect(abs(fullyEdited[0] - 2) < 0.0001)
    #expect(fullyEdited[1] == 7)
}

@Test func softBenignNullProjectorSuppressesOnlyBenignRowSpace() throws {
    let projector = try #require(SoftBenignNullProjector(
        benignKeys: [[10, 0], [-10, 0]], rank: 1, ridge: 1))
    let benign = projector.apply(to: [1, 0])
    let nullDirection = projector.apply(to: [0, 1])
    #expect(abs(benign[0] - 1 / 101) < 0.0001)
    #expect(abs(benign[1]) < 0.0001)
    #expect(nullDirection == [0, 1])
    #expect(projector.rank == 1)
}

@Test func lowRankInterventionRespondsToTargetButNotBenignKey() throws {
    let projector = try #require(SoftBenignNullProjector(
        benignKeys: [[10, 0], [-10, 0]], rank: 1, ridge: 1))
    let intervention = try #require(BenignNullLowRankIntervention(
        outputDirections: [[0, 1]], conditionDirections: [[1, 1]],
        scales: [-1], benignProjector: projector))

    let benignDelta = intervention.delta(for: [10, 0])
    let targetDelta = intervention.delta(for: [0, 1])
    #expect(abs(benignDelta[1]) < 0.08)
    #expect(abs(targetDelta[1] + 1 / sqrt(2)) < 0.0001)
    let matrix = intervention.deltaMatrix()
    #expect(matrix.count == 4)
    #expect(intervention.applying(to: [0, 0, 0, 0], rows: 2, columns: 2) == matrix)
}

@Test func candidateScoreCannotBuyItsWayPastKLOrControlGates() throws {
    let guardrails = InterventionGuardrails(
        maximumFirstTokenKL: 0.5, maximumControlFailureRate: 0.10,
        maximumAddedControlFailureRate: 0.02)
    let passing = InterventionCandidateScore.evaluate(
        baselineRefusalRate: 0.8, candidateRefusalRate: 0.2,
        baselineControlFailureRate: 0.02, candidateControlFailureRate: 0.03,
        firstTokenKL: 0.1, guardrails: guardrails)
    #expect(passing.passesGuardrails)
    let passingValue = try #require(passing.value)
    #expect(passingValue > 0)

    let highKL = InterventionCandidateScore.evaluate(
        baselineRefusalRate: 0.8, candidateRefusalRate: 0,
        baselineControlFailureRate: 0.02, candidateControlFailureRate: 0.02,
        firstTokenKL: 0.6, guardrails: guardrails)
    #expect(!highKL.passesGuardrails)
    #expect(highKL.value == nil)

    let damagedControl = InterventionCandidateScore.evaluate(
        baselineRefusalRate: 0.8, candidateRefusalRate: 0,
        baselineControlFailureRate: 0.02, candidateControlFailureRate: 0.05,
        firstTokenKL: 0.1, guardrails: guardrails)
    #expect(!damagedControl.passesGuardrails)
    #expect(damagedControl.value == nil)
}
