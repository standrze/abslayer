import Foundation
import MLX
import Testing

@testable import ProbeCore

@Suite("Answer-centered residual transport")
struct AnswerCenteredTransportTests {
    @Test("ACE targets the answered coordinate and preserves its complement")
    func affineCentroidMap() throws {
        let plan = try #require(AnswerCenteredTransportPlan(
            method: .affineCentroid,
            basis: [[1, 0, 0]],
            sourceMeanCoordinates: [4],
            targetMeanCoordinates: [-2],
            transportMatrix: [[0]],
            strength: 1))

        let edited = plan.apply(
            gateActivation: [8, 7, 9], to: [8, 7, 9])
        #expect(abs(edited[0] + 2) < 0.000_001)
        #expect(edited[1] == 7)
        #expect(edited[2] == 9)
    }

    @Test("gated plan closes on answered states and opens on refused states")
    func gateCalibration() throws {
        let probe = try #require(ControlledEvasionProbe(weights: [1, 0]))
        let gate = try #require(AnswerCenteredTransportGate(
            probe: probe, margin: 1, transitionWidth: 2))
        let plan = try #require(AnswerCenteredTransportPlan(
            method: .affineCentroid,
            basis: [[1, 0]],
            sourceMeanCoordinates: [3],
            targetMeanCoordinates: [0],
            transportMatrix: [[0]],
            gate: gate,
            strength: 1))

        #expect(plan.apply(
            gateActivation: [1, 0], to: [4, 5]) == [4, 5])
        let half = plan.apply(
            gateActivation: [2, 0], to: [4, 5])
        #expect(abs(half[0] - 2) < 0.000_001)
        #expect(half[1] == 5)
        let full = plan.apply(
            gateActivation: [3, 0], to: [4, 5])
        #expect(abs(full[0]) < 0.000_001)
        #expect(full[1] == 5)
    }

    @Test("PCA Gaussian OT aligns two-dimensional moments")
    func gaussianOTMomentAlignment() throws {
        let signs: [(Float, Float)] = [
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        let refused = signs.map { first, second in
            [3 + 2 * first, -2 + 0.5 * second, 10]
        }
        let answered = signs.map { first, second in
            // Rotated/scaled target covariance, not just a mean shift.
            [-1 + first + 0.5 * second, 4 + 1.5 * second, 10]
        }
        let fit = try #require(AnswerCenteredTransportFitter.fit(
            refusedActivations: refused,
            answeredActivations: answered,
            method: .pcaGaussianOT,
            rank: 2,
            covarianceRidge: 1e-7,
            controlGateQuantile: nil,
            strength: 1))

        #expect(fit.plan.rank == 2)
        #expect(fit.diagnostics.explainedVarianceFraction > 0.999)
        #expect(fit.diagnostics.projectedMeanAlignmentRMSE < 0.000_01)
        #expect(fit.diagnostics.projectedCovarianceAlignmentRMSE < 0.000_1)

        let mapped = refused.map {
            fit.plan.apply(gateActivation: $0, to: $0)
        }
        let mappedMean = mean(mapped)
        let targetMean = mean(answered)
        for index in mappedMean.indices {
            #expect(abs(mappedMean[index] - targetMean[index]) < 0.000_1)
        }
        // The rank-two map must not collapse or shift the orthogonal third
        // coordinate. This catches the destructive P A P^T lift.
        #expect(mapped.allSatisfy { abs($0[2] - 10) < 0.000_1 })

        let mappedCovariance = covariance2D(mapped)
        let targetCovariance = covariance2D(answered)
        for row in 0 ..< 2 {
            for column in 0 ..< 2 {
                #expect(abs(
                    mappedCovariance[row][column]
                        - targetCovariance[row][column]) < 0.000_5)
            }
        }
    }

    @Test("MLX runtime uses the same identity-preserving lift and schedule")
    func mlxRuntimeMap() throws {
        let plan = try #require(AnswerCenteredTransportPlan(
            method: .pcaGaussianOT,
            basis: [[1, 0]],
            sourceMeanCoordinates: [2],
            targetMeanCoordinates: [-1],
            transportMatrix: [[0.5]],
            strength: 1))
        let intervention = try #require(
            AnswerCenteredResidualIntervention(
                layer: 2, plan: plan,
                schedule: ExactResidualInterventionSchedule(
                    includePostInstruction: false,
                    generationStart: 0,
                    generationEnd: 0)))
        let transform = intervention.makeTransform()

        let prefill = transform(
            1, MLXArray([4, 9, 3, 8] as [Float], [1, 2, 2]))
        let token0 = transform(
            1, MLXArray([5, 7] as [Float], [1, 1, 2]))
        let token1 = transform(
            1, MLXArray([6, 6] as [Float], [1, 1, 2]))
        eval(prefill, token0, token1)

        #expect(prefill.asArray(Float.self) == [4, 9, 3, 8])
        // z' = 0.5 * (5 - 2) - 1 = 0.5; y is orthogonal.
        let first = token0.asArray(Float.self)
        #expect(abs(first[0] - 0.5) < 0.000_001)
        #expect(first[1] == 7)
        #expect(token1.asArray(Float.self) == [6, 6])
    }

    @Test("fit artifacts round trip with explicit method provenance")
    func serialization() throws {
        let fit = try #require(AnswerCenteredTransportFitter.fit(
            refusedActivations: [[2, 0], [3, 1], [4, -1]],
            answeredActivations: [[-2, 0], [-3, 1], [-4, -1]],
            method: .affineCentroid,
            rank: 1,
            controlGateQuantile: 0.95))
        let intervention = try #require(AnswerCenteredResidualIntervention(
            layer: 20, plan: fit.plan,
            schedule: ExactResidualInterventionSchedule(
                includePostInstruction: false,
                generationStart: 0,
                generationEnd: 7)))
        let artifact = AnswerCenteredTransportArtifact(
            modelPath: "/tmp/model",
            measurementPath: "/tmp/outcome-screened-dev.json",
            pairNames: ["dev-a", "dev-b", "dev-c"],
            tokenPosition: .postInstruction,
            layerZeroBased: 19,
            fit: fit,
            intervention: intervention)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        defer { try? FileManager.default.removeItem(at: url) }

        try artifact.write(to: url.path)
        let decoded = try AnswerCenteredTransportArtifact.read(
            from: url.path)
        #expect(decoded == artifact)
        #expect(decoded.fit.plan.method == .affineCentroid)
        #expect(decoded.status == "development_candidate")
        #expect(decoded.intervention.layer == 20)
        let refusedGate = try #require(
            decoded.fit.diagnostics.meanRefusedGate)
        let answeredGate = try #require(
            decoded.fit.diagnostics.meanAnsweredGate)
        #expect(refusedGate > answeredGate)
        let derived = try #require(decoded.derived(
            from: url.path,
            strength: 0.5,
            schedule: ExactResidualInterventionSchedule(
                includePostInstruction: true,
                generationStart: nil)))
        #expect(derived.parentArtifactPath == url.standardizedFileURL.path)
        #expect(derived.fit.plan.strength == 0.5)
        #expect(derived.fit.diagnostics == decoded.fit.diagnostics)
        #expect(derived.intervention.plan == derived.fit.plan)
        #expect(derived.intervention.schedule?.diagnosticName == "post")
    }

    @Test("derive can disable the gate without changing fitted geometry")
    func deriveWithoutGate() throws {
        let fit = try #require(AnswerCenteredTransportFitter.fit(
            refusedActivations: [[3, 0], [4, 1], [5, -1]],
            answeredActivations: [[-3, 0], [-4, 1], [-5, -1]],
            method: .pcaGaussianOT,
            rank: 2,
            controlGateQuantile: 0.99))
        let intervention = try #require(AnswerCenteredResidualIntervention(
            layer: 24, plan: fit.plan,
            schedule: ExactResidualInterventionSchedule(
                includePostInstruction: false,
                generationStart: 0,
                generationEnd: 7)))
        let source = AnswerCenteredTransportArtifact(
            modelPath: "/tmp/model",
            measurementPath: "/tmp/measurement-dev.json",
            pairNames: ["dev-a", "dev-b", "dev-c"],
            tokenPosition: .postInstruction,
            layerZeroBased: 23,
            fit: fit,
            intervention: intervention)
        let parentPath = "/tmp/pca-ot-k2-l23-gated.json"
        let derived = try #require(source.derived(
            from: parentPath,
            strength: 0.25,
            schedule: intervention.schedule,
            gateOverride: .disable))

        #expect(source.fit.plan.gate != nil)
        #expect(derived.fit.plan.gate == nil)
        #expect(derived.derivationGateOverride == .disable)
        #expect(derived.parentArtifactPath
            == URL(fileURLWithPath: parentPath).standardizedFileURL.path)
        #expect(derived.modelPath == source.modelPath)
        #expect(derived.measurementPath == source.measurementPath)
        #expect(derived.pairNames == source.pairNames)
        #expect(derived.tokenPosition == source.tokenPosition)
        #expect(derived.layerZeroBased == source.layerZeroBased)
        #expect(derived.fit.plan.method == source.fit.plan.method)
        #expect(derived.fit.plan.basis == source.fit.plan.basis)
        #expect(derived.fit.plan.sourceMeanCoordinates
            == source.fit.plan.sourceMeanCoordinates)
        #expect(derived.fit.plan.targetMeanCoordinates
            == source.fit.plan.targetMeanCoordinates)
        #expect(derived.fit.plan.transportMatrix
            == source.fit.plan.transportMatrix)
        #expect(derived.fit.plan.strength == 0.25)
        #expect(derived.intervention.schedule == intervention.schedule)
        #expect(derived.fit.diagnostics.meanRefusedGate == 1)
        #expect(derived.fit.diagnostics.meanAnsweredGate == 1)
        #expect(derived.fit.diagnostics.explainedVarianceFraction
            == source.fit.diagnostics.explainedVarianceFraction)
        #expect(derived.fit.diagnostics.projectedCovarianceAlignmentRMSE
            == source.fit.diagnostics.projectedCovarianceAlignmentRMSE)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        defer { try? FileManager.default.removeItem(at: url) }
        try derived.write(to: url.path)
        #expect(try AnswerCenteredTransportArtifact.read(from: url.path)
            == derived)
    }

    @Test("legacy artifacts decode without optional gate telemetry")
    func legacyArtifactWithoutGateTelemetry() throws {
        let fit = try #require(AnswerCenteredTransportFitter.fit(
            refusedActivations: [[2, 0], [3, 1], [4, -1]],
            answeredActivations: [[-2, 0], [-3, 1], [-4, -1]],
            method: .pcaGaussianOT,
            rank: 2,
            controlGateQuantile: 0.99))
        let intervention = try #require(AnswerCenteredResidualIntervention(
            layer: 24, plan: fit.plan,
            schedule: ExactResidualInterventionSchedule(
                includePostInstruction: false,
                generationStart: 0,
                generationEnd: 7)))
        let artifact = AnswerCenteredTransportArtifact(
            modelPath: "/tmp/model",
            measurementPath: "/tmp/measurement-dev.json",
            pairNames: ["dev-a", "dev-b", "dev-c"],
            tokenPosition: .postInstruction,
            layerZeroBased: 23,
            fit: fit,
            intervention: intervention)
        let encoded = try JSONEncoder().encode(artifact)
        var root = try #require(
            JSONSerialization.jsonObject(with: encoded)
                as? [String: Any])
        var fitObject = try #require(root["fit"] as? [String: Any])
        var diagnostics = try #require(
            fitObject["diagnostics"] as? [String: Any])
        diagnostics.removeValue(forKey: "meanRefusedGate")
        diagnostics.removeValue(forKey: "meanAnsweredGate")
        fitObject["diagnostics"] = diagnostics
        root["fit"] = fitObject
        let legacyData = try JSONSerialization.data(withJSONObject: root)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        defer { try? FileManager.default.removeItem(at: url) }
        try legacyData.write(to: url, options: .atomic)

        let decoded = try AnswerCenteredTransportArtifact.read(
            from: url.path)
        #expect(decoded.fit.diagnostics.meanRefusedGate == nil)
        #expect(decoded.fit.diagnostics.meanAnsweredGate == nil)
        let derived = try #require(decoded.derived(
            from: url.path,
            strength: 0.75,
            schedule: decoded.intervention.schedule))
        #expect(derived.fit.diagnostics.meanRefusedGate == nil)
        #expect(derived.fit.diagnostics.meanAnsweredGate == nil)
    }

    private func mean(_ values: [[Float]]) -> [Float] {
        var result = Array(repeating: Float.zero, count: values[0].count)
        for value in values {
            for index in result.indices { result[index] += value[index] }
        }
        return result.map { $0 / Float(values.count) }
    }

    private func covariance2D(_ values: [[Float]]) -> [[Float]] {
        let center = mean(values)
        var result = Array(
            repeating: Array(repeating: Float.zero, count: 2), count: 2)
        for value in values {
            let x = value[0] - center[0]
            let y = value[1] - center[1]
            result[0][0] += x * x
            result[0][1] += x * y
            result[1][0] += y * x
            result[1][1] += y * y
        }
        return result.map { row in
            row.map { $0 / Float(values.count - 1) }
        }
    }
}
