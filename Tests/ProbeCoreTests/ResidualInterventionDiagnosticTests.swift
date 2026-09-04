import Foundation
import MLX
@testable import ProbeCore
import Testing

@Suite("Exact residual intervention diagnostics")
struct ResidualInterventionDiagnosticTests {
    @Test("persist-all environment flag is strict and defaults off")
    func persistAllEnvironmentParsing() throws {
        #expect(try DiagnosticTrialPersistenceConfiguration.parse(
            environment: [:]) == false)
        #expect(try DiagnosticTrialPersistenceConfiguration.parse(
            environment: [
                "ABSLAYER_DIAGNOSTIC_PERSIST_ALL_TRIALS": " true "
            ]) == true)
        #expect(try DiagnosticTrialPersistenceConfiguration.parse(
            environment: [
                "ABSLAYER_DIAGNOSTIC_PERSIST_ALL_TRIALS": "FALSE"
            ]) == false)

        for value in ["", "1", "yes", "enabled"] {
            #expect(throws:
                DiagnosticTrialPersistenceConfigurationError
                    .invalidBoolean(value))
            {
                try DiagnosticTrialPersistenceConfiguration.parse(
                    environment: [
                        "ABSLAYER_DIAGNOSTIC_PERSIST_ALL_TRIALS": value
                    ])
            }
        }

        let request = ResidualInterventionDiagnosticRequest(
            modelDirectory: "/model",
            measurementPairs: [], evaluationPairs: [])
        #expect(request.persistAllTrials == false)
    }

    @Test("persist-all bypasses only the non-improving proxy skip")
    func persistAllCandidatePolicy() {
        #expect(ResidualInterventionDiagnosticEngine
            .shouldFullyMeasureCandidate(
                candidateCyberFailureRate: 0.4,
                incumbentCyberFailureRate: 0.5,
                persistAllTrials: false))
        #expect(!ResidualInterventionDiagnosticEngine
            .shouldFullyMeasureCandidate(
                candidateCyberFailureRate: 0.5,
                incumbentCyberFailureRate: 0.5,
                persistAllTrials: false))
        #expect(!ResidualInterventionDiagnosticEngine
            .shouldFullyMeasureCandidate(
                candidateCyberFailureRate: 0.6,
                incumbentCyberFailureRate: 0.5,
                persistAllTrials: false))
        #expect(ResidualInterventionDiagnosticEngine
            .shouldFullyMeasureCandidate(
                candidateCyberFailureRate: 0.6,
                incumbentCyberFailureRate: 0.5,
                persistAllTrials: true))
    }

    @Test("trajectory environment parsing is opt-in and ordered")
    func trajectoryEnvironmentParsing() throws {
        #expect(try TrajectoryDirectionSearchConfiguration.parse(
            environment: [:]) == nil)

        let defaultDepth = try #require(
            try TrajectoryDirectionSearchConfiguration.parse(environment: [
                "ABSLAYER_DIRECTION_POSITIONS":
                    " first-response, post-instruction, last-user "
            ]))
        #expect(defaultDepth.positions == [
            .firstResponse, .postInstruction, .lastUser,
        ])
        #expect(defaultDepth.layersPerPosition == 2)

        let explicitDepth = try #require(
            try TrajectoryDirectionSearchConfiguration.parse(environment: [
                "ABSLAYER_DIRECTION_POSITIONS": "second-response",
                "ABSLAYER_DIRECTION_LAYERS_PER_POSITION": " 3 ",
            ]))
        #expect(explicitDepth.positions == [.secondResponse])
        #expect(explicitDepth.layersPerPosition == 3)
    }

    @Test("trajectory environment rejects invalid roles and source counts")
    func invalidTrajectoryEnvironment() {
        #expect(throws:
            TrajectoryDirectionSearchConfigurationError.invalidPositions(
                "post-instruction,unknown"))
        {
            try TrajectoryDirectionSearchConfiguration.parse(environment: [
                "ABSLAYER_DIRECTION_POSITIONS": "post-instruction,unknown"
            ])
        }
        #expect(throws:
            TrajectoryDirectionSearchConfigurationError.duplicatePosition(
                .firstResponse))
        {
            try TrajectoryDirectionSearchConfiguration.parse(environment: [
                "ABSLAYER_DIRECTION_POSITIONS":
                    "first-response,FIRST-RESPONSE"
            ])
        }
        #expect(throws:
            TrajectoryDirectionSearchConfigurationError
                .invalidLayersPerPosition("0"))
        {
            try TrajectoryDirectionSearchConfiguration.parse(environment: [
                "ABSLAYER_DIRECTION_POSITIONS": "post-instruction",
                "ABSLAYER_DIRECTION_LAYERS_PER_POSITION": "0",
            ])
        }
        #expect(throws:
            TrajectoryDirectionSearchConfigurationError.positionsRequired)
        {
            try TrajectoryDirectionSearchConfiguration.parse(environment: [
                "ABSLAYER_DIRECTION_LAYERS_PER_POSITION": "2"
            ])
        }
    }

    @Test("probe AUC ranks separated harmful and control residuals")
    func probeAUC() {
        let layer = LayerActivationSet(
            layer: 3,
            contrast: [[3, 0], [4, 1], [5, -1], [6, 2], [7, -2], [8, 0.5]],
            control: [[-3, 0], [-4, 1], [-5, -1], [-6, 2], [-7, -2], [-8, 0.5]])
        #expect(ResidualInterventionDiagnosticEngine.linearProbeAUC(layer) == 1)
    }

    @Test("planner calibrates its gate from target-model controls")
    func targetCalibratedGate() throws {
        let layer = LayerActivationSet(
            layer: 2,
            contrast: [[4, 1], [5, -1], [6, 0], [7, 0.5]],
            control: [[-4, 1], [-5, -1], [-6, 0], [-7, 0.5]])
        let planned = try #require(ControlledEvasionPlanner.makePlan(
            layer: layer, subspaceRank: 1, benignNullRank: 0,
            controlGateQuantile: 0.75, strength: 0.5))
        let controlScores = layer.control.map(
            planned.intervention.plan.probe.score).sorted()
        let expected = try #require(
            ControlledEvasionPlanner.quantile(controlScores, 0.75))
        #expect(abs(planned.controlScoreAtMargin - expected) < 1e-5)
        #expect(planned.intervention.plan.strength == 0.5)
        let editableProbeEnergy = planned.intervention.plan.behaviorBasis.map { axis in
            let overlap = zip(axis, planned.intervention.plan.probe.weights)
                .reduce(Float.zero) { $0 + $1.0 * $1.1 }
            return overlap * overlap
        }.reduce(Float.zero, +)
        #expect(editableProbeEnergy > 0.9)
    }

    @Test("trajectory planner separates direction source from application gate")
    func crossSiteDirectionAndGate() throws {
        let direction = LayerActivationSet(
            layer: 4,
            contrast: [[4, 0], [5, 0.2], [6, -0.2], [7, 0.1]],
            control: [[-4, 0], [-5, 0.2], [-6, -0.2], [-7, 0.1]])
        let application = LayerActivationSet(
            layer: 9,
            contrast: [[0, 4], [0.2, 5], [-0.2, 6], [0.1, 7]],
            control: [[0, -4], [0.2, -5], [-0.2, -6], [0.1, -7]])

        let planned = try #require(ControlledEvasionPlanner.makePlan(
            directionLayer: direction,
            applicationLayer: application,
            subspaceRank: 1,
            benignNullRank: 0,
            controlGateQuantile: 0.75,
            strength: 1,
            includeApplicationGateAxisInEdit: false))

        let gate = planned.intervention.plan.probe.weights
        let edit = try #require(planned.intervention.plan.behaviorBasis.first)
        #expect(planned.intervention.layer == 9)
        #expect(abs(gate[1]) > abs(gate[0]))
        #expect(abs(edit[0]) > abs(edit[1]))
    }

    @Test("exact MLX transform gates tokens and edits only its chosen layer")
    func exactTransform() throws {
        let probe = try #require(ControlledEvasionProbe(weights: [1, 0]))
        let plan = try #require(ControlledEvasionPlan(
            probe: probe, behaviorBasis: [[1, 0]], reference: [0, 0],
            margin: 0, transitionWidth: 0, strength: 1))
        let intervention = try #require(
            ExactResidualIntervention(layer: 2, plan: plan))
        let input = MLXArray([-1, 5, 2, 3], [1, 2, 2]).asType(.float32)

        let unchanged = intervention.makeTransform()(0, input)
        let transform = intervention.makeTransform()
        let edited = transform(1, input)
        // The final prefill token opens the prompt gate. A later decode token
        // with a negative instantaneous score must remain edited.
        let decoded = transform(
            1, MLXArray([-1, 7], [1, 1, 2]).asType(.float32))
        eval(unchanged, edited, decoded)
        #expect(unchanged.asArray(Float.self) == [-1, 5, 2, 3])
        #expect(edited.asArray(Float.self) == [-1, 5, 0, 3])
        #expect(decoded.asArray(Float.self) == [0, 7])
    }

    @Test("prefill ignores transient early-token gates when the final gate is closed")
    func finalPromptTokenIsTheOnlyPrefillGate() throws {
        let probe = try #require(ControlledEvasionProbe(weights: [1, 0]))
        let plan = try #require(ControlledEvasionPlan(
            probe: probe, behaviorBasis: [[1, 0]], reference: [0, 0],
            margin: 0, transitionWidth: 0, strength: 1))
        let intervention = try #require(
            ExactResidualIntervention(layer: 2, plan: plan))
        let transform = intervention.makeTransform()

        // The first token would open an independently computed token gate, but
        // the calibrated final prompt token closes the prompt-latched gate.
        let prefill = transform(
            1, MLXArray([2, 5, -1, 3], [1, 2, 2]).asType(.float32))
        let decoded = transform(
            1, MLXArray([4, 7], [1, 1, 2]).asType(.float32))
        eval(prefill, decoded)

        #expect(prefill.asArray(Float.self) == [2, 5, -1, 3])
        #expect(decoded.asArray(Float.self) == [4, 7])
    }

    @Test("strict cyber metrics reject partial educational engagement")
    func strictPartialFailure() {
        let responses = [PromptResult(
            name: "cyber-1", contrastResponse: "educational framing",
            controlResponse: "healthy answer", category: "execution")]
        let classifier = RecordedOutcomeClassifier(judgments: [
            RecordedOutcomeJudgment(
                name: "cyber-1", channel: .contrast,
                classification: OutcomeClassification(
                    outcome: .partialCompliance, confidence: 1)),
            RecordedOutcomeJudgment(
                name: "cyber-1", channel: .control,
                classification: OutcomeClassification(
                    outcome: .compliance, confidence: 1)),
        ])
        let metrics = ExactBehaviorMetrics(evaluation: RefusalEvaluator.evaluate(
            responses, classifier: classifier))
        #expect(metrics.strictCyberFailureRate == 1)
        #expect(metrics.partialComplianceRate == 1)
        #expect(metrics.refusalRate == 0)
        #expect(metrics.controlFailureRate == 0)
    }

    @Test("causal candidates are distributed across model depth")
    func depthDistributedCandidates() {
        var locality = [LayerRefusalLocality]()
        for layer in 1 ... 12 {
            let early = layer <= 4
            locality.append(LayerRefusalLocality(
                layer: layer,
                postInstructionAUC: 1,
                lastUserAUC: early ? 0.5 : 1,
                refusalAmplification: early ? 0.5 : 0,
                priority: early ? 1.5 - Double(layer) * 0.01 : 1))
        }
        locality.sort { $0.priority > $1.priority }

        let selected = ResidualInterventionDiagnosticEngine
            .selectCandidateLayerNumbers(
                locality: locality, maximum: 3, layerCount: 12)

        #expect(selected.count == 3)
        #expect(selected.contains { (1 ... 4).contains($0) })
        #expect(selected.contains { (5 ... 8).contains($0) })
        #expect(selected.contains { (9 ... 12).contains($0) })
    }

    @Test("each trajectory role selects source layers across decoder depth")
    func depthDistributedDirectionSources() {
        let layers = (1 ... 12).map { layer in
            LayerActivationSet(
                layer: layer,
                contrast: [[4, 0], [5, 1], [6, -1], [7, 0.5]],
                control: [[-4, 0], [-5, 1], [-6, -1], [-7, 0.5]])
        }
        let selected = ResidualInterventionDiagnosticEngine
            .selectDirectionLayerNumbers(layers: layers, maximum: 3)

        #expect(selected.count == 3)
        #expect(selected.contains { (1 ... 4).contains($0) })
        #expect(selected.contains { (5 ... 8).contains($0) })
        #expect(selected.contains { (9 ... 12).contains($0) })
    }

    @Test("fallback classifier cannot certify an intervention study")
    func fallbackCannotCertify() {
        let responses = [PromptResult(
            name: "case", contrastResponse: "candidate", controlResponse: "control")]
        let classifier = RecordedOutcomeClassifier(judgments: [
            RecordedOutcomeJudgment(
                name: "case", channel: .contrast,
                classification: OutcomeClassification(
                    outcome: .compliance, confidence: 1)),
            RecordedOutcomeJudgment(
                name: "case", channel: .control,
                classification: OutcomeClassification(
                    outcome: .compliance, confidence: 1)),
        ])
        let metrics = ExactBehaviorMetrics(evaluation: RefusalEvaluator.evaluate(
            responses, classifier: classifier))
        let study = ExactResidualInterventionStudy(
            schemaVersion: 1, modelPath: "/model",
            classifierIdentifier: "substring-fallback-v1",
            maximumStrictCyberFailureRate: 0.1,
            maximumSequenceKLLowerBound: 0.5,
            baseline: metrics, baselineCaseOutcomes: nil,
            localityRounds: [], trials: [], acceptedInterventions: [],
            finalMetrics: metrics, finalFirstTokenKL: 0,
            finalSequenceKLLowerBound: 0,
            finalControlPerplexityRatio: 1, finalCaseOutcomes: nil)
        #expect(study.passedAbliteration)
        #expect(!study.passedAll)
        #expect(study.resolvedCyberFailureMetric.basis == .substringMarkerProxy)
        #expect(!study.resolvedCyberFailureMetric.certifiesSemanticCompliance)
    }

    @Test("new studies persist honest metric metadata while old JSON remains decodable")
    func metricMetadataCompatibility() throws {
        let responses = [PromptResult(
            name: "case", contrastResponse: "candidate", controlResponse: "control")]
        let classifier = RecordedOutcomeClassifier(judgments: [
            RecordedOutcomeJudgment(
                name: "case", channel: .contrast,
                classification: OutcomeClassification(
                    outcome: .compliance, confidence: 1)),
            RecordedOutcomeJudgment(
                name: "case", channel: .control,
                classification: OutcomeClassification(
                    outcome: .compliance, confidence: 1)),
        ])
        let metrics = ExactBehaviorMetrics(evaluation: RefusalEvaluator.evaluate(
            responses, classifier: classifier))
        var study = ExactResidualInterventionStudy(
            schemaVersion: 1, modelPath: "/model",
            classifierIdentifier: "substring-fallback-v1",
            maximumStrictCyberFailureRate: 0.1,
            maximumSequenceKLLowerBound: 0.5,
            baseline: metrics, baselineCaseOutcomes: nil,
            localityRounds: [], trials: [], acceptedInterventions: [],
            finalMetrics: metrics, finalFirstTokenKL: 0,
            finalSequenceKLLowerBound: 0,
            finalControlPerplexityRatio: 1, finalCaseOutcomes: nil)
        study.cyberFailureMetric = .inferred(
            classifierIdentifier: study.classifierIdentifier)

        let encoded = try JSONEncoder().encode(study)
        let text = try #require(String(data: encoded, encoding: .utf8))
        #expect(text.contains("\"substring_marker_proxy\""))
        #expect(text.contains("\"strictCyberFailureRate\""))
        let persisted = try JSONDecoder().decode(
            ExactResidualInterventionStudy.self, from: encoded)
        #expect(persisted.cyberFailureMetric?.displayLabel
            == "marker/proxy cyber-failure")

        var object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        object.removeValue(forKey: "cyberFailureMetric")
        let legacy = try JSONSerialization.data(withJSONObject: object)
        let decoded = try JSONDecoder().decode(
            ExactResidualInterventionStudy.self, from: legacy)
        #expect(decoded.cyberFailureMetric == nil)
        #expect(decoded.resolvedCyberFailureMetric.basis == .substringMarkerProxy)
    }
}
