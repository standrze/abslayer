import Foundation
@testable import ProbeCore
import Testing

@Suite("Directional adapter authored-strength planning")
struct DirectionalAdapterStrengthPlanTests {
    @Test("omitting the sweep preserves the historical unsuffixed output")
    func defaultOutput() throws {
        let plans = try DirectionalAdapterStrengthPlan.parse(
            environment: [:], baseOutputPath: "/tmp/candidate")
        #expect(plans == [
            .init(multiplier: 1, outputPath: "/tmp/candidate")
        ])
    }

    @Test("explicit strengths produce deterministic safe sibling names")
    func namedSweep() throws {
        let plans = try DirectionalAdapterStrengthPlan.parse(
            environment: ["ABSLAYER_STRENGTHS": " 0.3, 0.5,8e-1 "],
            baseOutputPath: "/tmp/candidate")
        #expect(plans.map(\.multiplier) == [0.3, 0.5, 0.8])
        #expect(plans.map(\.outputPath) == [
            "/tmp/candidate-s0p3",
            "/tmp/candidate-s0p5",
            "/tmp/candidate-s0p8",
        ])
        for plan in plans {
            #expect(URL(fileURLWithPath: plan.outputPath).deletingLastPathComponent().path == "/tmp")
            #expect(!plan.outputPath.contains(","))
        }
    }

    @Test("invalid empty nonfinite nonpositive and duplicate strengths fail")
    func invalidSweeps() {
        for raw in ["", "0.3,", ",0.3", "nan", "inf", "0", "-0.1", "garbage"] {
            #expect(throws: DirectionalAdapterStrengthPlanError.self) {
                try DirectionalAdapterStrengthPlan.parse(
                    environment: ["ABSLAYER_STRENGTHS": raw],
                    baseOutputPath: "/tmp/candidate")
            }
        }
        #expect(throws: DirectionalAdapterStrengthPlanError.duplicateStrength(0.3)) {
            try DirectionalAdapterStrengthPlan.parse(
                environment: ["ABSLAYER_STRENGTHS": "0.3,0.30"],
                baseOutputPath: "/tmp/candidate")
        }
    }
}
