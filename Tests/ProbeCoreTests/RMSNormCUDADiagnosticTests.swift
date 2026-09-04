import Testing

@testable import ProbeCore

@Suite("CUDA RMSNorm remediation diagnostic")
struct RMSNormCUDADiagnosticTests {
    @Test func certificationRequiresExactGraphDisableAndCUDA() throws {
        let configuration = try RMSNormCUDADiagnosticConfiguration.resolve(
            repetitions: 3,
            environment: ["MLX_USE_CUDA_GRAPHS": "0"],
            backend: .cuda)
        #expect(configuration.repetitions == 3)
        #expect(configuration.mlxUseCUDAGraphs == "0")
        #expect(!configuration.cudaGraphsEnabled)

        #expect(
            throws: RMSNormCUDADiagnosticError.explicitGraphDisableRequired(nil)
        ) {
            try RMSNormCUDADiagnosticConfiguration.resolve(
                environment: [:], backend: .cuda)
        }
        #expect(
            throws: RMSNormCUDADiagnosticError.explicitGraphDisableRequired("00")
        ) {
            try RMSNormCUDADiagnosticConfiguration.resolve(
                environment: ["MLX_USE_CUDA_GRAPHS": "00"], backend: .cuda)
        }
        #expect(
            throws: RMSNormCUDADiagnosticError.cudaBackendRequired(.cpu)
        ) {
            try RMSNormCUDADiagnosticConfiguration.resolve(
                environment: ["MLX_USE_CUDA_GRAPHS": "0"], backend: .cpu)
        }
        #expect(
            throws: RMSNormCUDADiagnosticError.invalidRepetitions(1)
        ) {
            try RMSNormCUDADiagnosticConfiguration.resolve(
                repetitions: 1,
                environment: ["MLX_USE_CUDA_GRAPHS": "0"], backend: .cuda)
        }
    }

    @Test func graphOnRequiresAnExplicitDiagnosticOptIn() throws {
        let configuration = try RMSNormCUDADiagnosticConfiguration.resolve(
            repetitions: 2,
            allowCUDAGraphs: true,
            environment: ["MLX_USE_CUDA_GRAPHS": "1"],
            backend: .cuda)
        #expect(configuration.cudaGraphsEnabled)
        #expect(configuration.mlxUseCUDAGraphs == "1")
    }

    @Test func fixtureExercisesTwoDistinctBF16SizedRows() {
        let dimension = RMSNormCUDADiagnosticFixture.gemmaBF16Dimension
        let inputs = RMSNormCUDADiagnosticFixture.inputValues(dimension: dimension)
        let cotangents = RMSNormCUDADiagnosticFixture.cotangentValues(
            dimension: dimension)
        let rowRMS = RMSNormCUDADiagnosticFixture.rowRootMeanSquares(
            dimension: dimension)

        #expect(RMSNormCUDADiagnosticFixture.rows == 2)
        #expect(dimension == 512)
        #expect(inputs.count == 2 * dimension)
        #expect(cotangents.count == inputs.count)
        #expect(rowRMS.count == 2)
        #expect(rowRMS[1] > 4 * rowRMS[0])
    }

    @Test func caseVerdictRequiresToleranceAndRepeatableHashes() {
        func measurement(
            repetition: Int, hash: String, passed: Bool = true
        ) -> RMSNormCUDADiagnosticMeasurement {
            RMSNormCUDADiagnosticMeasurement(
                repetition: repetition,
                forwardSHA256: "forward-\(hash)",
                inputGradientSHA256: "input-\(hash)",
                weightGradientSHA256: "weight-\(hash)",
                allFastValuesFinite: passed,
                allReferenceValuesFinite: true,
                forwardMaximumAbsoluteError: passed ? 0.001 : nil,
                inputGradientMaximumAbsoluteError: passed ? 0.001 : nil,
                weightGradientMaximumAbsoluteError: passed ? 0.001 : nil,
                passedTolerance: passed)
        }

        let exact = RMSNormCUDADiagnosticCaseReport.make(
            label: "bf16", dtype: "bfloat16", dimension: 512,
            forwardTolerance: 0.01, gradientTolerance: 0.02,
            measurements: [
                measurement(repetition: 1, hash: "same"),
                measurement(repetition: 2, hash: "same"),
            ])
        #expect(exact.deterministicHashes)
        #expect(exact.passed)

        let divergent = RMSNormCUDADiagnosticCaseReport.make(
            label: "bf16", dtype: "bfloat16", dimension: 512,
            forwardTolerance: 0.01, gradientTolerance: 0.02,
            measurements: [
                measurement(repetition: 1, hash: "first"),
                measurement(repetition: 2, hash: "second"),
            ])
        #expect(!divergent.deterministicHashes)
        #expect(!divergent.passed)
    }
}
