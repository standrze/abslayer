import Foundation
import MLX
import MLXFast

/// Configuration for the focused CUDA regression diagnostic corresponding to
/// upstream MLX commit a5a684db596c117f13f7bacaea9902d0ad6d28a6.
public struct RMSNormCUDADiagnosticConfiguration: Equatable, Sendable {
    public static let upstreamFixCommit = "a5a684db596c117f13f7bacaea9902d0ad6d28a6"
    public static let defaultRepetitions = 5
    public static let maximumRepetitions = 100

    public let repetitions: Int
    public let mlxUseCUDAGraphs: String
    public let cudaGraphsEnabled: Bool

    public static func resolve(
        repetitions: Int = defaultRepetitions,
        allowCUDAGraphs: Bool = false,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        backend: MLXExecutionBackend = .compiled
    ) throws -> Self {
        guard backend == .cuda else {
            throw RMSNormCUDADiagnosticError.cudaBackendRequired(backend)
        }
        guard (2...maximumRepetitions).contains(repetitions) else {
            throw RMSNormCUDADiagnosticError.invalidRepetitions(repetitions)
        }

        let rawGraphSetting = environment["MLX_USE_CUDA_GRAPHS"]
        // MLX defaults this setting to enabled and parses an explicit value
        // with C atoi. Certification is intentionally stricter: require the
        // unambiguous literal "0" unless a graph-on diagnostic was requested.
        let graphsEnabled = rawGraphSetting != "0"
        guard !graphsEnabled || allowCUDAGraphs else {
            throw RMSNormCUDADiagnosticError.explicitGraphDisableRequired(
                rawGraphSetting)
        }
        return Self(
            repetitions: repetitions,
            mlxUseCUDAGraphs: rawGraphSetting ?? "unset",
            cudaGraphsEnabled: graphsEnabled)
    }
}

public enum RMSNormCUDADiagnosticFixture {
    public static let rows = 2
    public static let gemmaBF16Dimension = 512
    public static let upstreamFP32Dimension = 256
    public static let epsilon: Float = 1e-6

    public static func inputValues(dimension: Int) -> [Float] {
        precondition(dimension > 0)
        let first = (0..<dimension).map { index in
            Float(16 + index % 16) / 64
        }
        let second = (0..<dimension).map { index in
            Float(24 + index % 16) / 16
        }
        return first + second
    }

    public static func weightValues(dimension: Int) -> [Float] {
        precondition(dimension > 0)
        return (0..<dimension).map { index in
            Float(12 + index % 8) / 16
        }
    }

    public static func cotangentValues(dimension: Int) -> [Float] {
        precondition(dimension > 0)
        return (0..<(rows * dimension)).map { index in
            let magnitude = Float(4 + index % 11) / 16
            return index.isMultiple(of: 2) ? magnitude : -magnitude
        }
    }

    public static func rowRootMeanSquares(dimension: Int) -> [Float] {
        let values = inputValues(dimension: dimension)
        return (0..<rows).map { row in
            let start = row * dimension
            let sumSquares = values[start..<(start + dimension)].reduce(Float(0)) {
                $0 + $1 * $1
            }
            return sqrt(sumSquares / Float(dimension))
        }
    }
}

public struct RMSNormCUDADiagnosticMeasurement: Codable, Equatable, Sendable {
    public let repetition: Int
    public let forwardSHA256: String
    public let inputGradientSHA256: String
    public let weightGradientSHA256: String
    public let allFastValuesFinite: Bool
    public let allReferenceValuesFinite: Bool
    public let forwardMaximumAbsoluteError: Float?
    public let inputGradientMaximumAbsoluteError: Float?
    public let weightGradientMaximumAbsoluteError: Float?
    public let passedTolerance: Bool
}

public struct RMSNormCUDADiagnosticCaseReport: Codable, Equatable, Sendable {
    public let label: String
    public let dtype: String
    public let rows: Int
    public let dimension: Int
    public let epsilon: Float
    public let inputRowRootMeanSquares: [Float]
    public let forwardTolerance: Float
    public let gradientTolerance: Float
    public let measurements: [RMSNormCUDADiagnosticMeasurement]
    public let deterministicHashes: Bool
    public let passed: Bool

    static func make(
        label: String,
        dtype: String,
        dimension: Int,
        forwardTolerance: Float,
        gradientTolerance: Float,
        measurements: [RMSNormCUDADiagnosticMeasurement]
    ) -> Self {
        let deterministic = !measurements.isEmpty
            && Set(measurements.map(\.forwardSHA256)).count == 1
            && Set(measurements.map(\.inputGradientSHA256)).count == 1
            && Set(measurements.map(\.weightGradientSHA256)).count == 1
        return Self(
            label: label,
            dtype: dtype,
            rows: RMSNormCUDADiagnosticFixture.rows,
            dimension: dimension,
            epsilon: RMSNormCUDADiagnosticFixture.epsilon,
            inputRowRootMeanSquares:
                RMSNormCUDADiagnosticFixture.rowRootMeanSquares(
                    dimension: dimension),
            forwardTolerance: forwardTolerance,
            gradientTolerance: gradientTolerance,
            measurements: measurements,
            deterministicHashes: deterministic,
            passed: deterministic && measurements.allSatisfy(\.passedTolerance))
    }
}

public struct RMSNormCUDADiagnosticReport: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let upstreamFixCommit: String
    public let compiledBackend: String
    public let mlxUseCUDAGraphs: String
    public let cudaGraphsEnabled: Bool
    public let repetitions: Int
    public let cases: [RMSNormCUDADiagnosticCaseReport]
    public let passed: Bool
}

public enum RMSNormCUDADiagnosticEngine {
    public static func run(
        repetitions: Int = RMSNormCUDADiagnosticConfiguration.defaultRepetitions,
        allowCUDAGraphs: Bool = false,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> RMSNormCUDADiagnosticReport {
        let configuration = try RMSNormCUDADiagnosticConfiguration.resolve(
            repetitions: repetitions,
            allowCUDAGraphs: allowCUDAGraphs,
            environment: environment)
        return try Device.withDefaultDevice(.gpu) {
            let cases = try [
                runCase(
                    plan: CasePlan(
                        label: "gemma4-bfloat16-d512",
                        dtype: .bfloat16,
                        dtypeName: "bfloat16",
                        dimension: RMSNormCUDADiagnosticFixture.gemmaBF16Dimension,
                        forwardTolerance: 1e-2,
                        gradientTolerance: 2e-2),
                    repetitions: configuration.repetitions),
                runCase(
                    plan: CasePlan(
                        label: "upstream-float32-d256",
                        dtype: .float32,
                        dtypeName: "float32",
                        dimension: RMSNormCUDADiagnosticFixture.upstreamFP32Dimension,
                        forwardTolerance: 1e-5,
                        gradientTolerance: 1e-5),
                    repetitions: configuration.repetitions),
            ]
            return RMSNormCUDADiagnosticReport(
                schemaVersion: 1,
                upstreamFixCommit:
                    RMSNormCUDADiagnosticConfiguration.upstreamFixCommit,
                compiledBackend: MLXExecutionBackend.compiled.rawValue,
                mlxUseCUDAGraphs: configuration.mlxUseCUDAGraphs,
                cudaGraphsEnabled: configuration.cudaGraphsEnabled,
                repetitions: configuration.repetitions,
                cases: cases,
                passed: cases.allSatisfy(\.passed))
        }
    }

    private struct CasePlan {
        let label: String
        let dtype: DType
        let dtypeName: String
        let dimension: Int
        let forwardTolerance: Float
        let gradientTolerance: Float
    }

    private static func runCase(
        plan: CasePlan, repetitions: Int
    ) throws -> RMSNormCUDADiagnosticCaseReport {
        var measurements = [RMSNormCUDADiagnosticMeasurement]()
        measurements.reserveCapacity(repetitions)
        for repetition in 1...repetitions {
            measurements.append(try measure(plan: plan, repetition: repetition))
        }
        return RMSNormCUDADiagnosticCaseReport.make(
            label: plan.label,
            dtype: plan.dtypeName,
            dimension: plan.dimension,
            forwardTolerance: plan.forwardTolerance,
            gradientTolerance: plan.gradientTolerance,
            measurements: measurements)
    }

    private static func measure(
        plan: CasePlan, repetition: Int
    ) throws -> RMSNormCUDADiagnosticMeasurement {
        let shape = [RMSNormCUDADiagnosticFixture.rows, plan.dimension]
        let input = MLXArray(
            RMSNormCUDADiagnosticFixture.inputValues(dimension: plan.dimension),
            shape).asType(plan.dtype)
        let weight = MLXArray(
            RMSNormCUDADiagnosticFixture.weightValues(dimension: plan.dimension)
        ).asType(plan.dtype)
        let cotangent = MLXArray(
            RMSNormCUDADiagnosticFixture.cotangentValues(dimension: plan.dimension),
            shape)

        let fastForward = MLXFast.rmsNorm(
            input, weight: weight, eps: RMSNormCUDADiagnosticFixture.epsilon)
            .asType(.float32)
        let referenceForward = referenceRMSNorm(
            input, weight: weight, eps: RMSNormCUDADiagnosticFixture.epsilon)

        let fastValueAndGrad = valueAndGrad(
            { arrays -> [MLXArray] in
                let output = MLXFast.rmsNorm(
                    arrays[0], weight: arrays[1],
                    eps: RMSNormCUDADiagnosticFixture.epsilon)
                    .asType(.float32)
                return [(output * arrays[2]).sum()]
            },
            argumentNumbers: [0, 1])
        let referenceValueAndGrad = valueAndGrad(
            { arrays -> [MLXArray] in
                let output = referenceRMSNorm(
                    arrays[0], weight: arrays[1],
                    eps: RMSNormCUDADiagnosticFixture.epsilon)
                return [(output * arrays[2]).sum()]
            },
            argumentNumbers: [0, 1])
        let (_, fastGradients) = fastValueAndGrad([input, weight, cotangent])
        let (_, referenceGradients) = referenceValueAndGrad(
            [input, weight, cotangent])
        guard fastGradients.count == 2, referenceGradients.count == 2 else {
            throw RMSNormCUDADiagnosticError.unexpectedGradientCount(
                fast: fastGradients.count,
                reference: referenceGradients.count)
        }
        let fastInputGradient = fastGradients[0].asType(.float32)
        let fastWeightGradient = fastGradients[1].asType(.float32)
        let referenceInputGradient = referenceGradients[0].asType(.float32)
        let referenceWeightGradient = referenceGradients[1].asType(.float32)

        let forwardError = abs(fastForward - referenceForward).max()
        let inputGradientError = abs(
            fastInputGradient - referenceInputGradient).max()
        let weightGradientError = abs(
            fastWeightGradient - referenceWeightGradient).max()
        let fastFinite = [fastForward, fastInputGradient, fastWeightGradient].map {
            isFinite($0).all()
        }
        let referenceFinite = [
            referenceForward, referenceInputGradient, referenceWeightGradient,
        ].map { isFinite($0).all() }
        eval(
            [
                fastForward, referenceForward,
                fastInputGradient, fastWeightGradient,
                referenceInputGradient, referenceWeightGradient,
                forwardError, inputGradientError, weightGradientError,
            ] + fastFinite + referenceFinite)

        let allFastValuesFinite = fastFinite.allSatisfy {
            $0.item(Bool.self)
        }
        let allReferenceValuesFinite = referenceFinite.allSatisfy {
            $0.item(Bool.self)
        }
        let forwardMaximumAbsoluteError = finiteScalar(forwardError)
        let inputGradientMaximumAbsoluteError = finiteScalar(inputGradientError)
        let weightGradientMaximumAbsoluteError = finiteScalar(weightGradientError)
        let passedTolerance = allFastValuesFinite && allReferenceValuesFinite
            && forwardMaximumAbsoluteError.map { $0 <= plan.forwardTolerance } == true
            && inputGradientMaximumAbsoluteError.map { $0 <= plan.gradientTolerance } == true
            && weightGradientMaximumAbsoluteError.map { $0 <= plan.gradientTolerance } == true

        return RMSNormCUDADiagnosticMeasurement(
            repetition: repetition,
            forwardSHA256: tensorSHA256(fastForward),
            inputGradientSHA256: tensorSHA256(fastInputGradient),
            weightGradientSHA256: tensorSHA256(fastWeightGradient),
            allFastValuesFinite: allFastValuesFinite,
            allReferenceValuesFinite: allReferenceValuesFinite,
            forwardMaximumAbsoluteError: forwardMaximumAbsoluteError,
            inputGradientMaximumAbsoluteError: inputGradientMaximumAbsoluteError,
            weightGradientMaximumAbsoluteError: weightGradientMaximumAbsoluteError,
            passedTolerance: passedTolerance)
    }

    private static func referenceRMSNorm(
        _ input: MLXArray, weight: MLXArray, eps: Float
    ) -> MLXArray {
        let input32 = input.asType(.float32)
        let weight32 = weight.asType(.float32)
        let inverseRMS = rsqrt(
            input32.square().mean(axis: -1, keepDims: true) + eps)
        return input32 * inverseRMS * weight32
    }

    private static func finiteScalar(_ array: MLXArray) -> Float? {
        let value = array.item(Float.self)
        return value.isFinite ? value : nil
    }

    static func tensorSHA256(_ array: MLXArray) -> String {
        let values = array.asType(.float32).asArray(Float.self)
        var bytes = Data("abslayer-rmsnorm-tensor-v1\0".utf8)
        for dimension in array.shape {
            var canonicalDimension = UInt64(dimension).littleEndian
            Swift.withUnsafeBytes(of: &canonicalDimension) {
                bytes.append(contentsOf: $0)
            }
        }
        for value in values {
            var canonicalBits = value.bitPattern.littleEndian
            Swift.withUnsafeBytes(of: &canonicalBits) {
                bytes.append(contentsOf: $0)
            }
        }
        return ScreeningReviewProvenance.sha256(bytes)
    }
}

public enum RMSNormCUDADiagnosticError: LocalizedError, Equatable {
    case cudaBackendRequired(MLXExecutionBackend)
    case invalidRepetitions(Int)
    case explicitGraphDisableRequired(String?)
    case unexpectedGradientCount(fast: Int, reference: Int)

    public var errorDescription: String? {
        switch self {
        case .cudaBackendRequired(let backend):
            "CUDA RMSNorm diagnostics require a CUDA build, not \(backend.rawValue)."
        case .invalidRepetitions(let count):
            "RMSNorm diagnostic repetitions must be between 2 and "
                + "\(RMSNormCUDADiagnosticConfiguration.maximumRepetitions), not \(count)."
        case .explicitGraphDisableRequired(let value):
            "Set MLX_USE_CUDA_GRAPHS=0 for the certification run; current value is "
                + "\(value ?? "unset"). Pass --allow-cuda-graphs only for a later diagnostic."
        case .unexpectedGradientCount(let fast, let reference):
            "RMSNorm VJP returned unexpected gradient counts "
                + "(fast=\(fast), reference=\(reference))."
        }
    }
}
