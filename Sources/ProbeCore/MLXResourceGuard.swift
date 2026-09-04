import Foundation
import MLX

public enum MLXExecutionBackend: String, Equatable, Sendable {
    case metal
    case cuda
    case cpu

    public static var compiled: Self {
        #if MLX_CUDA_BACKEND
        .cuda
        #elseif MLX_CPU_BACKEND
        .cpu
        #else
        .metal
        #endif
    }
}

/// Applies a conservative, process-wide ceiling to MLX allocations before a
/// model is loaded. Metal is bounded against unified host memory; CUDA uses a
/// separate envelope that leaves headroom on a 24-GiB RTX 4090.
public struct MLXResourceLimits: Equatable, Sendable {
    public static let memoryLimitKey = "ABSLAYER_MLX_MEMORY_LIMIT_GIB"
    public static let cacheLimitKey = "ABSLAYER_MLX_CACHE_LIMIT_MIB"

    public static let metalDefaultMemoryLimitGiB = 24.0
    public static let metalDefaultCacheLimitMiB = 256.0
    public static let metalMaximumMemoryLimitGiB = 32.0
    public static let metalMaximumCacheLimitMiB = 256.0
    public static let cudaDefaultMemoryLimitGiB = 20.0
    public static let cudaDefaultCacheLimitMiB = 128.0
    public static let cudaMaximumMemoryLimitGiB = 22.0
    public static let cudaMaximumCacheLimitMiB = 128.0
    public static let cudaDeviceMemoryGiB = 24.0
    public static let cpuDefaultMemoryLimitGiB = 18.0
    public static let cpuDefaultCacheLimitMiB = 128.0
    public static let cpuMaximumMemoryLimitGiB = 20.0
    public static let cpuMaximumCacheLimitMiB = 128.0
    public static let minimumMemoryLimitBytes = 1_073_741_824

    public let backend: MLXExecutionBackend
    public let memoryLimitBytes: Int
    public let cacheLimitBytes: Int
    public let maximumMemoryLimitBytes: Int
    public let maximumCacheLimitBytes: Int

    public init(
        backend: MLXExecutionBackend,
        memoryLimitBytes: Int,
        cacheLimitBytes: Int,
        maximumMemoryLimitBytes: Int,
        maximumCacheLimitBytes: Int
    ) {
        self.backend = backend
        self.memoryLimitBytes = memoryLimitBytes
        self.cacheLimitBytes = cacheLimitBytes
        self.maximumMemoryLimitBytes = maximumMemoryLimitBytes
        self.maximumCacheLimitBytes = maximumCacheLimitBytes
    }

    /// Parses limits without touching global MLX state. `physicalMemoryBytes`
    /// is injectable so boundary behavior can be tested deterministically.
    public static func parse(
        environment: [String: String],
        backend: MLXExecutionBackend = .compiled,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) throws -> Self {
        let policy = try Policy(
            backend: backend,
            physicalMemoryBytes: physicalMemoryBytes)
        let memoryBytes = try parseOverride(
            environment: environment,
            key: memoryLimitKey,
            multiplier: gibibyte,
            minimumBytes: minimumMemoryLimitBytes,
            maximumBytes: policy.maximumMemoryBytes)
            ?? policy.defaultMemoryBytes
        let cacheMaximum = min(
            policy.maximumCacheBytes,
            memoryBytes)
        let cacheBytes = try parseOverride(
            environment: environment,
            key: cacheLimitKey,
            multiplier: mebibyte,
            minimumBytes: 0,
            maximumBytes: cacheMaximum)
            ?? min(policy.defaultCacheBytes, cacheMaximum)
        return Self(
            backend: backend,
            memoryLimitBytes: memoryBytes,
            cacheLimitBytes: cacheBytes,
            maximumMemoryLimitBytes: policy.maximumMemoryBytes,
            maximumCacheLimitBytes: policy.maximumCacheBytes)
    }

    /// Converts the controller's RTX 4090 utilization fraction into the
    /// existing absolute MLX allocation ceiling. The one-GiB floor is the
    /// guard's minimum safe setting; the 22-GiB cap leaves driver headroom.
    public static func cudaMemoryLimitGiB(utilization: Double) throws -> Double {
        guard utilization.isFinite, (0.01 ... 1).contains(utilization) else {
            throw MLXResourceGuardError.invalidGPUUtilization(utilization)
        }
        return min(
            cudaMaximumMemoryLimitGiB,
            max(1, cudaDeviceMemoryGiB * utilization))
    }

    private static let mebibyte = 1_048_576
    private static let gibibyte = 1_073_741_824

    private struct Policy {
        let defaultMemoryBytes: Int
        let maximumMemoryBytes: Int
        let defaultCacheBytes: Int
        let maximumCacheBytes: Int

        init(
            backend: MLXExecutionBackend,
            physicalMemoryBytes: UInt64
        ) throws {
            switch backend {
            case .cuda:
                self.init(
                    defaultMemoryBytes: Int(cudaDefaultMemoryLimitGiB) * gibibyte,
                    maximumMemoryBytes: Int(cudaMaximumMemoryLimitGiB) * gibibyte,
                    defaultCacheBytes: Int(cudaDefaultCacheLimitMiB) * mebibyte,
                    maximumCacheBytes: Int(cudaMaximumCacheLimitMiB) * mebibyte)
            case .metal:
                let physicalMaximum = try Self.halfPhysicalMemory(
                    physicalMemoryBytes)
                let maximum = min(
                    Int(metalMaximumMemoryLimitGiB) * gibibyte,
                    physicalMaximum)
                guard maximum >= gibibyte else {
                    throw MLXResourceGuardError.insufficientPhysicalMemory(
                        physicalMemoryBytes)
                }
                self.init(
                    defaultMemoryBytes: min(
                        Int(metalDefaultMemoryLimitGiB) * gibibyte,
                        maximum),
                    maximumMemoryBytes: maximum,
                    defaultCacheBytes: Int(metalDefaultCacheLimitMiB) * mebibyte,
                    maximumCacheBytes: Int(metalMaximumCacheLimitMiB) * mebibyte)
            case .cpu:
                let physicalMaximum = try Self.halfPhysicalMemory(
                    physicalMemoryBytes)
                let maximum = min(
                    Int(cpuMaximumMemoryLimitGiB) * gibibyte,
                    physicalMaximum)
                guard maximum >= gibibyte else {
                    throw MLXResourceGuardError.insufficientPhysicalMemory(
                        physicalMemoryBytes)
                }
                self.init(
                    defaultMemoryBytes: min(
                        Int(cpuDefaultMemoryLimitGiB) * gibibyte,
                        maximum),
                    maximumMemoryBytes: maximum,
                    defaultCacheBytes: Int(cpuDefaultCacheLimitMiB) * mebibyte,
                    maximumCacheBytes: Int(cpuMaximumCacheLimitMiB) * mebibyte)
            }
        }

        init(
            defaultMemoryBytes: Int,
            maximumMemoryBytes: Int,
            defaultCacheBytes: Int,
            maximumCacheBytes: Int
        ) {
            self.defaultMemoryBytes = defaultMemoryBytes
            self.maximumMemoryBytes = maximumMemoryBytes
            self.defaultCacheBytes = defaultCacheBytes
            self.maximumCacheBytes = maximumCacheBytes
        }

        private static func halfPhysicalMemory(
            _ physicalMemoryBytes: UInt64
        ) throws -> Int {
            guard physicalMemoryBytes > 0 else {
                throw MLXResourceGuardError.insufficientPhysicalMemory(0)
            }
            return Int(min(physicalMemoryBytes / 2, UInt64(Int.max)))
        }
    }

    private static func parseOverride(
        environment: [String: String],
        key: String,
        multiplier: Int,
        minimumBytes: Int,
        maximumBytes: Int
    ) throws -> Int? {
        guard let raw = environment[key] else { return nil }
        guard isPlainDecimal(raw),
              let units = Double(raw),
              units.isFinite
        else {
            throw MLXResourceGuardError.invalidValue(key: key, value: raw)
        }
        let bytes = units * Double(multiplier)
        guard bytes.isFinite,
              bytes >= Double(minimumBytes)
        else {
            throw MLXResourceGuardError.belowSafeMinimum(key: key, value: raw)
        }
        guard bytes <= Double(maximumBytes), bytes <= Double(Int.max) else {
            throw MLXResourceGuardError.exceedsSafeMaximum(key: key, value: raw)
        }
        return Int(bytes.rounded(.down))
    }

    private static func isPlainDecimal(_ value: String) -> Bool {
        guard !value.isEmpty else { return false }
        let pieces = value.split(
            separator: ".",
            omittingEmptySubsequences: false)
        guard pieces.count == 1 || pieces.count == 2,
              !pieces[0].isEmpty,
              pieces[0].allSatisfy(\.isNumber)
        else { return false }
        if pieces.count == 2 {
            guard !pieces[1].isEmpty,
                  pieces[1].allSatisfy(\.isNumber)
            else { return false }
        }
        return true
    }
}

public enum MLXResourceGuardError: LocalizedError, Equatable {
    case invalidValue(key: String, value: String)
    case belowSafeMinimum(key: String, value: String)
    case exceedsSafeMaximum(key: String, value: String)
    case insufficientPhysicalMemory(UInt64)
    case invalidGPUUtilization(Double)
    case applicationFailed(
        requestedMemoryBytes: Int,
        appliedMemoryBytes: Int,
        requestedCacheBytes: Int,
        appliedCacheBytes: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidValue(let key, let value):
            "\(key) must be a finite plain decimal number, not '\(value)'."
        case .belowSafeMinimum(let key, let value):
            "\(key) is below ABSlayer's safe minimum: '\(value)'."
        case .exceedsSafeMaximum(let key, let value):
            "\(key) exceeds ABSlayer's safe maximum: '\(value)'."
        case .insufficientPhysicalMemory(let bytes):
            "The host reports only \(bytes) bytes of physical memory; ABSlayer cannot establish its minimum MLX safety limit."
        case .invalidGPUUtilization(let value):
            "GPU memory utilization must be finite and in 0.01...1, not '\(value)'."
        case .applicationFailed(
            let requestedMemory,
            let appliedMemory,
            let requestedCache,
            let appliedCache):
            "MLX rejected ABSlayer's resource guard "
                + "(memory requested/applied: \(requestedMemory)/\(appliedMemory), "
                + "cache requested/applied: \(requestedCache)/\(appliedCache)); "
                + "refusing to load the model."
        }
    }
}

public enum MLXResourceGuard {
    /// Must be called before any model container is created in the process.
    @discardableResult
    public static func apply(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        emitStatus: Bool = true
    ) throws -> MLXResourceLimits {
        let limits = try MLXResourceLimits.parse(environment: environment)
        Memory.memoryLimit = limits.memoryLimitBytes
        Memory.cacheLimit = limits.cacheLimitBytes
        Memory.clearCache()
        let appliedMemoryLimit = Memory.memoryLimit
        let appliedCacheLimit = Memory.cacheLimit
        guard appliedMemoryLimit == limits.memoryLimitBytes,
              appliedCacheLimit == limits.cacheLimitBytes
        else {
            throw MLXResourceGuardError.applicationFailed(
                requestedMemoryBytes: limits.memoryLimitBytes,
                appliedMemoryBytes: appliedMemoryLimit,
                requestedCacheBytes: limits.cacheLimitBytes,
                appliedCacheBytes: appliedCacheLimit)
        }
        if emitStatus {
            print(
                "MLX resource guard (\(limits.backend.rawValue)): memory limit "
                    + format(limits.memoryLimitBytes, divisor: 1_073_741_824, suffix: "GiB")
                    + ", cache limit "
                    + format(limits.cacheLimitBytes, divisor: 1_048_576, suffix: "MiB")
                    + " (cache cleared)")
        }
        return limits
    }

    /// Applies the harness-provided RTX 4090 utilization fraction without
    /// mutating the process environment or permitting an ambient override to
    /// silently change the controller-bound resource request.
    @discardableResult
    public static func apply(
        gpuMemoryUtilization: Double,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        emitStatus: Bool = true
    ) throws -> MLXResourceLimits {
        var boundEnvironment = environment
        let limit = try MLXResourceLimits.cudaMemoryLimitGiB(
            utilization: gpuMemoryUtilization)
        boundEnvironment[MLXResourceLimits.memoryLimitKey] = String(
            format: "%.9f", limit)
        return try apply(environment: boundEnvironment, emitStatus: emitStatus)
    }

    private static func format(
        _ bytes: Int, divisor: Int, suffix: String
    ) -> String {
        let value = Double(bytes) / Double(divisor)
        return String(format: "%.2f %@", value, suffix)
    }
}

/// Generation settings kept in ProbeCore so command-line entry points use one
/// strict parser instead of silently accepting malformed token caps.
public struct EvaluationGenerationOptions: Equatable, Sendable {
    public static let maximumTokensKey = "ABSLAYER_EVAL_MAX_TOKENS"
    public static let defaultMaximumTokens = 96
    public static let maximumAllowedTokens = 512

    public let maximumTokens: Int

    public static func parse(
        environment: [String: String]
    ) throws -> Self {
        guard let raw = environment[maximumTokensKey] else {
            return Self(maximumTokens: defaultMaximumTokens)
        }
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int(normalized), value > 0,
              value <= maximumAllowedTokens
        else {
            throw EvaluationGenerationOptionsError.invalidMaximumTokens(raw)
        }
        return Self(maximumTokens: value)
    }
}

public enum EvaluationGenerationOptionsError: LocalizedError, Equatable {
    case invalidMaximumTokens(String)

    public var errorDescription: String? {
        switch self {
        case .invalidMaximumTokens(let value):
            "ABSLAYER_EVAL_MAX_TOKENS must be an integer in 1...512, not '\(value)'."
        }
    }
}
