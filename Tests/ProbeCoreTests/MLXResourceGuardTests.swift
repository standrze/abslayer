import Foundation
import Testing
@testable import ProbeCore

@Suite("MLX resource guard parsing")
struct MLXResourceGuardTests {
    private let gib = 1_073_741_824
    private let mib = 1_048_576
    private let sixtyFourGiB = UInt64(64 * 1_073_741_824)

    @Test("uses conservative Metal defaults on a 64 GiB host")
    func conservativeMetalDefaults() throws {
        let limits = try MLXResourceLimits.parse(
            environment: [:], backend: .metal,
            physicalMemoryBytes: sixtyFourGiB)
        #expect(limits.backend == .metal)
        #expect(limits.memoryLimitBytes == 24 * gib)
        #expect(limits.cacheLimitBytes == 256 * mib)
        #expect(limits.maximumMemoryLimitBytes == 32 * gib)
        #expect(limits.maximumCacheLimitBytes == 256 * mib)
    }

    @Test("CUDA defaults leave headroom on a 24 GiB RTX 4090")
    func conservativeCUDADefaults() throws {
        let limits = try MLXResourceLimits.parse(
            environment: [:], backend: .cuda,
            physicalMemoryBytes: sixtyFourGiB)
        #expect(limits.backend == .cuda)
        #expect(limits.memoryLimitBytes == 20 * gib)
        #expect(limits.cacheLimitBytes == 128 * mib)
        #expect(limits.maximumMemoryLimitBytes == 22 * gib)
        #expect(limits.maximumCacheLimitBytes == 128 * mib)
    }

    @Test("accepts finite in-range decimal limits")
    func explicitLimits() throws {
        let limits = try MLXResourceLimits.parse(
            environment: [
                MLXResourceLimits.memoryLimitKey: "20.5",
                MLXResourceLimits.cacheLimitKey: "128",
            ],
            backend: .metal,
            physicalMemoryBytes: sixtyFourGiB)
        #expect(limits.memoryLimitBytes == Int(20.5 * Double(gib)))
        #expect(limits.cacheLimitBytes == 128 * mib)
    }

    @Test("defaults to at most half of physical memory")
    func physicalMemoryBound() throws {
        let limits = try MLXResourceLimits.parse(
            environment: [:], backend: .metal,
            physicalMemoryBytes: UInt64(16 * gib))
        #expect(limits.memoryLimitBytes == 8 * gib)
        #expect(limits.cacheLimitBytes == 256 * mib)
    }

    @Test("rejects malformed and nonfinite values")
    func malformedValues() {
        for raw in ["", " ", "garbage", "nan", "inf", "-1", "+1", "1e1", ".5", "5."] {
            #expect(throws: MLXResourceGuardError.self) {
                try MLXResourceLimits.parse(
                    environment: [MLXResourceLimits.memoryLimitKey: raw],
                    backend: .cuda,
                    physicalMemoryBytes: sixtyFourGiB)
            }
        }
    }

    @Test("rejects values outside the fixed safety envelope")
    func boundedValues() {
        #expect(throws: MLXResourceGuardError.belowSafeMinimum(
            key: MLXResourceLimits.memoryLimitKey, value: "0.5")) {
            try MLXResourceLimits.parse(
                environment: [MLXResourceLimits.memoryLimitKey: "0.5"],
                backend: .metal,
                physicalMemoryBytes: sixtyFourGiB)
        }
        #expect(throws: MLXResourceGuardError.exceedsSafeMaximum(
            key: MLXResourceLimits.memoryLimitKey, value: "33")) {
            try MLXResourceLimits.parse(
                environment: [MLXResourceLimits.memoryLimitKey: "33"],
                backend: .metal,
                physicalMemoryBytes: sixtyFourGiB)
        }
        #expect(throws: MLXResourceGuardError.exceedsSafeMaximum(
            key: MLXResourceLimits.cacheLimitKey, value: "257")) {
            try MLXResourceLimits.parse(
                environment: [MLXResourceLimits.cacheLimitKey: "257"],
                backend: .metal,
                physicalMemoryBytes: sixtyFourGiB)
        }
    }

    @Test("explicit memory limits cannot exceed half of host RAM")
    func hostBound() {
        #expect(throws: MLXResourceGuardError.exceedsSafeMaximum(
            key: MLXResourceLimits.memoryLimitKey, value: "9")) {
            try MLXResourceLimits.parse(
                environment: [MLXResourceLimits.memoryLimitKey: "9"],
                backend: .metal,
                physicalMemoryBytes: UInt64(16 * gib))
        }
    }

    @Test("CUDA overrides cannot exceed the fixed 4090 envelope")
    func cudaBounds() throws {
        let accepted = try MLXResourceLimits.parse(
            environment: [
                MLXResourceLimits.memoryLimitKey: "22",
                MLXResourceLimits.cacheLimitKey: "64.5",
            ],
            backend: .cuda,
            physicalMemoryBytes: sixtyFourGiB)
        #expect(accepted.memoryLimitBytes == 22 * gib)
        #expect(accepted.cacheLimitBytes == Int(64.5 * Double(mib)))

        #expect(throws: MLXResourceGuardError.self) {
            try MLXResourceLimits.parse(
                environment: [MLXResourceLimits.memoryLimitKey: "22.0001"],
                backend: .cuda,
                physicalMemoryBytes: sixtyFourGiB)
        }
        #expect(throws: MLXResourceGuardError.self) {
            try MLXResourceLimits.parse(
                environment: [MLXResourceLimits.cacheLimitKey: "129"],
                backend: .cuda,
                physicalMemoryBytes: sixtyFourGiB)
        }
    }

    @Test("zero cache disables recycling while zero memory is rejected")
    func zeroCache() throws {
        let limits = try MLXResourceLimits.parse(
            environment: [MLXResourceLimits.cacheLimitKey: "0"],
            backend: .cuda,
            physicalMemoryBytes: sixtyFourGiB)
        #expect(limits.cacheLimitBytes == 0)
        #expect(throws: MLXResourceGuardError.self) {
            try MLXResourceLimits.parse(
                environment: [MLXResourceLimits.memoryLimitKey: "0"],
                backend: .cuda,
                physicalMemoryBytes: sixtyFourGiB)
        }
    }
}

@Suite("Evaluation generation option parsing")
struct EvaluationGenerationOptionsTests {
    @Test("defaults evaluation generations to 96 tokens")
    func defaults() throws {
        let options = try EvaluationGenerationOptions.parse(environment: [:])
        #expect(options.maximumTokens == 96)
    }

    @Test("accepts a bounded explicit token cap")
    func explicitValue() throws {
        let options = try EvaluationGenerationOptions.parse(environment: [
            EvaluationGenerationOptions.maximumTokensKey: " 64 "
        ])
        #expect(options.maximumTokens == 64)
    }

    @Test("rejects malformed, nonpositive, and excessive token caps")
    func invalidValues() {
        for raw in ["", "1.5", "0", "-1", "513"] {
            #expect(throws: EvaluationGenerationOptionsError.invalidMaximumTokens(raw)) {
                try EvaluationGenerationOptions.parse(environment: [
                    EvaluationGenerationOptions.maximumTokensKey: raw
                ])
            }
        }
    }
}

@Suite("Model-load resource guard coverage")
struct ModelLoadResourceGuardCoverageTests {
    @Test("every Hugging Face model-container load is immediately guarded")
    func allModelLoadsAreGuarded() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let sources = packageRoot.appendingPathComponent("Sources")
        let enumerator = FileManager.default.enumerator(
            at: sources,
            includingPropertiesForKeys: nil)
        var loadCount = 0
        var uncovered = [String]()
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let lines = try String(contentsOf: url, encoding: .utf8)
                .components(separatedBy: .newlines)
            for index in lines.indices where
                lines[index].contains("#huggingFaceLoadModelContainer")
            {
                loadCount += 1
                let priorLine = index > 0 ? lines[index - 1] : ""
                if !priorLine.contains("MLXResourceGuard.apply") {
                    uncovered.append(
                        "\(url.lastPathComponent):\(index + 1)")
                }
            }
        }
        #expect(loadCount > 0)
        #expect(uncovered.isEmpty, "unguarded model loads: \(uncovered)")
    }
}
