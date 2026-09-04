import Foundation

/// One output requested from a directional-adapter authoring run.
public struct DirectionalAdapterStrengthPlan: Sendable, Equatable {
    public let multiplier: Float
    public let outputPath: String

    public init(multiplier: Float, outputPath: String) {
        self.multiplier = multiplier
        self.outputPath = outputPath
    }

    /// Parses an optional comma-separated authored-strength sweep. Omitting the
    /// variable retains the historical, unsuffixed single output at strength 1.
    /// An explicit sweep always writes sibling directories with deterministic,
    /// path-safe suffixes such as `adapter-s0p3` and `adapter-s0p8`.
    public static func parse(
        environment: [String: String], baseOutputPath: String
    ) throws -> [DirectionalAdapterStrengthPlan] {
        let base = URL(fileURLWithPath: baseOutputPath).standardizedFileURL
        let baseName = base.lastPathComponent
        guard !baseName.isEmpty, base.path != "/" else {
            throw DirectionalAdapterStrengthPlanError.invalidOutputPath(base.path)
        }
        guard let raw = environment["ABSLAYER_STRENGTHS"] else {
            return [.init(multiplier: 1, outputPath: base.path)]
        }

        let tokens = raw.split(separator: ",", omittingEmptySubsequences: false)
        guard !tokens.isEmpty else {
            throw DirectionalAdapterStrengthPlanError.invalidStrengths(raw)
        }
        var seen = Set<Float>()
        var plans = [DirectionalAdapterStrengthPlan]()
        plans.reserveCapacity(tokens.count)
        for token in tokens {
            let valueText = token.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !valueText.isEmpty, let value = Float(valueText),
                  value.isFinite, value > 0
            else {
                throw DirectionalAdapterStrengthPlanError.invalidStrengths(raw)
            }
            guard seen.insert(value).inserted else {
                throw DirectionalAdapterStrengthPlanError.duplicateStrength(value)
            }
            let output = base.deletingLastPathComponent().appendingPathComponent(
                baseName + "-s" + safeTag(value), isDirectory: true)
            plans.append(.init(multiplier: value, outputPath: output.path))
        }
        return plans
    }

    private static func safeTag(_ value: Float) -> String {
        // Swift's Float description is a locale-independent shortest
        // round-tripping representation. Map its punctuation to filename-safe
        // letters while preserving a deterministic one-to-one numeric tag.
        String(value)
            .replacingOccurrences(of: ".", with: "p")
            .replacingOccurrences(of: "+", with: "p")
            .replacingOccurrences(of: "-", with: "m")
    }
}

public enum DirectionalAdapterStrengthPlanError: LocalizedError, Equatable {
    case invalidStrengths(String)
    case duplicateStrength(Float)
    case invalidOutputPath(String)

    public var errorDescription: String? {
        switch self {
        case .invalidStrengths(let value):
            "ABSLAYER_STRENGTHS must be a comma-separated list of finite positive numbers, not '\(value)'."
        case .duplicateStrength(let value):
            "ABSLAYER_STRENGTHS contains duplicate numeric strength \(value)."
        case .invalidOutputPath(let path):
            "Directional-adapter output path cannot be suffixed safely: \(path)"
        }
    }
}
