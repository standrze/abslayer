import Foundation

public struct PromptPair: Codable, Equatable, Sendable {
    public let name: String
    public let contrast: String
    public let control: String
    /// Stable corpus metadata. These fields are optional so prompt files made
    /// before schema version 2 remain readable.
    public let category: String?
    public let source: String?
    public let controlSource: String?
    public let split: String?
    public let requestType: String?
    /// Optional benign assistant answer used only for teacher-forced utility
    /// preservation. Older prompt corpora omit this field and remain valid.
    public let controlReferenceResponse: String?

    public init(
        name: String,
        contrast: String,
        control: String,
        category: String? = nil,
        source: String? = nil,
        controlSource: String? = nil,
        split: String? = nil,
        requestType: String? = nil,
        controlReferenceResponse: String? = nil
    ) {
        self.name = name
        self.contrast = contrast
        self.control = control
        self.category = category
        self.source = source
        self.controlSource = controlSource
        self.split = split
        self.requestType = requestType
        self.controlReferenceResponse = controlReferenceResponse
    }
}

public struct LayerScore: Sendable {
    public let layer: Int
    public let cosineDistance: Double
    public let directionAgreement: Double
    public let medianDirectionAgreement: Double
    public let silhouette: Double

    public init(
        layer: Int,
        cosineDistance: Double,
        directionAgreement: Double,
        medianDirectionAgreement: Double = 0,
        silhouette: Double = 0
    ) {
        self.layer = layer
        self.cosineDistance = cosineDistance
        self.directionAgreement = directionAgreement
        self.medianDirectionAgreement = medianDirectionAgreement
        self.silhouette = silhouette
    }
}

public struct PromptResult: Codable, Sendable, Equatable {
    public let name: String
    public let category: String?
    /// Optional generation condition used for controlled system-prompt
    /// experiments. Legacy response artifacts omit this field.
    public let systemPrompt: String?
    public let contrastPrompt: String?
    public let controlPrompt: String?
    public let contrastResponse: String
    public let controlResponse: String

    public init(
        name: String, contrastResponse: String, controlResponse: String,
        category: String? = nil, systemPrompt: String? = nil,
        contrastPrompt: String? = nil, controlPrompt: String? = nil
    ) {
        self.name = name
        self.category = category
        self.systemPrompt = systemPrompt
        self.contrastPrompt = contrastPrompt
        self.controlPrompt = controlPrompt
        self.contrastResponse = contrastResponse
        self.controlResponse = controlResponse
    }
}

public struct ProbeReport: Sendable {
    public let model: String
    public let pairCount: Int
    public let layers: [LayerScore]
    public let responses: [PromptResult]
    public let directions: [[Float]]
    public let subspaces: [[[Float]]]
    /// Full, stable SOM candidates and diagnostics, keyed by zero-based layer.
    /// This is empty for non-SOM extraction and excludes unselected layers.
    public let somResultsByLayer: [Int: SOMDirectionResult]

    public init(
        model: String,
        pairCount: Int,
        layers: [LayerScore],
        responses: [PromptResult],
        directions: [[Float]],
        subspaces: [[[Float]]],
        somResultsByLayer: [Int: SOMDirectionResult] = [:]
    ) {
        self.model = model
        self.pairCount = pairCount
        self.layers = layers
        self.responses = responses
        self.directions = directions
        self.subspaces = subspaces
        self.somResultsByLayer = somResultsByLayer
    }

    public var rendered: String {
        let maximum = layers.map(\.cosineDistance).max() ?? 1
        var lines = [
            "Model: \(model)",
            "Pairs: \(pairCount)",
            "",
            "Layer  distance  delta agreement  median agreement  silhouette",
            "       (pair separation)  (shared delta)  (mean vs median)  (class separation)",
        ]

        for score in layers {
            let width = maximum > 0 ? Int((score.cosineDistance / maximum) * 32) : 0
            let bar = String(repeating: "█", count: width)
            lines.append(String(
                format: "L%02d    %7.4f       %+7.4f          %+7.4f       %+7.4f  %@",
                score.layer,
                score.cosineDistance,
                score.directionAgreement,
                score.medianDirectionAgreement,
                score.silhouette,
                bar
            ))
        }

        lines.append("")
        lines.append("Generated answers (first 500 characters):")
        for response in responses {
            lines.append("")
            lines.append("[\(response.name)] contrast: \(Self.clip(response.contrastResponse))")
            lines.append("[\(response.name)] control:  \(Self.clip(response.controlResponse))")
        }
        lines.append("")
        lines.append(RefusalEvaluator.evaluate(
            responses, classifier: SubstringOutcomeClassifier()).rendered)
        return lines.joined(separator: "\n")
    }

    private static func clip(_ value: String) -> String {
        String(value.replacingOccurrences(of: "\n", with: " ").prefix(500))
    }
}

public enum LayerMath {
    public static func cosineSimilarity(_ lhs: [Float], _ rhs: [Float]) -> Double {
        precondition(lhs.count == rhs.count)
        var dot = 0.0
        var leftSquared = 0.0
        var rightSquared = 0.0
        for (left, right) in zip(lhs, rhs) {
            let l = Double(left)
            let r = Double(right)
            dot += l * r
            leftSquared += l * l
            rightSquared += r * r
        }
        let denominator = sqrt(leftSquared) * sqrt(rightSquared)
        return denominator == 0 ? 0 : dot / denominator
    }

    public static func cosineDistance(_ lhs: [Float], _ rhs: [Float]) -> Double {
        1 - cosineSimilarity(lhs, rhs)
    }

    public static func meanPairwiseAgreement(_ vectors: [[Float]]) -> Double {
        guard vectors.count >= 2 else { return 0 }
        var total = 0.0
        var comparisons = 0
        for left in 0 ..< vectors.count {
            for right in (left + 1) ..< vectors.count {
                total += cosineSimilarity(vectors[left], vectors[right])
                comparisons += 1
            }
        }
        return total / Double(comparisons)
    }

    /// Weiszfeld geometric median. This is robust to activation outliers and is
    /// used as a diagnostic alongside the ordinary centroid direction.
    public static func geometricMedian(
        _ vectors: [[Float]], tolerance: Float = 1e-5, maximumIterations: Int = 128
    ) -> [Float] {
        guard let first = vectors.first else { return [] }
        precondition(vectors.allSatisfy { $0.count == first.count })
        var estimate = AbliterationMath.mean(vectors)
        for _ in 0 ..< maximumIterations {
            var numerator = Array(repeating: Float.zero, count: first.count)
            var denominator: Float = 0
            var exact: [Float]?
            for vector in vectors {
                let distance = sqrt(zip(vector, estimate).reduce(Float.zero) {
                    let delta = $1.0 - $1.1
                    return $0 + delta * delta
                })
                if distance < tolerance {
                    exact = vector
                    break
                }
                let weight = 1 / distance
                denominator += weight
                for index in numerator.indices { numerator[index] += vector[index] * weight }
            }
            if let exact { return exact }
            guard denominator > 0 else { return estimate }
            let next = numerator.map { $0 / denominator }
            let movement = sqrt(zip(next, estimate).reduce(Float.zero) {
                let delta = $1.0 - $1.1
                return $0 + delta * delta
            })
            estimate = next
            if movement < tolerance { break }
        }
        return estimate
    }

    /// Mean two-cluster silhouette using cosine distance, matching the role of
    /// Heretic's residual-geometry diagnostic without external dependencies.
    public static func binarySilhouette(_ first: [[Float]], _ second: [[Float]]) -> Double {
        guard first.count > 1, second.count > 1 else { return 0 }
        func score(index: Int, own: [[Float]], other: [[Float]]) -> Double {
            let vector = own[index]
            let a = own.indices.filter { $0 != index }.reduce(0) {
                $0 + cosineDistance(vector, own[$1])
            } / Double(own.count - 1)
            let b = other.reduce(0) { $0 + cosineDistance(vector, $1) } / Double(other.count)
            let scale = max(a, b)
            return scale == 0 ? 0 : (b - a) / scale
        }
        let values = first.indices.map { score(index: $0, own: first, other: second) }
            + second.indices.map { score(index: $0, own: second, other: first) }
        return values.reduce(0, +) / Double(values.count)
    }
}

public enum PromptFile {
    public static func load(_ path: String) throws -> [PromptPair] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let decoder = JSONDecoder()
        let pairs: [PromptPair]
        if let explicit = try? decoder.decode([PromptPair].self, from: data) {
            pairs = explicit
        } else if let document = try? decoder.decode(PromptPairDocument.self, from: data) {
            pairs = document.pairs
        } else {
            let holdout = try decoder.decode(HoldoutPairs.self, from: data)
            guard holdout.harmful.count == holdout.harmless.count else {
                throw ProbeError.unpairedPromptFile(
                    harmful: holdout.harmful.count, harmless: holdout.harmless.count)
            }
            pairs = zip(holdout.harmful, holdout.harmless).enumerated().map {
                PromptPair(name: "holdout-\($0.offset + 1)", contrast: $0.element.0,
                           control: $0.element.1)
            }
        }
        guard !pairs.isEmpty else { throw ProbeError.emptyPromptFile }
        return pairs
    }

    /// Writes the metadata-preserving schema. `load` also accepts the legacy
    /// harmful/harmless parallel-array format for old experiment artifacts.
    public static func write(
        _ pairs: [PromptPair], to path: String, modelCondition: String? = nil
    ) throws {
        guard !pairs.isEmpty else { throw ProbeError.emptyPromptFile }
        let document = PromptPairDocument(
            schemaVersion: 2, modelCondition: modelCondition, pairs: pairs)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(document).write(
            to: URL(fileURLWithPath: path).standardizedFileURL, options: .atomic)
    }

    private struct PromptPairDocument: Codable {
        let schemaVersion: Int
        let modelCondition: String?
        let pairs: [PromptPair]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version"
            case modelCondition = "model_condition"
            case pairs
        }
    }

    private struct HoldoutPairs: Codable {
        let harmful: [String]
        let harmless: [String]
    }
}

public enum ProbeError: LocalizedError {
    case emptyPromptFile
    case unpairedPromptFile(harmful: Int, harmless: Int)
    case unsupportedModel(String)
    case emptyPromptTokenization(name: String)
    case promptTooLong(name: String, tokenCount: Int)
    case referenceContinuationTokenizationMismatch(name: String)

    public var errorDescription: String? {
        switch self {
        case .emptyPromptFile:
            "The prompt file contains no pairs."
        case .unpairedPromptFile(let harmful, let harmless):
            "The prompt file has \(harmful) harmful prompts and \(harmless) harmless prompts."
        case .unsupportedModel(let type):
            "This small PoC can inspect Gemma 3 only; MLX loaded \(type)."
        case .emptyPromptTokenization(let name):
            "Prompt pair '\(name)' tokenized to an empty sequence."
        case .promptTooLong(let name, let count):
            "Prompt pair '\(name)' has \(count) tokens. Keep PoC prompts at 512 tokens or fewer."
        case .referenceContinuationTokenizationMismatch(let name):
            "Prompt pair '\(name)' has a control reference response that cannot be tokenized as a continuation of the rendered assistant-generation prefix."
        }
    }
}
