import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Materialized FP32 module-I/O clouds for one zero-based attention layer.
/// Control prompts are the good cloud and contrast prompts are the bad cloud.
public struct ARAProjectionLayerCapture: Sendable {
    public let layer: Int
    public let goodInput: [[Float]]
    public let goodOutput: [[Float]]
    public let badInput: [[Float]]
    public let badOutput: [[Float]]

    public func moduleSamples() -> ARAModuleSamples {
        func matrix(_ rows: [[Float]]) -> MLXArray {
            MLXArray(rows.flatMap { $0 }).reshaped(rows.count, rows[0].count)
        }
        return ARAModuleSamples(
            goodInput: matrix(goodInput), goodOutput: matrix(goodOutput),
            badInput: matrix(badInput), badOutput: matrix(badOutput))
    }
}

public struct ARAProjectionIOCollection: Sendable {
    public let modelPath: String
    public let pairNames: [String]
    public let layers: [ARAProjectionLayerCapture]
}

public struct ARAProjectionIOCollectionEngine: Sendable {
    public let modelDirectory: String
    public let pairs: [PromptPair]
    /// Zero-based layer indices.
    public let layers: [Int]

    public init(modelDirectory: String, pairs: [PromptPair], layers: [Int]) {
        self.modelDirectory = modelDirectory
        self.pairs = pairs
        self.layers = Array(Set(layers)).sorted()
    }

    public func run(
        progress: (@Sendable (_ completed: Int, _ total: Int) -> Void)? = nil
    ) async throws -> ARAProjectionIOCollection {
        guard !pairs.isEmpty else { throw ProbeError.emptyPromptFile }
        guard !layers.isEmpty, layers.allSatisfy({ $0 >= 0 }) else {
            throw ARAProjectionCollectionError.invalidLayers
        }
        let url = URL(
            fileURLWithPath: NSString(string: modelDirectory).expandingTildeInPath
        ).standardizedFileURL
        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(
                directory: url, extraEOSTokens: ["<end_of_turn>"]))
        let requested = Set(layers)
        var goodInputs = Dictionary(uniqueKeysWithValues: layers.map { ($0, [[Float]]()) })
        var goodOutputs = Dictionary(uniqueKeysWithValues: layers.map { ($0, [[Float]]()) })
        var badInputs = Dictionary(uniqueKeysWithValues: layers.map { ($0, [[Float]]()) })
        var badOutputs = Dictionary(uniqueKeysWithValues: layers.map { ($0, [[Float]]()) })

        for (index, pair) in pairs.enumerated() {
            let capture = try await container.perform { context in
                guard let good = try Gemma4Probe.attentionOutputProjectionIO(
                    context: context, prompt: pair.control, pairName: pair.name,
                    layers: requested),
                      let bad = try Gemma4Probe.attentionOutputProjectionIO(
                        context: context, prompt: pair.contrast, pairName: pair.name,
                        layers: requested)
                else {
                    throw ARAProjectionCollectionError.unsupportedModel(
                        String(describing: type(of: context.model)))
                }
                return PairProjectionCapture(good: good, bad: bad)
            }
            guard capture.good.map(\.layer) == layers,
                  capture.bad.map(\.layer) == layers
            else { throw ARAProjectionCollectionError.incompleteCapture(pair.name) }
            for vector in capture.good {
                goodInputs[vector.layer, default: []].append(vector.input)
                goodOutputs[vector.layer, default: []].append(vector.output)
            }
            for vector in capture.bad {
                badInputs[vector.layer, default: []].append(vector.input)
                badOutputs[vector.layer, default: []].append(vector.output)
            }
            progress?(index + 1, pairs.count)
        }

        return ARAProjectionIOCollection(
            modelPath: url.path,
            pairNames: pairs.map(\.name),
            layers: try layers.map { layer in
                guard let goodInput = goodInputs[layer], let goodOutput = goodOutputs[layer],
                      let badInput = badInputs[layer], let badOutput = badOutputs[layer],
                      goodInput.count == pairs.count, goodOutput.count == pairs.count,
                      badInput.count == pairs.count, badOutput.count == pairs.count
                else {
                    throw ARAProjectionCollectionError.incompleteCapture("layer \(layer)")
                }
                return ARAProjectionLayerCapture(
                    layer: layer, goodInput: goodInput, goodOutput: goodOutput,
                    badInput: badInput, badOutput: badOutput)
            })
    }
}

private struct PairProjectionCapture: Sendable {
    let good: [Gemma4Probe.ProjectionVector]
    let bad: [Gemma4Probe.ProjectionVector]
}

public enum ARAProjectionCollectionError: LocalizedError {
    case invalidLayers
    case unsupportedModel(String)
    case incompleteCapture(String)

    public var errorDescription: String? {
        switch self {
        case .invalidLayers:
            "ARA requires at least one non-negative, zero-based layer index."
        case .unsupportedModel(let type):
            "ARA projection capture currently requires Gemma 4; MLX loaded \(type)."
        case .incompleteCapture(let name):
            "ARA projection I/O capture was incomplete for \(name)."
        }
    }
}
