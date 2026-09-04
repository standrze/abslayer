import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public struct ActivationPairMetadata: Codable, Equatable, Sendable {
    public let name: String
    public let category: String?
    public let split: String?

    public init(name: String, category: String?, split: String?) {
        self.name = name
        self.category = category
        self.split = split
    }
}

/// Dense paired residuals for one decoder layer. Rows use the same ordering as
/// `ActivationCollection.pairs`.
public struct LayerActivationSet: Sendable {
    public let layer: Int
    public let contrast: [[Float]]
    public let control: [[Float]]

    public init(layer: Int, contrast: [[Float]], control: [[Float]]) {
        self.layer = layer
        self.contrast = contrast
        self.control = control
    }
}

public struct ActivationCollection: Sendable {
    public let model: String
    public let tokenPosition: ActivationTokenPosition
    public let pairs: [ActivationPairMetadata]
    public let layers: [LayerActivationSet]

    public var hiddenWidth: Int {
        layers.first?.contrast.first?.count ?? 0
    }

    public init(
        model: String,
        tokenPosition: ActivationTokenPosition,
        pairs: [ActivationPairMetadata],
        layers: [LayerActivationSet]
    ) {
        self.model = model
        self.tokenPosition = tokenPosition
        self.pairs = pairs
        self.layers = layers
    }
}

/// Collects harmful/control residuals at both the final literal user token and
/// the post-template decision token. Gemma 4 captures both positions in one
/// forward pass per prompt so the diagnostic does not double model time.
public struct ActivationCollectionEngine: Sendable {
    public enum Source: Sendable {
        case hub(String)
        case directory(String)
    }

    public let source: Source
    public let pairs: [PromptPair]

    public init(modelID: String, pairs: [PromptPair]) {
        source = .hub(modelID)
        self.pairs = pairs
    }

    public init(modelDirectory: String, pairs: [PromptPair]) {
        source = .directory(modelDirectory)
        self.pairs = pairs
    }

    public func run(
        positions: Set<ActivationTokenPosition> = [.lastUser, .postInstruction],
        progress: (@Sendable (_ completed: Int, _ total: Int) -> Void)? = nil
    ) async throws -> [ActivationTokenPosition: ActivationCollection] {
        guard !pairs.isEmpty else { throw ProbeError.emptyPromptFile }
        guard !positions.isEmpty else { return [:] }

        let configuration: ModelConfiguration
        let modelName: String
        switch source {
        case .hub(let modelID):
            configuration = ModelConfiguration(
                id: modelID, extraEOSTokens: ["<end_of_turn>"])
            modelName = modelID
        case .directory(let path):
            let url = URL(
                fileURLWithPath: NSString(string: path).expandingTildeInPath
            ).standardizedFileURL
            configuration = ModelConfiguration(
                directory: url, extraEOSTokens: ["<end_of_turn>"])
            modelName = url.path
        }

        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: configuration)
        return try await collectActivationCollections(
            container: container, modelName: modelName, pairs: pairs,
            positions: positions, progress: progress)
    }
}

func collectActivationCollections(
    container: ModelContainer,
    modelName: String,
    pairs: [PromptPair],
    positions: Set<ActivationTokenPosition>,
    progress: (@Sendable (_ completed: Int, _ total: Int) -> Void)? = nil
) async throws -> [ActivationTokenPosition: ActivationCollection] {
    guard !pairs.isEmpty else { throw ProbeError.emptyPromptFile }
    guard !positions.isEmpty else { return [:] }

    var contrastByPosition = [ActivationTokenPosition: [[[Float]]]]()
    var controlByPosition = [ActivationTokenPosition: [[[Float]]]]()

    for (pairIndex, pair) in pairs.enumerated() {
        let captured = try await container.perform { context in
            if let contrast = try Gemma4Probe.layerVectors(
                context: context, prompt: pair.contrast, pairName: pair.name,
                tokenPositions: positions),
               let control = try Gemma4Probe.layerVectors(
                context: context, prompt: pair.control, pairName: pair.name,
                tokenPositions: positions)
            {
                return PairedCapture(contrast: contrast, control: control)
            }

            var contrast = [ActivationTokenPosition: [[Float]]]()
            var control = [ActivationTokenPosition: [[Float]]]()
            for position in positions {
                contrast[position] = try Gemma3Probe.layerVectors(
                    context: context, prompt: pair.contrast, pairName: pair.name,
                    tokenPosition: position)
                control[position] = try Gemma3Probe.layerVectors(
                    context: context, prompt: pair.control, pairName: pair.name,
                    tokenPosition: position)
            }
            return PairedCapture(contrast: contrast, control: control)
        }

        for position in positions {
            guard let contrastRows = captured.contrast[position],
                  let controlRows = captured.control[position],
                  contrastRows.count == controlRows.count
            else {
                throw ActivationCollectionError.incompleteCapture(
                    name: pair.name, position: position)
            }

            var contrastLayers = contrastByPosition[position]
                ?? Array(repeating: [], count: contrastRows.count)
            var controlLayers = controlByPosition[position]
                ?? Array(repeating: [], count: controlRows.count)
            guard contrastLayers.count == contrastRows.count,
                  controlLayers.count == controlRows.count
            else {
                throw ActivationCollectionError.layerCountChanged(name: pair.name)
            }
            for layer in contrastRows.indices {
                contrastLayers[layer].append(contrastRows[layer])
                controlLayers[layer].append(controlRows[layer])
            }
            contrastByPosition[position] = contrastLayers
            controlByPosition[position] = controlLayers
        }
        progress?(pairIndex + 1, pairs.count)
    }

    let metadata = pairs.map {
        ActivationPairMetadata(name: $0.name, category: $0.category, split: $0.split)
    }
    return try Dictionary(uniqueKeysWithValues: positions.map { position in
        guard let contrast = contrastByPosition[position],
              let control = controlByPosition[position]
        else {
            throw ActivationCollectionError.incompleteCapture(
                name: "collection", position: position)
        }
        let layers = zip(contrast, control).enumerated().map { index, rows in
            LayerActivationSet(
                layer: index + 1, contrast: rows.0, control: rows.1)
        }
        return (position, ActivationCollection(
            model: modelName, tokenPosition: position,
            pairs: metadata, layers: layers))
    })
}

struct PairedCapture: Sendable {
    let contrast: [ActivationTokenPosition: [[Float]]]
    let control: [ActivationTokenPosition: [[Float]]]
}

extension ActivationCollectionError {
    static func incompleteCapture(
        name: String, position: ActivationTokenPosition
    ) -> Self {
        .captureFailure(
            "Activation capture for '\(name)' is missing \(position.rawValue) residuals.")
    }

    static func layerCountChanged(name: String) -> Self {
        .captureFailure(
            "The decoder layer count changed while collecting prompt pair '\(name)'.")
    }
}
