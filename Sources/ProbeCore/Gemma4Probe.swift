import Foundation
import MLX
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon

public enum ActivationTokenPosition: String, Codable, Hashable, Sendable, CaseIterable {
    /// The final token in the rendered chat template, immediately before the
    /// model begins its answer. This is the conventional refusal-direction tap.
    case postInstruction = "post-instruction"
    /// The final token belonging to the user's literal request. Comparing this
    /// with `postInstruction` helps separate harmfulness recognition from the
    /// later decision to refuse.
    case lastUser = "last-user"
    /// Residual at the first greedily generated assistant token.
    case firstResponse = "first-response"
    /// Residual at the second greedily generated assistant token.
    case secondResponse = "second-response"
}

enum Gemma4Probe {
    struct TrajectoryTokenIndices: Equatable, Sendable {
        let postInstruction: Int
        let firstResponse: Int?
        let secondResponse: Int?
    }

    struct ProjectionVector: Sendable {
        let layer: Int
        let input: [Float]
        let output: [Float]
    }

    /// Resolves prompt/response trajectory positions without inspecting model
    /// state. `postInstruction` is anchored to the original prompt boundary,
    /// even after generated response tokens have been appended.
    static func trajectoryTokenIndices(
        promptTokenCount: Int,
        responseTokenCount: Int
    ) -> TrajectoryTokenIndices? {
        guard promptTokenCount > 0, responseTokenCount >= 0 else { return nil }
        return TrajectoryTokenIndices(
            postInstruction: promptTokenCount - 1,
            firstResponse: responseTokenCount >= 1 ? promptTokenCount : nil,
            secondResponse: responseTokenCount >= 2 ? promptTokenCount + 1 : nil)
    }

    /// Captures the actual module boundary used by ARA rather than a decoder
    /// residual. Layer numbers here are zero-based to match checkpoint keys.
    static func attentionOutputProjectionIO(
        context: ModelContext,
        prompt: String,
        pairName: String,
        layers requestedLayers: Set<Int>
    ) throws -> [ProjectionVector]? {
        let textModel: Gemma4TextModel
        if let model = context.model as? Gemma4TextModel {
            textModel = model
        } else if let model = context.model as? Gemma4Model {
            textModel = model.languageModel
        } else {
            return nil
        }
        let tokens = try context.tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": prompt]
        ])
        guard !tokens.isEmpty else {
            throw ProbeError.emptyPromptTokenization(name: pairName)
        }
        guard tokens.count <= 512 else {
            throw ProbeError.promptTooLong(name: pairName, tokenCount: tokens.count)
        }
        let input = MLXArray(tokens).expandedDimensions(axis: 0)
        let captures = textModel.model.attentionOutputProjectionIO(input)
        guard requestedLayers.allSatisfy(captures.indices.contains) else {
            throw ActivationCollectionError.captureFailure(
                "Requested ARA layer is outside the model's decoder range.")
        }
        return requestedLayers.sorted().map { layer in
            let capture = captures[layer]
            return ProjectionVector(
                layer: layer,
                input: capture.input.asArray(Float.self),
                output: capture.output.asArray(Float.self))
        }
    }

    static func layerVectors(
        context: ModelContext,
        prompt: String,
        pairName: String,
        tokenPosition: ActivationTokenPosition = .postInstruction,
        maximumSequenceLength: Int = 512
    ) throws -> [[Float]]? {
        try layerVectors(
            context: context, prompt: prompt, pairName: pairName,
            tokenPositions: [tokenPosition],
            maximumSequenceLength: maximumSequenceLength)?[tokenPosition]
    }

    /// Captures several semantically meaningful token positions in one model
    /// forward pass. This matters on Apple Silicon where activation collection,
    /// rather than the small linear probes, dominates experiment time.
    static func layerVectors(
        context: ModelContext,
        prompt: String,
        pairName: String,
        tokenPositions: Set<ActivationTokenPosition>,
        maximumSequenceLength: Int = 512
    ) throws -> [ActivationTokenPosition: [[Float]]]? {
        let textModel: Gemma4TextModel
        if let model = context.model as? Gemma4TextModel {
            textModel = model
        } else if let model = context.model as? Gemma4Model {
            textModel = model.languageModel
        } else {
            return nil
        }

        var tokens = try context.tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": prompt]
        ])
        guard !tokens.isEmpty else {
            throw ProbeError.emptyPromptTokenization(name: pairName)
        }
        guard tokens.count <= maximumSequenceLength else {
            throw ProbeError.promptTooLong(name: pairName, tokenCount: tokens.count)
        }

        let requestedResponseTokens = tokenPositions.contains(.secondResponse) ? 2
            : (tokenPositions.contains(.firstResponse) ? 1 : 0)
        let promptTokenCount = tokens.count
        if requestedResponseTokens > 0 {
            for _ in 0 ..< requestedResponseTokens {
                let input = MLXArray(tokens).expandedDimensions(axis: 0)
                let logits = context.model(input, cache: nil)[0, -1]
                let nextToken = argMax(logits, axis: -1).item(Int.self)
                tokens.append(nextToken)
            }
        }

        guard let trajectoryIndices = trajectoryTokenIndices(
            promptTokenCount: promptTokenCount,
            responseTokenCount: tokens.count - promptTokenCount)
        else {
            throw ActivationCollectionError.captureFailure(
                "Could not resolve prompt and response trajectory token indices.")
        }

        var tokenIndices = [ActivationTokenPosition: Int]()
        for tokenPosition in tokenPositions {
            switch tokenPosition {
            case .postInstruction:
                tokenIndices[tokenPosition] = trajectoryIndices.postInstruction
            case .lastUser:
                let literalTokens = context.tokenizer.encode(
                    text: prompt, addSpecialTokens: false)
                guard let match = lastSubsequenceRange(
                    haystack: tokens, needle: literalTokens)
                else {
                    throw ActivationCollectionError.userTokenSpanNotFound(name: pairName)
                }
                tokenIndices[tokenPosition] = match.upperBound - 1
            case .firstResponse:
                guard let tokenIndex = trajectoryIndices.firstResponse else {
                    throw ActivationCollectionError.captureFailure(
                        "First-response activation requested before a response token was generated.")
                }
                tokenIndices[tokenPosition] = tokenIndex
            case .secondResponse:
                guard let tokenIndex = trajectoryIndices.secondResponse else {
                    throw ActivationCollectionError.captureFailure(
                        "Second-response activation requested before two response tokens were generated.")
                }
                tokenIndices[tokenPosition] = tokenIndex
            }
        }

        let input = MLXArray(tokens).expandedDimensions(axis: 0)
        var result = Dictionary(
            uniqueKeysWithValues: tokenPositions.map { ($0, [[Float]]()) })
        for hidden in textModel.model.layerHiddenStates(input) {
            for (position, tokenIndex) in tokenIndices {
                let selectedToken = hidden[0, tokenIndex].asType(.float32)
                eval(selectedToken)
                result[position, default: []].append(selectedToken.asArray(Float.self))
            }
        }
        return result
    }

    static func lastSubsequenceRange(
        haystack: [Int], needle: [Int]
    ) -> Range<Int>? {
        guard !needle.isEmpty, needle.count <= haystack.count else { return nil }
        for start in stride(
            from: haystack.count - needle.count, through: 0, by: -1
        ) where Array(haystack[start ..< start + needle.count]) == needle {
            return start ..< start + needle.count
        }
        return nil
    }
}

public enum ActivationCollectionError: LocalizedError {
    case userTokenSpanNotFound(name: String)
    case captureFailure(String)

    public var errorDescription: String? {
        switch self {
        case .userTokenSpanNotFound(let name):
            "Could not locate the literal user-token span for prompt pair '\(name)' in the rendered chat template."
        case .captureFailure(let message):
            message
        }
    }
}
