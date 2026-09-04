import Foundation
import MLX
import MLXFast
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon

enum Gemma3Probe {
    static func layerVectors(
        context: ModelContext,
        prompt: String,
        pairName: String,
        tokenPosition: ActivationTokenPosition = .postInstruction,
        maximumSequenceLength: Int = 512
    ) throws -> [[Float]] {
        guard let model = context.model as? Gemma3TextModel else {
            throw ProbeError.unsupportedModel(String(describing: type(of: context.model)))
        }

        let tokens = try context.tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": prompt]
        ])
        guard !tokens.isEmpty else {
            throw ProbeError.emptyPromptTokenization(name: pairName)
        }
        guard tokens.count <= maximumSequenceLength else {
            throw ProbeError.promptTooLong(name: pairName, tokenCount: tokens.count)
        }

        let tokenIndex: Int
        switch tokenPosition {
        case .postInstruction:
            tokenIndex = tokens.count - 1
        case .lastUser:
            let literalTokens = context.tokenizer.encode(
                text: prompt, addSpecialTokens: false)
            guard let match = Gemma4Probe.lastSubsequenceRange(
                haystack: tokens, needle: literalTokens)
            else {
                throw ActivationCollectionError.userTokenSpanNotFound(name: pairName)
            }
            tokenIndex = match.upperBound - 1
        case .firstResponse, .secondResponse:
            throw ActivationCollectionError.captureFailure(
                "Generated-response activation capture is currently implemented for Gemma 4 only.")
        }

        let input = MLXArray(tokens).expandedDimensions(axis: 0)
        var hidden = model.model.embedTokens(input)
        let scale = MLXArray(
            sqrt(Float(model.model.config.hiddenSize)), dtype: .bfloat16
        )
        hidden = hidden * scale.asType(hidden.dtype)

        var result = [[Float]]()
        result.reserveCapacity(model.model.layers.count)
        for layer in model.model.layers {
            // At <= 512 tokens, Gemma 3's local and global causal masks agree.
            hidden = layer(hidden, mask: .causal, cache: nil)
            let selectedToken = hidden[0, tokenIndex]
            eval(selectedToken)
            result.append(selectedToken.asArray(Float.self))
        }
        return result
    }
}
