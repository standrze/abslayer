import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

/// Runs exact, teacher-forced, same-prompt residual replacement on a freshly
/// loaded Gemma 4 BF16 model. No adapter or persistent intervention is loaded.
public enum MatchedResponseStatePatchingEngine {
    public static let maximumTeacherForcedPrefixTokens = 512

    public static func run(
        modelDirectory: String,
        document: MatchedResponsePatchDocument,
        configuration: MatchedResponsePatchConfiguration,
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> MatchedResponsePatchStudy {
        try document.validate()
        let inspection = try ModelFolderValidator.validateFullBF16(
            path: modelDirectory)
        if let decoderLayerCount = inspection.decoderLayerCount {
            try configuration.validate(
                caseCount: document.cases.count,
                decoderLayerCount: decoderLayerCount)
        }

        let started = ContinuousClock.now
        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(
                directory: URL(fileURLWithPath: inspection.path),
                extraEOSTokens: ["<end_of_turn>"]))
        progress?(
            "fresh untouched BF16 model loaded in "
                + started.duration(to: .now).formatted(
                    .units(
                        allowed: [.seconds, .milliseconds],
                        width: .abbreviated)))

        let selectedCases = evenlySpaced(
            document.cases, maximum: configuration.maximumCases)
        let study = try await container.perform {
            (context: ModelContext) throws -> MatchedResponsePatchStudy in
            guard let backbone = gemma4Backbone(context.model) else {
                throw MatchedResponsePatchError.unsupportedModel
            }
            try configuration.validate(
                caseCount: selectedCases.count,
                decoderLayerCount: backbone.layers.count)
            // A fresh load should already be clear. Explicitly setting and
            // restoring nil makes the untouched baseline condition auditable.
            backbone.residualIntervention = nil
            defer { backbone.residualIntervention = nil }

            var prepared = [PreparedCase]()
            prepared.reserveCapacity(selectedCases.count)
            for (index, item) in selectedCases.enumerated() {
                prepared.append(try prepare(
                    item, context: context, backbone: backbone,
                    configuration: configuration))
                progress?(
                    "captured untouched donor/recipient states "
                        + "\(index + 1)/\(selectedCases.count) [\(item.name)]")
            }

            let sites = configuration.layersZeroBased.flatMap { layer in
                configuration.responseTokenPositionsZeroBased.map {
                    MatchedResponsePatchSite(
                        layerZeroBased: layer,
                        responseTokenPositionZeroBased: $0)
                }
            }
            let total = prepared.count * sites.count
            var completed = 0
            var results = [MatchedResponsePatchResult]()
            results.reserveCapacity(total)
            for caseIndex in prepared.indices {
                let item = prepared[caseIndex]
                for site in sites {
                    results.append(try patchResult(
                        caseIndex: caseIndex, prepared: prepared,
                        site: site, context: context, backbone: backbone,
                        configuration: configuration))
                    completed += 1
                    progress?(
                        "exact matched/reverse/control patches \(completed)/\(total) "
                            + "[\(item.source.name) L0="
                            + "\(site.layerZeroBased) t="
                            + "\(site.responseTokenPositionZeroBased)]")
                }
            }
            return MatchedResponsePatchStudy(
                modelPath: inspection.path,
                inputSplit: document.split,
                inputModelCondition: document.modelCondition,
                inputProvenance: document.provenance,
                configuration: configuration,
                cases: selectedCases,
                results: results)
        }
        return study
    }

    private struct PreparedCase {
        let source: MatchedResponsePatchCase
        let promptTokens: [Int]
        let donorContinuationTokens: [Int]
        let recipientContinuationTokens: [Int]
        let donorStates: [MatchedResponsePatchSite: [Float]]
        let recipientStates: [MatchedResponsePatchSite: [Float]]
        let donorBaselines: [Int: TeacherForcedTrajectory]
        let recipientBaselines: [Int: TeacherForcedTrajectory]
    }

    private struct TeacherForcedTrajectory {
        let immediateLogProbabilities: [Float]
        let continuationTargetLogProbabilities: [Double]
        let nextTokenID: Int
    }

    private struct TokenizedContinuation {
        let promptTokens: [Int]
        let continuationTokens: [Int]
    }

    private static func prepare(
        _ item: MatchedResponsePatchCase,
        context: ModelContext,
        backbone: Gemma4TextModelInner,
        configuration: MatchedResponsePatchConfiguration
    ) throws -> PreparedCase {
        let donor = try tokenize(
            prompt: item.prompt, continuation: item.donorContinuation,
            caseName: item.name, context: context)
        let recipient = try tokenize(
            prompt: item.prompt, continuation: item.recipientContinuation,
            caseName: item.name, context: context)
        guard donor.promptTokens == recipient.promptTokens else {
            throw MatchedResponsePatchError.promptTokenizationMismatch(
                item.name)
        }
        let maximumPosition =
            configuration.responseTokenPositionsZeroBased.max() ?? 0
        guard donor.continuationTokens.count > maximumPosition + 1,
              recipient.continuationTokens.count > maximumPosition + 1
        else {
            throw MatchedResponsePatchError.responsePositionUnavailable(
                caseName: item.name, position: maximumPosition,
                donorTokens: donor.continuationTokens.count,
                recipientTokens: recipient.continuationTokens.count)
        }

        var donorStates = [MatchedResponsePatchSite: [Float]]()
        var recipientStates = [MatchedResponsePatchSite: [Float]]()
        var donorBaselines = [Int: TeacherForcedTrajectory]()
        var recipientBaselines = [Int: TeacherForcedTrajectory]()
        for position in configuration.responseTokenPositionsZeroBased {
            let donorPrefix = donor.promptTokens
                + Array(donor.continuationTokens.prefix(position + 1))
            let recipientPrefix = recipient.promptTokens
                + Array(recipient.continuationTokens.prefix(position + 1))
            try validatePrefix(
                donorPrefix, caseName: item.name)
            try validatePrefix(
                recipientPrefix, caseName: item.name)

            let donorCaptured = try captureStates(
                prefixTokens: donorPrefix,
                layers: configuration.layersZeroBased,
                backbone: backbone)
            let recipientCaptured = try captureStates(
                prefixTokens: recipientPrefix,
                layers: configuration.layersZeroBased,
                backbone: backbone)
            for layer in configuration.layersZeroBased {
                let site = MatchedResponsePatchSite(
                    layerZeroBased: layer,
                    responseTokenPositionZeroBased: position)
                donorStates[site] = donorCaptured[layer]
                recipientStates[site] = recipientCaptured[layer]
            }

            donorBaselines[position] = try trajectory(
                promptTokens: donor.promptTokens,
                continuationTokens: donor.continuationTokens,
                responsePosition: position,
                maximumSuffixTokens: configuration.maximumSuffixTokens,
                replacement: nil,
                context: context, backbone: backbone)
            recipientBaselines[position] = try trajectory(
                promptTokens: recipient.promptTokens,
                continuationTokens: recipient.continuationTokens,
                responsePosition: position,
                maximumSuffixTokens: configuration.maximumSuffixTokens,
                replacement: nil,
                context: context, backbone: backbone)
        }
        return PreparedCase(
            source: item,
            promptTokens: donor.promptTokens,
            donorContinuationTokens: donor.continuationTokens,
            recipientContinuationTokens: recipient.continuationTokens,
            donorStates: donorStates,
            recipientStates: recipientStates,
            donorBaselines: donorBaselines,
            recipientBaselines: recipientBaselines)
    }

    private static func patchResult(
        caseIndex: Int,
        prepared: [PreparedCase],
        site: MatchedResponsePatchSite,
        context: ModelContext,
        backbone: Gemma4TextModelInner,
        configuration: MatchedResponsePatchConfiguration
    ) throws -> MatchedResponsePatchResult {
        let item = prepared[caseIndex]
        guard let donorState = item.donorStates[site],
              let recipientState = item.recipientStates[site],
              let donorBaseline = item.donorBaselines[
                site.responseTokenPositionZeroBased],
              let recipientBaseline = item.recipientBaselines[
                site.responseTokenPositionZeroBased]
        else { throw MatchedResponsePatchError.incompatibleStates }

        let forward = try trajectory(
            promptTokens: item.promptTokens,
            continuationTokens: item.recipientContinuationTokens,
            responsePosition: site.responseTokenPositionZeroBased,
            maximumSuffixTokens: configuration.maximumSuffixTokens,
            replacement: (site.layerZeroBased, donorState),
            context: context, backbone: backbone)
        let reverse = try trajectory(
            promptTokens: item.promptTokens,
            continuationTokens: item.donorContinuationTokens,
            responsePosition: site.responseTokenPositionZeroBased,
            maximumSuffixTokens: configuration.maximumSuffixTokens,
            replacement: (site.layerZeroBased, recipientState),
            context: context, backbone: backbone)
        let forwardEffect = try MatchedResponsePatchMath.effect(
            sourceBaselineLogProbabilities:
                donorBaseline.immediateLogProbabilities,
            targetBaselineLogProbabilities:
                recipientBaseline.immediateLogProbabilities,
            patchedLogProbabilities: forward.immediateLogProbabilities,
            sourceNextTokenID: donorBaseline.nextTokenID,
            targetNextTokenID: recipientBaseline.nextTokenID,
            targetBaselineContinuationLogProbabilities:
                recipientBaseline.continuationTargetLogProbabilities,
            patchedTargetContinuationLogProbabilities:
                forward.continuationTargetLogProbabilities)
        let reverseEffect = try MatchedResponsePatchMath.effect(
            sourceBaselineLogProbabilities:
                recipientBaseline.immediateLogProbabilities,
            targetBaselineLogProbabilities:
                donorBaseline.immediateLogProbabilities,
            patchedLogProbabilities: reverse.immediateLogProbabilities,
            sourceNextTokenID: recipientBaseline.nextTokenID,
            targetNextTokenID: donorBaseline.nextTokenID,
            targetBaselineContinuationLogProbabilities:
                donorBaseline.continuationTargetLogProbabilities,
            patchedTargetContinuationLogProbabilities:
                reverse.continuationTargetLogProbabilities)

        var randomEffect: MatchedResponsePatchEffect?
        var randomName: String?
        if let randomIndex = MatchedResponsePatchMath.randomControlIndex(
            caseIndex: caseIndex, caseCount: prepared.count,
            seed: configuration.randomControlSeed),
           let unrelated = prepared[randomIndex].donorStates[site]
        {
            randomName = prepared[randomIndex].source.name
            if let replacement = try? MatchedResponsePatchMath
                .normMatchedControlReplacement(
                    recipient: recipientState,
                    unrelatedDonor: unrelated,
                    matchedDonor: donorState)
            {
                let control = try trajectory(
                    promptTokens: item.promptTokens,
                    continuationTokens: item.recipientContinuationTokens,
                    responsePosition: site.responseTokenPositionZeroBased,
                    maximumSuffixTokens: configuration.maximumSuffixTokens,
                    replacement: (site.layerZeroBased, replacement),
                    context: context, backbone: backbone)
                randomEffect = try MatchedResponsePatchMath.effect(
                    sourceBaselineLogProbabilities:
                        donorBaseline.immediateLogProbabilities,
                    targetBaselineLogProbabilities:
                        recipientBaseline.immediateLogProbabilities,
                    patchedLogProbabilities:
                        control.immediateLogProbabilities,
                    sourceNextTokenID: donorBaseline.nextTokenID,
                    targetNextTokenID: recipientBaseline.nextTokenID,
                    targetBaselineContinuationLogProbabilities:
                        recipientBaseline.continuationTargetLogProbabilities,
                    patchedTargetContinuationLogProbabilities:
                        control.continuationTargetLogProbabilities)
            }
        }

        return MatchedResponsePatchResult(
            caseName: item.source.name,
            category: item.source.category,
            site: site,
            donorNextTokenID: donorBaseline.nextTokenID,
            recipientNextTokenID: recipientBaseline.nextTokenID,
            donorPatchedTokenID: item.donorContinuationTokens[
                site.responseTokenPositionZeroBased],
            recipientPatchedTokenID: item.recipientContinuationTokens[
                site.responseTokenPositionZeroBased],
            responsePrefixesMatchThroughPatchedToken:
                item.donorContinuationTokens.prefix(
                    site.responseTokenPositionZeroBased + 1)
                == item.recipientContinuationTokens.prefix(
                    site.responseTokenPositionZeroBased + 1),
            donorRecipientStateL2Distance:
                MatchedResponsePatchMath.l2Distance(
                    donorState, recipientState),
            donorRecipientStateCosineSimilarity:
                LayerMath.cosineSimilarity(donorState, recipientState),
            donorIntoRecipient: forwardEffect,
            recipientIntoDonor: reverseEffect,
            normMatchedRandomControlIntoRecipient: randomEffect,
            randomControlCaseName: randomName)
    }

    private static func tokenize(
        prompt: String, continuation: String, caseName: String,
        context: ModelContext
    ) throws -> TokenizedContinuation {
        let promptTokens = try context.tokenizer.applyChatTemplate(messages: [
            ["role": "user", "content": prompt]
        ])
        let renderedPrompt = context.tokenizer.decode(
            tokenIds: promptTokens, skipSpecialTokens: false)
        let promptPlusContinuation = context.tokenizer.encode(
            text: renderedPrompt + continuation,
            addSpecialTokens: false)
        var templatedConversation: [Int]?
        if !promptPlusContinuation.starts(with: promptTokens) {
            templatedConversation = try context.tokenizer.applyChatTemplate(
                messages: [
                    ["role": "user", "content": prompt],
                    ["role": "assistant", "content": continuation],
                ],
                tools: nil,
                additionalContext: ["add_generation_prompt": false])
        }
        do {
            return TokenizedContinuation(
                promptTokens: promptTokens,
                continuationTokens: try
                    TeacherForcedContinuationTokenDerivation.continuation(
                        promptTokens: promptTokens,
                        promptPlusReferenceTokens: promptPlusContinuation,
                        templatedConversationTokens: templatedConversation))
        } catch {
            throw MatchedResponsePatchError.promptTokenizationMismatch(
                caseName)
        }
    }

    private static func captureStates(
        prefixTokens: [Int], layers: [Int],
        backbone: Gemma4TextModelInner
    ) throws -> [Int: [Float]] {
        backbone.residualIntervention = nil
        let input = MLXArray(prefixTokens).expandedDimensions(axis: 0)
        let hiddenStates = backbone.layerHiddenStates(input)
        guard layers.allSatisfy(hiddenStates.indices.contains) else {
            throw MatchedResponsePatchError.layerOutsideDecoder(
                layers: layers, decoderLayerCount: hiddenStates.count)
        }
        var result = [Int: [Float]]()
        for layer in layers {
            let selected = hiddenStates[layer][0, -1].asType(.float32)
            eval(selected)
            result[layer] = selected.asArray(Float.self)
        }
        return result
    }

    /// Runs a real cached prefill followed by a bounded teacher-forced decode.
    /// A replacement is applied exactly once, at the selected layer and final
    /// response token of the prefill. It is not re-applied to decode tokens.
    private static func trajectory(
        promptTokens: [Int], continuationTokens: [Int],
        responsePosition: Int, maximumSuffixTokens: Int,
        replacement: (layer: Int, state: [Float])?,
        context: ModelContext, backbone: Gemma4TextModelInner
    ) throws -> TeacherForcedTrajectory {
        guard continuationTokens.count > responsePosition + 1 else {
            throw MatchedResponsePatchError.incompatibleMetrics
        }
        let prefix = promptTokens
            + Array(continuationTokens.prefix(responsePosition + 1))
        try validatePrefix(prefix, caseName: "teacher-forced-trajectory")
        let prefillTokenCount = prefix.count
        if let replacement {
            backbone.residualIntervention = replacementTransform(
                layer: replacement.layer,
                prefillTokenCount: prefillTokenCount,
                state: replacement.state)
        } else {
            backbone.residualIntervention = nil
        }
        defer { backbone.residualIntervention = nil }

        let cache = try context.model.newCache(parameters: nil)
        var logits = context.model(
            MLXArray(prefix).expandedDimensions(axis: 0), cache: cache)[0, -1]
        var logs = MLXNN.logSoftmax(
            logits.asType(.float32), axis: -1)
        eval(logs)
        let immediate = logs.asArray(Float.self)

        let end = min(
            continuationTokens.count,
            responsePosition + 1 + maximumSuffixTokens)
        var targetLogs = [Double]()
        targetLogs.reserveCapacity(end - responsePosition - 1)
        for targetPosition in (responsePosition + 1) ..< end {
            let targetToken = continuationTokens[targetPosition]
            let targetLog = logs[targetToken]
            eval(targetLog)
            targetLogs.append(Double(targetLog.item(Float.self)))
            if targetPosition + 1 < end {
                logits = context.model(
                    MLXArray([targetToken]).expandedDimensions(axis: 0),
                    cache: cache)[0, -1]
                logs = MLXNN.logSoftmax(
                    logits.asType(.float32), axis: -1)
            }
        }
        return TeacherForcedTrajectory(
            immediateLogProbabilities: immediate,
            continuationTargetLogProbabilities: targetLogs,
            nextTokenID: continuationTokens[responsePosition + 1])
    }

    private static func replacementTransform(
        layer: Int, prefillTokenCount: Int, state replacement: [Float]
    ) -> @Sendable (_ layer: Int, _ state: MLXArray) -> MLXArray {
        return { currentLayer, current in
            guard currentLayer == layer,
                  current.dim(-2) == prefillTokenCount,
                  current.dim(-1) == replacement.count
            else { return current }
            let replacementArray = MLXArray(replacement)
                .asType(current.dtype)
            let selected = current[0, prefillTokenCount - 1]
            let delta = (replacementArray - selected)
                .reshaped(1, 1, replacement.count)
            var mask = Array(repeating: Float.zero, count: prefillTokenCount)
            mask[prefillTokenCount - 1] = 1
            let selector = MLXArray(mask)
                .reshaped(1, prefillTokenCount, 1)
                .asType(current.dtype)
            return current + selector * delta
        }
    }

    private static func validatePrefix(
        _ tokens: [Int], caseName: String
    ) throws {
        guard !tokens.isEmpty,
              tokens.count <= maximumTeacherForcedPrefixTokens
        else {
            throw MatchedResponsePatchError.promptTooLong(
                caseName: caseName, tokenCount: tokens.count,
                maximum: maximumTeacherForcedPrefixTokens)
        }
    }

    private static func gemma4Backbone(
        _ model: LanguageModel
    ) -> Gemma4TextModelInner? {
        if let model = model as? Gemma4TextModel { return model.model }
        if let model = model as? Gemma4Model {
            return model.languageModel.model
        }
        return nil
    }

    private static func evenlySpaced<T>(
        _ values: [T], maximum: Int
    ) -> [T] {
        guard values.count > maximum else { return values }
        return (0 ..< maximum).map {
            values[$0 * values.count / maximum]
        }
    }
}
