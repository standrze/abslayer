import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

public enum BehaviorEvaluationEngine {
    public struct TimedResponse: Equatable, Sendable {
        public let response: String
        public let latencyMilliseconds: Int64

        public init(response: String, latencyMilliseconds: Int64) {
            self.response = response
            self.latencyMilliseconds = latencyMilliseconds
        }
    }

    public static func request(
        modelDirectory: String,
        prompt: String,
        adapterDirectory: String? = nil,
        adapterScaleOverride: Float? = nil,
        systemPrompt: String? = nil,
        maximumTokens: Int = EvaluationGenerationOptions.defaultMaximumTokens
    ) async throws -> String {
        let responses = try await requestBatch(
            modelDirectory: modelDirectory, prompts: [prompt],
            adapterDirectory: adapterDirectory,
            adapterScaleOverride: adapterScaleOverride,
            systemPrompt: systemPrompt, maximumTokens: maximumTokens)
        return responses[0].response
    }

    /// Loads the model and optional adapter once, then evaluates every prompt
    /// sequentially with the same deterministic generation parameters.
    public static func requestBatch(
        modelDirectory: String,
        prompts: [String],
        adapterDirectory: String? = nil,
        adapterScaleOverride: Float? = nil,
        systemPrompt: String? = nil,
        maximumTokens: Int = EvaluationGenerationOptions.defaultMaximumTokens
    ) async throws -> [TimedResponse] {
        guard !prompts.isEmpty else { return [] }
        let url = URL(fileURLWithPath: modelDirectory).standardizedFileURL
        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(directory: url, extraEOSTokens: ["<end_of_turn>"]))
        if let adapterDirectory {
            let adapter = try LoRAAdapterLoader.load(
                directory: adapterDirectory, scaleOverride: adapterScaleOverride)
            try await container.perform { context in
                try adapter.load(into: context.model)
            }
        }
        let parameters = GenerateParameters(maxTokens: maximumTokens, temperature: 0)
        var results = [TimedResponse]()
        results.reserveCapacity(prompts.count)
        for prompt in prompts {
            let started = ContinuousClock.now
            let generated = try await response(
                container: container, parameters: parameters,
                prompt: prompt, systemPrompt: systemPrompt)
            results.append(TimedResponse(
                response: generated,
                latencyMilliseconds: elapsedMilliseconds(since: started)))
        }
        return results
    }

    public static func run(
        modelDirectory: String,
        pairs: [PromptPair],
        maximumCases: Int,
        adapterDirectory: String? = nil,
        adapterScaleOverride: Float? = nil,
        systemPrompt: String? = nil,
        maximumTokens: Int = EvaluationGenerationOptions.defaultMaximumTokens
    ) async throws -> [PromptResult] {
        let url = URL(fileURLWithPath: modelDirectory).standardizedFileURL
        let loadStarted = ContinuousClock.now
        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(directory: url, extraEOSTokens: ["<end_of_turn>"]))
        let loadTime = loadStarted.duration(to: .now)
        print("model loaded in \(loadTime.formatted(.units(allowed: [.seconds, .milliseconds], width: .abbreviated)))")
        if let adapterDirectory {
            let adapter = try LoRAAdapterLoader.load(
                directory: adapterDirectory, scaleOverride: adapterScaleOverride)
            try await container.perform { context in
                try adapter.load(into: context.model)
            }
            let suffix = adapterScaleOverride.map { " at scale \($0)" } ?? ""
            print("loaded adapter from \(adapterDirectory)\(suffix)")
        }
        let selected = evenlySpaced(pairs, maximum: maximumCases)
        var results = [PromptResult]()
        results.reserveCapacity(selected.count)
        let parameters = GenerateParameters(maxTokens: maximumTokens, temperature: 0)
        for (index, pair) in selected.enumerated() {
            let contrast = try await response(
                container: container, parameters: parameters,
                prompt: pair.contrast, systemPrompt: systemPrompt)
            let control = try await response(
                container: container, parameters: parameters,
                prompt: pair.control, systemPrompt: systemPrompt)
            results.append(PromptResult(
                name: pair.name, contrastResponse: contrast, controlResponse: control,
                category: pair.category, systemPrompt: systemPrompt,
                contrastPrompt: pair.contrast, controlPrompt: pair.control))
            print("evaluated \(index + 1)/\(selected.count)")
        }
        return results
    }

    private static func response(
        container: ModelContainer, parameters: GenerateParameters,
        prompt: String, systemPrompt: String?
    ) async throws -> String {
        let session = ChatSession(container, generateParameters: parameters)
        if let systemPrompt {
            return try await session.respond(to: [
                .system(systemPrompt), .user(prompt),
            ])
        }
        return try await session.respond(to: prompt)
    }

    private static func elapsedMilliseconds(since started: ContinuousClock.Instant) -> Int64 {
        let components = started.duration(to: .now).components
        return components.seconds * 1_000 + components.attoseconds / 1_000_000_000_000_000
    }

    private static func evenlySpaced(_ pairs: [PromptPair], maximum: Int) -> [PromptPair] {
        guard maximum > 0, pairs.count > maximum else { return pairs }
        return (0 ..< maximum).map { index in
            pairs[index * pairs.count / maximum]
        }
    }
}
