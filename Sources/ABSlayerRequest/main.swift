#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

private let responseMarker = "ABSLAYER_RESPONSE_JSON\t"
private let batchResponseMarker = "ABSLAYER_BATCH_RESPONSE_JSON\t"

private enum RequestRuntimeError: LocalizedError {
    case unableToDisableCUDAGraphs(Int32)

    var errorDescription: String? {
        switch self {
        case .unableToDisableCUDAGraphs(let code):
            "Could not disable MLX CUDA graphs for deterministic inference (errno \(code))."
        }
    }
}

private struct ResponseEnvelope: Encodable {
    let schemaVersion = 1
    let response: String
}

@main
enum ABSlayerRequest {
    static func main() async throws {
        // MLX CUDA graph replay is not numerically safe for this Gemma 4 LoRA
        // cached-decoding path in the pinned Linux/CUDA runtime. It can
        // nondeterministically corrupt generation, including repeated token 0
        // (<pad>). Configure this before the first MLX/model operation, and
        // override inherited process state.
        guard setenv("MLX_USE_CUDA_GRAPHS", "0", 1) == 0 else {
            throw RequestRuntimeError.unableToDisableCUDAGraphs(errno)
        }
        let args = CommandLine.arguments
        if args.count == 3, args[1] == "--batch" {
            try await runBatch(modelPath: args[2])
            return
        }
        guard args.count == 3 else {
            writeUsageAndExit()
        }
        try await runSingle(modelPath: args[1], prompt: args[2])
    }

    private static func runSingle(modelPath: String, prompt: String) async throws {
        let model = try ModelFolderValidator.validateFullBF16(path: modelPath)
        let environment = ProcessInfo.processInfo.environment
        let adapter = try AdapterRuntimeOptions.parse(environment: environment)
        let options = try EvaluationGenerationOptions.parse(environment: environment)
        let response = try await BehaviorEvaluationEngine.request(
            modelDirectory: model.path, prompt: prompt,
            adapterDirectory: adapter.directory,
            adapterScaleOverride: adapter.scaleOverride,
            systemPrompt: environment["ABSLAYER_SYSTEM_PROMPT"],
            maximumTokens: options.maximumTokens)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        try printMarker(responseMarker, data: encoder.encode(ResponseEnvelope(response: response)))
    }

    private static func runBatch(modelPath: String) async throws {
        let input = FileHandle.standardInput.readDataToEndOfFile()
        let request = try AgentBatchInferenceProtocol.decodeRequest(input)
        let model = try ModelFolderValidator.validateFullBF16(path: modelPath)
        let environment = ProcessInfo.processInfo.environment
        let adapter = try AdapterRuntimeOptions.parse(environment: environment)
        let options = try EvaluationGenerationOptions.parse(environment: environment)
        let generated = try await BehaviorEvaluationEngine.requestBatch(
            modelDirectory: model.path,
            prompts: request.requests.map(\.prompt),
            adapterDirectory: adapter.directory,
            adapterScaleOverride: adapter.scaleOverride,
            systemPrompt: environment["ABSLAYER_SYSTEM_PROMPT"],
            maximumTokens: options.maximumTokens)
        guard generated.count == request.requests.count else {
            throw AgentBatchInferenceProtocolError.tooManyResponses(generated.count)
        }
        let responses = zip(request.requests, generated).map { request, result in
            AgentBatchInferenceResponseItem(
                id: request.id, latencyMilliseconds: result.latencyMilliseconds,
                response: result.response)
        }
        let envelope = AgentBatchInferenceResponse(responses: responses)
        try printMarker(
            batchResponseMarker,
            data: AgentBatchInferenceProtocol.encodeResponse(envelope))
    }

    private static func printMarker(_ marker: String, data: Data) throws {
        guard let line = String(data: data, encoding: .utf8) else {
            throw CocoaError(.fileWriteInapplicableStringEncoding)
        }
        print(marker + line)
    }

    private static func writeUsageAndExit() -> Never {
        FileHandle.standardError.write(Data(
            "usage: abslayer-request MODEL_FOLDER PROMPT\n".utf8))
        FileHandle.standardError.write(Data(
            "       abslayer-request --batch MODEL_FOLDER < canonical-request.json\n".utf8))
        exit(2)
    }
}
