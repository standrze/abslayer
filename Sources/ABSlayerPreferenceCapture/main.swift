#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

private struct CaptureInput: Codable {
    let prompt: String
    let target: String
    let category: String?
}

private struct CapturedPreference: Codable {
    let prompt: String
    let target: String
    let rejected: String
    let category: String?
}

@main
enum ABSlayerPreferenceCapture {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard (4 ... 5).contains(arguments.count) else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-preference-capture MODEL_FOLDER INPUT_JSON OUTPUT_JSON [MAX_TOKENS]\n".utf8))
            exit(2)
        }
        let maximumTokens = arguments.count == 5 ? Int(arguments[4]) : 96
        guard let maximumTokens, maximumTokens > 0 else {
            FileHandle.standardError.write(Data("MAX_TOKENS must be positive.\n".utf8))
            exit(2)
        }
        let adapter = try AdapterRuntimeOptions.parse(
            environment: ProcessInfo.processInfo.environment)
        try MLXResourceGuard.apply(
            environment: ProcessInfo.processInfo.environment)
        let model = try ModelFolderValidator.validateFullBF16(path: arguments[1])
        let inputURL = URL(fileURLWithPath: arguments[2]).standardizedFileURL
        let outputURL = URL(fileURLWithPath: arguments[3]).standardizedFileURL
        let decoder = JSONDecoder()
        let inputs = try decoder.decode(
            [CaptureInput].self, from: Data(contentsOf: inputURL))
        guard !inputs.isEmpty else {
            FileHandle.standardError.write(Data("INPUT_JSON is empty.\n".utf8))
            exit(2)
        }

        var captured = [CapturedPreference]()
        if FileManager.default.fileExists(atPath: outputURL.path) {
            captured = try decoder.decode(
                [CapturedPreference].self, from: Data(contentsOf: outputURL))
            print("resuming after \(captured.count)/\(inputs.count) captured preferences")
        }
        guard captured.count <= inputs.count,
              zip(captured, inputs).allSatisfy({ pair in
                  pair.0.prompt == pair.1.prompt && pair.0.target == pair.1.target
              })
        else {
            FileHandle.standardError.write(Data(
                "OUTPUT_JSON is not a prefix of INPUT_JSON; refusing an unsafe resume.\n".utf8))
            exit(2)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let runtime = try await ResidentTrialRuntime(modelDirectory: model.path)
        if let directory = adapter.directory {
            let loaded = try LoRAAdapterLoader.load(
                directory: directory, scaleOverride: adapter.scaleOverride)
            try await runtime.load(loaded)
            print(
                "loaded capture adapter from \(directory)"
                    + (adapter.scaleOverride.map { " at scale \($0)" } ?? ""))
        }
        for index in captured.count ..< inputs.count {
            let input = inputs[index]
            let response = try await runtime.response(
                to: input.prompt, maximumTokens: maximumTokens)
            captured.append(CapturedPreference(
                prompt: input.prompt,
                target: input.target,
                rejected: response,
                category: input.category))
            // Atomic incremental publication makes a long capture safely resumable.
            try encoder.encode(captured).write(to: outputURL, options: .atomic)
            print("captured baseline response \(index + 1)/\(inputs.count)")
        }
        print("Saved \(captured.count) model-conditioned preference candidates to \(outputURL.path)")
    }
}
