#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerEval {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count == 5, let maximum = Int(args[4]) else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-eval MODEL_FOLDER HOLDOUT_JSON REPORT_PATH MAX_CASES\n".utf8))
            exit(2)
        }
        let adapter: AdapterRuntimeOptions
        let evaluationOptions: EvaluationGenerationOptions
        do {
            adapter = try AdapterRuntimeOptions.parse(
                environment: ProcessInfo.processInfo.environment)
            evaluationOptions = try EvaluationGenerationOptions.parse(
                environment: ProcessInfo.processInfo.environment)
            try MLXResourceGuard.apply(
                environment: ProcessInfo.processInfo.environment)
        } catch {
            FileHandle.standardError.write(Data("Error: \(error.localizedDescription)\n".utf8))
            exit(2)
        }
        let model = try ModelFolderValidator.validateFullBF16(path: args[1])
        let pairs = try PromptFile.load(args[2])
        let systemPrompt = ProcessInfo.processInfo.environment["ABSLAYER_SYSTEM_PROMPT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if ProcessInfo.processInfo.environment["ABSLAYER_SYSTEM_PROMPT"] != nil,
           systemPrompt?.isEmpty != false
        {
            FileHandle.standardError.write(Data(
                "Error: ABSLAYER_SYSTEM_PROMPT cannot be empty.\n".utf8))
            exit(2)
        }
        if systemPrompt != nil {
            print("using one recorded system prompt for both contrast and control channels")
        }
        let responses = try await BehaviorEvaluationEngine.run(
            modelDirectory: model.path, pairs: pairs, maximumCases: maximum,
            adapterDirectory: adapter.directory,
            adapterScaleOverride: adapter.scaleOverride,
            systemPrompt: systemPrompt,
            maximumTokens: evaluationOptions.maximumTokens)
        if let responsesPath = ProcessInfo.processInfo.environment["ABSLAYER_RESPONSES_JSON"] {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(responses).write(
                to: URL(fileURLWithPath: responsesPath).standardizedFileURL,
                options: .atomic)
        }
        let evaluation: RefusalEvaluation
        if let judgmentsPath = ProcessInfo.processInfo.environment["ABSLAYER_OUTCOME_JUDGMENTS"] {
            let data = try Data(contentsOf: URL(fileURLWithPath: judgmentsPath))
            let judgments = try JSONDecoder().decode([RecordedOutcomeJudgment].self, from: data)
            evaluation = RefusalEvaluator.evaluate(
                responses, classifier: RecordedOutcomeClassifier(judgments: judgments))
        } else {
            FileHandle.standardError.write(Data(
                "WARNING: ABSLAYER_OUTCOME_JUDGMENTS is unset; using the explicit substring fallback.\n".utf8))
            evaluation = RefusalEvaluator.evaluate(
                responses, classifier: SubstringOutcomeClassifier())
        }
        var details = [evaluation.rendered, ""]
        for response in responses {
            details.append("[\(response.name)] contrast: \(response.contrastResponse)")
            details.append("[\(response.name)] control: \(response.controlResponse)")
            details.append("")
        }
        try details.joined(separator: "\n").write(
            toFile: args[3], atomically: true, encoding: .utf8)
        print(evaluation.rendered)
        print("Detailed local report: \(args[3])")
        if !evaluation.allPassed { exit(1) }
    }
}
