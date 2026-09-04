#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerJudge {
    static func main() async {
        do {
            try await run()
        } catch {
            let message = "abslayer-judge: \(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(message.utf8))
            exit(2)
        }
    }

    private static func run() async throws {
        let args = CommandLine.arguments
        guard args.count == 5, let trialIndex = Int(args[2]) else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-judge STUDY_JSON TRIAL_INDEX JUDGE_MODEL JUDGMENTS_JSON\n".utf8))
            exit(2)
        }
        let inputData = try Data(contentsOf: URL(fileURLWithPath: args[1]))
        let decoder = JSONDecoder()
        let responses: [PromptResult]
        if let directResponses = try? decoder.decode([PromptResult].self, from: inputData) {
            responses = directResponses
        } else {
            let study = try decoder.decode(ExactResidualInterventionStudy.self, from: inputData)
            guard study.trials.indices.contains(trialIndex),
                  let trialResponses = study.trials[trialIndex].responses
            else { throw SemanticJudgeError.missingResponse }
            responses = trialResponses
        }
        let judgments = try await SemanticJudgeEngine.judge(
            modelDirectory: args[3], responses: responses)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(judgments).write(
            to: URL(fileURLWithPath: args[4]).standardizedFileURL,
            options: .atomic)
        let evaluation = RefusalEvaluator.evaluate(
            responses,
            classifier: RecordedOutcomeClassifier(
                judgments: judgments, identifier: "local-semantic-model-judge-v1"))
        print(evaluation.rendered)
        print("Saved semantic judgments \(args[4])")
        if !evaluation.allPassed { exit(1) }
    }
}
