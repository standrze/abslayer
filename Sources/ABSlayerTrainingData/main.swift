import Foundation
import ProbeCore

@main
enum ABSlayerTrainingData {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count == 3, args[1] == "validate" else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-training-data validate HARNESS_JSON\n".utf8))
            exit(2)
        }
        let examples = try HarnessTrainingDatasetLoader.load(args[2])
        let preferenceCount = examples.count(where: { $0.rejected != nil })
        let objective = preferenceCount == 0 ? "supervised completions" : "preferences"
        print("Valid: \(examples.count) \(objective) record(s)")
    }
}
