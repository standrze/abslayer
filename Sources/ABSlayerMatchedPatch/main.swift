#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerMatchedPatch {
    static func main() async throws {
        let arguments = CommandLine.arguments
        if (7 ... 8).contains(arguments.count),
           arguments[1] == "build-input"
        {
            try buildInput(arguments)
            return
        }
        guard arguments.count == 4 else {
            usage()
            exit(2)
        }
        let environment = ProcessInfo.processInfo.environment
        let layers = try requiredIntegerList(
            environment, key: "ABSLAYER_PATCH_LAYERS")
        let positions = try requiredIntegerList(
            environment, key: "ABSLAYER_PATCH_RESPONSE_POSITIONS")
        let maximumCases = try boundedInteger(
            environment, key: "ABSLAYER_PATCH_MAX_CASES", fallback: 4)
        let maximumSuffixTokens = try boundedInteger(
            environment, key: "ABSLAYER_PATCH_SUFFIX_TOKENS", fallback: 8)
        let randomSeed = try unsignedInteger(
            environment, key: "ABSLAYER_PATCH_RANDOM_SEED",
            fallback: MatchedResponsePatchConfiguration.defaultRandomSeed)
        let configuration = try MatchedResponsePatchConfiguration(
            layersZeroBased: layers,
            responseTokenPositionsZeroBased: positions,
            maximumCases: maximumCases,
            maximumSuffixTokens: maximumSuffixTokens,
            randomControlSeed: randomSeed)
        let input = try MatchedResponsePatchDocument.read(from: arguments[2])

        FileHandle.standardError.write(Data(
            "DIAGNOSTIC ONLY: matched activation patching cannot certify abliteration. Input must contain independently judged dev-only donor/recipient continuations.\n".utf8))
        let study = try await MatchedResponseStatePatchingEngine.run(
            modelDirectory: arguments[1],
            document: input,
            configuration: configuration,
            progress: { message in print(message) })
        try study.write(to: arguments[3])

        let summary = study.summary
        let random = summary.randomControlMeanClosenessGain.map {
            String(format: "%.6f", $0)
        } ?? "n/a"
        let specificity = summary.matchedMinusRandomMeanClosenessGain.map {
            String(format: "%.6f", $0)
        } ?? "n/a"
        print(String(
            format: "Matched patch diagnostic: sites=%d matched-gain=%.6f reverse-gain=%.6f random-gain=%@ matched-minus-random=%@ -> %@",
            summary.resultCount,
            summary.matchedMeanClosenessGain,
            summary.reverseMeanClosenessGain,
            random,
            specificity,
            URL(fileURLWithPath: arguments[3]).standardizedFileURL.path))
        print(study.warning)
    }

    private static func usage() {
        FileHandle.standardError.write(Data(
            """
            usage: abslayer-matched-patch MODEL_BF16 DEV_MATCHED_JSON OUTPUT_JSON

              abslayer-matched-patch build-input \
                RECIPIENT_RESPONSES RECIPIENT_JUDGMENTS \
                DONOR_RESPONSES DONOR_JUDGMENTS OUTPUT_JSON [MODEL_CONDITION]

            required environment:
              ABSLAYER_PATCH_LAYERS=16,19,23
              ABSLAYER_PATCH_RESPONSE_POSITIONS=0,1,3

            optional environment:
              ABSLAYER_PATCH_MAX_CASES=4
              ABSLAYER_PATCH_SUFFIX_TOKENS=8
              ABSLAYER_PATCH_RANDOM_SEED=2823923682

            All layer and response-token positions are zero-based.
            """.utf8))
    }

    private static func buildInput(_ arguments: [String]) throws {
        let decoder = JSONDecoder()
        let recipientResponses = try decoder.decode(
            [PromptResult].self,
            from: Data(contentsOf: URL(
                fileURLWithPath: arguments[2]).standardizedFileURL))
        let recipientJudgments = try decoder.decode(
            [RecordedOutcomeJudgment].self,
            from: Data(contentsOf: URL(
                fileURLWithPath: arguments[3]).standardizedFileURL))
        let donorResponses = try decoder.decode(
            [PromptResult].self,
            from: Data(contentsOf: URL(
                fileURLWithPath: arguments[4]).standardizedFileURL))
        let donorJudgments = try decoder.decode(
            [RecordedOutcomeJudgment].self,
            from: Data(contentsOf: URL(
                fileURLWithPath: arguments[5]).standardizedFileURL))
        let provenance = MatchedResponsePatchProvenance(
            recipientResponsesPath: normalizedPath(arguments[2]),
            recipientJudgmentsPath: normalizedPath(arguments[3]),
            donorResponsesPath: normalizedPath(arguments[4]),
            donorJudgmentsPath: normalizedPath(arguments[5]))
        let document = try MatchedResponsePatchDocument.build(
            recipientResponses: recipientResponses,
            recipientJudgments: recipientJudgments,
            donorResponses: donorResponses,
            donorJudgments: donorJudgments,
            modelCondition: arguments.count == 8 ? arguments[7] : nil,
            provenance: provenance)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(document).write(
            to: URL(fileURLWithPath: arguments[6]).standardizedFileURL,
            options: .atomic)
        print(
            "Saved \(document.cases.count) exact-prompt dev cases "
                + "(recipient=refusal, donor=compliance) -> "
                + normalizedPath(arguments[6]))
    }

    private static func normalizedPath(_ path: String) -> String {
        URL(fileURLWithPath: path).standardizedFileURL.path
    }

    private static func requiredIntegerList(
        _ environment: [String: String], key: String
    ) throws -> [Int] {
        guard let raw = environment[key] else {
            throw CLIError.missing(key)
        }
        let pieces = raw.split(
            separator: ",", omittingEmptySubsequences: false)
        let values = pieces.compactMap {
            Int($0.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        guard !pieces.isEmpty, values.count == pieces.count else {
            throw CLIError.invalid(key: key, value: raw)
        }
        return values
    }

    private static func boundedInteger(
        _ environment: [String: String], key: String, fallback: Int
    ) throws -> Int {
        guard let raw = environment[key] else { return fallback }
        guard let value = Int(raw.trimmingCharacters(
            in: .whitespacesAndNewlines))
        else { throw CLIError.invalid(key: key, value: raw) }
        return value
    }

    private static func unsignedInteger(
        _ environment: [String: String], key: String, fallback: UInt64
    ) throws -> UInt64 {
        guard let raw = environment[key] else { return fallback }
        guard let value = UInt64(raw.trimmingCharacters(
            in: .whitespacesAndNewlines))
        else { throw CLIError.invalid(key: key, value: raw) }
        return value
    }

    private enum CLIError: LocalizedError {
        case missing(String)
        case invalid(key: String, value: String)

        var errorDescription: String? {
            switch self {
            case .missing(let key):
                "Required environment variable \(key) is not set."
            case .invalid(let key, let value):
                "\(key) has an invalid value: '\(value)'."
            }
        }
    }
}
