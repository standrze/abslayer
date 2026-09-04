#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerPromptEndPatch {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 5 else {
            usage()
            exit(2)
        }
        let environment = ProcessInfo.processInfo.environment
        let layers = try requiredIntegerList(
            environment, key: "ABSLAYER_PROMPT_END_LAYERS")
        let configuration = try PromptEndPatchConfiguration(
            layersZeroBased: layers,
            caseOffset: try integer(
                environment, key: "ABSLAYER_PROMPT_END_CASE_OFFSET",
                fallback: 0),
            maximumCases: try integer(
                environment, key: "ABSLAYER_PROMPT_END_MAX_CASES",
                fallback: 4),
            maximumGenerationTokens: try integer(
                environment, key: "ABSLAYER_PROMPT_END_MAX_TOKENS",
                fallback: 96),
            randomControlSeed: try unsignedInteger(
                environment, key: "ABSLAYER_PROMPT_END_RANDOM_SEED",
                fallback: PromptEndPatchConfiguration.defaultRandomSeed))
        let scale = try optionalFloat(
            environment, key: "ABSLAYER_ADAPTER_SCALE")
        let document = try MatchedResponsePatchDocument.read(
            from: arguments[3])

        FileHandle.standardError.write(Data(
            "DIAGNOSTIC ONLY: same-token prompt-end patching uses dev data and cannot certify abliteration. Do not pass a frozen audit artifact.\n".utf8))
        let study = try await PromptEndCrossConditionPatchingEngine.run(
            modelDirectory: arguments[1],
            adapterDirectory: arguments[2],
            adapterScaleOverride: scale,
            document: document,
            configuration: configuration,
            progress: { print($0) })
        try study.write(to: arguments[4])

        for summary in study.layerSummaries {
            let random = summary.randomMeanClosenessFraction.map {
                String(format: "%.4f", $0)
            } ?? "n/a"
            let gap = summary.matchedMinusRandomMeanClosenessFraction.map {
                String(format: "%.4f", $0)
            } ?? "n/a"
            let beats = summary.matchedBeatsRandomRate.map {
                String(format: "%.1f%%", 100 * $0)
            } ?? "n/a"
            print(String(
                format: "L0=%d matched-closure=%.4f random-closure=%@ gap=%@ matched>random=%@",
                summary.layerZeroBased,
                summary.matchedMeanClosenessFraction,
                random, gap, beats))
        }
        print(
            "Prompt-end cross-condition diagnostic -> "
                + URL(fileURLWithPath: arguments[4])
                    .standardizedFileURL.path)
        print(study.warning)
    }

    private static func usage() {
        FileHandle.standardError.write(Data(
            """
            usage: abslayer-prompt-end-patch \
              MODEL_BF16 DONOR_ADAPTER DEV_MATCHED_JSON OUTPUT_JSON

            required environment:
              ABSLAYER_PROMPT_END_LAYERS=16,19,23

            optional environment:
              ABSLAYER_PROMPT_END_CASE_OFFSET=0
              ABSLAYER_PROMPT_END_MAX_CASES=4
              ABSLAYER_PROMPT_END_MAX_TOKENS=96
              ABSLAYER_PROMPT_END_RANDOM_SEED=3232080254
              ABSLAYER_ADAPTER_SCALE=32

            Layer indices are zero-based. Use disjoint offsets for discovery
            and validation; never use a frozen audit artifact as input.
            """.utf8))
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
        guard !pieces.isEmpty, pieces.count == values.count else {
            throw CLIError.invalid(key: key, value: raw)
        }
        return values
    }

    private static func integer(
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

    private static func optionalFloat(
        _ environment: [String: String], key: String
    ) throws -> Float? {
        guard let raw = environment[key] else { return nil }
        guard let value = Float(raw.trimmingCharacters(
            in: .whitespacesAndNewlines)), value.isFinite, value >= 0
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
