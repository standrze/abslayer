import Foundation
import ProbeCore

private let reportMarker = "ABSLAYER_GEMMA_PARITY_JSON\t"

@main
enum ABSlayerGemmaParity {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard (3...4).contains(arguments.count) else {
            writeUsageAndExit()
        }
        let environment = ProcessInfo.processInfo.environment
        let adapter = try AdapterRuntimeOptions.parse(environment: environment)
        let report = try await GemmaParityDiagnosticEngine.run(
            modelDirectory: arguments[1],
            datasetPath: arguments[2],
            recordID: arguments.count == 4 ? arguments[3] : nil,
            adapterDirectory: adapter.directory,
            adapterScaleOverride: adapter.scaleOverride,
            environment: environment)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(report)
        guard let json = String(data: data, encoding: .utf8) else {
            throw CocoaError(.fileWriteInapplicableStringEncoding)
        }
        print(reportMarker + json)
    }

    private static func writeUsageAndExit() -> Never {
        FileHandle.standardError.write(
            Data(
                "usage: abslayer-gemma-parity MODEL_FOLDER HARNESS_JSON [RECORD_ID]\n".utf8))
        FileHandle.standardError.write(
            Data(
                "optional adapter: ABSLAYER_ADAPTER_DIR and ABSLAYER_ADAPTER_SCALE\n".utf8))
        exit(2)
    }
}
