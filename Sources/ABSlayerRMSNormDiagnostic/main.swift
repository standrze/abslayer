import Foundation
import ProbeCore

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

private let reportMarker = "ABSLAYER_RMS_NORM_DIAGNOSTIC_JSON\t"

@main
enum ABSlayerRMSNormDiagnostic {
    static func main() {
        let arguments = Array(CommandLine.arguments.dropFirst())
        var repetitions = RMSNormCUDADiagnosticConfiguration.defaultRepetitions
        var sawRepetitions = false
        var allowCUDAGraphs = false
        for argument in arguments {
            if argument == "--allow-cuda-graphs" {
                allowCUDAGraphs = true
            } else if !sawRepetitions, let value = Int(argument) {
                repetitions = value
                sawRepetitions = true
            } else {
                writeUsageAndExit()
            }
        }

        do {
            let report = try RMSNormCUDADiagnosticEngine.run(
                repetitions: repetitions,
                allowCUDAGraphs: allowCUDAGraphs)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            encoder.keyEncodingStrategy = .convertToSnakeCase
            let data = try encoder.encode(report)
            guard let json = String(data: data, encoding: .utf8) else {
                throw CocoaError(.fileWriteInapplicableStringEncoding)
            }
            print(reportMarker + json)
            if !report.passed {
                exit(1)
            }
        } catch {
            FileHandle.standardError.write(
                Data("abslayer-rmsnorm-diagnostic: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    private static func writeUsageAndExit() -> Never {
        FileHandle.standardError.write(
            Data(
                "usage: abslayer-rmsnorm-diagnostic [REPETITIONS] [--allow-cuda-graphs]\n"
                    .utf8))
        exit(2)
    }
}
