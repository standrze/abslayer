#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerPreflightCommand {
    static func main() async {
        do {
            let invocation = try ABSlayerPreflightInvocation.parse(
                arguments: Array(CommandLine.arguments.dropFirst()))
            let report = try await ABSlayerPreflight.run(invocation)
            try writeJSON(report, to: .standardOutput)
        } catch {
            writeError(error.localizedDescription)
            exit(2)
        }
    }

    private struct ErrorLine: Encodable {
        let status = "error"
        let error: String
    }

    private static func writeJSON<T: Encodable>(
        _ value: T, to handle: FileHandle
    ) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var data = try encoder.encode(value)
        data.append(0x0a)
        try handle.write(contentsOf: data)
    }

    private static func writeError(_ message: String) {
        try? writeJSON(ErrorLine(error: message), to: .standardError)
    }
}
