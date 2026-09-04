#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
import Foundation
import ProbeCore

@main
enum ABSlayerBackendCommand {
    static func main() async {
        // This must precede every MLX inspection or model-container creation.
        // CUDA graph replay is intentionally disabled for the patched Gemma 4
        // path until upstream graph correctness is established.
        guard setenv("MLX_USE_CUDA_GRAPHS", "0", 1) == 0 else {
            writeError("Could not disable MLX CUDA graphs before initialization.")
            exit(2)
        }
        do {
            let invocation = try ABSlayerBackendInvocation.parse(
                arguments: Array(CommandLine.arguments.dropFirst()))
            let processLock = try ABSlayerProcessLock()
            defer { withExtendedLifetime(processLock) {} }
            switch invocation {
            case .doctor(let request):
                try writeJSON(ABSlayerBackendRuntime.doctor(request))
            case .measure(let request):
                try writeJSON(await ABSlayerBackendRuntime.measure(request))
            case .apply(let request):
                try writeJSON(ABSlayerBackendRuntime.apply(request))
            case .verify(let request):
                let report = try await ABSlayerBackendRuntime.verify(request)
                try writeJSON(report)
                if !report.passed { exit(5) }
            }
        } catch {
            writeError(error.localizedDescription)
            exit(2)
        }
    }

    private struct ErrorLine: Encodable {
        let status = "error"
        let error: String
    }

    private static func writeJSON<T: Encodable>(_ value: T) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var data = try encoder.encode(value)
        data.append(0x0a)
        try FileHandle.standardOutput.write(contentsOf: data)
    }

    private static func writeError(_ message: String) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard var data = try? encoder.encode(ErrorLine(error: message)) else { return }
        data.append(0x0a)
        try? FileHandle.standardError.write(contentsOf: data)
    }
}
