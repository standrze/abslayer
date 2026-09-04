import Foundation
import ABSlayerHarness
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

@main
enum JobHost {
    static func main() {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            guard args.count == 4, ["request", "drain"].contains(args[0]) else {
                throw HarnessError("invalid_invocation", "Internal host expects MODE WORKSPACE STATE_DIRECTORY SWIFT_EXECUTABLE.")
            }
            let workspace = URL(fileURLWithPath: args[1]).resolvingSymlinksInPath()
            // Resolve after creation: Foundation can canonicalize a missing
            // macOS /var path differently from the same path once it exists.
            let requestedState = URL(fileURLWithPath: args[2]).standardizedFileURL
            try FileManager.default.createDirectory(at: requestedState, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            let state = requestedState.resolvingSymlinksInPath()
            let plan = Harness.workspacePreflight(workspace: workspace, root: state,
                                                 swift: URL(fileURLWithPath: args[3]))
            let harness = try Harness(workspace: workspace, root: state, plan: plan)
            if args[0] == "drain" { try harness.drain(); return }
            var input = Data()
            while input.count <= 16_384 {
                let chunk = try FileHandle.standardInput.read(upToCount: 16_385 - input.count) ?? Data()
                if chunk.isEmpty { break }
                input.append(chunk)
            }
            let response = try harness.handle(ToolRequest.decode(input))
            try output(response)
        } catch {
            let problem = error as? HarnessError
            var response = ToolResponse(); response.ok = false
            response.code = problem?.code ?? "controller_error"
            response.message = error.localizedDescription
            try? output(response)
            exit(2)
        }
    }
    private static func output(_ response: ToolResponse) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        var data = try encoder.encode(response); data.append(0x0a)
        try FileHandle.standardOutput.write(contentsOf: data)
    }
}
