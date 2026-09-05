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
            var plans = ["workspace_preflight": Harness.workspacePreflight(
                workspace: workspace, root: state, swift: URL(fileURLWithPath: args[3]))]
            let environment = ProcessInfo.processInfo.environment
            if let worker = environment["ABSLAYER_LAGUNA_WORKER"],
               let model = environment["ABSLAYER_LAGUNA_MODEL"],
               let dataset = environment["ABSLAYER_LAGUNA_DATASET"],
               let generator = environment["ABSLAYER_LAGUNA_GENERATOR"],
               let output = environment["ABSLAYER_LAGUNA_OUTPUT"] {
                let workerURL = URL(fileURLWithPath: worker).standardizedFileURL
                plans["laguna_control_vector"] = WorkerPlan(
                    executable: URL(fileURLWithPath: "/usr/bin/env"),
                    arguments: ["ruby", workerURL.path, dataset, model, generator, output, "32"],
                    environment: ["PATH": "/usr/bin:/bin"],
                    inputs: [workerURL, URL(fileURLWithPath: dataset),
                             URL(fileURLWithPath: model), URL(fileURLWithPath: generator)])
            }
            if let worker = environment["ABSLAYER_LAGUNA_SCREEN_WORKER"],
               let server = environment["ABSLAYER_LAGUNA_SERVER"],
               let model = environment["ABSLAYER_LAGUNA_MODEL"],
               let vector = environment["ABSLAYER_LAGUNA_VECTOR"],
               let dataset = environment["ABSLAYER_LAGUNA_SCREEN_DATASET"],
               let output = environment["ABSLAYER_LAGUNA_SCREEN_OUTPUT"] {
                let workerURL = URL(fileURLWithPath: worker).standardizedFileURL
                let scale = environment["ABSLAYER_LAGUNA_SCREEN_SCALE"] ?? "-0.5"
                plans["laguna_vector_screen"] = WorkerPlan(
                    executable: URL(fileURLWithPath: "/usr/bin/env"),
                    arguments: ["ruby", workerURL.path, server, model, vector, dataset, output, scale],
                    environment: ["PATH": "/usr/bin:/bin"],
                    inputs: [workerURL, URL(fileURLWithPath: server),
                             URL(fileURLWithPath: model), URL(fileURLWithPath: vector),
                             URL(fileURLWithPath: dataset)])
            }
            let harness = try Harness(workspace: workspace, root: state, plans: plans)
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
