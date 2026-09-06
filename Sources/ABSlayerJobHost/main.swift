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
            func configuredURL(_ path: String) -> URL {
                URL(fileURLWithPath: path).standardizedFileURL
            }
            let ruby = URL(fileURLWithPath: environment["ABSLAYER_RUBY"] ?? "/usr/bin/ruby")
                .standardizedFileURL
            if let worker = environment["ABSLAYER_LAGUNA_VERIFY_WORKER"],
               let server = environment["ABSLAYER_LAGUNA_SERVER"],
               let model = environment["ABSLAYER_LAGUNA_MODEL"],
               let vector = environment["ABSLAYER_LAGUNA_VECTOR"],
               let fixture = environment["ABSLAYER_LAGUNA_VERIFY_FIXTURE"],
               let output = environment["ABSLAYER_LAGUNA_VERIFY_OUTPUT"] {
                let workerURL = configuredURL(worker), serverURL = configuredURL(server)
                let modelURL = configuredURL(model), vectorURL = configuredURL(vector)
                let fixtureURL = configuredURL(fixture), outputURL = configuredURL(output)
                let scale = environment["ABSLAYER_LAGUNA_VERIFY_SCALE"] ?? "-0.25"
                plans["laguna_independent_verify"] = WorkerPlan(
                    executable: ruby,
                    arguments: [workerURL.path, serverURL.path, modelURL.path, vectorURL.path,
                                fixtureURL.path, outputURL.path, scale],
                    environment: ["PATH": "/usr/bin:/bin"],
                    inputs: [workerURL, serverURL, modelURL, vectorURL, fixtureURL],
                    artifacts: ["private_results": outputURL
                        .appendingPathComponent("private-responses.json").standardizedFileURL])
            }
            if let worker = environment["ABSLAYER_LAGUNA_AUTHORIZED_SCREEN_WORKER"],
               let server = environment["ABSLAYER_LAGUNA_SERVER"],
               let model = environment["ABSLAYER_LAGUNA_MODEL"],
               let dataset = environment["ABSLAYER_LAGUNA_AUTHORIZED_DATASET"],
               let manifest = environment["ABSLAYER_LAGUNA_AUTHORIZED_MANIFEST"],
               let output = environment["ABSLAYER_LAGUNA_AUTHORIZED_SCREEN_OUTPUT"] {
                let workerURL = configuredURL(worker), serverURL = configuredURL(server)
                let modelURL = configuredURL(model), datasetURL = configuredURL(dataset)
                let manifestURL = configuredURL(manifest), outputURL = configuredURL(output)
                let mode = environment["ABSLAYER_LAGUNA_AUTHORIZED_SCREEN_MODE"] ?? "balanced"
                plans["laguna_authorized_screen"] = WorkerPlan(
                    executable: ruby,
                    arguments: [workerURL.path, serverURL.path, modelURL.path, datasetURL.path,
                                manifestURL.path, outputURL.path, mode],
                    environment: ["PATH": "/usr/bin:/bin"],
                    inputs: [workerURL, serverURL, modelURL, datasetURL, manifestURL],
                    artifacts: ["private_results": outputURL
                        .appendingPathComponent("private-responses.json").standardizedFileURL])
            }
            if let worker = environment["ABSLAYER_LAGUNA_REVIEWED_VECTOR_WORKER"],
               let selection = environment["ABSLAYER_LAGUNA_REVIEWED_SELECTION"],
               let screen = environment["ABSLAYER_LAGUNA_REVIEWED_SCREEN_RESULTS"],
               let dataset = environment["ABSLAYER_LAGUNA_AUTHORIZED_DATASET"],
               let manifest = environment["ABSLAYER_LAGUNA_AUTHORIZED_MANIFEST"],
               let model = environment["ABSLAYER_LAGUNA_MODEL"],
               let server = environment["ABSLAYER_LAGUNA_SERVER"],
               let generator = environment["ABSLAYER_LAGUNA_GENERATOR"],
               let output = environment["ABSLAYER_LAGUNA_REVIEWED_VECTOR_OUTPUT"] {
                let workerURL = configuredURL(worker), selectionURL = configuredURL(selection)
                let screenURL = configuredURL(screen), datasetURL = configuredURL(dataset)
                let manifestURL = configuredURL(manifest), modelURL = configuredURL(model)
                let serverURL = configuredURL(server), generatorURL = configuredURL(generator)
                let outputURL = configuredURL(output)
                plans["laguna_reviewed_vector"] = WorkerPlan(
                    executable: ruby,
                    arguments: [workerURL.path, selectionURL.path, screenURL.path, datasetURL.path,
                                manifestURL.path, modelURL.path, serverURL.path, generatorURL.path,
                                outputURL.path],
                    environment: ["PATH": "/usr/bin:/bin"],
                    inputs: [workerURL, selectionURL, screenURL, datasetURL, manifestURL,
                             modelURL, serverURL, generatorURL],
                    artifacts: ["vector": outputURL
                        .appendingPathComponent("control-vector.gguf").standardizedFileURL])
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
