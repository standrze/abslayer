import Foundation
import ProbeCore

@main
enum ABSlayerLoRAPipeline {
    static func main() throws {
        let args = CommandLine.arguments
        guard (6 ... 8).contains(args.count) else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-lora-pipeline BF16_MODEL HARNESS_JSON WORK_DIR OUTPUT_MODEL PROMPT [ITERS] [LR]\n".utf8))
            exit(2)
        }
        let source = try ModelFolderValidator.validateFullBF16(path: args[1])
        _ = try HarnessTrainingDatasetLoader.load(args[2])
        let manager = FileManager.default
        let work = URL(fileURLWithPath: args[3]).standardizedFileURL
        let output = URL(fileURLWithPath: args[4]).standardizedFileURL
        guard !manager.fileExists(atPath: work.path) else {
            throw PipelineError.pathAlreadyExists(work.path)
        }
        guard !manager.fileExists(atPath: output.path) else {
            throw PipelineError.pathAlreadyExists(output.path)
        }
        try manager.createDirectory(at: work, withIntermediateDirectories: true)
        let adapter = work.appendingPathComponent("adapter", isDirectory: true)
        let executableDirectory = URL(fileURLWithPath: args[0]).standardizedFileURL
            .deletingLastPathComponent()
        try run(
            executableDirectory.appendingPathComponent("abslayer-prefix-train"),
            [source.path, args[2], adapter.path,
             args.count >= 7 ? args[6] : "80", args.count >= 8 ? args[7] : "1e-5"])
        try run(
            executableDirectory.appendingPathComponent("abslayer-lora-merge"),
            [source.path, adapter.path, output.path])
        try run(
            executableDirectory.appendingPathComponent("abslayer-request"),
            [output.path, args[5]])
    }

    private static func run(_ executable: URL, _ arguments: [String]) throws {
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw PipelineError.missingSiblingExecutable(executable.path)
        }
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        process.standardInput = FileHandle.standardInput
        process.standardOutput = FileHandle.standardOutput
        process.standardError = FileHandle.standardError
        try process.run()
        process.waitUntilExit()
        guard process.terminationReason == .exit, process.terminationStatus == 0 else {
            throw PipelineError.stageFailed(executable.lastPathComponent, process.terminationStatus)
        }
    }
}

private enum PipelineError: LocalizedError {
    case pathAlreadyExists(String)
    case missingSiblingExecutable(String)
    case stageFailed(String, Int32)

    var errorDescription: String? {
        switch self {
        case .pathAlreadyExists(let path): "Refusing to overwrite existing path: \(path)"
        case .missingSiblingExecutable(let path):
            "Missing pipeline executable \(path). Build all products before running the pipeline."
        case .stageFailed(let stage, let status): "Pipeline stage \(stage) exited with status \(status)."
        }
    }
}
