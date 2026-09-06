import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

// Injectable only from Swift (tests/host), never from the JSON tool surface.
public struct WorkerPlan: Sendable {
    public var executable: URL
    public var arguments: [String]
    public var environment: [String: String]
    public var inputs: [URL]
    public var artifacts: [String: URL]
    public init(executable: URL, arguments: [String], environment: [String: String], inputs: [URL],
                artifacts: [String: URL] = [:]) {
        self.executable = executable; self.arguments = arguments
        self.environment = environment; self.inputs = inputs; self.artifacts = artifacts
    }
}

// Configuration is immutable; every persisted mutation is protected by flock.
public final class Harness: @unchecked Sendable {
    public let workspace: URL
    public let root: URL
    private let plans: [String: WorkerPlan]
    private let controller: FileBinding
    private let manager = FileManager.default
    static let outputLimit = 2 * 1_024 * 1_024
    static let artifactLimit = 64 * 1_024 * 1_024

    public convenience init(workspace: URL, root: URL, plan: WorkerPlan) throws {
        try self.init(workspace: workspace, root: root, plans: ["workspace_preflight": plan])
    }

    public init(workspace: URL, root: URL, plans: [String: WorkerPlan]) throws {
        let artifactNames = plans.values.flatMap { $0.artifacts.keys }
        guard !plans.isEmpty, plans.keys.allSatisfy({ !$0.isEmpty }),
              artifactNames.allSatisfy({ name in
                  !name.isEmpty && name != "stdout" && name != "stderr" &&
                  name.unicodeScalars.allSatisfy { CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_.-")).contains($0) }
              }),
              plans.values.allSatisfy({ plan in Set(plan.artifacts.values.map(\.standardizedFileURL.path)).count == plan.artifacts.count }) else {
            throw HarnessError("invalid_configuration", "At least one named worker plan is required.")
        }
        self.workspace = workspace.standardizedFileURL.resolvingSymlinksInPath()
        self.root = root.standardizedFileURL.resolvingSymlinksInPath()
        self.plans = plans
        self.controller = try Persistence.binding(URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL)
        try Persistence.directory(self.root)
        try Persistence.directory(self.root.appendingPathComponent("jobs"))
        _ = try transaction { _ in () }
    }

    public static func workspacePreflight(workspace: URL, root: URL, swift: URL) -> WorkerPlan {
        let inherited = ProcessInfo.processInfo.environment
        var environment = ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin"]
        for key in ["HOME", "TMPDIR", "DEVELOPER_DIR", "TOOLCHAINS", "SDKROOT", "SPM_CUDA"] {
            if let value = inherited[key] { environment[key] = value }
        }
        environment["CLANG_MODULE_CACHE_PATH"] = root.appendingPathComponent("module-cache").path
        environment["SWIFT_MODULECACHE_PATH"] = root.appendingPathComponent("module-cache").path
        if swift.lastPathComponent == "swift-driver" {
            environment["_ABSLAYER_EXECUTABLE_ARGV0"] = swift.deletingLastPathComponent()
                .appendingPathComponent("swift").path
        }
        return WorkerPlan(
            executable: swift,
            arguments: ["package", "--disable-sandbox", "--package-path", workspace.path,
                        "--cache-path", root.appendingPathComponent("swiftpm-cache").path,
                        "--scratch-path", root.appendingPathComponent("swiftpm-build").path,
                        "dump-package"],
            environment: environment,
            inputs: [workspace.appendingPathComponent("Package.swift"), workspace.appendingPathComponent("Package.resolved")])
    }

    public func handle(_ request: ToolRequest) throws -> ToolResponse {
        var response = ToolResponse()
        switch request.method {
        case "capabilities":
            response.operations = plans.keys.sorted()
            response.message = "Named, configuration-bound operations only. Execution never implies candidate acceptance. Methods: submit, status, cancel, evidence, recover."
        case "submit":
            response.job = try summary(submit(request))
        case "status":
            if request.jobID.isEmpty {
                let recent = try transaction { Array($0.jobs.suffix(20).reversed()) }
                response.jobs = try recent.map(summary)
                response.message = "Most recent 20 jobs; use jobID for a specific job."
            } else { response.job = try summary(job(request.jobID)) }
        case "cancel":
            let current = try transaction { store in
                let index = try index(request.jobID, in: store)
                if !store.jobs[index].state.terminal {
                    store.jobs[index].cancelRequested = true
                    store.jobs[index].updatedAt = Date()
                    if store.jobs[index].state == .queued { store.jobs[index].transition(.cancelled) }
                }
                return store.jobs[index]
            }
            response.job = try summary(current)
        case "recover":
            response.message = "A drain/reconciliation attempt is requested. An inherited worker lease must clear before interrupted jobs can be reconciled. Interrupted jobs are never retried automatically."
        case "evidence":
            let current = try job(request.jobID)
            guard current.state.terminal, let binding = current.outputs[request.stream] else {
                throw HarnessError("evidence_unavailable", "Bound evidence is available only after the worker has stopped and outputs have been recorded.")
            }
            try validateRecordedOutputs(current)
            let url = directory(current.id).appendingPathComponent(request.stream + ".log")
            let data = try Persistence.verifiedData(url, matching: binding, maxBytes: Self.outputLimit)
            let offset = min(request.offset, binding.bytes)
            let end = min(binding.bytes, offset + request.limit)
            response.text = String(decoding: data[offset..<end], as: UTF8.self)
            response.nextOffset = end < binding.bytes ? end : nil
            response.artifact = binding
        default: throw HarnessError("invalid_request", "Unknown method.")
        }
        response.needsDrain = try transaction { $0.jobs.contains { !$0.state.terminal } }
        return response
    }

    private func submit(_ request: ToolRequest) throws -> Job {
        guard let plan = plans[request.operation],
              (1...128).contains(request.idempotencyKey.utf8.count),
              !request.idempotencyKey.contains(where: { $0.isNewline }),
              (1...3_600).contains(request.timeoutSeconds) else {
            throw HarnessError("invalid_request", "Submit a listed operation with an idempotencyKey (1–128 bytes) and timeoutSeconds (1–3600).")
        }
        let inputs = try plan.inputs.map { try Persistence.binding($0) }
        let executable = try Persistence.binding(plan.executable)
        let digest = Persistence.digest(try Persistence.encode(ExecutionIdentity(
            workspace: workspace.path, operation: request.operation, timeout: request.timeoutSeconds,
            inputs: inputs, executable: executable, controller: controller,
            arguments: plan.arguments, environment: plan.environment,
            artifactPaths: plan.artifacts.mapValues { $0.standardizedFileURL.path })))
        let legacyDigest = Persistence.digest(try Persistence.encode(LegacyExecutionIdentity(
            workspace: workspace.path, operation: request.operation, timeout: request.timeoutSeconds,
            inputs: inputs, executable: executable, controller: controller,
            arguments: plan.arguments, environment: plan.environment)))
        return try transaction { store in
            if let previous = store.jobs.first(where: { $0.idempotencyKey == request.idempotencyKey }) {
                let sameRequest = previous.requestDigest == digest ||
                    (previous.artifactPaths == nil && plan.artifacts.isEmpty && previous.requestDigest == legacyDigest)
                guard sameRequest else {
                    throw HarnessError("idempotency_conflict", "This key already names a different request or input identity. Inspect its job; use a new key for an intentional new attempt.")
                }
                return previous
            }
            guard store.jobs.count < 1000 else { throw HarnessError("store_full", "This first-version store is limited to 1000 jobs.") }
            let now = Date()
            let id = UUID().uuidString.lowercased()
            try Persistence.directory(directory(id))
            let job = Job(id: id, idempotencyKey: request.idempotencyKey, requestDigest: digest,
                          operation: request.operation, timeoutSeconds: request.timeoutSeconds,
                          state: .queued, createdAt: now, updatedAt: now, cancelRequested: false,
                          inputs: inputs, executable: executable, controller: controller, arguments: plan.arguments,
                          environment: plan.environment,
                          artifactPaths: plan.artifacts.mapValues { $0.standardizedFileURL.path }, outputs: [:],
                          events: [.init(state: .queued, time: now)])
            store.jobs.append(job)
            return job
        }
    }

    func job(_ id: String) throws -> Job {
        try transaction { store in store.jobs[try index(id, in: store)] }
    }

    // Called by a detached host. All clients may request a drain; only one owns
    // the lease. Children inherit it, preventing overlap after a host crash.
    public func drain() throws {
        let owner: FileLock
        do { owner = try FileLock(root.appendingPathComponent("owner.lock"), nonblocking: true) }
        catch let error as HarnessError where error.code == "busy" { return }
        defer { owner.release() }
        try transaction { store in
            for i in store.jobs.indices where store.jobs[i].state == .running {
                store.jobs[i].transition(.interrupted, reason: "Previous supervisor ended without recording completion; worker lease is now clear. No automatic retry.")
            }
        }
        while true {
            let next: Job? = try transaction { store in
                guard let i = store.jobs.firstIndex(where: { $0.state == .queued }) else {
                    // Release under the store lock: a racing submission either
                    // precedes this check or starts a host after release.
                    owner.release()
                    return nil
                }
                store.jobs[i].transition(.running)
                return store.jobs[i]
            }
            guard let next else { return }
            try run(next, lease: owner.descriptor)
        }
    }

    private func run(_ submitted: Job, lease: Int32) throws {
        var result = submitted
        var attemptedOutputBinding = false
        do {
            try validateBindings(submitted)
            let worker = try SpawnedWorker(job: submitted, directory: directory(submitted.id),
                                           workspace: workspace, lease: lease)
            try transaction { store in
                store.jobs[try index(submitted.id, in: store)].workerPID = worker.pid
            }
            let started = Date()
            var stoppingAt: Date?
            var reason: String?
            var cancelled = false
            var exit: Int32?
            while exit == nil {
                exit = try worker.poll()
                if exit != nil { break }
                let current = try job(submitted.id)
                if stoppingAt == nil {
                    if current.cancelRequested { cancelled = true; reason = "Cancellation requested." }
                    else if Date().timeIntervalSince(started) >= Double(submitted.timeoutSeconds) { reason = "Worker exceeded its wall-clock budget." }
                    else if try logSize(submitted.id) > Self.outputLimit { reason = "Worker exceeded its 2 MiB output budget." }
                    if reason != nil { worker.signal(SIGTERM); stoppingAt = Date() }
                } else if Date().timeIntervalSince(stoppingAt!) >= 1 { worker.signal(SIGKILL) }
                Thread.sleep(forTimeInterval: 0.05)
            }
            // Reap/stop any ordinary descendants in this worker's process group.
            worker.signal(SIGKILL)
            result.exitCode = exit
            result.workerPID = worker.pid
            result.cancelRequested = try job(submitted.id).cancelRequested
            if result.cancelRequested { cancelled = true }
            try validateBindings(submitted)
            attemptedOutputBinding = true
            result.outputs = try bindingsForOutputs(submitted.id)
            if cancelled { result.transition(.cancelled, reason: reason) }
            else if let reason { result.transition(.failed, reason: reason) }
            else if try logSize(submitted.id) > Self.outputLimit { result.transition(.failed, reason: "Worker exceeded its output budget.") }
            else if exit == 0 {
                guard let stdout = result.outputs["stdout"] else {
                    throw HarnessError("invalid_worker_result", "Worker did not publish a stdout manifest.")
                }
                let data = try Persistence.verifiedData(
                    directory(submitted.id).appendingPathComponent("stdout.log"),
                    matching: stdout, maxBytes: Self.outputLimit)
                try validateUniqueJSONKeys(data)
                guard let manifest = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    throw HarnessError("invalid_worker_result", "Worker did not return one JSON object.")
                }
                if submitted.operation == "workspace_preflight" {
                    guard let name = manifest["name"] as? String, !name.isEmpty,
                          manifest["toolsVersion"] is [String: Any], manifest["targets"] is [Any] else {
                        throw HarnessError("invalid_worker_result", "Preflight worker did not return a SwiftPM manifest object.")
                    }
                } else {
                    guard manifest["operation"] as? String == submitted.operation,
                          manifest["status"] as? String == "completed" else {
                        throw HarnessError("invalid_worker_result", "Worker result does not identify the completed operation.")
                    }
                }
                for (name, url) in try planArtifacts(submitted) {
                    let binding = try Persistence.binding(url, maxBytes: Self.artifactLimit)
                    guard let described = manifest[name] as? [String: Any],
                          let bytes = manifestInteger(described["bytes"]),
                          bytes == binding.bytes else {
                        throw HarnessError("invalid_worker_result", "Worker result has an invalid artifact byte count: \(name)")
                    }
                    guard described["path"] as? String == url.path,
                          described["sha256"] as? String == binding.sha256,
                          bytes == binding.bytes else {
                        throw HarnessError("invalid_worker_result", "Worker result does not bind declared artifact: \(name)")
                    }
                    result.outputs[name] = binding
                }
                result.transition(.completed)
            } else { result.transition(.failed, reason: "Worker exited with status \(exit ?? -1).") }
        } catch {
            if !attemptedOutputBinding {
                attemptedOutputBinding = true
                result.outputs = (try? bindingsForOutputs(submitted.id)) ?? [:]
            }
            result.transition(.failed, reason: error.localizedDescription)
        }
        try transaction { store in
            let i = try index(submitted.id, in: store)
            result.cancelRequested = result.cancelRequested || store.jobs[i].cancelRequested
            if result.cancelRequested && result.state == .completed { result.transition(.cancelled) }
            store.jobs[i] = result
        }
    }

    private func validateBindings(_ job: Job) throws {
        guard let plan = plans[job.operation] else {
            throw HarnessError("request_changed", "The accepted operation is no longer configured.")
        }
        let expectedDigest: String
        if let artifactPaths = job.artifactPaths {
            expectedDigest = Persistence.digest(try Persistence.encode(ExecutionIdentity(
                workspace: workspace.path, operation: job.operation, timeout: job.timeoutSeconds,
                inputs: job.inputs, executable: job.executable, controller: job.controller,
                arguments: job.arguments, environment: job.environment, artifactPaths: artifactPaths)))
        } else {
            expectedDigest = Persistence.digest(try Persistence.encode(LegacyExecutionIdentity(
                workspace: workspace.path, operation: job.operation, timeout: job.timeoutSeconds,
                inputs: job.inputs, executable: job.executable, controller: job.controller,
                arguments: job.arguments, environment: job.environment)))
        }
        let plannedArtifactPaths = plan.artifacts.mapValues { $0.standardizedFileURL.path }
        let checks = [
            ("operation", plans[job.operation] != nil),
            ("timeout", (1...3_600).contains(job.timeoutSeconds)),
            ("request digest", job.requestDigest == expectedDigest),
            ("arguments", job.arguments == plan.arguments),
            ("environment", job.environment == plan.environment),
            ("artifact paths", job.artifactPaths.map { $0 == plannedArtifactPaths } ?? plannedArtifactPaths.isEmpty),
            ("executable path", job.executable.path == plan.executable.path),
            ("input paths", job.inputs.map(\.path) == plan.inputs.map(\.path)),
            ("controller", job.controller == controller),
        ]
        let changed = checks.filter { !$0.1 }.map { $0.0 }
        guard changed.isEmpty else {
            throw HarnessError("request_changed", "Configuration differs from the accepted request: \(changed.joined(separator: ", ")).")
        }
        for binding in job.inputs + [job.executable, job.controller] {
            guard try Persistence.binding(URL(fileURLWithPath: binding.path)) == binding else {
                throw HarnessError("input_changed", "Bound input or executable changed: \(binding.path)")
            }
        }
    }
    private func bindingsForOutputs(_ id: String) throws -> [String: FileBinding] {
        var bindings: [String: FileBinding] = [:]
        var remaining = Self.outputLimit
        for stream in ["stdout", "stderr"] {
            let path = directory(id).appendingPathComponent(stream + ".log")
            if manager.fileExists(atPath: path.path) {
                let binding = try Persistence.binding(path, maxBytes: remaining)
                bindings[stream] = binding
                remaining -= binding.bytes
            }
        }
        return bindings
    }
    private func planArtifacts(_ job: Job) throws -> [String: URL] {
        guard let plan = plans[job.operation],
              (job.artifactPaths ?? [:]) == plan.artifacts.mapValues({ $0.standardizedFileURL.path }) else {
            throw HarnessError("request_changed", "Declared artifact paths differ from the accepted request.")
        }
        return plan.artifacts.mapValues(\.standardizedFileURL)
    }
    private func summary(_ job: Job) throws -> JobSummary {
        if job.state.terminal { try validateRecordedOutputs(job) }
        return JobSummary(job, root: root)
    }
    private func validateRecordedOutputs(_ job: Job) throws {
        let persistedDigest: String
        if let artifactPaths = job.artifactPaths {
            persistedDigest = Persistence.digest(try Persistence.encode(ExecutionIdentity(
                workspace: workspace.path, operation: job.operation, timeout: job.timeoutSeconds,
                inputs: job.inputs, executable: job.executable, controller: job.controller,
                arguments: job.arguments, environment: job.environment, artifactPaths: artifactPaths)))
        } else {
            persistedDigest = Persistence.digest(try Persistence.encode(LegacyExecutionIdentity(
                workspace: workspace.path, operation: job.operation, timeout: job.timeoutSeconds,
                inputs: job.inputs, executable: job.executable, controller: job.controller,
                arguments: job.arguments, environment: job.environment)))
        }
        guard persistedDigest == job.requestDigest else {
            throw HarnessError("artifact_changed", "Persisted output identity differs from the accepted request.")
        }
        var expected = [
            "stdout": directory(job.id).appendingPathComponent("stdout.log").path,
            "stderr": directory(job.id).appendingPathComponent("stderr.log").path,
        ]
        for (name, path) in job.artifactPaths ?? [:] { expected[name] = path }
        let actualNames = Set(job.outputs.keys)
        let expectedNames = Set(expected.keys)
        guard actualNames.isSubset(of: expectedNames),
              job.state != .completed || actualNames == expectedNames else {
            throw HarnessError("artifact_changed", "Recorded output keys differ from the accepted request.")
        }
        for (name, binding) in job.outputs {
            guard binding.path == expected[name] else {
                throw HarnessError("artifact_changed", "Recorded output path changed: \(name)")
            }
            let limit = name == "stdout" || name == "stderr" ? Self.outputLimit : Self.artifactLimit
            guard try Persistence.binding(URL(fileURLWithPath: binding.path), maxBytes: limit) == binding else {
                throw HarnessError("artifact_changed", "Recorded output bytes changed: \(name)")
            }
        }
    }
    private func logSize(_ id: String) throws -> Int {
        try ["stdout", "stderr"].reduce(0) { size, stream in
            let attributes = try manager.attributesOfItem(atPath: directory(id).appendingPathComponent(stream + ".log").path)
            return size + ((attributes[.size] as? NSNumber)?.intValue ?? 0)
        }
    }
    private func directory(_ id: String) -> URL { root.appendingPathComponent("jobs").appendingPathComponent(id) }
    private func index(_ id: String, in store: Store) throws -> Int {
        guard UUID(uuidString: id) != nil, let i = store.jobs.firstIndex(where: { $0.id == id }) else {
            throw HarnessError("unknown_job", "Unknown job ID.")
        }
        return i
    }
    private func transaction<T>(_ body: (inout Store) throws -> T) throws -> T {
        let lock = try FileLock(root.appendingPathComponent("store.lock"))
        defer { withExtendedLifetime(lock) {} }
        let url = root.appendingPathComponent("state.json")
        var store: Store
        if manager.fileExists(atPath: url.path) {
            let data = try Data(contentsOf: url)
            guard data.count <= 16 * 1_024 * 1_024 else { throw HarnessError("invalid_store", "State file exceeds size limit.") }
            let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
            try validateUniqueJSONKeys(data)
            store = try decoder.decode(Store.self, from: data)
            guard store.schemaVersion == 1, store.workspace == workspace.path,
                  store.jobs.allSatisfy({ UUID(uuidString: $0.id) != nil && $0.acceptance == "not_evaluated" }),
                  Set(store.jobs.map(\.id)).count == store.jobs.count,
                  Set(store.jobs.map(\.idempotencyKey)).count == store.jobs.count else {
                throw HarnessError("invalid_store", "State schema or workspace binding differs.")
            }
        } else { store = Store(workspace: workspace.path) }
        let before = try Persistence.encode(store)
        let value = try body(&store)
        let after = try Persistence.encode(store)
        if !manager.fileExists(atPath: url.path) || before != after {
            try Persistence.write(store, to: url)
        }
        return value
    }
}

private func manifestInteger(_ value: Any?) -> Int? {
    guard let number = value as? NSNumber,
          String(cString: number.objCType) != "c",
          number.doubleValue.isFinite,
          number.doubleValue >= 0,
          number.doubleValue <= Double(Int.max),
          number.doubleValue.rounded() == number.doubleValue else { return nil }
    return number.intValue
}

private struct ExecutionIdentity: Encodable {
    let workspace: String
    let operation: String
    let timeout: Int
    let inputs: [FileBinding]
    let executable: FileBinding
    let controller: FileBinding
    let arguments: [String]
    let environment: [String: String]
    let artifactPaths: [String: String]
}

// Schema-1 jobs created before declared artifacts omitted artifactPaths from the
// request digest. They remain deduplicable only when the configured plan still
// declares no artifacts; controller/executable/input bindings still must match.
private struct LegacyExecutionIdentity: Encodable {
    let workspace: String
    let operation: String
    let timeout: Int
    let inputs: [FileBinding]
    let executable: FileBinding
    let controller: FileBinding
    let arguments: [String]
    let environment: [String: String]
}
