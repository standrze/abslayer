import Foundation
import Testing
@testable import ABSlayerHarness

private let fixtureShell = URL(fileURLWithPath: "/bin/sh").resolvingSymlinksInPath()

private struct Fixture: Sendable {
    let directory: URL
    let input: URL
    let harness: Harness
    init(script: String = "printf '%s' '{\"name\":\"fixture\",\"toolsVersion\":{},\"targets\":[]}'") throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        input = directory.appendingPathComponent("input")
        try Data("original".utf8).write(to: input)
        harness = try Harness(workspace: directory, root: directory.appendingPathComponent("state"),
                              plan: WorkerPlan(executable: fixtureShell,
                                               arguments: ["-c", script],
                                               environment: ["PATH": "/usr/bin:/bin"], inputs: [input]))
    }
    func remove() { try? FileManager.default.removeItem(at: directory) }
    func submit(_ key: String = "attempt-1", timeout: Int = 60) throws -> JobSummary {
        var request = ToolRequest(method: "submit")
        request.idempotencyKey = key; request.timeoutSeconds = timeout
        return try #require(harness.handle(request).job)
    }
    func waitUntilRunning(_ id: String) async throws {
        for _ in 0..<100 {
            if try harness.job(id).workerPID != nil { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        throw HarnessError("test_timeout", "Worker did not start.")
    }
}

private struct LegacyIdentity: Encodable {
    let workspace: String
    let operation: String
    let timeout: Int
    let inputs: [FileBinding]
    let executable: FileBinding
    let controller: FileBinding
    let arguments: [String]
    let environment: [String: String]
}

@Test func requestBoundaryRejectsAmbiguousAndUnsupportedInput() throws {
    for input in [
        "{\"method\":\"submit\",\"method\":\"cancel\"}",
        "{\"method\":\"status\",\"command\":\"anything\"}",
        "{\"method\":\"submit\",\"timeoutSeconds\":true}",
        "{\"method\":\"submit\",\"timeoutSeconds\":0}",
        "{\"method\":\"evidence\",\"limit\":99999}",
        "{\"method\":\"evidence\",\"stream\":\"../../secret\"}",
    ] {
        #expect(throws: (any Error).self) { try ToolRequest.decode(Data(input.utf8)) }
    }
    #expect(try ToolRequest.decode(Data("{\"method\":\"status\"}".utf8)).method == "status")
}

@Test func namedConfiguredOperationIsDiscoverableAndCompletes() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let input = directory.appendingPathComponent("input")
    try Data("bound".utf8).write(to: input)
    let plan = WorkerPlan(
        executable: fixtureShell,
        arguments: ["-c", "printf '%s' '{\"operation\":\"fixture_operation\",\"status\":\"completed\"}'"],
        environment: ["PATH": "/usr/bin:/bin"], inputs: [input])
    let harness = try Harness(
        workspace: directory, root: directory.appendingPathComponent("state"),
        plans: ["fixture_operation": plan])
    #expect(try harness.handle(ToolRequest(method: "capabilities")).operations == ["fixture_operation"])
    var request = ToolRequest(method: "submit")
    request.operation = "fixture_operation"
    request.idempotencyKey = "named-operation"
    request.timeoutSeconds = 300
    let job = try #require(harness.handle(request).job)
    try harness.drain()
    #expect(try harness.job(job.id).state == .completed)
    #expect(try harness.job(job.id).acceptance == "not_evaluated")
}

@Test func configuredExecutableArgumentZeroIsAppliedButNotInherited() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let script = "test \"$0\" = fixture-shell; test -z \"$_ABSLAYER_EXECUTABLE_ARGV0\"; " +
        "printf '%s' '{\"operation\":\"fixture_operation\",\"status\":\"completed\"}'"
    let plan = WorkerPlan(executable: fixtureShell, arguments: ["-c", script],
                          environment: ["PATH": "/usr/bin:/bin",
                                        "_ABSLAYER_EXECUTABLE_ARGV0": "fixture-shell"], inputs: [])
    let harness = try Harness(workspace: directory, root: directory.appendingPathComponent("state"),
                              plans: ["fixture_operation": plan])
    var request = ToolRequest(method: "submit")
    request.operation = "fixture_operation"; request.idempotencyKey = "custom-argv-zero"
    let job = try #require(harness.handle(request).job)
    try harness.drain()
    #expect(try harness.job(job.id).state == .completed)
}

@Test func declaredArtifactIsBoundAndTamperingBreaksStatus() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let input = directory.appendingPathComponent("input")
    let artifact = directory.appendingPathComponent("control-vector.gguf")
    try Data("bound".utf8).write(to: input)
    let manifest: [String: Any] = [
        "operation": "fixture_operation", "status": "completed",
        "vector": ["path": artifact.path,
                   "sha256": "21b7880109588a8c2561a6cdd2c94d4b42e17538055ef1c80f10be711b38edb1",
                   "bytes": 12],
    ]
    let manifestText = String(decoding: try JSONSerialization.data(withJSONObject: manifest), as: UTF8.self)
    func shellQuote(_ value: String) -> String { "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'" }
    let script = "printf '%s' 'vector-bytes' > \(shellQuote(artifact.path)); printf '%s' \(shellQuote(manifestText))"
    let plan = WorkerPlan(executable: fixtureShell, arguments: ["-c", script],
                          environment: ["PATH": "/usr/bin:/bin"], inputs: [input],
                          artifacts: ["vector": artifact])
    let harness = try Harness(workspace: directory, root: directory.appendingPathComponent("state"),
                              plans: ["fixture_operation": plan])
    var submit = ToolRequest(method: "submit")
    submit.operation = "fixture_operation"; submit.idempotencyKey = "artifact"
    let summary = try #require(harness.handle(submit).job)
    try harness.drain()
    let completed = try harness.job(summary.id)
    #expect(completed.state == .completed)
    #expect(completed.outputs["vector"]?.sha256 == "21b7880109588a8c2561a6cdd2c94d4b42e17538055ef1c80f10be711b38edb1")
    var status = ToolRequest(method: "status"); status.jobID = summary.id
    #expect(try harness.handle(status).job?.state == .completed)
    let state = directory.appendingPathComponent("state/state.json")
    let originalState = try Data(contentsOf: state)
    let copied = directory.appendingPathComponent("copied-vector.gguf")
    try FileManager.default.copyItem(at: artifact, to: copied)
    var object = try #require(JSONSerialization.jsonObject(with: originalState) as? [String: Any])
    var jobs = try #require(object["jobs"] as? [[String: Any]])
    var paths = try #require(jobs[0]["artifactPaths"] as? [String: String])
    var outputs = try #require(jobs[0]["outputs"] as? [String: [String: Any]])
    paths["vector"] = copied.path
    outputs["vector"]?["path"] = copied.path
    jobs[0]["artifactPaths"] = paths; jobs[0]["outputs"] = outputs; object["jobs"] = jobs
    try JSONSerialization.data(withJSONObject: object).write(to: state)
    #expect(throws: (any Error).self) { try harness.handle(status) }
    try originalState.write(to: state)
    try Data("tampered".utf8).write(to: artifact)
    #expect(throws: (any Error).self) { try harness.handle(status) }
}

@Test func falseDeclaredArtifactHashCannotComplete() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let artifact = directory.appendingPathComponent("artifact")
    let manifest = "{\"operation\":\"fixture_operation\",\"status\":\"completed\",\"vector\":{\"path\":\"\(artifact.path)\",\"sha256\":\"\(String(repeating: "0", count: 64))\",\"bytes\":12}}"
    let escaped = manifest.replacingOccurrences(of: "'", with: "'\"'\"'")
    let path = artifact.path.replacingOccurrences(of: "'", with: "'\"'\"'")
    let plan = WorkerPlan(executable: fixtureShell,
                          arguments: ["-c", "printf '%s' 'vector-bytes' > '\(path)'; printf '%s' '\(escaped)'"],
                          environment: ["PATH": "/usr/bin:/bin"], inputs: [], artifacts: ["vector": artifact])
    let harness = try Harness(workspace: directory, root: directory.appendingPathComponent("state"),
                              plans: ["fixture_operation": plan])
    var submit = ToolRequest(method: "submit")
    submit.operation = "fixture_operation"; submit.idempotencyKey = "spoof"
    let summary = try #require(harness.handle(submit).job)
    try harness.drain()
    #expect(try harness.job(summary.id).state == .failed)
}

@Test func persistedOutputPathCannotBeRedirectedAcrossSummaryMethods() throws {
    let fixture = try Fixture(); defer { fixture.remove() }
    let submitted = try fixture.submit()
    try fixture.harness.drain()
    let state = fixture.directory.appendingPathComponent("state/state.json")
    let copied = fixture.directory.appendingPathComponent("copied-stdout")
    let original = fixture.directory.appendingPathComponent("state/jobs/\(submitted.id)/stdout.log")
    try FileManager.default.copyItem(at: original, to: copied)
    var object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: state)) as? [String: Any])
    var jobs = try #require(object["jobs"] as? [[String: Any]])
    var outputs = try #require(jobs[0]["outputs"] as? [String: [String: Any]])
    outputs["stdout"]?["path"] = copied.path
    jobs[0]["outputs"] = outputs
    object["jobs"] = jobs
    try JSONSerialization.data(withJSONObject: object).write(to: state)
    var status = ToolRequest(method: "status"); status.jobID = submitted.id
    #expect(throws: (any Error).self) { try fixture.harness.handle(status) }
    #expect(throws: (any Error).self) { try fixture.harness.handle(ToolRequest(method: "status")) }
    var cancel = ToolRequest(method: "cancel"); cancel.jobID = submitted.id
    #expect(throws: (any Error).self) { try fixture.harness.handle(cancel) }
    #expect(throws: (any Error).self) { try fixture.submit() }
}

@Test func fractionalArtifactByteCountAndSymlinkCannotComplete() throws {
    for (suffix, artifactCommand, claimedBytes) in [
        ("fraction", "printf '%s' 'vector-bytes' > artifact", "12.5"),
        ("boolean", "printf '%s' 'v' > artifact", "true"),
        ("symlink", "printf '%s' 'vector-bytes' > target; ln -s target artifact", "12"),
    ] {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let artifact = directory.appendingPathComponent("artifact")
        let manifest = "{\"operation\":\"fixture_operation\",\"status\":\"completed\",\"vector\":{\"path\":\"\(artifact.path)\",\"sha256\":\"21b7880109588a8c2561a6cdd2c94d4b42e17538055ef1c80f10be711b38edb1\",\"bytes\":\(claimedBytes)}}"
        let escaped = manifest.replacingOccurrences(of: "'", with: "'\"'\"'")
        let script = "cd '\(directory.path)'; \(artifactCommand); printf '%s' '\(escaped)'"
        let plan = WorkerPlan(executable: fixtureShell, arguments: ["-c", script],
                              environment: ["PATH": "/usr/bin:/bin"], inputs: [], artifacts: ["vector": artifact])
        let harness = try Harness(workspace: directory, root: directory.appendingPathComponent("state"),
                                  plans: ["fixture_operation": plan])
        var submit = ToolRequest(method: "submit")
        submit.operation = "fixture_operation"; submit.idempotencyKey = suffix
        let summary = try #require(harness.handle(submit).job)
        try harness.drain()
        #expect(try harness.job(summary.id).state == .failed)
    }
}

@Test func bindingRejectsGrowthAndSameSizeMutationWhileHashing() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    let growing = directory.appendingPathComponent("growing")
    try Data("1234".utf8).write(to: growing)
    var appended = false
    #expect(throws: (any Error).self) {
        try Persistence.binding(growing, maxBytes: 4) { _ in
            guard !appended else { return }
            appended = true
            let handle = try FileHandle(forWritingTo: growing)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: Data("5".utf8))
        }
    }

    let changing = directory.appendingPathComponent("changing")
    try Data("before".utf8).write(to: changing)
    var changed = false
    #expect(throws: (any Error).self) {
        try Persistence.binding(changing, maxBytes: 6) { _ in
            guard !changed else { return }
            changed = true
            let handle = try FileHandle(forWritingTo: changing)
            defer { try? handle.close() }
            try handle.seek(toOffset: 0)
            try handle.write(contentsOf: Data("after!".utf8))
            try handle.synchronize()
        }
    }
}

@Test func verifiedEvidenceRejectsAPathSwapDuringItsDescriptorRead() throws {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let evidence = directory.appendingPathComponent("evidence")
    let moved = directory.appendingPathComponent("opened-evidence")
    let replacement = directory.appendingPathComponent("replacement")
    let original = Data(repeating: 0x41, count: 1_048_577)
    try original.write(to: evidence)
    try Data("unbound".utf8).write(to: replacement)
    let expected = try Persistence.binding(evidence, maxBytes: 2 * 1_024 * 1_024)
    var swapped = false
    #expect(throws: (any Error).self) {
        try Persistence.verifiedData(evidence, matching: expected,
                                     maxBytes: 2 * 1_024 * 1_024) { count in
            guard count == 1_048_576, !swapped else { return }
            swapped = true
            try FileManager.default.moveItem(at: evidence, to: moved)
            try FileManager.default.createSymbolicLink(at: evidence, withDestinationURL: replacement)
        }
    }
    #expect(throws: (any Error).self) { try Persistence.binding(evidence) }
}

@Test func oversizedLogsAreNeverPersistedAsBoundEvidence() throws {
    let fixture = try Fixture(script: "head -c 3145728 /dev/zero")
    defer { fixture.remove() }
    let submitted = try fixture.submit()
    try fixture.harness.drain()
    let job = try fixture.harness.job(submitted.id)
    #expect(job.state == .failed)
    #expect(job.outputs.values.reduce(0) { $0 + $1.bytes } <= 2 * 1_024 * 1_024)
}

@Test func duplicateSubmissionIsStableAndDriftIsRejected() throws {
    let fixture = try Fixture(); defer { fixture.remove() }
    let first = try fixture.submit()
    #expect(try fixture.submit().id == first.id)
    #expect(throws: (any Error).self) { try fixture.submit(timeout: 30) }
    try Data("changed".utf8).write(to: fixture.input)
    #expect(throws: (any Error).self) { try fixture.submit() }
    #expect(try fixture.submit("intentional-new-attempt").id != first.id)
}

@Test func legacyNoArtifactDigestRemainsDeduplicableAndRunnable() throws {
    let fixture = try Fixture(); defer { fixture.remove() }
    let submitted = try fixture.submit()
    let job = try fixture.harness.job(submitted.id)
    let identity = LegacyIdentity(workspace: fixture.directory.path, operation: job.operation,
                                  timeout: job.timeoutSeconds, inputs: job.inputs,
                                  executable: job.executable, controller: job.controller,
                                  arguments: job.arguments, environment: job.environment)
    let legacyDigest = Persistence.digest(try Persistence.encode(identity))
    let state = fixture.directory.appendingPathComponent("state/state.json")
    var object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: state)) as? [String: Any])
    var jobs = try #require(object["jobs"] as? [[String: Any]])
    jobs[0].removeValue(forKey: "artifactPaths")
    jobs[0]["requestDigest"] = legacyDigest
    object["jobs"] = jobs
    try JSONSerialization.data(withJSONObject: object).write(to: state)
    #expect(try fixture.submit().id == submitted.id)
    try fixture.harness.drain()
    #expect(try fixture.harness.job(submitted.id).state == .completed)
}

@Test func completedJobHasBoundEvidenceButNoAcceptance() throws {
    let fixture = try Fixture(); defer { fixture.remove() }
    let first = try fixture.submit()
    try fixture.harness.drain()
    let job = try fixture.harness.job(first.id)
    #expect(job.state == .completed)
    #expect(job.acceptance == "not_evaluated")
    #expect(job.events.map(\.state) == [.queued, .running, .completed])
    #expect(job.outputs["stdout"]?.sha256.count == 64)
    var request = ToolRequest(method: "evidence"); request.jobID = first.id; request.limit = 8
    let response = try fixture.harness.handle(request)
    #expect(response.text?.utf8.count == 8)
    #expect(response.nextOffset == 8)
    let output = try #require(job.outputs["stdout"])
    try Data("tampered".utf8).write(to: URL(fileURLWithPath: output.path))
    #expect(throws: (any Error).self) { try fixture.harness.handle(request) }
}

@Test func changedInputFailsBeforeWorkerStarts() throws {
    let fixture = try Fixture(); defer { fixture.remove() }
    let first = try fixture.submit()
    try Data("changed".utf8).write(to: fixture.input)
    try fixture.harness.drain()
    let job = try fixture.harness.job(first.id)
    #expect(job.state == .failed)
    #expect(job.workerPID == nil)
    #expect(job.failure?.contains("changed") == true)
}

@Test func queuedCancellationNeverLaunches() throws {
    let fixture = try Fixture(); defer { fixture.remove() }
    let first = try fixture.submit()
    var request = ToolRequest(method: "cancel"); request.jobID = first.id
    #expect(try fixture.harness.handle(request).job?.state == .cancelled)
    try fixture.harness.drain()
    #expect(try fixture.harness.job(first.id).workerPID == nil)
}

@Test func runningCancellationStopsWorkerBeforePublishing() async throws {
    let fixture = try Fixture(script: "sleep 10; printf '{}'"); defer { fixture.remove() }
    let first = try fixture.submit()
    let runner = Task.detached { try fixture.harness.drain() }
    try await fixture.waitUntilRunning(first.id)
    var request = ToolRequest(method: "cancel"); request.jobID = first.id
    _ = try fixture.harness.handle(request)
    try await runner.value
    let job = try fixture.harness.job(first.id)
    #expect(job.state == .cancelled)
    #expect(job.exitCode != nil)
}

@Test func timeoutStopsWorkerAndDoesNotPass() throws {
    let fixture = try Fixture(script: "trap '' TERM; sleep 10; printf '{}'"); defer { fixture.remove() }
    let first = try fixture.submit(timeout: 1)
    let start = Date()
    try fixture.harness.drain()
    let job = try fixture.harness.job(first.id)
    #expect(job.state == .failed)
    #expect(job.failure?.contains("wall-clock") == true)
    #expect(Date().timeIntervalSince(start) < 5)
}

@Test func secondSupervisorCannotOverlapTheFirst() async throws {
    let fixture = try Fixture(script: "sleep 0.3; printf '%s' '{\"name\":\"fixture\",\"toolsVersion\":{},\"targets\":[]}'")
    defer { fixture.remove() }
    let first = try fixture.submit("one")
    let second = try fixture.submit("two")
    let runner = Task.detached { try fixture.harness.drain() }
    try await fixture.waitUntilRunning(first.id)
    try fixture.harness.drain()
    #expect(try fixture.harness.job(second.id).state == .queued)
    try await runner.value
    #expect(try fixture.harness.job(first.id).state == .completed)
    #expect(try fixture.harness.job(second.id).state == .completed)
}

@Test func workerFailureAndMalformedOutputRemainFailures() throws {
    for script in ["exit 7", "printf 'not json'", "printf '{}'", "dd if=/dev/zero bs=4096 count=520 2>/dev/null"] {
        let fixture = try Fixture(script: script); defer { fixture.remove() }
        let first = try fixture.submit()
        try fixture.harness.drain()
        #expect(try fixture.harness.job(first.id).state == .failed)
    }
}

@Test func storeCannotBeReboundOrSilentlyReinitialized() throws {
    let fixture = try Fixture(); defer { fixture.remove() }
    let first = try fixture.submit()
    let root = fixture.directory.appendingPathComponent("state")
    let plan = WorkerPlan(executable: fixtureShell, arguments: [], environment: [:], inputs: [])
    #expect(throws: (any Error).self) { try Harness(workspace: fixture.directory.appendingPathComponent("elsewhere"), root: root, plan: plan) }
    #expect(try fixture.harness.job(first.id).state == .queued)
    try Data("corrupt".utf8).write(to: root.appendingPathComponent("state.json"))
    #expect(throws: (any Error).self) { try fixture.harness.job(first.id) }
}

@Test func changedPersistedCommandCannotExecute() throws {
    let fixture = try Fixture(); defer { fixture.remove() }
    let first = try fixture.submit()
    let state = fixture.directory.appendingPathComponent("state/state.json")
    var object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: state)) as? [String: Any])
    var jobs = try #require(object["jobs"] as? [[String: Any]])
    jobs[0]["arguments"] = ["-c", "printf '{\"unexpected\":true}'"]
    object["jobs"] = jobs
    try JSONSerialization.data(withJSONObject: object).write(to: state)
    try fixture.harness.drain()
    let job = try fixture.harness.job(first.id)
    #expect(job.state == .failed)
    #expect(job.workerPID == nil)
    #expect(job.failure?.contains("accepted request") == true)
}
