import Foundation
import Testing
@testable import ABSlayerHarness

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
                              plan: WorkerPlan(executable: URL(fileURLWithPath: "/bin/sh"),
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
        executable: URL(fileURLWithPath: "/bin/sh"),
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

@Test func duplicateSubmissionIsStableAndDriftIsRejected() throws {
    let fixture = try Fixture(); defer { fixture.remove() }
    let first = try fixture.submit()
    #expect(try fixture.submit().id == first.id)
    #expect(throws: (any Error).self) { try fixture.submit(timeout: 30) }
    try Data("changed".utf8).write(to: fixture.input)
    #expect(throws: (any Error).self) { try fixture.submit() }
    #expect(try fixture.submit("intentional-new-attempt").id != first.id)
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
    let plan = WorkerPlan(executable: URL(fileURLWithPath: "/bin/sh"), arguments: [], environment: [:], inputs: [])
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
