import Foundation

public struct HarnessError: Error, LocalizedError, Sendable {
    public let code: String
    public let message: String
    public init(_ code: String, _ message: String) {
        self.code = code; self.message = message
    }
    public var errorDescription: String? { message }
}

public enum JobState: String, Codable, Sendable {
    case queued, running, completed, failed, cancelled, interrupted
    public var terminal: Bool { self != .queued && self != .running }
}

public struct FileBinding: Codable, Equatable, Sendable {
    public var path: String
    public var sha256: String
    public var bytes: Int
}

public struct JobEvent: Codable, Sendable {
    public var state: JobState
    public var time: Date
    public var reason: String?
}

public struct Job: Codable, Sendable {
    public var id: String
    public var idempotencyKey: String
    public var requestDigest: String
    public var operation: String
    public var timeoutSeconds: Int
    public var state: JobState
    public var createdAt: Date
    public var updatedAt: Date
    public var cancelRequested: Bool
    public var inputs: [FileBinding]
    public var executable: FileBinding
    public var controller: FileBinding
    public var arguments: [String]
    public var environment: [String: String]
    public var workerPID: Int32?
    public var exitCode: Int32?
    public var failure: String?
    public var outputs: [String: FileBinding]
    public var events: [JobEvent]
    // Execution status must never be read as model/candidate acceptance.
    public private(set) var acceptance = "not_evaluated"

    mutating func transition(_ state: JobState, reason: String? = nil) {
        self.state = state; updatedAt = Date(); failure = reason
        events.append(JobEvent(state: state, time: updatedAt, reason: reason))
    }
}

struct Store: Codable {
    var schemaVersion = 1
    var workspace: String
    var jobs: [Job] = []
}

public struct ToolRequest: Sendable {
    public var method: String
    public var operation: String = "workspace_preflight"
    public var idempotencyKey: String = ""
    public var jobID: String = ""
    public var timeoutSeconds = 60
    public var stream = "stdout"
    public var offset = 0
    public var limit = 4096

    public init(method: String) { self.method = method }

    public static func decode(_ data: Data) throws -> Self {
        guard data.count <= 16_384,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let method = object["method"] as? String else {
            throw HarnessError("invalid_request", "Expected a JSON object with a method (maximum 16 KiB).")
        }
        try validateUniqueJSONKeys(data)
        let keys: [String: Set<String>] = [
            "capabilities": ["method"],
            "submit": ["method", "operation", "idempotencyKey", "timeoutSeconds"],
            "status": ["method", "jobID"],
            "cancel": ["method", "jobID"],
            "recover": ["method"],
            "evidence": ["method", "jobID", "stream", "offset", "limit"],
        ]
        guard let allowed = keys[method], Set(object.keys).isSubset(of: allowed) else {
            throw HarnessError("invalid_request", "Unknown method or unexpected request fields.")
        }
        var request = Self(method: method)
        func string(_ key: String, _ fallback: String) throws -> String {
            guard let value = object[key] else { return fallback }
            guard let text = value as? String else { throw HarnessError("invalid_request", "\(key) must be a string.") }
            return text
        }
        func integer(_ key: String, _ fallback: Int) throws -> Int {
            guard let value = object[key] else { return fallback }
            guard let number = value as? NSNumber, String(cString: number.objCType) != "c",
                  number.doubleValue.isFinite, number.doubleValue.rounded() == number.doubleValue,
                  number.doubleValue >= 0, number.doubleValue <= 16_777_216 else {
                throw HarnessError("invalid_request", "\(key) must be a bounded integer.")
            }
            return number.intValue
        }
        request.operation = try string("operation", request.operation)
        request.idempotencyKey = try string("idempotencyKey", "")
        request.jobID = try string("jobID", "")
        request.stream = try string("stream", "stdout")
        request.timeoutSeconds = try integer("timeoutSeconds", 60)
        request.offset = try integer("offset", 0)
        request.limit = try integer("limit", 4096)
        guard (1...120).contains(request.timeoutSeconds), (1...8192).contains(request.limit),
              ["stdout", "stderr"].contains(request.stream) else {
            throw HarnessError("invalid_request", "Timeout must be 1–120 seconds, evidence limit 1–8192 bytes, stream stdout or stderr.")
        }
        return request
    }
}

public struct ToolResponse: Encodable {
    public var schemaVersion = 1
    public var ok = true
    public var job: JobSummary?
    public var jobs: [JobSummary]?
    public var operations: [String]?
    public var message: String?
    public var text: String?
    public var nextOffset: Int?
    public var artifact: FileBinding?
    public var code: String?
    public var needsDrain = false
    public init() {}
}

public struct JobSummary: Encodable {
    public let id: String
    public let operation: String
    public let state: JobState
    public let cancelRequested: Bool
    public let failure: String?
    public let exitCode: Int32?
    public let requestDigest: String
    public let acceptance: String
    public let outputs: [String: FileBinding]
    public let stateFile: String
    init(_ job: Job, root: URL) {
        id = job.id; operation = job.operation; state = job.state
        cancelRequested = job.cancelRequested; failure = job.failure; exitCode = job.exitCode
        requestDigest = job.requestDigest; acceptance = job.acceptance; outputs = job.outputs
        stateFile = root.appendingPathComponent("state.json").path
    }
}

// JSONSerialization otherwise silently accepts repeated keys. Syntax is checked
// by the caller; scan string tokens to compare decoded keys in each object.
func validateUniqueJSONKeys(_ data: Data) throws {
    let bytes = Array(data)
    var stack: [Set<String>?] = []
    var i = 0
    while i < bytes.count {
        switch bytes[i] {
        case 123: stack.append([])
        case 91: stack.append(nil)
        case 125, 93: if !stack.isEmpty { stack.removeLast() }
        case 34:
            let start = i
            i += 1
            while i < bytes.count {
                if bytes[i] == 92 { i += 2; continue }
                if bytes[i] == 34 { break }
                i += 1
            }
            guard i < bytes.count else { throw HarnessError("invalid_request", "Invalid JSON string.") }
            var next = i + 1
            while next < bytes.count && [9, 10, 13, 32].contains(bytes[next]) { next += 1 }
            if next < bytes.count && bytes[next] == 58, !stack.isEmpty, var keys = stack[stack.count - 1] {
                let key = try JSONDecoder().decode(String.self, from: Data(bytes[start...i]))
                guard keys.insert(key).inserted else { throw HarnessError("invalid_request", "Duplicate JSON key: \(key)") }
                stack[stack.count - 1] = keys
            }
        default: break
        }
        i += 1
    }
}
