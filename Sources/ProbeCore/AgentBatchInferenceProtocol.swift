import Foundation

/// Closed, versioned protocol used by `abslayer-request --batch`. Requests are
/// canonical JSON so malformed, duplicated-key, and expanded envelopes fail
/// before the model is loaded.
public struct AgentBatchInferenceRequest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let requests: [AgentBatchInferenceRequestItem]

    public init(
        schemaVersion: Int = AgentBatchInferenceProtocol.schemaVersion,
        requests: [AgentBatchInferenceRequestItem]
    ) {
        self.schemaVersion = schemaVersion
        self.requests = requests
    }
}

public struct AgentBatchInferenceRequestItem: Codable, Equatable, Sendable {
    public let id: String
    public let prompt: String

    public init(id: String, prompt: String) {
        self.id = id
        self.prompt = prompt
    }
}

public struct AgentBatchInferenceResponse: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let responses: [AgentBatchInferenceResponseItem]

    public init(
        schemaVersion: Int = AgentBatchInferenceProtocol.schemaVersion,
        responses: [AgentBatchInferenceResponseItem]
    ) {
        self.schemaVersion = schemaVersion
        self.responses = responses
    }
}

public struct AgentBatchInferenceResponseItem: Codable, Equatable, Sendable {
    public let id: String
    public let latencyMilliseconds: Int64
    public let response: String

    public init(id: String, latencyMilliseconds: Int64, response: String) {
        self.id = id
        self.latencyMilliseconds = latencyMilliseconds
        self.response = response
    }
}

public enum AgentBatchInferenceProtocol {
    public static let schemaVersion = 1
    public static let maximumRequests = 4096
    public static let maximumIdentifierUTF8Bytes = 256
    public static let maximumPromptUTF8Bytes = 1_048_576
    public static let maximumEnvelopeBytes = 64 * 1_048_576

    public static func decodeRequest(_ data: Data) throws -> AgentBatchInferenceRequest {
        guard !data.isEmpty else { throw AgentBatchInferenceProtocolError.emptyEnvelope }
        guard data.count <= maximumEnvelopeBytes else {
            throw AgentBatchInferenceProtocolError.envelopeTooLarge(data.count)
        }
        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw AgentBatchInferenceProtocolError.invalidJSON
        }
        guard let envelope = object as? [String: Any], Set(envelope.keys) == [
            "requests", "schemaVersion",
        ] else {
            throw AgentBatchInferenceProtocolError.invalidEnvelopeShape
        }
        guard let rawRequests = envelope["requests"] as? [Any] else {
            throw AgentBatchInferenceProtocolError.invalidEnvelopeShape
        }
        for value in rawRequests {
            guard let request = value as? [String: Any], Set(request.keys) == ["id", "prompt"] else {
                throw AgentBatchInferenceProtocolError.invalidRequestShape
            }
        }
        let decoded: AgentBatchInferenceRequest
        do {
            decoded = try JSONDecoder().decode(AgentBatchInferenceRequest.self, from: data)
        } catch {
            throw AgentBatchInferenceProtocolError.invalidJSON
        }
        try validate(decoded)
        guard try encodeRequest(decoded) == data else {
            throw AgentBatchInferenceProtocolError.nonCanonicalJSON
        }
        return decoded
    }

    public static func encodeRequest(_ request: AgentBatchInferenceRequest) throws -> Data {
        try validate(request)
        return try canonicalEncoder().encode(request)
    }

    public static func encodeResponse(_ response: AgentBatchInferenceResponse) throws -> Data {
        try validate(response)
        return try canonicalEncoder().encode(response)
    }

    private static func validate(_ request: AgentBatchInferenceRequest) throws {
        guard request.schemaVersion == schemaVersion else {
            throw AgentBatchInferenceProtocolError.unsupportedSchema(request.schemaVersion)
        }
        guard !request.requests.isEmpty else {
            throw AgentBatchInferenceProtocolError.emptyRequestBatch
        }
        guard request.requests.count <= maximumRequests else {
            throw AgentBatchInferenceProtocolError.tooManyRequests(request.requests.count)
        }
        var identifiers = Set<String>()
        for item in request.requests {
            guard !item.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AgentBatchInferenceProtocolError.emptyIdentifier
            }
            guard item.id.utf8.count <= maximumIdentifierUTF8Bytes else {
                throw AgentBatchInferenceProtocolError.identifierTooLarge(item.id)
            }
            guard identifiers.insert(item.id).inserted else {
                throw AgentBatchInferenceProtocolError.duplicateIdentifier(item.id)
            }
            guard !item.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AgentBatchInferenceProtocolError.emptyPrompt(item.id)
            }
            guard item.prompt.utf8.count <= maximumPromptUTF8Bytes else {
                throw AgentBatchInferenceProtocolError.promptTooLarge(item.id)
            }
        }
    }

    private static func validate(_ response: AgentBatchInferenceResponse) throws {
        guard response.schemaVersion == schemaVersion else {
            throw AgentBatchInferenceProtocolError.unsupportedSchema(response.schemaVersion)
        }
        guard !response.responses.isEmpty else {
            throw AgentBatchInferenceProtocolError.emptyResponseBatch
        }
        guard response.responses.count <= maximumRequests else {
            throw AgentBatchInferenceProtocolError.tooManyResponses(response.responses.count)
        }
        var identifiers = Set<String>()
        for item in response.responses {
            guard !item.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AgentBatchInferenceProtocolError.emptyIdentifier
            }
            guard item.id.utf8.count <= maximumIdentifierUTF8Bytes else {
                throw AgentBatchInferenceProtocolError.identifierTooLarge(item.id)
            }
            guard identifiers.insert(item.id).inserted else {
                throw AgentBatchInferenceProtocolError.duplicateIdentifier(item.id)
            }
            guard item.latencyMilliseconds >= 0 else {
                throw AgentBatchInferenceProtocolError.negativeLatency(item.id)
            }
        }
    }

    private static func canonicalEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}

public enum AgentBatchInferenceProtocolError: LocalizedError, Equatable {
    case emptyEnvelope, invalidJSON, invalidEnvelopeShape, invalidRequestShape
    case nonCanonicalJSON, emptyRequestBatch, emptyResponseBatch, emptyIdentifier
    case envelopeTooLarge(Int), unsupportedSchema(Int), tooManyRequests(Int), tooManyResponses(Int)
    case identifierTooLarge(String), duplicateIdentifier(String), emptyPrompt(String)
    case promptTooLarge(String), negativeLatency(String)

    public var errorDescription: String? {
        switch self {
        case .emptyEnvelope: "Batch request envelope is empty."
        case .invalidJSON: "Batch request is not valid JSON for the closed protocol."
        case .invalidEnvelopeShape: "Batch request must contain exactly schemaVersion and requests."
        case .invalidRequestShape: "Each batch request must contain exactly id and prompt."
        case .nonCanonicalJSON: "Batch request must be canonical sorted-key UTF-8 JSON."
        case .emptyRequestBatch: "Batch request must contain at least one request."
        case .emptyResponseBatch: "Batch response must contain at least one response."
        case .emptyIdentifier: "Batch request identifiers must be non-empty."
        case .envelopeTooLarge(let size): "Batch request envelope is too large: \(size) bytes."
        case .unsupportedSchema(let version): "Unsupported batch schema version: \(version)."
        case .tooManyRequests(let count): "Batch contains too many requests: \(count)."
        case .tooManyResponses(let count): "Batch contains too many responses: \(count)."
        case .identifierTooLarge(let id): "Batch identifier is too large: \(id)."
        case .duplicateIdentifier(let id): "Duplicate batch identifier: \(id)."
        case .emptyPrompt(let id): "Batch request \(id) has an empty prompt."
        case .promptTooLarge(let id): "Batch request \(id) has a prompt above the size limit."
        case .negativeLatency(let id): "Batch response \(id) has a negative latency."
        }
    }
}
