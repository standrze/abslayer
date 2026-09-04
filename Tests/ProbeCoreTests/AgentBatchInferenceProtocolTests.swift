import Foundation
import Testing
@testable import ProbeCore

@Suite("Agent batch inference protocol")
struct AgentBatchInferenceProtocolTests {
    @Test func roundTripsCanonicalRequest() throws {
        let request = AgentBatchInferenceRequest(requests: [
            AgentBatchInferenceRequestItem(id: "case-01", prompt: "first prompt"),
            AgentBatchInferenceRequestItem(id: "case-02", prompt: "second prompt"),
        ])
        let data = try AgentBatchInferenceProtocol.encodeRequest(request)
        #expect(String(decoding: data, as: UTF8.self) ==
            #"{"requests":[{"id":"case-01","prompt":"first prompt"},{"id":"case-02","prompt":"second prompt"}],"schemaVersion":1}"#)
        #expect(try AgentBatchInferenceProtocol.decodeRequest(data) == request)
    }

    @Test func rejectsNonCanonicalAndExpandedEnvelopes() throws {
        let nonCanonical = Data(#"{ "requests": [{"id":"case","prompt":"prompt"}], "schemaVersion":1}"#.utf8)
        #expect(throws: AgentBatchInferenceProtocolError.nonCanonicalJSON) {
            try AgentBatchInferenceProtocol.decodeRequest(nonCanonical)
        }

        let expanded = Data(#"{"extra":false,"requests":[{"id":"case","prompt":"prompt"}],"schemaVersion":1}"#.utf8)
        #expect(throws: AgentBatchInferenceProtocolError.invalidEnvelopeShape) {
            try AgentBatchInferenceProtocol.decodeRequest(expanded)
        }
    }

    @Test func rejectsUnsupportedSchemaAndDuplicateIDs() throws {
        let unsupported = Data(#"{"requests":[{"id":"case","prompt":"prompt"}],"schemaVersion":2}"#.utf8)
        #expect(throws: AgentBatchInferenceProtocolError.unsupportedSchema(2)) {
            try AgentBatchInferenceProtocol.decodeRequest(unsupported)
        }

        let duplicate = Data(#"{"requests":[{"id":"case","prompt":"one"},{"id":"case","prompt":"two"}],"schemaVersion":1}"#.utf8)
        #expect(throws: AgentBatchInferenceProtocolError.duplicateIdentifier("case")) {
            try AgentBatchInferenceProtocol.decodeRequest(duplicate)
        }
    }

    @Test func encodesClosedCanonicalResponseEnvelope() throws {
        let response = AgentBatchInferenceResponse(responses: [
            AgentBatchInferenceResponseItem(
                id: "case-01", latencyMilliseconds: 123,
                response: #"{"action":"blocked"}"#),
        ])
        let data = try AgentBatchInferenceProtocol.encodeResponse(response)
        #expect(String(decoding: data, as: UTF8.self) ==
            #"{"responses":[{"id":"case-01","latencyMilliseconds":123,"response":"{\"action\":\"blocked\"}"}],"schemaVersion":1}"#)
    }
}
