import Foundation
import Testing
@testable import ProbeCore

@Suite("Recorded system-prompt generation condition")
struct PromptResultSystemPromptTests {
    @Test("round-trips the exact system prompt")
    func roundTrip() throws {
        let result = PromptResult(
            name: "case-1", contrastResponse: "contrast",
            controlResponse: "control", systemPrompt: "direct test condition",
            contrastPrompt: "request", controlPrompt: "control request")
        let decoded = try JSONDecoder().decode(
            PromptResult.self, from: JSONEncoder().encode(result))
        #expect(decoded == result)
        #expect(decoded.systemPrompt == "direct test condition")
    }

    @Test("legacy response JSON decodes without a system prompt")
    func legacyDecode() throws {
        let json = #"{"name":"legacy","contrastResponse":"a","controlResponse":"b"}"#
        let decoded = try JSONDecoder().decode(
            PromptResult.self, from: Data(json.utf8))
        #expect(decoded.systemPrompt == nil)
    }
}
