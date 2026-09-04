@testable import ProbeCore
import Testing

@Suite("Control reference binding")
struct ControlReferenceBinderTests {
    @Test("binds exact untouched control response and preserves metadata")
    func bindsResponse() throws {
        let pair = PromptPair(
            name: "dev-1", contrast: "contrast", control: "control",
            category: "network", source: "source", controlSource: "control-source",
            split: "dev", requestType: "procedure",
            controlReferenceResponse: "stale")
        let response = PromptResult(
            name: "dev-1", contrastResponse: "unused",
            controlResponse: "exact base answer", category: "network",
            contrastPrompt: "contrast", controlPrompt: "control")

        let bound = try ControlReferenceBinder.bind(
            pairs: [pair], responses: [response])

        #expect(bound.count == 1)
        #expect(bound[0].controlReferenceResponse == "exact base answer")
        #expect(bound[0].category == "network")
        #expect(bound[0].source == "source")
        #expect(bound[0].controlSource == "control-source")
        #expect(bound[0].split == "dev")
        #expect(bound[0].requestType == "procedure")
    }

    @Test("rejects detached response text")
    func rejectsDetachedResponse() {
        let pair = PromptPair(name: "dev-1", contrast: "a", control: "b")
        let response = PromptResult(
            name: "dev-1", contrastResponse: "x", controlResponse: "y",
            controlPrompt: "different")
        #expect(throws: ControlReferenceBindingError.self) {
            try ControlReferenceBinder.bind(pairs: [pair], responses: [response])
        }
    }

    @Test("rejects count, order, duplicate, and empty-response errors")
    func rejectsInvalidBindings() {
        let one = PromptPair(name: "one", contrast: "a", control: "b")
        let duplicate = PromptPair(name: "one", contrast: "c", control: "d")
        let valid = PromptResult(
            name: "one", contrastResponse: "x", controlResponse: "answer")
        #expect(throws: ControlReferenceBindingError.self) {
            try ControlReferenceBinder.bind(pairs: [one], responses: [])
        }
        #expect(throws: ControlReferenceBindingError.self) {
            try ControlReferenceBinder.bind(
                pairs: [one],
                responses: [PromptResult(
                    name: "two", contrastResponse: "x", controlResponse: "answer")])
        }
        #expect(throws: ControlReferenceBindingError.self) {
            try ControlReferenceBinder.bind(
                pairs: [one, duplicate], responses: [valid, valid])
        }
        #expect(throws: ControlReferenceBindingError.self) {
            try ControlReferenceBinder.bind(
                pairs: [one],
                responses: [PromptResult(
                    name: "one", contrastResponse: "x", controlResponse: "  ")])
        }
    }
}
