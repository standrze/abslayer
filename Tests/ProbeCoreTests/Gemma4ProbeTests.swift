@testable import ProbeCore
import Testing

@Suite("Gemma 4 activation token selection")
struct Gemma4ProbeTests {
    @Test("keeps post-instruction anchored to the prompt boundary")
    func postInstructionIndex() {
        let beforeGeneration = Gemma4Probe.trajectoryTokenIndices(
            promptTokenCount: 7, responseTokenCount: 0)
        let afterTwoResponses = Gemma4Probe.trajectoryTokenIndices(
            promptTokenCount: 7, responseTokenCount: 2)

        #expect(beforeGeneration?.postInstruction == 6)
        #expect(afterTwoResponses?.postInstruction == 6)
    }

    @Test("places first response immediately after the prompt")
    func firstResponseIndex() {
        let beforeGeneration = Gemma4Probe.trajectoryTokenIndices(
            promptTokenCount: 7, responseTokenCount: 0)
        let afterFirstResponse = Gemma4Probe.trajectoryTokenIndices(
            promptTokenCount: 7, responseTokenCount: 1)

        #expect(beforeGeneration?.firstResponse == nil)
        #expect(afterFirstResponse?.firstResponse == 7)
    }

    @Test("places second response immediately after the first")
    func secondResponseIndex() {
        let afterFirstResponse = Gemma4Probe.trajectoryTokenIndices(
            promptTokenCount: 7, responseTokenCount: 1)
        let afterSecondResponse = Gemma4Probe.trajectoryTokenIndices(
            promptTokenCount: 7, responseTokenCount: 2)

        #expect(afterFirstResponse?.secondResponse == nil)
        #expect(afterSecondResponse?.secondResponse == 8)
    }

    @Test("finds the final occurrence of a user-token span")
    func finalOccurrence() {
        let range = Gemma4Probe.lastSubsequenceRange(
            haystack: [9, 1, 2, 3, 8, 1, 2, 3, 7],
            needle: [1, 2, 3])
        #expect(range == 5 ..< 8)
    }

    @Test("rejects absent and empty spans")
    func absentSpan() {
        #expect(Gemma4Probe.lastSubsequenceRange(
            haystack: [1, 2, 3], needle: [2, 4]) == nil)
        #expect(Gemma4Probe.lastSubsequenceRange(
            haystack: [1, 2, 3], needle: []) == nil)
    }
}
