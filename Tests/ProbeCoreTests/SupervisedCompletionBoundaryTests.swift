import Testing
@testable import ProbeCore

@Suite("Supervised completion boundary")
struct SupervisedCompletionBoundaryTests {
    @Test func includesAnswerAndEndOfTurnSuffix() throws {
        let range = try SupervisedCompletionBoundary.resolve(
            promptTokens: [2, 10, 11, 12],
            completedTokens: [2, 10, 11, 12, 90, 91, 106, 13])

        #expect(range == 4 ..< 8)
    }

    @Test func rejectsNonPrefixAndEmptyCompletion() {
        #expect(throws: SupervisedCompletionBoundaryError.incompatibleTokenization) {
            try SupervisedCompletionBoundary.resolve(
                promptTokens: [2, 10], completedTokens: [2, 99, 90])
        }
        #expect(throws: SupervisedCompletionBoundaryError.incompatibleTokenization) {
            try SupervisedCompletionBoundary.resolve(
                promptTokens: [2, 10], completedTokens: [2, 10])
        }
    }
}
