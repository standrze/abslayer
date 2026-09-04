import Foundation

/// Resolves the exact assistant suffix supervised by prefix-masked SFT.
///
/// `promptTokens` must be the deployed generation prompt, including its
/// assistant header. `completedTokens` must render that same prompt followed by
/// the assistant answer and its end-of-turn framing. The returned range keeps
/// the end-of-turn token supervised so generation learns to terminate.
public enum SupervisedCompletionBoundary {
    public static func resolve(
        promptTokens: [Int], completedTokens: [Int]
    ) throws -> Range<Int> {
        guard !promptTokens.isEmpty,
              completedTokens.starts(with: promptTokens),
              completedTokens.count > promptTokens.count
        else { throw SupervisedCompletionBoundaryError.incompatibleTokenization }
        return promptTokens.count ..< completedTokens.count
    }
}

public enum SupervisedCompletionBoundaryError: LocalizedError, Equatable {
    case incompatibleTokenization

    public var errorDescription: String? {
        switch self {
        case .incompatibleTokenization:
            "Completed chat tokens do not extend the deployed generation prompt."
        }
    }
}
