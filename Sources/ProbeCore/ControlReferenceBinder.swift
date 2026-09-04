import Foundation

/// Binds untouched-model control answers to an existing prompt corpus for
/// exact teacher-forced preservation metrics. This operation does not screen
/// or relabel behavior: it only copies response-bound control text after
/// verifying pair identity and, when present, the exact stored control prompt.
public enum ControlReferenceBinder {
    public static func bind(
        pairs: [PromptPair], responses: [PromptResult]
    ) throws -> [PromptPair] {
        guard !pairs.isEmpty else { throw ControlReferenceBindingError.emptyPairs }
        guard pairs.count == responses.count else {
            throw ControlReferenceBindingError.countMismatch(
                pairs: pairs.count, responses: responses.count)
        }

        var seen = Set<String>()
        return try zip(pairs, responses).map { pair, response in
            guard !pair.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  seen.insert(pair.name).inserted
            else { throw ControlReferenceBindingError.duplicateOrEmptyName(pair.name) }
            guard pair.name == response.name else {
                throw ControlReferenceBindingError.nameMismatch(
                    expected: pair.name, actual: response.name)
            }
            if let boundPrompt = response.controlPrompt,
               boundPrompt != pair.control
            {
                throw ControlReferenceBindingError.controlPromptMismatch(pair.name)
            }
            let reference = response.controlResponse.trimmingCharacters(
                in: .whitespacesAndNewlines)
            guard !reference.isEmpty else {
                throw ControlReferenceBindingError.emptyControlResponse(pair.name)
            }
            return PromptPair(
                name: pair.name,
                contrast: pair.contrast,
                control: pair.control,
                category: pair.category,
                source: pair.source,
                controlSource: pair.controlSource,
                split: pair.split,
                requestType: pair.requestType,
                controlReferenceResponse: response.controlResponse)
        }
    }
}

public enum ControlReferenceBindingError: LocalizedError, Equatable {
    case emptyPairs
    case countMismatch(pairs: Int, responses: Int)
    case duplicateOrEmptyName(String)
    case nameMismatch(expected: String, actual: String)
    case controlPromptMismatch(String)
    case emptyControlResponse(String)

    public var errorDescription: String? {
        switch self {
        case .emptyPairs:
            "No prompt pairs were supplied for control-reference binding."
        case .countMismatch(let pairs, let responses):
            "Control-reference binding received \(pairs) pairs and \(responses) responses."
        case .duplicateOrEmptyName(let name):
            "Control-reference binding found an empty or duplicate pair name '\(name)'."
        case .nameMismatch(let expected, let actual):
            "Control-reference binding expected pair '\(expected)', not '\(actual)'."
        case .controlPromptMismatch(let name):
            "Stored control prompt for '\(name)' does not match the source pair."
        case .emptyControlResponse(let name):
            "Stored control response for '\(name)' is empty."
        }
    }
}
