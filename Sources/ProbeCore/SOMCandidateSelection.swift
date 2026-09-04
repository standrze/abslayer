import Foundation

/// One reproducible, ordered subset of a trained 4x4 SOM lattice.
///
/// `latticeIDs` are row-major IDs in `0 ... 15`. Their order is significant
/// when the downstream intervention composes non-orthogonal projections
/// sequentially.
public struct SOMCandidateSelection: Sendable, Equatable {
    public let label: String
    public let latticeIDs: [Int]
    public let directions: [[Float]]

    public init(label: String, latticeIDs: [Int], directions: [[Float]]) {
        self.label = label
        self.latticeIDs = latticeIDs
        self.directions = directions
    }
}

/// Strict parser and resolver for `ABSLAYER_SOM_SELECTIONS`.
///
/// The optional value is a semicolon-separated list of any of:
///
/// - `coverage`: the extractor's existing deterministic selection (default)
/// - `coverage-reversed`: the same rank-sized coverage subset in reverse order
/// - `occupancy`: occupied neurons sorted by occupancy descending, then mean
///   quantization error ascending, then lattice ID ascending
/// - `occupancy-reversed`: the same rank-sized occupancy subset in reverse order
/// - `ids:<label>=7,6,15,2`: an explicitly labeled, ordered list of lattice IDs
///
/// Explicit lists must contain exactly `rank` IDs. Labels are intentionally
/// restricted to ASCII letters, digits, `.`, `_`, and `-`, must begin with an
/// alphanumeric character, and are compared case-insensitively for uniqueness
/// so they remain safe on case-insensitive filesystems.
public enum SOMCandidateSelector {
    public static let environmentVariable = "ABSLAYER_SOM_SELECTIONS"
    public static let latticeRows = 4
    public static let latticeColumns = 4
    public static let latticeCount = latticeRows * latticeColumns

    public static func resolve(
        result: SOMDirectionResult,
        rank: Int,
        specification: String? = nil
    ) throws -> [SOMCandidateSelection] {
        guard (1 ... latticeCount).contains(rank) else {
            throw SOMCandidateSelectionError.invalidRank(rank)
        }
        try validateResult(result)

        let trimmed = specification?.trimmingCharacters(in: .whitespacesAndNewlines)
        let specs = try parsedSpecs(trimmed?.isEmpty == false ? trimmed! : "coverage")
        var seenLabels = Set<String>()
        var selections = [SOMCandidateSelection]()
        selections.reserveCapacity(specs.count)

        for spec in specs {
            let label: String
            let latticeIDs: [Int]
            switch spec {
            case .coverage:
                label = "coverage"
                latticeIDs = try coverageIDs(result: result, rank: rank)
            case .coverageReversed:
                label = "coverage-reversed"
                latticeIDs = Array(try coverageIDs(result: result, rank: rank).reversed())
            case .occupancy:
                label = "occupancy"
                latticeIDs = try occupancyIDs(result: result, rank: rank)
            case .occupancyReversed:
                label = "occupancy-reversed"
                latticeIDs = Array(try occupancyIDs(result: result, rank: rank).reversed())
            case .explicit(let explicitLabel, let ids):
                label = explicitLabel
                guard ids.count == rank else {
                    throw SOMCandidateSelectionError.explicitRankMismatch(
                        label: label, expected: rank, actual: ids.count)
                }
                latticeIDs = ids
            }

            let foldedLabel = label.lowercased()
            guard seenLabels.insert(foldedLabel).inserted else {
                throw SOMCandidateSelectionError.duplicateLabel(label)
            }
            try validateSelectionIDs(latticeIDs, label: label, result: result)
            selections.append(SOMCandidateSelection(
                label: label,
                latticeIDs: latticeIDs,
                directions: latticeIDs.map { result.candidateDirections[$0] }))
        }
        return selections
    }

    private enum ParsedSpec {
        case coverage
        case coverageReversed
        case occupancy
        case occupancyReversed
        case explicit(label: String, ids: [Int])
    }

    private static func parsedSpecs(_ raw: String) throws -> [ParsedSpec] {
        let tokens = raw.split(separator: ";", omittingEmptySubsequences: false)
        guard !tokens.isEmpty else {
            throw SOMCandidateSelectionError.malformedSpecification(raw)
        }
        return try tokens.map { token in
            let value = token.trimmingCharacters(in: .whitespacesAndNewlines)
            switch value {
            case "coverage":
                return .coverage
            case "coverage-reversed":
                return .coverageReversed
            case "occupancy":
                return .occupancy
            case "occupancy-reversed":
                return .occupancyReversed
            default:
                return try parseExplicit(value, fullSpecification: raw)
            }
        }
    }

    private static func parseExplicit(
        _ value: String, fullSpecification: String
    ) throws -> ParsedSpec {
        guard value.hasPrefix("ids:") else {
            throw SOMCandidateSelectionError.unknownStrategy(value)
        }
        let body = String(value.dropFirst(4))
        let pieces = body.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
        guard pieces.count == 2 else {
            throw SOMCandidateSelectionError.malformedSpecification(fullSpecification)
        }
        let label = String(pieces[0]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard isSafeLabel(label) else {
            throw SOMCandidateSelectionError.unsafeLabel(label)
        }
        let idTokens = pieces[1].split(separator: ",", omittingEmptySubsequences: false)
        guard !idTokens.isEmpty else {
            throw SOMCandidateSelectionError.malformedSpecification(fullSpecification)
        }
        let ids = try idTokens.map { token -> Int in
            let text = token.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty, let id = Int(text) else {
                throw SOMCandidateSelectionError.malformedSpecification(fullSpecification)
            }
            return id
        }
        return .explicit(label: label, ids: ids)
    }

    private static func isSafeLabel(_ label: String) -> Bool {
        guard (1 ... 64).contains(label.utf8.count),
              let first = label.unicodeScalars.first,
              isASCIILetter(first) || isASCIIDigit(first)
        else { return false }
        return label.unicodeScalars.allSatisfy { scalar in
            isASCIILetter(scalar) || isASCIIDigit(scalar)
                || scalar == "." || scalar == "_" || scalar == "-"
        }
    }

    private static func isASCIILetter(_ scalar: Unicode.Scalar) -> Bool {
        (65 ... 90).contains(scalar.value) || (97 ... 122).contains(scalar.value)
    }

    private static func isASCIIDigit(_ scalar: Unicode.Scalar) -> Bool {
        (48 ... 57).contains(scalar.value)
    }

    private static func coverageIDs(
        result: SOMDirectionResult, rank: Int
    ) throws -> [Int] {
        guard result.selectedNeuronIndices.count >= rank else {
            throw SOMCandidateSelectionError.insufficientCandidates(
                strategy: "coverage", requested: rank,
                available: result.selectedNeuronIndices.count)
        }
        return Array(result.selectedNeuronIndices.prefix(rank))
    }

    private static func occupancyIDs(
        result: SOMDirectionResult, rank: Int
    ) throws -> [Int] {
        let usable = (0 ..< latticeCount).filter { isUsable($0, in: result) }
            .sorted { lhs, rhs in
                if result.occupancy[lhs] != result.occupancy[rhs] {
                    return result.occupancy[lhs] > result.occupancy[rhs]
                }
                if result.quantizationErrorByNeuron[lhs]
                    != result.quantizationErrorByNeuron[rhs]
                {
                    return result.quantizationErrorByNeuron[lhs]
                        < result.quantizationErrorByNeuron[rhs]
                }
                return lhs < rhs
            }
        guard usable.count >= rank else {
            throw SOMCandidateSelectionError.insufficientCandidates(
                strategy: "occupancy", requested: rank, available: usable.count)
        }
        return Array(usable.prefix(rank))
    }

    private static func validateResult(_ result: SOMDirectionResult) throws {
        let counts = [
            "candidateDirections": result.candidateDirections.count,
            "occupancy": result.occupancy.count,
            "quantizationErrorByNeuron": result.quantizationErrorByNeuron.count,
        ]
        for (field, count) in counts where count != latticeCount {
            throw SOMCandidateSelectionError.invalidResultCount(
                field: field, expected: latticeCount, actual: count)
        }
        guard let width = result.candidateDirections.first?.count, width > 0,
              result.candidateDirections.allSatisfy({ $0.count == width })
        else {
            throw SOMCandidateSelectionError.inconsistentDirectionWidth
        }
    }

    private static func validateSelectionIDs(
        _ ids: [Int], label: String, result: SOMDirectionResult
    ) throws {
        var seen = Set<Int>()
        for id in ids {
            guard (0 ..< latticeCount).contains(id) else {
                throw SOMCandidateSelectionError.latticeIDOutOfRange(label: label, id: id)
            }
            guard seen.insert(id).inserted else {
                throw SOMCandidateSelectionError.duplicateLatticeID(label: label, id: id)
            }
            guard isUsable(id, in: result) else {
                throw SOMCandidateSelectionError.unusableLatticeID(label: label, id: id)
            }
        }
    }

    fileprivate static func isUsable(
        _ id: Int, in result: SOMDirectionResult
    ) -> Bool {
        guard (0 ..< latticeCount).contains(id), result.occupancy[id] > 0,
              result.quantizationErrorByNeuron[id].isFinite
        else { return false }
        let direction = result.candidateDirections[id]
        guard !direction.isEmpty, direction.allSatisfy(\.isFinite) else { return false }
        let squaredNorm = direction.reduce(Float.zero) { $0 + $1 * $1 }
        return squaredNorm.isFinite && squaredNorm > Float.ulpOfOne
    }
}

/// JSON-safe projection of every SOM candidate and every resolved selection.
/// Empty-neuron `infinity` errors are encoded as `null`, avoiding the special
/// non-conforming-float configuration otherwise required by `JSONEncoder`.
public struct SOMSelectionDiagnostics: Codable, Sendable, Equatable {
    public let latticeRows: Int
    public let latticeColumns: Int
    public let quantizationError: Float?
    public let extractorSelectedLatticeIDs: [Int]
    public let candidates: [SOMCandidateDiagnostic]
    public let selectedSets: [SOMSelectedSetDiagnostic]

    public static func project(
        result: SOMDirectionResult,
        selections: [SOMCandidateSelection]
    ) throws -> Self {
        try SOMCandidateSelector.validateForDiagnostics(result)
        let candidates = (0 ..< SOMCandidateSelector.latticeCount).map { id in
            let error = result.quantizationErrorByNeuron[id]
            return SOMCandidateDiagnostic(
                latticeID: id,
                row: id / SOMCandidateSelector.latticeColumns,
                column: id % SOMCandidateSelector.latticeColumns,
                occupancy: result.occupancy[id],
                quantizationError: error.isFinite ? error : nil,
                usable: SOMCandidateSelector.isUsable(id, in: result))
        }
        return Self(
            latticeRows: SOMCandidateSelector.latticeRows,
            latticeColumns: SOMCandidateSelector.latticeColumns,
            quantizationError: result.quantizationError.isFinite
                ? result.quantizationError : nil,
            extractorSelectedLatticeIDs: result.selectedNeuronIndices,
            candidates: candidates,
            selectedSets: selections.map {
                SOMSelectedSetDiagnostic(label: $0.label, latticeIDs: $0.latticeIDs)
            })
    }
}

public struct SOMCandidateDiagnostic: Codable, Sendable, Equatable {
    public let latticeID: Int
    public let row: Int
    public let column: Int
    public let occupancy: Int
    public let quantizationError: Float?
    public let usable: Bool
}

public struct SOMSelectedSetDiagnostic: Codable, Sendable, Equatable {
    public let label: String
    public let latticeIDs: [Int]
}

public enum SOMCandidateSelectionError: LocalizedError, Equatable {
    case invalidRank(Int)
    case malformedSpecification(String)
    case unknownStrategy(String)
    case unsafeLabel(String)
    case duplicateLabel(String)
    case explicitRankMismatch(label: String, expected: Int, actual: Int)
    case invalidResultCount(field: String, expected: Int, actual: Int)
    case inconsistentDirectionWidth
    case insufficientCandidates(strategy: String, requested: Int, available: Int)
    case latticeIDOutOfRange(label: String, id: Int)
    case duplicateLatticeID(label: String, id: Int)
    case unusableLatticeID(label: String, id: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidRank(let rank):
            "SOM selection rank must be in 1...16, not \(rank)."
        case .malformedSpecification(let value):
            "Malformed ABSLAYER_SOM_SELECTIONS value '\(value)'."
        case .unknownStrategy(let strategy):
            "Unknown SOM selection strategy '\(strategy)'."
        case .unsafeLabel(let label):
            "SOM selection label '\(label)' is not a safe ASCII label."
        case .duplicateLabel(let label):
            "SOM selection label '\(label)' is duplicated."
        case .explicitRankMismatch(let label, let expected, let actual):
            "Explicit SOM selection '\(label)' must contain exactly \(expected) IDs, not \(actual)."
        case .invalidResultCount(let field, let expected, let actual):
            "SOM result \(field) must contain \(expected) lattice entries, not \(actual)."
        case .inconsistentDirectionWidth:
            "SOM candidate directions must have one consistent, nonzero width."
        case .insufficientCandidates(let strategy, let requested, let available):
            "SOM strategy '\(strategy)' requested \(requested) usable candidates but only \(available) are available."
        case .latticeIDOutOfRange(let label, let id):
            "SOM selection '\(label)' contains out-of-range lattice ID \(id); valid IDs are 0...15."
        case .duplicateLatticeID(let label, let id):
            "SOM selection '\(label)' repeats lattice ID \(id)."
        case .unusableLatticeID(let label, let id):
            "SOM selection '\(label)' contains unoccupied or unusable lattice ID \(id)."
        }
    }
}

private extension SOMCandidateSelector {
    static func validateForDiagnostics(_ result: SOMDirectionResult) throws {
        try validateResult(result)
    }
}
