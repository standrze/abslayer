import Foundation

/// Provenance for a deterministic, name-based train/dev subtraction.
///
/// The manifest intentionally contains no timestamp: identical inputs and
/// source digests produce byte-for-byte identical encoded metadata when the
/// caller uses sorted-key JSON output.
public struct PromptPairSubtractionManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let artifactRole: String
    public let operation: String
    public let trainingSourcePath: String
    public let trainingSourceSHA256: String
    public let exclusionSourcePath: String
    public let exclusionSourceSHA256: String
    public let inputTrainingPairs: Int
    public let inputExclusionPairs: Int
    public let removedPairs: Int
    public let outputPairs: Int
    public let removedNames: [String]
    public let trainingNamesSHA256: String
    public let exclusionNamesSHA256: String
    public let outputNamesSHA256: String
    public let preservesTrainingOrder: Bool
    public let preservesRetainedPairContent: Bool
    public let frozenAuditAccessPermitted: Bool
}

public struct PromptPairSubtractionResult: Equatable, Sendable {
    public let pairs: [PromptPair]
    public let manifest: PromptPairSubtractionManifest
}

/// Removes every dev name from a training PromptFile without modifying or
/// reordering retained rows. Recorded split metadata, when present, must agree
/// with the input's declared role; legacy PromptFiles without split metadata
/// remain supported.
public enum PromptPairSubtractor {
    public static func subtract(
        trainingPairs: [PromptPair],
        excludingNamesIn exclusionPairs: [PromptPair],
        trainingSourcePath: String,
        trainingSourceSHA256: String,
        exclusionSourcePath: String,
        exclusionSourceSHA256: String
    ) throws -> PromptPairSubtractionResult {
        guard !trainingPairs.isEmpty else {
            throw PromptPairSubtractionError.emptyTrainingInput
        }
        guard !exclusionPairs.isEmpty else {
            throw PromptPairSubtractionError.emptyExclusionInput
        }
        guard !trainingSourcePath.trimmingCharacters(
            in: .whitespacesAndNewlines).isEmpty,
              !exclusionSourcePath.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty,
              !trainingSourceSHA256.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty,
              !exclusionSourceSHA256.trimmingCharacters(
                in: .whitespacesAndNewlines).isEmpty
        else { throw PromptPairSubtractionError.missingProvenance }

        let trainingNames = try validatedNames(
            trainingPairs, role: .training)
        let exclusionNames = try validatedNames(
            exclusionPairs, role: .exclusion)
        let trainingNameSet = Set(trainingNames)
        let exclusionNameSet = Set(exclusionNames)
        let missing = exclusionNameSet.subtracting(trainingNameSet).sorted()
        guard missing.isEmpty else {
            throw PromptPairSubtractionError.exclusionNamesMissingFromTraining(
                missing)
        }

        let retained = zip(trainingPairs, trainingNames).compactMap {
            pair, name in exclusionNameSet.contains(name) ? nil : pair
        }
        guard !retained.isEmpty else {
            throw PromptPairSubtractionError.emptyOutput
        }
        let outputNames = try validatedNames(retained, role: .output)
        let canonicalTrainingPath = URL(fileURLWithPath: trainingSourcePath)
            .standardizedFileURL.path
        let canonicalExclusionPath = URL(fileURLWithPath: exclusionSourcePath)
            .standardizedFileURL.path
        let removedNames = exclusionNameSet.sorted()
        let manifest = PromptPairSubtractionManifest(
            schemaVersion: 1,
            artifactRole: "split-safe-training-prompt-subtraction",
            operation: "subtract_names",
            trainingSourcePath: canonicalTrainingPath,
            trainingSourceSHA256: trainingSourceSHA256,
            exclusionSourcePath: canonicalExclusionPath,
            exclusionSourceSHA256: exclusionSourceSHA256,
            inputTrainingPairs: trainingPairs.count,
            inputExclusionPairs: exclusionPairs.count,
            removedPairs: trainingPairs.count - retained.count,
            outputPairs: retained.count,
            removedNames: removedNames,
            trainingNamesSHA256: namesSHA256(trainingNames),
            exclusionNamesSHA256: namesSHA256(exclusionNames),
            outputNamesSHA256: namesSHA256(outputNames),
            preservesTrainingOrder: true,
            preservesRetainedPairContent: true,
            frozenAuditAccessPermitted: false)
        return PromptPairSubtractionResult(pairs: retained, manifest: manifest)
    }

    private enum Role {
        case training
        case exclusion
        case output
    }

    private static func validatedNames(
        _ pairs: [PromptPair], role: Role
    ) throws -> [String] {
        var seen = Set<String>()
        var result = [String]()
        result.reserveCapacity(pairs.count)
        for pair in pairs {
            let name = pair.name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty, seen.insert(name).inserted else {
                switch role {
                case .training, .output:
                    throw PromptPairSubtractionError
                        .duplicateOrEmptyTrainingName(pair.name)
                case .exclusion:
                    throw PromptPairSubtractionError
                        .duplicateOrEmptyExclusionName(pair.name)
                }
            }
            try validateSplit(pair.split, name: name, role: role)
            result.append(name)
        }
        return result
    }

    private static func validateSplit(
        _ rawSplit: String?, name: String, role: Role
    ) throws {
        guard let rawSplit else { return }
        let split = rawSplit.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch role {
        case .training, .output:
            guard split == CounterfactualDatasetSplit.train.rawValue else {
                throw PromptPairSubtractionError.invalidTrainingSplit(
                    name: name, split: rawSplit)
            }
        case .exclusion:
            guard split == CounterfactualDatasetSplit.dev.rawValue else {
                throw PromptPairSubtractionError.invalidExclusionSplit(
                    name: name, split: rawSplit)
            }
        }
    }

    private static func namesSHA256(_ names: [String]) -> String {
        let canonical = names.map {
            "\(Data($0.utf8).count):\($0)"
        }.joined(separator: "\n")
        return ScreeningReviewProvenance.sha256(canonical)
    }
}

public enum PromptPairSubtractionError: LocalizedError, Equatable {
    case emptyTrainingInput
    case emptyExclusionInput
    case duplicateOrEmptyTrainingName(String)
    case duplicateOrEmptyExclusionName(String)
    case invalidTrainingSplit(name: String, split: String)
    case invalidExclusionSplit(name: String, split: String)
    case exclusionNamesMissingFromTraining([String])
    case emptyOutput
    case missingProvenance

    public var errorDescription: String? {
        switch self {
        case .emptyTrainingInput:
            "The training PromptFile is empty."
        case .emptyExclusionInput:
            "The dev exclusion PromptFile is empty."
        case .duplicateOrEmptyTrainingName(let name):
            "The training PromptFile contains an empty or duplicate name '\(name)'."
        case .duplicateOrEmptyExclusionName(let name):
            "The dev exclusion PromptFile contains an empty or duplicate name '\(name)'."
        case .invalidTrainingSplit(let name, let split):
            "Training pair '\(name)' records non-train split '\(split)'."
        case .invalidExclusionSplit(let name, let split):
            "Dev exclusion pair '\(name)' records non-dev split '\(split)'."
        case .exclusionNamesMissingFromTraining(let names):
            "Dev exclusion names are absent from training: \(names.joined(separator: ", "))."
        case .emptyOutput:
            "Subtracting the dev names would leave no training pairs."
        case .missingProvenance:
            "Training/exclusion source paths and SHA-256 digests are required."
        }
    }
}
