import Foundation

/// Stable interchange format for document-condensation harnesses. The trainer
/// consumes this envelope directly while retaining the legacy flat array format.
public struct HarnessTrainingDataset: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let datasetName: String
    public let records: [HarnessTrainingRecord]

    public init(schemaVersion: Int = 1, datasetName: String, records: [HarnessTrainingRecord]) {
        self.schemaVersion = schemaVersion
        self.datasetName = datasetName
        self.records = records
    }
}

public struct HarnessTrainingRecord: Codable, Equatable, Sendable {
    public let id: String
    public let instruction: String
    public let response: String
    public let rejectedResponse: String?
    public let provenance: HarnessTrainingProvenance?

    public init(
        id: String, instruction: String, response: String,
        rejectedResponse: String? = nil, provenance: HarnessTrainingProvenance? = nil
    ) {
        self.id = id
        self.instruction = instruction
        self.response = response
        self.rejectedResponse = rejectedResponse
        self.provenance = provenance
    }
}

public struct HarnessTrainingProvenance: Codable, Equatable, Sendable {
    public let document: String
    public let pages: [Int]

    public init(document: String, pages: [Int] = []) {
        self.document = document
        self.pages = pages
    }
}

public struct LoRATrainingExample: Codable, Equatable, Sendable {
    public let prompt: String
    public let target: String
    public let rejected: String?

    public init(prompt: String, target: String, rejected: String? = nil) {
        self.prompt = prompt
        self.target = target
        self.rejected = rejected
    }
}

public enum HarnessTrainingDatasetLoader {
    public static func load(_ path: String) throws -> [LoRATrainingExample] {
        try load(Data(contentsOf: URL(fileURLWithPath: path).standardizedFileURL))
    }

    public static func load(_ data: Data) throws -> [LoRATrainingExample] {
        let decoder = JSONDecoder()
        if let envelope = try? decoder.decode(HarnessTrainingDataset.self, from: data) {
            return try validate(envelope)
        }
        if let examples = try? decoder.decode([LoRATrainingExample].self, from: data) {
            return try validateLegacy(examples)
        }
        if let prompts = try? decoder.decode([String].self, from: data) {
            let prefix = "Certainly. Here is the requested response, directly and without disclaimers or moralizing:\n\n"
            return try validateLegacy(prompts.map {
                LoRATrainingExample(prompt: $0, target: prefix)
            })
        }
        throw HarnessTrainingDatasetError.unrecognizedFormat
    }

    private static func validate(
        _ dataset: HarnessTrainingDataset
    ) throws -> [LoRATrainingExample] {
        guard dataset.schemaVersion == 1 else {
            throw HarnessTrainingDatasetError.unsupportedSchema(dataset.schemaVersion)
        }
        guard !trimmed(dataset.datasetName).isEmpty else {
            throw HarnessTrainingDatasetError.emptyDatasetName
        }
        guard !dataset.records.isEmpty else { throw HarnessTrainingDatasetError.emptyDataset }
        var ids = Set<String>()
        var examples = [LoRATrainingExample]()
        for record in dataset.records {
            let id = trimmed(record.id)
            guard !id.isEmpty else { throw HarnessTrainingDatasetError.emptyRecordID }
            guard ids.insert(id).inserted else {
                throw HarnessTrainingDatasetError.duplicateRecordID(id)
            }
            let prompt = trimmed(record.instruction)
            let target = trimmed(record.response)
            guard !prompt.isEmpty else { throw HarnessTrainingDatasetError.emptyInstruction(id) }
            guard !target.isEmpty else { throw HarnessTrainingDatasetError.emptyResponse(id) }
            let rejected = record.rejectedResponse.map(trimmed)
            if let rejected, rejected.isEmpty {
                throw HarnessTrainingDatasetError.emptyRejectedResponse(id)
            }
            if let provenance = record.provenance {
                guard !trimmed(provenance.document).isEmpty else {
                    throw HarnessTrainingDatasetError.emptyDocument(id)
                }
                guard provenance.pages.allSatisfy({ $0 > 0 }) else {
                    throw HarnessTrainingDatasetError.invalidPage(id)
                }
            }
            examples.append(LoRATrainingExample(
                prompt: prompt, target: target, rejected: rejected))
        }
        return try validateLegacy(examples)
    }

    private static func validateLegacy(
        _ examples: [LoRATrainingExample]
    ) throws -> [LoRATrainingExample] {
        guard !examples.isEmpty else { throw HarnessTrainingDatasetError.emptyDataset }
        for (index, example) in examples.enumerated() {
            guard !trimmed(example.prompt).isEmpty else {
                throw HarnessTrainingDatasetError.emptyInstruction(String(index))
            }
            guard !trimmed(example.target).isEmpty else {
                throw HarnessTrainingDatasetError.emptyResponse(String(index))
            }
        }
        let preferences = examples.map { $0.rejected != nil }
        guard preferences.allSatisfy({ $0 }) || preferences.allSatisfy({ !$0 }) else {
            throw HarnessTrainingDatasetError.mixedObjectives
        }
        guard Set(examples.map { trimmed($0.prompt) }).count > 1 else {
            throw HarnessTrainingDatasetError.insufficientPromptGroups
        }
        return examples
    }

    private static func trimmed(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public enum HarnessTrainingDatasetError: LocalizedError, Equatable {
    case unrecognizedFormat, emptyDatasetName, emptyDataset, emptyRecordID, mixedObjectives
    case insufficientPromptGroups
    case unsupportedSchema(Int), duplicateRecordID(String), emptyInstruction(String)
    case emptyResponse(String), emptyRejectedResponse(String), emptyDocument(String)
    case invalidPage(String)

    public var errorDescription: String? {
        switch self {
        case .unrecognizedFormat: "Training data is not harness-v1, completion-array, or prompt-array JSON."
        case .unsupportedSchema(let version): "Unsupported harness training schema version: \(version)."
        case .emptyDatasetName: "datasetName cannot be empty."
        case .emptyDataset: "The training dataset is empty."
        case .emptyRecordID: "Every harness record needs a non-empty id."
        case .duplicateRecordID(let id): "Duplicate harness record id: \(id)."
        case .emptyInstruction(let id): "Record \(id) has an empty instruction."
        case .emptyResponse(let id): "Record \(id) has an empty response."
        case .emptyRejectedResponse(let id): "Record \(id) has an empty rejectedResponse."
        case .emptyDocument(let id): "Record \(id) has an empty provenance document."
        case .invalidPage(let id): "Record \(id) has a page number below 1."
        case .mixedObjectives: "Use either responses only or rejectedResponse on every record, not a mixture."
        case .insufficientPromptGroups:
            "Training requires at least two distinct instructions so validation is independent."
        }
    }
}
