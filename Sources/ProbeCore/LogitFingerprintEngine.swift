import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Tokenizers

public struct LogitFingerprint: Sendable {
    public let promptNames: [String]
    public let vocabularySize: Int
    public let logProbabilities: [[Float]]

    public init(promptNames: [String], vocabularySize: Int, logProbabilities: [[Float]]) {
        self.promptNames = promptNames
        self.vocabularySize = vocabularySize
        self.logProbabilities = logProbabilities
    }

    public func write(to path: String) throws {
        let header = Header(
            version: 1, promptNames: promptNames, vocabularySize: vocabularySize)
        let headerData = try JSONEncoder().encode(header)
        var output = Data("ABSLKL01".utf8)
        var length = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &length) { output.append(contentsOf: $0) }
        output.append(headerData)
        for row in logProbabilities {
            guard row.count == vocabularySize else { throw FingerprintError.malformed(path) }
            row.withUnsafeBytes { output.append(contentsOf: $0) }
        }
        try output.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    public static func read(from path: String) throws -> LogitFingerprint {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard data.count >= 16, String(decoding: data.prefix(8), as: UTF8.self) == "ABSLKL01"
        else { throw FingerprintError.malformed(path) }
        let headerLength = data[8 ..< 16].withUnsafeBytes {
            UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
        }
        let headerEnd = 16 + Int(headerLength)
        guard headerEnd <= data.count else { throw FingerprintError.malformed(path) }
        let header = try JSONDecoder().decode(Header.self, from: data[16 ..< headerEnd])
        let valueCount = header.promptNames.count * header.vocabularySize
        guard data.count == headerEnd + valueCount * MemoryLayout<Float>.size else {
            throw FingerprintError.malformed(path)
        }
        let values: [Float] = data[headerEnd...].withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        let rows = stride(from: 0, to: values.count, by: header.vocabularySize).map {
            Array(values[$0 ..< $0 + header.vocabularySize])
        }
        return LogitFingerprint(
            promptNames: header.promptNames,
            vocabularySize: header.vocabularySize,
            logProbabilities: rows)
    }

    private struct Header: Codable {
        let version: Int
        let promptNames: [String]
        let vocabularySize: Int
    }
}

public enum LogitFingerprintEngine {
    public static func capture(
        modelDirectory: String, pairs: [PromptPair], maximumCases: Int,
        adapterDirectory: String? = nil, adapterScaleOverride: Float? = nil
    ) async throws -> LogitFingerprint {
        let selected = evenlySpaced(pairs, maximum: maximumCases)
        let url = URL(fileURLWithPath: modelDirectory).standardizedFileURL
        let loadStarted = ContinuousClock.now
        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(directory: url, extraEOSTokens: ["<end_of_turn>"]))
        let loadTime = loadStarted.duration(to: .now)
        print("model loaded in \(loadTime.formatted(.units(allowed: [.seconds, .milliseconds], width: .abbreviated)))")
        if let adapterDirectory {
            let adapter = try LoRAAdapterLoader.load(
                directory: adapterDirectory, scaleOverride: adapterScaleOverride)
            try await container.perform { context in
                try adapter.load(into: context.model)
            }
            let suffix = adapterScaleOverride.map { " at scale \($0)" } ?? ""
            print("loaded adapter from \(adapterDirectory)\(suffix)")
        }
        var rows = [[Float]]()
        rows.reserveCapacity(selected.count)
        for (index, pair) in selected.enumerated() {
            let row = try await container.perform { context in
                let tokens = try context.tokenizer.applyChatTemplate(messages: [
                    ["role": "user", "content": pair.control]
                ])
                guard tokens.count <= 512 else {
                    throw ProbeError.promptTooLong(name: pair.name, tokenCount: tokens.count)
                }
                let input = MLXArray(tokens).expandedDimensions(axis: 0)
                let logits = context.model(input, cache: nil)[0, -1].asType(.float32)
                let probabilities = MLXNN.logSoftmax(logits, axis: -1)
                eval(probabilities)
                return probabilities.asArray(Float.self)
            }
            rows.append(row)
            print("fingerprinted \(index + 1)/\(selected.count)")
        }
        return LogitFingerprint(
            promptNames: selected.map(\.name),
            vocabularySize: rows.first?.count ?? 0,
            logProbabilities: rows)
    }

    /// Captures every next-token distribution along each benign control prompt.
    /// A baseline chooses its top-K support; candidate captures must reuse that
    /// support so top-K-plus-tail KL is directly comparable at every position.
    public static func captureSequence(
        modelDirectory: String, pairs: [PromptPair], maximumCases: Int,
        topK: Int = 64, reference: SequenceLogitFingerprint? = nil,
        adapterDirectory: String? = nil, adapterScaleOverride: Float? = nil
    ) async throws -> SequenceLogitFingerprint {
        let selected = evenlySpaced(pairs, maximum: maximumCases)
        guard topK > 0, reference == nil || reference?.cases.count == selected.count else {
            throw FingerprintError.incompatible
        }
        let requestedTopK = reference?.topK ?? topK
        let url = URL(fileURLWithPath: modelDirectory).standardizedFileURL
        let loadStarted = ContinuousClock.now
        try MLXResourceGuard.apply()
        let container = try await #huggingFaceLoadModelContainer(
            configuration: ModelConfiguration(
                directory: url, extraEOSTokens: ["<end_of_turn>"]))
        let loadTime = loadStarted.duration(to: .now)
        print("model loaded in \(loadTime.formatted(.units(allowed: [.seconds, .milliseconds], width: .abbreviated)))")
        if let adapterDirectory {
            let adapter = try LoRAAdapterLoader.load(
                directory: adapterDirectory, scaleOverride: adapterScaleOverride)
            try await container.perform { context in
                try adapter.load(into: context.model)
            }
            let suffix = adapterScaleOverride.map { " at scale \($0)" } ?? ""
            print("loaded adapter from \(adapterDirectory)\(suffix)")
        }
        var cases = [SequenceCaseFingerprint]()
        var vocabularySize: Int?

        for (index, pair) in selected.enumerated() {
            let referenceCase = reference?.cases[index]
            guard referenceCase == nil || referenceCase?.name == pair.name else {
                throw FingerprintError.incompatible
            }
            let captured = try await container.perform { context in
                let tokens = try context.tokenizer.applyChatTemplate(messages: [
                    ["role": "user", "content": pair.control]
                ])
                guard tokens.count <= 512 else {
                    throw ProbeError.promptTooLong(name: pair.name, tokenCount: tokens.count)
                }
                guard tokens.count >= 2 else { throw FingerprintError.incompatible }
                guard referenceCase == nil || referenceCase?.tokenIDs == tokens else {
                    throw FingerprintError.incompatible
                }

                let positionCount = tokens.count - 1
                let input = MLXArray(tokens).expandedDimensions(axis: 0)
                let logits = context.model(input, cache: nil)[0, 0 ..< positionCount]
                    .asType(.float32)
                let logProbabilities = MLXNN.logSoftmax(logits, axis: -1)
                let vocabulary = logProbabilities.dim(-1)
                guard requestedTopK < vocabulary else { throw FingerprintError.incompatible }

                let supportIndices: MLXArray
                if let referenceCase {
                    guard referenceCase.positions.count == positionCount,
                          referenceCase.positions.allSatisfy({
                              $0.supportTokenIDs.count == requestedTopK
                          })
                    else { throw FingerprintError.incompatible }
                    supportIndices = MLXArray(
                        referenceCase.positions.flatMap(\.supportTokenIDs),
                        [positionCount, requestedTopK])
                } else {
                    let partitioned = argPartition(
                        logProbabilities, kth: -requestedTopK, axis: -1)
                    supportIndices = partitioned[.ellipsis, (-requestedTopK)...]
                }
                let supportLogProbabilities = takeAlong(
                    logProbabilities, supportIndices, axis: -1)
                let retainedMass = exp(supportLogProbabilities).sum(axis: -1)
                let tailLogProbability = log(clip(1 - retainedMass, min: Float(1e-30)))
                let targetIDs = Array(tokens.dropFirst())
                let targetIndices = MLXArray(targetIDs).expandedDimensions(axis: -1)
                let targetLogProbability = takeAlong(
                    logProbabilities, targetIndices, axis: -1).squeezed(axis: -1)
                eval(
                    supportIndices, supportLogProbabilities,
                    tailLogProbability, targetLogProbability)

                let supportIDs = supportIndices.asArray(Int.self)
                let supportLogs = supportLogProbabilities.asArray(Float.self)
                let tailLogs = tailLogProbability.asArray(Float.self)
                let targetLogs = targetLogProbability.asArray(Float.self)
                let positions = (0 ..< positionCount).map { position in
                    let range = position * requestedTopK ..< (position + 1) * requestedTopK
                    return SequencePositionFingerprint(
                        targetTokenID: targetIDs[position],
                        supportTokenIDs: Array(supportIDs[range]),
                        supportLogProbabilities: Array(supportLogs[range]),
                        tailLogProbability: tailLogs[position],
                        targetLogProbability: targetLogs[position])
                }
                return (
                    vocabulary,
                    SequenceCaseFingerprint(
                        name: pair.name, tokenIDs: tokens, positions: positions))
            }
            if let vocabularySize, vocabularySize != captured.0 {
                throw FingerprintError.incompatible
            }
            vocabularySize = captured.0
            cases.append(captured.1)
            print("sequence-fingerprinted \(index + 1)/\(selected.count)")
        }
        guard let vocabularySize else { throw FingerprintError.incompatible }
        if let reference, reference.vocabularySize != vocabularySize {
            throw FingerprintError.incompatible
        }
        return SequenceLogitFingerprint(
            vocabularySize: vocabularySize, topK: requestedTopK, cases: cases)
    }

    /// KL(P_baseline || P_candidate), averaged over prompts.
    public static func divergence(
        baseline: LogitFingerprint, candidate: LogitFingerprint
    ) throws -> Double {
        guard baseline.promptNames == candidate.promptNames,
              baseline.vocabularySize == candidate.vocabularySize,
              baseline.logProbabilities.count == candidate.logProbabilities.count,
              !baseline.logProbabilities.isEmpty
        else { throw FingerprintError.incompatible }
        var total = 0.0
        for (base, trial) in zip(baseline.logProbabilities, candidate.logProbabilities) {
            guard base.count == trial.count else { throw FingerprintError.incompatible }
            total += zip(base, trial).reduce(0.0) {
                let logP = Double($1.0)
                return $0 + exp(logP) * (logP - Double($1.1))
            }
        }
        return total / Double(baseline.logProbabilities.count)
    }

    private static func evenlySpaced(_ pairs: [PromptPair], maximum: Int) -> [PromptPair] {
        guard maximum > 0, pairs.count > maximum else { return pairs }
        return (0 ..< maximum).map { pairs[$0 * pairs.count / maximum] }
    }
}

public enum FingerprintError: LocalizedError {
    case malformed(String)
    case incompatible

    public var errorDescription: String? {
        switch self {
        case .malformed(let path): "Malformed logit fingerprint: \(path)"
        case .incompatible: "Logit fingerprints use different prompts or vocabularies."
        }
    }
}
