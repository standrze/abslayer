import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXLinalg
import Tokenizers

public enum DirectionExtractionMethod: String, Codable, Sendable {
    /// Legacy projected mean difference retained for study reproducibility.
    case meanDifference
    /// Conventional abliteration: harmful centroid minus harmless centroid.
    case centroidDifference
    /// Normalize the paired difference matrix by harmless covariance first.
    case whitenedSVD
    /// Multiple non-orthogonal directions from a harmful-activation SOM.
    case som
}

public struct ProbeEngine: Sendable {
    public enum Source: Sendable {
        case hub(String)
        case directory(String)
    }

    public let source: Source
    public let pairs: [PromptPair]
    public let subspaceRank: Int
    public let extractionMethod: DirectionExtractionMethod
    public let generateResponses: Bool
    public let winsorizationQuantile: Float?
    public let tokenPosition: ActivationTokenPosition
    public let maximumSequenceLength: Int
    public let gpuMemoryUtilization: Double?
    public let emitResourceStatus: Bool
    /// Optional zero-based layer indices on which to train expensive extractors.
    ///
    /// This currently constrains `.som` only. Unselected layers retain a
    /// rank-one centroid direction so reports remain shape-compatible. Callers
    /// seeking a purely SOM edit should apply a zero-radius edit only to one of
    /// the selected layers.
    public let extractionLayers: Set<Int>?

    public init(
        modelID: String, pairs: [PromptPair], subspaceRank: Int = 4,
        extractionMethod: DirectionExtractionMethod = .centroidDifference,
        generateResponses: Bool = true, winsorizationQuantile: Float? = nil,
        tokenPosition: ActivationTokenPosition = .postInstruction,
        extractionLayers: Set<Int>? = nil,
        maximumSequenceLength: Int = 512,
        gpuMemoryUtilization: Double? = nil,
        emitResourceStatus: Bool = true
    ) {
        self.source = .hub(modelID)
        self.pairs = pairs
        self.subspaceRank = max(1, subspaceRank)
        self.extractionMethod = extractionMethod
        self.generateResponses = generateResponses
        self.winsorizationQuantile = winsorizationQuantile
        self.tokenPosition = tokenPosition
        self.extractionLayers = extractionLayers
        self.maximumSequenceLength = maximumSequenceLength
        self.gpuMemoryUtilization = gpuMemoryUtilization
        self.emitResourceStatus = emitResourceStatus
    }

    public init(
        modelDirectory: String, pairs: [PromptPair], subspaceRank: Int = 4,
        extractionMethod: DirectionExtractionMethod = .centroidDifference,
        generateResponses: Bool = true, winsorizationQuantile: Float? = nil,
        tokenPosition: ActivationTokenPosition = .postInstruction,
        extractionLayers: Set<Int>? = nil,
        maximumSequenceLength: Int = 512,
        gpuMemoryUtilization: Double? = nil,
        emitResourceStatus: Bool = true
    ) {
        self.source = .directory(modelDirectory)
        self.pairs = pairs
        self.subspaceRank = max(1, subspaceRank)
        self.extractionMethod = extractionMethod
        self.generateResponses = generateResponses
        self.winsorizationQuantile = winsorizationQuantile
        self.tokenPosition = tokenPosition
        self.extractionLayers = extractionLayers
        self.maximumSequenceLength = maximumSequenceLength
        self.gpuMemoryUtilization = gpuMemoryUtilization
        self.emitResourceStatus = emitResourceStatus
    }

    public func run() async throws -> ProbeReport {
        let configuration: ModelConfiguration
        let displayName: String
        switch source {
        case .hub(let modelID):
            configuration = ModelConfiguration(
                id: modelID,
                extraEOSTokens: ["<end_of_turn>"]
            )
            displayName = modelID
        case .directory(let path):
            let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
                .standardizedFileURL
            configuration = ModelConfiguration(
                directory: url,
                extraEOSTokens: ["<end_of_turn>"]
            )
            displayName = url.path
        }
        guard maximumSequenceLength > 0 else {
            throw ProbeEngineError.invalidMaximumSequenceLength(maximumSequenceLength)
        }
        var resourceEnvironment = ProcessInfo.processInfo.environment
        if let gpuMemoryUtilization {
            let limit = try MLXResourceLimits.cudaMemoryLimitGiB(
                utilization: gpuMemoryUtilization)
            resourceEnvironment[MLXResourceLimits.memoryLimitKey] = String(
                format: "%.9f", limit)
        }
        try MLXResourceGuard.apply(environment: resourceEnvironment, emitStatus: emitResourceStatus)
        let container = try await #huggingFaceLoadModelContainer(
            configuration: configuration
        )

        var totals: [Double] = []
        var deltasByLayer: [[[Float]]] = []
        var contrastByLayer: [[[Float]]] = []
        var controlByLayer: [[[Float]]] = []
        for pair in pairs {
            let observations = try await container.perform { context in
                let contrast = try layerVectors(
                    context: context, prompt: pair.contrast, pairName: pair.name,
                    tokenPosition: tokenPosition,
                    maximumSequenceLength: maximumSequenceLength)
                let control = try layerVectors(
                    context: context, prompt: pair.control, pairName: pair.name,
                    tokenPosition: tokenPosition,
                    maximumSequenceLength: maximumSequenceLength)
                return zip(contrast, control).map { contrastLayer, controlLayer in
                    let processedContrast = AbliterationMath.winsorized(
                        contrastLayer, quantile: winsorizationQuantile)
                    let processedControl = AbliterationMath.winsorized(
                        controlLayer, quantile: winsorizationQuantile)
                    return LayerObservation(
                        distance: LayerMath.cosineDistance(processedContrast, processedControl),
                        delta: zip(processedContrast, processedControl).map(-),
                        contrast: processedContrast,
                        control: processedControl
                    )
                }
            }
            if totals.isEmpty {
                totals = Array(repeating: 0, count: observations.count)
                deltasByLayer = Array(repeating: [], count: observations.count)
                contrastByLayer = Array(repeating: [], count: observations.count)
                controlByLayer = Array(repeating: [], count: observations.count)
            }
            for index in observations.indices {
                totals[index] += observations[index].distance
                deltasByLayer[index].append(observations[index].delta)
                contrastByLayer[index].append(observations[index].contrast)
                controlByLayer[index].append(observations[index].control)
            }
        }

        var responses = [PromptResult]()
        let generation = GenerateParameters(maxTokens: 160, temperature: 0)
        if generateResponses {
            for pair in pairs {
                // Separate sessions prevent one prompt from affecting the next.
                let contrast = try await ChatSession(
                    container, generateParameters: generation).respond(to: pair.contrast)
                let control = try await ChatSession(
                    container, generateParameters: generation).respond(to: pair.control)
                responses.append(PromptResult(
                    name: pair.name,
                    contrastResponse: contrast,
                    controlResponse: control,
                    category: pair.category,
                    contrastPrompt: pair.contrast,
                    controlPrompt: pair.control
                ))
            }
        }

        let divisor = Double(pairs.count)
        let scores = totals.enumerated().map {
            let meanDirection = AbliterationMath.normalized(zip(
                AbliterationMath.mean(contrastByLayer[$0.offset]),
                AbliterationMath.mean(controlByLayer[$0.offset])
            ).map(-))
            let medianDirection = AbliterationMath.normalized(zip(
                LayerMath.geometricMedian(contrastByLayer[$0.offset]),
                LayerMath.geometricMedian(controlByLayer[$0.offset])
            ).map(-))
            return LayerScore(
                layer: $0.offset + 1,
                cosineDistance: $0.element / divisor,
                directionAgreement: LayerMath.meanPairwiseAgreement(deltasByLayer[$0.offset]),
                medianDirectionAgreement: LayerMath.cosineSimilarity(
                    meanDirection, medianDirection),
                silhouette: LayerMath.binarySilhouette(
                    contrastByLayer[$0.offset], controlByLayer[$0.offset])
            )
        }
        if let extractionLayers {
            guard !extractionLayers.isEmpty,
                  extractionLayers.allSatisfy(contrastByLayer.indices.contains)
            else {
                throw ProbeEngineError.invalidExtractionLayers(
                    requested: extractionLayers.sorted(), available: contrastByLayer.count)
            }
        }
        let subspaces: [[[Float]]]
        var somResultsByLayer: [Int: SOMDirectionResult] = [:]
        if extractionMethod == .som {
            let results = SOMDirectionExtractor.extractLayers(
                harmfulByLayer: contrastByLayer,
                harmlessByLayer: controlByLayer,
                rank: subspaceRank,
                selectedLayers: extractionLayers)
            somResultsByLayer = SOMDirectionExtractor.indexedTrainedResults(results)
            subspaces = contrastByLayer.indices.map { index in
                if !results[index].directions.isEmpty {
                    return results[index].directions
                }
                // Explicit compatibility placeholder for layers excluded from
                // SOM training. It is intentionally rank one and is not a SOM
                // result; see `extractionLayers` above.
                return [AbliterationMath.direction(
                    contrast: contrastByLayer[index],
                    control: controlByLayer[index],
                    projectAwayFromControl: false,
                    winsorQuantile: nil)]
            }
        } else {
            subspaces = contrastByLayer.indices.map { index -> [[Float]] in
                switch extractionMethod {
                case .meanDifference:
                    let primary = AbliterationMath.direction(
                        contrast: contrastByLayer[index],
                        control: controlByLayer[index],
                        projectAwayFromControl: true,
                        winsorQuantile: nil)
                    return rawSubspace(
                        contrast: contrastByLayer[index], control: controlByLayer[index],
                        primary: primary, rank: subspaceRank)
                case .centroidDifference:
                    let primary = AbliterationMath.direction(
                        contrast: contrastByLayer[index],
                        control: controlByLayer[index],
                        projectAwayFromControl: false,
                        winsorQuantile: nil)
                    return rawSubspace(
                        contrast: contrastByLayer[index], control: controlByLayer[index],
                        primary: primary, rank: subspaceRank)
                case .whitenedSVD:
                    return whitenedSubspace(
                        contrast: contrastByLayer[index], control: controlByLayer[index],
                        rank: subspaceRank)
                case .som:
                    preconditionFailure("SOM extraction is handled in one batched pass")
                }
            }
        }
        // Keep the single-direction API consistent with the actual first
        // direction used by rank-k editing.  Previously this was a centroid
        // direction while the remaining basis came from raw SVD, so rank 1
        // and rank k were optimizing different concepts.
        let directions = subspaces.map { $0.first ?? [] }
        return ProbeReport(
            model: displayName, pairCount: pairs.count, layers: scores,
            responses: responses, directions: directions, subspaces: subspaces,
            somResultsByLayer: somResultsByLayer)
    }
}

public enum ProbeEngineError: LocalizedError, Sendable {
    case invalidExtractionLayers(requested: [Int], available: Int)
    case invalidMaximumSequenceLength(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidExtractionLayers(let requested, let available):
            "Zero-based extraction layers \(requested) are invalid for \(available) layers."
        case .invalidMaximumSequenceLength(let value):
            "Maximum activation sequence length must be positive, not \(value)."
        }
    }
}

private func rawSubspace(
    contrast: [[Float]], control: [[Float]], primary: [Float], rank: Int
) -> [[Float]] {
    guard let width = contrast.first?.count, width > 0 else { return [] }
    let differences = zip(contrast, control).map { zip($0, $1).map(-) }
    let matrix = MLXArray(differences.flatMap { $0 }).reshaped(differences.count, width)
    let (_, _, vt) = MLXLinalg.svd(matrix.asType(.float32), stream: .cpu)
    let available = min(rank, differences.count, width)
    let selected = vt[0 ..< available]
    eval(selected)
    var basis = [AbliterationMath.normalized(primary)]
    for row in 0 ..< available where basis.count < rank {
        var candidate = selected[row].asArray(Float.self)
        for existing in basis {
            let projection = zip(candidate, existing).reduce(Float.zero) {
                $0 + $1.0 * $1.1
            }
            candidate = zip(candidate, existing).map { $0 - projection * $1 }
        }
        candidate = AbliterationMath.normalized(candidate)
        let norm = sqrt(candidate.reduce(Float.zero) { $0 + $1 * $1 })
        if norm > 0.99 { basis.append(candidate) }
    }
    return basis
}

/// Covariance-whitened SVD in the low-rank control sample space.
///
/// With n prompt pairs and hidden width d (n << d), the non-zero eigenvectors
/// of the d x d harmless covariance are exactly the right singular vectors of
/// the centered n x d control matrix. The orthogonal complement is still
/// important: after ridge regularization it has variance `epsilon`, and is
/// often where a behavior-specific paired delta lives. We therefore apply the
/// inverse square root as an identity-plus-low-rank transform instead of
/// projecting the deltas into the observed harmless span. This is equivalent
/// to whitening by `cov(control) + epsilon I` without materializing a d x d
/// matrix.
func whitenedSubspace(
    contrast: [[Float]], control: [[Float]], rank: Int,
    regularizationEpsilon: Float = 1e-4, minimumVarianceRatio: Float = 0.01
) -> [[Float]] {
    guard contrast.count == control.count, contrast.count >= 2,
          let width = contrast.first?.count, width > 0,
          control.allSatisfy({ $0.count == width }),
          contrast.allSatisfy({ $0.count == width })
    else { return [] }

    let controlMean = AbliterationMath.mean(control)
    let centeredControl = control.map { row in zip(row, controlMean).map(-) }
    let differences = zip(contrast, control).map { zip($0, $1).map(-) }

    let controlMatrix = MLXArray(centeredControl.flatMap { $0 })
        .reshaped(centeredControl.count, width).asType(.float32)
    let (_, controlSingular, controlVT) = MLXLinalg.svd(controlMatrix, stream: .cpu)
    eval(controlSingular, controlVT)
    let singularValues = controlSingular.asArray(Float.self)
    let denominator = Float(max(control.count - 1, 1))
    let differenceMatrix = MLXArray(differences.flatMap { $0 })
        .reshaped(differences.count, width).asType(.float32)

    let largestEigenvalue = singularValues.first.map {
        $0 * $0 / denominator
    } ?? 0
    let validCount = largestEigenvalue > 0
        ? singularValues.prefix { value in
            let eigenvalue = value * value / denominator
            return eigenvalue >= largestEigenvalue * minimumVarianceRatio
                && eigenvalue > Float.ulpOfOne
        }.count
        : 0
    let ridgeInverseScale = 1 / sqrt(regularizationEpsilon)
    var whitenedDifferences = differenceMatrix * ridgeInverseScale
    if validCount > 0 {
        let eigenvalues = singularValues.prefix(validCount).map {
            $0 * $0 / denominator
        }
        let inverseCorrections = MLXArray(eigenvalues.map {
            1 / sqrt($0 + regularizationEpsilon) - ridgeInverseScale
        })
        let controlBasis = controlVT[0 ..< validCount, 0...]
        let coordinates = matmul(differenceMatrix, controlBasis.T)
        whitenedDifferences = whitenedDifferences
            + matmul(coordinates * inverseCorrections, controlBasis)
    }

    let (_, _, whitenedVT) = MLXLinalg.svd(whitenedDifferences, stream: .cpu)
    let available = min(rank, differences.count, width)
    var originalDirections = whitenedVT[0 ..< available, 0...]
        * sqrt(regularizationEpsilon)
    if validCount > 0 {
        let eigenvalues = singularValues.prefix(validCount).map {
            $0 * $0 / denominator
        }
        let scaleCorrections = MLXArray(eigenvalues.map {
            sqrt($0 + regularizationEpsilon) - sqrt(regularizationEpsilon)
        })
        let controlBasis = controlVT[0 ..< validCount, 0...]
        let coordinates = matmul(originalDirections, controlBasis.T)
        originalDirections = originalDirections
            + matmul(coordinates * scaleCorrections / sqrt(regularizationEpsilon),
                     controlBasis)
    }
    eval(originalDirections)

    var basis: [[Float]] = []
    for row in 0 ..< available {
        var candidate = originalDirections[row].asArray(Float.self)
        for existing in basis {
            let projection = zip(candidate, existing).reduce(Float.zero) {
                $0 + $1.0 * $1.1
            }
            candidate = zip(candidate, existing).map { $0 - projection * $1 }
        }
        candidate = AbliterationMath.normalized(candidate)
        let norm = sqrt(candidate.reduce(Float.zero) { $0 + $1 * $1 })
        if norm > 0.99 { basis.append(candidate) }
    }
    return basis
}

private func layerVectors(
    context: ModelContext,
    prompt: String,
    pairName: String,
    tokenPosition: ActivationTokenPosition,
    maximumSequenceLength: Int
) throws -> [[Float]] {
    if let vectors = try Gemma4Probe.layerVectors(
        context: context, prompt: prompt, pairName: pairName,
        tokenPosition: tokenPosition,
        maximumSequenceLength: maximumSequenceLength
    ) {
        return vectors
    }
    return try Gemma3Probe.layerVectors(
        context: context, prompt: prompt, pairName: pairName,
        tokenPosition: tokenPosition,
        maximumSequenceLength: maximumSequenceLength
    )
}

private struct LayerObservation: Sendable {
    let distance: Double
    let delta: [Float]
    let contrast: [Float]
    let control: [Float]
}
