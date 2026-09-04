import Foundation
import Testing
@testable import ProbeCore

@Suite("SOM multidirectional extraction", .serialized)
struct SOMDirectionExtractorTests {
    private let fastConfiguration = SOMDirectionConfiguration(
        rows: 4, columns: 4, iterations: 400,
        initialLearningRate: 0.01, sigma: 0.3,
        seed: 0x534F_4D5F_5445_5354,
        trainingMode: .paperText)

    @Test func paperDefaultsAreExplicit() {
        let configuration = SOMDirectionConfiguration.paperDefault
        #expect(configuration.rows == 4)
        #expect(configuration.columns == 4)
        #expect(configuration.iterations == 10_000)
        #expect(configuration.initialLearningRate == 0.01)
        #expect(configuration.sigma == 0.3)
        #expect(configuration.trainingMode == .paperText)
    }

    @Test func releasedCodeDefaultsAndDecayAreExplicit() {
        let configuration = SOMDirectionConfiguration.officialMiniSom235
        #expect(configuration.rows == 4)
        #expect(configuration.columns == 4)
        #expect(configuration.iterations == 10_000)
        #expect(configuration.initialLearningRate == 0.01)
        #expect(configuration.sigma == 0.33)
        #expect(configuration.trainingMode == .officialMiniSom235)

        let halfway = SOMDirectionExtractor.trainingParameters(
            at: 5_000, configuration: configuration)
        #expect(abs(halfway.learningRate - 0.005) < 1e-7)
        #expect(abs(halfway.sigma - 0.165) < 1e-7)

        let paperHalfway = SOMDirectionExtractor.trainingParameters(
            at: 5_000, configuration: .paperDefault)
        #expect(abs(paperHalfway.learningRate - 0.005) < 1e-7)
        #expect(abs(paperHalfway.sigma - 0.3) < 1e-7)
    }

    @Test func releasedCodeScheduleIsDeterministicBalancedAndShuffledOnce() {
        let configuration = SOMDirectionConfiguration(
            iterations: 23, seed: 0xBADC_0FFE,
            trainingMode: .officialMiniSom235)
        let first = SOMDirectionExtractor.trainingSampleIndices(
            sampleCount: 5, neuronCount: 16, configuration: configuration)
        let second = SOMDirectionExtractor.trainingSampleIndices(
            sampleCount: 5, neuronCount: 16, configuration: configuration)
        let unshuffled = (0 ..< 23).map { $0 % 5 }
        let counts = (0 ..< 5).map { sample in
            first.count { $0 == sample }
        }

        #expect(first == second)
        #expect(first != unshuffled)
        #expect(counts == [5, 5, 5, 4, 4])
    }

    @Test func probeReportDefaultsToNoSOMDiagnostics() {
        let report = ProbeReport(
            model: "synthetic", pairCount: 0, layers: [], responses: [],
            directions: [], subspaces: [])
        #expect(report.somResultsByLayer.isEmpty)
    }

    @Test func extractionIsDeterministicAndDirectionsAreUnitLength() {
        let data = orthogonalClusters()
        let first = SOMDirectionExtractor.extract(
            harmful: data.harmful, harmless: data.harmless, rank: 3,
            configuration: fastConfiguration)
        let second = SOMDirectionExtractor.extract(
            harmful: data.harmful, harmless: data.harmless, rank: 3,
            configuration: fastConfiguration)

        #expect(first.selectedNeuronIndices == second.selectedNeuronIndices)
        #expect(first.occupancy == second.occupancy)
        #expect(first.candidateDirections.count == 16)
        #expect(first.directions.count == 3)
        for (position, pair) in zip(first.directions, second.directions).enumerated() {
            let (lhs, rhs) = pair
            #expect(maximumAbsoluteDifference(lhs, rhs) < 1e-6)
            #expect(abs(norm(lhs) - 1) < 1e-5)
            #expect(maximumAbsoluteDifference(
                lhs, first.candidateDirections[first.selectedNeuronIndices[position]]) < 1e-6)
        }
    }

    @Test func multipleDirectionsCoverSeparatedModesBetterThanCentroid() {
        let data = orthogonalClusters()
        let result = SOMDirectionExtractor.extract(
            harmful: data.harmful, harmless: data.harmless, rank: 2,
            configuration: fastConfiguration)
        let harmlessMean = AbliterationMath.mean(data.harmless)
        let harmfulMean = AbliterationMath.mean(data.harmful)
        let centroid = AbliterationMath.normalized(
            zip(harmfulMean, harmlessMean).map(-))
        let modes: [[Float]] = [[1, 0], [0, 1]]
        let centroidCoverage = modes.map { dot($0, centroid) }
            .reduce(0, +) / Float(modes.count)
        let somCoverage = modes.map { mode in
            result.directions.map { dot(mode, $0) }.max() ?? -1
        }.reduce(0, +) / Float(modes.count)

        #expect(result.directions.count == 2)
        #expect(somCoverage > centroidCoverage + 0.15)
        #expect(somCoverage > 0.90)
    }

    @Test func relatedDirectionsRemainNonorthogonal() {
        let harmless = Array(repeating: [Float](arrayLiteral: 0, 0), count: 12)
        let firstMode = clustered(center: [4, 0], count: 18)
        let secondMode = clustered(center: [3, 2], count: 18)
        let result = SOMDirectionExtractor.extract(
            harmful: interleaved(firstMode, secondMode), harmless: harmless,
            rank: 2, configuration: fastConfiguration)

        #expect(result.directions.count == 2)
        let alignment = abs(dot(result.directions[0], result.directions[1]))
        // Gram-Schmidt would make this approximately zero. SOM prototypes are
        // intentionally kept as related, non-orthogonal manifold directions.
        #expect(alignment > 0.25)
        #expect(alignment < 0.99)
    }

    @Test func selectedLayersAvoidTrainingUnrequestedLayers() {
        let data = orthogonalClusters()
        let results = SOMDirectionExtractor.extractLayers(
            harmfulByLayer: [data.harmful, data.harmful],
            harmlessByLayer: [data.harmless, data.harmless],
            rank: 2, selectedLayers: [1], configuration: fastConfiguration)

        #expect(results.count == 2)
        #expect(results[0].directions.isEmpty)
        #expect(results[0].neurons.isEmpty)
        #expect(results[1].directions.count == 2)

        let retained = SOMDirectionExtractor.indexedTrainedResults(results)
        #expect(retained.keys.sorted() == [1])
        #expect(retained[1]?.candidateDirections.count == 16)
        let report = ProbeReport(
            model: "synthetic", pairCount: data.harmful.count,
            layers: [], responses: [], directions: [], subspaces: [],
            somResultsByLayer: retained)
        #expect(report.somResultsByLayer.keys.sorted() == [1])
    }

    @Test func nilLayerShortlistTrainsEveryAvailableLayer() {
        let data = orthogonalClusters()
        let results = SOMDirectionExtractor.extractLayers(
            harmfulByLayer: [data.harmful, data.harmful, data.harmful],
            harmlessByLayer: [data.harmless, data.harmless, data.harmless],
            rank: 2, selectedLayers: nil, configuration: fastConfiguration)

        #expect(results.count == 3)
        #expect(results.allSatisfy { $0.candidateDirections.count == 16 })
        #expect(SOMDirectionExtractor.indexedTrainedResults(results).keys.sorted()
            == [0, 1, 2])
    }

    private func orthogonalClusters() -> (harmful: [[Float]], harmless: [[Float]]) {
        let first = clustered(center: [4, 0], count: 24)
        let second = clustered(center: [0, 4], count: 24)
        let harmless = (0 ..< 24).map { index -> [Float] in
            let x: Float = index.isMultiple(of: 2) ? -0.04 : 0.04
            let y: Float = index.isMultiple(of: 3) ? 0.03 : -0.03
            return [x, y]
        }
        return (interleaved(first, second), harmless)
    }

    private func clustered(center: [Float], count: Int) -> [[Float]] {
        (0 ..< count).map { index in
            let x = Float((index % 5) - 2) * 0.035
            let y = Float(((index * 3) % 7) - 3) * 0.025
            return [center[0] + x, center[1] + y]
        }
    }

    private func interleaved(_ lhs: [[Float]], _ rhs: [[Float]]) -> [[Float]] {
        (0 ..< max(lhs.count, rhs.count)).flatMap { index in
            var values: [[Float]] = []
            if index < lhs.count { values.append(lhs[index]) }
            if index < rhs.count { values.append(rhs[index]) }
            return values
        }
    }

    private func norm(_ vector: [Float]) -> Float {
        sqrt(vector.reduce(Float.zero) { $0 + $1 * $1 })
    }

    private func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).reduce(Float.zero) { $0 + $1.0 * $1.1 }
    }

    private func maximumAbsoluteDifference(_ lhs: [Float], _ rhs: [Float]) -> Float {
        zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
    }
}
