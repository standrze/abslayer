@testable import ProbeCore
import Testing

@Suite("Decoder-layer selection")
struct DecoderLayerSelectionTests {
    @Test("default stack is model-derived rather than a Heretic window")
    func fullStackIsModelDerived() throws {
        let layers = try DecoderLayerSelection.fullStackZeroBased(layerCount: 35)
        #expect(layers.count == 35)
        #expect(layers.first == 0)
        #expect(layers.last == 34)
        #expect(layers.contains(16))
        #expect(layers.contains(24))
    }

    @Test("legacy one-based overrides convert only at the boundary")
    func legacyConversion() throws {
        let canonical = try DecoderLayerSelection.zeroBasedFromLegacyOneBased(
            [1, 18, 35], layerCount: 35)
        #expect(canonical == [0, 17, 34])
        #expect(try DecoderLayerSelection.runtimeOneBased(
            fromZeroBased: canonical, layerCount: 35) == [1, 18, 35])
    }

    @Test("explicit selections never silently drop invalid or duplicate layers")
    func explicitSelectionValidation() {
        #expect(throws: DecoderLayerSelectionError.self) {
            try DecoderLayerSelection.validateZeroBased(
                [0, 35], layerCount: 35)
        }
        #expect(throws: DecoderLayerSelectionError.self) {
            try DecoderLayerSelection.validateZeroBased(
                [7, 7], layerCount: 35)
        }
        #expect(throws: DecoderLayerSelectionError.self) {
            try DecoderLayerSelection.zeroBasedFromLegacyOneBased(
                [0, 20], layerCount: 35)
        }
    }

    @Test("automatic candidates cover the complete decoder depth")
    func depthDistributedCandidates() throws {
        // Make the first band globally dominant. The selector must still keep
        // a data-ranked candidate from the middle and final bands.
        let ranked = (0 ..< 12).map { layer in
            DecoderLayerSelection.RankedLayer(
                zeroBasedIndex: layer,
                priority: layer < 4 ? 100 - Double(layer) : 10 - Double(layer))
        }
        let selected = try DecoderLayerSelection.depthDistributedZeroBased(
            rankedLayers: ranked, maximum: 3, layerCount: 12)
        #expect(selected.count == 3)
        #expect(selected.contains { (0 ..< 4).contains($0) })
        #expect(selected.contains { (4 ..< 8).contains($0) })
        #expect(selected.contains { (8 ..< 12).contains($0) })
    }
}
