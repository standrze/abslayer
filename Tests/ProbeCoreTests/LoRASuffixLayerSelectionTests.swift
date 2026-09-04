import Testing
@testable import ProbeCore

@Suite("LoRA suffix layer selection")
struct LoRASuffixLayerSelectionTests {
    @Test("resolves the last 18 of 35 layers without an off-by-one")
    func resolvesGemmaE2BRange() throws {
        let selection = try LoRASuffixLayerSelection.resolve(
            requestedCount: 18, availableCount: 35)
        #expect(selection.startIndex == 17)
        #expect(selection.endIndex == 35)
        #expect(Array(selection.range) == Array(17 ..< 35))
        #expect(selection.range.count == 18)
    }

    @Test("allows selecting every available layer")
    func allLayers() throws {
        let selection = try LoRASuffixLayerSelection.resolve(
            requestedCount: 35, availableCount: 35)
        #expect(selection.range == 0 ..< 35)
    }

    @Test("rejects a count that Swift suffix would silently clamp")
    func rejectsOversizedCount() {
        #expect(throws: LoRASuffixLayerSelectionError.exceedsAvailableLayers(
            requested: 36, available: 35)) {
            try LoRASuffixLayerSelection.resolve(
                requestedCount: 36, availableCount: 35)
        }
    }

    @Test("rejects non-positive and empty-model bounds")
    func invalidBounds() {
        #expect(throws: LoRASuffixLayerSelectionError.invalidRequestedCount(0)) {
            try LoRASuffixLayerSelection.resolve(
                requestedCount: 0, availableCount: 35)
        }
        #expect(throws: LoRASuffixLayerSelectionError.modelHasNoLayers) {
            try LoRASuffixLayerSelection.resolve(
                requestedCount: 1, availableCount: 0)
        }
    }
}
