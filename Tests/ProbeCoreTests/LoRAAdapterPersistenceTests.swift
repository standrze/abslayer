import Testing
@testable import ProbeCore

@Suite("LoRA adapter persistence")
struct LoRAAdapterPersistenceTests {
    @Test("accepts a flat manifest filename")
    func acceptsManifestName() throws {
        try LoRAAdapterPersistence.validateAdditionalFileName(
            "abslayer_manifest.json")
    }

    @Test("rejects traversal and core adapter filenames")
    func rejectsUnsafeNames() {
        for name in [
            "../manifest.json", "nested/manifest.json",
            "adapter_config.json", "adapters.safetensors",
        ] {
            #expect(throws: LoRAAdapterPersistenceError.invalidAdditionalFileName(name)) {
                try LoRAAdapterPersistence.validateAdditionalFileName(name)
            }
        }
    }

    @Test("O-only layer 20 uses the minimal suffix and no MLP key")
    func sparseOProjectionCoverage() throws {
        let coverage = try AbliterationAdapterFactory.sparseCoverage(
            layerCount: 35, targetLayers: [20],
            includesAttention: true, includesMLP: false)
        #expect(coverage.numLayers == 15)
        #expect(coverage.keys == ["self_attn.o_proj"])
    }
}
