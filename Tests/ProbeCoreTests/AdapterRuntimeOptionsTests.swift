import Testing
@testable import ProbeCore

@Suite("Adapter runtime environment parsing")
struct AdapterRuntimeOptionsTests {
    @Test("accepts an adapter without a scale override")
    func adapterWithoutScale() throws {
        let options = try AdapterRuntimeOptions.parse(environment: [
            "ABSLAYER_ADAPTER_DIR": "/tmp/adapter"
        ])
        #expect(options.directory == "/tmp/adapter")
        #expect(options.scaleOverride == nil)
    }

    @Test("accepts finite non-negative scales and surrounding whitespace")
    func validScales() throws {
        let zero = try AdapterRuntimeOptions.parse(environment: [
            "ABSLAYER_ADAPTER_DIR": "/tmp/adapter",
            "ABSLAYER_ADAPTER_SCALE": "0",
        ])
        #expect(zero.scaleOverride == 0)

        let positive = try AdapterRuntimeOptions.parse(environment: [
            "ABSLAYER_ADAPTER_DIR": "/tmp/adapter",
            "ABSLAYER_ADAPTER_SCALE": " 8.5 ",
        ])
        #expect(positive.scaleOverride == 8.5)
    }

    @Test("rejects malformed, nonfinite, and negative scales")
    func invalidScales() {
        for rawValue in ["", "garbage", "nan", "inf", "-1"] {
            #expect(throws: AdapterRuntimeOptionsError.invalidScale(rawValue)) {
                try AdapterRuntimeOptions.parse(environment: [
                    "ABSLAYER_ADAPTER_DIR": "/tmp/adapter",
                    "ABSLAYER_ADAPTER_SCALE": rawValue,
                ])
            }
        }
    }

    @Test("rejects a scale without a usable adapter directory")
    func scaleWithoutAdapter() {
        #expect(throws: AdapterRuntimeOptionsError.scaleRequiresAdapterDirectory) {
            try AdapterRuntimeOptions.parse(environment: [
                "ABSLAYER_ADAPTER_SCALE": "8"
            ])
        }
        #expect(throws: AdapterRuntimeOptionsError.scaleRequiresAdapterDirectory) {
            try AdapterRuntimeOptions.parse(environment: [
                "ABSLAYER_ADAPTER_DIR": "  ",
                "ABSLAYER_ADAPTER_SCALE": "8",
            ])
        }
    }
}
