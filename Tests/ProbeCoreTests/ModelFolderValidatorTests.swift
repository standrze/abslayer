import Foundation
import Testing
@testable import ProbeCore

@Test func acceptsQuantizedAndBF16Folders() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let quantized = root.appendingPathComponent("quantized", isDirectory: true)
    let bf16 = root.appendingPathComponent("bf16", isDirectory: true)
    try makeModelFolder(quantized, dtype: "U8", quantized: true)
    try makeModelFolder(bf16, dtype: "BF16", quantized: false)

    let result = try ModelFolderValidator.validatePair(
        quantizedPath: quantized.path,
        fullBF16Path: bf16.path
    )
    #expect(result.quantized.detectedDTypes == ["U8"])
    #expect(result.fullBF16.detectedDTypes == ["BF16"])
    #expect(result.quantized.decoderLayerCount == 35)
    #expect(result.fullBF16.decoderLayerCount == 35)
}

@Test func prefersTextDecoderDepthOverOtherModalityDepths() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = root.appendingPathComponent("multimodal", isDirectory: true)
    try makeModelFolder(model, dtype: "BF16", quantized: false)

    let config: [String: Any] = [
        "model_type": "gemma4",
        "audio_config": ["num_hidden_layers": 12],
        "text_config": [
            "model_type": "gemma4_text",
            "num_hidden_layers": 35,
            "hidden_size": 2_560,
        ],
        "vision_config": ["num_hidden_layers": 16],
    ]
    try JSONSerialization.data(withJSONObject: config).write(
        to: model.appendingPathComponent("config.json"))

    let result = try ModelFolderValidator.validateFullBF16(path: model.path)
    #expect(result.decoderLayerCount == 35)
    #expect(result.hiddenSize == 2_560)
    #expect(result.modelType == "gemma4")
    #expect(result.textModelType == "gemma4_text")
}

@Test func rejectsFullPrecisionFolderInQuantizedSlot() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let first = root.appendingPathComponent("first", isDirectory: true)
    let second = root.appendingPathComponent("second", isDirectory: true)
    try makeModelFolder(first, dtype: "BF16", quantized: false)
    try makeModelFolder(second, dtype: "BF16", quantized: false)

    #expect(throws: ModelFolderValidationError.self) {
        try ModelFolderValidator.validatePair(
            quantizedPath: first.path,
            fullBF16Path: second.path
        )
    }
}

@Test func rejectsMixedOrMetadataQuantizedFolderInFullBF16Slot() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = root.appendingPathComponent("mixed", isDirectory: true)
    try makeModelFolder(model, dtype: "BF16", quantized: true)
    #expect(throws: ModelFolderValidationError.self) {
        try ModelFolderValidator.validateFullBF16(path: model.path)
    }
}

@Test func rejectsUnreferencedBF16DecoyForIndexedNonBF16Weights() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = root.appendingPathComponent("indexed", isDirectory: true)
    try makeModelMetadata(model)
    try writeSafeTensor(
        model.appendingPathComponent("real.safetensors"),
        tensors: ["real.weight": ("F16", [1], [0, 2])], payloadBytes: 2)
    try writeSafeTensor(
        model.appendingPathComponent("decoy.safetensors"),
        tensors: ["decoy.weight": ("BF16", [1], [0, 2])], payloadBytes: 2)
    try JSONSerialization.data(withJSONObject: [
        "weight_map": ["real.weight": "real.safetensors"],
    ]).write(to: model.appendingPathComponent("model.safetensors.index.json"))

    #expect(throws: ModelFolderValidationError.self) {
        try ModelFolderValidator.validateFullBF16(path: model.path)
    }
}

@Test func rejectsMalformedDuplicateOverlappingAndSymlinkedModelMembers() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: root) }

    func folder(_ name: String) throws -> URL {
        let value = root.appendingPathComponent(name, isDirectory: true)
        try makeModelMetadata(value)
        return value
    }

    let missing = try folder("missing-field")
    try writeRawSafeTensor(
        missing.appendingPathComponent("model.safetensors"),
        header: "{\"weight\":{\"shape\":[1],\"data_offsets\":[0,2]}}",
        payloadBytes: 2)
    #expect(throws: ModelFolderValidationError.self) {
        try ModelFolderValidator.validateFullBF16(path: missing.path)
    }

    let duplicate = try folder("duplicate-field")
    try writeRawSafeTensor(
        duplicate.appendingPathComponent("model.safetensors"),
        header: "{\"weight\":{\"dtype\":\"BF16\",\"d\\u0074ype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}",
        payloadBytes: 2)
    #expect(throws: ModelFolderValidationError.self) {
        try ModelFolderValidator.validateFullBF16(path: duplicate.path)
    }

    let overlap = try folder("overlap")
    try writeSafeTensor(
        overlap.appendingPathComponent("model.safetensors"),
        tensors: [
            "a": ("BF16", [1], [0, 2]),
            "b": ("BF16", [1], [1, 3]),
        ], payloadBytes: 3)
    #expect(throws: ModelFolderValidationError.self) {
        try ModelFolderValidator.validateFullBF16(path: overlap.path)
    }

    let outOfBounds = try folder("out-of-bounds")
    try writeSafeTensor(
        outOfBounds.appendingPathComponent("model.safetensors"),
        tensors: ["weight": ("BF16", [2], [0, 4])], payloadBytes: 2)
    #expect(throws: ModelFolderValidationError.self) {
        try ModelFolderValidator.validateFullBF16(path: outOfBounds.path)
    }

    let symlink = try folder("symlink")
    try writeSafeTensor(
        symlink.appendingPathComponent("model.safetensors"),
        tensors: ["weight": ("BF16", [1], [0, 2])], payloadBytes: 2)
    let tokenizer = symlink.appendingPathComponent("tokenizer_config.json")
    try FileManager.default.removeItem(at: tokenizer)
    try FileManager.default.createSymbolicLink(
        at: tokenizer,
        withDestinationURL: symlink.appendingPathComponent("config.json"))
    #expect(throws: ModelFolderValidationError.self) {
        try ModelFolderValidator.validateFullBF16(path: symlink.path)
    }
}

private func makeModelFolder(_ folder: URL, dtype: String, quantized: Bool) throws {
    try makeModelMetadata(folder, quantized: quantized)
    let width = dtype == "BF16" ? 2 : 1
    try writeSafeTensor(
        folder.appendingPathComponent("model.safetensors"),
        tensors: ["model.layers.0.weight": (dtype, [1], [0, width])],
        payloadBytes: width)
}

private func makeModelMetadata(_ folder: URL, quantized: Bool = false) throws {
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let config: [String: Any] = quantized
        ? [
            "model_type": "gemma3", "num_hidden_layers": 35, "hidden_size": 2_560,
            "quantization": ["group_size": 64, "bits": 4],
        ]
        : [
            "model_type": "gemma3", "num_hidden_layers": 35, "hidden_size": 2_560,
            "torch_dtype": "bfloat16",
        ]
    let configData = try JSONSerialization.data(withJSONObject: config)
    try configData.write(to: folder.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: folder.appendingPathComponent("tokenizer_config.json"))
}

private func writeSafeTensor(
    _ file: URL,
    tensors: [String: (dtype: String, shape: [Int], offsets: [Int])],
    payloadBytes: Int
) throws {
    let headerObject: [String: Any] = tensors.mapValues { value in
        [
            "dtype": value.dtype,
            "shape": value.shape,
            "data_offsets": value.offsets,
        ] as [String: Any]
    }
    let header = try JSONSerialization.data(withJSONObject: headerObject)
    try writeRawSafeTensor(
        file, headerData: header, payloadBytes: payloadBytes)
}

private func writeRawSafeTensor(
    _ file: URL, header: String, payloadBytes: Int
) throws {
    try writeRawSafeTensor(
        file, headerData: Data(header.utf8), payloadBytes: payloadBytes)
}

private func writeRawSafeTensor(
    _ file: URL, headerData header: Data, payloadBytes: Int
) throws {
    var size = UInt64(header.count).littleEndian
    var fileData = withUnsafeBytes(of: &size) { Data($0) }
    fileData.append(header)
    fileData.append(Data(repeating: 0, count: payloadBytes))
    try fileData.write(to: file)
}
