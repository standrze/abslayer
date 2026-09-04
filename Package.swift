// swift-tools-version: 6.3

import PackageDescription

#if os(macOS)
let backendSwiftSettings: [SwiftSetting] = [
    .define("MLX_METAL_BACKEND")
]
#elseif os(Linux)
let backendSwiftSettings: [SwiftSetting] =
    Context.environment["SPM_CUDA"] == "0"
    ? [.define("MLX_CPU_BACKEND")]
    : [.define("MLX_CUDA_BACKEND")]
#else
let backendSwiftSettings: [SwiftSetting] = [
    .define("MLX_CPU_BACKEND")
]
#endif

#if os(Linux)
// FoundationModels is an Apple-only framework. Passing an empty trait set is
// required because the pinned mlx-swift-lm revision enables it by default.
let mlxSwiftLMDependency: Package.Dependency = .package(
    url: "https://github.com/ml-explore/mlx-swift-lm",
    revision: "14414441fa44f45eee35a61e9fa0bab577cf9734",
    traits: []
)
#else
let mlxSwiftLMDependency: Package.Dependency = .package(
    url: "https://github.com/ml-explore/mlx-swift-lm",
    revision: "14414441fa44f45eee35a61e9fa0bab577cf9734"
)
#endif

var products: [Product] = [
    .executable(name: "abslayer", targets: ["ABSlayerBackend"]),
    .executable(name: "abslayer-cli", targets: ["ABSlayerCLI"]),
    .executable(name: "abslayer-eval", targets: ["ABSlayerEval"]),
    .executable(name: "abslayer-preflight", targets: ["ABSlayerPreflight"]),
    .executable(name: "abslayer-kl", targets: ["ABSlayerKL"]),
    .executable(
        name: "abslayer-continuation-kl",
        targets: ["ABSlayerContinuationKL"]),
    .executable(name: "abslayer-optimize", targets: ["ABSlayerOptimize"]),
    .executable(name: "abslayer-intervene", targets: ["ABSlayerIntervene"]),
    .executable(
        name: "abslayer-answer-transport",
        targets: ["ABSlayerAnswerTransport"]),
    .executable(
        name: "abslayer-matched-patch",
        targets: ["ABSlayerMatchedPatch"]),
    .executable(
        name: "abslayer-prompt-end-patch",
        targets: ["ABSlayerPromptEndPatch"]),
    .executable(name: "abslayer-judge", targets: ["ABSlayerJudge"]),
    .executable(name: "abslayer-dataset", targets: ["ABSlayerDataset"]),
    .executable(name: "abslayer-prefix-train", targets: ["ABSlayerPrefixTrain"]),
    .executable(name: "abslayer-lora-merge", targets: ["ABSlayerLoRAMerge"]),
    .executable(name: "abslayer-request", targets: ["ABSlayerRequest"]),
    .executable(name: "abslayer-lora-pipeline", targets: ["ABSlayerLoRAPipeline"]),
    .executable(name: "abslayer-training-data", targets: ["ABSlayerTrainingData"]),
    .executable(name: "abslayer-gemma-parity", targets: ["ABSlayerGemmaParity"]),
    .executable(
        name: "abslayer-rmsnorm-diagnostic",
        targets: ["ABSlayerRMSNormDiagnostic"]),
    .executable(
        name: "abslayer-directional-adapter",
        targets: ["ABSlayerDirectionalAdapter"]),
    .executable(
        name: "abslayer-preference-capture",
        targets: ["ABSlayerPreferenceCapture"]),
    .executable(
        name: "abslayer-first-token-screen",
        targets: ["ABSlayerFirstTokenScreen"]),
    .executable(
        name: "abslayer-som-search",
        targets: ["ABSlayerSOMSearch"]),
    .executable(
        name: "abslayer-som-multisource-search",
        targets: ["ABSlayerSOMMultiSourceSearch"]),
    .executable(name: "abslayer-ara", targets: ["ABSlayerARA"]),
]

var dependencies: [Package.Dependency] = [
    .package(
        url: "https://github.com/ml-explore/mlx-swift",
        exact: "0.31.6"),
    mlxSwiftLMDependency,
    .package(
        url: "https://github.com/huggingface/swift-huggingface",
        .upToNextMajor(from: "0.9.0")),
    .package(
        url: "https://github.com/huggingface/swift-transformers",
        .upToNextMajor(from: "1.3.0")),
    .package(
        url: "https://github.com/apple/swift-crypto.git",
        exact: "4.5.1"),
]

var targets: [Target] = [
    .target(
        name: "ProbeCore",
        dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXLinalg", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXOptimizers", package: "mlx-swift"),
            .product(name: "MLXLLM", package: "mlx-swift-lm"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "Crypto", package: "swift-crypto"),
        ],
        swiftSettings: backendSwiftSettings
    ),
    .executableTarget(name: "ABSlayerBackend", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerCLI", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerEval", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerPreflight", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerKL", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerContinuationKL", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerOptimize", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerIntervene", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerAnswerTransport", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerMatchedPatch", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerPromptEndPatch", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerJudge", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerDataset", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerARA", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerDirectionalAdapter", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerPreferenceCapture", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerFirstTokenScreen", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerSOMSearch", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerSOMMultiSourceSearch", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerPrefixTrain",
        dependencies: [
            "ProbeCore",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXOptimizers", package: "mlx-swift"),
            .product(name: "MLXLLM", package: "mlx-swift-lm"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
            .product(name: "Tokenizers", package: "swift-transformers"),
        ]
    ),
    .executableTarget(name: "ABSlayerLoRAMerge", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerRequest", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerLoRAPipeline", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerTrainingData", dependencies: ["ProbeCore"]),
    .executableTarget(name: "ABSlayerGemmaParity", dependencies: ["ProbeCore"]),
    .executableTarget(
        name: "ABSlayerRMSNormDiagnostic", dependencies: ["ProbeCore"]),
    .testTarget(name: "ProbeCoreTests", dependencies: ["ProbeCore"]),
]

#if os(macOS)
products.insert(
    .executable(name: "abslayer-probe", targets: ["ABSlayerProbe"]),
    at: 0)
dependencies.append(
    .package(url: "https://github.com/SwiftTUI/swift-tui.git", exact: "0.9.7"))
targets.append(
    .executableTarget(
        name: "ABSlayerProbe",
        dependencies: [
            "ProbeCore",
            .product(name: "SwiftTUI", package: "swift-tui"),
        ]
    ))
#endif

// The controller can be built/tested without resolving the model runtime.
// Default builds retain the original worker graph and dependency pins.
let harnessOnly = Context.environment["ABSLAYER_HARNESS_ONLY"] == "1"
let harnessCrypto: [Target.Dependency]
#if os(Linux)
harnessCrypto = [.product(name: "Crypto", package: "swift-crypto")]
#else
harnessCrypto = []
#endif
let harnessTargets: [Target] = [
    .target(name: "ABSlayerHarness", dependencies: harnessCrypto),
    .executableTarget(name: "ABSlayerJobHost", dependencies: ["ABSlayerHarness"]),
    .testTarget(name: "ABSlayerHarnessTests", dependencies: ["ABSlayerHarness"]),
]
let harnessProducts: [Product] = [
    .executable(name: "abslayer-job-host", targets: ["ABSlayerJobHost"]),
]
if harnessOnly {
    products = harnessProducts
    targets = harnessTargets
    #if os(Linux)
    dependencies = [.package(url: "https://github.com/apple/swift-crypto.git", exact: "4.5.1")]
    #else
    dependencies = []
    #endif
} else {
    products += harnessProducts
    targets += harnessTargets
}

let package = Package(
    name: "ABSlayerStudio",
    platforms: [.macOS(.v15)],
    products: products,
    dependencies: dependencies,
    targets: targets,
    swiftLanguageModes: [.v6]
)
