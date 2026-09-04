import Foundation
import Testing

@Suite("Linux and CUDA portability contract")
struct LinuxPortabilityTests {
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    @Test("Darwin command entry points provide a Glibc fallback")
    func conditionalCLibImports() throws {
        let sources = packageRoot.appendingPathComponent("Sources")
        let enumerator = try #require(FileManager.default.enumerator(
            at: sources,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]))
        var conditionalImportCount = 0
        var uncovered = [String]()
        for case let file as URL in enumerator where file.pathExtension == "swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
            guard text.contains("import Darwin") else { continue }
            conditionalImportCount += 1
            if !text.contains("#if canImport(Darwin)")
                || !text.contains("#elseif canImport(Glibc)")
                || !text.contains("import Glibc")
            {
                uncovered.append(file.path)
            }
        }
        #expect(conditionalImportCount > 0)
        #expect(uncovered.isEmpty, "unportable C-library imports: \(uncovered)")
    }

    @Test("CryptoKit call sites provide swift-crypto imports")
    func portableCryptoImports() throws {
        let probeCore = packageRoot
            .appendingPathComponent("Sources/ProbeCore")
        let enumerator = try #require(FileManager.default.enumerator(
            at: probeCore,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]))
        var cryptoImportCount = 0
        var uncovered = [String]()
        for case let file as URL in enumerator where file.pathExtension == "swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
            guard text.contains("import CryptoKit") else { continue }
            cryptoImportCount += 1
            if !text.contains("#if canImport(CryptoKit)")
                || !text.contains("#elseif canImport(Crypto)")
                || !text.contains("import Crypto")
            {
                uncovered.append(file.path)
            }
        }
        #expect(cryptoImportCount == 4)
        #expect(uncovered.isEmpty, "unportable CryptoKit imports: \(uncovered)")
    }

    @Test("manifest pins MLX and excludes Apple-only UI and traits on Linux")
    func manifestContract() throws {
        let text = try String(
            contentsOf: packageRoot.appendingPathComponent("Package.swift"),
            encoding: .utf8)
        #expect(text.contains("exact: \"0.31.6\""))
        #expect(text.contains("#if os(Linux)"))
        #expect(text.contains("traits: []"))
        #expect(text.contains("#if os(macOS)"))
        #expect(text.contains("name: \"ABSlayerProbe\""))
        #expect(text.contains("name: \"abslayer\""))
        #expect(text.contains("name: \"ABSlayerBackend\""))
        #expect(text.contains("name: \"abslayer-preflight\""))
        #expect(text.contains("name: \"ABSlayerPreflight\""))
        #expect(text.contains("name: \"abslayer-rmsnorm-diagnostic\""))
        #expect(text.contains("name: \"ABSlayerRMSNormDiagnostic\""))
        #expect(text.contains(".product(name: \"Crypto\", package: \"swift-crypto\")"))
        #expect(text.contains(".define(\"MLX_CUDA_BACKEND\")"))
    }

    @Test("RTX 4090 wrapper is bounded and checks its toolchain")
    func cudaBuildWrapperContract() throws {
        let script = packageRoot.appendingPathComponent("build-cuda-4090.sh")
        let text = try String(contentsOf: script, encoding: .utf8)
        #expect(text.contains("SWIFT_BUILD_JOBS=\"${SWIFT_BUILD_JOBS:-1}\""))
        #expect(text.contains("export SPM_CUDA=1"))
        #expect(text.contains("export CUDA_ARCH=sm_89"))
        #expect(text.contains("nvcc --list-gpu-code"))
        #expect(text.contains("cudnn-frontend-v1.16.0"))
        #expect(text.contains("cutlass-v4.3.5"))
        #expect(text.contains("cutlass/cutlass.h"))
        #expect(text.contains("cutlass/version.h"))
        #expect(text.contains("cute/tensor.hpp"))
        #expect(text.contains("CUTLASS_MAJOR[[:space:]]+4"))
        #expect(text.contains("CUTLASS_MINOR[[:space:]]+3"))
        #expect(text.contains("CUTLASS_PATCH[[:space:]]+5"))
        #expect(text.contains("CUDA_CCCL_INCLUDE_DIR=\"$(readlink -f /usr/local/cuda/include/cccl)\""))
        #expect(text.contains("export CPATH=\"$CUDA_CCCL_INCLUDE_DIR:$CUTLASS_INCLUDE_DIR:$CUDNN_FRONTEND_INCLUDE_DIR"))
        #expect(text.contains(":cccl=$CUDA_CCCL_INCLUDE_DIR:"))
        #expect(text.contains(":cutlass=$CUTLASS_INCLUDE_DIR:cutlass-version="))
        #expect(text.contains("abslayer_resolve_cuda_host_cxx"))
        #expect(text.contains("host-cxx=$ABSLAYER_CUDA_HOST_CXX_PATH"))
        #expect(text.contains("host-cxx-version=$ABSLAYER_CUDA_HOST_CXX_VERSION_FINGERPRINT"))
        #expect(text.contains("ABSLAYER_SWIFT_BUILD_SCRATCH_ARGS"))
        #expect(text.contains("PROFILE_MARKER=\"$ABSLAYER_SWIFTPM_SCRATCH_PATH/.abslayer-build-profile\""))
        #expect(text.contains("CACHE_PROFILE_MARKER=\"$ABSLAYER_SWIFTPM_SCRATCH_PATH/.abslayer-cache-profile\""))
        #expect(text.contains("mv -f \"$CACHE_PROFILE_TEMP\" \"$CACHE_PROFILE_MARKER\""))
        #expect(text.contains("mv -f \"$SUCCESS_PROFILE_TEMP\" \"$PROFILE_MARKER\""))
        #expect(text.contains("-c release"))
        #expect(text.contains("if [[ \"$BIN_DIR\" != */release ]]"))
        #expect(text.contains("RELEASE_EXECUTABLES=("))
        #expect(text.contains("  abslayer\n"))
        #expect(text.contains("abslayer-preflight"))
        #expect(text.contains("abslayer-rmsnorm-diagnostic"))

        let attributes = try FileManager.default.attributesOfItem(atPath: script.path)
        let permissions = try #require(attributes[.posixPermissions] as? NSNumber)
        #expect(permissions.intValue & 0o111 != 0)
    }

    @Test("dependency preparation applies every pinned MLX LM patch")
    func dependencyPatchContract() throws {
        let text = try String(
            contentsOf: packageRoot.appendingPathComponent("prepare-dependencies.sh"),
            encoding: .utf8)
        #expect(text.contains("EXPECTED_REVISION=\"14414441fa44f45eee35a61e9fa0bab577cf9734\""))
        #expect(text.contains("MLX_SWIFT_EXPECTED_REVISION=\"0bb916c67f4b9e5c682cbe02a42c701c93ab5021\""))
        #expect(text.contains("mlx-swift-cuda-host-cxx.patch"))
        #expect(text.contains("mlx-cuda-rms-norm-small-row.patch"))
        #expect(text.contains("a5a684db596c117f13f7bacaea9902d0ad6d28a6"))
        #expect(text.contains("ABSLAYER_SWIFTPM_SCRATCH_PATH/checkouts/mlx-swift"))
        #expect(text.contains("mlx-swift-lm-gemma4-layer-states.patch"))
        #expect(text.contains("mlx-swift-lm-gemma4-residual-intervention.patch"))
        #expect(text.contains("mlx-swift-lm-gemma4-attention-projection-io.patch"))
        #expect(text.contains("mlx-swift-lm-ignore-readmes.patch"))
        #expect(text.contains("mlx-swift-lm-lora-numerics.patch"))
    }
}
