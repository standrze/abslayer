import Foundation
import ProbeCore

@main
enum ABSlayerLoRAMerge {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count == 4 else {
            FileHandle.standardError.write(Data(
                "usage: abslayer-lora-merge BF16_MODEL ADAPTER_DIR OUTPUT_MODEL\n".utf8))
            exit(2)
        }
        let source = try ModelFolderValidator.validateFullBF16(path: args[1])
        print("Merging adapter into BF16 weights shard by shard…")
        let summary = try LoRABF16Merger.merge(
            sourcePath: source.path, adapterPath: args[2], outputPath: args[3])
        print("Merged \(summary.mergedMatrices) matrices -> \(summary.outputPath)")
    }
}
