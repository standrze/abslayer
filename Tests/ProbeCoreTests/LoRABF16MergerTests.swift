import MLX
import Testing
@testable import ProbeCore

@Suite("LoRA BF16 merger")
struct LoRABF16MergerTests {
    @Test func appliesScaledLowRankDelta() throws {
        let weight = MLXArray([Float](repeating: 1, count: 6)).reshaped(2, 3)
        let a = MLXArray([Float(1), 2, 3]).reshaped(3, 1)
        let b = MLXArray([Float(4), 5]).reshaped(1, 2)
        let merged = try LoRABF16Merger.mergedMatrix(
            weight: weight, loraA: a, loraB: b, scale: 0.5)
        eval(merged)
        #expect(merged.asArray(Float.self) == [3, 5, 7, 3.5, 6, 8.5])
    }

    @Test func rejectsIncompatibleShapes() {
        #expect(throws: LoRAMergeError.self) {
            try LoRABF16Merger.mergedMatrix(
                weight: MLXArray.zeros([2, 3]),
                loraA: MLXArray.zeros([4, 1]),
                loraB: MLXArray.zeros([1, 2]), scale: 1)
        }
    }
}
