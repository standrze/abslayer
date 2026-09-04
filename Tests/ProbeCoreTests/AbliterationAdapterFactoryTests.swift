import MLX
@testable import ProbeCore
import Testing

struct AbliterationAdapterFactoryTests {
    @Test func noNormalizationFactorsMatchExactProjection() {
        assertParity(.none)
    }

    @Test func preNormalizationFactorsMatchExactProjection() {
        assertParity(.pre)
    }

    @Test func fullNormalizationRankThreeMatchesSmallExactEdit() {
        MLXRandom.seed(17)
        assertParity(.full)
    }

    @Test func sequentialNoNormalizationAdapterMatchesBF16Edit() {
        assertSequentialParity(.none, adapterRank: 2)
    }

    @Test func sequentialPreNormalizationAdapterMatchesBF16Edit() {
        assertSequentialParity(.pre, adapterRank: 2)
    }

    @Test func sequentialFullNormalizationAdapterMatchesBF16EditAtFullMatrixRank() {
        MLXRandom.seed(17)
        // This fixture has only three columns, so rank three exactly represents
        // the otherwise potentially full-rank row-renormalization delta.
        assertSequentialParity(.full, adapterRank: 3)
    }

    private func assertParity(_ normalization: WeightNormalization) {
        let rows = 4
        let columns = 3
        let values: [Float] = [
            0.2, -0.8, 1.1,
            1.5, 0.3, -0.2,
            -0.4, 0.9, 0.7,
            0.6, -1.2, 0.1,
        ]
        let direction = AbliterationMath.normalized([0.4, -0.2, 0.8, 0.1])
        let strength: Float = 0.73
        let weight = MLXArray(values).reshaped(rows, columns)
        let factors = AbliterationAdapterFactory.factors(
            weight: weight, basis: [direction], strength: strength,
            normalization: normalization, adapterRank: normalization == .full ? 3 : 1)
        let reconstructed = weight + matmul(factors.b.T, factors.a.T)
        eval(reconstructed)
        let expected = AbliterationMath.edit(
            matrix: values, rows: rows, columns: columns,
            direction: direction, strength: strength, normalization: normalization)
        let actual = reconstructed.asArray(Float.self)
        #expect(actual.count == expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            // Metal reductions and the CPU QR/SVD boundary are not bit-identical.
            #expect(abs(lhs - rhs) < 2e-3)
        }
    }


    private func assertSequentialParity(
        _ normalization: WeightNormalization, adapterRank: Int
    ) {
        let rows = 4
        let columns = 3
        let values: [Float] = [
            0.2, -0.8, 1.1,
            1.5, 0.3, -0.2,
            -0.4, 0.9, 0.7,
            0.6, -1.2, 0.1,
        ]
        // Deliberately non-orthogonal and non-unit to exercise the SOM path's
        // ordered per-direction normalization semantics.
        let basis: [[Float]] = [
            [0.4, -0.2, 0.8, 0.1],
            [0.7, 0.3, 0.2, -0.4],
        ]
        let strength: Float = 0.73
        let weight = MLXArray(values).reshaped(rows, columns)
        let factors = AbliterationAdapterFactory.factors(
            weight: weight, basis: basis, strength: strength,
            normalization: normalization, composition: .sequential,
            adapterRank: adapterRank)
        let reconstructed = weight + matmul(factors.b.T, factors.a.T)
        let bf16 = BF16WeightEditor.editMatrix(
            weight, basis: basis, strength: strength,
            normalization: normalization, composition: .sequential)
            .asType(.float32)
        eval(reconstructed, bf16)
        let actual = reconstructed.asArray(Float.self)
        let expected = bf16.asArray(Float.self)
        #expect(actual.count == expected.count)
        // The exact low-rank algebra sums the same rank-one terms through one
        // GEMM while the BF16 path applies them as separate GEMMs, so Metal's
        // accumulation order differs slightly even in float32.
        let tolerance: Float = normalization == .full ? 3e-3 : 5e-4
        for (lhs, rhs) in zip(actual, expected) {
            #expect(abs(lhs - rhs) < tolerance)
        }
    }
}
