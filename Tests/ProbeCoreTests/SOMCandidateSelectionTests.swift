import Foundation
import Testing
@testable import ProbeCore

@Suite("SOM candidate selection")
struct SOMCandidateSelectionTests {
    @Test func omittedSpecificationPreservesExtractorSelection() throws {
        let result = makeResult(selected: [5, 2, 9, 7])
        let selections = try SOMCandidateSelector.resolve(result: result, rank: 3)

        #expect(selections.count == 1)
        #expect(selections[0].label == "coverage")
        #expect(selections[0].latticeIDs == [5, 2, 9])
        #expect(selections[0].directions == [
            result.candidateDirections[5],
            result.candidateDirections[2],
            result.candidateDirections[9],
        ])

        let blank = try SOMCandidateSelector.resolve(
            result: result, rank: 3, specification: "  \n")
        #expect(blank == selections)
    }

    @Test func resolvesBuiltinsAndExplicitOrderedIDs() throws {
        let result = makeResult(selected: [5, 2, 9, 7])
        let selections = try SOMCandidateSelector.resolve(
            result: result,
            rank: 3,
            specification: "coverage;coverage-reversed;occupancy;occupancy-reversed;ids:paper-best=7,6,3")

        #expect(selections.map(\.label) == [
            "coverage", "coverage-reversed", "occupancy",
            "occupancy-reversed", "paper-best",
        ])
        #expect(selections.map(\.latticeIDs) == [
            [5, 2, 9],
            [9, 2, 5],
            [15, 14, 13],
            [13, 14, 15],
            [7, 6, 3],
        ])
    }

    @Test func occupancyTiesUseQuantizationErrorThenID() throws {
        var occupancy = Array(repeating: 1, count: 16)
        occupancy[4] = 9
        occupancy[7] = 9
        occupancy[11] = 9
        var errors = (0 ..< 16).map { Float($0) / 10 + 0.1 }
        errors[4] = 0.3
        errors[7] = 0.2
        errors[11] = 0.2
        let result = makeResult(
            selected: [0, 1, 2], occupancy: occupancy, errors: errors)

        let selections = try SOMCandidateSelector.resolve(
            result: result, rank: 3, specification: "occupancy")
        #expect(selections.count == 1)
        let selection = selections[0]
        #expect(selection.latticeIDs == [7, 11, 4])
    }

    @Test func rejectsInvalidRankLabelsIDsAndUnusableCandidates() throws {
        let result = makeResult(selected: [5, 2, 9, 7])

        #expect(throws: SOMCandidateSelectionError.invalidRank(0)) {
            try SOMCandidateSelector.resolve(result: result, rank: 0)
        }
        #expect(throws: SOMCandidateSelectionError.invalidRank(17)) {
            try SOMCandidateSelector.resolve(result: result, rank: 17)
        }
        #expect(throws: SOMCandidateSelectionError.unsafeLabel("bad label")) {
            try SOMCandidateSelector.resolve(
                result: result, rank: 3, specification: "ids:bad label=1,2,3")
        }
        #expect(throws: SOMCandidateSelectionError.duplicateLabel("Trial")) {
            try SOMCandidateSelector.resolve(
                result: result, rank: 3,
                specification: "ids:trial=1,2,3;ids:Trial=4,5,6")
        }
        #expect(throws: SOMCandidateSelectionError.explicitRankMismatch(
            label: "short", expected: 3, actual: 2))
        {
            try SOMCandidateSelector.resolve(
                result: result, rank: 3, specification: "ids:short=1,2")
        }
        #expect(throws: SOMCandidateSelectionError.duplicateLatticeID(
            label: "dupe", id: 1))
        {
            try SOMCandidateSelector.resolve(
                result: result, rank: 3, specification: "ids:dupe=1,1,2")
        }
        #expect(throws: SOMCandidateSelectionError.latticeIDOutOfRange(
            label: "range", id: 16))
        {
            try SOMCandidateSelector.resolve(
                result: result, rank: 3, specification: "ids:range=1,2,16")
        }

        var occupancy = Array(1 ... 16)
        occupancy[4] = 0
        let unoccupied = makeResult(
            selected: [5, 2, 9], occupancy: occupancy,
            errors: (0 ..< 16).map { $0 == 4 ? .infinity : Float($0 + 1) / 10 })
        #expect(throws: SOMCandidateSelectionError.unusableLatticeID(
            label: "empty", id: 4))
        {
            try SOMCandidateSelector.resolve(
                result: unoccupied, rank: 3, specification: "ids:empty=1,4,2")
        }
    }

    @Test func rejectsInsufficientDefaultCoverageWithoutReinventingIt() throws {
        let result = makeResult(selected: [5, 2])
        #expect(throws: SOMCandidateSelectionError.insufficientCandidates(
            strategy: "coverage", requested: 3, available: 2))
        {
            try SOMCandidateSelector.resolve(result: result, rank: 3)
        }
    }

    @Test func diagnosticsEncodeEmptyNeuronErrorsAsNull() throws {
        var occupancy = Array(1 ... 16)
        occupancy[6] = 0
        var errors = (0 ..< 16).map { Float($0 + 1) / 10 }
        errors[6] = .infinity
        let result = makeResult(
            selected: [5, 2, 9], occupancy: occupancy, errors: errors,
            overallError: .infinity)
        let selections = try SOMCandidateSelector.resolve(
            result: result, rank: 3,
            specification: "coverage;ids:manual=7,8,3")
        let diagnostics = try SOMSelectionDiagnostics.project(
            result: result, selections: selections)

        #expect(diagnostics.latticeRows == 4)
        #expect(diagnostics.latticeColumns == 4)
        #expect(diagnostics.quantizationError == nil)
        #expect(diagnostics.extractorSelectedLatticeIDs == [5, 2, 9])
        #expect(diagnostics.candidates.count == 16)
        #expect(diagnostics.candidates[6].latticeID == 6)
        #expect(diagnostics.candidates[6].row == 1)
        #expect(diagnostics.candidates[6].column == 2)
        #expect(diagnostics.candidates[6].occupancy == 0)
        #expect(diagnostics.candidates[6].quantizationError == nil)
        #expect(!diagnostics.candidates[6].usable)
        #expect(diagnostics.selectedSets == [
            SOMSelectedSetDiagnostic(label: "coverage", latticeIDs: [5, 2, 9]),
            SOMSelectedSetDiagnostic(label: "manual", latticeIDs: [7, 8, 3]),
        ])

        let encoded = try JSONEncoder().encode(diagnostics)
        let decoded = try JSONDecoder().decode(SOMSelectionDiagnostics.self, from: encoded)
        #expect(decoded == diagnostics)
    }

    private func makeResult(
        selected: [Int],
        occupancy: [Int] = Array(1 ... 16),
        errors: [Float] = (0 ..< 16).map { Float($0 + 1) / 10 },
        overallError: Float = 0.25
    ) -> SOMDirectionResult {
        let candidates = (0 ..< 16).map { id -> [Float] in
            let angle = Float(id + 1) * 0.17
            return [cos(angle), sin(angle)]
        }
        return SOMDirectionResult(
            directions: selected.map { candidates[$0] },
            selectedNeuronIndices: selected,
            candidateDirections: candidates,
            neurons: candidates,
            occupancy: occupancy,
            quantizationErrorByNeuron: errors,
            quantizationError: overallError)
    }
}
