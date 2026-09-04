import Foundation
import Testing
@testable import ProbeCore

@Suite("Harness training dataset")
struct HarnessTrainingDatasetTests {
    @Test func loadsVersionedDocumentRecords() throws {
        let dataset = HarnessTrainingDataset(
            datasetName: "manual",
            records: [
                HarnessTrainingRecord(
                    id: "p3-a", instruction: "Explain the reset procedure",
                    response: "First disconnect power.",
                    provenance: HarnessTrainingProvenance(document: "manual.pdf", pages: [3])),
                HarnessTrainingRecord(
                    id: "p4-b", instruction: "Explain the startup procedure",
                    response: "Reconnect power and press Start."),
            ])
        let data = try JSONEncoder().encode(dataset)
        let examples = try HarnessTrainingDatasetLoader.load(data)
        #expect(examples == [
            LoRATrainingExample(
                prompt: "Explain the reset procedure", target: "First disconnect power."),
            LoRATrainingExample(
                prompt: "Explain the startup procedure", target: "Reconnect power and press Start."),
        ])
    }

    @Test func rejectsDuplicateIDs() throws {
        let record = HarnessTrainingRecord(id: "same", instruction: "Q", response: "A")
        let data = try JSONEncoder().encode(HarnessTrainingDataset(
            datasetName: "duplicate", records: [record, record]))
        #expect(throws: HarnessTrainingDatasetError.duplicateRecordID("same")) {
            try HarnessTrainingDatasetLoader.load(data)
        }
    }

    @Test func rejectsMixedSFTAndPreferenceRows() throws {
        let data = try JSONEncoder().encode(HarnessTrainingDataset(
            datasetName: "mixed", records: [
                HarnessTrainingRecord(id: "a", instruction: "Q1", response: "A1"),
                HarnessTrainingRecord(
                    id: "b", instruction: "Q2", response: "A2", rejectedResponse: "bad"),
            ]))
        #expect(throws: HarnessTrainingDatasetError.mixedObjectives) {
            try HarnessTrainingDatasetLoader.load(data)
        }
    }
}
