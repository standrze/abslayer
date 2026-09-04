import AppKit
import Foundation
import ProbeCore
import SwiftTUI

@main
struct ABSlayerProbeApp: App, SwiftTUICommand {
    @OptionGroup(title: "SwiftTUI Options") var swiftTUIOptions: SwiftTUIOptions
    @Option(name: .long, help: "Folder containing the quantized model")
    var quantized = ""
    @Option(name: .long, help: "Folder containing the full BF16 model")
    var bf16 = ""
    @Flag(name: .long, help: "Use the BF16 folder for measurement as well as final editing")
    var bf16Only = false
    @Option(name: .long, help: "Path to the paired prompt JSON file")
    var prompts = "prompts.json"
    @Option(name: .long, help: "Separate paired holdout JSON used only for evaluation")
    var holdout = "holdout-pairs.json"
    @Option(name: .long, help: "Path for the complete text report")
    var output = "probe-report.txt"
    @Option(name: .long, help: "New folder for the edited BF16 checkpoint")
    var outputModel = ""

    var body: some Scene {
        WindowGroup("ABSlayer layer probe", id: WindowIdentifier("abslayer-probe")) {
            ProbeView(options: Options(
                quantizedPath: quantized,
                fullBF16Path: bf16,
                useBF16ForMeasurement: bf16Only,
                promptPath: prompts,
                holdoutPath: holdout,
                outputPath: output,
                outputModelPath: outputModel
            ))
        }
    }
}

struct Options: Sendable {
    let quantizedPath: String
    let fullBF16Path: String
    let useBF16ForMeasurement: Bool
    let promptPath: String
    let holdoutPath: String
    let outputPath: String
    let outputModelPath: String
}

struct ProbeView: View {
    let options: Options
    @State private var quantizedPath: String
    @State private var fullBF16Path: String
    @State private var useBF16ForMeasurement: Bool
    @State private var outputModelPath: String
    @State private var measurementPromptPath: String
    @State private var holdoutPath: String
    @State private var optimizeParameters = true
    @State private var trialCount = "200"
    @State private var startupTrialCount = "60"
    @State private var rowNormalization = "full"
    @State private var fullNormalizationRank = "3"
    @State private var winsorizationQuantile = "1.0"
    @State private var maximumRefusalRate = "0.10"
    @State private var finalEvaluationCases = "90"
    @State private var status = "Select both model folders to continue."
    @State private var output = ""
    @State private var isRunning = false

    init(options: Options) {
        self.options = options
        _quantizedPath = State(initialValue: options.quantizedPath)
        _fullBF16Path = State(initialValue: options.fullBF16Path)
        _useBF16ForMeasurement = State(initialValue: options.useBF16ForMeasurement)
        _outputModelPath = State(initialValue: options.outputModelPath)
        _measurementPromptPath = State(initialValue: options.promptPath)
        _holdoutPath = State(initialValue: options.holdoutPath)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text("ABSlayer")
                .foregroundStyle(.cyan)
            Text("Quantized model: fast residual measurement and trial evaluation")
            HStack {
                TextField("/path/to/quantized-model", text: $quantizedPath)
                Button("Browse…") {
                    if let path = chooseFolder(startingAt: quantizedPath) {
                        quantizedPath = path
                    }
                }
            }
            Toggle(
                "Use the full BF16 model for measurement too (no quantized copy required)",
                isOn: $useBF16ForMeasurement
            )
            Text("Full BF16 model: final weight editing and export")
            HStack {
                TextField("/path/to/full-bf16-model", text: $fullBF16Path)
                Button("Browse…") {
                    if let path = chooseFolder(startingAt: fullBF16Path) {
                        fullBF16Path = path
                    }
                }
            }
            Text("Output BF16 model: must be a new folder")
            TextField("/path/to/new-abliterated-model", text: $outputModelPath)
            Text("Measurement pairs (used to calculate directions)")
            TextField("/path/to/measurement-pairs.json", text: $measurementPromptPath)
            Text("Holdout pairs (never used to calculate directions)")
            TextField("/path/to/holdout-pairs.json", text: $holdoutPath)
            Toggle("Optimize strengths and layer kernels with TPE", isOn: $optimizeParameters)
            if optimizeParameters {
                HStack {
                    Text("Trials")
                    TextField("200", text: $trialCount)
                    Text("Random startup trials")
                    TextField("60", text: $startupTrialCount)
                }
                Text("Optimization checkpoints beside the output model and can resume after interruption.")
                HStack {
                    Text("Row normalization (none/pre/full)")
                    TextField("full", text: $rowNormalization)
                }
                HStack {
                    Text("Full-normalization adapter rank")
                    TextField("3", text: $fullNormalizationRank)
                }
                HStack {
                    Text("Winsorization quantile (1.0 disables)")
                    TextField("1.0", text: $winsorizationQuantile)
                }
                HStack {
                    Text("Maximum strict refusal rate for success")
                    TextField("0.10", text: $maximumRefusalRate)
                }
                HStack {
                    Text("Exact final-verification cases")
                    TextField("90", text: $finalEvaluationCases)
                }
            }
            Button(isRunning ? "Analyzing…" : "Validate folders and analyze") {
                guard !isRunning else { return }
                validateAndRun()
            }
            Text(status)
            Text("")
            ScrollView {
                Text(output)
            }
        }
        .padding(1)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    @MainActor
    private func validateAndRun() {
        do {
            guard !outputModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw UIError.missingOutputPath
            }
            let measurementPath: String
            if useBF16ForMeasurement {
                let inspection = try ModelFolderValidator.validateFullBF16(path: fullBF16Path)
                measurementPath = inspection.path
                status = "Validated \(inspection.weightFiles) BF16 weight file(s). Loading BF16 model…"
            } else {
                let inspection = try ModelFolderValidator.validatePair(
                    quantizedPath: quantizedPath,
                    fullBF16Path: fullBF16Path
                )
                measurementPath = inspection.quantized.path
                status = "Validated \(inspection.quantized.weightFiles) quantized and "
                    + "\(inspection.fullBF16.weightFiles) BF16 weight file(s). Loading quantized model…"
            }
            isRunning = true
            Task { @MainActor in
                do {
                    let pairs = try PromptFile.load(measurementPromptPath)
                    if optimizeParameters {
                        guard let trials = Int(trialCount), trials > 0,
                              let startupTrials = Int(startupTrialCount),
                              (0 ... trials).contains(startupTrials)
                        else {
                            throw UIError.invalidTrialCount
                        }
                        guard let normalization = WeightNormalization(
                            rawValue: rowNormalization.lowercased()),
                              let normalizationRank = Int(fullNormalizationRank),
                              normalizationRank > 0,
                              let winsor = Float(winsorizationQuantile),
                              (0 ... 1).contains(winsor),
                              let refusalLimit = Double(maximumRefusalRate),
                              (0 ... 1).contains(refusalLimit),
                              let finalCases = Int(finalEvaluationCases), finalCases > 0
                        else { throw UIError.invalidTechniqueSettings }
                        status = "Optimizing \(trials) trials. This can take a while…"
                        let workPath = outputModelPath + ".study"
                        let evaluationPairs = try PromptFile.load(holdoutPath)
                        let study = try await OptimizationEngine.run(OptimizationRequest(
                            sourceModel: fullBF16Path,
                            measurementModel: measurementPath,
                            measurementPairs: pairs,
                            evaluationPairs: evaluationPairs,
                            workDirectory: workPath,
                            outputModel: outputModelPath,
                            trialCount: trials,
                            measurementCases: min(32, pairs.count),
                            evaluationCases: min(20, evaluationPairs.count),
                            finalEvaluationCases: min(finalCases, evaluationPairs.count),
                            startupTrialCount: startupTrials,
                            maximumRefusalRate: refusalLimit,
                            normalization: normalization,
                            fullNormalizationRank: normalizationRank,
                            winsorizationQuantile: winsor < 1 ? winsor : nil))
                        if let best = study.best {
                            if let final = study.finalVerification {
                                status = String(
                                    format: "Exported; exact BF16 overall %@ (utility %@, abliteration %@). Refusal %.1f%%, control regression %.1f%%, KL %.4f",
                                    final.passedAll ? "PASS" : "FAIL",
                                    final.passedGuardrails ? "PASS" : "FAIL",
                                    final.passedAbliteration == true ? "PASS" : "FAIL",
                                    final.metrics.refusalRate * 100,
                                    final.metrics.controlFailureRate * 100,
                                    final.metrics.firstTokenKL)
                            } else {
                                status = String(
                                    format: "Dry run complete. Best trial %d: refusal %.1f%%, control failures %.1f%%, KL %.4f",
                                    best.index + 1, best.metrics.refusalRate * 100,
                                    best.metrics.controlFailureRate * 100,
                                    best.metrics.firstTokenKL)
                            }
                        }
                        output = "Study checkpoint: \(workPath)\nOutput model: \(outputModelPath)"
                        isRunning = false
                        return
                    }
                    let report = try await ProbeEngine(
                        modelDirectory: measurementPath,
                        pairs: pairs
                    ).run()
                    output = report.rendered
                    try report.rendered.write(
                        toFile: options.outputPath, atomically: true, encoding: .utf8
                    )
                    let peak = report.layers.max(by: {
                        $0.cosineDistance * max(0, $0.directionAgreement)
                            * max(0, $0.medianDirectionAgreement) * max(0, $0.silhouette)
                            < $1.cosineDistance * max(0, $1.directionAgreement)
                            * max(0, $1.medianDirectionAgreement) * max(0, $1.silhouette)
                    })?.layer ?? max(1, report.layers.count / 2)
                    let summary = try BF16WeightEditor.edit(
                        sourcePath: fullBF16Path,
                        outputPath: outputModelPath,
                        directions: report.directions,
                        subspaces: report.subspaces,
                        configuration: AbliterationConfiguration(
                            attention: LayerAblationKernel(
                                maximum: 1, peakLayer: Float(peak - 1), minimum: 0.3, radius: 25),
                            mlp: LayerAblationKernel(
                                maximum: 0.15, peakLayer: Float(peak - 1), minimum: 0.03, radius: 15),
                            normalization: .full))
                    status = "Complete. Edited \(summary.editedAttentionMatrices) attention and "
                        + "\(summary.editedMLPMatrices) MLP matrices. Report: \(options.outputPath)"
                } catch {
                    status = "Analysis error"
                    output = error.localizedDescription
                }
                isRunning = false
            }
        } catch {
            status = "Folder validation failed"
            output = error.localizedDescription
        }
    }

    @MainActor
    private func chooseFolder(startingAt path: String) -> String? {
        let panel = NSOpenPanel()
        panel.title = "Select model folder"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        let expanded = NSString(string: path).expandingTildeInPath
        if !expanded.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: expanded, isDirectory: true)
        }
        return panel.runModal() == .OK ? panel.url?.path : nil
    }
}

private enum UIError: LocalizedError {
    case invalidTrialCount
    case missingOutputPath
    case invalidTechniqueSettings
    var errorDescription: String? {
        switch self {
        case .invalidTrialCount: "Trial count must be a positive integer."
        case .missingOutputPath: "Choose a new output model folder."
        case .invalidTechniqueSettings:
            "Use row normalization none/pre/full, a positive adapter rank, and a winsorization quantile from 0 to 1."
        }
    }
}
