import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Performs the final evaluation of a trained model
 * on an independent test dataset (blind test).
 *
 *
 * This class loads a previously trained classifier
 * and evaluates it on data it has never seen,
 * to measure the real generalization ability of the model.
 *
 *
 *
 * This evaluation does not tune any parameter,
 * and that is why it is the most reliable one.
 *
 *
 * @version 1.0
 */
public class TestModel {
    /**
     * Program entry point.
     *
     *
     * This method:
     * <ul>
     * <li>Loads the test data</li>
     * <li>Loads the previously trained model</li>
     * <li>Evaluates the model on unseen data</li>
     * <li>Computes the main metrics (Accuracy, F-Measure, etc.)</li>
     * <li>Shows the results on screen and saves them to a file</li>
     * </ul>
     *
     *
     * @param args Arguments:
     *             {@code <test_final.arff> <model.model> <results.txt> }
     */
    public static void main(String[] args) {
        try {
            if (args.length > 3) {
                System.err.println(
                        "Usage: java -cp \"lib/weka.jar:bin\" TestModel <test_final.arff> <model.model> <results.txt>");
                return;
            }

            // Default values
            String testPath = args.length >= 1 ? args[0] : "data/splits/test_final.arff";
            String modelPath = args.length >= 2 ? args[1] : "model/mlp.model";
            String resultsPath = args.length == 3 ? args[2] : "results/test_results.txt";

            System.out.println("==================================================");
            System.out.println("BLIND EVALUATION CONFIGURATION (Blind Test):");
            System.out.println("Test file       : " + testPath);
            System.out.println("Model (.model)  : " + modelPath);
            System.out.println("Results (.txt)  : " + resultsPath);
            System.out.println("==================================================\n");

            if (!new File(testPath).exists() || !new File(modelPath).exists()) {
                System.err.println("ERROR: One of the files does not exist.");
                return;
            }

            // Load the data and the model
            Instances test = new DataSource(testPath).getDataSet();
            EmailLoader.ensureClassIndex(test);

            Classifier model = (Classifier) SerializationHelper.read(modelPath);

            // Evaluation
            System.out.println("Evaluating the model on unseen data...\n");
            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(model, test);

            // Prepare the results
            int spamIndex = test.classAttribute().indexOfValue("SPAM");
            if (spamIndex == -1)
                spamIndex = test.classAttribute().indexOfValue("spam");

            StringBuilder fullReport = new StringBuilder();
            fullReport.append("==================================================\n");
            fullReport.append("               EVALUATION RESULTS                  \n");
            fullReport.append("==================================================\n\n");

            fullReport.append("--- OVERALL METRICS ---\n");
            fullReport.append(String.format("Correct Percentage (Accuracy): %.2f%%\n", eval.pctCorrect()));
            fullReport.append(String.format("Weighted F-Measure (WAvg)   : %.4f\n\n", eval.weightedFMeasure()));

            if (spamIndex != -1) {
                fullReport.append("--- SPAM CLASS METRICS ---\n");
                fullReport.append(String.format("Precision               : %.4f\n", eval.precision(spamIndex)));
                fullReport.append(String.format("Recall                  : %.4f\n", eval.recall(spamIndex)));
                fullReport.append(String.format("F-Measure               : %.4f\n\n", eval.fMeasure(spamIndex)));
            }

            fullReport.append(eval.toMatrixString("--- CONFUSION MATRIX ---"));
            fullReport.append("\n==================================================");

            System.out.println(fullReport.toString());

            // Save the file
            File resultsFile = new File(resultsPath);
            if (resultsFile.getParentFile() != null) {
                resultsFile.getParentFile().mkdirs();
            }
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFile))) {
                writer.write(fullReport.toString());
            }

            // Save the log of predictions
            // Fix C4: only replace the ".txt" suffix; if there is no suffix, append one
            // (before, a path without ".txt" made this file overwrite the report)
            String predFilePath = resultsPath.endsWith(".txt")
                    ? resultsPath.substring(0, resultsPath.length() - ".txt".length()) + "_predictions.txt"
                    : resultsPath + "_predictions.txt";
            try (BufferedWriter predWriter = new BufferedWriter(new FileWriter(predFilePath))) {
                for (int i = 0; i < test.numInstances(); i++) {
                    double realClassIdx = test.instance(i).classValue();
                    String realClass = test.classAttribute().value((int) realClassIdx);
                    double predClassIdx = model.classifyInstance(test.instance(i));
                    String predClass = test.classAttribute().value((int) predClassIdx);
                    String line = String.format("%3d. instance:\tReal class: %-8s Predicted class: %-8s\n",
                        (i + 1), realClass, predClass);
                    predWriter.write(line);
                }
            }

            // Experiment tracking (optional)
            ExperimentLogger.log("4. Blind Evaluation (TestModel)", "Model: " + modelPath,
                    "\n" + fullReport.toString());

            System.out.println("\nSUCCESS! The detailed results report was saved here: " + resultsPath);

        } catch (Exception e) {
            System.err.println("Critical error while running the evaluation.");
            e.printStackTrace();
        }
    }
}
