import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Optimizes the parameters (fine-tuning) of the
 * Multilayer Perceptron (MLP) algorithm.
 *
 *
 * This class uses the training (train) and validation (dev) datasets
 * to evaluate different parameter combinations
 * (grid search), and chooses the model with the best performance.
 *
 *
 *
 * The main optimization criterion is the F-Measure of the spam class.
 *
 *
 *
 * Finally, the best model is saved to disk and a detailed
 * report of the results is generated.
 *
 *
 * @version 1.0
 */
public class GetModel {
    /**
     * Program entry point.
     *
     *
     * This method:
     * <ul>
     * <li>Loads the data (train and dev)</li>
     * <li>Runs a grid search over the parameters</li>
     * <li>Trains and evaluates each model</li>
     * <li>Selects the model that maximizes the F-Measure (spam)</li>
     * <li>Saves the best model to a file</li>
     * </ul>
     *
     *
     * @param args Arguments:
     *             {@code <train_final.arff> <dev_final.arff> <model.model> <results.txt>}
     * @throws Exception If an error happens during the process
     */
    public static void main(String[] args) throws Exception {
        if (args.length > 4) {
            System.err.println(
                    "Usage: java -cp \"lib/weka.jar:bin\" GetModel <train_final.arff> <dev_final.arff> <model.model> <results.txt>");
            return;
        }

        String trainPath = args.length >= 1 ? args[0] : "data/splits/train_final.arff";
        String devPath = args.length >= 2 ? args[1] : "data/splits/dev_final.arff";
        String modelPath = args.length >= 3 ? args[2] : "model/mlp.model";
        String resultsPath = args.length == 4 ? args[3] : "results/finetuning.txt";

        System.out.println("==================================================");
        System.out.println("MLP FINE-TUNING CONFIGURATION:");
        System.out.println("Train file       : " + trainPath);
        System.out.println("Dev file         : " + devPath);
        System.out.println("Model (.model)   : " + modelPath);
        System.out.println("Results (.txt)   : " + resultsPath);
        System.out.println("==================================================\n");

        // ================================================
        // 2. LOAD THE DATA
        // ================================================
        Instances train = new DataSource(trainPath).getDataSet();
        Instances dev = new DataSource(devPath).getDataSet();

        EmailLoader.ensureClassIndex(train);
        EmailLoader.ensureClassIndex(dev);

        System.out.println("Train: " + train.numInstances() + " instances loaded.");
        System.out.println("Dev  : " + dev.numInstances() + " instances loaded.\n");

        int spamIndex = train.classAttribute().indexOfValue("SPAM");
        if (spamIndex == -1)
            spamIndex = train.classAttribute().indexOfValue("spam");

        // Search space (grid search)
        double[] learningRates = { 0.003, 0.005, 0.01 };
        double[] momentums = { 0.1, 0.2, 0.3 };
        String[] hiddenLayers = { "3", "5", "10" };
        int[] epochs = { 200, 300, 500 };

        MultilayerPerceptron bestModel = null;
        double bestFMeasure = -1;
        String bestParams = "";
        double bestTime = 0.0;

        File resultsFile = new File(resultsPath);
        if (resultsFile.getParentFile() != null) {
            resultsFile.getParentFile().mkdirs();
        }

        BufferedWriter writer = new BufferedWriter(new FileWriter(resultsPath));
        writer.write("=== MultilayerPerceptron FINE-TUNING ===\n");
        writer.write("Optimization metric: F-Measure (spam)\n");
        writer.write("Train: " + trainPath + "\n");
        writer.write("Dev  : " + devPath + "\n\n");
        writer.write(String.format("%-15s %-12s %-14s %-8s %-12s %-12s %-12s %-12s%n",
                "LearningRate", "Momentum", "HiddenLayers", "Epochs",
                "F-spam", "F-ham", "WAvg-F", "Time(s)"));
        writer.write(repeat("-", 100) + "\n");

        int total = learningRates.length * momentums.length * hiddenLayers.length * epochs.length;
        int current = 0;

        System.out.println("Neural network training has started. This can take a long time...");

        // Training and evaluation loops
        for (double lr : learningRates) {
            for (double mom : momentums) {
                for (String hidden : hiddenLayers) {
                    for (int ep : epochs) {
                        current++;
                        System.out.printf("[%d/%d] LR=%.3f Mom=%.1f Hidden=%s Epochs=%d... ",
                                current, total, lr, mom, hidden, ep);

                        // MLP configuration
                        MultilayerPerceptron mlp = Evaluate.buildMLP(lr, mom, hidden, ep);

                        // Measure the computational cost (train)
                        long startTime = System.currentTimeMillis();
                        mlp.buildClassifier(train);
                        long endTime = System.currentTimeMillis();
                        double timeSeconds = (endTime - startTime) / 1000.0;

                        // Evaluation (dev)
                        Evaluation eval = new Evaluation(train);
                        eval.evaluateModel(mlp, dev);

                        double fSpam = eval.fMeasure(spamIndex);
                        double fHam = eval.fMeasure(1 - spamIndex);
                        double fWAvg = eval.weightedFMeasure();

                        System.out.printf("F-Spam: %.4f | Time: %.2fs%n", fSpam, timeSeconds);

                        writer.write(String.format("%-15.3f %-12.1f %-14s %-8d %-12.4f %-12.4f %-12.4f %-12.2f%n",
                                lr, mom, hidden, ep, fSpam, fHam, fWAvg, timeSeconds));

                        // Update the best model
                        if (fSpam > bestFMeasure) {
                            bestFMeasure = fSpam;
                            bestModel = mlp;
                            bestTime = timeSeconds;
                            bestParams = String.format(
                                    "LearningRate=%.3f | Momentum=%.1f | HiddenLayers=%s | Epochs=%d",
                                    lr, mom, hidden, ep);
                        }
                    }
                }
            }
        }

        // ================================================
        // 5. SUMMARY AND SAVE THE MODEL
        // ================================================
        String summary = "\n" + repeat("=", 100) + "\n" +
                "BEST MODEL:\n" +
                "  Parameters  : " + bestParams + "\n" +
                "  F-Measure (spam): " + String.format("%.4f", bestFMeasure) + "\n" +
                "  Training time: " + String.format("%.2f", bestTime) + " sec\n";

        writer.write(summary);
        writer.close();

        System.out.println(summary);

        // Save to disk (.model)
        // Fix C5: a path without a folder (e.g. "mlp.model") has no parent
        File modelParent = new File(modelPath).getParentFile();
        if (modelParent != null) {
            modelParent.mkdirs();
        }
        SerializationHelper.write(modelPath, bestModel);

        System.out.println("Model saved here           : " + modelPath);
        System.out.println("Results report here        : " + resultsPath);

        // Automatic logging (uncomment it when you have ExperimentLogger)
        String resultLog = String.format("F-Spam: %.4f | Time: %.2fs", bestFMeasure, bestTime);
        ExperimentLogger.log("3. MLP Fine-Tuning (GetModel)", "Best Parameters: " + bestParams, resultLog);
    }

    /**
     * Repeats a string n times.
     *
     * @param s String to repeat
     * @param n Number of repetitions
     * @return The new string
     */
    private static String repeat(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++)
            sb.append(s);
        return sb.toString();
    }
}
