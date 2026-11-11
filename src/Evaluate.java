import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;

/**
 * Evaluates the quality of a classifier
 * with two methods: Cross Validation and Hold-Out.
 *
 *
 * For the evaluation, the train and dev datasets are merged,
 * and then two strategies are applied:
 *
 *
 * <ul>
 * <li><b>5-fold Cross Validation:</b>
 * <ul>
 * <li>The data is split into 5 parts</li>
 * <li>In each iteration, 4 parts are used for training + 1 for testing</li>
 * <li>Gives a robust estimate of the error</li>
 * </ul>
 * </li>
 * <li><b>Repeated Random Hold-Out (70/30):</b>
 * <ul>
 * <li>A random 70% of the data (without replacement) is used for training
 * and the other 30% for testing; the class is not used to draw the sample</li>
 * <li>The process is repeated several times</li>
 * <li>The mean and the standard deviation are computed</li>
 * </ul>
 * </li>
 * </ul>
 *
 *
 * Metrics computed in the evaluation:
 *
 * <ul>
 * <li>Accuracy</li>
 * <li>Precision</li>
 * <li>Recall</li>
 * <li>F-Measure (spam and legitimate)</li>
 * <li>Weighted average</li>
 * </ul>
 *
 *
 * The model used is a neural network (MultilayerPerceptron),
 * with configurable parameters.
 *
 *
 *
 * The results are saved to a file and are also recorded in the
 * experiment log.
 *
 *
 * @version 1.0
 */
public class Evaluate {
    /**
     * Main method: runs the complete evaluation of the
     * classifier.
     *
     * <p>
     * The train and dev datasets are merged and then
     * two evaluation methods are applied: Cross Validation and
     * Repeated Hold-Out.
     * </p>
     *
     * @param args Command-line arguments:
     *             <ul>
     *             <li>args[0] - train_final.arff file</li>
     *             <li>args[1] - dev_final.arff file</li>
     *             <li>args[2] - file where the results are saved</li>
     *             <li>args[3] - learning rate</li>
     *             <li>args[4] - momentum</li>
     *             <li>args[5] - hidden layers configuration</li>
     *             <li>args[6] - number of epochs</li>
     *             </ul>
     *
     * @throws Exception If an error happens during the run
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 7) {
            System.out.println(
                    "Usage: java -cp \"lib\\weka.jar;bin\" Evaluate <train_final.arff> <dev_final.arff> <results.txt> <learning_rate> <momentum> <hidden_layers> <epochs>");
            return;
        }

        String trainPath = args[0];
        String devPath = args[1];
        String resultsPath = args[2];

        double learningRate = Double.parseDouble(args[3]);
        double momentum = Double.parseDouble(args[4]);
        String hiddenLayers = args[5];
        int epochs = Integer.parseInt(args[6]);

        // The train and dev datasets are merged for a more robust evaluation
        Instances train = new DataSource(trainPath).getDataSet();
        Instances dev = new DataSource(devPath).getDataSet();

        EmailLoader.ensureClassIndex(train);
        EmailLoader.ensureClassIndex(dev);

        Instances allData = new Instances(train);
        for (int i = 0; i < dev.numInstances(); i++) {
            allData.add(dev.instance(i));
        }
        EmailLoader.ensureClassIndex(allData);

        System.out.println("Train+Dev: " + allData.numInstances() + " instances");
        System.out.println("Optimal parameters: LR=" + learningRate + " Mom=" + momentum
                + " Hidden=" + hiddenLayers + " Epochs=" + epochs);

        int spamIndex = allData.classAttribute().indexOfValue("spam");
        if (spamIndex == -1)
            spamIndex = allData.classAttribute().indexOfValue("SPAM");

        // Fix C5: a path without a folder (e.g. "quality.txt") has no parent
        File resultsParent = new File(resultsPath).getParentFile();
        if (resultsParent != null) {
            resultsParent.mkdirs();
        }
        BufferedWriter writer = new BufferedWriter(new FileWriter(resultsPath));

        writer.write("==================================================\n");
        writer.write("      ESTIMATED QUALITY REPORT                    \n");
        writer.write("==================================================\n");
        writer.write("Optimal parameters:\n");
        writer.write("  LearningRate : " + learningRate + "\n");
        writer.write("  Momentum     : " + momentum + "\n");
        writer.write("  HiddenLayers : " + hiddenLayers + "\n");
        writer.write("  Epochs       : " + epochs + "\n");
        writer.write("  Train+Dev    : " + allData.numInstances() + " instances\n\n");

        // 5-fold cross validation is applied to get a reliable estimate
        // of the error
        System.out.println("\n[1/2] 5-fold Cross Validation...");

        MultilayerPerceptron mlpCV = buildMLP(learningRate, momentum, hiddenLayers, epochs);
        Evaluation evalCV = new Evaluation(allData);
        evalCV.crossValidateModel(mlpCV, allData, 5, new Random(1));

        writer.write("==================================================\n");
        writer.write("1. 5-FOLD CROSS VALIDATION\n");
        writer.write("==================================================\n");
        writer.write(formatResults(evalCV, spamIndex));
        writer.write(evalCV.toMatrixString("Confusion matrix:") + "\n\n");

        System.out.println("  5-fCV F-spam: " + String.format("%.4f", evalCV.fMeasure(spamIndex)));

        // Random hold-out: the train/test split is repeated with 5 different seeds
        System.out.println("[2/2] Repeated Random Hold-Out (5 repetitions)...");

        int repetitions = 5;
        double[] fSpam = new double[repetitions];
        double[] fHam = new double[repetitions];
        double[] fWAvg = new double[repetitions];
        double[] prSpam = new double[repetitions];
        double[] prHam = new double[repetitions];
        double[] reSpam = new double[repetitions];
        double[] reHam = new double[repetitions];

        writer.write("==================================================\n");
        writer.write("2. 5 REPEATED RANDOM HOLD-OUT (70/30)\n");
        writer.write("==================================================\n");

        for (int k = 0; k < repetitions; k++) {
            System.out.println("  Iteration " + (k + 1) + "/" + repetitions);
            // Fix C3: Weka's unsupervised Resample filter draws a random 70% sample
            // without replacement; it does not look at the class, so the split is
            // random (not stratified). The same seed with invertSelection gives the
            // complementary 30%.
            // Train 70%
            Resample rTrain = new Resample();
            rTrain.setRandomSeed(k + 1);
            rTrain.setNoReplacement(true);
            rTrain.setSampleSizePercent(70.0);
            rTrain.setInvertSelection(false);
            rTrain.setInputFormat(allData);
            Instances iterTrain = Filter.useFilter(allData, rTrain);
            iterTrain.setClassIndex(iterTrain.numAttributes() - 1);

            // Dev 30%
            Resample rDev = new Resample();
            rDev.setRandomSeed(k + 1);
            rDev.setNoReplacement(true);
            rDev.setSampleSizePercent(70.0);
            rDev.setInvertSelection(true);
            rDev.setInputFormat(allData);
            Instances iterDev = Filter.useFilter(allData, rDev);
            iterDev.setClassIndex(iterDev.numAttributes() - 1);

            MultilayerPerceptron mlpIter = buildMLP(learningRate, momentum, hiddenLayers, epochs);
            mlpIter.buildClassifier(iterTrain);

            Evaluation evalIter = new Evaluation(iterTrain);
            evalIter.evaluateModel(mlpIter, iterDev);

            // Precision, Recall and F-Measure are computed for each class
            fSpam[k] = evalIter.fMeasure(spamIndex);
            fHam[k] = evalIter.fMeasure(1 - spamIndex);
            fWAvg[k] = evalIter.weightedFMeasure();
            prSpam[k] = evalIter.precision(spamIndex);
            prHam[k] = evalIter.precision(1 - spamIndex);
            reSpam[k] = evalIter.recall(spamIndex);
            reHam[k] = evalIter.recall(1 - spamIndex);

            writer.write((k + 1) + ". iteration: F-spam=" + String.format("%.4f", fSpam[k])
                    + " F-ham=" + String.format("%.4f", fHam[k])
                    + " WAvg-F=" + String.format("%.4f", fWAvg[k]) + "\n");
        }

        writer.write("\n--- MEAN +- STANDARD DEVIATION ---\n");
        writer.write(String.format("Spam  Precision : %.4f +- %.4f%n", mean(prSpam), stddev(prSpam)));
        writer.write(String.format("Spam  Recall    : %.4f +- %.4f%n", mean(reSpam), stddev(reSpam)));
        writer.write(String.format("Spam  F-Measure : %.4f +- %.4f%n", mean(fSpam), stddev(fSpam)));
        writer.write(String.format("Ham   Precision : %.4f +- %.4f%n", mean(prHam), stddev(prHam)));
        writer.write(String.format("Ham   Recall    : %.4f +- %.4f%n", mean(reHam), stddev(reHam)));
        writer.write(String.format("Ham   F-Measure : %.4f +- %.4f%n", mean(fHam), stddev(fHam)));
        writer.write(String.format("WAvg  F-Measure : %.4f +- %.4f%n", mean(fWAvg), stddev(fWAvg)));
        writer.write("\n==================================================\n");
        writer.close();

        System.out.println("\nResults saved: " + resultsPath);
        System.out.println("  Repeated HO F-spam: "
                + String.format("%.4f", mean(fSpam)) + " +- " + String.format("%.4f", stddev(fSpam)));

        // Experiment tracking (optional)
        String paramsLog = String.format(
                "Dataset: merged Train+Dev (%d instances)\n" +
                        "Neural Network Settings: LR=%.3f | Mom=%.1f | Hidden=%s | Epochs=%d\n" +
                        "Evaluation schemes: 5-fold CV & 5 Repeated Random Hold-Out (70/30)",
                allData.numInstances(), learningRate, momentum, hiddenLayers, epochs);

        String resultsLog = String.format(
                "[1] 5-FOLD CROSS VALIDATION:\n" +
                        "    Accuracy: %.2f%%\n" +
                        "    F-Spam  : %.4f\n" +
                        "    WAvg-F  : %.4f\n\n" +
                        "[2] 5 REPEATED RANDOM HOLD-OUT (70/30):\n" +
                        "    Spam Precision : %.4f +- %.4f\n" +
                        "    Spam Recall    : %.4f +- %.4f\n" +
                        "    Spam F-Measure : %.4f +- %.4f\n" +
                        "    WAvg F-Measure : %.4f +- %.4f",
                evalCV.pctCorrect(), evalCV.fMeasure(spamIndex), evalCV.weightedFMeasure(),
                mean(prSpam), stddev(prSpam),
                mean(reSpam), stddev(reSpam),
                mean(fSpam), stddev(fSpam),
                mean(fWAvg), stddev(fWAvg));

        ExperimentLogger.log("5. Estimated Quality (Evaluate)", paramsLog, resultsLog);
    }

    /**
     * Creates and configures a Multilayer Perceptron neural network.
     *
     * <p>
     * This method sets the parameters of the MLP algorithm,
     * taking the input values and creating the MLP model with them.
     * </p>
     *
     * @param learningRate Learning rate
     * @param momentum     Momentum value
     * @param hiddenLayers Hidden layers configuration (e.g. "3")
     * @param epochs       Number of training iterations (training time)
     * @return The configured {@link MultilayerPerceptron} model
     * @throws Exception If a configuration error happens
     */
    public static MultilayerPerceptron buildMLP(double learningRate, double momentum, String hiddenLayers, int epochs)
            throws Exception {
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setLearningRate(learningRate);
        mlp.setMomentum(momentum);
        mlp.setHiddenLayers(hiddenLayers);
        mlp.setTrainingTime(epochs);
        mlp.setNominalToBinaryFilter(true);
        mlp.setNormalizeAttributes(true);
        mlp.setGUI(false);
        mlp.setDebug(false);
        return mlp;
    }

    /**
     * Converts the evaluation results to text format.
     *
     * @param eval      Evaluation object (Weka)
     * @param spamIndex Index of the "spam" class
     * @return The formatted results text
     * @throws Exception If an error happens
     */
    private static String formatResults(Evaluation eval, int spamIndex) throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Accuracy          : %.2f%%%n", eval.pctCorrect()));
        sb.append(String.format("Spam  Precision   : %.4f%n", eval.precision(spamIndex)));
        sb.append(String.format("Spam  Recall      : %.4f%n", eval.recall(spamIndex)));
        sb.append(String.format("Spam  F-Measure   : %.4f%n", eval.fMeasure(spamIndex)));
        sb.append(String.format("Ham   Precision   : %.4f%n", eval.precision(1 - spamIndex)));
        sb.append(String.format("Ham   Recall      : %.4f%n", eval.recall(1 - spamIndex)));
        sb.append(String.format("Ham   F-Measure   : %.4f%n", eval.fMeasure(1 - spamIndex)));
        sb.append(String.format("WAvg  Precision   : %.4f%n", eval.weightedPrecision()));
        sb.append(String.format("WAvg  Recall      : %.4f%n", eval.weightedRecall()));
        sb.append(String.format("WAvg  F-Measure   : %.4f%n%n", eval.weightedFMeasure()));
        return sb.toString();
    }

    /**
     * Computes the mean of a set of values.
     *
     * @param arr Array of values
     * @return The mean
     */
    private static double mean(double[] arr) {
        double sum = 0;
        for (double v : arr)
            sum += v;
        return sum / arr.length;
    }

    /**
     * Computes the (population) standard deviation of a set of values,
     * dividing by the number of values.
     *
     * @param arr Array of values
     * @return The standard deviation
     */
    private static double stddev(double[] arr) {
        double m = mean(arr);
        double sum = 0;
        for (double v : arr)
            sum += Math.pow(v - m, 2);
        return Math.sqrt(sum / arr.length);
    }
}
