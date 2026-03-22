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
 * Evaluate
 *
 * Kalitate estimatua: sortutako sailkatzailerako aurreikusitako
 * kalitate-txostena.
 * Bi ebaluazio eskema erabiltzen dira train+dev instantziekin:
 * 1. 5-fold Cross Validation
 * 2. 5 Repeated Stratified Hold-Out (70/30)
 *
 * Uso:
 * java -cp "lib\weka.jar;bin" Evaluate <train_final.arff> <dev_final.arff>
 * <emaitzak.txt>
 *
 * Ejemplo:
 * java -cp "lib\weka.jar;bin" Evaluate Partiketak/train_final.arff
 * Partiketak/dev_final.arff emaitzak/kalitatea.txt
 */
public class Evaluate {

    // Parametro optimoak GetModel-etik lortutakoak
    private static final double LEARNING_RATE = 0.005;
    private static final double MOMENTUM = 0.2;
    private static final String HIDDEN_LAYERS = "3";
    private static final int EPOCHS = 300;

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println(
                    "Uso: java -cp \"lib\\weka.jar;bin\" Evaluate <train_final.arff> <dev_final.arff> <emaitzak.txt>");
            return;
        }

        String trainPath = args[0];
        String devPath = args[1];
        String resultsPath = args[2];

        // ================================
        // 1. Train eta dev kargatu eta elkartu
        // ================================
        Instances train = new DataSource(trainPath).getDataSet();
        Instances dev = new DataSource(devPath).getDataSet();

        train.setClassIndex(train.numAttributes() - 1);
        dev.setClassIndex(dev.numAttributes() - 1);

        Instances allData = new Instances(train);
        for (int i = 0; i < dev.numInstances(); i++) {
            allData.add(dev.instance(i));
        }
        allData.setClassIndex(allData.numAttributes() - 1);

        System.out.println("Train+Dev: " + allData.numInstances() + " instanzia");
        System.out.println("Parametro optimoak: LR=" + LEARNING_RATE + " Mom=" + MOMENTUM
                + " Hidden=" + HIDDEN_LAYERS + " Epochs=" + EPOCHS);

        int spamIndex = allData.classAttribute().indexOfValue("spam");

        new File(resultsPath).getParentFile().mkdirs();
        BufferedWriter writer = new BufferedWriter(new FileWriter(resultsPath));

        writer.write("==================================================\n");
        writer.write("      AURREIKUSITAKO KALITATE TXOSTENA            \n");
        writer.write("==================================================\n");
        writer.write("Parametro optimoak:\n");
        writer.write("  LearningRate : " + LEARNING_RATE + "\n");
        writer.write("  Momentum     : " + MOMENTUM + "\n");
        writer.write("  HiddenLayers : " + HIDDEN_LAYERS + "\n");
        writer.write("  Epochs       : " + EPOCHS + "\n");
        writer.write("  Train+Dev    : " + allData.numInstances() + " instanzia\n\n");

        // ================================
        // 2. 5-fold Cross Validation
        // ================================
        System.out.println("\n[1/2] 5-fold Cross Validation...");

        MultilayerPerceptron mlpCV = buildMLP();
        Evaluation evalCV = new Evaluation(allData);
        evalCV.crossValidateModel(mlpCV, allData, 5, new Random(1));

        writer.write("==================================================\n");
        writer.write("1. 5-FOLD CROSS VALIDATION\n");
        writer.write("==================================================\n");
        writer.write(formatResults(evalCV, spamIndex));
        writer.write(evalCV.toMatrixString("Nahaste-matrizea:") + "\n\n");

        System.out.println("  5-fCV F-spam: " + String.format("%.4f", evalCV.fMeasure(spamIndex)));

        // ================================
        // 3. 5 Repeated Stratified Hold-Out (70/30)
        // ================================
        System.out.println("[2/2] Repeated Stratified Hold-Out (5 errepikapen)...");

        int repeticiones = 5;
        double[] fSpam = new double[repeticiones];
        double[] fHam = new double[repeticiones];
        double[] fWAvg = new double[repeticiones];
        double[] prSpam = new double[repeticiones];
        double[] prHam = new double[repeticiones];
        double[] reSpam = new double[repeticiones];
        double[] reHam = new double[repeticiones];

        writer.write("==================================================\n");
        writer.write("2. 5 REPEATED STRATIFIED HOLD-OUT (70/30)\n");
        writer.write("==================================================\n");

        for (int k = 0; k < repeticiones; k++) {
            System.out.println("  Iterazioa " + (k + 1) + "/" + repeticiones);

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

            // Entrenar y evaluar
            MultilayerPerceptron mlpIter = buildMLP();
            mlpIter.buildClassifier(iterTrain);

            Evaluation evalIter = new Evaluation(iterTrain);
            evalIter.evaluateModel(mlpIter, iterDev);

            fSpam[k] = evalIter.fMeasure(spamIndex);
            fHam[k] = evalIter.fMeasure(1 - spamIndex);
            fWAvg[k] = evalIter.weightedFMeasure();
            prSpam[k] = evalIter.precision(spamIndex);
            prHam[k] = evalIter.precision(1 - spamIndex);
            reSpam[k] = evalIter.recall(spamIndex);
            reHam[k] = evalIter.recall(1 - spamIndex);

            writer.write((k + 1) + ". iterazioa: F-spam=" + String.format("%.4f", fSpam[k])
                    + " F-ham=" + String.format("%.4f", fHam[k])
                    + " WAvg-F=" + String.format("%.4f", fWAvg[k]) + "\n");
        }

        writer.write("\n--- BATEZBESTEKOA +- DESBIDERATZE ESTANDARRA ---\n");
        writer.write(String.format("Spam  Precision : %.4f +- %.4f%n", mean(prSpam), stddev(prSpam)));
        writer.write(String.format("Spam  Recall    : %.4f +- %.4f%n", mean(reSpam), stddev(reSpam)));
        writer.write(String.format("Spam  F-Measure : %.4f +- %.4f%n", mean(fSpam), stddev(fSpam)));
        writer.write(String.format("Ham   Precision : %.4f +- %.4f%n", mean(prHam), stddev(prHam)));
        writer.write(String.format("Ham   Recall    : %.4f +- %.4f%n", mean(reHam), stddev(reHam)));
        writer.write(String.format("Ham   F-Measure : %.4f +- %.4f%n", mean(fHam), stddev(fHam)));
        writer.write(String.format("WAvg  F-Measure : %.4f +- %.4f%n", mean(fWAvg), stddev(fWAvg)));
        writer.write("\n==================================================\n");
        writer.close();

        System.out.println("\nEmaitzak gordeta: " + resultsPath);
        System.out.println("  Repeated HO F-spam: "
                + String.format("%.4f", mean(fSpam)) + " +- " + String.format("%.4f", stddev(fSpam)));

        ExperimentLogger.log("5. Kalitate Estimatua (Evaluate)",
                "LR=" + LEARNING_RATE + " Mom=" + MOMENTUM + " Hidden=" + HIDDEN_LAYERS + " Epochs=" + EPOCHS,
                "5-fCV F-spam: " + String.format("%.4f", evalCV.fMeasure(spamIndex))
                        + "\nRepeated HO F-spam: " + String.format("%.4f", mean(fSpam))
                        + " +- " + String.format("%.4f", stddev(fSpam)));
    }

    private static MultilayerPerceptron buildMLP() throws Exception {
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setLearningRate(LEARNING_RATE);
        mlp.setMomentum(MOMENTUM);
        mlp.setHiddenLayers(HIDDEN_LAYERS);
        mlp.setTrainingTime(EPOCHS);
        mlp.setNominalToBinaryFilter(true);
        mlp.setNormalizeAttributes(true);
        mlp.setGUI(false);
        mlp.setDebug(false);
        return mlp;
    }

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

    private static double mean(double[] arr) {
        double sum = 0;
        for (double v : arr)
            sum += v;
        return sum / arr.length;
    }

    private static double stddev(double[] arr) {
        double m = mean(arr);
        double sum = 0;
        for (double v : arr)
            sum += Math.pow(v - m, 2);
        return Math.sqrt(sum / arr.length);
    }
}