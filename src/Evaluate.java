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
 * Klase honek sailkatzaile baten kalitatea ebaluatzen du,
 * bi metodologia erabiliz: Cross Validation eta Hold-Out.
 *
 * 
 * Ebaluazioa egiteko, train eta dev dataset-ak bateratzen dira,
 * eta ondoren bi estrategia aplikatzen dira:
 * 
 *
 * <ul>
 * <li><b>5-fold Cross Validation:</b>
 * <ul>
 * <li>Datuak 5 zatitan banatzen dira</li>
 * <li>Iterazio bakoitzean 4 train + 1 test erabiltzen da</li>
 * <li>Errorearen estimazio sendoa ematen du</li>
 * </ul>
 * </li>
 * <li><b>Repeated Stratified Hold-Out (70/30):</b>
 * <ul>
 * <li>Datuak %70 train eta %30 test moduan banatzen dira</li>
 * <li>Prozesua hainbat aldiz errepikatzen da</li>
 * <li>Batezbestekoa eta desbideratze estandarra kalkulatzen dira</li>
 * </ul>
 * </li>
 * </ul>
 *
 * 
 * Ebaluazioan kalkulatutako metrikak:
 * 
 * <ul>
 * <li>Accuracy</li>
 * <li>Precision</li>
 * <li>Recall</li>
 * <li>F-Measure (spam eta legitimoa)</li>
 * <li>Weighted average</li>
 * </ul>
 *
 * 
 * Erabilitako eredua sare neuronal bat da (MultilayerPerceptron),
 * parametro konfigurableekin.
 * 
 *
 * 
 * Emaitzak fitxategi batean gordetzen dira eta esperimentuen
 * erregistroan ere jasotzen dira.
 * 
 *
 * @version 1.0
 */
public class Evaluate {
    /**
     * Programa exekutatzen duen metodo nagusia, sailkatzailearen
     * ebaluazio osoa egiten duena.
     *
     * <p>
     * Train eta dev dataset-ak bateratzen dira eta ondoren
     * bi ebaluazio-metodo aplikatzen dira: Cross Validation eta
     * Repeated Hold-Out.
     * </p>
     *
     * @param args Komando lerroko argumentuak:
     *             <ul>
     *             <li>args[0] - train_final.arff fitxategia</li>
     *             <li>args[1] - dev_final.arff fitxategia</li>
     *             <li>args[2] - emaitzak gordetzeko fitxategia</li>
     *             <li>args[3] - learning rate</li>
     *             <li>args[4] - momentum</li>
     *             <li>args[5] - hidden layers konfigurazioa</li>
     *             <li>args[6] - epoch kopurua</li>
     *             </ul>
     *
     * @throws Exception Exekuzioan errore bat gertatzen bada
     */
    public static void main(String[] args) throws Exception {
        if (args.length < 7) {
            System.out.println(
                    "Erabilera: java -cp \"lib\\weka.jar;bin\" Evaluate <train_final.arff> <dev_final.arff> <emaitzak.txt> <learning_rate> <momentum> <hidden_layers> <epochs>");
            return;
        }

        String trainPath = args[0];
        String devPath = args[1];
        String resultsPath = args[2];

        double learningRate = Double.parseDouble(args[3]);
        double momentum = Double.parseDouble(args[4]);
        String hiddenLayers = args[5];
        int epochs = Integer.parseInt(args[6]);

        // Train eta dev dataset-ak bateratzen dira ebaluazio sendoagoa egiteko
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
        System.out.println("Parametro optimoak: LR=" + learningRate + " Mom=" + momentum
                + " Hidden=" + hiddenLayers + " Epochs=" + epochs);

        int spamIndex = allData.classAttribute().indexOfValue("spam");
        if (spamIndex == -1)
            spamIndex = allData.classAttribute().indexOfValue("SPAM");

        new File(resultsPath).getParentFile().mkdirs();
        BufferedWriter writer = new BufferedWriter(new FileWriter(resultsPath));

        writer.write("==================================================\n");
        writer.write("      AURREIKUSITAKO KALITATE TXOSTENA            \n");
        writer.write("==================================================\n");
        writer.write("Parametro optimoak:\n");
        writer.write("  LearningRate : " + learningRate + "\n");
        writer.write("  Momentum     : " + momentum + "\n");
        writer.write("  HiddenLayers : " + hiddenLayers + "\n");
        writer.write("  Epochs       : " + epochs + "\n");
        writer.write("  Train+Dev    : " + allData.numInstances() + " instanzia\n\n");

        // 5-fold cross validation aplikatzen da, errorearen estimazio fidagarria
        // lortzeko
        System.out.println("\n[1/2] 5-fold Cross Validation...");

        /**
         * Multilayer Perceptron sare neuronala sortu eta konfiguratzen du.
         *
         * @param learningRate Ikasketa tasa
         * @param momentum     Momentum balioa
         * @param hiddenLayers Geruza ezkutuen konfigurazioa
         * @param epochs       Entrenamendu iterazio kopurua
         * @return Konfiguratutako MultilayerPerceptron eredua
         * @throws Exception Konfigurazioan errorea badago
         */
        MultilayerPerceptron mlpCV = buildMLP(learningRate, momentum, hiddenLayers, epochs);
        Evaluation evalCV = new Evaluation(allData);
        evalCV.crossValidateModel(mlpCV, allData, 5, new Random(1));

        /**
         * Ebaluazioaren emaitzak testu formatuan bihurtzen ditu.
         *
         * @param eval      Evaluation objektua (Weka)
         * @param spamIndex "spam" klasearen indizea
         * @return Formateatutako emaitzen testua
         * @throws Exception Errorea gertatzen bada
         */
        writer.write("==================================================\n");
        writer.write("1. 5-FOLD CROSS VALIDATION\n");
        writer.write("==================================================\n");
        writer.write(formatResults(evalCV, spamIndex));
        writer.write(evalCV.toMatrixString("Nahaste-matrizea:") + "\n\n");

        System.out.println("  5-fCV F-spam: " + String.format("%.4f", evalCV.fMeasure(spamIndex)));

        // Stratified hold-out erabiliz, train/test banaketa errepikatzen da
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
            // Weka-ko Resample filtroa erabiltzen da banaketa estratifikatua egiteko
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

            // Precision, Recall eta F-Measure kalkulatzen dira klase bakoitzerako
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

        /**
         * Balio multzo baten batezbestekoa kalkulatzen du.
         *
         * @param arr Balioen array-a
         * @return Batezbestekoa
         */
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

        // Experiment Tracking (Aukerakoa)
        String parametrosLog = String.format(
                "Datu-multzoa: Train+Dev bateratua (%d instantzia)\n" +
                        "Sare Neuronalaren Ezarpenak: LR=%.3f | Mom=%.1f | Hidden=%s | Epochs=%d\n" +
                        "Ebaluazio eskemak: 5-fold CV & 5 Repeated Stratified Hold-Out (70/30)",
                allData.numInstances(), learningRate, momentum, hiddenLayers, epochs);

        String resultadosLog = String.format(
                "[1] 5-FOLD CROSS VALIDATION:\n" +
                        "    Accuracy: %.2f%%\n" +
                        "    F-Spam  : %.4f\n" +
                        "    WAvg-F  : %.4f\n\n" +
                        "[2] 5 REPEATED STRATIFIED HOLD-OUT (70/30):\n" +
                        "    Spam Precision : %.4f +- %.4f\n" +
                        "    Spam Recall    : %.4f +- %.4f\n" +
                        "    Spam F-Measure : %.4f +- %.4f\n" +
                        "    WAvg F-Measure : %.4f +- %.4f",
                evalCV.pctCorrect(), evalCV.fMeasure(spamIndex), evalCV.weightedFMeasure(),
                mean(prSpam), stddev(prSpam),
                mean(reSpam), stddev(reSpam),
                mean(fSpam), stddev(fSpam),
                mean(fWAvg), stddev(fWAvg));

        ExperimentLogger.log("5. Kalitate Estimatua (Evaluate)", parametrosLog, resultadosLog);
    }

    private static MultilayerPerceptron buildMLP(double learningRate, double momentum, String hiddenLayers, int epochs)
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

    /**
     * Balio multzo baten desbideratze estandarra kalkulatzen du.
     *
     * @param arr Balioen array-a
     * @return Desbideratze estandarra
     */
    private static double stddev(double[] arr) {
        double m = mean(arr);
        double sum = 0;
        for (double v : arr)
            sum += Math.pow(v - m, 2);
        return Math.sqrt(sum / arr.length);
    }
}