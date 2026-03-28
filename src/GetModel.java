import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Multilayer Perceptron (MLP) algoritmoaren parametroen optimizazioa
 * (fine-tuning) egiten duen klasea.
 *
 * 
 * Klase honek entrenamendu (train) eta balidazio (dev) datu-multzoak
 * erabiltzen ditu parametro desberdinen konbinazioak ebaluatzeko
 * (grid search bidez), eta errendimendu onena duen eredua aukeratzen du.
 *
 *
 * 
 * Optimizazio irizpide nagusia spam klasearen F-Measure da.
 * 
 *
 * 
 * Azkenik, eredurik onena diskoan gordetzen da eta emaitzen txosten
 * zehatza sortzen da.
 * 
 *
 * @version 1.0
 */
public class GetModel {
    /**
     * Programaren sarrera-puntua.
     *
     * <p>
     * Metodo honek:
     * <ul>
     * <li>Datuak kargatzen ditu (train eta dev)</li>
     * <li>Parametroen grid search bat exekutatzen du</li>
     * <li>Eredu bakoitza entrenatu eta ebaluatzen du</li>
     * <li>F-Measure (spam) maximizatzen duen eredua hautatzen du</li>
     * <li>Eredu onena fitxategi batean gordetzen du</li>
     * </ul>
     * </p>
     *
     * @param args Argumentuak:
     *             {@code <train_final.arff> <dev_final.arff> <modelo.model> <resultados.txt>}
     * @throws Exception Prozesuan errore bat gertatzen bada
     */
    public static void main(String[] args) throws Exception {
        if (args.length > 4) {
            System.err.println(
                    "Erabilera: java -cp \"lib/weka.jar:bin\" GetModel <train_final.arff> <dev_final.arff> <modelo.model> <resultados.txt>");
            return;
        }

        String trainPath = args.length >= 1 ? args[0] : "Partiketak/train_final.arff";
        String devPath = args.length >= 2 ? args[1] : "Partiketak/dev_final.arff";
        String modelPath = args.length >= 3 ? args[2] : "modelo/mlp.model";
        String resultsPath = args.length == 4 ? args[3] : "emaitzak/finetuning.txt";

        System.out.println("==================================================");
        System.out.println("MLP FINE-TUNING KONFIGURAZIOA:");
        System.out.println("Train fitxategia : " + trainPath);
        System.out.println("Dev fitxategia   : " + devPath);
        System.out.println("Eredua (.model)  : " + modelPath);
        System.out.println("Emaitzak (.txt)  : " + resultsPath);
        System.out.println("==================================================\n");

        // ================================================
        // 2. DATUAK KARGATU
        // ================================================
        Instances train = new DataSource(trainPath).getDataSet();
        Instances dev = new DataSource(devPath).getDataSet();

        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);
        if (dev.classIndex() == -1)
            dev.setClassIndex(dev.numAttributes() - 1);

        System.out.println("Train: " + train.numInstances() + " instantzia kargatuta.");
        System.out.println("Dev  : " + dev.numInstances() + " instantzia kargatuta.\n");

        int spamIndex = train.classAttribute().indexOfValue("SPAM");
        if (spamIndex == -1)
            spamIndex = train.classAttribute().indexOfValue("spam");

        // Bilaketa espazioa (Grid Search)
        double[] learningRates = { 0.005 };
        double[] momentums = { 0.2 };
        String[] hiddenLayers = { "3" };
        int[] epochs = { 300 };

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
        writer.write("Optimizazio metrika: F-Measure (spam)\n");
        writer.write("Train: " + trainPath + "\n");
        writer.write("Dev  : " + devPath + "\n\n");
        writer.write(String.format("%-15s %-12s %-14s %-8s %-12s %-12s %-12s %-12s%n",
                "LearningRate", "Momentum", "HiddenLayers", "Epochs",
                "F-spam", "F-ham", "WAvg-F", "Denbora(s)"));
        writer.write(repeat("-", 100) + "\n");

        int total = learningRates.length * momentums.length * hiddenLayers.length * epochs.length;
        int current = 0;

        System.out.println("Sare Neuronalaren entrenamendua hasi da. Hau denbora luzea har dezake...");

        // Entrenamendu eta ebaluazio begiztak
        for (double lr : learningRates) {
            for (double mom : momentums) {
                for (String hidden : hiddenLayers) {
                    for (int ep : epochs) {
                        current++;
                        System.out.printf("[%d/%d] LR=%.3f Mom=%.1f Hidden=%s Epochs=%d... ",
                                current, total, lr, mom, hidden, ep);

                        // MLP Konfigurazioa
                        MultilayerPerceptron mlp = new MultilayerPerceptron();
                        mlp.setLearningRate(lr);
                        mlp.setMomentum(mom);
                        mlp.setHiddenLayers(hidden);
                        mlp.setTrainingTime(ep);
                        mlp.setNominalToBinaryFilter(true);
                        mlp.setNormalizeAttributes(true);

                        // Koste konputazionala neurtu (Train)
                        long startTime = System.currentTimeMillis();
                        mlp.buildClassifier(train);
                        long endTime = System.currentTimeMillis();
                        double timeSeconds = (endTime - startTime) / 1000.0;

                        // Ebaluazioa (Dev)
                        Evaluation eval = new Evaluation(train);
                        eval.evaluateModel(mlp, dev);

                        double fSpam = eval.fMeasure(spamIndex);
                        double fHam = eval.fMeasure(1 - spamIndex);
                        double fWAvg = eval.weightedFMeasure();

                        System.out.printf("F-Spam: %.4f | Denbora: %.2fs%n", fSpam, timeSeconds);

                        writer.write(String.format("%-15.3f %-12.1f %-14s %-8d %-12.4f %-12.4f %-12.4f %-12.2f%n",
                                lr, mom, hidden, ep, fSpam, fHam, fWAvg, timeSeconds));

                        // Eredu onena eguneratu
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
        // 5. LABURPENA ETA EREDUA GORDE
        // ================================================
        String laburpena = "\n" + repeat("=", 100) + "\n" +
                "EREDU ONENA:\n" +
                "  Parametroak : " + bestParams + "\n" +
                "  F-Measure (spam): " + String.format("%.4f", bestFMeasure) + "\n" +
                "  Entrenamendu denbora: " + String.format("%.2f", bestTime) + " seg\n";

        writer.write(laburpena);
        writer.close();

        System.out.println(laburpena);

        // Diskoan gorde (.model)
        new File(modelPath).getParentFile().mkdirs();
        SerializationHelper.write(modelPath, bestModel);

        System.out.println("Eredua ondo gorde da hemen : " + modelPath);
        System.out.println("Emaitzen txostena hemen    : " + resultsPath);

        // Erregistro Automatikoa (Deskomentatu ExperimentLogger duzuenean)
        String emaitzaLog = String.format("F-Spam: %.4f | Denbora: %.2fs", bestFMeasure, bestTime);
        ExperimentLogger.log("3. MLP Fine-Tuning (GetModel)", "Parametro Onenak: " + bestParams, emaitzaLog);
    }

    /**
     * Kate bat n aldiz errepikatzen du.
     *
     * @param s Errepikatu nahi den katea
     * @param n Errepikapen kopurua
     * @return Sortutako kate berria
     */
    private static String repeat(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++)
            sb.append(s);
        return sb.toString();
    }
}