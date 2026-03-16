import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Klase honek MultilayerPerceptron (MLP) algoritmoaren parametroen ekorketa (fine-tuning) egiten du.
 * Parametro konbinazio desberdinak probatzen ditu, bakoitza 'dev' multzoan ebaluatuz, eta
 * Spam klaseko F-Measure onena lortzen duen modeloa gordetzen du.
 * * HELBURUAK:
 * - Sare neuronalaren parametro optimoak bilatzea (LearningRate, Momentum, HiddenLayers, Epochs).
 * - Koste konputazionala (entrenamendu denbora) neurtzea eredu bakoitzarentzat.
 * - Eredu optimoa diskorako esportatzea (.model formatuan).
 * * AURREBALDINTZAK:
 * - Bektorizatutako 'train_final.arff' eta 'dev_final.arff' fitxategiak existitzea.
 * * ONDORENGO BALDINTZAK:
 * - Eredu optimoa gordeko da adierazitako karpetan.
 * - Emaitzen laburpen taula bat '.txt' fitxategian idatziko da.
 * * EXEKUZIO ADIBIDEA:
 * java -cp "lib/weka.jar:bin" GetModel Partiketak/train_final.arff Partiketak/dev_final.arff modelo/mlp.model emaitzak/finetuning.txt
 * * @author WekaProyecto2026 Taldea
 */
public class GetModel {

    public static void main(String[] args) throws Exception {
        // ================================================
        // 1. PARAMETROEN KUDEAKETA DINAMIKOA
        // ================================================
        if (args.length > 4) {
            System.err.println("Erabilera: java -cp \"lib/weka.jar:bin\" GetModel <train_final.arff> <dev_final.arff> <modelo.model> <resultados.txt>");
            return;
        }

        // Balio lehenetsiak zure egiturara egokituta
        String trainPath   = args.length >= 1 ? args[0] : "Partiketak/train_final.arff";
        String devPath     = args.length >= 2 ? args[1] : "Partiketak/dev_final.arff";
        String modelPath   = args.length >= 3 ? args[2] : "modelo/mlp.model";
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

        if (train.classIndex() == -1) train.setClassIndex(train.numAttributes() - 1);
        if (dev.classIndex() == -1) dev.setClassIndex(dev.numAttributes() - 1);

        System.out.println("Train: " + train.numInstances() + " instantzia kargatuta.");
        System.out.println("Dev  : " + dev.numInstances() + " instantzia kargatuta.\n");

        // Bilatu SPAM klasearen indizea (segurtasun neurria maiuskula/minuskula arazoentzat)
        int spamIndex = train.classAttribute().indexOfValue("SPAM");
        if (spamIndex == -1) spamIndex = train.classAttribute().indexOfValue("spam");

        // ================================================
        // 3. EKORKETA PRESTATU (Grid Search)
        // ================================================
        double[] learningRates = { 0.003, 0.005, 0.008 }; // Afinamos alrededor del ganador anterior
        double[] momentums = { 0.4, 0.6, 0.8 };           // Subimos el techo
        String[] hiddenLayers = { "10", "30", "a" };      // Probamos arquitecturas mucho más grandes ('a' = ~501 neuronas)
        int[] epochs = { 100, 300, 500 };                 // Le damos más tiempo de aprendizaje

        MultilayerPerceptron bestModel = null;
        double bestFMeasure = -1;
        String bestParams = "";
        double bestTime = 0.0;

        // Sortu emaitzak gordetzeko karpeta
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

        // ================================================
        // 4. ENTRENAMENDU ETA EBALUAZIO BEGIZTAK
        // ================================================
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

                        // Emaitza idatzi taulan
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

    private static String repeat(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++)
            sb.append(s);
        return sb.toString();
    }
}