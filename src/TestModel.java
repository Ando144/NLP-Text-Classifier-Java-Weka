import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Entrenatutako eredu baten azken ebaluazioa egiten duen klasea,
 * test datu-multzo independente baten gainean (blind test).
 *
 * 
 * Klase honek aurrez entrenatutako sailkatzaile bat kargatzen du
 * eta inoiz ikusi ez dituen datuen gainean ebaluatzen du,
 * ereduaren benetako generalizazio gaitasuna neurtzeko.
 * 
 *
 * 
 * Ebaluazio honek ez du inolako parametro doikuntzarik egiten,
 * eta horregatik da fidagarriena.
 * 
 *
 * @version 1.0
 */
public class TestModel {
    /**
     * Programaren sarrera-puntua.
     *
     * 
     * Metodo honek:
     * <ul>
     * <li>Test datuak kargatzen ditu</li>
     * <li>Aurrez entrenatutako eredua kargatzen du</li>
     * <li>Eredua ebaluatzen du datu ezezagunetan</li>
     * <li>Metrika nagusiak kalkulatzen ditu (Accuracy, F-Measure, etab.)</li>
     * <li>Emaitzak pantailan erakutsi eta fitxategian gordetzen ditu</li>
     * </ul>
     * 
     *
     * @param args Argumentuak:
     *             {@code <test_final.arff> <eredua.model> <emaitzak.txt> }
     */
    public static void main(String[] args) {
        try {
            if (args.length > 3) {
                System.err.println(
                        "Erabilera: java -cp \"lib/weka.jar:bin\" TestModel <test_final.arff> <eredua.model> <emaitzak.txt>");
                return;
            }

            // Balio lehenetsiak
            String testPath = args.length >= 1 ? args[0] : "Partiketak/test_final.arff";
            String modelPath = args.length >= 2 ? args[1] : "modelo/mlp.model";
            String resultsPath = args.length == 3 ? args[2] : "emaitzak/test_emaitzak.txt";

            System.out.println("==================================================");
            System.out.println("EBALUAZIO ITSUAREN KONFIGURAZIOA (Blind Test):");
            System.out.println("Test fitxategia : " + testPath);
            System.out.println("Eredua (.model) : " + modelPath);
            System.out.println("Emaitzak (.txt) : " + resultsPath);
            System.out.println("==================================================\n");

            if (!new File(testPath).exists() || !new File(modelPath).exists()) {
                System.err.println("ERROREA: Fitxategiren bat ez da existitzen.");
                return;
            }

            // Datuak eta eredua kargatu
            Instances test = new DataSource(testPath).getDataSet();
            EmailLoader.ensureClassIndex(test);

            Classifier eredua = (Classifier) SerializationHelper.read(modelPath);

            // Ebaluaketa
            System.out.println("Eredua ebaluatzen datu ezezagunekin...\n");
            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(eredua, test);

            // Emaitzak prestatu
            int spamIndex = test.classAttribute().indexOfValue("SPAM");
            if (spamIndex == -1)
                spamIndex = test.classAttribute().indexOfValue("spam");

            StringBuilder emaitzaOsoa = new StringBuilder();
            emaitzaOsoa.append("==================================================\n");
            emaitzaOsoa.append("               EBALUAZIOAREN EMAITZAK                  \n");
            emaitzaOsoa.append("==================================================\n\n");

            emaitzaOsoa.append("--- METRIKA OROKORRAK ---\n");
            emaitzaOsoa.append(String.format("Zuzenen Ehunekoa (Accuracy) : %.2f%%\n", eval.pctCorrect()));
            emaitzaOsoa.append(String.format("F-Measure Haztatua (WAvg)   : %.4f\n\n", eval.weightedFMeasure()));

            if (spamIndex != -1) {
                emaitzaOsoa.append("--- SPAM KLASEAREN METRIKAK ---\n");
                emaitzaOsoa.append(String.format("Zehaztasuna (Precision) : %.4f\n", eval.precision(spamIndex)));
                emaitzaOsoa.append(String.format("Estaldura (Recall)      : %.4f\n", eval.recall(spamIndex)));
                emaitzaOsoa.append(String.format("F-Measure               : %.4f\n\n", eval.fMeasure(spamIndex)));
            }

            emaitzaOsoa.append(eval.toMatrixString("--- NAHASTE MATRIZEA (Confusion Matrix) ---"));
            emaitzaOsoa.append("\n==================================================");

            System.out.println(emaitzaOsoa.toString());

            // Fitxategia gorde
            File resultsFile = new File(resultsPath);
            if (resultsFile.getParentFile() != null) {
                resultsFile.getParentFile().mkdirs();
            }
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFile))) {
                writer.write(emaitzaOsoa.toString());
            }

            // Iragarpenen erregistroa gorde
            String predFilePath = resultsPath.replace(".txt", "_iragarpenak.txt");
            try (BufferedWriter predWriter = new BufferedWriter(new FileWriter(predFilePath))) {
                for (int i = 0; i < test.numInstances(); i++) {
                    double realClassIdx = test.instance(i).classValue();
                    String realClass = test.classAttribute().value((int) realClassIdx);
                    double predClassIdx = eredua.classifyInstance(test.instance(i));
                    String predClass = test.classAttribute().value((int) predClassIdx);
                    String linea = String.format("%3d. instantzia:\tKlase erreala: %-8s Iragarritako klasea: %-8s\n",
                        (i + 1), realClass, predClass);
                    predWriter.write(linea);
                }
            }

            // Experiment Tracking (Aukerakoa)
            ExperimentLogger.log("4. Ebaluazio Itsua (TestModel)", "Eredua: " + modelPath,
                    "\n" + emaitzaOsoa.toString());

            System.out.println("\nARRAKASTA! Emaitzen txosten zehatza hemen gorde da: " + resultsPath);

        } catch (Exception e) {
            System.err.println("Errore kritikoa ebaluazioa burutzean.");
            e.printStackTrace();
        }
    }
}