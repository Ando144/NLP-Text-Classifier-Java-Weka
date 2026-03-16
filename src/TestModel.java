import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * Klase honek entrenatutako eredu bat (.model) kargatzen du eta 
 * proba-multzo itsu baten gainean (test set) ebaluatzen du.
 * * HELBURUAK:
 * - Ereduaren kalitatea benetako datu ezezagunekin neurtzea (Generalizazio gaitasuna).
 * - Nahaste-matrizea (Confusion Matrix) eta metrika nagusiak (F-Measure, Precision, Recall) lortzea.
 * - Emaitza guztiak fitxategi batean (.txt) eta erregistro nagusian (Logger) gordetzea.
 * * AURREBALDINTZAK:
 * - 'test_final.arff' fitxategia existitzea (bektorizatuta eta iragazita).
 * - Eredu entrenatu bat diskoan gordeta egotea (adibidez, 'mlp.model').
 * * ONDORENGO BALDINTZAK:
 * - Ebaluazioaren emaitzak kontsolan inprimatuko dira eta adierazitako .txt fitxategian gordeko dira.
 * * EXEKUZIO ADIBIDEA:
 * java -cp "lib/weka.jar:bin" TestModel Partiketak/test_final.arff modelo/mlp.model emaitzak/test_emaitzak.txt
 * * @author WekaProyecto2026 Taldea
 */
public class TestModel {

    public static void main(String[] args) {
        try {
            // ================================================
            // 1. PARAMETROEN KUDEAKETA DINAMIKOA
            // ================================================
            if (args.length > 3) {
                System.err.println("Erabilera: java -cp \"lib/weka.jar:bin\" TestModel <test_final.arff> <eredua.model> <emaitzak.txt>");
                return;
            }

            // Balio lehenetsiak
            String testPath    = args.length >= 1 ? args[0] : "Partiketak/test_final.arff";
            String modelPath   = args.length >= 2 ? args[1] : "modelo/mlp.model";
            String resultsPath = args.length == 3 ? args[2] : "emaitzak/test_emaitzak.txt";

            System.out.println("==================================================");
            System.out.println("EBALUAZIO ITSUAREN KONFIGURAZIOA (Blind Test):");
            System.out.println("Test fitxategia : " + testPath);
            System.out.println("Eredua (.model) : " + modelPath);
            System.out.println("Emaitzak (.txt) : " + resultsPath);
            System.out.println("==================================================\n");

            if (!new File(testPath).exists() || !new File(modelPath).exists()) {
                System.err.println("ERROREA: Fitxategiren bat ez da existitzen. Bideak ondo begiratu.");
                return;
            }

            // ================================================
            // 2. DATUAK ETA EREDUA KARGATU
            // ================================================
            System.out.println("Test datuak kargatzen...");
            Instances test = new DataSource(testPath).getDataSet();
            if (test.classIndex() == -1) {
                test.setClassIndex(test.numAttributes() - 1);
            }

            System.out.println("Sare neuronala (eredua) kargatzen...");
            Classifier eredua = (Classifier) SerializationHelper.read(modelPath);

            // ================================================
            // 3. EBALUAZIOA BURUTU
            // ================================================
            System.out.println("Eredua ebaluatzen datu ezezagunekin...\n");
            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(eredua, test);

            // ================================================
            // 4. EMAITZAK INPRIMITU ETA GORDE
            // ================================================
            int spamIndex = test.classAttribute().indexOfValue("SPAM");
            if (spamIndex == -1) spamIndex = test.classAttribute().indexOfValue("spam");

            // String bat sortuko dugu dena gordetzeko (Kontsola, txt eta Logger-arentzat)
            StringBuilder emaitzaOsoa = new StringBuilder();
            emaitzaOsoa.append("==================================================\n");
            emaitzaOsoa.append("               EBALUAZIOAREN EMAITZAK             \n");
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

            // 4.1. Kontsolan inprimatu
            System.out.println(emaitzaOsoa.toString());

            // 4.2. Fitxategi dedikatuan gorde (adib: emaitzak/test_emaitzak.txt)
            File resultsFile = new File(resultsPath);
            if (resultsFile.getParentFile() != null) {
                resultsFile.getParentFile().mkdirs();
            }
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFile))) {
                writer.write(emaitzaOsoa.toString());
            }

            // 4.3. Erregistro Automatikoa (Logger)
            ExperimentLogger.log("4. Ebaluazio Itsua (TestModel)", "Eredua: " + modelPath, "\n" + emaitzaOsoa.toString());

            System.out.println("\nARRAKASTA! Emaitzen txosten zehatza hemen gorde da: " + resultsPath);

        } catch (Exception e) {
            System.err.println("Errore kritikoa ebaluazioa burutzean.");
            e.printStackTrace();
        }
    }
}