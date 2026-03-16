import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;

import java.io.File;

/**
 * Klase honek hasierako datu-multzoa (emails_raw.arff) hiru azpimultzotan banatzen du:
 * Train (%80), Dev (%10) eta Test (%10), banaketa estratifikatua (Stratified) erabiliz.
 * * HELBURUAK:
 * - Datuen ausazkotzea (Randomize) hazi (seed) finko batekin, erreproduzibilitatea bermatzeko.
 * - Klaseen banaketa proportzionala mantentzea partiketa guztietan.
 * * AURREBALDINTZAK:
 * - 'emails_raw.arff' fitxategia existitu behar da (EmailLoader klaseak sortutakoa).
 * * ONDORENGO BALDINTZAK:
 * - 'Partiketak' karpetan 'train.arff', 'dev.arff' eta 'test.arff' fitxategiak sortuko dira.
 * * EXEKUZIO ADIBIDEA:
 * java -cp "lib/weka.jar:bin" DataSplit emails_raw.arff Partiketak/train.arff Partiketak/dev.arff Partiketak/test.arff
 * * @author WekaProyecto2026 Taldea
 */
public class DataSplit {

    public static void main(String[] args) {
        // ================================================
        // 1. PARAMETROEN KUDEAKETA DINAMIKOA
        // ================================================
        if (args.length > 4) {
            System.err.println("Erabilera: java -cp \"lib/weka.jar:bin\" DataSplit [input.arff] [train.arff] [dev.arff] [test.arff]");
            return;
        }

        // Zure karpeten egiturara egokitutako balio lehenetsiak (Irudian oinarrituta)
        String inputPath = args.length >= 1 ? args[0] : "emails_raw.arff";
        String trainPath = args.length >= 2 ? args[1] : "Partiketak/train.arff";
        String devPath   = args.length >= 3 ? args[2] : "Partiketak/dev.arff";
        String testPath  = args.length == 4 ? args[3] : "Partiketak/test.arff";

        System.out.println("==================================================");
        System.out.println("DATUEN PARTIKETA KONFIGURAZIOA (80 / 10 / 10):");
        System.out.println("Sarrera (Input) : " + inputPath);
        System.out.println("Train fitxategia: " + trainPath);
        System.out.println("Dev fitxategia  : " + devPath);
        System.out.println("Test fitxategia : " + testPath);
        System.out.println("==================================================\n");

        try {
            // 1. Datu-multzoa kargatu
            File inputFile = new File(inputPath);
            if (!inputFile.exists()) {
                System.err.println("ERROREA: Sarrerako fitxategia ez da existitzen -> " + inputFile.getAbsolutePath());
                return;
            }
            
            DataSource source = new DataSource(inputPath);
            Instances data = source.getDataSet();

            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Ausazkotu (Randomize) erreproduzibilitatea bermatzeko (Seed = 1)
            System.out.println("Datuak ausazkotzen (Randomize, seed=1)...");
            Randomize rand = new Randomize();
            rand.setRandomSeed(1);
            rand.setInputFormat(data);
            data = Filter.useFilter(data, rand);

            // ================================================
            // 2. PARTIKETA ESTRATIFIKATUA (80% Train, 10% Dev, 10% Test)
            // ================================================
            System.out.println("Partiketa estratifikatua aplikatzen...");

            // A. TEST sortu (%10 -> 1 fold 10etik)
            StratifiedRemoveFolds testFilter = new StratifiedRemoveFolds();
            testFilter.setNumFolds(10);
            testFilter.setFold(1);
            testFilter.setSeed(1);
            testFilter.setInputFormat(data);
            Instances test = Filter.useFilter(data, testFilter);

            // Gainerako %90a lortu (Train + Dev)
            StratifiedRemoveFolds remainderFilter = new StratifiedRemoveFolds();
            remainderFilter.setNumFolds(10);
            remainderFilter.setFold(1);
            remainderFilter.setSeed(1);
            remainderFilter.setInvertSelection(true); // Folds 2-10 mantendu
            remainderFilter.setInputFormat(data);
            Instances remainder = Filter.useFilter(data, remainderFilter);

            // B. DEV sortu (Gainerakoaren 1 fold 9tik -> Totalaren %10a)
            StratifiedRemoveFolds devFilter = new StratifiedRemoveFolds();
            devFilter.setNumFolds(9);
            devFilter.setFold(1);
            devFilter.setSeed(1);
            devFilter.setInputFormat(remainder);
            Instances dev = Filter.useFilter(remainder, devFilter);

            // C. TRAIN sortu (Gainerakoaren 8 folds 9tik -> Totalaren %80a)
            StratifiedRemoveFolds trainFilter = new StratifiedRemoveFolds();
            trainFilter.setNumFolds(9);
            trainFilter.setFold(1);
            trainFilter.setSeed(1);
            trainFilter.setInvertSelection(true); // Fold 1 kendu, beste guztiak mantendu
            trainFilter.setInputFormat(remainder);
            Instances train = Filter.useFilter(remainder, trainFilter);

            System.out.println("Banaketa amaituta:");
            System.out.println(" -> Train : " + train.numInstances() + " instantzia");
            System.out.println(" -> Dev   : " + dev.numInstances() + " instantzia");
            System.out.println(" -> Test  : " + test.numInstances() + " instantzia");

            // ================================================
            // 3. FITXATEGIAK GORDE
            // ================================================
            // 'Partiketak' karpeta sortu ez bada existitzen
            new File(trainPath).getParentFile().mkdirs();

            ArffSaver saver = new ArffSaver();

            saver.setInstances(train);
            saver.setFile(new File(trainPath));
            saver.writeBatch();

            saver.setInstances(dev);
            saver.setFile(new File(devPath));
            saver.writeBatch();

            saver.setInstances(test);
            saver.setFile(new File(testPath));
            saver.writeBatch();

            System.out.println("\nARRAKASTA! Fitxategiak ondo gorde dira 'Partiketak' karpetan.");
            System.out.println("==================================================");

            // Experiment Tracking (Aukerakoa)
            String parametrosUsados = "Train: 80% | Dev: 10% | Test: 10% (Seed=1)";
            String resultadosObtenidos = "Train: " + train.numInstances() + " | Dev: " + dev.numInstances() + " | Test: " + test.numInstances();
            ExperimentLogger.log("2. Datuen Partiketa (DataSplit)", parametrosUsados, resultadosObtenidos);

        } catch (Exception e) {
            System.err.println("Errore kritikoa datuak banatzean.");
            e.printStackTrace();
        }
    }
}