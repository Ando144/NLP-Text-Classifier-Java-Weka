import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;

import java.io.File;

/**
 * Datuen partiketa estratifikatua: Train (80%), Dev (10%), Test (10%).
 */
public class DataSplit {

    public static void main(String[] args) {
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

            // Randomize (Seed = 1)
            System.out.println("Datuak ausazkotzen (Randomize, seed=1)...");
            Randomize rand = new Randomize();
            rand.setRandomSeed(1);
            rand.setInputFormat(data);
            data = Filter.useFilter(data, rand);

            // Partiketa aplikatu
            System.out.println("Partiketa estratifikatua aplikatzen...");

            // TEST (%10)
            StratifiedRemoveFolds testFilter = new StratifiedRemoveFolds();
            testFilter.setNumFolds(10);
            testFilter.setFold(1);
            testFilter.setSeed(1);
            testFilter.setInputFormat(data);
            Instances test = Filter.useFilter(data, testFilter);

            StratifiedRemoveFolds remainderFilter = new StratifiedRemoveFolds();
            remainderFilter.setNumFolds(10);
            remainderFilter.setFold(1);
            remainderFilter.setSeed(1);
            remainderFilter.setInvertSelection(true);
            remainderFilter.setInputFormat(data);
            Instances remainder = Filter.useFilter(data, remainderFilter);

            // DEV (%10)
            StratifiedRemoveFolds devFilter = new StratifiedRemoveFolds();
            devFilter.setNumFolds(9);
            devFilter.setFold(1);
            devFilter.setSeed(1);
            devFilter.setInputFormat(remainder);
            Instances dev = Filter.useFilter(remainder, devFilter);

            // TRAIN (%80)
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

            // Fitxategiak gorde
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