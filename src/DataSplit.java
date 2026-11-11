import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;

import java.io.File;

/**
 * Splits an ARFF dataset into three subsets:
 * training (train), development (dev) and test.
 *
 *
 * The split is stratified using the Weka library, so the class
 * proportion (spam / legitimate) is kept in every subset.
 *
 *
 *
 * The process has these steps:
 * <ul>
 * <li>Shuffle the data (randomization)</li>
 * <li>Split into 10 folds</li>
 * <li>Assign the folds as follows:
 * <ul>
 * <li>10% → Test</li>
 * <li>10% → Dev</li>
 * <li>80% → Train</li>
 * </ul>
 * </li>
 * </ul>
 *
 *
 *
 * The results are saved as independent ARFF files.
 *
 *
 *
 * This class is part of the email classification pipeline
 * (spam vs legitimate).
 *
 *
 * @version 1.0
 */
public class DataSplit {
    /**
     * Main method: runs the complete data split.
     *
     * <p>
     * The input and output file paths can be given as arguments.
     * Otherwise, default values are used.
     * </p>
     *
     * @param args Command-line arguments:
     *             <ul>
     *             <li>args[0] - Input ARFF file</li>
     *             <li>args[1] - Output path of the train file</li>
     *             <li>args[2] - Output path of the dev file</li>
     *             <li>args[3] - Output path of the test file</li>
     *             </ul>
     */
    public static void main(String[] args) {
        if (args.length > 4) {
            System.err.println(
                    "Usage: java -cp \"lib/weka.jar:bin\" DataSplit [input.arff] [train.arff] [dev.arff] [test.arff]");
            return;
        }

        // Default values adapted to the project folder structure
        String inputPath = args.length >= 1 ? args[0] : "emails_raw.arff";
        String trainPath = args.length >= 2 ? args[1] : "data/splits/train.arff";
        String devPath = args.length >= 3 ? args[2] : "data/splits/dev.arff";
        String testPath = args.length == 4 ? args[3] : "data/splits/test.arff";

        System.out.println("==================================================");
        System.out.println("DATA SPLIT CONFIGURATION (80 / 10 / 10):");
        System.out.println("Input           : " + inputPath);
        System.out.println("Train file      : " + trainPath);
        System.out.println("Dev file        : " + devPath);
        System.out.println("Test file       : " + testPath);
        System.out.println("==================================================\n");

        try {
            File inputFile = new File(inputPath);
            if (!inputFile.exists()) {
                System.err.println("ERROR: The input file does not exist -> " + inputFile.getAbsolutePath());
                return;
            }

            DataSource source = new DataSource(inputPath);
            Instances data = source.getDataSet();

            EmailLoader.ensureClassIndex(data);

            // The data is shuffled to avoid bias in the split
            System.out.println("Shuffling the data (Randomize, seed=1)...");
            Randomize rand = new Randomize();
            rand.setRandomSeed(1);
            rand.setInputFormat(data);
            data = Filter.useFilter(data, rand);

            // Apply the split
            System.out.println("Applying the stratified split...");

            // 10% is taken as the test set (1 of 10 folds)
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

            // With the test fold removed, the remaining 90% is obtained

            // ~10% of the remaining data is taken as the dev set
            StratifiedRemoveFolds devFilter = new StratifiedRemoveFolds();
            devFilter.setNumFolds(9);
            devFilter.setFold(1);
            devFilter.setSeed(1);
            devFilter.setInputFormat(remainder);
            Instances dev = Filter.useFilter(remainder, devFilter);

            // The remaining ~80% is used for training
            StratifiedRemoveFolds trainFilter = new StratifiedRemoveFolds();
            trainFilter.setNumFolds(9);
            trainFilter.setFold(1);
            trainFilter.setSeed(1);
            trainFilter.setInvertSelection(true); // Remove fold 1, keep all the others
            trainFilter.setInputFormat(remainder);
            Instances train = Filter.useFilter(remainder, trainFilter);

            System.out.println("Split finished:");

            EmailLoader.ensureClassIndex(train);
            EmailLoader.ensureClassIndex(dev);
            EmailLoader.ensureClassIndex(test);

            System.out.println(" -> Train : " + train.numInstances() + " instances");
            System.out.println(" -> Dev   : " + dev.numInstances() + " instances");
            System.out.println(" -> Test  : " + test.numInstances() + " instances");

            // The generated subsets are saved as ARFF files
            EmailLoader.saveToArff(train, new File(trainPath));
            EmailLoader.saveToArff(dev, new File(devPath));
            EmailLoader.saveToArff(test, new File(testPath));

            System.out.println("\nSUCCESS! The files were saved in the 'data/splits' folder.");
            System.out.println("==================================================");

            // Experiment tracking (optional)
            String usedParams = "Train: 80% | Dev: 10% | Test: 10% (Seed=1)";
            String obtainedResults = "Train: " + train.numInstances() + " | Dev: " + dev.numInstances()
                    + " | Test: " + test.numInstances();
            ExperimentLogger.log("2. Data Split (DataSplit)", usedParams, obtainedResults);

        } catch (Exception e) {
            System.err.println("Critical error while splitting the data.");
            e.printStackTrace();
        }
    }
}
