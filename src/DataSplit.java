import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.File;

public class DataSplit {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.out.println(
                    "Uso: java -cp \"lib\\weka.jar;bin\" DatasetSplitter <data.arff> <train.arff> <dev.arff> <test.arff>");
            return;
        }

        String inputPath = args[0];
        String trainPath = args[1];
        String devPath = args[2];
        String testPath = args[3];

        // ================================
        // 1. Cargar dataset
        // ================================
        DataSource source = new DataSource(inputPath);
        Instances data = source.getDataSet();

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // ================================
        // 2. Dividir en 3 particiones estratificadas 60/20/20
        // Usamos 5 folds (20% cada uno):
        // fold 1 -> test (20%)
        // fold 2 -> dev (20%)
        // folds 3+4+5 -> train (60%)
        // ================================

        // --- Test: fold 1 ---
        StratifiedRemoveFolds testFilter = new StratifiedRemoveFolds();
        testFilter.setNumFolds(5);
        testFilter.setFold(1);
        testFilter.setInvertSelection(false);
        testFilter.setInputFormat(data);
        Instances test = Filter.useFilter(data, testFilter);

        // --- Dev: fold 2 ---
        StratifiedRemoveFolds devFilter = new StratifiedRemoveFolds();
        devFilter.setNumFolds(5);
        devFilter.setFold(2);
        devFilter.setInvertSelection(false);
        devFilter.setInputFormat(data);
        Instances dev = Filter.useFilter(data, devFilter);

        // --- Train: complemento de fold 1, luego complemento de fold 2
        // -> nos quedamos con los folds 3+4+5 (60%) ---
        StratifiedRemoveFolds trainFilter1 = new StratifiedRemoveFolds();
        trainFilter1.setNumFolds(5);
        trainFilter1.setFold(1);
        trainFilter1.setInvertSelection(true); // todo menos fold 1
        trainFilter1.setInputFormat(data);
        Instances sinFold1 = Filter.useFilter(data, trainFilter1);

        StratifiedRemoveFolds trainFilter2 = new StratifiedRemoveFolds();
        trainFilter2.setNumFolds(4); // sobre los 4 folds restantes
        trainFilter2.setFold(1);
        trainFilter2.setInvertSelection(true); // todo menos fold 2 (ahora es fold 1 de los 4)
        trainFilter2.setInputFormat(sinFold1);
        Instances train = Filter.useFilter(sinFold1, trainFilter2);

        System.out.println("Train : " + train.numInstances() + " instancias");
        System.out.println("Dev   : " + dev.numInstances() + " instancias");
        System.out.println("Test  : " + test.numInstances() + " instancias");

        // ================================
        // 3. Guardar train.arff, dev.arff y test.arff
        // ================================
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

        System.out.println("Ficheros guardados:");
        System.out.println("  " + trainPath);
        System.out.println("  " + devPath);
        System.out.println("  " + testPath);
    }
}