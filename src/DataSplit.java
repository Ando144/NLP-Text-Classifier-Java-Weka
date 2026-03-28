import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Randomize;

import java.io.File;

/**
 * Klase honek ARFF formatuko dataset bat hiru azpimultzotan banatzen du:
 * entrenamendua (train), garapena (dev) eta testa (test).
 *
 * 
 * Banaketa modu estratifikatuan egiten da Weka liburutegia erabiliz,
 * klaseen proportzioa (spam / legitimoa) azpimultzo guztietan mantenduz.
 * 
 *
 * 
 * Prozesuak urrats hauek ditu:
 * <ul>
 * <li>Datuen ausazkotzea (randomizazioa)</li>
 * <li>10 fold-etan banaketa</li>
 * <li>Honako esleipena:
 * <ul>
 * <li>%10 → Test</li>
 * <li>%10 → Dev</li>
 * <li>%80 → Train</li>
 * </ul>
 * </li>
 * </ul>
 * 
 *
 * 
 * Emaitzak ARFF fitxategi independenteetan gordetzen dira.
 * 
 *
 * 
 * Klase hau emailen sailkapen pipeline-aren parte da
 * (spam vs legitimoa).
 * 
 *
 * @version 1.0
 */
public class DataSplit {
    /**
     * Programa exekutatzen duen metodo nagusia, datuen banaketa osoa egiten duena.
     *
     * <p>
     * Sarrerako eta irteerako fitxategien bideak argumentuen bidez pasa daitezke.
     * Bestela, balio lehenetsiak erabiliko dira.
     * </p>
     *
     * @param args Komando lerroko argumentuak:
     *             <ul>
     *             <li>args[0] - Sarrerako ARFF fitxategia</li>
     *             <li>args[1] - Train fitxategiaren irteera</li>
     *             <li>args[2] - Dev fitxategiaren irteera</li>
     *             <li>args[3] - Test fitxategiaren irteera</li>
     *             </ul>
     */
    public static void main(String[] args) {
        if (args.length > 4) {
            System.err.println(
                    "Erabilera: java -cp \"lib/weka.jar:bin\" DataSplit [input.arff] [train.arff] [dev.arff] [test.arff]");
            return;
        }

        // Zure karpeten egiturara egokitutako balio lehenetsiak (Irudian oinarrituta)
        String inputPath = args.length >= 1 ? args[0] : "emails_raw.arff";
        String trainPath = args.length >= 2 ? args[1] : "Partiketak/train.arff";
        String devPath = args.length >= 3 ? args[2] : "Partiketak/dev.arff";
        String testPath = args.length == 4 ? args[3] : "Partiketak/test.arff";

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

            EmailLoader.ensureClassIndex(data);

            // Datuak ausaz nahasten dira, banaketan bias-a saihesteko
            System.out.println("Datuak ausazkotzen (Randomize, seed=1)...");
            Randomize rand = new Randomize();
            rand.setRandomSeed(1);
            rand.setInputFormat(data);
            data = Filter.useFilter(data, rand);

            // Partiketa aplikatu
            System.out.println("Partiketa estratifikatua aplikatzen...");

            // %10 test multzo gisa hartzen da (10 fold-etik 1)
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

            // Test-eko fold-a kenduta, %90eko multzoa lortzen da

            // Gainerako datuetatik ~%10 dev multzo gisa hartzen da
            StratifiedRemoveFolds devFilter = new StratifiedRemoveFolds();
            devFilter.setNumFolds(9);
            devFilter.setFold(1);
            devFilter.setSeed(1);
            devFilter.setInputFormat(remainder);
            Instances dev = Filter.useFilter(remainder, devFilter);

            // Gainerako ~%80 entrenamendurako erabiltzen da
            StratifiedRemoveFolds trainFilter = new StratifiedRemoveFolds();
            trainFilter.setNumFolds(9);
            trainFilter.setFold(1);
            trainFilter.setSeed(1);
            trainFilter.setInvertSelection(true); // Fold 1 kendu, beste guztiak mantendu
            trainFilter.setInputFormat(remainder);
            Instances train = Filter.useFilter(remainder, trainFilter);

            System.out.println("Banaketa amaituta:");

            EmailLoader.ensureClassIndex(train);
            EmailLoader.ensureClassIndex(dev);
            EmailLoader.ensureClassIndex(test);

            System.out.println(" -> Train : " + train.numInstances() + " instantzia");
            System.out.println(" -> Dev   : " + dev.numInstances() + " instantzia");
            System.out.println(" -> Test  : " + test.numInstances() + " instantzia");

            // Sortutako azpimultzoak ARFF fitxategietan gordetzen dira
            EmailLoader.saveToArff(train, new File(trainPath));
            EmailLoader.saveToArff(dev, new File(devPath));
            EmailLoader.saveToArff(test, new File(testPath));

            System.out.println("\nARRAKASTA! Fitxategiak ondo gorde dira 'Partiketak' karpetan.");
            System.out.println("==================================================");

            // Experiment Tracking (Aukerakoa)
            String parametrosUsados = "Train: 80% | Dev: 10% | Test: 10% (Seed=1)";
            String resultadosObtenidos = "Train: " + train.numInstances() + " | Dev: " + dev.numInstances()
                    + " | Test: " + test.numInstances();
            ExperimentLogger.log("2. Datuen Partiketa (DataSplit)", parametrosUsados, resultadosObtenidos);

        } catch (Exception e) {
            System.err.println("Errore kritikoa datuak banatzean.");
            e.printStackTrace();
        }
    }
}