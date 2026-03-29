import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.stemmers.LovinsStemmer;
import weka.core.tokenizers.WordTokenizer;
import weka.core.tokenizers.NGramTokenizer;
import weka.core.stopwords.Rainbow;
import weka.core.converters.ArffSaver;
import java.io.File;

/**
 * Klase honek testu-datuak bektorizatu eta atributu esanguratsuenak hautatzen
 * ditu,
 * emailen sailkapenerako (spam vs legitimoa).
 *
 * 
 * Pipeline hau bi urrats nagusitan banatzen da:
 * 
 *
 * <ul>
 * <li><b>Bektorizazioa (StringToWordVector):</b>
 * <ul>
 * <li>Testua bektore numeriko bihurtzen du (Bag-of-Words eredua)</li>
 * <li>TF-IDF transformazioa aplikatzen da</li>
 * <li>Tokenizazioa (hitzak edo n-gramak)</li>
 * <li>Stopword-en ezabaketa (Rainbow)</li>
 * <li>Stemming (Lovins)</li>
 * </ul>
 * </li>
 * <li><b>Atributu-aukeraketa (InfoGain):</b>
 * <ul>
 * <li>Informazio-irabazia (Information Gain) kalkulatzen du</li>
 * <li>Atributu garrantzitsuenak hautatzen ditu</li>
 * <li>Bi modu:
 * <ul>
 * <li>Top-N atributuak</li>
 * <li>Atalase baten gainetik (threshold)</li>
 * </ul>
 * </li>
 * </ul>
 * </li>
 * </ul>
 *
 * 
 * Gainera, aukerazko konfigurazio hauek onartzen dira:
 * 
 * <ul>
 * <li>TextNormalizer aplikatzea</li>
 * <li>N-gramen erabilera</li>
 * <li>Tokenizazio pertsonalizatua</li>
 * <li>MinTermFreq parametroa</li>
 * </ul>
 *
 * 
 * Azken emaitza train/dev/test dataset eraldatuak dira, eta filtroak
 * serializatzen dira ondorengo inferentziarako.
 * 
 *
 * 
 * Klase hau pipeline-aren erdigunea da, feature engineering fasea kudeatzen
 * duena.
 * 
 *
 * @version 1.0
 */
public class EmailVectorizerAndSelector {

    private static final double INFOGAIN_THRESHOLD = 0.005;

    /**
     * Programa exekutatzen duen metodo nagusia, bektorizazio eta
     * atributu-aukeraketa pipeline osoa aplikatzen duena.
     *
     * <p>
     * Parametroen bidez pipeline-aren konfigurazioa molda daiteke
     * (ablation experiments egiteko diseinatua).
     * </p>
     *
     * @param args Komando lerroko argumentuak:
     *             <ul>
     *             <li>args[0] - baseDir (train/dev/test fitxategien
     *             direktorioa)</li>
     *             <li>args[1] - wordsToKeep (hitz kopuru maximoa)</li>
     *             <li>args[2] - numToSelect (aukeratutako atributuak)</li>
     *             <li>args[3] - digitsAsDelim (digituak bereizle gisa)</li>
     *             <li>args[4] - minTermFreq (gutxieneko maiztasuna)</li>
     *             <li>args[5] - useThreshold (InfoGain atalasea erabili)</li>
     *             <li>args[6] - useNormalizer (TextNormalizer aplikatu)</li>
     *             <li>args[7] - maxNGram (n-gram tamaina maximoa)</li>
     *             </ul>
     */
    public static void main(String[] args) {
        try {
            if (args.length > 8) {
                System.err.println("Erabilera:");
                System.err.println("  java -cp \"lib/weka.jar:bin\" EmailVectorizerAndSelector");
                System.err.println("       [baseDir] [wordsToKeep] [numToSelect] [digitsAsDelim]");
                System.err.println("       [minTermFreq] [useThreshold] [useNormalizer] [maxNGram]");
                return;
            }

            String baseDir = args.length >= 1 ? args[0] : "Partiketak";
            int wordsToKeep = args.length >= 2 ? Integer.parseInt(args[1]) : 25000;
            int numToSelect = args.length >= 3 ? Integer.parseInt(args[2]) : 1000;
            boolean digitsAsDelim = args.length >= 4 ? Boolean.parseBoolean(args[3]) : false;
            int minTermFreq = args.length >= 5 ? Integer.parseInt(args[4]) : 1;
            boolean useThreshold = args.length >= 6 ? Boolean.parseBoolean(args[5]) : false;
            boolean useNormalizer = args.length >= 7 ? Boolean.parseBoolean(args[6]) : false;
            int maxNGram = args.length >= 8 ? Integer.parseInt(args[7]) : 1;

            boolean thresholdMode = (numToSelect == -1) || useThreshold;
            printConfig(baseDir, wordsToKeep, numToSelect, digitsAsDelim,
                    minTermFreq, thresholdMode, useNormalizer, maxNGram);

            // Fitxategiak kargatu
            File trainRawFile = new File(baseDir, "train.arff");
            File devRawFile = new File(baseDir, "dev.arff");
            File testRawFile = new File(baseDir, "test.arff");

            if (!trainRawFile.exists() || !devRawFile.exists() || !testRawFile.exists()) {
                System.err.println("Sarrerako fitxategiak falta dira: " +
                        new File(baseDir).getAbsolutePath());
                System.err.println("Espero: train.arff, dev.arff, test.arff");
                return;
            }

            Instances train = new DataSource(trainRawFile.getPath()).getDataSet();
            Instances dev = new DataSource(devRawFile.getPath()).getDataSet();
            Instances test = new DataSource(testRawFile.getPath()).getDataSet();

            EmailLoader.ensureClassIndex(train);
            EmailLoader.ensureClassIndex(dev);
            EmailLoader.ensureClassIndex(test);

            // Aukeran, testuaren aurreprozesamendua aplikatzen da (garbiketa,
            // normalizazioa)
            if (useNormalizer) {
                System.out.println("\nTextNormalizer aplikatzen...");
                TextNormalizer normalizer = new TextNormalizer();
                normalizer.setInputFormat(train);
                train = Filter.useFilter(train, normalizer);
                dev = Filter.useFilter(dev, normalizer);
                test = Filter.useFilter(test, normalizer);
                System.out.println("  TextNormalizer aplikatuta.");
            } else {
                System.out.println("\nTextNormalizer desaktibatua");
            }

            // Testua bektore numeriko bihurtzen da TF-IDF erabiliz
            System.out.println("\nStringToWordVector aplikatzen...");
            StringToWordVector stwv = new StringToWordVector();

            // Tokenizazioa: hitzak edo n-gramak, konfigurazioaren arabera
            stwv.setIDFTransform(true);
            stwv.setTFTransform(true);
            stwv.setLowerCaseTokens(true);
            stwv.setWordsToKeep(wordsToKeep);

            // MinTermFreq (Aukerakoa)
            stwv.setMinTermFreq(minTermFreq);
            System.out.println("  MinTermFreq    = " + minTermFreq);

            // N-grama (Aukerakoa)
            if (maxNGram > 1) {
                NGramTokenizer ngramTokenizer = new NGramTokenizer();
                ngramTokenizer.setNGramMinSize(1);
                ngramTokenizer.setNGramMaxSize(maxNGram);
                ngramTokenizer.setDelimiters(" \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|\\[\\]{}");
                stwv.setTokenizer(ngramTokenizer);
                System.out.println("  Tokenizadorea  = NGram (min=1, max=" + maxNGram + ")");
            } else {
                WordTokenizer tokenizador = new WordTokenizer();
                if (digitsAsDelim) {
                    tokenizador.setDelimiters(
                            " \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}0123456789=%");
                    System.out.println("  Tokenizadorea  = Digituak bereizle");
                } else {
                    tokenizador.setDelimiters(
                            " \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}");
                    System.out.println("  Tokenizadorea  = Digituak token barruan");
                }
                stwv.setTokenizer(tokenizador);
            }

            stwv.setStemmer(new LovinsStemmer());
            stwv.setStopwordsHandler(new Rainbow());
            stwv.setInputFormat(train);

            Instances trainVec = Filter.useFilter(train, stwv);
            Instances devVec = Filter.useFilter(dev, stwv);
            Instances testVec = Filter.useFilter(test, stwv);

            System.out.println("  Atributuak bektorizazioaren ostean: " + trainVec.numAttributes());

            // Atributu garrantzitsuenak hautatzen dira Information Gain erabiliz
            System.out.println("\nInfoGain aukeraketa aplikatzen...");
            AttributeSelection filterSelector = new AttributeSelection();
            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            Ranker search = new Ranker();

            // Bi aukera: atributu kopuru finkoa edo atalase bidezko aukeraketa
            if (thresholdMode) {
                search.setNumToSelect(-1);
                search.setThreshold(INFOGAIN_THRESHOLD);
                System.out.println("  Modua          = atalasea >= " + INFOGAIN_THRESHOLD);
            } else {
                search.setNumToSelect(numToSelect);
                search.setThreshold(-Double.MAX_VALUE);
                System.out.println("  Modua          = numToSelect=" + numToSelect);
            }

            filterSelector.setEvaluator(eval);
            filterSelector.setSearch(search);
            filterSelector.setInputFormat(trainVec);

            Instances trainFinal = Filter.useFilter(trainVec, filterSelector);
            Instances devFinal = Filter.useFilter(devVec, filterSelector);
            Instances testFinal = Filter.useFilter(testVec, filterSelector);

            System.out.println("  Amaierako atributuak: " + trainFinal.numAttributes());

            EmailLoader.sanitizeAttributeNames(trainFinal);
            EmailLoader.sanitizeAttributeNames(devFinal);
            EmailLoader.sanitizeAttributeNames(testFinal);

            // Gorde filtroak inferentziarako
            File modelDir = new File("modelo");
            if (!modelDir.exists())
                modelDir.mkdirs();

            // Filtroak gordetzen dira, gero inferentzian berrerabiltzeko
            FilterSerializationHelper.saveFilter(stwv, "modelo/vectorizer.ser");
            FilterSerializationHelper.saveFilter(filterSelector, "modelo/selector.ser");
            System.out.println("\nFiltroak gordeta inferentziarako (modelo/vectorizer.ser, modelo/selector.ser)");

            // Azken dataset eraldatuak fitxategietan gordetzen dira
            EmailLoader.saveToArff(trainFinal, new File(baseDir, "train_final.arff"));
            EmailLoader.saveToArff(devFinal, new File(baseDir, "dev_final.arff"));
            EmailLoader.saveToArff(testFinal, new File(baseDir, "test_final.arff"));

            System.out.println("\n==================================================");
            System.out.println("PROZESUA AMAITUTA!");
            System.out.println("Amaierako atributuak: " + trainFinal.numAttributes());
            System.out.println("==================================================");

            // Erregistroa
            String config = String.format(
                    "WordsToKeep=%d | MinTermFreq=%d | DigitsAsDelim=%b | " +
                            "ThresholdMode=%b | TextNormalizer=%b | MaxNGram=%d",
                    wordsToKeep, minTermFreq, digitsAsDelim, thresholdMode,
                    useNormalizer, maxNGram);
            String result = "Amaierako atributuak: " + trainFinal.numAttributes();
            ExperimentLogger.log("Bektorizazioa eta Aukeraketa (ablation)", config, result);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Exekuzioaren konfigurazioa kontsolan inprimatzen du,
     * esperimentuen trazabilitatea errazteko.
     *
     * @param baseDir       Datuen direktorioa
     * @param wordsToKeep   Hitz kopuru maximoa
     * @param numToSelect   Hautatutako atributu kopurua
     * @param digitsAsDelim Digituak bereizle diren ala ez
     * @param minTermFreq   Gutxieneko maiztasuna
     * @param thresholdMode Atalase bidezko aukeraketa aktibatuta dagoen
     * @param useNormalizer TextNormalizer erabilera
     * @param maxNGram      N-gram tamaina maximoa
     */
    private static void printConfig(String baseDir, int wordsToKeep, int numToSelect,
            boolean digitsAsDelim, int minTermFreq,
            boolean thresholdMode, boolean useNormalizer,
            int maxNGram) {
        System.out.println("\n--- KONFIGURAZIOA ---");
        System.out.println("  baseDir       : " + baseDir);
        System.out.println("  wordsToKeep   : " + wordsToKeep);
        System.out.println("  minTermFreq   : " + minTermFreq);
        System.out.println("  digitsAsDelim : " + digitsAsDelim);
        System.out.println("  useNormalizer : " + useNormalizer);
        System.out.println("  thresholdMode : " + thresholdMode
                + (thresholdMode ? " (threshold=" + INFOGAIN_THRESHOLD + ")" : " (numToSelect=" + numToSelect + ")"));
        System.out.println("  maxNGram      : " + maxNGram);
        System.out.println();
    }
}