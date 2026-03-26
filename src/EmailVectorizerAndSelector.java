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
 * Bektorizazioa (StringToWordVector) eta atributu-aukeraketa (InfoGain).
 * + Aukerako konfigurazioa
 *
 * PARAMETROAK:
 *   1. baseDir
 *   2. wordsToKeep: Gordetako hitz kopurua
 *   3. numToSelect: Atributu kopurua
 *   4. digitsAsDelim: Digituak erabili bereizle gisa
 *   5. minTermFreq: Gutxieneko maiztasuna
 *   6. useThreshold: InfoGain atalasea 0.005
 *   7. useNormalizer: TextNormalizer aplikatu datu gordinetan
 *   8. maxNGram: N-gramen tamaina maximoa (1, 2, edo 3)
 *
 */
public class EmailVectorizerAndSelector {

    private static final double INFOGAIN_THRESHOLD = 0.005;

    public static void main(String[] args) {
        try {
            if (args.length > 8) {
                System.err.println("Erabilera:");
                System.err.println("  java -cp \"lib/weka.jar:bin\" EmailVectorizerAndSelector");
                System.err.println("       [baseDir] [wordsToKeep] [numToSelect] [digitsAsDelim]");
                System.err.println("       [minTermFreq] [useThreshold] [useNormalizer] [maxNGram]");
                return;
            }

            String  baseDir       = args.length >= 1 ? args[0]                          : "Partiketak";
            int     wordsToKeep   = args.length >= 2 ? Integer.parseInt(args[1])        : 25000;
            int     numToSelect   = args.length >= 3 ? Integer.parseInt(args[2])        : 1000;
            boolean digitsAsDelim = args.length >= 4 ? Boolean.parseBoolean(args[3])    : false;
            int     minTermFreq   = args.length >= 5 ? Integer.parseInt(args[4])        : 1;
            boolean useThreshold  = args.length >= 6 ? Boolean.parseBoolean(args[5])    : false;
            boolean useNormalizer = args.length >= 7 ? Boolean.parseBoolean(args[6])    : false;
            int     maxNGram      = args.length >= 8 ? Integer.parseInt(args[7])        : 1;

            boolean thresholdMode = (numToSelect == -1) || useThreshold;

            printConfig(baseDir, wordsToKeep, numToSelect, digitsAsDelim,
                        minTermFreq, thresholdMode, useNormalizer, maxNGram);

            // Fitxategiak kargatu
            File trainRawFile = new File(baseDir, "train.arff");
            File devRawFile   = new File(baseDir, "dev.arff");
            File testRawFile  = new File(baseDir, "test.arff");

            if (!trainRawFile.exists() || !devRawFile.exists() || !testRawFile.exists()) {
                System.err.println("Sarrerako fitxategiak falta dira: " +
                        new File(baseDir).getAbsolutePath());
                System.err.println("Espero: train.arff, dev.arff, test.arff");
                return;
            }

            Instances train = new DataSource(trainRawFile.getPath()).getDataSet();
            Instances dev   = new DataSource(devRawFile.getPath()).getDataSet();
            Instances test  = new DataSource(testRawFile.getPath()).getDataSet();

            train.setClassIndex(train.numAttributes() - 1);
            dev.setClassIndex(dev.numAttributes() - 1);
            test.setClassIndex(test.numAttributes() - 1);

            // TextNormalizer (Aukerakoa)
            if (useNormalizer) {
                System.out.println("\nTextNormalizer aplikatzen...");
                TextNormalizer normalizer = new TextNormalizer();
                normalizer.setInputFormat(train);
                train = Filter.useFilter(train, normalizer);
                dev   = Filter.useFilter(dev,   normalizer);
                test  = Filter.useFilter(test,  normalizer);
                System.out.println("  TextNormalizer aplikatuta.");
            } else {
                System.out.println("\nTextNormalizer desaktibatua");
            }

            // StringToWordVector
            System.out.println("\nStringToWordVector aplikatzen...");
            StringToWordVector stwv = new StringToWordVector();

            // Transformazio matematikoak
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
                        " \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}0123456789=%"
                    );
                    System.out.println("  Tokenizadorea  = Digituak bereizle");
                } else {
                    tokenizador.setDelimiters(
                        " \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}"
                    );
                    System.out.println("  Tokenizadorea  = Digituak token barruan");
                }
                stwv.setTokenizer(tokenizador);
            }

            stwv.setStemmer(new LovinsStemmer());
            stwv.setStopwordsHandler(new Rainbow());
            stwv.setInputFormat(train);

            Instances trainVec = Filter.useFilter(train, stwv);
            Instances devVec   = Filter.useFilter(dev,   stwv);
            Instances testVec  = Filter.useFilter(test,  stwv);

            System.out.println("  Atributuak bektorizazioaren ostean: " + trainVec.numAttributes());

            // InfoGain
            System.out.println("\nInfoGain aukeraketa aplikatzen...");
            AttributeSelection filterSelector = new AttributeSelection();
            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            Ranker search = new Ranker();

            // Aukeraketa modua: numToSelect edo threshold
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
            Instances devFinal   = Filter.useFilter(devVec,   filterSelector);
            Instances testFinal  = Filter.useFilter(testVec,  filterSelector);

            System.out.println("  Amaierako atributuak: " + trainFinal.numAttributes());

            // Gorde filtroak inferentziarako
            File modelDir = new File("modelo");
            if (!modelDir.exists()) modelDir.mkdirs();
            
            FilterSerializationHelper.saveFilter(stwv, "modelo/vectorizer.ser");
            FilterSerializationHelper.saveFilter(filterSelector, "modelo/selector.ser");
            System.out.println("\nFiltroak gordeta inferentziarako (modelo/vectorizer.ser, modelo/selector.ser)");

            // Gorde
            ArffSaver saver = new ArffSaver();

            saver.setInstances(trainFinal);
            saver.setFile(new File(baseDir, "train_final.arff"));
            saver.writeBatch();

            saver.setInstances(devFinal);
            saver.setFile(new File(baseDir, "dev_final.arff"));
            saver.writeBatch();

            saver.setInstances(testFinal);
            saver.setFile(new File(baseDir, "test_final.arff"));
            saver.writeBatch();

            System.out.println("\n==================================================");
            System.out.println("PROZESUA AMAITUTA!");
            System.out.println("Amaierako atributuak: " + trainFinal.numAttributes());
            System.out.println("==================================================");

            // Erregistroa
            String config = String.format(
                "WordsToKeep=%d | MinTermFreq=%d | DigitsAsDelim=%b | " +
                "ThresholdMode=%b | TextNormalizer=%b | MaxNGram=%d",
                wordsToKeep, minTermFreq, digitsAsDelim, thresholdMode,
                useNormalizer, maxNGram
            );
            String result = "Amaierako atributuak: " + trainFinal.numAttributes();
            ExperimentLogger.log("Bektorizazioa eta Aukeraketa (ablation)", config, result);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

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
        System.out.println("  thresholdMode : " + thresholdMode + (thresholdMode ? " (threshold=" + INFOGAIN_THRESHOLD + ")" : " (numToSelect=" + numToSelect + ")"));
        System.out.println("  maxNGram      : " + maxNGram);
        System.out.println();
    }
}