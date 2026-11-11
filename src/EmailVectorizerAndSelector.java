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
 * Vectorizes the text data and selects the most relevant
 * attributes,
 * for email classification (spam vs legitimate).
 *
 *
 * This pipeline has two main steps:
 *
 *
 * <ul>
 * <li><b>Vectorization (StringToWordVector):</b>
 * <ul>
 * <li>Turns the text into a numeric vector (Bag-of-Words model)</li>
 * <li>Applies the TF-IDF transformation</li>
 * <li>Tokenization (words or n-grams)</li>
 * <li>Stopword removal (Rainbow)</li>
 * <li>Stemming (Lovins)</li>
 * </ul>
 * </li>
 * <li><b>Attribute selection (InfoGain):</b>
 * <ul>
 * <li>Computes the Information Gain</li>
 * <li>Selects the most important attributes</li>
 * <li>Two modes:
 * <ul>
 * <li>Top-N attributes</li>
 * <li>Above a threshold</li>
 * </ul>
 * </li>
 * </ul>
 * </li>
 * </ul>
 *
 *
 * In addition, these optional settings are supported:
 *
 * <ul>
 * <li>Applying TextNormalizer</li>
 * <li>Using n-grams</li>
 * <li>Custom tokenization</li>
 * <li>MinTermFreq parameter</li>
 * </ul>
 *
 *
 * The final result is the transformed train/dev/test datasets, and the
 * filters are serialized for later inference.
 *
 *
 *
 * This class is the core of the pipeline: it manages the feature
 * engineering phase.
 *
 *
 * @version 1.0
 */
public class EmailVectorizerAndSelector {

    private static final double INFOGAIN_THRESHOLD = 0.005;

    /**
     * Main method: applies the complete vectorization and
     * attribute selection pipeline.
     *
     * <p>
     * The pipeline configuration can be changed with the parameters
     * (designed for ablation experiments).
     * </p>
     *
     * @param args Command-line arguments:
     *             <ul>
     *             <li>args[0] - baseDir (directory of the train/dev/test
     *             files)</li>
     *             <li>args[1] - wordsToKeep (maximum number of words)</li>
     *             <li>args[2] - numToSelect (number of selected attributes)</li>
     *             <li>args[3] - digitsAsDelim (digits used as delimiters)</li>
     *             <li>args[4] - minTermFreq (minimum term frequency)</li>
     *             <li>args[5] - useThreshold (use the InfoGain threshold)</li>
     *             <li>args[6] - useNormalizer (apply TextNormalizer)</li>
     *             <li>args[7] - maxNGram (maximum n-gram size)</li>
     *             </ul>
     */
    public static void main(String[] args) {
        try {
            if (args.length > 8) {
                System.err.println("Usage:");
                System.err.println("  java -cp \"lib/weka.jar:bin\" EmailVectorizerAndSelector");
                System.err.println("       [baseDir] [wordsToKeep] [numToSelect] [digitsAsDelim]");
                System.err.println("       [minTermFreq] [useThreshold] [useNormalizer] [maxNGram]");
                return;
            }

            String baseDir = args.length >= 1 ? args[0] : "data/splits";
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

            // Load the files
            File trainRawFile = new File(baseDir, "train.arff");
            File devRawFile = new File(baseDir, "dev.arff");
            File testRawFile = new File(baseDir, "test.arff");

            if (!trainRawFile.exists() || !devRawFile.exists() || !testRawFile.exists()) {
                System.err.println("Input files are missing: " +
                        new File(baseDir).getAbsolutePath());
                System.err.println("Expected: train.arff, dev.arff, test.arff");
                return;
            }

            Instances train = new DataSource(trainRawFile.getPath()).getDataSet();
            Instances dev = new DataSource(devRawFile.getPath()).getDataSet();
            Instances test = new DataSource(testRawFile.getPath()).getDataSet();

            EmailLoader.ensureClassIndex(train);
            EmailLoader.ensureClassIndex(dev);
            EmailLoader.ensureClassIndex(test);

            // Optionally, text preprocessing is applied (cleaning,
            // normalization)
            TextNormalizer normalizer = null;
            if (useNormalizer) {
                System.out.println("\nApplying TextNormalizer...");
                normalizer = new TextNormalizer();
                normalizer.setInputFormat(train);
                train = Filter.useFilter(train, normalizer);
                dev = Filter.useFilter(dev, normalizer);
                test = Filter.useFilter(test, normalizer);
                System.out.println("  TextNormalizer applied.");
            } else {
                System.out.println("\nTextNormalizer disabled");
            }

            // The text is turned into a numeric vector using TF-IDF
            System.out.println("\nApplying StringToWordVector...");
            StringToWordVector stwv = new StringToWordVector();

            // Tokenization: words or n-grams, depending on the configuration
            stwv.setIDFTransform(true);
            stwv.setTFTransform(true);
            stwv.setLowerCaseTokens(true);
            stwv.setWordsToKeep(wordsToKeep);

            // MinTermFreq (optional)
            stwv.setMinTermFreq(minTermFreq);
            System.out.println("  MinTermFreq    = " + minTermFreq);

            // N-grams (optional)
            if (maxNGram > 1) {
                NGramTokenizer ngramTokenizer = new NGramTokenizer();
                ngramTokenizer.setNGramMinSize(1);
                ngramTokenizer.setNGramMaxSize(maxNGram);
                ngramTokenizer.setDelimiters(" \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|\\[\\]{}");
                stwv.setTokenizer(ngramTokenizer);
                System.out.println("  Tokenizer      = NGram (min=1, max=" + maxNGram + ")");
            } else {
                WordTokenizer wordTokenizer = new WordTokenizer();
                if (digitsAsDelim) {
                    wordTokenizer.setDelimiters(
                            " \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}0123456789=%");
                    System.out.println("  Tokenizer      = Digits as delimiters");
                } else {
                    wordTokenizer.setDelimiters(
                            " \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}");
                    System.out.println("  Tokenizer      = Digits kept inside tokens");
                }
                stwv.setTokenizer(wordTokenizer);
            }

            stwv.setStemmer(new LovinsStemmer());
            stwv.setStopwordsHandler(new Rainbow());
            stwv.setInputFormat(train);

            Instances trainVec = Filter.useFilter(train, stwv);
            Instances devVec = Filter.useFilter(dev, stwv);
            Instances testVec = Filter.useFilter(test, stwv);

            System.out.println("  Attributes after vectorization: " + trainVec.numAttributes());

            // The most important attributes are selected with Information Gain
            System.out.println("\nApplying InfoGain selection...");
            AttributeSelection filterSelector = new AttributeSelection();
            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            Ranker search = new Ranker();

            // Two options: fixed number of attributes or threshold-based selection
            if (thresholdMode) {
                search.setNumToSelect(-1);
                search.setThreshold(INFOGAIN_THRESHOLD);
                System.out.println("  Mode           = threshold >= " + INFOGAIN_THRESHOLD);
            } else {
                search.setNumToSelect(numToSelect);
                search.setThreshold(-Double.MAX_VALUE);
                System.out.println("  Mode           = numToSelect=" + numToSelect);
            }

            filterSelector.setEvaluator(eval);
            filterSelector.setSearch(search);
            filterSelector.setInputFormat(trainVec);

            Instances trainFinal = Filter.useFilter(trainVec, filterSelector);
            Instances devFinal = Filter.useFilter(devVec, filterSelector);
            Instances testFinal = Filter.useFilter(testVec, filterSelector);

            System.out.println("  Final attributes: " + trainFinal.numAttributes());

            EmailLoader.sanitizeAttributeNames(trainFinal);
            EmailLoader.sanitizeAttributeNames(devFinal);
            EmailLoader.sanitizeAttributeNames(testFinal);

            // Save the filters for inference
            File modelDir = new File("model");
            if (!modelDir.exists())
                modelDir.mkdirs();

            // The filters are saved so they can be reused at inference time
            FilterSerializationHelper.saveFilter(stwv, "model/vectorizer.ser");
            FilterSerializationHelper.saveFilter(filterSelector, "model/selector.ser");
            if (normalizer != null) {
                FilterSerializationHelper.saveFilter(normalizer, "model/normalizer.ser");
                System.out.println("  TextNormalizer saved for inference (model/normalizer.ser)");
            } else {
                File nf = new File("model/normalizer.ser");
                if (nf.exists())
                    nf.delete();
            }
            System.out.println("\nFilters saved for inference (model/vectorizer.ser, model/selector.ser)");

            // The final transformed datasets are saved to files
            EmailLoader.saveToArff(trainFinal, new File(baseDir, "train_final.arff"));
            EmailLoader.saveToArff(devFinal, new File(baseDir, "dev_final.arff"));
            EmailLoader.saveToArff(testFinal, new File(baseDir, "test_final.arff"));

            System.out.println("\n==================================================");
            System.out.println("PROCESS FINISHED!");
            System.out.println("Final attributes: " + trainFinal.numAttributes());
            System.out.println("==================================================");

            // Logging
            String config = String.format(
                    "WordsToKeep=%d | MinTermFreq=%d | DigitsAsDelim=%b | " +
                            "ThresholdMode=%b | TextNormalizer=%b | MaxNGram=%d",
                    wordsToKeep, minTermFreq, digitsAsDelim, thresholdMode,
                    useNormalizer, maxNGram);
            String result = "Final attributes: " + trainFinal.numAttributes();
            ExperimentLogger.log("Vectorization and Selection (ablation)", config, result);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Prints the run configuration to the console,
     * to make the experiments easier to trace.
     *
     * @param baseDir       Data directory
     * @param wordsToKeep   Maximum number of words
     * @param numToSelect   Number of selected attributes
     * @param digitsAsDelim Whether digits are delimiters or not
     * @param minTermFreq   Minimum term frequency
     * @param thresholdMode Whether threshold-based selection is enabled
     * @param useNormalizer Whether TextNormalizer is used
     * @param maxNGram      Maximum n-gram size
     */
    private static void printConfig(String baseDir, int wordsToKeep, int numToSelect,
            boolean digitsAsDelim, int minTermFreq,
            boolean thresholdMode, boolean useNormalizer,
            int maxNGram) {
        System.out.println("\n--- CONFIGURATION ---");
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
