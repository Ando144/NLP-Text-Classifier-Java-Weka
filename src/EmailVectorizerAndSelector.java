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
 * Klase honek mezu elektronikoen testuak bektorizatzen ditu (StringToWordVector)
 * eta, ondoren, ezaugarri onenak aukeratzen ditu (AttributeSelection - InfoGain)
 * dimentsionalitatea murrizteko.
 *
 * EXEKUZIO ADIBIDEAK (ablation study):
 *
 *   -- Oinarrizko bertsioa (zaharra, erreferentzia gisa) --
 *   java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 20000 1000 false 1 false false 1
 *
 *   -- N-grama bakarrik (unigrama + bigrama) --
 *   java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 20000 1000 false 1 false false 2
 *
 *   -- N-grama bakarrik (unigrama + bigrama + trigrama) --
 *   java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 20000 1000 false 1 false false 3
 *
 *   -- N-grama + minTermFreq=2 (gomendatua) --
 *   java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 20000 1000 false 2 false false 2
 *
 *   -- TextNormalizer bakarrik --
 *   java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 20000 1000 false 1 false true 1
 *
 *   -- ALDAKETA 1 bakarrik (minTermFreq=2) --
 *   java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 20000 1000 false 2 false false 1
 *
 *   -- ALDAKETA 2 bakarrik (digits tokenizer) --
 *   java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 20000 1000 true 1 false false 1
 *
 *   -- ALDAKETA 3 bakarrik (InfoGain atalasea) --
 *   java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 20000 -1 false 1 true false 1
 *
 *   -- Dena batera --
 *   java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 30000 -1 true 2 true true 2
 *
 * PARAMETROAK:
 *   1. baseDir         : karpeta (adib. "Partiketak")
 *   2. wordsToKeep     : gordetako hitz kopurua (adib. 20000)
 *   3. numToSelect     : InfoGain kopurua; -1 = atalasea erabili
 *   4. digitsAsDelim   : true = digituak bereizle gisa (ALDAKETA 2)
 *   5. minTermFreq     : gutxieneko maiztasuna (1 = zaharra, 2 = ALDAKETA 1)
 *   6. useThreshold    : true = InfoGain atalasea 0.005 (ALDAKETA 3)
 *   7. useNormalizer   : true = TextNormalizer aplikatu aurretik
 *   8. maxNGram        : n-gramen tamaina maximoa (1 = unigrama zaharra,
 *                        2 = unigrama+bigrama, 3 = unigrama+bigrama+trigrama)
 *
 * OHARRA N-GRAMEI BURUZ:
 *   N-gramak hitz konbinazioak dira. Adibidez, "click here" edo "free offer"
 *   spam-ean oso ohikoak dira, baina hitz bakoitza bere aldetik ere ham-ean
 *   ager daiteke. Bigramak konbinazio horiek atributu gisa kapturatzen ditu,
 *   unigramak baino seinale aberatsagoa emanez.
 *
 *   Kontuan hartu: n-gramak atributu kopurua asko handitzen du bektorizazio
 *   ostean (20000tik 60000+ ere joan daiteke). Horregatik InfoGain aukeraketa
 *   are garrantzitsuagoa da n-gramak erabiltzean.
 *
 *   wordsToKeep handiagoa izatea gomendatzen da n-gramak erabiltzean (adib.
 *   30000), bestela n-gramek unigrama onak kanpo utz ditzakete lehiaketan.
 *
 * @author WekaProyecto2026 Taldea
 */
public class EmailVectorizerAndSelector {

    private static final double INFOGAIN_THRESHOLD = 0.005;

    public static void main(String[] args) {
        try {
            // ================================================
            // 1. PARAMETROEN KUDEAKETA
            // ================================================
            if (args.length > 8) {
                printUsage();
                return;
            }

            String  baseDir       = args.length >= 1 ? args[0]                       : "Partiketak";
            int     wordsToKeep   = args.length >= 2 ? Integer.parseInt(args[1])      : 20000;
            int     numToSelect   = args.length >= 3 ? Integer.parseInt(args[2])      : 1000;
            boolean digitsAsDelim = args.length >= 4 ? Boolean.parseBoolean(args[3]) : false;
            int     minTermFreq   = args.length >= 5 ? Integer.parseInt(args[4])      : 1;
            boolean useThreshold  = args.length >= 6 ? Boolean.parseBoolean(args[5]) : false;
            boolean useNormalizer = args.length >= 7 ? Boolean.parseBoolean(args[6]) : false;
            int     maxNGram      = args.length >= 8 ? Integer.parseInt(args[7])      : 1;

            boolean thresholdMode = (numToSelect == -1) || useThreshold;

            printConfig(baseDir, wordsToKeep, numToSelect, digitsAsDelim,
                        minTermFreq, thresholdMode, useNormalizer, maxNGram);

            // ================================================
            // 2. FITXATEGIAK EGIAZTATU ETA KARGATU
            // ================================================
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

            // ================================================
            // 3. TextNormalizer (StringToWordVector AURRETIK)
            // ================================================
            if (useNormalizer) {
                System.out.println("\nTextNormalizer aplikatzen...");
                TextNormalizer normalizer = new TextNormalizer();
                normalizer.setInputFormat(train);
                train = Filter.useFilter(train, normalizer);
                dev   = Filter.useFilter(dev,   normalizer);
                test  = Filter.useFilter(test,  normalizer);
                System.out.println("  TextNormalizer aplikatuta.");
            } else {
                System.out.println("\nTextNormalizer: desaktibatua (zaharra)");
            }

            // ================================================
            // 4. StringToWordVector
            // ================================================
            System.out.println("\nStringToWordVector aplikatzen...");
            StringToWordVector stwv = new StringToWordVector();

            // Transformazio matematikoak
            stwv.setIDFTransform(true);
            stwv.setTFTransform(true);
            stwv.setLowerCaseTokens(true);
            stwv.setWordsToKeep(wordsToKeep);

            // ---------------------------------------------------
            // ALDAKETA 1 — MinTermFreq
            // ---------------------------------------------------
            stwv.setMinTermFreq(minTermFreq);
            System.out.println("  MinTermFreq    = " + minTermFreq +
                    (minTermFreq == 1 ? " (zaharra)" : " (ALDAKETA 1 — aktibatua)"));

            // ---------------------------------------------------
            // ALDAKETA 2 vs N-GRAMA — Tokenizadorea
            // ---------------------------------------------------
            // N-gramak aktibatuta daudenean (maxNGram > 1), NGramTokenizer
            // erabiltzen da WordTokenizer-en ordez. Bi tokenizadoreak
            // bateraezinak dira: NGramTokenizer-ek bere bereizleak ditu
            // eta gainera n-gramen eraikuntza kudeatzen du berak.
            //
            // digitsAsDelim=true eta maxNGram>1 batera erabiltzea ez da
            // gomendatzen, NGramTokenizer-ek bereizle propioak baititu.
            if (maxNGram > 1) {
                // N-GRAMA TOKENIZADOREA
                // NGramTokenizer-ek zuzenean sortzen ditu n-gramak:
                //   minNGram=1 → unigramak ere sartzen dira
                //   maxNGram=2 → unigrama + bigrama
                //   maxNGram=3 → unigrama + bigrama + trigrama
                NGramTokenizer ngramTokenizer = new NGramTokenizer();
                ngramTokenizer.setNGramMinSize(1);
                ngramTokenizer.setNGramMaxSize(maxNGram);
                // Bereizleak: hitz-mugak zehazten dituzte
                ngramTokenizer.setDelimiters(" \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|\\[\\]{}");
                stwv.setTokenizer(ngramTokenizer);
                System.out.println("  Tokenizadorea  = NGram (min=1, max=" + maxNGram + ") — AKTIBATUA");
                if (digitsAsDelim) {
                    System.out.println("  OHARRA: digitsAsDelim=true ignoratzen da NGram moduan.");
                }
            } else {
                // JOKABIDE ZAHARRA — WordTokenizer
                WordTokenizer tokenizador = new WordTokenizer();
                if (digitsAsDelim) {
                    tokenizador.setDelimiters(
                        " \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}0123456789=%"
                    );
                    System.out.println("  Tokenizadorea  = digituak bereizle (ALDAKETA 2 — aktibatua)");
                } else {
                    tokenizador.setDelimiters(
                        " \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}"
                    );
                    System.out.println("  Tokenizadorea  = zaharra (digituak token barruan)");
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

            // ================================================
            // 5. ATRIBUTUEN AUKERAKETA (InfoGain)
            // ================================================
            System.out.println("\nInfoGain Aukeraketa aplikatzen...");
            AttributeSelection filterSelector = new AttributeSelection();
            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            Ranker search = new Ranker();

            if (thresholdMode) {
                search.setNumToSelect(-1);
                search.setThreshold(INFOGAIN_THRESHOLD);
                System.out.println("  Modua          = atalasea >= " + INFOGAIN_THRESHOLD +
                        " (ALDAKETA 3 — aktibatua)");
            } else {
                search.setNumToSelect(numToSelect);
                search.setThreshold(-Double.MAX_VALUE);
                System.out.println("  Modua          = numToSelect=" + numToSelect + " (zaharra)");
            }

            filterSelector.setEvaluator(eval);
            filterSelector.setSearch(search);
            filterSelector.setInputFormat(trainVec);

            Instances trainFinal = Filter.useFilter(trainVec, filterSelector);
            Instances devFinal   = Filter.useFilter(devVec,   filterSelector);
            Instances testFinal  = Filter.useFilter(testVec,  filterSelector);

            System.out.println("  Amaierako atributuak: " + trainFinal.numAttributes());

            // ================================================
            // 6. GORDE
            // ================================================
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

            // ================================================
            // 7. ERREGISTROA
            // ================================================
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
        System.out.println("==================================================");
        System.out.println("ABLATION STUDY — KONFIGURAZIOA:");
        System.out.println("  baseDir        : " + baseDir);
        System.out.println("  wordsToKeep    : " + wordsToKeep);
        System.out.println("  useNormalizer  : " + useNormalizer +
                           (useNormalizer ? " ← TextNormalizer aktibo" : " (zaharra)"));
        System.out.println("  digitsAsDelim  : " + digitsAsDelim +
                           (digitsAsDelim ? " ← ALDAKETA 2 aktibo" : " (zaharra)"));
        System.out.println("  minTermFreq    : " + minTermFreq +
                           (minTermFreq > 1 ? " ← ALDAKETA 1 aktibo" : " (zaharra)"));
        System.out.println("  thresholdMode  : " + thresholdMode +
                           (thresholdMode
                               ? " ← ALDAKETA 3 aktibo (threshold=" + INFOGAIN_THRESHOLD + ")"
                               : " (zaharra, numToSelect=" + numToSelect + ")"));
        System.out.println("  maxNGram       : " + maxNGram +
                           (maxNGram == 1 ? " (unigrama zaharra)"
                           : maxNGram == 2 ? " ← unigrama + bigrama aktibo"
                                           : " ← unigrama + bigrama + trigrama aktibo"));
        System.out.println("==================================================");
    }

    private static void printUsage() {
        System.err.println("Erabilera:");
        System.err.println("  java -cp \"lib/weka.jar:bin\" EmailVectorizerAndSelector");
        System.err.println("       [baseDir] [wordsToKeep] [numToSelect] [digitsAsDelim]");
        System.err.println("       [minTermFreq] [useThreshold] [useNormalizer] [maxNGram]");
        System.err.println();
        System.err.println("Ablation study adibideak:");
        System.err.println("  Zaharra (erreferentzia):");
        System.err.println("    ... Partiketak 20000 1000 false 1 false false 1");
        System.err.println("  N-grama bakarrik (unigrama+bigrama):");
        System.err.println("    ... Partiketak 20000 1000 false 1 false false 2");
        System.err.println("  N-grama + minTermFreq=2 (gomendatua):");
        System.err.println("    ... Partiketak 20000 1000 false 2 false false 2");
        System.err.println("  N-grama + trigrama:");
        System.err.println("    ... Partiketak 20000 1000 false 1 false false 3");
    }
}