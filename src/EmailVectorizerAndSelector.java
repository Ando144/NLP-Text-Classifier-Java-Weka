import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.stemmers.LovinsStemmer;
// Cambiamos el WordTokenizer por AlphabeticTokenizer para limpiar números y basura
import weka.core.tokenizers.AlphabeticTokenizer; 
import weka.core.stopwords.Rainbow;
import weka.core.converters.ArffSaver;
import java.io.File;

/**
 * Klase honek mezu elektronikoen testuak bektorizatzen ditu (StringToWordVector) 
 * eta, ondoren, ezaugarri onenak aukeratzen ditu (AttributeSelection - InfoGain) 
 * dimentsionalitatea murrizteko.
 * * HELBURUAK:
 * - Testu gordinak zenbakizko bektore bihurtzea TF-IDF, LovinsStemmer eta Rainbow stopwords erabiliz.
 * - InfoGainAttributeEval erabiliz informazio gehien ematen duten atributuak (hitzak) iragaztea.
 * - Esperimentazioa erraztea parametro dinamikoen bidez (Experiment Tracking).
 * * AURREBALDINTZAK:
 * - 'train.arff', 'dev.arff' eta 'test.arff' fitxategiak zehaztutako karpetan egon behar dira.
 * * ONDORENGO BALDINTZAK:
 * - 'train_final.arff', 'dev_final.arff' eta 'test_final.arff' fitxategiak sortuko dira 
 * hiztegi optimizatuarekin (atributu kopuru murriztuarekin).
 * * EXEKUZIO ADIBIDEA:
 * java -cp "lib/weka.jar:bin" EmailVectorizerAndSelector Partiketak 20000 700
 * * @author WekaProyecto2026 Taldea
 */
public class EmailVectorizerAndSelector {
    public static void main(String[] args) {
        try {
            // ================================================
            // 1. PARAMETROEN KUDEAKETA DINAMIKOA (Aldagaiak)
            // ================================================
            if (args.length > 3) {
                System.err.println("Erabilera: java -cp \"lib/weka.jar:bin\" EmailVectorizerAndSelector [karpeta] [wordsToKeep] [numToSelect]");
                return;
            }

            // Argumenturik pasatzen ez bada, balio lehenetsiak erabiliko dira
            String baseDir = args.length >= 1 ? args[0] : "Partiketak";
            int wordsToKeep = args.length >= 2 ? Integer.parseInt(args[1]) : 20000;
            // CAMBIO CLAVE: Bajamos por defecto a 700 atributos para cumplir con la profesora y limpiar ruido
            int numToSelect = args.length == 3 ? Integer.parseInt(args[2]) : 700; 

            System.out.println("==================================================");
            System.out.println("ESPERIMENTUAREN KONFIGURAZIOA:");
            System.out.println("Lan-direktorioa       : " + baseDir);
            System.out.println("Gordetzeko hitzak max : " + wordsToKeep);
            System.out.println("InfoGain aukeraketa   : " + numToSelect + " atributu");
            System.out.println("==================================================\n");

            File trainRawFile = new File(baseDir, "train.arff");
            File devRawFile = new File(baseDir, "dev.arff");
            File testRawFile = new File(baseDir, "test.arff");

            if (!trainRawFile.exists() || !devRawFile.exists() || !testRawFile.exists()) {
                System.err.println("Sarrerako fitxategiak falta dira hemen: " + new File(baseDir).getAbsolutePath());
                System.err.println("Hauek espero dira: train.arff, dev.arff eta test.arff");
                return;
            }

            // ================================================
            // 2. DATU-MULTZOAK KARGATU
            // ================================================
            Instances train = new DataSource(trainRawFile.getPath()).getDataSet();
            Instances dev = new DataSource(devRawFile.getPath()).getDataSet();
            Instances test = new DataSource(testRawFile.getPath()).getDataSet();

            train.setClassIndex(train.numAttributes() - 1);
            dev.setClassIndex(dev.numAttributes() - 1);
            test.setClassIndex(test.numAttributes() - 1);

            // ================================================
            // 3. StringToWordVector (Bektorizazioa eta NLP)
            // ================================================
            System.out.println("StringToWordVector aplikatzen...");
            StringToWordVector stwv = new StringToWordVector();

            // Transformazio matematikoak
            stwv.setIDFTransform(true);
            stwv.setTFTransform(true);
            stwv.setLowerCaseTokens(true);
            
            // ALDAGAIA HEMEN APLIKATZEN DUGU
            stwv.setWordsToKeep(wordsToKeep);

            // OPTIMIZAZIOA: AlphabeticTokenizer-ek bakarrik letrak onartzen ditu (A-Z).
            // Zenbakiak, ikurrak eta HTML etiketak automatikoki garbitzen dira.
            AlphabeticTokenizer tokenizer = new AlphabeticTokenizer();
            stwv.setTokenizer(tokenizer);

            // NLP Tresnak
            LovinsStemmer stemmer = new LovinsStemmer();
            stwv.setStemmer(stemmer);
            stwv.setStopwordsHandler(new Rainbow());
            stwv.setInputFormat(train);

            Instances trainVec = Filter.useFilter(train, stwv);
            Instances devVec = Filter.useFilter(dev, stwv);
            Instances testVec = Filter.useFilter(test, stwv);

            System.out.println("Atributuak bektorizazioaren ostean: " + trainVec.numAttributes());

            // ================================================
            // 4. ATRIBUTUEN AUKERAKETA (InfoGain)
            // ================================================
            System.out.println("Atributuen Aukeraketa (InfoGain) aplikatzen...");
            AttributeSelection filterSelector = new AttributeSelection();
            InfoGainAttributeEval eval = new InfoGainAttributeEval();

            Ranker search = new Ranker();
            // ALDAGAIA HEMEN APLIKATZEN DUGU (Orain 700 lehenetsita)
            search.setNumToSelect(numToSelect);

            filterSelector.setEvaluator(eval);
            filterSelector.setSearch(search);
            filterSelector.setInputFormat(trainVec);

            Instances trainFinal = Filter.useFilter(trainVec, filterSelector);
            Instances devFinal = Filter.useFilter(devVec, filterSelector);
            Instances testFinal = Filter.useFilter(testVec, filterSelector);

            // ================================================
            // 5. AMAIERAKO DATU-MULTZOAK GORDE
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

            System.out.println("==================================================");
            System.out.println("PROZESUA AMAITUTA!");
            System.out.println("Amaierako atributuak: " + trainFinal.numAttributes());
            System.out.println("==================================================");

            // ================================================
            // 6. ERREGISTRO AUTOMATIKOA (Experiment Tracking)
            // ================================================
            String parametrosUsados = String.format("WordsToKeep: %d | InfoGain(NumToSelect): %d | Tokenizer: Alphabetic | Stemmer: Lovins | Stopwords: Rainbow", wordsToKeep, numToSelect);
            String resultadosObtenidos = "Amaierako hiztegiaren atributuak: " + trainFinal.numAttributes();
            ExperimentLogger.log("Bektorizazioa eta Atributuen Aukeraketa", parametrosUsados, resultadosObtenidos);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}