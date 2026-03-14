import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.stemmers.LovinsStemmer;
import weka.core.tokenizers.WordTokenizer;
import weka.core.stopwords.Rainbow;
import weka.core.converters.ArffSaver;
import java.io.File;

public class EmailVectorizerAndSelector {
    public static void main(String[] args) {
        try {
            String baseDir = args.length >= 1 ? args[0] : "Partiketak";
            if (args.length > 1) {
                System.err.println("Uso: java -cp \"lib/weka.jar:bin\" EmailVectorizerAndSelector [carpeta_particiones]");
                return;
            }

            File trainRawFile = new File(baseDir, "train.arff");
            File devRawFile = new File(baseDir, "dev.arff");
            File testRawFile = new File(baseDir, "test.arff");

            if (!trainRawFile.exists() || !devRawFile.exists() || !testRawFile.exists()) {
                System.err.println("Faltan archivos de entrada en: " + new File(baseDir).getAbsolutePath());
                System.err.println("Se esperan: train.arff, dev.arff y test.arff");
                return;
            }

            // 1. Cargar datasets
            Instances train = new DataSource(trainRawFile.getPath()).getDataSet();
            Instances dev = new DataSource(devRawFile.getPath()).getDataSet();
            Instances test = new DataSource(testRawFile.getPath()).getDataSet();

            train.setClassIndex(train.numAttributes() - 1);
            dev.setClassIndex(dev.numAttributes() - 1);
            test.setClassIndex(test.numAttributes() - 1);

            // ================================
            // 2. StringToWordVector
            // ================================

            StringToWordVector stwv = new StringToWordVector();

            stwv.setIDFTransform(true);
            stwv.setTFTransform(true);
            stwv.setLowerCaseTokens(true);
            stwv.setWordsToKeep(20000);

            WordTokenizer tokenizador = new WordTokenizer();
            tokenizador.setDelimiters(" \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}");
            stwv.setTokenizer(tokenizador);

            LovinsStemmer stemmer = new LovinsStemmer();
            stwv.setStemmer(stemmer);

            stwv.setStopwordsHandler(new Rainbow());

            stwv.setInputFormat(train);

            Instances trainVec = Filter.useFilter(train, stwv);
            Instances devVec = Filter.useFilter(dev, stwv);
            Instances testVec = Filter.useFilter(test, stwv);

            System.out.println("Atributos tras vectorización: " + trainVec.numAttributes());

            // ================================
            // 3. Selección de atributos
            // ================================

            AttributeSelection filterSelector = new AttributeSelection();

            InfoGainAttributeEval eval = new InfoGainAttributeEval();

            Ranker search = new Ranker();
            search.setNumToSelect(1000);

            filterSelector.setEvaluator(eval);
            filterSelector.setSearch(search);
            filterSelector.setInputFormat(trainVec);

            Instances trainFinal = Filter.useFilter(trainVec, filterSelector);
            Instances devFinal = Filter.useFilter(devVec, filterSelector);
            Instances testFinal = Filter.useFilter(testVec, filterSelector);

            // ================================
            // 4. Guardar datasets finales
            // ================================

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
            System.out.println("¡PROCESO COMPLETADO!");
            System.out.println("Atributos finales: " + trainFinal.numAttributes());
            System.out.println("==================================================");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}