import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.stemmers.LovinsStemmer;
import weka.core.tokenizers.WordTokenizer;
import weka.core.converters.ArffSaver;
import java.io.File;

public class EmailVectorizerAndSelector {
    public static void main(String[] args) {
        try {
            // 1. Cargar datos crudos
            DataSource source = new DataSource("data/emails_raw.arff"); 
            Instances dataRaw = source.getDataSet();
            if (dataRaw.classIndex() == -1) {
                dataRaw.setClassIndex(dataRaw.numAttributes() - 1); 
 
            } else {
                System.out.println("El dataset tiene que tener el indice en el -1");
            }
            //Vectorizamos con StringToWordVector
            StringToWordVector stwv = new StringToWordVector();
            stwv.setInputFormat(dataRaw);
            stwv.setIDFTransform(true);
            stwv.setTFTransform(true);
            stwv.setLowerCaseTokens(true);
            
            WordTokenizer tokenizador = new WordTokenizer();
            tokenizador.setDelimiters(" \r\n\t.,;:'\"()?!-+/\\<>@#$%^&*_=~`|[]{}");
            stwv.setTokenizer(tokenizador);
            
            //el stemmer lo que hace es reducir las palabras a su raíz,
            //  por ejemplo "running" se convierte en "run". 
            // Esto ayuda a filtrar las palabras y que no tarde en ejecutar dos años.

            LovinsStemmer stemmer = new LovinsStemmer();
            stwv.setStemmer(stemmer);
            
            //el WordsToKeep lo mas grande posible 
            stwv.setWordsToKeep(1000000); 

            System.out.println("Aplicando StringToWordVector masivo...");
            Instances dataVectorized = Filter.useFilter(dataRaw, stwv);
            
            System.out.println("Atributos ANTES de la selección: " + dataVectorized.numAttributes());

            //fase de selección de atributos con InfoGain
            AttributeSelection filterSelector = new AttributeSelection();
            
            // Usamos Ganancia de Información (InfoGain)
            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            
            // Usamos Ranker para ordenar las palabras de mejor a peor
            Ranker search = new Ranker();
            
            // aqui elegimos el tamaño del vocabulario final, habra que jugar con este numero para ver cual es el 
            //mejor resultado
            search.setNumToSelect(1000); 
            
            filterSelector.setEvaluator(eval);
            filterSelector.setSearch(search);
            filterSelector.setInputFormat(dataVectorized);
            
            System.out.println("Aplicando Selección de Atributos (InfoGain)...");
            Instances dataFinal = Filter.useFilter(dataVectorized, filterSelector);
            
            // ==========================================
            // FASE 3: GUARDAR EL DATASET FINAL
            // ==========================================
            ArffSaver saver = new ArffSaver();
            saver.setInstances(dataFinal);
            saver.setFile(new File("data/emails_final_bow.arff"));
            saver.writeBatch();

            System.out.println("==================================================");
            System.out.println("¡PROCESO COMPLETADO CON ÉXITO!");
            System.out.println("Vocabulario Final (Atributos): " + dataFinal.numAttributes());
            System.out.println("==================================================");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}