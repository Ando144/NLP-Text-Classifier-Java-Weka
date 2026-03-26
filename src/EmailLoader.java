import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.converters.ArffSaver;
import java.io.File;

/**
 * Datu gordinak ARFF formatura bihurtu.
 */
public class EmailLoader {
    public static void main(String[] args) {
        if (args.length > 2) {
            System.err.println("Erabilera: java -cp \"lib/weka.jar:bin\" EmailLoader [jatorrizkoa] [helburua]");
            return;
        }

        String inputPath = args.length >= 1 ? args[0] : "DatuakRaw";
        String outputPath = args.length == 2 ? args[1] : "emails_raw.arff";

        System.out.println("==================================================");
        System.out.println("KARGATZE KONFIGURAZIOA:");
        System.out.println("Jatorrizko direktorioa : " + inputPath);
        System.out.println("Helburuko fitxategia   : " + outputPath);
        System.out.println("==================================================\n");

        try {
            // Direktorioak kargatu
            TextDirectoryLoader loader = new TextDirectoryLoader();
            
            File sourceDirectory = new File(inputPath);
            if (!sourceDirectory.exists() || !sourceDirectory.isDirectory()) {
                System.err.println("ERROREA: Jatorrizko direktorioa ez da existitzen -> " + sourceDirectory.getAbsolutePath());
                return;
            }
            
            loader.setDirectory(sourceDirectory);
            System.out.println("Direktorioak eskaneatzen eta mezu elektronikoak kargatzen... Honek segundo batzuk har ditzake.");
            
            Instances dataRaw = loader.getDataSet();
            
            if (dataRaw.classIndex() == -1) {
                dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
            }
            
            // ARFF Gorde
            ArffSaver saver = new ArffSaver();
            saver.setInstances(dataRaw);
            
            File outputArff = new File(outputPath);
            if (outputArff.getParentFile() != null) {
                outputArff.getParentFile().mkdirs();
            }
            
            saver.setFile(outputArff);
            saver.writeBatch();
            
            System.out.println("==================================================");
            System.out.println("ARRAKASTA! Korreoen datu-multzoa ondo sortu da.");
            System.out.println("Prozesatutako mezuak guztira: " + dataRaw.numInstances());
            System.out.println("Sortutako atributuak: " + dataRaw.numAttributes() + " (Klasea eta Testua)");
            System.out.println("Fitxategia hemen gorde da: " + outputArff.getAbsolutePath());
            System.out.println("==================================================");

            // ================================================
            // 4. ERREGISTRO AUTOMATIKOA (Experiment Tracking)
            // ================================================
            String parametrosUsados = "Jatorria: " + inputPath + " | Helburua: " + outputPath;
            String resultadosObtenidos = "Kargatutako instantziak: " + dataRaw.numInstances();
            ExperimentLogger.log("1. Datu Gordinen Karga (EmailLoader)", parametrosUsados, resultadosObtenidos);

        } catch (Exception e) {
            System.err.println("Errore kritikoa mezu elektronikoen korpusa kargatzean.");
            e.printStackTrace();
        }
    }
}