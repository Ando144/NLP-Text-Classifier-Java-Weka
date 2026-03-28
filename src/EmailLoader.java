import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.converters.ArffSaver;
import java.io.File;

/**
 * Klase honek mezu elektronikoen testu-fitxategiak kargatzen ditu
 * eta Weka-k erabil dezakeen ARFF formatura bihurtzen ditu.
 *
 * 
 * Weka-ko {@code TextDirectoryLoader} erabiliz, direktorio-egitura
 * batean dauden testu-fitxategiak automatikoki irakurtzen dira, non
 * karpeta bakoitza klase bati dagokion (adibidez: spam / legitimoa).
 * 
 *
 * 
 * Sortutako dataset-ak bi atributu nagusi ditu:
 * <ul>
 * <li>Testua (emailaren edukia)</li>
 * <li>Klasea (spam edo legitimoa)</li>
 * </ul>
 * 
 *
 * 
 * Azken emaitza ARFF fitxategi batean gordetzen da, ondorengo
 * prozesamenduetarako (behin betiko pipeline-an erabiltzeko).
 * 
 *
 * 
 * Klase hau emailen sailkapen sistemaren lehen urratsa da.
 * 
 *
 * @version 1.0
 */
public class EmailLoader {
    /**
     * Programa exekutatzen duen metodo nagusia, testu-fitxategiak kargatu
     * eta ARFF dataset bihurtzen dituena.
     *
     * <p>
     * Sarrerako direktorioa eta irteerako fitxategia argumentuen bidez
     * zehaztu daitezke. Bestela, balio lehenetsiak erabiliko dira.
     * </p>
     *
     * @param args Komando lerroko argumentuak:
     *             <ul>
     *             <li>args[0] - Jatorrizko direktorioa (email testuak)</li>
     *             <li>args[1] - Irteerako ARFF fitxategia</li>
     *             </ul>
     */
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
            // Weka-ko TextDirectoryLoader erabiliz direktorio egitura kargatzen da
            TextDirectoryLoader loader = new TextDirectoryLoader();

            // Ziurtatzen da sarrerako direktorioa existitzen dela eta baliozkoa dela
            File sourceDirectory = new File(inputPath);
            if (!sourceDirectory.exists() || !sourceDirectory.isDirectory()) {
                System.err.println(
                        "ERROREA: Jatorrizko direktorioa ez da existitzen -> " + sourceDirectory.getAbsolutePath());
                return;
            }

            loader.setDirectory(sourceDirectory);
            System.out.println(
                    "Direktorioak eskaneatzen eta mezu elektronikoak kargatzen... Honek segundo batzuk har ditzake.");

            // Testu-fitxategiak Weka Instances objektu bihurtzen dira
            Instances dataRaw = loader.getDataSet();

            // Klase atributua azken atributu gisa ezartzen da (spam/legitimoa)
            if (dataRaw.classIndex() == -1) {
                dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
            }

            // Sortutako dataset-a ARFF formatuan gordetzen da
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