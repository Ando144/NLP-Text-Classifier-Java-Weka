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
     * Datu-multzo bat ARFF formatuko fitxategi batean gordetzen du.
     *
     * <p>
     * Metodo honek, existitzen ez badira, irteerako fitxategiaren direktorioak
     * sortzen ditu, eta Weka-ko {@link ArffSaver} erabiliz datuak idazten
     * ditu.
     * </p>
     *
     * @param data       Gorde nahi den Instances objektua
     * @param outputFile Irteerako fitxategiaren File objektua
     * @throws Exception Fitxategia idaztean errore bat gertatzen bada
     */
    public static void saveToArff(Instances data, File outputFile) throws Exception {
        if (outputFile.getParentFile() != null) {
            outputFile.getParentFile().mkdirs();
        }
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(outputFile);
        saver.writeBatch();
    }

    /**
     * Datu-multzo baten klase-atributua azken indizean dagoela ziurtatzen du.
     *
     * <p>
     * Weka-k klase-atributua esplizituki ezarrita izatea behar du prozesu
     * gehienetarako. Metodo honek ezarrita ez badago soilik aldatzen du,
     * klasea azken atributua dela ezarriz.
     * </p>
     *
     * @param data Prozesatu nahi den datu-multzoa
     */
    public static void ensureClassIndex(Instances data) {
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
    }

    /**
     * Atributu guztien izenak garbitzen ditu, Weka-k ARFF fitxategiak
     * kargatzean erroreak eman ditzaketen kontrol-karaktereak eta
     * karaktere bereziak ezabatuz.
     *
     * @param data Garbitu nahi den datu-multzoa
     */
    public static void sanitizeAttributeNames(Instances data) {
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            String oldName = data.attribute(i).name();
            String sanitizedName = oldName.replaceAll("[^a-zA-Z0-9_.-]", "_");
            if (!oldName.equals(sanitizedName)) {
                data.renameAttribute(i, sanitizedName);
            }
        }
    }

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
            ensureClassIndex(dataRaw);

            // Sortutako dataset-a ARFF formatuan gordetzen da
            saveToArff(dataRaw, new File(outputPath));

            System.out.println("==================================================");
            System.out.println("ARRAKASTA! Korreoen datu-multzoa ondo sortu da.");
            System.out.println("Prozesatutako mezuak guztira: " + dataRaw.numInstances());
            System.out.println("Sortutako atributuak: " + dataRaw.numAttributes() + " (Klasea eta Testua)");
            System.out.println("Fitxategia hemen gorde da: " + new File(outputPath).getAbsolutePath());
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