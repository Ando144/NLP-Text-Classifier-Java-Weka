import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.converters.ArffSaver;
import java.io.File;

/**
 * Klase honek SPAM eta HAM karpetetan egituratutako direktorio bat irakurtzen du 
 * eta mezu elektroniko guztiak ARFF formatuko datu-multzo (dataset) bakar batean bihurtzen ditu.
 * * HELBURUAK:
 * - Testu libreko .txt fitxategiak kargatzea klaseen arabera multzokatuta.
 * - Prozesamendurako beharrezkoa den hasierako ARFF fitxategi gordina (raw) sortzea.
 * * AURREBALDINTZAK:
 * - Jatorrizko karpetak existitu behar du eta zehazki bi azpikarpeta eduki behar ditu, 
 * klaseen izenekin (adibidez, "SPAM", "HAM").
 * * ONDORENGO BALDINTZAK:
 * - ARFF fitxategi bat sortuko da testuaren atributuarekin eta klase nominalarekin.
 * * EXEKUZIO ADIBIDEA:
 * java -cp "lib/weka.jar:bin" EmailLoader dataset_correos emails_raw.arff
 * * @author WekaProyecto2026 Taldea
 */
public class EmailLoader {

    /**
     * Datuak kargatu eta bihurtzeko prozesua exekutatzen duen metodo nagusia.
     * @param args Terminaleko argumentuak: [0] jatorrizko_bidea, [1] helburuko_arff_bidea
     */
    public static void main(String[] args) {
        // ================================================
        // 1. PARAMETROEN KUDEAKETA DINAMIKOA
        // ================================================
        if (args.length > 2) {
            System.err.println("Erabilera: java -cp \"lib/weka.jar:bin\" EmailLoader [jatorrizko_bidea] [helburuko_arff_bidea]");
            System.err.println("Jatorrizko bideak bi azpikarpeta izan behar ditu: 'SPAM' eta 'HAM'");
            return;
        }

        // Balio lehenetsiak argumenturik ez badago (Eclipse/IntelliJ-n exekutatzeko erraza)
        String inputPath = args.length >= 1 ? args[0] : "DatuakRaw";
        String outputPath = args.length == 2 ? args[1] : "emails_raw.arff";

        System.out.println("==================================================");
        System.out.println("KARGATZE KONFIGURAZIOA:");
        System.out.println("Jatorrizko direktorioa : " + inputPath);
        System.out.println("Helburuko fitxategia   : " + outputPath);
        System.out.println("==================================================\n");

        try {
            // ================================================
            // 2. DIREKTORIOEN EGIAZTAPENA ETA KARGA
            // ================================================
            TextDirectoryLoader loader = new TextDirectoryLoader();
            
            File sourceDirectory = new File(inputPath);
            if (!sourceDirectory.exists() || !sourceDirectory.isDirectory()) {
                System.err.println("ERROREA: Jatorrizko direktorioa ez da existitzen -> " + sourceDirectory.getAbsolutePath());
                return;
            }
            
            loader.setDirectory(sourceDirectory);
            System.out.println("Direktorioak eskaneatzen eta mezu elektronikoak kargatzen... Honek segundo batzuk har ditzake.");
            
            Instances dataRaw = loader.getDataSet();
            
            // Weka-k klasearen atributua zein den jakin dezan ziurtatu (Index 0 edo azkena)
            if (dataRaw.classIndex() == -1) {
                dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
            }
            
            // ================================================
            // 3. ARFF FITXATEGIA GORDE
            // ================================================
            ArffSaver saver = new ArffSaver();
            saver.setInstances(dataRaw);
            
            File outputArff = new File(outputPath);
            // Karpeta ez bada existitzen, sortu egingo dugu erroreak ekiditeko
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