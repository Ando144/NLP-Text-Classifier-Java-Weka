import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Klase honek esperimentuen erregistro automatikoa (Logger) kudeatzen du.
 * Parametroak eta emaitzak gordetzeaz gain, sistemaren informazioa, memoriaren kontsumoa
 * eta prozesadorearen (CPU) datuak automatikoki harrapatzen ditu erreproduzibilidadea bermatzeko.
 * * @author WekaProyecto2026 Taldea
 */
public class ExperimentLogger {
    
    private static final String LOG_FILE = "registro_experimentos.txt";

    /**
     * Esperimentu baten sarrera berri bat idazten du erregistroan.
     * @param fasea Exekutatutako prozesua (adib. "MLP Fine-Tuning")
     * @param parametroak Erabilitako ezarpenak
     * @param emaitzak Lortutako metrikak edo denborak
     */
    public static void log(String fasea, String parametroak, String emaitzak) {
        try (FileWriter fw = new FileWriter(LOG_FILE, true);
             PrintWriter pw = new PrintWriter(fw)) {
            
            // 1. Data eta Ordua
            LocalDateTime now = LocalDateTime.now();
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            String timestamp = now.format(formatter);
            
            // 2. Sistemaren Informazioa eta CPU-a (Erreproduzibilidadea - RQ4)
            String osName = System.getProperty("os.name");
            String osArch = System.getProperty("os.arch");
            String javaVersion = System.getProperty("java.version");
            
            // CPU nukleoak (Cores) eta modeloa (Windows sistemetan)
            int cores = Runtime.getRuntime().availableProcessors();
            String cpuModel = System.getenv("PROCESSOR_IDENTIFIER");
            if (cpuModel == null) {
                cpuModel = "Ezezaguna (Ez da Windows)";
            }
            
            // 3. Memoriaren Kontsumoa (Koste Konputazionala - RQ2)
            Runtime runtime = Runtime.getRuntime();
            runtime.gc(); // Garbiketa azkarra memoria zehatzagoa izateko
            long memoriaErabiliaBytes = runtime.totalMemory() - runtime.freeMemory();
            long memoriaErabiliaMB = memoriaErabiliaBytes / (1024 * 1024);
            
            // 4. Fitxategian idatzi formatu profesional eta garbi batekin
            pw.println("=========================================================================");
            pw.println("DATA ETA ORDUA  : " + timestamp);
            pw.println("SISTEMA         : " + osName + " (" + osArch + ") | Java: " + javaVersion);
            pw.println("PROZESADOREA    : " + cpuModel + " (" + cores + " nukleo/hari)");
            pw.println("MEMORIA (RAM)   : " + memoriaErabiliaMB + " MB erabilita sare neuronalarako");
            pw.println("-------------------------------------------------------------------------");
            pw.println("FASEA           : " + fasea);
            pw.println("PARAMETROAK     : " + parametroak);
            pw.println("EMAITZAK        : \n" + emaitzak);
            pw.println("=========================================================================\n");
            
        } catch (IOException e) {
            System.err.println("Errorea esperimentuen erregistroan idaztean: " + e.getMessage());
        }
    }
}