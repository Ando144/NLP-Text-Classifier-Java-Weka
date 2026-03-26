import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Esperimentuen erregistro automatikoa.
 */
public class ExperimentLogger {
    
    private static final String LOG_FILE = "registro_experimentos.txt";

    public static void log(String fasea, String parametroak, String emaitzak) {
        try (FileWriter fw = new FileWriter(LOG_FILE, true);
             PrintWriter pw = new PrintWriter(fw)) {
            
            LocalDateTime now = LocalDateTime.now();
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            String timestamp = now.format(formatter);
            
            // Sistemaren informazioa
            String osName = System.getProperty("os.name");
            String osArch = System.getProperty("os.arch");
            String javaVersion = System.getProperty("java.version");
            
            int cores = Runtime.getRuntime().availableProcessors();
            String cpuModel = System.getenv("PROCESSOR_IDENTIFIER");
            if (cpuModel == null) {
                cpuModel = "Ezezaguna (Ez da Windows)";
            }
            
            // Memoriaren kontsumoa
            Runtime runtime = Runtime.getRuntime();
            runtime.gc();
            long memoriaErabiliaBytes = runtime.totalMemory() - runtime.freeMemory();
            long memoriaErabiliaMB = memoriaErabiliaBytes / (1024 * 1024);
            
            // Fitxategian idatzi
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