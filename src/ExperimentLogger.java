import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Klase honek esperimentuen erregistro automatikoa kudeatzen du,
 * exekuzio bakoitzeko informazio garrantzitsua fitxategi batean gordez.
 *
 * <p>
 * Erregistroak honako informazioa jasotzen du:
 * </p>
 * <ul>
 * <li>Data eta ordua (timestamp)</li>
 * <li>Sistemaren informazioa (OS, Java bertsioa)</li>
 * <li>Prozesadorearen informazioa eta nukleo kopurua</li>
 * <li>Memoriaren erabilera (RAM)</li>
 * <li>Exekuzio fasea (pipeline-ko pausoa)</li>
 * <li>Erabilitako parametroak</li>
 * <li>Lortutako emaitzak</li>
 * </ul>
 *
 * <p>
 * Informazio hau {@code registro_experimentos.txt} fitxategian
 * gordetzen da, esperimentuen trazabilitatea eta erreprodukzioa
 * errazteko.
 * </p>
 *
 * <p>
 * Klase hau oso erabilgarria da machine learning pipeline-etan,
 * konfigurazio ezberdinen arteko konparaketak egiteko.
 * </p>
 *
 * @version 1.0
 */
public class ExperimentLogger {

    private static final String LOG_FILE = "registro_experimentos.txt";

    /**
     * Esperimentu baten informazioa erregistratzen du log fitxategian.
     *
     * <p>
     * Metodo honek automatikoki gehitzen du sistemaren egoera eta
     * exekuzioaren testuingurua, parametroekin eta emaitzekin batera.
     * </p>
     *
     * @param fasea       Pipeline-aren fasea (adibidez: "Datuen karga",
     *                    "Ebaluazioa")
     * @param parametroak Erabilitako parametroen deskribapena
     * @param emaitzak    Lortutako emaitzak (metrikak edo bestelakoak)
     */
    public static void log(String fasea, String parametroak, String emaitzak) {
        try (FileWriter fw = new FileWriter(LOG_FILE, true);
                PrintWriter pw = new PrintWriter(fw)) {

            LocalDateTime now = LocalDateTime.now();
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            String timestamp = now.format(formatter);

            // Sistema eragilearen eta Java ingurunearen informazioa jasotzen da
            String osName = System.getProperty("os.name");
            String osArch = System.getProperty("os.arch");
            String javaVersion = System.getProperty("java.version");

            int cores = Runtime.getRuntime().availableProcessors();
            String cpuModel = System.getenv("PROCESSOR_IDENTIFIER");
            if (cpuModel == null) {
                cpuModel = "Ezezaguna (Ez da Windows)";
            }

            // Prozesadorearen informazioa eta nukleo kopurua eskuratzen dira
            Runtime runtime = Runtime.getRuntime();
            runtime.gc();
            long memoriaErabiliaBytes = runtime.totalMemory() - runtime.freeMemory();
            long memoriaErabiliaMB = memoriaErabiliaBytes / (1024 * 1024);

            // Informazio guztia log fitxategian formatu egituratuan idazten da
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