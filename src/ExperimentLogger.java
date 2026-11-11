import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Manages the automatic logging of experiments, saving the
 * relevant information of each run to a file.
 *
 *
 * Each log entry contains the following information:
 *
 * <ul>
 * <li>Date and time (timestamp)</li>
 * <li>System information (OS, Java version)</li>
 * <li>Processor information and number of cores</li>
 * <li>Memory used by the JVM (RAM)</li>
 * <li>Execution phase (pipeline step)</li>
 * <li>Parameters used</li>
 * <li>Results obtained</li>
 * </ul>
 *
 *
 * This information is saved in the {@code experiment_log.txt} file,
 * to make the experiments easier to trace and to
 * reproduce.
 *
 *
 *
 * This class is very useful in machine learning pipelines,
 * to compare different configurations.
 *
 *
 * @version 1.0
 */
public class ExperimentLogger {

    private static final String LOG_FILE = "experiment_log.txt";

    /**
     * Logs the information of one experiment in the log file.
     *
     * <p>
     * This method automatically adds the system state and the
     * run context, together with the parameters and the results.
     * </p>
     *
     * @param phase      Pipeline phase (for example: "Data loading",
     *                   "Evaluation")
     * @param parameters Description of the parameters used
     * @param results    Results obtained (metrics or others)
     */
    public static void log(String phase, String parameters, String results) {
        try (FileWriter fw = new FileWriter(LOG_FILE, true);
                PrintWriter pw = new PrintWriter(fw)) {

            LocalDateTime now = LocalDateTime.now();
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            String timestamp = now.format(formatter);

            // Information about the operating system and the Java environment
            String osName = System.getProperty("os.name");
            String osArch = System.getProperty("os.arch");
            String javaVersion = System.getProperty("java.version");

            // Processor information and number of cores
            int cores = Runtime.getRuntime().availableProcessors();
            String cpuModel = System.getenv("PROCESSOR_IDENTIFIER");
            if (cpuModel == null) {
                cpuModel = "Unknown (not Windows)";
            }

            // Memory currently used by the JVM (after asking for a garbage collection)
            Runtime runtime = Runtime.getRuntime();
            runtime.gc();
            long usedMemoryBytes = runtime.totalMemory() - runtime.freeMemory();
            long usedMemoryMB = usedMemoryBytes / (1024 * 1024);

            // All the information is written to the log file in a structured format
            pw.println("=========================================================================");
            pw.println("DATE AND TIME   : " + timestamp);
            pw.println("SYSTEM          : " + osName + " (" + osArch + ") | Java: " + javaVersion);
            pw.println("PROCESSOR       : " + cpuModel + " (" + cores + " cores/threads)");
            pw.println("MEMORY (RAM)    : " + usedMemoryMB + " MB used by the JVM");
            pw.println("-------------------------------------------------------------------------");
            pw.println("PHASE           : " + phase);
            pw.println("PARAMETERS      : " + parameters);
            pw.println("RESULTS         : \n" + results);
            pw.println("=========================================================================\n");

        } catch (IOException e) {
            System.err.println("Error while writing the experiment log: " + e.getMessage());
        }
    }
}
