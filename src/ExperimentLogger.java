import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Clase utilitaria para registrar automáticamente los experimentos, parámetros
 * y resultados en un archivo de texto. Para ayudar con la trazabilidad.
 * 
 */
public class ExperimentLogger {
    
    // Archivo donde se guardará todo el historial
private static final String LOG_FILE = "emaitzak/registro_experimentos.txt";
    /**
     * Escribe una nueva entrada en el log del proyecto.
     * @param fase El nombre del script o fase (ej. "VECTORIZACIÓN", "MLP FINE-TUNING")
     * @param parametros Los parámetros usados (ej. "WordsToKeep: 1000, InfoGain: 500")
     * @param resultados Los resultados obtenidos o tiempos (ej. "F-Measure: 0.98, Tiempo: 4.5s")
     */
    public static void log(String fase, String parametros, String resultados) {
        try (FileWriter fw = new FileWriter(LOG_FILE, true); // El 'true' hace que se añada al final sin borrar lo anterior
             PrintWriter pw = new PrintWriter(fw)) {
            
            // Obtener la fecha y hora actual
            LocalDateTime now = LocalDateTime.now();
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
            String timestamp = now.format(formatter);
            
            // Escribir en el archivo con un formato limpio y tabular
            pw.println("==================================================");
            pw.println("FECHA      : " + timestamp);
            pw.println("FASE       : " + fase);
            pw.println("PARÁMETROS : " + parametros);
            pw.println("RESULTADOS : " + resultados);
            pw.println("==================================================\n");
            
        } catch (IOException e) {
            System.err.println("Error al escribir en el log de experimentos: " + e.getMessage());
        }
    }
}