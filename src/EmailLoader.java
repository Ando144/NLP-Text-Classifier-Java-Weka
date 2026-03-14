import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.converters.ArffSaver;
import java.io.File;

/**
 * Clase encargada de leer un directorio estructurado por carpetas (SPAM y HAM)
 * y convertir todos los correos electrónicos a un único dataset en formato ARFF.
 *
 */
public class EmailLoader {

    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Uso: java EmailLoader <ruta_base>");
            System.err.println("La ruta base debe contener dos subcarpetas: 'SPAM' y 'HAM'");
            return;
        }
        String path = args[0];

        try {
            // 1. Instanciar el cargador de directorios de texto de Weka
            TextDirectoryLoader loader = new TextDirectoryLoader();
            
            // La carpeta padre debe contener exactamente dos subcarpetas: "SPAM" y "HAM"
            File sourceDirectory = new File(path);
            loader.setDirectory(sourceDirectory);
            
            System.out.println("Escaneando directorios y cargando correos electrónicos... Esto puede tardar unos segundos.");
            
            // 2. Construir el objeto Instances (Dataset crudo)
            Instances dataRaw = loader.getDataSet();
            
            // 3. Configurar el guardado del archivo ARFF
            ArffSaver saver = new ArffSaver();
            saver.setInstances(dataRaw);
            
            // Definir la ruta de salida de nuestro archivo crudo
            File outputArff = new File(path + "\\emails_raw.arff");
            saver.setFile(outputArff);
            saver.writeBatch();
            
            System.out.println("==================================================");
            System.out.println("¡ÉXITO! Dataset de correos creado correctamente.");
            System.out.println("Total de correos procesados: " + dataRaw.numInstances());
            System.out.println("Atributos creados: " + dataRaw.numAttributes() + " (Clase y Texto)");
            System.out.println("Archivo guardado en: " + outputArff.getAbsolutePath());
            System.out.println("==================================================");

        } catch (Exception e) {
            System.err.println("Error crítico durante la carga del corpus de correos.");
            e.printStackTrace();
        }
    }
}