import weka.filters.Filter;
import java.io.*;

/**
 * Weka filtroak gorde eta kargatzeko laguntzailea.
 */
public class FilterSerializationHelper {

    // Filtroak gorde
    public static void saveFilter(Filter filter, String filepath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filepath))) {
            oos.writeObject(filter);
        }
    }

    // Filtroak kargatu
    public static Filter loadFilter(String filepath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filepath))) {
            return (Filter) ois.readObject();
        }
    }
}
