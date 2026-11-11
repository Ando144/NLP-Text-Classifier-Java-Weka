import weka.filters.Filter;
import java.io.*;

/**
 * Helper class to serialize (save) and deserialize (load)
 * Weka filters.
 *
 *
 * In a machine learning pipeline, the filters used during training
 * (for example: vectorization or attribute selection) must be
 * saved, so that the same transformations can be applied to new
 * data at inference time.
 *
 *
 *
 * This class makes it possible to save {@link weka.filters.Filter}
 * objects to a file and load them again later.
 *
 *
 *
 * Serialization is done with Java's {@code ObjectOutputStream} and
 * {@code ObjectInputStream}.
 *
 *
 * @version 1.0
 */
public class FilterSerializationHelper {

    /**
     * Saves a Weka filter to a file (serialization).
     *
     * <p>
     * This method saves the complete state of the filter, so it can be
     * reused later in the inference phase.
     * </p>
     *
     * @param filter   Filter to save
     * @param filepath Path of the output file
     * @throws IOException If an error happens while writing
     */
    public static void saveFilter(Filter filter, String filepath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filepath))) {
            oos.writeObject(filter);
        }
    }

    /**
     * Loads a Weka filter from a file (deserialization).
     *
     * <p>
     * The loaded filter is the same one used during training,
     * and it is used to apply the same transformations to new data.
     * </p>
     *
     * @param filepath Path of the file to load
     * @return The loaded filter
     * @throws IOException            If an error happens while reading
     * @throws ClassNotFoundException If the class cannot be found
     */
    public static Filter loadFilter(String filepath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filepath))) {
            return (Filter) ois.readObject();
        }
    }
}
