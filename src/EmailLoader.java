import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.SortLabels;
import java.io.File;
import java.util.Comparator;

/**
 * Loads the text files of the emails and converts them
 * to the ARFF format that Weka can use.
 *
 *
 * Using Weka's {@code TextDirectoryLoader}, the text files of a
 * directory structure are read automatically, where each
 * folder corresponds to one class (for example: spam / legitimate).
 *
 *
 *
 * The generated dataset has two main attributes:
 * <ul>
 * <li>Text (the content of the email)</li>
 * <li>Class (spam or legitimate)</li>
 * </ul>
 *
 *
 *
 * The final result is saved in an ARFF file for the following
 * processing steps (to be used in the final pipeline).
 *
 *
 *
 * This class is the first step of the email classification system.
 *
 * <p>
 * The result does not depend on the operating system: the files are read as
 * UTF-8, the class labels are sorted alphabetically and the emails are sorted
 * by class and file name (Weka's loader alone uses the order in which the
 * file system lists the folders, and the default charset of the platform).
 * </p>
 *
 * @version 1.0
 */
public class EmailLoader {
    /**
     * Saves a dataset to an ARFF file.
     *
     * <p>
     * If they do not exist, this method creates the directories of the
     * output file, and writes the data using Weka's {@link ArffSaver}.
     * </p>
     *
     * @param data       Instances object to save
     * @param outputFile File object of the output file
     * @throws Exception If an error happens while writing the file
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
     * Makes sure that the class attribute of a dataset is the last attribute.
     *
     * <p>
     * Weka needs the class attribute to be set explicitly for most
     * processes. This method only changes it if it is not set,
     * making the last attribute the class.
     * </p>
     *
     * @param data Dataset to process
     */
    public static void ensureClassIndex(Instances data) {
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
    }

    /**
     * Cleans the names of all attributes, removing the control characters
     * and special characters that can cause errors when Weka loads
     * ARFF files.
     *
     * @param data Dataset to clean
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
     * Puts a dataset loaded with {@code TextDirectoryLoader} (with the file
     * name attribute enabled) in a fixed order, so that the result is the same
     * on every operating system.
     *
     * <ul>
     * <li>The labels of the class attribute are sorted alphabetically.</li>
     * <li>The emails are sorted by class label and then by file name.</li>
     * <li>The file name attribute is removed.</li>
     * <li>The relation name is the name of the source folder (instead of the
     * absolute path on the computer).</li>
     * </ul>
     *
     * @param data         Dataset with the attributes text, filename and class
     * @param relationName Relation name of the result
     * @return The dataset in a fixed order, with the attributes text and class
     * @throws Exception If a Weka filter fails
     */
    public static Instances makeDeterministic(Instances data, String relationName) throws Exception {
        ensureClassIndex(data);

        SortLabels sortLabels = new SortLabels();
        sortLabels.setAttributeIndices(String.valueOf(data.classIndex() + 1));
        sortLabels.setInputFormat(data);
        Instances sorted = Filter.useFilter(data, sortLabels);
        sorted.setClassIndex(data.classIndex());

        final int classIndex = sorted.classIndex();
        final int fileIndex = sorted.attribute("filename").index();
        sorted.sort(Comparator
                .comparing((Instance inst) -> inst.stringValue(classIndex))
                .thenComparing(inst -> inst.stringValue(fileIndex)));

        Remove remove = new Remove();
        remove.setAttributeIndices(String.valueOf(fileIndex + 1));
        remove.setInputFormat(sorted);
        Instances result = Filter.useFilter(sorted, remove);
        result.setClassIndex(result.numAttributes() - 1);
        result.setRelationName(relationName);
        return result;
    }

    /**
     * Main method: loads the text files and converts them
     * into an ARFF dataset.
     *
     * <p>
     * The input directory and the output file can be given as
     * arguments. Otherwise, default values are used.
     * </p>
     *
     * @param args Command-line arguments:
     *             <ul>
     *             <li>args[0] - Source directory (email texts)</li>
     *             <li>args[1] - Output ARFF file</li>
     *             </ul>
     */
    public static void main(String[] args) {

        if (args.length > 2) {
            System.err.println("Usage: java -cp \"lib/weka.jar:bin\" EmailLoader [source] [target]");
            return;
        }

        String inputPath = args.length >= 1 ? args[0] : "data/raw";
        String outputPath = args.length == 2 ? args[1] : "emails_raw.arff";

        System.out.println("==================================================");
        System.out.println("LOADING CONFIGURATION:");
        System.out.println("Source directory : " + inputPath);
        System.out.println("Target file      : " + outputPath);
        System.out.println("==================================================\n");

        try {
            // The directory structure is loaded with Weka's TextDirectoryLoader
            TextDirectoryLoader loader = new TextDirectoryLoader();

            // Check that the input directory exists and is valid
            File sourceDirectory = new File(inputPath);
            if (!sourceDirectory.exists() || !sourceDirectory.isDirectory()) {
                System.err.println(
                        "ERROR: The source directory does not exist -> " + sourceDirectory.getAbsolutePath());
                return;
            }

            loader.setDirectory(sourceDirectory);
            // Fix C10: same result on every OS (UTF-8, and the file name is kept
            // to sort the emails; see makeDeterministic)
            loader.setCharSet("UTF-8");
            loader.setOutputFilename(true);
            System.out.println(
                    "Scanning the directories and loading the emails... This can take a few seconds.");

            // The text files are converted into a Weka Instances object
            Instances dataRaw = loader.getDataSet();

            // The class attribute is set as the last attribute (spam/legitimate)
            ensureClassIndex(dataRaw);

            // Fixed order of labels and emails, independent of the file system
            dataRaw = makeDeterministic(dataRaw, sourceDirectory.getName());

            // The generated dataset is saved in ARFF format
            saveToArff(dataRaw, new File(outputPath));

            System.out.println("==================================================");
            System.out.println("SUCCESS! The email dataset was created.");
            System.out.println("Total processed emails: " + dataRaw.numInstances());
            System.out.println("Generated attributes: " + dataRaw.numAttributes() + " (Class and Text)");
            System.out.println("File saved here: " + new File(outputPath).getAbsolutePath());
            System.out.println("==================================================");

            // ================================================
            // 4. AUTOMATIC LOGGING (Experiment Tracking)
            // ================================================
            String usedParams = "Source: " + inputPath + " | Target: " + outputPath;
            String obtainedResults = "Loaded instances: " + dataRaw.numInstances();
            ExperimentLogger.log("1. Raw Data Loading (EmailLoader)", usedParams, obtainedResults);

        } catch (Exception e) {
            System.err.println("Critical error while loading the email corpus.");
            e.printStackTrace();
        }
    }
}
