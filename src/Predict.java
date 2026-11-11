/*
 * =============================================================================
 *  Predict.java - Inference entry point of the spam/ham email classifier
 * =============================================================================
 *  What it does : Loads the trained Multilayer Perceptron (model/mlp.model) and
 *                 the serialized preprocessing filters (TextNormalizer,
 *                 StringToWordVector TF-IDF, InfoGain AttributeSelection), and
 *                 classifies every .txt email of an input folder as SPAM or HAM,
 *                 with the probability of the predicted class.
 *  Platform     : Any OS with a Java 11+ runtime (JVM). No special hardware;
 *                 runs on CPU.
 *  Tools        : Java, Weka 3.8.6 (lib/weka.jar), Bash (pipeline.sh), Javadoc.
 *  Authors      : Ibai Olaziregi, Gorka Piedra, Andoni Ortiz de Zarate
 *  Course       : Decision Support Systems (Erabakiak Hartzeko Euskarri
 *                 Sistemak), 3rd year, Bachelor's Degree in Management and
 *                 Information Systems Engineering, Bilbao School of
 *                 Engineering (UPV/EHU).
 * =============================================================================
 */
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Utils;
import weka.core.converters.TextDirectoryLoader;
import weka.core.converters.ArffSaver;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;

import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * Makes predictions on new emails using a trained
 * classifier (MLP model).
 *
 *
 * It runs the complete pipeline in the inference phase:
 *
 * <ul>
 * <li>Load the emails (from a directory structure or a flat folder)</li>
 * <li>Apply the text normalization</li>
 * <li>Reuse the filters used during training (vectorization and
 * attribute selection)</li>
 * <li>Load the model and make the predictions</li>
 * <li>Save the results to a file</li>
 * </ul>
 *
 *
 * The goal is to guarantee consistency between training and inference.
 *
 *
 * @version 1.0
 */
public class Predict {
    /**
     * Loads the .txt files of a folder and of all its subfolders
     * (recursive search), sorted by path.
     *
     * <p>
     * Each file becomes one instance,
     * with a single text attribute and an unknown class.
     * </p>
     *
     * @param inputDirectory Input folder
     * @return The generated Instances object
     * @throws IOException If a read error happens
     */
    private static Instances loadFlatInferenceData(File inputDirectory) throws IOException {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("text", (List<String>) null));
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("ham");
        classValues.add("spam");
        attributes.add(new Attribute("@@class@@", classValues));

        Instances data = new Instances("inference_emails", attributes, 0);
        data.setClassIndex(data.numAttributes() - 1);

        List<Path> txtFiles = new ArrayList<>();
        // Fix C6: the stream returned by Files.walk is now closed (try-with-resources)
        try (Stream<Path> paths = Files.walk(inputDirectory.toPath())) {
            paths.filter(Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().toLowerCase().endsWith(".txt"))
                    .forEach(txtFiles::add);
        }

        txtFiles.sort(Comparator.comparing(Path::toString));

        for (Path txtFile : txtFiles) {
            // Fix C6: decode as UTF-8 but replace invalid bytes (instead of
            // aborting the whole run when one file is not valid UTF-8)
            String content = new String(Files.readAllBytes(txtFile), StandardCharsets.UTF_8);
            double[] values = new double[data.numAttributes()];
            values[0] = data.attribute(0).addStringValue(content);
            values[data.classIndex()] = Utils.missingValue();
            data.add(new weka.core.DenseInstance(1.0, values));
        }

        return data;
    }

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
     * Program entry point.
     *
     * <p>
     * This method loads the trained model and
     * classifies the emails of an input directory.
     * </p>
     *
     * @param args Arguments:
     *             {@code  <model> <input_directory> <output_directory>}
     */
    public static void main(String[] args) {
        try {
            if (args.length > 3) {
                System.err.println("Usage: java Predict <model> <input> <output>");
                return;
            }

            String modelPath = args.length >= 1 ? args[0] : "model/mlp.model";
            String inputDir = args.length >= 2 ? args[1] : "data/samples/";
            String outputDir = args.length == 3 ? args[2] : "results/";

            System.out.println("\n--- Predictions ---");
            System.out.println("  Model    : " + modelPath);
            System.out.println("  Input    : " + inputDir);
            System.out.println("  Output   : " + outputDir);

            File modelFile = new File(modelPath);
            File inputDirectory = new File(inputDir);
            File outputDirectory = new File(outputDir);

            if (!modelFile.exists()) {
                System.err.println("ERROR: The model does not exist -> " + modelFile.getAbsolutePath());
                return;
            }

            if (!inputDirectory.exists() || !inputDirectory.isDirectory()) {
                System.err.println(
                        "ERROR: The input directory does not exist -> " + inputDirectory.getAbsolutePath());
                return;
            }

            outputDirectory.mkdirs();

            // Load the data
            System.out.println("\nReading the emails...");
            Instances rawData = loadFlatInferenceData(inputDirectory);

            if (rawData.classIndex() == -1) {
                rawData.setClassIndex(rawData.numAttributes() - 1);
            }

            for (int i = 0; i < rawData.numInstances(); i++) {
                rawData.instance(i).setClassMissing();
            }

            System.out.println("Loaded emails: " + rawData.numInstances());
            System.out.println("Attributes: " + rawData.numAttributes());
            if (rawData.numInstances() == 0) {
                System.out.println(
                        "WARNING: No email was loaded. Check the structure of the input directory.");
            }

            saveToArff(rawData, new File(outputDir, "inference_raw.arff"));
            System.out.println("  RAW ARFF saved.");

            // Load the filters
            // Fix C7: the filters are read from the folder of the given model,
            // so a model is always used with its own filters
            File modelDir = modelFile.getAbsoluteFile().getParentFile();
            File normalizerFile = new File(modelDir, "normalizer.ser");
            File vectorizerFile = new File(modelDir, "vectorizer.ser");
            File selectorFile = new File(modelDir, "selector.ser");

            Instances finalData;
            Instances normalizedData;

            // Apply TextNormalizer (if the model was trained with it)
            if (normalizerFile.exists()) {
                System.out.println("\nApplying TextNormalizer (" + normalizerFile.getPath() + ")...");
                TextNormalizer normalizer = (TextNormalizer) FilterSerializationHelper.loadFilter(normalizerFile.getPath());
                normalizedData = Filter.useFilter(rawData, normalizer);
                System.out.println("  TextNormalizer applied.");
            } else {
                System.out.println("\nTextNormalizer not found. Keeping the original text.");
                normalizedData = rawData;
            }

            if (vectorizerFile.exists() && selectorFile.exists()) {
                System.out.println("\nLoading the training filters (vectorizer.ser, selector.ser)...");

                StringToWordVector stwv = (StringToWordVector) FilterSerializationHelper.loadFilter(vectorizerFile.getPath());
                AttributeSelection selector = (AttributeSelection) FilterSerializationHelper.loadFilter(selectorFile.getPath());

                Instances vectorizedData = Filter.useFilter(normalizedData, stwv);
                System.out.println("  Attributes after vectorization: " + vectorizedData.numAttributes());

                finalData = Filter.useFilter(vectorizedData, selector);
                System.out.println("  Selection applied. Attributes: " + finalData.numAttributes());

            } else {
                throw new Exception(
                        "The serialized filters were not found in the folder " + modelDir.getPath()
                                + ". Run EmailVectorizerAndSelector first.");
            }

            saveToArff(finalData, new File(outputDir, "inference_final.arff"));
            System.out.println("  FINAL ARFF saved.");

            System.out.println("\nLoading the model (neural network)...");
            Classifier classifier = (Classifier) SerializationHelper.read(modelPath);
            System.out.println("  Model loaded.");

            // Make the predictions
            System.out.println("\nThe classification process has started...");
            StringBuilder results = new StringBuilder();
            results.append("==================================================\n");
            results.append("               PREDICTIONS - RESULTS              \n");
            results.append("==================================================\n\n");

            int totalCount = finalData.numInstances();

            for (int i = 0; i < totalCount; i++) {
                Instance currentInstance = finalData.instance(i);

                double[] probabilities = classifier.distributionForInstance(currentInstance);
                double prediction = classifier.classifyInstance(currentInstance);

                String className = finalData.classAttribute().value((int) prediction);
                double confidence = probabilities[(int) prediction];

                results.append(String.format("%d. email: %s (confidence: %.4f)%n",
                        i + 1, className.toUpperCase(), confidence));
            }

            results.append("\n==================================================\n");
            results.append(String.format("Total classified emails: %d%n", totalCount));
            results.append("==================================================\n");

            // Save the results
            File resultsFile = new File(outputDir, "predictions.txt");
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFile))) {
                writer.write(results.toString());
            }

            System.out.println("\n" + results.toString());
            System.out.println("Results saved: " + resultsFile.getAbsolutePath());

            // Experiment tracking (optional)
            /*
            String paramsLog = String.format(
                    "Model: %s\n" +
                            "Input directory: %s\n" +
                            "Test emails: %d\n" +
                            "Filters: vectorizer.ser, selector.ser (model folder)",
                    modelPath, inputDir, totalCount);

            String resultsLog = String.format(
                    "Classified emails: %d\n" +
                            "Output file: %s",
                    totalCount, resultsFile.getAbsolutePath());

            ExperimentLogger.log("6. Predictions (Predict)", paramsLog, resultsLog);
            */
        } catch (Exception e) {
            System.err.println("Critical error while making the predictions.");
            e.printStackTrace();
        }
    }
}
