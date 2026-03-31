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

/**
 * Entrenatutako sailkatzaile bat erabiliz (MLP eredua),
 * email berrien gaineko iragarpenak egiten dituen klasea.
 *
 * 
 * Pipeline osoa exekutatzen du inferentzia fasean:
 * 
 * <ul>
 * <li>Emailak kargatu (direktorio egituratik edo fitxategi lautik)</li>
 * <li>Testuaren normalizazioa aplikatu</li>
 * <li>Entrenamenduan erabilitako filtroak berrerabili (bektorizazioa eta
 * atributu-aukeraketa)</li>
 * <li>Eredua kargatu eta iragarpenak egin</li>
 * <li>Emaitzak fitxategi batean gorde</li>
 * </ul>
 *
 * 
 * Helburua da entrenamendu eta inferentzia arteko koherentzia bermatzea.
 * 
 *
 * @version 1.0
 */
public class Iragarri {
    /**
     * Karpeta batean dauden .txt fitxategiak kargatzen ditu,
     * azpikarpetarik gabe (egitura laua).
     *
     * <p>
     * Fitxategi bakoitza instantzia bat bihurtzen da,
     * testu atributu bakarrarekin eta klase ezezagunarekin.
     * </p>
     *
     * @param inputDirectory Sarrerako karpeta
     * @return Sortutako Instances objektua
     * @throws IOException Irakurketa errorea gertatzen bada
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
        Files.walk(inputDirectory.toPath())
                .filter(Files::isRegularFile)
                .filter(p -> p.getFileName().toString().toLowerCase().endsWith(".txt"))
                .forEach(txtFiles::add);

        txtFiles.sort(Comparator.comparing(Path::toString));

        for (Path txtFile : txtFiles) {
            String content = Files.readString(txtFile, StandardCharsets.UTF_8);
            double[] values = new double[data.numAttributes()];
            values[0] = data.attribute(0).addStringValue(content);
            values[data.classIndex()] = Utils.missingValue();
            data.add(new weka.core.DenseInstance(1.0, values));
        }

        return data;
    }


    /**
     * Programaren sarrera-puntua.
     *
     * <p>
     * Metodo honek entrenatutako eredua kargatzen du eta
     * sarrera direktorio bateko emailak sailkatzen ditu.
     * </p>
     *
     * @param args Argumentuak:
     *             {@code  <eredua> <sarrera_direktorioa> <irteera_direktorioa>}
     */
    public static void main(String[] args) {
        try {
            if (args.length > 3) {
                System.err.println("Erabilera: java Iragarri <eredua> <sarrera> <irteera>");
                return;
            }

            String modelPath = args.length >= 1 ? args[0] : "modelo/mlp.model";
            String inputDir = args.length >= 2 ? args[1] : "data_proba/";
            String outputDir = args.length == 3 ? args[2] : "emaitzak/";

            System.out.println("\n--- Iragarpenak ---");
            System.out.println("  Eredua   : " + modelPath);
            System.out.println("  Sarrera  : " + inputDir);
            System.out.println("  Irteera  : " + outputDir);

            File modelFile = new File(modelPath);
            File inputDirectory = new File(inputDir);
            File outputDirectory = new File(outputDir);

            if (!modelFile.exists()) {
                System.err.println("ERROREA: Eredua ez da existitzen -> " + modelFile.getAbsolutePath());
                return;
            }

            if (!inputDirectory.exists() || !inputDirectory.isDirectory()) {
                System.err.println(
                        "ERROREA: Sarrera direktorioa ez da existitzen -> " + inputDirectory.getAbsolutePath());
                return;
            }

            outputDirectory.mkdirs();

            // Datuak kargatu
            System.out.println("\nEmailak irakurtzen...");
            Instances probaRaw = loadFlatInferenceData(inputDirectory);

            EmailLoader.ensureClassIndex(probaRaw);
            for (int i = 0; i < probaRaw.numInstances(); i++) {
                probaRaw.instance(i).setClassMissing();
            }

            System.out.println("Kargatutako mezuak: " + probaRaw.numInstances());
            System.out.println("Atributuak: " + probaRaw.numAttributes());
            if (probaRaw.numInstances() == 0) {
                System.out.println(
                        "ABISUA: Ez da mezu elektronikorik kargatu. Egiaztatu sarrera direktorioaren egitura.");
            }

            EmailLoader.saveToArff(probaRaw, new File(outputDir, "proba_raw.arff"));
            System.out.println("  RAW ARFF gordeta.");

            // Filtroak kargatu
            File normalizerFile = new File("modelo/normalizer.ser");
            File vectorizerFile = new File("modelo/vectorizer.ser");
            File selectorFile = new File("modelo/selector.ser");

            Instances probaFinal;
            Instances probaNorm;

            // TestNormalizer aplikatu (horrekin entrenatu bada)
            if (normalizerFile.exists()) {
                System.out.println("\nTextNormalizer aplikatzen (modelo/normalizer.ser)...");
                TextNormalizer normalizer = (TextNormalizer) FilterSerializationHelper
                        .loadFilter(normalizerFile.getPath());
                probaNorm = Filter.useFilter(probaRaw, normalizer);
                System.out.println("  TextNormalizer aplikatuta.");
            } else {
                System.out.println("\nTextNormalizer ez da aurkitu. Jatorrizko testua mantentzen.");
                probaNorm = probaRaw;
            }

            if (vectorizerFile.exists() && selectorFile.exists()) {
                System.out.println("\nEntrenamenduko filtroak kargatzen (vectorizer.ser, selector.ser)...");

                StringToWordVector stwv = (StringToWordVector) FilterSerializationHelper
                        .loadFilter(vectorizerFile.getPath());
                AttributeSelection selector = (AttributeSelection) FilterSerializationHelper
                        .loadFilter(selectorFile.getPath());

                Instances probaVec = Filter.useFilter(probaNorm, stwv);
                System.out.println("  Bektorizazioaren ostean atributuak: " + probaVec.numAttributes());

                probaFinal = Filter.useFilter(probaVec, selector);
                System.out.println("  Aukeraketa aplikatuta. Atributuak: " + probaFinal.numAttributes());

            } else {
                throw new Exception(
                        "Filtro serializatuak ez dira aurkitu 'modelo/' karpetan. Exekutatu EmailVectorizerAndSelector lehenik.");
            }

            EmailLoader.saveToArff(probaFinal, new File(outputDir, "proba_final.arff"));
            System.out.println("  FINAL ARFF gordeta.");

            System.out.println("\nEredua (sare neuronala) kargatzen...");
            Classifier modelo = (Classifier) SerializationHelper.read(modelPath);
            System.out.println("  Eredua ondo kargatuta.");

            // Iragaerpenak egin
            System.out.println("\nSailkatzearen prozesua hasi da...");
            StringBuilder emaitzak = new StringBuilder();
            emaitzak.append("==================================================\n");
            emaitzak.append("               IRAGARPENAK - EMAITZAK             \n");
            emaitzak.append("==================================================\n\n");

            int totalCount = probaFinal.numInstances();

            for (int i = 0; i < totalCount; i++) {
                Instance instancia = probaFinal.instance(i);

                double[] probabilidades = modelo.distributionForInstance(instancia);
                double prediccion = modelo.classifyInstance(instancia);

                String klaseIzena = probaFinal.classAttribute().value((int) prediccion);
                double confidence = probabilidades[(int) prediccion];

                emaitzak.append(String.format("%d. mezua: %s (zehaztasuna: %.4f)%n",
                        i + 1, klaseIzena.toUpperCase(), confidence));
            }

            emaitzak.append("\n==================================================\n");
            emaitzak.append(String.format("Guztira sailkatutako mezuak: %d%n", totalCount));
            emaitzak.append("==================================================\n");

            // Emaitzak gorde
            File resultsFile = new File(outputDir, "iragarpenak.txt");
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(resultsFile))) {
                writer.write(emaitzak.toString());
            }

            System.out.println("\n" + emaitzak.toString());
            System.out.println("Emaitzak gordeta: " + resultsFile.getAbsolutePath());

            // Experiment Tracking (Aukerakoa)
            String parametrosLog = String.format(
                    "Eredua: %s\n" +
                            "Sarrera direktorioa: %s\n" +
                            "Proba mezuak: %d\n" +
                            "Filtroak: modelo/vectorizer.ser, modelo/selector.ser",
                    modelPath, inputDir, totalCount);

            String resultadosLog = String.format(
                    "Sailkatutako mezuak: %d\n" +
                            "Irteera fitxategia: %s",
                    totalCount, resultsFile.getAbsolutePath());

            ExperimentLogger.log("6. Iragarpenak (Iragarri)", parametrosLog, resultadosLog);

        } catch (Exception e) {
            System.err.println("Errore kritikoa iragarpenak egitean.");
            e.printStackTrace();
        }
    }
}