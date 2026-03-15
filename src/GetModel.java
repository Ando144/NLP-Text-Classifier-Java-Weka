import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 * GetModel
 *
 * Realiza el fine-tuning del MultilayerPerceptron probando combinaciones
 * de parámetros, evalúa cada combinación sobre dev, y guarda el modelo
 * óptimo (mayor F-Measure de spam) en un fichero .model
 *
 * Uso:
 * java -cp "lib\weka.jar;bin" GetModel <train_final.arff> <dev_final.arff>
 * <modelo_salida.model> <resultados.txt>
 *
 * Ejemplo:
 * java -cp "lib\weka.jar;bin" GetModel Partiketak/train_final.arff
 * Partiketak/dev_final.arff modelo/mlp.model resultados/finetuning.txt
 */
public class GetModel {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.out.println(
                    "Uso: java -cp \"lib\\weka.jar;bin\" GetModel <train_final.arff> <dev_final.arff> <modelo.model> <resultados.txt>");
            return;
        }

        String trainPath = args[0];
        String devPath = args[1];
        String modelPath = args[2];
        String resultsPath = args[3];

        // 1. Cargar datasets
        Instances train = new DataSource(trainPath).getDataSet();
        Instances dev = new DataSource(devPath).getDataSet();

        train.setClassIndex(train.numAttributes() - 1);
        dev.setClassIndex(dev.numAttributes() - 1);

        System.out.println("Train: " + train.numInstances() + " instancias");
        System.out.println("Dev  : " + dev.numInstances() + " instancias");

        // 2. Parámetros a explorar
        double[] learningRates = { 0.001, 0.005, 0.01 };
        double[] momentums = { 0.2, 0.3, 0.4 };
        String[] hiddenLayers = { "3", "5", "10" };
        int[] epochs = { 100 };

        // 3. Fine-tuning: probar todas las combinaciones
        MultilayerPerceptron bestModel = null;
        double bestFMeasure = -1;
        String bestParams = "";

        // Índice de la clase spam (para obtener su F-Measure)
        int spamIndex = train.classAttribute().indexOfValue("spam");

        // Crear directorio de resultados si no existe
        File resultsFile = new File(resultsPath);
        if (resultsFile.getParentFile() != null) {
            resultsFile.getParentFile().mkdirs();
        }

        BufferedWriter writer = new BufferedWriter(new FileWriter(resultsPath));
        writer.write("=== FINE-TUNING MultilayerPerceptron ===\n");
        writer.write("Metrica optimizacion: F-Measure (spam)\n");
        writer.write("Train: " + trainPath + "\n");
        writer.write("Dev  : " + devPath + "\n\n");
        writer.write(String.format("%-15s %-12s %-14s %-8s %-12s %-12s %-12s%n",
                "LearningRate", "Momentum", "HiddenLayers", "Epochs",
                "F-spam", "F-ham", "WAvg-F"));
        writer.write(repeat("-", 85) + "\n");

        int total = learningRates.length * momentums.length * hiddenLayers.length * epochs.length;
        int current = 0;

        for (double lr : learningRates) {
            for (double mom : momentums) {
                for (String hidden : hiddenLayers) {
                    for (int ep : epochs) {
                        current++;
                        System.out.printf("[%d/%d] LR=%.3f Mom=%.1f Hidden=%s Epochs=%d%n",
                                current, total, lr, mom, hidden, ep);

                        // Configurar MLP
                        MultilayerPerceptron mlp = new MultilayerPerceptron();
                        mlp.setLearningRate(lr);
                        mlp.setMomentum(mom);
                        mlp.setHiddenLayers(hidden);
                        mlp.setTrainingTime(ep);
                        mlp.setNominalToBinaryFilter(true);
                        mlp.setNormalizeAttributes(true);

                        // Entrenar con train
                        mlp.buildClassifier(train);

                        // Evaluar con dev
                        Evaluation eval = new Evaluation(train);
                        eval.evaluateModel(mlp, dev);

                        double fSpam = eval.fMeasure(spamIndex);
                        double fHam = eval.fMeasure(1 - spamIndex);
                        double fWAvg = eval.weightedFMeasure();

                        // Guardar resultado en fichero
                        writer.write(String.format("%-15.3f %-12.1f %-14s %-8d %-12.4f %-12.4f %-12.4f%n",
                                lr, mom, hidden, ep, fSpam, fHam, fWAvg));

                        // Actualizar mejor modelo
                        if (fSpam > bestFMeasure) {
                            bestFMeasure = fSpam;
                            bestModel = mlp;
                            bestParams = String.format(
                                    "LearningRate=%.3f | Momentum=%.1f | HiddenLayers=%s | Epochs=%d",
                                    lr, mom, hidden, ep);
                        }
                    }
                }
            }
        }

        // 4. Resumen del mejor modelo
        writer.write("\n" + repeat("=", 85) + "\n");
        writer.write("MEJOR MODELO:\n");
        writer.write("  Parametros : " + bestParams + "\n");
        writer.write("  F-Measure (spam): " + String.format("%.4f", bestFMeasure) + "\n");
        writer.close();

        System.out.println("\n=== MEJOR MODELO ===");
        System.out.println("  " + bestParams);
        System.out.printf("  F-Measure (spam): %.4f%n", bestFMeasure);

        // 5. Guardar el modelo óptimo
        new File(modelPath).getParentFile().mkdirs();
        SerializationHelper.write(modelPath, bestModel);

        System.out.println("\nModelo guardado en : " + modelPath);
        System.out.println("Resultados en      : " + resultsPath);
    }

    private static String repeat(String s, int n) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++)
            sb.append(s);
        return sb.toString();
    }
}