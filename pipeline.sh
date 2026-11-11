#!/bin/bash
# ==========================================================
# FULL PIPELINE - Text Classifier (SPAM/HAM)
# Final configuration - the one that gave the best results
# ==========================================================
# What it does : Builds the ARFF corpus from raw .txt emails,
#                splits it 80/10/10, normalizes and vectorizes the
#                text (TF-IDF + InfoGain), tunes a Multilayer
#                Perceptron with grid search, evaluates it (blind
#                test, 5-fold CV, repeated hold-out) and classifies
#                new emails.
# Platform     : Linux / macOS / Git Bash, Java 11+ (JDK, needs
#                javac and javadoc). Runs on CPU.
# Tools        : Java, Weka 3.8.6 (lib/weka.jar), Bash.
# Authors      : Ibai Olaziregi, Gorka Piedra, Andoni Ortiz de Zarate
# Course       : Decision Support Systems (Erabakiak Hartzeko
#                Euskarri Sistemak), Bilbao School of Engineering
#                (UPV/EHU).
# ==========================================================
# Usage:
#   chmod +x pipeline.sh
#   ./pipeline.sh

set -e  # Stop on any error

# Fix C10: all java commands use -Dfile.encoding=UTF-8, because Weka reads and
# writes the ARFF files with the default charset of the platform (it is not
# UTF-8 on Windows with Java 11-17). With UTF-8 the outputs are identical on
# every operating system.

# Fix C1: run from the folder of this script (before, an absolute path of
# one team member's computer was used, so the script failed anywhere else)
cd "$(dirname "$0")"

# ----------------------------------------------------------
# FINAL CONFIGURATION
# Parameters chosen after the ablation study:
#   - digitsAsDelim:  false (keeps the informative Enron tokens)
#   - minTermFreq:    1 (original behaviour)
#   - useThreshold:   false (fixed numToSelect)
#   - useNormalizer:  true
# ----------------------------------------------------------
WORDS_TO_KEEP=25000
NUM_TO_SELECT=1000
DIGITS_AS_DELIM=false
MIN_TERM_FREQ=1
USE_THRESHOLD=false
USE_NORMALIZER=true
MAX_NGRAM=1

# ----------------------------------------------------------
# OPTIMAL PARAMETERS
LEARNING_RATE=0.01;
MOMENTUM=0.1;
HIDDEN_LAYERS="3";
EPOCHS=500;

# ----------------------------------------------------------
# PREPARE THE DIRECTORIES
# ----------------------------------------------------------
mkdir -p bin data/splits results model

echo "=========================================================="
echo "  PIPELINE START"
echo "=========================================================="

# ----------------------------------------------------------
# 1) COMPILE
# ----------------------------------------------------------
echo ""
echo "[1/6] Compiling..."
javac -encoding UTF-8 -cp "lib/weka.jar:src" -d bin src/*.java
echo "  OK"

# ----------------------------------------------------------
# 2) CREATE THE RAW ARFF
# ----------------------------------------------------------
echo ""
echo "[2/6] Loading the data (EmailLoader)..."
java -Dfile.encoding=UTF-8 --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     EmailLoader data/raw data/raw/emails_raw.arff
echo "  OK"

# ----------------------------------------------------------
# 3) SPLIT INTO TRAIN / DEV / TEST
# ----------------------------------------------------------
echo ""
echo "[3/6] Splitting the data 80/10/10 (DataSplit)..."
java -Dfile.encoding=UTF-8 --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     DataSplit \
     data/raw/emails_raw.arff \
     data/splits/train.arff \
     data/splits/dev.arff \
     data/splits/test.arff
echo "  OK"

# ----------------------------------------------------------
# 4) VECTORIZE AND SELECT ATTRIBUTES
# Argument order:
#   baseDir wordsToKeep numToSelect digitsAsDelim
#   minTermFreq useThreshold useNormalizer
# ----------------------------------------------------------
echo ""
echo "[4/6] Vectorizing and selecting attributes (EmailVectorizerAndSelector)..."
java -Dfile.encoding=UTF-8 --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     EmailVectorizerAndSelector \
     data/splits \
     "$WORDS_TO_KEEP" \
     "$NUM_TO_SELECT" \
     "$DIGITS_AS_DELIM" \
     "$MIN_TERM_FREQ" \
     "$USE_THRESHOLD" \
     "$USE_NORMALIZER" \
     "$MAX_NGRAM"
echo "  OK"

# ----------------------------------------------------------
# 5) FINE-TUNING AND SAVE THE MODEL
# Fix C2: the outputs go to results/ (before, they overwrote the
# archived experiment results in experiments/)
# ----------------------------------------------------------
echo ""
echo "[5/6] MLP fine-tuning and saving the model (GetModel)..."
java -Dfile.encoding=UTF-8 --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     GetModel \
     data/splits/train_final.arff \
     data/splits/dev_final.arff \
     model/mlp.model \
     results/finetuning.txt
echo "  OK"

# ----------------------------------------------------------
# 6) BLIND EVALUATION (TEST)
# ----------------------------------------------------------
echo ""
echo "[6/6] Blind test evaluation (TestModel)..."
java -Dfile.encoding=UTF-8 --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     TestModel \
     data/splits/test_final.arff \
     model/mlp.model \
     results/test_results.txt
echo "  OK"

# ----------------------------------------------------------
# 7) ESTIMATED QUALITY (EVALUATE)
# ----------------------------------------------------------
echo ""
echo "[7/7] Estimated quality (Evaluate)..."
java -Dfile.encoding=UTF-8 --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     Evaluate \
     data/splits/train_final.arff \
     data/splits/dev_final.arff \
     results/quality.txt \
     "$LEARNING_RATE" \
     "$MOMENTUM" \
     "$HIDDEN_LAYERS" \
     "$EPOCHS"
echo "  OK"

# ----------------------------------------------------------
# 8) NEW PREDICTIONS (Predict)
# ----------------------------------------------------------
echo ""
echo "[8/8] Making new predictions (Predict)..."
java -Dfile.encoding=UTF-8 --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     Predict \
     model/mlp.model \
     data/samples/ \
     predictions/
echo "  OK"


# ----------------------------------------------------------
# 9) GENERATE THE JAVADOC
# ----------------------------------------------------------
javadoc -d docs/javadoc -sourcepath src -cp "lib/*" src/*.java


# ----------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------
echo ""
echo "=========================================================="
echo "  PIPELINE FINISHED"
echo "=========================================================="
echo "Results here:"
echo "  - results/finetuning.txt             (fine-tuning table)"
echo "  - results/test_results.txt           (blind test results)"
echo "  - results/quality.txt                (estimated quality)"
echo "  - predictions/predictions.txt        (new predictions)"
echo "  - experiment_log.txt                 (full log)"
echo "  - model/mlp.model                    (best model)"
echo "=========================================================="
