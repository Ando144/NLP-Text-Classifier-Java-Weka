#!/bin/bash
# ==========================================================
# PIPELINE OSOA — Testu-Sailkatzailea (SPAM/HAM)
# Konfigurazio definitiboa — emaitza onenak lortu dituenak
# ==========================================================
# Erabilera:
#   chmod +x pipeline.sh
#   ./pipeline.sh

set -e  # Edozein erroretan gelditu

cd /home/ibai/Dokumentuak/Testu-Sailkatzailea

# ----------------------------------------------------------
# KONFIGURAZIO DEFINITIBOA
# Ablation study-aren ondoren aukeratutako parametroak:
#   - TextNormalizer: desaktibatua (ez du hobetzen)
#   - digitsAsDelim:  false (Enron token informatiboak babesten)
#   - minTermFreq:    1 (jokabide originala)
#   - useThreshold:   false (numToSelect finkoa)
#   - useNormalizer:  false
# ----------------------------------------------------------
WORDS_TO_KEEP=25000
NUM_TO_SELECT=1000
MIN_TERM_FREQ=1
USE_THRESHOLD=false
USE_NORMALIZER=false
# MAX_NGRAM: 1 = unigrama zaharra, 2 = unigrama+bigrama, 3 = unigrama+bigrama+trigrama
MAX_NGRAM=1

# ----------------------------------------------------------
# PARAMETRO OPTIMOAK
LEARNING_RATE=0.005;
MOMENTUM=0.2;
HIDDEN_LAYERS="3";
EPOCHS=300;

# ----------------------------------------------------------
# DIREKTORIOAK PRESTATU
# ----------------------------------------------------------
mkdir -p bin Partiketak emaitzak modelo

echo "=========================================================="
echo "  PIPELINE HASIERA"
echo "=========================================================="

# ----------------------------------------------------------
# 1) KONPILATU
# ----------------------------------------------------------
echo ""
echo "[1/6] Konpilatzen..."
javac -encoding UTF-8 -cp "lib/weka.jar:src" -d bin src/*.java
echo "  OK"

# ----------------------------------------------------------
# 2) ARFF GORDINA SORTU
# ----------------------------------------------------------
echo ""
echo "[2/6] Datuak kargatzen (EmailLoader)..."
java --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     EmailLoader DatuakRaw DatuakRaw/emails_raw.arff
echo "  OK"

# ----------------------------------------------------------
# 3) TRAIN / DEV / TEST BANATU
# ----------------------------------------------------------
echo ""
echo "[3/6] Datuak banatzen 80/10/10 (DataSplit)..."
java --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     DataSplit \
     DatuakRaw/emails_raw.arff \
     Partiketak/train.arff \
     Partiketak/dev.arff \
     Partiketak/test.arff
echo "  OK"

# ----------------------------------------------------------
# 4) BEKTORIZATU ETA ATRIBUTUAK AUKERATU
# Argumentu-ordena:
#   baseDir wordsToKeep numToSelect digitsAsDelim
#   minTermFreq useThreshold useNormalizer
# ----------------------------------------------------------
echo ""
echo "[4/6] Bektorizatzen eta atributuak aukeratzen (EmailVectorizerAndSelector)..."
java --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     EmailVectorizerAndSelector \
     Partiketak \
     "$WORDS_TO_KEEP" \
     "$NUM_TO_SELECT" \
     "$DIGITS_AS_DELIM" \
     "$MIN_TERM_FREQ" \
     "$USE_THRESHOLD" \
     "$USE_NORMALIZER" \
     "$MAX_NGRAM"
echo "  OK"

# ----------------------------------------------------------
# 5) FINE-TUNING ETA EREDUA GORDE
# ----------------------------------------------------------
echo ""
echo "[5/6] MLP fine-tuning eta eredua gordetzen (GetModel)..."
java --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     GetModel \
     Partiketak/train_final.arff \
     Partiketak/dev_final.arff \
     modelo/mlp.model \
     emaitzak/finetuning_finala.txt
echo "  OK"

# ----------------------------------------------------------
# 6) EBALUAZIO ITSUA (TEST)
# ----------------------------------------------------------
echo ""
echo "[6/6] Test itsuaren ebaluazioa (TestModel)..."
java --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     TestModel \
     Partiketak/test_final.arff \
     modelo/mlp.model \
     emaitzak/test_emaitzak_finala.txt
echo "  OK"

# ----------------------------------------------------------
# 7) KALITATE ESTIMATUA (EVALUATE)
# ----------------------------------------------------------
echo ""
echo "[7/7] Kalitate estimatua (Evaluate)..."
java --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     Evaluate \
     Partiketak/train_final.arff \
     Partiketak/dev_final.arff \
     emaitzak/kalitatea_finala.txt \
     "$LEARNING_RATE" \
     "$MOMENTUM" \
     "$HIDDEN_LAYERS" \
     "$EPOCHS"
echo "  OK"

# ----------------------------------------------------------
# 8) IRAGARPEN BERRIAK (Iragarri)
# ----------------------------------------------------------
echo ""
echo "[8/8] Iragarpen berriak egiten (Iragarri)..."
java --add-opens java.base/java.lang=ALL-UNNAMED \
     -cp "lib/weka.jar:bin" \
     Iragarri \
     modelo/mlp.model \
     DatuakRaw/proba_data \
     emaitzak/
echo "  OK"

# ----------------------------------------------------------
# LABURPENA
# ----------------------------------------------------------
echo ""
echo "=========================================================="
echo "  PIPELINE AMAITUTA"
echo "=========================================================="
echo "Emaitzak hemen:"
echo "  - emaitzak/finetuning_finala.txt      (fine-tuning taula)"
echo "  - emaitzak/test_emaitzak_finala.txt   (test itsuaren emaitzak)"
echo "  - emaitzak/kalitatea_finala.txt       (kalitate estimatua)"
echo "  - emaitzak/iragarpenak.txt            (iragarpen berriak)"
echo "  - registro_experimentos.txt           (log osoa)"
echo "  - modelo/mlp.model                    (eredu onena)"
echo "=========================================================="