# NLP Spam Classifier & Inference Pipeline (Java & Weka)

An end-to-end Machine Learning pipeline and CLI inference tool developed in Java using the Weka API. The system cleans, normalizes, and vectorizes raw email text to classify messages into Spam or Ham using a tuned Multilayer Perceptron (MLP) neural network.

Trained and evaluated on 5,172 emails from the Enron-Spam corpus, achieving **96.61% accuracy** and an **F-Measure of 0.9450** on the minority Spam class under repeated stratified hold-out validation.

---

## 🛠 Tech Stack & Architecture

- **Language:** Java (JDK 17+)
- **ML Engine:** Weka API 3.8+ (`weka.jar`)
- **Neural Network:** Multilayer Perceptron (Backpropagation, Momentum, Multi-layer topology)
- **Feature Engineering:** Custom regex-based normalization, TF-IDF vectorization, Information Gain attribute ranking
- **Logging & Evaluation:** Automated experiment tracking (`ExperimentLogger`), Stratified 5-Fold Cross-Validation, Repeated Hold-Out (70/30)

---

## 📊 Key Results & Model Stability

| Evaluation Strategy | Metric | Overall / WAvg | Spam Class |
| :--- | :--- | :--- | :--- |
| **5-Fold Cross-Validation** | Accuracy / F-Measure | 96.61% | 0.9415 |
| **5x Repeated Hold-Out (70/30)** | Mean F-Measure ($\mu \pm \sigma$) | 0.9678 ($\pm 0.0039$) | 0.9450 ($\pm 0.0078$) |
| **Blind Test (517 unseen emails)** | Accuracy / Precision | 95.95% | 95.74% Precision |

> **Feature Space Optimization:** Applying regex normalization and Information Gain selection reduced vocabulary dimensionality by **97%** (from 33,863 raw string attributes down to 1,000 numeric features), minimizing computational overhead and eliminating overfitting ($\sigma = 0.0078$).

---

## 🚀 Execution & CLI Inference

The production package includes serialized pipeline filters (`normalizer.ser`, `vectorizer.ser`, `selector.ser`) and the trained neural model (`mlp.model`) to process new unseen `.txt` files without training overhead.

### 1. Requirements
Ensure Java is available in your environment:
```bash
java -version
```

### 2. Run Predictions on Raw Text Files
Pass the model and the input directory containing raw `.txt` emails:

```bash
java --add-opens java.base/java.lang=ALL-UNNAMED \
  -cp "lib/weka.jar:bin" Iragarri \
  modelo/mlp.model data_proba/ iragarpenak/
```

Console output logs confidence scores for each classified email:
```text
================== IRAGARPENAK (RESULTS) ==================
1. email: SPAM (confidence: 0.9939)
2. email: HAM  (confidence: 0.9993)
3. email: SPAM (confidence: 0.9939)
===========================================================
```

---

## 📂 Project Architecture

```text
.
├── bin/                       # Compiled Java bytecode
├── lib/                       # Weka framework binaries (weka.jar)
├── modelo/                    # Serialized models & pipeline filters
│   ├── mlp.model              # Pre-trained Multilayer Perceptron
│   ├── normalizer.ser         # Regex cleaning pipeline state
│   ├── vectorizer.ser         # TF-IDF StringToWordVector dictionary
│   └── selector.ser           # InfoGain feature selection mask
├── src/                       # Java source codebase
│   ├── DataSplit.java         # Stratified dataset partitioning (80/10/10)
│   ├── EmailLoader.java       # Raw text to Weka Instances parser
│   ├── TextNormalizer.java    # Regex cleaner (URLs, emails, spacing)
│   ├── EmailVectorizer.java   # TF-IDF transformation & feature filtering
│   ├── GetModel.java          # Hyperparameter grid search & training
│   ├── Evaluate.java          # Cross-validation & error matrix analysis
│   ├── ExperimentLogger.java  # Persistent metrics tracking and logging
│   └── Iragarri.java          # CLI inference engine for new files
└── README.md
```

---

## 👥 Authors & Academic Context

Developed as part of the *Decision Support Systems* course at the **School of Engineering of Bilbao (UPV/EHU)**.

- **Andoni Ortiz de Zarate:** Neural network design (MLP), hyperparameter optimization (Grid Search), experimental tracking infrastructure (`ExperimentLogger`), and statistical stability benchmarking.
- **Gorka Piedra:** Technical documentation, data partitioning logic (`DataSplit`), and model evaluation framework (`Evaluate`).
- **Ibai Olaziregi:** Text preprocessing pipeline (`TextNormalizer`), TF-IDF feature selection (`EmailVectorizer`), and CLI inference engine (`Iragarri`).
