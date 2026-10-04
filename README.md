# NLP Text Classifier: Java & Weka Integration

An automated Machine Learning pipeline developed in Java, utilizing the Weka framework to perform Natural Language Processing (NLP) and text classification. The project focuses on transforming unstructured raw text corpora into structured, analyzable datasets for predictive modeling.

---

## 🛠 Tech Stack & Frameworks

- **Core Language:** Java
- **Machine Learning Framework:** Weka API (Waikato Environment for Knowledge Analysis)
- **Data Architecture:** ARFF (Attribute-Relation File Format) structured generation
- **Development Environment:** VS Code (`.vscode/settings.json`)

---

## 🚀 Pipeline Architecture

### 1. Data Ingestion & Parsing
The system processes unstructured text documents directly from the file system. The current implementation parses raw email data (`.txt` formats) located in the `DatuakRaw/ham/` directory, handling diverse character encodings and formatting inconsistencies.

### 2. Preprocessing & Feature Extraction
Unstructured text is programmatically transformed into structured Weka instances. 
- **Tokenization:** Breaking down text blocks into individual tokens.
- **Vectorization:** Utilizing Weka's `StringToWordVector` filter to convert string attributes into numeric feature vectors (e.g., Term Frequency-Inverse Document Frequency / TF-IDF).
- **ARFF Generation:** Compiling the processed features into formatted `.arff` datasets (`emails_raw.arff`) ready for algorithmic processing.

### 3. Classification Modeling
With the structured `.arff` data, the pipeline supports training and evaluating various classification algorithms provided by the Weka API:
- Naïve Bayes (Standard for probabilistic text classification)
- Support Vector Machines (SMO)
- K-Nearest Neighbors (IBk)

Model performance is evaluated through rigorous cross-validation techniques to ensure precision, recall, and F-Measure reliability.

👤 Academic Context
This project demonstrates practical application of Data Mining and Artificial Intelligence methodologies. 
Developed during engineering studies at the School of Engineering of Bilbao (UPV/EHU).
---

## 📂 Project Structure

```text
/
├── .vscode/               # IDE configuration settings
├── DatuakRaw/             # Primary dataset storage
│   ├── ham/               # Unstructured raw text files (.txt)
│   └── emails_raw.arff    # Processed and vectorized Weka dataset
├── src/                   # Java source code containing the NLP pipeline
└── README.md              # Project documentation
