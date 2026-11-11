# Changelog

All notable changes to this repository after the original course submission.

## [1.1.0] – 2026-11-11 · English translation and portfolio clean-up

### Translation (no behaviour change)

- Comments, Javadoc, console messages and report texts translated from Basque/Spanish to English.
- 26 local identifiers renamed 1:1 (see the table in [docs/DESIGN.md](docs/DESIGN.md#appendix-a--name-equivalence-table)). The class `Iragarri` was renamed `Predict` (file `src/Predict.java`).
- The class name `TextNormalizer` was **kept on purpose**: an instance of it is serialized in `model/normalizer.ser`, so renaming it would break the trained model shipped with the repository.
- Folders and output files renamed to English. The paths written in the code and in `pipeline.sh` were updated with the same mapping.
- How the translation was verified (before the bug fixes below were applied):
  1. **Token-by-token comparison:** 8,808 tokens in 10 files. Every keyword, operator and number is identical and in the same order. The only differences are the 26 renamed identifiers and 178 translated string literals, and every translated literal keeps the same `printf` format specifiers and the same `\n`/`\t` count.
  2. **Bytecode comparison:** both versions were compiled with `javac -g:none` and their `javap -c` output was compared. All 10 classes have identical instructions once the string constants and the `Iragarri`→`Predict` name are masked.
  3. **Execution:** `EmailLoader` → `DataSplit` → `EmailVectorizerAndSelector` produced identical ARFF data with both versions. With the shipped model, `TestModel` (518 predictions and all metrics) and `Predict` (12 predictions with their confidences) gave identical numbers.

### Added

- `LICENSE` file (MIT License, © the three authors), badges and a License section in the README. The License section notes that `lib/weka.jar` stays under the GNU GPL v3.

### Fixed

| ID | Problem | Effect | Fix |
|----|---------|--------|-----|
| C1 | `pipeline.sh` ran `cd /home/ibai/Dokumentuak/Testu-Sailkatzailea`, an absolute path on one team member's computer. | The pipeline failed on any other machine. | `cd "$(dirname "$0")"`: the script now runs from its own folder. |
| C2 | Steps 5–7 of `pipeline.sh` wrote to `esperimentuak/*_norm.txt`. | Running the pipeline again overwrote the archived results that the report cites. | The outputs now go to `results/finetuning.txt`, `results/test_results.txt` and `results/quality.txt`. |
| C3 | `Evaluate` called its hold-out "Repeated Stratified Hold-Out", but it uses `weka.filters.unsupervised.instance.Resample`, which ignores the class. | The reports described the method wrongly. The splits are random 70/30 samples without replacement. | The program now prints "Repeated Random Hold-Out", and the comments explain how the split is drawn. The method was not changed, so every published hold-out figure is still reproducible. A truly stratified version (`supervised.instance.Resample`) would change all the numbers and require re-running the five configurations. |
| C4 | `TestModel` built the predictions file name with `resultsPath.replace(".txt", "_iragarpenak.txt")`. | If the results path did not contain `.txt`, the predictions overwrote the report (reproduced: the report file contained the prediction list). | Only a trailing `.txt` is replaced. Otherwise `_predictions.txt` is appended. |
| C5 | `GetModel` and `Evaluate` called `new File(path).getParentFile().mkdirs()` without a null check. | `NullPointerException` when the path has no folder, e.g. `quality.txt` (reproduced with `Evaluate`). | The parent folder is created only if it exists in the path. |
| C6 | `Predict` (`Iragarri`) read the files with `Files.readString(..., UTF_8)` and never closed the `Files.walk` stream. Its Javadoc said "without subfolders", but the search is recursive. | A single non-UTF-8 file aborted the whole run with `MalformedInputException` (reproduced with a Latin-1 file). The stream leaked a file handle. | The bytes are decoded as UTF-8 and invalid bytes are replaced. `Files.walk` is closed with try-with-resources. The Javadoc now says the search is recursive. |
| C7 | `Predict` always loaded the filters from `modelo/*.ser`, whatever model path was given. | A model stored in another folder was used with the wrong filters, or the run failed when started from another directory (reproduced). | The filters are read from the folder that contains the given model file. |
| C8 | Three Javadoc blocks (for `buildMLP`, `formatResults` and `mean`) were inside `Evaluate.main()`. | The Javadoc tool ignored them, and `formatResults` and `mean` had no documentation. | The duplicated block was removed, and the other two were moved onto their methods. Only comments changed. |
| C9 | `ExperimentLogger` wrote "MB used for the neural network" in every phase, and one comment was in the wrong place. | The log was misleading for phases that train no network (loading, split, vectorization). | The log now says "MB used by the JVM", and the comments were fixed. |
| C10 | Rebuilding the corpus depended on the computer: a Windows checkout converted the emails to CRLF; `TextDirectoryLoader` used the file-system order of folders and files and the platform charset; Weka read and wrote ARFF files with the platform charset. | `EmailLoader` produced a different ARFF on each OS (other texts, class order `{ham,spam}` or `{spam,ham}`, other e-mail order). As a result, `DataSplit` gave other splits and the pipeline gave another model. | `.gitattributes` keeps `data/**` byte-identical (no CRLF conversion); the working copy of `data/` was restored to the exact committed bytes. `EmailLoader` reads as UTF-8, sorts the class labels, sorts the emails by class and file name, and uses the folder name as relation name. `pipeline.sh` runs Java with `-Dfile.encoding=UTF-8`. |

Re-checked after the fixes:

- With the shipped model, the blind-test metrics, the 518 test predictions, the 12 sample predictions and the vectorized inference ARFF are still identical to the original program's output.
- In `Evaluate` and `ExperimentLogger`, the only token changes are the 5 output texts of C3 and C9.
- C10: `EmailLoader` + `DataSplit` produced byte-identical ARFF files (same MD5) on two machines (Java 11 on the user's computer, Java 21 on Linux). The new ARFF contains exactly the same 5,172 (text, class) pairs as the published `emails_raw.arff`; only the order differs.

**Note on C10.** The published `emails_raw.arff`, the splits and the model were made in the original run, with the e-mail order given by the file system of that computer. That order cannot be rebuilt from the files. Running the pipeline again therefore gives a different, but now reproducible, split and model; their scores will differ slightly from the published ones.

### Repository

- `Dokumentazioa/` (LaTeX report, diagrams, `Graficos varios.txt`) moved to `docs/original-eu/`, unchanged. `registro_experimentos.txt` renamed `experiment_log.txt`, with its contents unchanged.
- New: `README.md`, `docs/DESIGN.md` (English translation of the report, checked against the code), and `docs/images/` (the report's charts redrawn in English from the result files).
- The Basque Javadoc HTML in `docs/` was replaced by English Javadoc generated in `docs/javadoc/`.
- Older exploratory runs from `emaitzak/` moved to `results/archive/`, unchanged. The original prediction outputs (`iragarpenak/`) moved to `results/archive/iragarpenak/`. `predictions/` now holds the output of the translated `Predict`, whose numbers are identical.
- `repomix-output.xml` removed (a 7 MB dump of the repository that no program used).
- `bin/` recompiled from the translated sources with `--release 11`.
- New `.gitattributes` (see C10).
